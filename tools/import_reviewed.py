"""Bring the REVIEWED labels back from CVAT (or Roboflow) and build the training dataset.

Annotation tools quietly change things on the way out:
  * Roboflow sorts class names alphabetically and renames files (bench1_00012_jpg.rf.<hash>.jpg)
  * CVAT's free plan exports labels WITHOUT images
  * either tool may re-split train/val or add augmented copies
This script undoes all of that, so the dataset you train on is exactly what you reviewed:
  * class ids are remapped BY NAME to the canonical order (from dataset/bas/split.json)
  * each label file is matched back to your ORIGINAL full-resolution frame in dataset/raw
  * the train/val split is re-made BY SESSION FOLDER (same val session as tools/auto_label.py)
  * frames you deleted in the tool are left out; frames with no boxes become negative examples

    python tools/import_reviewed.py --export "%USERPROFILE%\\Downloads\\bas-review.zip"

Output (default dataset/bas_reviewed): images/{train,val} labels/{train,val} data.yaml split.json
plus dataset/bas_reviewed_colab.zip = this dataset + tools/train.py + the base weights, ready to upload
to tools/colab_train.ipynb (paths inside the zip match the project layout).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LIST_FILES = {"train.txt", "val.txt", "valid.txt", "test.txt"}
RF_SUFFIX = re.compile(r"^(?P<stem>.+)_(?:jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-fA-F]+$")


def original_stem(stem: str) -> str:
    m = RF_SUFFIX.match(stem)
    return m.group("stem") if m else stem


def norm(name: str) -> str:
    return " ".join(str(name).strip().lower().replace("_", " ").split())


def read_names(root: Path) -> dict[int, str]:
    import yaml

    for y in sorted(root.rglob("*.yaml")) + sorted(root.rglob("*.yml")):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            continue
        names = d.get("names") if isinstance(d, dict) else None
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return dict(enumerate(map(str, names)))
    for fname in ("obj.names", "classes.txt", "_darknet.labels"):
        for f in root.rglob(fname):
            lines = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            return dict(enumerate(lines))
    raise SystemExit("No class names found in the export (expected data.yaml or obj.names). "
                     "Export as 'Ultralytics YOLO Detection 1.0' (CVAT) or 'YOLOv11' (Roboflow).")


def is_label_file(p: Path, root: Path) -> bool:
    if p.suffix.lower() != ".txt" or p.name in LIST_FILES or p.name.lower().startswith("readme"):
        return False
    parts = [q.lower() for q in p.relative_to(root).parts[:-1]]
    return "labels" in parts or any(q.startswith("obj_") and q.endswith("_data") for q in parts)


def parse_label(p: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        tok = line.split()
        if not tok:
            continue
        try:
            vals = [float(t) for t in tok]
        except ValueError:
            raise SystemExit(f"{p.name} line {n}: not a YOLO label line: {line!r}")
        cls = int(vals[0])
        if len(vals) in (5, 6):            # box (+ optional track id / confidence)
            cx, cy, w, h = vals[1:5]
        elif len(vals) >= 7 and len(vals) % 2 == 1:   # polygon -> enclosing box
            xs, ys = vals[1::2], vals[2::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        else:
            raise SystemExit(f"{p.name} line {n}: unexpected number of values: {line!r}")
        boxes.append((cls, cx, cy, w, h))
    return boxes


def clip_box(cx, cy, w, h):
    x0, y0 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
    x1, y1 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
    if x1 - x0 < 1e-3 or y1 - y0 < 1e-3:
        return None
    return (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0


def letterbox_suspected(img_path: Path, raw_path: Path) -> bool:
    """Roboflow 'Fit (black/white edges)' resize pads the frame, which makes normalised boxes wrong
    for the original frame. 'Stretch' (the default) is fine. Detect uniform bars on a resized export."""
    try:
        import cv2
    except ImportError:
        return False
    a, b = cv2.imread(str(img_path)), cv2.imread(str(raw_path))
    if a is None or b is None:
        return False
    ra, rb = a.shape[1] / a.shape[0], b.shape[1] / b.shape[0]
    if abs(ra - rb) < 0.02:
        return False
    h, w = a.shape[:2]
    bands = [a[: max(2, h // 25)], a[-max(2, h // 25):], a[:, : max(2, w // 25)], a[:, -max(2, w // 25):]]
    flat = [band.std() < 2.0 for band in bands]
    return (flat[0] and flat[1]) or (flat[2] and flat[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="zip or folder exported from CVAT / Roboflow (YOLO format)")
    ap.add_argument("--raw", default="dataset/raw", help="original frames, one sub-folder per session")
    ap.add_argument("--prelabels", default="dataset/bas", help="auto_label.py output (split.json lives here)")
    ap.add_argument("--out", default="dataset/bas_reviewed")
    ap.add_argument("--val-session", nargs="+", default=None, help="override the val session(s) from split.json")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--colab-zip", action=argparse.BooleanOptionalAction, default=True,
                    help="also write <out>_colab.zip for GPU training in Colab/Kaggle")
    args = ap.parse_args()

    raw, pre, out = ROOT / args.raw, ROOT / args.prelabels, ROOT / args.out
    split_file = pre / "split.json"
    if not split_file.exists():
        raise SystemExit(f"{split_file} not found - run tools/auto_label.py first (it records classes and val session).")
    meta = json.loads(split_file.read_text(encoding="utf-8"))
    classes: list[str] = meta["classes"]
    val_sessions = set(args.val_session or meta["val_sessions"])
    canon = {norm(c): i for i, c in enumerate(classes)}

    # ---- original frames: stem -> (path, session)
    frames: dict[str, tuple[Path, str]] = {}
    for p in raw.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and len(p.relative_to(raw).parts) >= 2:
            if p.stem in frames:
                raise SystemExit(f"Duplicate frame name {p.stem} in {raw}")
            frames[p.stem] = (p, p.relative_to(raw).parts[0])
    if not frames:
        raise SystemExit(f"No frames found in {raw}/<session>/")
    unknown_val = val_sessions - {s for _, s in frames.values()}
    if unknown_val:
        raise SystemExit(f"Val session(s) {sorted(unknown_val)} not found in {raw}")

    # ---- open the export
    src = Path(args.export).expanduser()
    if not src.is_absolute():
        src = (Path.cwd() / src) if (Path.cwd() / src).exists() else ROOT / src
    if not src.exists():
        raise SystemExit(f"Export not found: {src}")
    tmp = None
    if src.is_file():
        tmp = Path(tempfile.mkdtemp(prefix="bas_export_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        root = tmp
    else:
        root = src
    try:
        names = read_names(root)
        label_files: dict[str, Path] = {}
        seen_as: dict[str, str] = {}
        export_images: dict[str, Path] = {}
        listed: set[str] = set()
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() in IMG_EXTS:
                export_images.setdefault(original_stem(p.stem), p)
            elif p.name in LIST_FILES:
                for ln in p.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        listed.add(original_stem(Path(ln.strip().replace("\\", "/")).stem))
            elif is_label_file(p, root):
                stem = original_stem(p.stem)
                if stem in label_files:
                    raise SystemExit(
                        f"Two label files for frame {stem} ({seen_as[stem]} and {p.name}). The export contains "
                        f"augmented copies - export again with NO augmentation (Roboflow: generate a version "
                        f"with augmentations turned off).")
                label_files[stem] = p
                seen_as[stem] = p.name
        if not label_files:
            raise SystemExit("No YOLO label files found in the export.")

        present = set(label_files) | set(export_images) | listed
        unmatched = sorted(s for s in present if s not in frames)
        if unmatched:
            raise SystemExit(f"{len(unmatched)} exported frame(s) have no original in {raw}, e.g. {unmatched[:5]}")

        # ---- class mapping by name
        used = Counter()
        parsed = {s: parse_label(p) for s, p in label_files.items()}
        for boxes in parsed.values():
            used.update(b[0] for b in boxes)
        id_map, problems = {}, []
        for i, n in names.items():
            if norm(n) in canon:
                id_map[i] = canon[norm(n)]
            elif used[i]:
                problems.append(f"'{n}' ({used[i]} boxes)")
        bad_ids = [i for i in used if i not in names]
        if bad_ids:
            problems.append(f"class ids {bad_ids} with no name")
        if problems:
            raise SystemExit(f"Export uses classes that are not in {classes}: {', '.join(problems)}. "
                             f"Rename/delete them in the annotation tool and export again.")
        ignored = [n for i, n in names.items() if i not in id_map]
        if ignored:
            print(f"Note: ignoring unused export class name(s) {ignored}")

        # ---- Roboflow 'Fit' resize check (a few samples)
        for stem in list(export_images)[:5]:
            if stem in frames and letterbox_suspected(export_images[stem], frames[stem][0]):
                raise SystemExit("The exported images look letterboxed (padded). Boxes would not line up with the "
                                 "original frames. In Roboflow set Preprocessing > Resize to 'Stretch' or remove it, "
                                 "and export again.")

        # ---- build the dataset
        if out.exists() and any(out.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"{out} already exists. Re-run with --overwrite to replace it.")
            for sub in ("images", "labels"):
                shutil.rmtree(out / sub, ignore_errors=True)
        stats = {"train": Counter(), "val": Counter()}
        n_img, n_neg = Counter(), Counter()
        kept_per_session, fixed = Counter(), Counter()
        for stem in sorted(present):
            img, session = frames[stem]
            split = "val" if session in val_sessions else "train"
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out / "images" / split / img.name)
            lines = []
            for cls, cx, cy, w, h in parsed.get(stem, []):
                b = clip_box(cx, cy, w, h)
                if b is None:
                    fixed["zero-size box dropped"] += 1
                    continue
                if max(abs(u - v) for u, v in zip(b, (cx, cy, w, h))) > 1e-4:
                    fixed["box clipped to the image edge"] += 1
                c = id_map[cls]
                lines.append(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                stats[split][classes[c]] += 1
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
            n_img[split] += 1
            n_neg[split] += not lines
            kept_per_session[session] += 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    import yaml

    (out / "data.yaml").write_text(
        "# Written by tools/import_reviewed.py - HUMAN-REVIEWED labels, split by session\n"
        + yaml.safe_dump({"train": "images/train", "val": "images/val", "names": dict(enumerate(classes))},
                         sort_keys=False), encoding="utf-8")
    all_sessions = Counter(s for _, s in frames.values())
    (out / "split.json").write_text(json.dumps({
        "split_by": "session", "reviewed": True, "classes": classes,
        "val_sessions": sorted(val_sessions),
        "train_sessions": sorted(s for s in all_sessions if s not in val_sessions),
        "frames_per_session": {s: kept_per_session[s] for s in sorted(all_sessions)},
        "source_export": src.name,
    }, indent=2), encoding="utf-8")

    print(f"\nReviewed dataset written to {out}")
    print(f"  {'session':10s} {'split':6s} {'kept':>5s} {'deleted':>8s}")
    for s in sorted(all_sessions):
        print(f"  {s:10s} {'val' if s in val_sessions else 'train':6s} {kept_per_session[s]:5d} "
              f"{all_sessions[s] - kept_per_session[s]:8d}")
    print(f"\n  images: train {n_img['train']} ({n_neg['train']} with no objects), "
          f"val {n_img['val']} ({n_neg['val']} with no objects)")
    print(f"  {'class':14s} {'train':>6s} {'val':>6s}   (reviewed boxes)")
    for c in classes:
        print(f"  {c:14s} {stats['train'][c]:6d} {stats['val'][c]:6d}")
    for k, v in fixed.items():
        print(f"  note: {v} {k}")
    for split in ("train", "val"):
        empty = [c for c in classes if stats[split][c] == 0]
        if empty:
            print(f"\nWARNING: no {split} boxes for {empty}. The model cannot learn/be measured on these classes.")
    if args.colab_zip:
        zpath = out.parent / f"{out.name}_colab.zip"
        extra = [ROOT / "tools" / "train.py", ROOT / "backend" / "models" / "yolo11n.pt"]
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
            for fpath in sorted(out.rglob("*")) + [e for e in extra if e.exists()]:
                if fpath.is_file():
                    z.write(fpath, fpath.relative_to(ROOT).as_posix())
        print(f"\nColab upload: {zpath}  ({zpath.stat().st_size / 1e6:.0f} MB)")
    print(f"NEXT: train on {out.relative_to(ROOT).as_posix()}/data.yaml (docs/finetune-runbook.md, step 4)")


if __name__ == "__main__":
    main()
