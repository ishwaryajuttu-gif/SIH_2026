"""Auto-annotate recorded frames with YOLO-World (open-vocabulary detector) -> YOLO dataset.

Optional tooling for the Level-2 data strategy (docs/dataset-strategy.md). YOLO-World detects objects from a TEXT
description ("petri dish", "pipette", "sample tube"...). We use it as a "teacher" to
pre-label your own recordings, you review/fix the labels (CVAT, Label Studio or Roboflow),
then fine-tune a small fast YOLO "student" (tools/train.py). Pre-labels are never trained on unreviewed.

The train/val split is BY SESSION FOLDER, never by random frame. Frames are expected in
dataset/raw/<session>/*.jpg (tools/record_dataset.py --name <session> does this). Neighbouring
video frames are near-duplicates, so a random frame split leaks into validation and inflates mAP.
Whole sessions go to val (default: the last session, bench10 after bench9); the rest go to train.
Optionally, --test-session keeps whole sessions for a final score that nothing was tuned on (val picks the
best training epoch, so a val score is slightly optimistic).

    python tools/auto_label.py --src dataset/raw --classes person bottle cup "cell phone" scissors tray
        --val-session bench5 --test-session bench6

Class ids follow the order of --classes exactly. Output (default dataset/bas):
    images/{train,val[,test]}/  labels/{train,val[,test]}/  data.yaml  split.json  and  <out>_upload.zip (for CVAT)
"""
import argparse
import json
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def natural_key(name: str):
    """bench2 < bench10 (plain sorting would put bench10 first)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def split_of(session: str, val_sessions, test_sessions) -> str:
    return "test" if session in test_sessions else "val" if session in val_sessions else "train"


def find_sessions(src: Path) -> dict[str, list[Path]]:
    """session name (first folder below src) -> sorted frames."""
    sessions: dict[str, list[Path]] = defaultdict(list)
    loose = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        rel = p.relative_to(src)
        if len(rel.parts) < 2:
            loose.append(p)
        else:
            sessions[rel.parts[0]].append(p)
    if loose:
        raise SystemExit(
            f"{len(loose)} image(s) sit directly in {src} (e.g. {loose[0].name}). Put every frame in a session "
            f"folder such as {src / 'bench1'} - the train/val split is made per session.")
    return dict(sessions)


def write_data_yaml(out: Path, classes: list[str], has_test: bool = False):
    import yaml

    # No 'path:' key on purpose: Ultralytics then resolves images relative to this file, so the
    # same dataset folder works on your laptop, in Colab and on Kaggle without editing.
    # No 'nc:' key either: CVAT's importer treats every unknown key as a subset name and fails.
    data = {"train": "images/train", "val": "images/val"}
    if has_test:
        data["test"] = "images/test"
    data["names"] = {i: c for i, c in enumerate(classes)}
    (out / "data.yaml").write_text(
        "# Written by tools/auto_label.py - class order must match backend/config.yaml detector.class_map names\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_upload_zip(out: Path) -> Path:
    """data.yaml + images/<split> + labels/<split> at the zip root = CVAT 'Ultralytics YOLO Detection 1.0'."""
    zpath = out.parent / f"{out.name}_upload.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:   # JPEGs don't compress; STORED is fast
        z.write(out / "data.yaml", "data.yaml")
        for sub in ("images", "labels"):
            for f in sorted((out / sub).rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(out).as_posix())
    return zpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/raw", help="folder with one sub-folder of .jpg frames per session")
    ap.add_argument("--out", default="dataset/bas")
    ap.add_argument("--classes", nargs="+", required=True, help='text prompts = class names, in class-id order')
    ap.add_argument("--model", default="backend/models/yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--val-session", nargs="+", default=None,
                    help="session folder(s) used ONLY for validation (default: the last session, natural order)")
    ap.add_argument("--test-session", nargs="+", default=[],
                    help="optional session folder(s) held out for the final score (never used for training or "
                         "epoch selection)")
    ap.add_argument("--agnostic-nms", action=argparse.BooleanOptionalAction, default=True,
                    help="drop a lower-confidence box of another class on the same object (e.g. cup AND bottle)")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing --out dataset")
    args = ap.parse_args()

    src = ROOT / args.src
    out = ROOT / args.out
    sessions = find_sessions(src)
    if len(sessions) < 2:
        raise SystemExit(f"Found {len(sessions)} session folder(s) in {src}; need at least 2 (train + val).")
    test_sessions = list(args.test_session)
    val_sessions = args.val_session or [sorted((s for s in sessions if s not in test_sessions), key=natural_key)[-1]]
    unknown = [s for s in val_sessions + test_sessions if s not in sessions]
    if unknown:
        raise SystemExit(f"--val-session/--test-session {unknown} not found. Sessions: {sorted(sessions)}")
    if set(val_sessions) & set(test_sessions):
        raise SystemExit(f"{sorted(set(val_sessions) & set(test_sessions))} is in both val and test.")
    if len(set(val_sessions) | set(test_sessions)) >= len(sessions):
        raise SystemExit("Every session is in val/test - leave at least one for training.")
    train_sessions = sorted((s for s in sessions if split_of(s, val_sessions, test_sessions) == "train"), key=natural_key)

    names = Counter(p.name for frames in sessions.values() for p in frames)
    dupes = [n for n, k in names.items() if k > 1]
    if dupes:
        raise SystemExit(f"Same file name in several sessions (e.g. {dupes[0]}). Record each session with its own "
                         f"--name so file names are unique.")

    # A stale dataset would keep old frames in the wrong split -> leakage. Start clean.
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} already exists. Re-run with --overwrite to replace it.")
        for sub in ("images", "labels"):
            shutil.rmtree(out / sub, ignore_errors=True)

    from ultralytics import YOLO

    model_path = ROOT / args.model
    model = YOLO(str(model_path) if model_path.exists() else "yolov8s-worldv2.pt")
    model.set_classes(args.classes)   # first run downloads the CLIP text encoder (needs internet once)

    print("Class ids (keep this order everywhere):")
    for i, c in enumerate(args.classes):
        print(f"  {i}: {c}")
    print(f"Sessions: {', '.join(f'{s} ({len(f)})' for s, f in sorted(sessions.items()))}")
    print(f"VAL = {', '.join(val_sessions)}   TEST = {', '.join(test_sessions) or '-'}   "
          f"TRAIN = {', '.join(train_sessions)}\n")

    stats = {"train": Counter(), "val": Counter(), "test": Counter()}
    n_imgs = Counter()
    total = sum(len(f) for f in sessions.values())
    done = 0
    for session, frames in sorted(sessions.items()):
        split = split_of(session, val_sessions, test_sessions)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in frames:
            shutil.copy2(img, out / "images" / split / img.name)
            res = model.predict(str(img), conf=args.conf, agnostic_nms=args.agnostic_nms, verbose=False)[0]
            lines = []
            for (cx, cy, w, h), cls in zip(res.boxes.xywhn.tolist(), res.boxes.cls.tolist()):
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                stats[split][args.classes[int(cls)]] += 1
            (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
            n_imgs[split] += 1
            done += 1
            if done % 50 == 0:
                print(f"  labelled {done}/{total}")

    write_data_yaml(out, args.classes, has_test=bool(test_sessions))
    (out / "split.json").write_text(json.dumps({
        "split_by": "session",
        "classes": args.classes,
        "val_sessions": sorted(val_sessions),
        "test_sessions": sorted(test_sessions),
        "train_sessions": train_sessions,
        "frames_per_session": {s: len(f) for s, f in sorted(sessions.items())},
    }, indent=2), encoding="utf-8")
    zpath = write_upload_zip(out)

    splits = ["train", "val"] + (["test"] if test_sessions else [])
    print(f"\nDataset written to {out}  (" + ", ".join(f"{s} {n_imgs[s]} images" for s in splits) + ")")
    print(f"  {'class':14s} " + " ".join(f"{s:>6s}" for s in splits) + "   (pre-label boxes, NOT reviewed)")
    for c in args.classes:
        print(f"  {c:14s} " + " ".join(f"{stats[s][c]:6d}" for s in splits))
    for s in splits[1:]:
        missing = [c for c in args.classes if stats[s][c] == 0]
        if missing:
            print(f"\nWARNING: no pre-label boxes for {missing} in the {s} session. If the object really is in "
                  f"those frames, add the boxes during review; if it is not, record it in the {s} session too.")
    print(f"\nUpload for review: {zpath}")
    print("NEXT: review the labels by hand (docs/finetune-runbook.md, step 3), then tools/import_reviewed.py")


if __name__ == "__main__":
    main()
