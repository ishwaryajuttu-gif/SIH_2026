"""Auto-annotate recorded frames with YOLO-World (open-vocabulary detector) -> YOLO dataset.

Optional tooling for the Level-2 data strategy (docs/dataset-strategy.md). YOLO-World detects objects from a TEXT
description ("petri dish", "pipette", "sample tube"...). We use it as a "teacher" to
pre-label your own recordings, you quickly review/fix the labels (e.g. in Label Studio,
CVAT or Roboflow), then fine-tune a small fast YOLO "student" (tools/train.py).

    python tools/auto_label.py --src dataset/raw --classes "petri dish" "test tube" "pipette" "glove"
"""
import argparse
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/raw", help="folder with recorded .jpg frames (searched recursively)")
    ap.add_argument("--out", default="dataset/bas")
    ap.add_argument("--classes", nargs="+", required=True, help='text prompts, e.g. "petri dish" "test tube"')
    ap.add_argument("--model", default="backend/models/yolov8s-worldv2.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--val", type=float, default=0.2, help="validation split fraction")
    args = ap.parse_args()

    from ultralytics import YOLO

    model_path = ROOT / args.model
    model = YOLO(str(model_path) if model_path.exists() else "yolov8s-worldv2.pt")
    model.set_classes(args.classes)

    src = ROOT / args.src
    out = ROOT / args.out
    images = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise SystemExit(f"No images found in {src}")
    random.seed(0)
    random.shuffle(images)
    n_val = max(1, int(len(images) * args.val))

    stats = {c: 0 for c in args.classes}
    for i, img in enumerate(images):
        split = "val" if i < n_val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / split / img.name)
        res = model.predict(str(img), conf=args.conf, verbose=False)[0]
        lines = []
        for (cx, cy, w, h), cls in zip(res.boxes.xywhn.tolist(), res.boxes.cls.tolist()):
            lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[args.classes[int(cls)]] += 1
        (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
        if (i + 1) % 50 == 0:
            print(f"  labelled {i + 1}/{len(images)}")

    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(args.classes))
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
    print(f"\nDataset written to {out}  ({len(images)} images, {n_val} val)")
    for c, k in stats.items():
        print(f"  {c:20s} {k} boxes")
    print("\nNEXT: review the labels (fix misses / wrong boxes), then run tools/train.py")


if __name__ == "__main__":
    main()
