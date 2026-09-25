"""FUTURE WORK: fine-tune a small YOLO model on a domain-specific dataset (transfer learning).

    python tools/train.py --data dataset/bas/data.yaml --epochs 60

The best weights are copied to backend/models/bas_custom.pt. Then in backend/config.yaml set
    detector.mode: custom
and map the new class names in detector.class_map if you want nicer labels.
CPU training works for a few hundred images (slow but fine); a free Colab/Kaggle GPU is much faster.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/bas/data.yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="pretrained starting point (COCO)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="'cpu', '0' for GPU ...")
    args = ap.parse_args()

    from ultralytics import YOLO

    base = ROOT / "backend" / "models" / args.base
    model = YOLO(str(base) if base.exists() else args.base)
    res = model.train(
        data=str(ROOT / args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, project=str(ROOT / "runs"), name="bas", exist_ok=True,
        # augmentations that mimic on-board conditions: lighting changes, blur, rotation (no "up" in microgravity)
        hsv_v=0.5, degrees=15, flipud=0.3, fliplr=0.5, mosaic=1.0, patience=20,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    dest = ROOT / "backend" / "models" / "bas_custom.pt"
    shutil.copy2(best, dest)
    print(f"\nSaved {dest}\nSet detector.mode: custom in backend/config.yaml to use it.")


if __name__ == "__main__":
    main()
