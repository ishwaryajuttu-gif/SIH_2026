"""Download the pretrained models once (needs internet ONE time, during setup - never during the demo).
After this the system runs fully offline; the backend refuses to download anything at run time.

    python scripts/download_models.py            # YOLO11n + MediaPipe hand landmarker
    python scripts/download_models.py --world    # also YOLO-World (open-vocabulary, text-prompted detection)
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
HAND_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/latest/hand_landmarker.task")


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [ok] {dest.name} already present")
        return
    print(f"  downloading {dest.name} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")      # never leave a half-written model behind
    urllib.request.urlretrieve(url, tmp)
    import zipfile
    if dest.suffix == ".task" and not zipfile.is_zipfile(tmp):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"  [error] downloaded {dest.name} is not a valid model bundle - retry")
    tmp.replace(dest)
    print(f"  [ok] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB) cached locally in {dest.parent}")


def yolo(name: str):
    dest = MODELS / name
    if dest.exists():
        print(f"  [ok] {name} already present")
        return
    from ultralytics import YOLO
    import os
    cwd = os.getcwd()
    os.chdir(MODELS)           # ultralytics downloads into the working directory
    try:
        YOLO(name)
    finally:
        os.chdir(cwd)
    print(f"  [ok] {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", action="store_true", help="also download YOLO-World + CLIP text encoder")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    print("MediaPipe hand landmarker:")
    fetch(HAND_URL, MODELS / "hand_landmarker.task")
    print("YOLO object detector:")
    yolo("yolo11n.pt")
    if args.world:
        print("YOLO-World (open vocabulary):")
        yolo("yolov8s-worldv2.pt")
        from ultralytics import YOLO
        m = YOLO(str(MODELS / "yolov8s-worldv2.pt"))
        m.set_classes(["petri dish", "test tube"])   # triggers the one-time CLIP download
        print("  [ok] CLIP text encoder cached")
    print("\nAll models ready. The system can now run offline.")


if __name__ == "__main__":
    sys.exit(main())
