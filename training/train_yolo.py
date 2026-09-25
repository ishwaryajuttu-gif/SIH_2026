"""Custom YOLO11n Training Pipeline for BAS-HAR Stand-in Objects.

Trains Ultralytics YOLO on representative demonstration objects:
1. person
2. sample_container
3. culture_vessel
4. data_tablet
5. restricted_tool

IMPORTANT NOTE:
These are laboratory demonstration stand-ins used to validate explainable
human activity recognition. They are NOT official ISRO BAS flight hardware.

Usage:
    python training/train_yolo.py
    python training/train_yolo.py --epochs 50 --batch 16 --imgsz 640
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bas.training")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "data.yaml"
DEFAULT_PRETRAINED = PROJECT_ROOT / "backend" / "models" / "yolo11n.pt"
FALLBACK_PRETRAINED = "yolo11n.pt"
DEFAULT_TARGET_MODEL = PROJECT_ROOT / "models" / "best.pt"
BACKEND_TARGET_MODEL = PROJECT_ROOT / "backend" / "models" / "best.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLO11 on BAS-HAR custom stand-in dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to dataset data.yaml file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_PRETRAINED) if DEFAULT_PRETRAINED.exists() else FALLBACK_PRETRAINED,
        help="Initial pretrained weights (e.g. yolo11n.pt or path to existing weights).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30, student laptop friendly).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size (default: 8, safe for laptops and integrated/low-VRAM GPUs).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (default: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use ('cpu', '0', 'cuda:0'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Dataloader workers (default: 2, avoids Windows multiprocessing issues).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(PROJECT_ROOT / "training" / "runs"),
        help="Directory to save training run logs and checkpoints.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="bas_standins",
        help="Name of this training experiment run.",
    )
    return parser.parse_args()


def detect_device(requested_device: str) -> str:
    """Auto-detect CUDA GPU availability; fallback gracefully to CPU."""
    if requested_device:
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            log.info("CUDA GPU detected: %s (using device '0')", dev_name)
            return "0"
    except ImportError:
        pass
    log.info("No CUDA GPU detected or torch not imported. Using CPU for training.")
    return "cpu"


def check_dataset_ready(data_yaml_path: Path) -> tuple[bool, str]:
    """Verify that dataset directory and at least one training image exist."""
    if not data_yaml_path.exists():
        return False, f"Dataset config file not found: {data_yaml_path}"

    train_img_dir = PROJECT_ROOT / "dataset" / "images" / "train"
    if not train_img_dir.exists():
        return False, f"Training images directory missing: {train_img_dir}"

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in train_img_dir.iterdir() if p.suffix.lower() in valid_exts]
    if len(images) == 0:
        return (
            False,
            f"No training images found in {train_img_dir}.\n"
            f"To train the model, please add collected images to dataset/images/train/ "
            f"and labels to dataset/labels/train/ (see docs/custom-training.md).",
        )
    return True, f"Found {len(images)} training image(s)."


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).resolve()

    log.info("================================================================")
    log.info("  BAS-HAR Custom YOLO11n Training Pipeline")
    log.info("  Representative Stand-in Objects (Not official ISRO hardware)")
    log.info("================================================================")

    # 1. Dataset Readiness Check
    is_ready, msg = check_dataset_ready(data_yaml)
    if not is_ready:
        log.error("Dataset check failed:\n%s", msg)
        return 1
    log.info("Dataset check: %s", msg)

    # 2. Hardware Selection
    device = detect_device(args.device)
    log.info("Training device selected: %s", device)
    log.info("Hyperparameters: epochs=%d, batch=%d, imgsz=%d", args.epochs, args.batch, args.imgsz)

    # 3. Load YOLO Pretrained Architecture
    from ultralytics import YOLO

    weights_path = Path(args.weights)
    if weights_path.exists():
        log.info("Loading base weights from local path: %s", weights_path)
        model = YOLO(str(weights_path))
    else:
        log.info("Loading base weights from Ultralytics: %s", args.weights)
        model = YOLO(args.weights)

    # 4. Execute Fine-Tuning
    log.info("Starting YOLO training run '%s' on %s...", args.name, data_yaml)
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )
    except Exception as e:
        log.error("Training failed with error: %s", e)
        return 1

    # 5. Extract and Deploy Trained Weights
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / args.name
    best_pt = save_dir / "weights" / "best.pt"

    if best_pt.exists():
        log.info("Training complete! Best weights produced at: %s", best_pt)

        # Copy to primary models/best.pt
        DEFAULT_TARGET_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, DEFAULT_TARGET_MODEL)
        log.info("Copied custom model to: %s", DEFAULT_TARGET_MODEL)

        # Also copy to backend/models/best.pt for direct backend access
        BACKEND_TARGET_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, BACKEND_TARGET_MODEL)
        log.info("Copied custom model to: %s", BACKEND_TARGET_MODEL)

        log.info("================================================================")
        log.info("Custom model successfully trained and deployed!")
        log.info("Run evaluation: python evaluation/evaluate_detector.py")
        log.info("Run application: python backend/scripts/launch.py (or run.bat)")
        log.info("================================================================")
        return 0
    else:
        log.warning("Training finished but weights/best.pt not found in %s", save_dir)
        return 1


if __name__ == "__main__":
    sys.exit(main())
