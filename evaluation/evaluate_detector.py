"""Evaluation Script for Trained BAS-HAR Custom YOLO Detector.

Evaluates trained model weights against the test split defined in dataset/data.yaml.
Computes and reports real, measured metrics:
- Precision (P)
- Recall (R)
- mAP@0.50 (mAP50)
- mAP@0.50:0.95 (mAP50-95)
- F1-Score (overall and per-class)

IMPORTANT NOTE:
All numbers are generated dynamically from the actual test dataset.
Metrics are NEVER hard-coded or fabricated.

Usage:
    python evaluation/evaluate_detector.py
    python evaluation/evaluate_detector.py --model models/best.pt --split test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("bas.evaluation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "dataset" / "data.yaml"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best.pt"
BACKEND_MODEL = PROJECT_ROOT / "backend" / "models" / "best.pt"
BASE_FALLBACK_MODEL = PROJECT_ROOT / "backend" / "models" / "yolo11n.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO detector on BAS-HAR test dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to trained weights (default: models/best.pt or backend/models/best.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to dataset data.yaml file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on ('test' or 'val', default: 'test').",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Evaluation batch size (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Evaluation device ('cpu', '0', 'cuda:0'). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "results"),
        help="Directory to save evaluation reports and metrics JSON.",
    )
    return parser.parse_args()


def resolve_model_path(requested: str) -> Path | None:
    if requested:
        p = Path(requested)
        return p if p.exists() else None
    if DEFAULT_MODEL.exists():
        return DEFAULT_MODEL
    if BACKEND_MODEL.exists():
        return BACKEND_MODEL
    if BASE_FALLBACK_MODEL.exists():
        return BASE_FALLBACK_MODEL
    return None


def detect_device(requested_device: str) -> str:
    if requested_device:
        return requested_device
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def check_test_dataset(split: str) -> tuple[bool, str]:
    split_dir = PROJECT_ROOT / "dataset" / "images" / split
    if not split_dir.exists():
        return False, f"Directory does not exist: {split_dir}"
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in split_dir.iterdir() if p.suffix.lower() in valid_exts]
    if len(images) == 0:
        return (
            False,
            f"No images found in dataset/images/{split}/.\n"
            f"Please populate the '{split}' split before evaluating (see docs/custom-training.md).",
        )
    return True, f"Found {len(images)} image(s) in '{split}' split."


def calculate_f1(p: float, r: float) -> float:
    if p + r <= 0:
        return 0.0
    return 2.0 * (p * r) / (p + r)


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).resolve()

    log.info("================================================================")
    log.info("  BAS-HAR Object Detector Evaluation Engine")
    log.info("  Strict Empirical Evaluation (Zero Fabricated Metrics)")
    log.info("================================================================")

    # 1. Check Model Existence
    model_path = resolve_model_path(args.model)
    if model_path is None or not model_path.exists():
        log.error(
            "Model weights not found. Looked for:\n"
            "  1. %s\n"
            "  2. %s\n"
            "  3. %s\n"
            "Please train a model first using: python training/train_yolo.py",
            DEFAULT_MODEL,
            BACKEND_MODEL,
            BASE_FALLBACK_MODEL,
        )
        return 1

    # 2. Check Dataset Split
    split_ok, split_msg = check_test_dataset(args.split)
    if not split_ok:
        log.error("Dataset check failed:\n%s", split_msg)
        return 1
    log.info("Dataset check: %s", split_msg)

    # 3. Setup Device & Load Model
    device = detect_device(args.device)
    log.info("Evaluating model: %s", model_path)
    log.info("Evaluating split: %s on device: %s", args.split, device)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # 4. Run Ultralytics Validation / Evaluation
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(out_dir),
            name=f"eval_{args.split}",
            exist_ok=True,
            verbose=False,
        )
    except Exception as e:
        log.error("Evaluation run failed: %s", e)
        return 1

    # 5. Extract Real Metrics
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    map50 = float(box.map50)
    map50_95 = float(box.map)
    f1 = calculate_f1(precision, recall)

    # Per-class metrics
    class_names = metrics.names
    per_class = {}
    p_per_class = box.p.tolist() if hasattr(box, "p") and hasattr(box.p, "tolist") else []
    r_per_class = box.r.tolist() if hasattr(box, "r") and hasattr(box.r, "tolist") else []
    ap50_per_class = box.ap50.tolist() if hasattr(box, "ap50") and hasattr(box.ap50, "tolist") else []
    ap_per_class = box.ap.tolist() if hasattr(box, "ap") and hasattr(box.ap, "tolist") else []

    for i, cname in class_names.items():
        cp = float(p_per_class[i]) if i < len(p_per_class) else 0.0
        cr = float(r_per_class[i]) if i < len(r_per_class) else 0.0
        c50 = float(ap50_per_class[i]) if i < len(ap50_per_class) else 0.0
        c_all = float(ap_per_class[i]) if i < len(ap_per_class) else 0.0
        cf1 = calculate_f1(cp, cr)
        per_class[cname] = {
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "map50": round(c50, 4),
            "map50_95": round(c_all, 4),
            "f1": round(cf1, 4),
        }

    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "dataset_config": str(data_yaml),
        "split": args.split,
        "device": device,
        "image_size": args.imgsz,
        "overall_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
        },
        "per_class_metrics": per_class,
    }

    # 6. Save JSON & Report
    json_path = out_dir / "evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "=" * 64,
        "  BAS-HAR OBJECT DETECTOR EVALUATION REPORT",
        "=" * 64,
        f"Evaluated Model : {model_path}",
        f"Dataset Split   : {args.split}",
        f"Date & Time     : {summary['evaluation_timestamp']}",
        "-" * 64,
        "OVERALL METRICS (Strict Empirical Measurement):",
        f"  Precision (P)   : {precision:.4f} ({precision * 100:.2f}%)",
        f"  Recall (R)      : {recall:.4f} ({recall * 100:.2f}%)",
        f"  F1-Score        : {f1:.4f} ({f1 * 100:.2f}%)",
        f"  mAP @ 0.50      : {map50:.4f} ({map50 * 100:.2f}%)",
        f"  mAP @ 0.50:0.95 : {map50_95:.4f} ({map50_95 * 100:.2f}%)",
        "-" * 64,
        f"{'Class':<20} {'P':<10} {'R':<10} {'F1':<10} {'mAP50':<10} {'mAP50-95':<10}",
        "-" * 64,
    ]
    for cname, cm in per_class.items():
        report_lines.append(
            f"{cname:<20} {cm['precision']:<10.4f} {cm['recall']:<10.4f} "
            f"{cm['f1']:<10.4f} {cm['map50']:<10.4f} {cm['map50_95']:<10.4f}"
        )
    report_lines.append("=" * 64)

    report_path = out_dir / "evaluation_report.txt"
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # 7. Print Terminal Output
    print("\n" + report_text + "\n")
    log.info("Evaluation results saved to:")
    log.info("  JSON   : %s", json_path)
    log.info("  Report : %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
