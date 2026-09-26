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

Classes are matched BY NAME between the dataset and the model, so the same held-out split can be
scored for the fine-tuned model AND for the pretrained COCO model (a fair before/after comparison).
Dataset classes the model does not know (e.g. 'tray' for COCO) are reported as not evaluable.
--classes limits scoring to the named classes (use it to compare both models on the same classes).

Usage:
    python evaluation/evaluate_detector.py
    python evaluation/evaluate_detector.py --model models/best.pt --split test
    python evaluation/evaluate_detector.py --data dataset/bas_reviewed/data.yaml --split val --imgsz 480 \
        --model backend/models/yolo11n.pt --classes person bottle cup "cell phone" scissors \
        --output-dir evaluation/results/coco_baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
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
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help='Only score these dataset class names, e.g. person bottle cup "cell phone" scissors.',
    )
    return parser.parse_args()


def resolve_model_path(requested: str) -> Path | None:
    if requested:
        p = Path(requested)
        if not p.exists() and not p.is_absolute() and (PROJECT_ROOT / p).exists():
            p = PROJECT_ROOT / p
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


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_split(data_yaml: Path, split: str) -> tuple[list[str], list[Path], str]:
    """Class names and image files of one split, resolved from data.yaml (not a hard-coded folder)."""
    import yaml

    if not data_yaml.exists():
        raise ValueError(f"Dataset config file not found: {data_yaml}")
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names or [])
    if not names:
        raise ValueError(f"No class names in {data_yaml}")
    root = Path(cfg.get("path") or ".")
    if not root.is_absolute():
        root = (data_yaml.parent / root) if (data_yaml.parent / root).exists() else Path.cwd() / root
    entry = cfg.get(split)
    if not entry:
        raise ValueError(f"'{split}' is not defined in {data_yaml}")
    images: list[Path] = []
    for e in entry if isinstance(entry, list) else [entry]:
        loc = (root / e).resolve()
        if loc.is_dir():
            images += sorted(p for p in loc.rglob("*") if p.suffix.lower() in VALID_EXTS)
        elif loc.suffix == ".txt" and loc.exists():
            images += [(root / ln.strip()).resolve() for ln in loc.read_text().splitlines() if ln.strip()]
    if not images:
        raise ValueError(
            f"No images found for split '{split}' ({entry}) of {data_yaml}.\n"
            f"Please populate the '{split}' split before evaluating (see docs/custom-training.md)."
        )
    return names, images, f"Found {len(images)} image(s) in '{split}' split."


def build_aligned_split(names: list[str], images: list[Path], model_names: dict[int, str], split: str,
                        keep: set[str] | None, work: Path) -> dict:
    """Copy one split with label ids rewritten to the MODEL's ids (matched by class name).

    GT boxes of classes the model cannot predict, or outside --classes, are dropped and reported,
    so they are not counted as misses. Returns the data.yaml path and bookkeeping for the report.
    """
    from ultralytics.data.utils import img2label_paths

    by_name = {str(n).strip().lower(): i for i, n in model_names.items()}
    shutil.rmtree(work, ignore_errors=True)
    (work / "images" / split).mkdir(parents=True)
    (work / "labels" / split).mkdir(parents=True)
    gt, dropped = {}, {}
    for img, lbl in zip(images, img2label_paths([str(p) for p in images])):
        dst = work / "images" / split / img.name
        try:
            os.link(img, dst)          # hard link: no extra disk space
        except OSError:
            shutil.copy2(img, dst)
        lines = []
        lp = Path(lbl)
        for ln in (lp.read_text().splitlines() if lp.exists() else []):
            t = ln.split()
            if len(t) < 5:
                continue
            cname = names[int(t[0])]
            mid = by_name.get(cname.strip().lower())
            if mid is None or (keep is not None and cname not in keep):
                dropped[cname] = dropped.get(cname, 0) + 1
                continue
            gt[cname] = gt.get(cname, 0) + 1
            lines.append(" ".join([str(mid)] + t[1:5]))
        (work / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines))
    yaml_path = work / "data.yaml"
    import yaml

    yaml_path.write_text(yaml.safe_dump({
        "path": str(work), "train": f"images/{split}", "val": f"images/{split}", "test": f"images/{split}",
        "names": {int(k): str(v) for k, v in model_names.items()},
    }, sort_keys=False), encoding="utf-8")
    return {"yaml": yaml_path, "gt_instances": gt, "dropped": dropped}


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

    # 2. Check Dataset Split (resolved from data.yaml)
    try:
        ds_names, split_images, split_msg = load_split(data_yaml, args.split)
    except ValueError as e:
        log.error("Dataset check failed:\n%s", e)
        return 1
    log.info("Dataset check: %s", split_msg)
    keep = set(args.classes) if args.classes else None
    if keep and not keep <= set(ds_names):
        log.error("--classes %s not in dataset classes %s", sorted(keep - set(ds_names)), ds_names)
        return 1

    # 3. Setup Device & Load Model
    device = detect_device(args.device)
    log.info("Evaluating model: %s", model_path)
    log.info("Evaluating split: %s on device: %s", args.split, device)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    # 4. Run Ultralytics Validation / Evaluation on a copy of the split aligned to the model's class ids
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aligned = build_aligned_split(ds_names, split_images, model.names, args.split, keep, out_dir / "_aligned_split")
    if not aligned["gt_instances"]:
        log.error("None of the dataset classes %s can be scored with this model (%d classes).",
                  ds_names, len(model.names))
        return 1
    for cname, n in sorted(aligned["dropped"].items()):
        why = "outside --classes" if keep and cname not in keep else "model has no such class"
        log.info("Not scored: %-12s %4d boxes (%s)", cname, n, why)

    try:
        metrics = model.val(
            data=str(aligned["yaml"]),
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

    # Per-class metrics. Ultralytics stores them in the order of box.ap_class_index (classes that have
    # ground truth in this split), NOT by class id - index by position, then map back to the name.
    class_names = metrics.names
    per_class = {}
    for pos, cid in enumerate(int(c) for c in box.ap_class_index):
        cp, cr, c50, c_all = (float(v) for v in box.class_result(pos))
        per_class[class_names[int(cid)]] = {
            "gt_instances": aligned["gt_instances"].get(class_names[int(cid)], 0),
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "map50": round(c50, 4),
            "map50_95": round(c_all, 4),
            "f1": round(calculate_f1(cp, cr), 4),
        }

    summary = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_path": str(model_path),
        "dataset_config": str(data_yaml),
        "split": args.split,
        "device": device,
        "image_size": args.imgsz,
        "images": len(split_images),
        "classes_scored": sorted(per_class),
        "boxes_not_scored": aligned["dropped"],
        "note": "Overall = mean over classes_scored only. Frame-level detection metrics, not activity accuracy.",
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
        f"Dataset         : {data_yaml}",
        f"Dataset Split   : {args.split}  ({len(split_images)} images, imgsz {args.imgsz})",
        f"Date & Time     : {summary['evaluation_timestamp']}",
        "-" * 64,
        "OVERALL METRICS (Strict Empirical Measurement):",
        f"  Precision (P)   : {precision:.4f} ({precision * 100:.2f}%)",
        f"  Recall (R)      : {recall:.4f} ({recall * 100:.2f}%)",
        f"  F1-Score        : {f1:.4f} ({f1 * 100:.2f}%)",
        f"  mAP @ 0.50      : {map50:.4f} ({map50 * 100:.2f}%)",
        f"  mAP @ 0.50:0.95 : {map50_95:.4f} ({map50_95 * 100:.2f}%)",
        "-" * 64,
        f"{'Class':<14} {'Boxes':<7} {'P':<9} {'R':<9} {'F1':<9} {'mAP50':<9} {'mAP50-95':<9}",
        "-" * 64,
    ]
    for cname, cm in per_class.items():
        report_lines.append(
            f"{cname:<14} {cm['gt_instances']:<7d} {cm['precision']:<9.4f} {cm['recall']:<9.4f} "
            f"{cm['f1']:<9.4f} {cm['map50']:<9.4f} {cm['map50_95']:<9.4f}"
        )
    for cname, n in sorted(aligned["dropped"].items()):
        report_lines.append(f"{cname:<14} {n:<7d} not scored (not a class of this model, or outside --classes)")
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
