"""Put several evaluate_detector.py runs side by side as a Markdown table (numbers copied from the JSON,
never typed by hand).

    python evaluation/compare_results.py ^
        --run "Pretrained YOLO11n (COCO)=evaluation/results/coco_baseline" ^
        --run "Fine-tuned (same 5 classes)=evaluation/results/finetuned_5cls" ^
        --run "Fine-tuned (all 6 classes)=evaluation/results/finetuned_all" ^
        --out docs/finetune-results.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"Label=path/to/results_dir"')
    ap.add_argument("--out", default="", help="also write the table to this Markdown file")
    args = ap.parse_args()

    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        p = Path(path)
        p = p if p.is_absolute() or p.exists() else ROOT / p
        runs.append((label.strip(), json.loads((p / "evaluation_metrics.json").read_text(encoding="utf-8"))))

    # Side by side only means something when every run scored the same images the same way.
    same = ("dataset_config", "split", "image_size", "images")
    first = runs[0][1]
    diffs = [f"  {k}: " + ", ".join(f"{label}={r.get(k)!r}" for label, r in runs)
             for k in same if any(r.get(k) != first.get(k) for _, r in runs)]
    if diffs:
        raise SystemExit("Runs are not comparable - they differ in:\n" + "\n".join(diffs) +
                         "\nRe-run evaluate_detector.py with the same --data, --split and --imgsz for every model.")
    split_json = Path(first["dataset_config"]).parent / "split.json"
    meta = json.loads(split_json.read_text(encoding="utf-8")) if split_json.exists() else {}
    classes = []
    for _, r in runs:
        classes += [c for c in r["per_class_metrics"] if c not in classes]
    lines = [
        f"Split `{first['split']}` of `{Path(first['dataset_config']).parent.name}` - {first['images']} images, "
        f"imgsz {first['image_size']}. Frame-level detection metrics (not activity accuracy).",
        "",
        "| Model | Classes scored | mAP50 | mAP50-95 | Precision | Recall |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in runs:
        o = r["overall_metrics"]
        lines.append(f"| {label} | {', '.join(r['classes_scored'])} | {o['map50']:.3f} | {o['map50_95']:.3f} | "
                     f"{o['precision']:.3f} | {o['recall']:.3f} |")
    lines += ["", "mAP50 per class (boxes in this split):", "",
              "| Class | " + " | ".join(label for label, _ in runs) + " |",
              "|---|" + "---|" * len(runs)]
    for c in classes:
        cells = []
        for _, r in runs:
            m = r["per_class_metrics"].get(c)
            cells.append(f"{m['map50']:.3f} ({m['gt_instances']})" if m else "not scored")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    if meta:
        fps = meta.get("frames_per_session", {})
        val = meta.get("val_sessions", [])
        test = meta.get("test_sessions", [])
        n = {s: sum(v for k, v in fps.items() if (k in test if s == "test" else k in val if s == "val"
                                                   else k not in val and k not in test))
             for s in ("train", "val", "test")}
        held_out = f"val = {', '.join(val)}" + (f", test = {', '.join(test)}" if test else "")
        if first["split"] == "test":
            bias = ("The scored test session was used for nothing else: not for training and not for picking the "
                    "training epoch (that was the val session).")
        else:
            bias = ("The scored session was also used to pick the best training epoch, so it is a slightly "
                    "optimistic estimate.")
        lines = [
            "## Detector fine-tune on our demo table - measured results",
            "",
            f"- Data: our own webcam recordings of the demo table, {len(fps)} sessions, "
            f"{n['train']} train / {n['val']} val" + (f" / {n['test']} test" if test else "") + " frames "
            f"({'labels reviewed by hand' if meta.get('reviewed') else 'pre-labels NOT reviewed by a human'}).",
            f"- Split by session: {held_out} (each a whole separate recording), "
            f"train = {', '.join(meta.get('train_sessions', []))}.",
            f"- Classes: {', '.join(meta.get('classes', []))}.",
            "",
        ] + lines + [
            "",
            "What these numbers do NOT show: accuracy on real BAS hardware, other rooms, tables or cameras, or "
            f"activity-recognition accuracy. {bias} All sessions show the same table, room and objects.",
        ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
        out.write_text(text, encoding="utf-8")
        print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
