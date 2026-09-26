"""Export the detector for edge deployment.

    python tools/export_edge.py --format onnx              # portable, CPU-friendly (any PC)
    python tools/export_edge.py --format openvino          # fast on Intel CPUs / iGPUs
    python tools/export_edge.py --format engine --half     # TensorRT FP16 - run this ON the Jetson

After export, point detector.model (or model_path) in backend/config.yaml at the exported file.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="backend/models/yolo11n.pt")
    ap.add_argument("--format", default="onnx", choices=["onnx", "openvino", "engine", "tflite", "ncnn"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="FP16 (TensorRT / GPU)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(ROOT / args.weights))
    path = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
