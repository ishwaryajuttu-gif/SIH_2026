# BAS-HAR Custom Object Detector Training & Evaluation Guide

This guide details the custom YOLO11 training pipeline for the **BAS-HAR** (AI Human Activity Recognition for On-board BAS Experiments) prototype.

---

## 1. Overview of the Custom Dataset

The default BAS-HAR prototype utilizes a pretrained YOLO11n model trained on the COCO dataset (mapping everyday objects like bottles and cups to equipment stand-ins). The custom training pipeline enables fine-tuning the YOLO architecture to detect domain-specific, BAS-like laboratory stand-in objects directly.

This bridges the gap between generic computer vision classes and specialized experiment apparatus without altering the downstream explainable temporal state machine, hand tracking, or safety rule verification engine.

---

## 2. The Five Target Stand-in Classes

The custom dataset defines five discrete object categories:

| Class ID | Class Name | Description & Laboratory Stand-in Role |
| :---: | :--- | :--- |
| `0` | `person` | Experimenter or operator present in the field of view. |
| `1` | `sample_container` | Test tubes, centrifuge vials, or capped sample bottles used for biological assays. |
| `2` | `culture_vessel` | Petri dishes, flask stand-ins, or specimen wells used for cell/microbial growth. |
| `3` | `data_tablet` | Tablet, digital notebook, or interface display used for procedure logging. |
| `4` | `restricted_tool` | Sharp instruments, scissors, or high-hazard tools requiring safety boundary alerts. |

---

## 3. How to Collect Dataset Images

High-quality, diverse image collection is essential for reliable tracking and interaction detection:

1. **Camera Angles & Positions:**
   - Capture images from the primary webcam perspective matching the demonstration layout (top-down, 45-degree angled view, and frontal view).
2. **Lighting Variations:**
   - Record under direct desk illumination, diffuse room lighting, and slight shadows to prevent over-reliance on brightness.
3. **Occlusions and Hands in Contact:**
   - Crucially, capture objects while being approached, held, and partially occluded by one or both hands (fingers grasping, palm resting).
   - Capture objects placed at various locations across the workstation tray.
4. **Negative Samples & Clutter:**
   - Include images of the empty workstation and background laboratory environment without stand-ins to minimize false positives.
5. **Collection Tools:**
   - You can use the built-in webcam collection tool:
     ```bash
     python tools/record_dataset.py
     ```
   - Alternatively, extract still frames at 2–3 FPS from webcam recordings to ensure variance across consecutive frames.

---

## 4. How to Annotate Images

Annotations must follow standard **YOLO bounding-box format**:
- Each image (`image_name.jpg`) has an accompanying text file (`image_name.txt`) with the exact same base name.
- Each line in the `.txt` file represents one object instance:
  ```text
  <class_id> <x_center> <y_center> <width> <height>
  ```
- All coordinates are normalized between `0.0` and `1.0` relative to image dimensions.

### Recommended Annotation Tools
- **CVAT (Computer Vision Annotation Tool):** Open-source, web-based tool supporting YOLO export.
- **Label Studio:** Self-hosted multi-modal annotation tool.
- **Roboflow:** Web-based labeling tool with direct YOLOv11 export options.

### Semi-Automated Labeling
You can run the existing auto-label helper script in `tools/auto_label.py` to bootstrap initial bounding boxes before manual correction.

---

## 5. Dataset Train / Validation / Test Split

Organize your annotated dataset into a standard 70% / 20% / 10% split:

```text
dataset/
├── images/
│   ├── train/    # ~70% of collected images
│   ├── val/      # ~20% for validation during training
│   └── test/     # ~10% held-out images for final evaluation
├── labels/
│   ├── train/    # YOLO .txt labels for train images
│   ├── val/      # YOLO .txt labels for val images
│   └── test/     # YOLO .txt labels for test images
└── data.yaml     # Dataset configuration descriptor
```

Ensure that:
1. Every image in `images/<split>/` has a corresponding `.txt` in `labels/<split>/` (images with no target objects can have empty `.txt` files).
2. The `dataset/data.yaml` points to `images/train`, `images/val`, and `images/test`.

---

## 6. How to Run Custom YOLO Training

Execute the training script from the project root:

```bash
python training/train_yolo.py
```

### Configurable Options
```bash
python training/train_yolo.py --epochs 40 --batch 8 --imgsz 640 --device cpu
```

- `--epochs`: Number of passes over the dataset (default: `30`, optimized for student laptops).
- `--batch`: Batch size (default: `8`, low VRAM and memory friendly).
- `--imgsz`: Training image resolution (default: `640`).
- `--device`: Automatic selection: selects CUDA device `0` if available, otherwise runs on `cpu`.
- `--weights`: Pretrained starting weights (defaults to local `backend/models/yolo11n.pt`).

---

## 7. Where `best.pt` Is Generated & Deployed

Upon training completion, the best checkpoint based on validation loss and mAP is saved to:
1. `training/runs/bas_standins/weights/best.pt` (training artifact).
2. Automatically copied to **`models/best.pt`** (project-level model storage).
3. Automatically copied to **`backend/models/best.pt`** (backend runtime access).

The backend is configured in `backend/config.yaml` to load `models/best.pt`:
```yaml
detector:
  mode: custom
  model_path: models/best.pt
```

### Safe Fallback Mechanism
If `models/best.pt` does not exist (e.g. before training is executed), the backend **does not crash**. It logs a clear warning and automatically falls back to `backend/models/yolo11n.pt`:
- UI Status read-out displays: `BASE PRETRAINED MODEL (fallback: custom model not found)`
- When `models/best.pt` exists: UI Status read-out displays `CUSTOM MODEL`.

---

## 8. How to Run Evaluation

Run the evaluation script to test the model on the held-out test split:

```bash
python evaluation/evaluate_detector.py
```

Optional arguments:
```bash
python evaluation/evaluate_detector.py --model models/best.pt --split test --imgsz 640
```

The script evaluates the weights and outputs:
- Real-time terminal metrics summary.
- Comprehensive JSON metrics file: `evaluation/results/evaluation_metrics.json`.
- Formatted text report: `evaluation/results/evaluation_report.txt`.

---

## 9. Metrics Reported

The evaluation script computes strictly empirical, non-fabricated metrics:

1. **Precision (P):** Ratio of correctly predicted positive observations to total predicted positive observations.
2. **Recall (R):** Ratio of correctly predicted positive observations to all observations in the ground truth.
3. **F1-Score:** Harmonic mean of Precision and Recall ($F_1 = 2 \cdot \frac{P \cdot R}{P + R}$).
4. **mAP @ 0.50 (mAP50):** Mean Average Precision at an Intersection over Union (IoU) threshold of 0.50.
5. **mAP @ 0.50:0.95 (mAP50-95):** Mean Average Precision averaged over 10 IoU thresholds from 0.50 to 0.95 in increments of 0.05.
6. **Per-Class Breakdown:** Individual P, R, F1, and mAP metrics for each of the 5 custom classes.

> **CRITICAL RULE:** Accuracy and performance numbers must never be invented or hard-coded. They must always originate from running `evaluate_detector.py` on real annotated test data.

---

## 10. Important Scope Limitation & Flight Hardware Disclaimer

- **Representative Stand-in Objects:** The items used in this prototype (commercial sample bottles, culture dishes, tablets, scissors) are **demonstration stand-ins** designed to validate explainable temporal activity recognition, hand tracking, and safety alerting algorithms in a university laboratory setup.
- **Not Official ISRO Flight Hardware:** These objects are not official ISRO Biological Experiment / Biological Apparatus System (BAS) flight hardware.
- **Future Production Path:** Transitioning from this proof-of-concept prototype to a flight-qualified system would require:
  1. High-fidelity imaging of actual BAS mission apparatus and payload enclosures.
  2. Mission-specific training datasets captured under simulated microgravity operational lighting.
  3. Formal flight-software verification and validation under Space Grade software engineering standards.
