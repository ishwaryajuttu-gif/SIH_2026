# Dataset strategy

## WHAT EXISTS NOW (current prototype)

- **No ISRO, BAS, astronaut or mission dataset is used or included.** Nothing in this repository was trained or validated on such data.
- **Perception** uses pretrained public models:
  - Ultralytics **YOLO11n**, pretrained on COCO, for people and everyday objects.
  - Google **MediaPipe HandLandmarker**, for hand landmarks.
- **Demonstration objects** are everyday items (bottle, cup, phone, scissors, tray) acting as *BAS equipment stand-ins* in a controlled set-up.
- **Activity recognition** is rule- and state-based. It uses temporal confirmation, a state machine, a representative demo workflow and safety rules, so it needs no training data.
- **Evaluation to date:**
  - Unit tests of the reasoning logic on synthetic inputs.
  - An end-to-end run of the real models on a public sample hand video.
  - We have **not** measured detection or activity accuracy on a labelled test set, so we quote no accuracy numbers.

> The current prototype uses pretrained computer-vision models for perception and a controlled demonstration
> environment for activity recognition. The architecture is designed so that mission-specific BAS datasets can
> later be used for domain-specific fine-tuning and validation.

## WHAT IS FUTURE WORK

### Step 1: Collect mission-specific BAS data (with the owning agency)

Record video from the real camera positions around the real BAS hardware (or high-fidelity mock-ups). Cover:

- normal procedures performed by trained operators
- deliberately abnormal scenarios, run safely:
  - reaching into restricted areas
  - touching the wrong item
  - skipping or reordering steps
  - loose or drifting items
- variation in lighting, gloves, clothing, occlusion, camera angle, and several operators

### Step 2: Annotate

| Annotation | Used for |
|---|---|
| Bounding boxes of **BAS equipment** classes | Detector fine-tuning |
| **Crew / operator actions** with start and end times | Temporal activity model |
| **Hand-object interactions** (which hand, which object, contact yes/no) | Interaction thresholds, HOI model |
| **Experiment states** / procedure step boundaries | Workflow engine validation |
| **Normal vs abnormal** segment labels | Safety-rule tuning, anomaly detection |

Tools: CVAT, Label Studio or Roboflow.

- *Optional speed-up:* `tools/auto_label.py` pre-labels frames with the open-vocabulary YOLO-World model, and humans then correct them.
- Pre-labels are **never** used without human review.

### Step 3: Train and validate

1. **Fine-tune the object detector** (`tools/train.py`, transfer learning from YOLO11n). Evaluate mAP on a held-out set recorded in *different sessions* from the training data.
2. **Collect temporal activity sequences** (landmarks plus object boxes over time) from the annotated videos.
3. **If enough labelled sequences exist,** train and validate a temporal activity model (e.g. a GRU or transformer over pose and box sequences) to add to, or replace, parts of the rule-based layer. Otherwise keep the transparent rules and tune their thresholds on the data.
4. **Validate under operational conditions:**
   - different camera angles, lighting and occlusion
   - different operators and gloves
   - long sessions

   Report per-class precision and recall, false-alert rate per hour, and alert latency.
5. **Keep a frozen test set** that is never used for tuning, for the final report.

### Optional interim data (before real BAS data is available)

These sources help with *development* but prove nothing about BAS performance. Check every licence before use.

- Recordings of your own mock bench (`tools/record_dataset.py`).
- Public lab-equipment detection sets (e.g. on Roboflow Universe).
- Public hand-object interaction datasets (e.g. 100DOH, EPIC-KITCHENS, HOI4D), for testing interaction thresholds.
