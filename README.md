# BAS-HAR: AI-assisted Human Activity Recognition for On-board BAS Experiments

**Prototype / controlled demonstration (SIH 2026).**

> The current prototype uses pretrained computer-vision models for perception and a controlled demonstration
> environment for activity recognition. The architecture is designed so that mission-specific BAS datasets can
> later be used for domain-specific fine-tuning and validation.

Nothing in this repository has been trained on ISRO, BAS or astronaut data. It has not been validated in a real
experiment or in space, and it is not mission-ready or flight-ready.

---

## 1. Project overview

BAS-HAR watches an experiment work area through a **live laptop webcam**. It estimates what the person is doing:

- approaching the workstation
- handling a sample container
- moving a vessel
- reaching into a restricted area

It then shows this on a local dashboard, logs every step with a timestamp, and raises visual and offline voice alerts.

## 2. Problem statement

During on-board experiments, people interact with sensitive equipment in a confined space. Plain video recording
does not tell an operator:

- which equipment is being handled
- whether steps happen in the expected order
- whether a hand enters a restricted area
- whether something moved unexpectedly

## 3. Proposed solution

The system does vision-based Human Activity Recognition using pretrained perception models and temporal reasoning.
It works in two stages:

1. **Perception.** Pretrained models detect people, objects and hands.
2. **Reasoning.** Transparent rules and state machines turn the per-frame detections into activities, sequence checks and safety alerts.

## 4. Architecture

```
LIVE WEBCAM → OpenCV → YOLO11n → MediaPipe → Human-Object Interaction → Temporal Reasoning
→ State Machine → Workflow (Procedure) Engine → Safety Engine → Events / Alerts → FastAPI → React Dashboard
```

[docs/architecture.md](docs/architecture.md) explains every layer, with its file and its limitations.

## 5. Current prototype scope

| CURRENTLY IMPLEMENTED | FUTURE ENHANCEMENTS |
|---|---|
| Live webcam input (primary), video file (fallback) | Multi-camera fusion |
| Pretrained YOLO11n (COCO) object/person detection | Detector fine-tuned on real BAS hardware |
| Pretrained MediaPipe hand landmarks | 3D hand/body pose, depth or stereo camera |
| 2D hand-object interaction estimation | True 3D contact estimation |
| Rule-based temporal state machine | Trained temporal activity model (needs labelled sequences) |
| Representative demo workflow (3 steps) | Real experiment procedures (with official documentation) |
| Rule-based safety engine (zones, restricted stand-ins, drift, missing objects) | Learned anomaly detection |
| Offline voice alerts, timestamped event log, CSV/JSON export | Mission data analytics |
| Runs locally on a laptop CPU | Optimised edge build (e.g. Jetson-class with TensorRT). Not tested yet |

## 6. AI components

| Component | Role | Trained by us? |
|---|---|---|
| **YOLO11n** (Ultralytics) | Object and person perception | No. Pretrained on COCO. |
| **MediaPipe HandLandmarker** | Hand landmark perception (21 points per hand) | No. Pretrained by Google. |
| Interaction engine | Human-object spatial relationship (2D) | Rules, no training |
| Temporal engine | Confirms an activity across several frames | Rules, no training |
| State machine | Tracks how an activity progresses | Rules, no training |
| Workflow engine | Checks the expected sequence | Rules, no training |
| Safety engine | Rule-based anomaly and safety detection | Rules, no training |

This is **AI-assisted HAR**: neural networks for perception, and explicit reasoning for activities. It is **not** a
trained end-to-end deep-learning activity classifier.

## 7. Activity recognition method

1. **Interaction estimate (per frame).** A fingertip or palm point inside the object's slightly enlarged 2D box counts as
   CONTACT. Within 1.6 × hand size counts as NEAR. Thresholds scale with hand size. This is 2D image geometry, not physical contact detection.
2. **Temporal confirmation.**
   - INTERACTING needs 4 consecutive contact frames.
   - A release needs 8 frames where the hand is visible and away from the object.
   - Contacts shorter than 0.5 s are logged as "brief" and not counted.
   - After a completed activity there is a 1.5 s cooldown per object.
3. **State machine (per object).** IDLE → HAND_NEAR_EQUIPMENT → INTERACTING → MANIPULATING (object displaced or held) → COMPLETED.
   The dashboard shows a scene-level headline:
   IDLE → PERSON_DETECTED → APPROACHING (workstation zone) → HAND_NEAR_EQUIPMENT → INTERACTING → MANIPULATING → COMPLETED.
   WARNING or CRITICAL overrides the headline.
4. **Perception drop-outs.**
   - If hand tracking is lost mid-interaction, the state is *held* and shown as **HAND TRACKING LOST**.
   - After 1.5 s it ends as "interrupted", never as "completed".
   - An object that drops out briefly is shown as **OBJECT TEMPORARILY LOST**.
5. **Activity Stability** is the share of recent frames whose evidence agrees with the current state. It is derived from
   detection confidence and temporal consistency; it is not a trained activity-classifier probability. The YOLO
   **Object Detection Confidence** is shown separately.

## 8. Safety logic

| Rule | Severity |
|---|---|
| Hand in a restricted zone for 3 frames | WARNING. Voice: "Warning. Restricted zone interaction detected." |
| Hand stays in the restricted zone for more than 3 s | CRITICAL. Voice: "Critical alert. Unsafe interaction detected." |
| Confirmed interaction with the restricted stand-in (scissors) | CRITICAL |
| Object moves while hands are tracked and none is near it | WARNING. Voice: "Warning. Unexpected activity detected." |
| All objects shift together | WARNING (camera bumped) |
| A stable object disappears without being handled | WARNING |
| Workflow step done out of order | WARNING |

- Every alert has a cooldown (8 s by default), so nothing repeats every frame.
- A zone clears only after the hand has been outside (or untracked) for 1 s.
- Every threshold is in `backend/config.yaml`. It is validated at start-up, and a unit test checks that every key is actually used.

## 9. Demo objects (BAS equipment stand-ins)

These are **representative demonstration objects**, not BAS equipment. YOLO detects the everyday object, and the
dashboard shows the stand-in role.

| Physical object | Detected YOLO class | Shown as |
|---|---|---|
| Transparent bottle | `bottle` | Sample Container Stand-in |
| Cup / mug | `cup` | Culture Vessel Stand-in |
| Mobile phone | `cell phone` | Data Tablet Stand-in |
| Scissors | `scissors` | Sharp Tool Stand-in (restricted) |
| Tray | *not a COCO class*. Marked with a **workstation zone** drawn on screen | Experimental Workstation Stand-in |

## 10. Dataset strategy

See [docs/dataset-strategy.md](docs/dataset-strategy.md). In short:

- **Now:** pretrained models, stand-in objects and rule-based reasoning, with no BAS data.
- **Future:** collect and annotate real BAS equipment and procedures, fine-tune the detector, and train a temporal model if enough sequences exist. Then validate across camera angles, lighting and occlusion.

## 11. Limitations

- **2D only.** "Contact" is a 2D overlap estimate. A hand in front of an object can look like contact. There is no depth.
- **Pretrained, generic perception.** The model knows COCO objects, not BAS hardware. Similar-looking objects confuse it. Transparent objects are harder to detect.
- **Lightweight tracking.** The IoU tracker with a centre-distance fallback is fine for a few well-separated objects. It can swap IDs when identical objects cross, and it is not mission-grade tracking.
- **Camera dependency.** Results depend on camera angle, lighting, resolution and occlusion. Hands must be visible.
- **Frame-based thresholds depend on FPS.** Re-tune them if your laptop runs much faster or slower.
- **Handedness labels.** MediaPipe's Left/Right label can be swapped on non-mirrored webcams.
- **No accuracy figures.** We have not measured detection or activity accuracy on a labelled test set, so none are quoted.
- **Performance.**
  - Measured in our cloud test environment: 2-core Intel Xeon at 2.1 GHz, no GPU, 1280×720 sample video. We saw about 17–22 processed FPS, with YOLO around 70 ms per run and MediaPipe around 35 ms per frame.
  - Your laptop will differ. The dashboard shows the live FPS.
- **The live webcam path has not been tested by us on the demo laptop** (see §12 and `docs/validation-checklist.md`).

## 12. Live demo instructions

1. Set up the table and camera using [docs/demo-guide.md](docs/demo-guide.md).
2. Run `verify.bat live` before the demo. It sends 15 s of real webcam frames through YOLO and MediaPipe.
3. Run `run.bat`. The backend starts, the script waits until it is ready, and then the browser opens http://localhost:8000.
   - Use `run.bat 1` for webcam 1.
   - Use `run.bat file` for the sample video fallback.
4. Check that **System health** is green for Camera, YOLO, MediaPipe, the activity and safety engines, and Voice.
5. Follow the 2–3 minute script in the demo guide.

## 13. Installation

Needs Windows 10/11 and **Python 3.11** (3.10–3.12 work) with "Add to PATH" ticked. Node.js is **not** needed, because the dashboard is pre-built.

```
setup.bat        # one time: creates .venv, installs packages, downloads the models into backend\models, runs verify
verify.bat live  # before every demo
run.bat          # start
```

Manual (any OS):

```bash
python -m venv .venv && .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
python backend/scripts/download_models.py               # one-time download; the demo never downloads anything
python backend/scripts/verify_setup.py --live
python backend/scripts/launch.py                        # start server → wait until ready → open browser
```

- Tests: `run_tests.bat`, or `cd backend && python -m pytest -q && python scripts/smoke_test.py`.
- Privacy: the server binds to `127.0.0.1` only. Frames are processed in memory and are **never recorded or uploaded**. Only event metadata is written to `backend/logs/`. The only video files on disk are ones you upload yourself for fallback mode.

## 14. Troubleshooting

| Symptom | Fix |
|---|---|
| Camera **ERROR** | Close Teams/Zoom/Camera. Windows Settings → Privacy & security → Camera → allow desktop apps. Try `run.bat 1`, or **⟳ Cameras** in the dashboard. |
| MediaPipe **ERROR** "Hand model not found" | Run `python backend\scripts\download_models.py` while online. |
| YOLO **ERROR** "weights not found" | Same as above. The backend never downloads at run time. |
| `mediapipe` fails to install | Use Python 3.10–3.12 (`py -3.11`). |
| Voice **ERROR** | Check that a Windows speech voice is installed. The dashboard then falls back to the browser's local voices, and visual alerts keep working. |
| Low FPS | In `config.yaml`: raise `detector.detect_every` to 3, lower `detector.imgsz` to 416, or lower `video.process_width` to 480. |
| Interaction triggers too easily or too late | Tune `interaction.contact_margin`, `interaction.near_factor` and `activity.interaction_confirm_frames`. |
| Phone detected on a bare hand | Raise `detector.class_conf.cell phone`. |

## 15. Future scope

1. Mission-specific dataset and a fine-tuned detector (see the dataset strategy).
2. A temporal activity model, if enough labelled sequences exist.
3. Depth, stereo or multi-camera fusion, and 3D pose.
4. Stronger multi-object tracking (e.g. ByteTrack or OC-SORT).
5. Learned anomaly detection.
6. A digital experiment twin driven by official procedures.
7. An edge build for Jetson-class hardware (TensorRT, FP16, dependency porting). Not tested yet.

More documents:

- [Architecture](docs/architecture.md)
- [Demo guide](docs/demo-guide.md)
- [Dataset strategy](docs/dataset-strategy.md)
- [Judge Q&A](docs/judge-qa.md)
- [Validation checklist](docs/validation-checklist.md)
- [Audit report](docs/audit-report.md)
