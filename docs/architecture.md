# BAS-HAR architecture

> The current prototype uses pretrained computer-vision models for perception and a controlled demonstration
> environment for activity recognition. The architecture is designed so that mission-specific BAS datasets can
> later be used for domain-specific fine-tuning and validation.

```
 LIVE WEBCAM  (primary; a video file is only a fallback)
     │  frames in memory only - nothing is recorded
     ▼
 OpenCV capture ─────────── backend/app/pipeline.py  (VideoSource, open_camera, probe_cameras)
     │  newest frame, downscaled to 640 px, capped at 15 FPS
     ▼
 YOLO11n (pretrained) ───── backend/app/detector.py  (ObjectDetector + SimpleTracker)
     │  person + stand-in objects, boxes, detection confidence, track IDs
     ▼
 MediaPipe (pretrained) ─── backend/app/hands.py     (HandTracker)
     │  21 landmarks per hand
     ▼
 Human-Object Interaction ─ backend/app/interaction.py   2D FAR / NEAR / CONTACT per hand-object pair
     ▼
 Temporal reasoning ──────┐
     ▼                    ├ backend/app/activity.py   confirm / release / min-duration / cooldown / grace periods
 State machine ───────────┘                           IDLE → HAND_NEAR → INTERACTING → MANIPULATING → COMPLETED
     ▼
 Procedure engine ───────── backend/app/procedure.py  Representative Demo Workflow (sequence check)
     ▼
 Safety engine ──────────── backend/app/safety.py     zones, restricted stand-ins, drift, missing, cooldowns
     ▼
 Events / Alerts ────────── backend/app/events.py (JSONL log, CSV/JSON export) · voice.py (offline TTS)
     │                      scene.py (headline state) · overlay.py (annotated frame)
     ▼
 FastAPI ────────────────── backend/app/main.py       MJPEG /video_feed · WebSocket /ws/state · REST
     ▼
 React dashboard ────────── frontend/src              live view, zones, activity, workflow, alerts, timeline, health
```

## Layers

### 1. Live webcam and OpenCV (`pipeline.py`)

- **Camera opening.** `open_camera()` tries the Windows capture back-ends in order (DirectShow, then Media Foundation, then any other). A camera only counts as "connected" after it has delivered a real frame.
- **Camera discovery.** `probe_cameras()` lists webcams 0–4. The dashboard's **⟳ Cameras** button runs it again.
- **Background capture.** A thread keeps only the newest frame, so lag never builds up.
- **Reported camera states:**
  - `connecting`
  - `connected`
  - `stalled`: no new frame for 3 s
  - `error`: with the reason, e.g. the camera is used by another app, or privacy settings block it
  - `ended`: a video file finished
- **No fake fallback.** If the webcam fails, the system reports the error. It never switches to demo data silently.

### 2. YOLO11n: object and person perception (`detector.py`)

- **Model.** Pretrained Ultralytics YOLO11n on COCO. It is loaded from `backend/models/yolo11n.pt`, and a missing file is an **error**: nothing is downloaded at run time.
- **Classes.** COCO classes are mapped to stand-in labels in `config.yaml` (`class_map`). There is **no BAS-specific training.**
- **Speed settings.** `detect_every: 2` runs YOLO on every 2nd analysed frame to save CPU; the tracker holds the boxes in between. The input size is 480.
- **Tracker.** `SimpleTracker` is a greedy IoU tracker with a same-label centre-distance fallback for fast moves.
  - An object must be seen `min_hits` times before it is trusted.
  - Boxes are held for `max_missed` detector runs while an object is hidden by a hand.
  - It is adequate for a few well-separated objects in a controlled demo. It is **not mission-grade**: identical objects that cross can swap IDs.
  - A future path is ByteTrack or OC-SORT with appearance features.

### 3. MediaPipe: hand landmark perception (`hands.py`)

- Uses the pretrained HandLandmarker (Tasks API, VIDEO mode, up to 2 hands). The model file is prepared by `setup.bat`.
- Duplicate hands (the same hand reported twice) are removed.
- If the model is missing, the component shows **ERROR** with the reason.

### 4. Human-object interaction (`interaction.py`)

Per hand and object, it uses the fingertips (thumb, index, middle) and the palm centre:

- **CONTACT:** a key point lies inside the object box, enlarged by 12%.
- **NEAR:** the distance to the box is at most 1.6 × hand size. Hand size is the wrist-to-middle-knuckle length, so this works at any distance from the camera.
- **FAR:** otherwise.

This is **2D vision-based interaction estimation**, not physical contact detection. The strongest (closest) hand per object is used.

### 5–6. Temporal reasoning and state machine (`activity.py`)

Each object has a state machine, and every transition needs temporal evidence:

| Transition | Evidence required (config key) |
|---|---|
| IDLE → HAND_NEAR_EQUIPMENT | `approach_confirm_frames` (3) consecutive NEAR frames |
| → INTERACTING | `interaction_confirm_frames` (4) consecutive CONTACT frames. A single frame never counts, and the validator rejects a value of 1. |
| INTERACTING → MANIPULATING | Object displaced ≥ `move_threshold` (6% of the frame diagonal), or hidden in the hand for `missing_as_held_frames` |
| → COMPLETED | Hand **visible** and not in contact for `release_confirm_frames` (8), and the object visible again |
| Too short | Duration < `min_interaction_s`: logged as BRIEF_CONTACT_IGNORED, not counted |
| COMPLETED → IDLE | After `interaction_cooldown_s`. No new interaction on that object until then. |
| Hand tracking lost | State is held (HAND TRACKING LOST). After `hand_lost_grace_s` it ends as INTERACTION_ENDED_TRACKING_LOST, never COMPLETED. |
| Object track dropped mid-interaction | OBJECT_LOST_DURING_INTERACTION, not counted as completed |

- **Presence.** PERSON_DETECTED needs `person_confirm_frames`, and PERSON_LEFT needs `person_lost_frames`.
- **Headline state** (`scene.py`): IDLE → PERSON_DETECTED → APPROACHING (hand in the workstation zone) → HAND_NEAR_EQUIPMENT → INTERACTING → MANIPULATING → COMPLETED. Safety WARNING or CRITICAL overrides the headline.
- **Activity Stability.** The share of the last `stability_window` frames whose evidence agrees with the current state. It is a heuristic for temporal consistency, **not** a classifier probability. The YOLO detection confidence is reported separately.

### 7. Procedure engine: Representative Demo Workflow (`procedure.py`)

- Completed activities are matched against the steps in `config.yaml → workflow`.
- A `move` step needs a real move; an `interact` step accepts any confirmed handling.
- A later step done early raises OUT_OF_SEQUENCE (WARNING). Repeating an earlier step is logged as info.
- This workflow is a representative demonstration sequence created to validate the prototype's activity-recognition architecture. It is **not** an official BAS/ISRO protocol.

### 8. Safety engine (`safety.py`)

- **Rule-based by design**: transparent, testable, and needs no training data.
- **Zones** are polygons in normalised coordinates, drawn in the dashboard (rectangle or polygon) and saved to `backend/data/zones.json`.
  - **Restricted zone:**
    - A hand present for `restricted_confirm_frames` gives WARNING.
    - Still inside after `critical_after_s` gives CRITICAL.
    - The zone clears only after `zone_exit_s` outside or untracked, so a tracking drop does not clear it and re-trigger.
  - **Workstation zone:** a hand inside emits APPROACHING_WORKSTATION.
  - **Enable toggle:** each zone can be switched on or off.
- **Restricted stand-in:** a confirmed interaction gives CRITICAL.
- **Unexpected movement:** an object moves more than `unexpected_move_threshold` while hands were tracked in at least 60% of the window and none was near it. This gives WARNING. It is *not* raised when hand tracking was lost, to avoid blaming a normal pick-up.
- **Camera shift:** all objects move together, giving WARNING.
- **Missing object:** a stable object disappears without having been handled, giving WARNING.
- **Cooldown:** every alert uses `alert_cooldown_s` per zone or object.

### 9. Events, alerts and voice (`events.py`, `voice.py`)

- **Event fields:** timestamp, event_type, activity (e.g. SAMPLE_CONTAINER_INTERACTION), state, severity, subject (object or zone), stability, description and data.
- **Storage.** Events are kept in memory and appended to `backend/logs/session_*.jsonl` (metadata only). They can be exported as CSV or JSON.
- **Voice.** Offline pyttsx3 runs in its own thread and speaks three fixed phrases, with a per-phrase cooldown.
  - If TTS fails to initialise, the voice component shows ERROR and visual alerts continue.
  - The dashboard can then use the browser's local OS voices as a fallback.

### 10. FastAPI and React (`main.py`, `frontend/`)

- **Server.** Binds to `127.0.0.1`. It serves the pre-built dashboard, the annotated MJPEG stream (JPEG encoding only happens while someone is watching), the WebSocket state (8 Hz) and REST endpoints (see `/docs`).
- **System health** shows each component: Camera, YOLO, MediaPipe, Activity engine, Safety engine, Voice, plus FPS.
  - Green appears only with evidence, e.g. a model loaded and ran recently, or the camera delivered frames within the last 3 s.
  - Otherwise the state is amber (waiting or muted) or red (error), with the reason.

## Threading model

| Thread | Job |
|---|---|
| Camera | Reads frames |
| Pipeline | Runs all inference and logic under one lock |
| Voice | Speaks queued phrases |
| Uvicorn | Serves the API |

API calls that change state (source, zones, workflow reset) take the same lock, so they never race with frame processing.

## Known limitations and future path

| Area | Now | Future |
|---|---|---|
| Geometry | 2D image-plane overlap | Depth camera, stereo, multi-camera fusion, 3D hand/body pose |
| Tracking | IoU + centre-distance | ByteTrack / OC-SORT with re-identification |
| Activities | Rules on top of pretrained perception | Temporal model (e.g. transformer or GRU over landmark and box sequences) trained on labelled BAS sequences |
| Objects | COCO stand-ins | Detector fine-tuned on BAS hardware |
| Deployment | Laptop CPU | Designed with edge deployment in mind; can be optimised for Jetson-class hardware using compatible dependencies, TensorRT, FP16 inference and hardware-specific tuning. **Not tested on Jetson.** |
