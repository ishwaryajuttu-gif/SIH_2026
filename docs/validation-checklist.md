# Validation checklist

## A. What was verified automatically (in the cloud build environment)

The build environment has **no webcam and no speakers**. Only the items below were actually run there.

| Check | How | Result |
|---|---|---|
| Python compile/import of all backend modules | `python -m compileall`, import `app.main` | passed |
| Unit tests: reasoning logic, config, events, voice mapping, tracker, camera-error path | `pytest tests/test_logic.py` | see the final report |
| Integration tests: real YOLO + MediaPipe on the sample video through the FastAPI app | `pytest tests/test_integration.py` | see the final report |
| Real-server smoke test: health, MJPEG stream, CSV/JSON export, dashboard | `python scripts/smoke_test.py` | passed |
| Frontend type-check, lint, production build | `npm run build`, `oxlint` | passed |
| Unavailable webcam gives an ERROR state and no frames (no fake data) | unit test | passed |
| TTS thread + phrase cooldown (Linux eSpeak, no audio device) | manual script | phrase queued and spoken once; **audio output not heard** |

## B. Must be checked manually on the demo laptop

Do these on the actual laptop, in the actual room, before the demo. Tick each one.

**Setup**
- [ ] `setup.bat` completes. `verify.bat` ends with `0 failed`.
- [ ] `backend\models\hand_landmarker.task` and `backend\models\yolo11n.pt` exist, so nothing needs to download during the demo.

**Camera**
- [ ] `verify.bat live`:
  - "webcam opens" PASS
  - frames processed > 0, FPS noted: ____
  - YOLO detected a person
  - MediaPipe detected a hand
  - stand-in objects detected: ____
- [ ] `run.bat` shows **[launch] component status** and then opens the browser only after the backend is ready.
- [ ] System health shows Camera CONNECTED, YOLO ACTIVE, MediaPipe ACTIVE, Activity ACTIVE, Safety ACTIVE, Voice READY.
- [ ] Unplug or cover the camera (or start Teams with the camera on). The dashboard shows a camera problem (STALLED or ERROR), not fake detections.
- [ ] The **⟳ Cameras** button lists the webcams, and switching webcam works.

**Zones**
- [ ] Draw the workstation and restricted zones and **Save zones**. They are visible on the live video and remain after restarting `run.bat`.
- [ ] A hand over the tray gives **APPROACHING**.

**Interaction and state machine**
- [ ] A quick hand swipe across the bottle gives **no** INTERACTION_STARTED (temporal persistence works).
- [ ] Holding the bottle for about 2 s gives INTERACTING. Releasing gives COMPLETED, and workflow step 1 is ticked.
- [ ] Moving the cup gives MANIPULATING, then COMPLETED (move), and step 2 is ticked.
- [ ] Tapping the phone before the cup (after a Reset) gives an OUT_OF_SEQUENCE warning.

**Safety and alerts**
- [ ] A hand in the restricted zone gives WARNING, the voice "Warning. Restricted zone interaction detected." is **heard**, and there is one alert, not one per frame.
- [ ] Holding the hand there for more than 3 s gives CRITICAL, the voice "Critical alert. Unsafe interaction detected." is **heard**, and the red banner appears.
- [ ] Leaving and re-entering within 8 s gives no second spoken alert (cooldown).
- [ ] Picking up the scissors gives a CRITICAL restricted-equipment alert.
- [ ] Hiding your hand behind the object briefly shows HAND TRACKING LOST, not COMPLETED.

**Logs**
- [ ] The event timeline shows the steps. **⤓ CSV** and **⤓ JSON** download and open correctly.

**Stability**
- [ ] Run for 10 minutes. The FPS stays stable (note it: ____) and the processing error counter stays 0.

## C. Not verified at all (do not claim)

- Accuracy of detection or activity recognition. No labelled test set has been measured.
- Performance on Jetson or any edge device.
- Any use with real BAS hardware, real procedures or in space.
