# Live demo guide (2–3 minutes)

The **primary** demo uses the **real laptop webcam**. The sample video (`run.bat file`) is only a fallback, for example if the webcam fails.

## 1. Physical set-up

### Objects (representative demonstration objects, i.e. BAS equipment stand-ins)

| Object | Role | Notes |
|---|---|---|
| Transparent or plastic **bottle** | Sample Container Stand-in | A labelled bottle is detected more reliably than a fully clear one |
| **Cup / mug** | Culture Vessel Stand-in | A solid colour works best |
| **Tray** | Experimental Workstation Stand-in | Not detected by YOLO. You mark it with a **workstation zone** on screen. |
| **Mobile phone** | Data Tablet Stand-in | Lay it flat or tilt it toward the camera |
| **Scissors** | Sharp Tool Stand-in (restricted) | Tongs are not a COCO class, so use scissors |
| Coloured tape / paper | Marks the **restricted zone** on the table | Draw the same area as a restricted zone on screen |

### Table layout (top view)

```
                    (back of table)
   ┌──────────────────────────────────────────────────┐
   │   [bottle]     [cup]                ┌──────────┐ │
   │                                     │RESTRICTED│ │   ← tape-marked area
   │   ┌────────── TRAY ──────────┐      │  ZONE    │ │     (+ scissors inside)
   │   │   workstation zone       │      │ [scissors]│ │
   │   │         [phone]          │      └──────────┘ │
   │   └──────────────────────────┘                   │
   └──────────────────────────────────────────────────┘
                 (presenter stands here)
                         ▲
                    ┌─────────┐
                    │ LAPTOP  │  webcam on top of the screen
                    └─────────┘
```

### Camera position

The camera must **not** mainly see the face. It must see the **upper body, both hands, the table top and all the objects**.

```
   side view
                 presenter
                    O
                   /|\  ← hands over the table
   laptop         / | \
   ┌──┐ webcam ── ── ── ── ── ► table top + objects
   │  │ ↘  (tilted DOWN)
   │  │   ↘
   └──┘______↘______________________________ table
```

- Put the laptop at the front edge of the table, with the screen tilted back so the webcam looks **down at the table** at roughly 30–45°.
- Raise the laptop on a box or books if needed. The camera should sit **slightly above table height**.
- Keep objects 40–80 cm from the camera, spread apart so their boxes don't overlap.
- Use even, bright light. Avoid a bright window behind the presenter.
- Keep the background plain, and don't let other people walk through the frame.

## 2. Before the professor arrives (10 minutes)

1. Run `verify.bat live`. Every line should be PASS. During the 15 s test, show your hand and the objects to the camera.
2. Run `run.bat`, or `run.bat 1` for a USB webcam. The browser opens by itself once the backend is ready.
3. Check **System health**: Camera **CONNECTED**, YOLO **ACTIVE**, MediaPipe **ACTIVE**, Activity and Safety engines **ACTIVE**, Voice **READY**, FPS ≥ 8.
4. Click **⬚ Zones**:
   - Type **Workstation (tray)** and drag a rectangle over the tray.
   - Type **Restricted zone** and drag over the taped area.
   - Press **Save zones**. Zones are remembered in `backend/data/zones.json`.
5. Click **Test voice** and confirm you can hear it.
6. Do one quick rehearsal, then press **Reset** on the workflow panel and **Ack all** on the alerts.

## 3. The demo (script)

| # | Presenter does | System shows |
|---|---|---|
| 1 | Start the application with `run.bat` | Backend starts, then the browser opens |
| 2 | — | Camera **CONNECTED**, mode **LIVE WEBCAM** |
| 3 | Point at System health | All components green, FPS visible |
| 4 | Point at the zones on the video | Red **RESTRICTED ZONE**, dashed **Workstation (tray)** |
| 5 | Professor steps into view, hands visible | **PERSON DETECTED** |
| 6 | Professor moves a hand over the tray | **APPROACHING** (APPROACHING_WORKSTATION event) |
| 7 | Hand moves close to the bottle | **HAND NEAR EQUIPMENT** |
| 8 | Hold the bottle for about 2 s, then take the hand away | **INTERACTING** → Activity Stability rises → **COMPLETED**. Workflow step 1 ✓. Event SAMPLE_CONTAINER_INTERACTION |
| 8b | *(optional)* Pick up the cup, put it down about 15 cm away, let go | **MANIPULATING** → **COMPLETED** (move). Step 2 ✓ |
| 9 | Put a hand into the taped restricted zone | **WARNING**. Voice: "Warning. Restricted zone interaction detected." |
| 10 | Keep the hand there for more than 3 s | **CRITICAL**, red banner. Voice: "Critical alert. Unsafe interaction detected." |
| 11 | Point at the event timeline | Every step, with time, state, subject and stability |
| 12 | Click **⤓ CSV** (and **⤓ JSON**) | The log downloads |

**Optional extras** (if time allows):

- Tap the phone *before* the cup to trigger an out-of-sequence WARNING.
- Pick up the scissors to trigger a CRITICAL restricted-equipment alert.
- Nudge the cup with a ruler, keeping your hand visible but far from it, to trigger an unexpected-movement WARNING.

**What to say** (keep it honest):

> "Perception uses pretrained YOLO and MediaPipe models. The activity recognition is our temporal reasoning layer:
> interaction must persist over several frames, release must persist, and safety rules have debouncing and cooldowns.
> These objects are stand-ins for BAS equipment, and the workflow is a representative demonstration sequence. With
> real BAS data the detector would be fine-tuned and validated."

## 4. If something goes wrong

| Problem | Do this |
|---|---|
| Camera ERROR | Close other camera apps. Choose another webcam in the dropdown, or use **⟳ Cameras**. Last resort: `run.bat file` (fallback) and say so. |
| Object not detected | Move it closer, turn its label toward the camera, improve the lighting. Check the Detections panel. |
| Interaction not confirmed | Keep your fingertips on the object for about 1 s. Make sure the hand is visible, since a hand hidden by the object causes HAND TRACKING LOST. |
| Too many alerts | Press **Ack all**. Make the zone smaller. |
| No sound | Check Windows volume. Voice ERROR in health means the dashboard uses the browser voice fallback. |
