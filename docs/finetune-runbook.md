# Fine-tune the detector on the demo table (half-day runbook)

Goal: fine-tune YOLO11n on **our own recordings of our demo table** so it detects the stand-in objects more
reliably and also detects the **tray**. Everything after the detector (hands, interaction, state machine,
workflow, safety, dashboard) stays the same.

This is still **not** BAS data. It makes the demo more reliable and gives us one honest, measured number.
It does not make the system mission-ready.

| Step | What | Time |
|---|---|---|
| 0 | Prepare (runs in the background during step 1) | – |
| 1 | Record 6 sessions, 550–800 frames | 35 min |
| 2 | Auto-label with YOLO-World, split by session | 30 min |
| 3 | Review the labels by hand in CVAT | 2 h |
| 4 | Train on a free GPU (Colab) | 45 min |
| 5 | Evaluate against the pretrained model, then switch | 45 min |
| 6 | Update the docs honestly | 15 min |

Class names and order (used everywhere, don't change them):

```
0 person   1 bottle   2 cup   3 cell phone   4 scissors   5 tray
```

They are the COCO names, so `backend/config.yaml` `class_map` keeps working. `tray` maps to
"Experimental Workstation Stand-in".

All commands are for **cmd** in the project folder, with the virtual environment active:

```
cd %USERPROFILE%\Documents\BAS-HAR
.venv\Scripts\activate
```

---

## 0. Prepare (5 min of your time)

1. Write down your Ultralytics version. You need it in step 4:
   ```
   python -c "import ultralytics; print(ultralytics.__version__)"
   ```
2. In a **second** cmd window (venv active, internet on), pre-download YOLO-World and its CLIP text encoder.
   This is about 370 MB and runs while you record:
   ```
   python -c "from ultralytics import YOLO; m = YOLO('backend/models/yolov8s-worldv2.pt'); m.set_classes(['tray']); print('YOLO-World ready')"
   ```
   If it fails with an error about **git**, install CLIP without git, then run the command again:
   ```
   pip install https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
   ```
3. Create a free account at **app.cvat.ai** (step 3) and make sure you can sign in to **colab.research.google.com** (step 4).
4. Ask the second person (session 4) whether they agree to be recorded. The frames are uploaded to CVAT and Colab.

---

## 1. Record (30 min)

**Set-up.** Build the table exactly as in [demo-guide.md §1](demo-guide.md): same laptop, same webcam, camera tilted down
30–45°, objects 40–80 cm away. Mark the laptop's position with tape so you can put it back exactly.
Use the real objects: bottle, cup, phone, scissors and tray, with the taped restricted zone.

**One session** = one command. Recording starts as soon as the window opens (one frame every 0.5 s). The counter at the
top left shows how many frames were saved. Press **Q** at about **120 saved** (about 60 s).

```
python tools\record_dataset.py --name bench1 --auto 0.5
```

**What to do in every session (about 60 s):**

| Time | Action |
|---|---|
| 0–5 s | Hands off the table. All objects visible (hands-free frames). |
| 5–15 s | Touch the bottle, hold it about 2 s, lift it, put it back. |
| 15–25 s | Pick up the cup, move it about 15 cm, let go. |
| 25–35 s | Tap the phone, pick it up, tilt it toward the camera, put it down. |
| 35–45 s | Pick up the scissors, open and close them, put them back in the restricted zone. |
| 45–55 s | Cover part of each object with your hand in turn (fingers in front of it). Use both hands. |
| 55–60 s | Rest a hand on the tray edge, slide the tray a little, then take both hands away. |

**What changes between sessions:**

| Session | Person | Light | Camera | Objects |
|---|---|---|---|---|
| `bench1` | you | demo light (lamp on) | exact demo position | demo layout |
| `bench2` | you | lamp **off** (room light only) | demo position | new positions; add a TV remote and a wallet as look-alike distractors |
| `bench3` | you | lamp on | about 10 cm higher and turned about 15° | new positions, tray rotated; lots of hand-covering |
| `bench4` | **another person** | lamp on or window light | about 10 cm lower / closer | demo layout |
| `bench5` | you | demo light | **back to the exact demo position** | an arrangement not used before; keep 1 distractor |
| `bench6` | you | demo light | exact demo position | another new arrangement; keep 1 distractor. Only about 40 s (~80 frames) |

`bench5` becomes the **validation** session: training uses it to pick the best epoch. `bench6` becomes the **test**
session: it is used for nothing except the final score, so that score is not tuned on. Both are recorded like the
live demo, so their scores tell you how the demo will behave. Make sure **all six classes are in bench5 and bench6**,
including the scissors and the tray.

Check the counts (aim for 550–800 in total):

```
python -c "from pathlib import Path; [print(d.name, len(list(d.glob('*.jpg')))) for d in sorted(Path('dataset/raw').iterdir())]"
```

Running the same `--name` again **adds** frames to that session.

---

## 2. Auto-label (30 min)

```
python tools\auto_label.py --src dataset/raw --classes person bottle cup "cell phone" scissors tray --val-session bench5 --test-session bench6
```

What it does:

- Pre-labels every frame with YOLO-World, using the six class names as text prompts. It runs in a few minutes on a CPU.
- **Splits by session folder**: all of `bench5` goes to `val`, all of `bench6` to `test`, and `bench1`–`bench4` to
  `train`. No frame of a session ever appears in two splits, so neighbouring near-identical frames can't leak.
- Writes `dataset/bas/` (images, labels, `data.yaml` with the class order above, and `split.json`, which records the split),
  plus **`dataset/bas_upload.zip`** for CVAT.
- Prints the number of pre-label boxes per class for train, val and test.

Read the warnings:

- **"no pre-label boxes for [...] in the val (or test) session"**: if the object is really in bench5/bench6, you will
  add the boxes in step 3. If it is not, record a few more seconds with `--name bench5` (or `bench6`) and re-run with
  `--overwrite`.
- A class with very few boxes overall (often `tray` or `scissors`) usually means YOLO-World misses it. That's fine;
  you draw those boxes in step 3.

Re-running needs `--overwrite`. The old images are removed, so nothing stale stays in the wrong split.

---

## 3. Review the labels by hand (2 h, the most important step)

Pre-labels are **never** trained on without review. The test labels decide the number you will report and the val
labels decide which epoch is kept, so review **test and val first and most carefully**.

### 3a. Set up CVAT (10 min)

We use **CVAT Online (app.cvat.ai), free plan**: private, 1 project, 3 tasks, 1 GB. That is enough for this.
(Roboflow's free plan makes datasets **public**, and these frames show people's faces. See the note at the end of step 3.)

1. **Projects → + → Create a new project.** Name: `BAS-HAR demo table`.
2. Add the labels **in exactly this order**: `person`, `bottle`, `cup`, `cell phone`, `scissors`, `tray`
   (type Rectangle or Any). Then **Submit & Open**.
3. On the project page: **Actions → Import dataset**. Format: **Ultralytics YOLO Detection 1.0**. File:
   `dataset\bas_upload.zip`. CVAT uploads the images and pre-labels and creates **three tasks, `train`, `val` and
   `test`** (exactly the free plan's limit). Progress shows on the **Requests** page.
4. Open task **test** → click its job. Review it, then do the same for **val**, then **train**.

### 3b. Keys that save time

| Key | Action |
|---|---|
| **F** / **D** | next / previous frame |
| **N** | draw a new box (repeats the last shape and label) |
| **Ctrl+1 … Ctrl+6** | change the selected box's label (person, bottle, cup, cell phone, scissors, tray, in label order) |
| **Del** | delete the selected box |
| **Alt+Del** | delete the whole frame (it is left out of the export) |
| **H** | hide the selected box, e.g. a big person box that covers small objects |
| **Ctrl+B** | propagate: copy the selected box to the next frames (good for a tray that doesn't move) |
| **Ctrl+Z** / **Ctrl+S** | undo / **save (save often)** |

Pace: about 680 frames in 110 minutes is about 10 s per frame. Most frames need one look and **F**.
Suggested split: **test 15 min, val 30 min, train 65 min**, then 10 min for export and import.

### 3c. Checklist: what to look for on every frame

**Missing boxes**

- [ ] **Partly hidden objects.** An object held in the hand, or with fingers in front of it, still gets a box.
      Box the **whole object**, including the part behind the fingers where you can tell its edge. Never include the hand.
      Only when about ¾ or more is hidden and you can't tell its size do you leave it without a box.
- [ ] Objects cut off at the image edge: box the visible part, up to the edge.
- [ ] Easy-to-miss objects: phone lying flat, scissors lying flat, a clear bottle against a light background.
- [ ] **Person**: every visible person gets **one** box, even if only the arms and shoulders are in frame.
- [ ] **Tray**: one box on every frame where the tray is visible.

**Wrong boxes**

- [ ] **Tray box covering the whole table**, the laptop, a book or a sheet of paper. The tray box must follow the
      **tray's rim** and cover the whole tray, including where objects stand on it or an arm covers an edge.
      Two boxes on one tray: keep one.
- [ ] Wrong class: bottle ↔ cup swaps. "cell phone" on a bare hand, the wallet, the remote or the laptop. "scissors" on
      a pen, cable or tape. "person" on a reflection, a photo, a poster or the laptop screen.
- [ ] Duplicates: two boxes on the same object (same or different class). Keep one, with the right class.
- [ ] Loose or shifted boxes: each edge should touch the object's outermost pixels (a few pixels of slack is fine).
      Pre-labels often trail a moving object or include the holding hand.
- [ ] Person box: one box around the whole visible person. Not one per arm, and not the chair behind them.
- [ ] Distractors (remote, wallet, pen, keys, laptop, charger) have **no** box.

**Frames to delete (Alt+Del)**

- [ ] Motion blur so strong that you can't see where an object's edges are.
- [ ] Frames where the camera was being moved, the lamp was switching, or a window covers the view.
- [ ] Long runs of near-identical frames (e.g. 20 frames of the untouched table): keep about 1 in 4.
- Delete at most about 10–15 %. **Don't delete frames just because they are hard.** Clear frames with hands covering
  objects are the most valuable ones.

**Be consistent.** Apply the rules above the same way on every frame. If you are unsure about a frame, apply the rule
and move on; don't delete it.

### 3d. Export and bring the labels back (10 min)

1. Project page → **Actions → Export dataset** → format **Ultralytics YOLO Detection 1.0** → *Save images* off →
   **OK**. Download the zip from the **Requests** page.
2. Import it. The script maps classes back **by name**, matches every label to your original full-resolution frame,
   re-makes the **same session split**, leaves out deleted frames and turns frames with no boxes into negative examples:
   ```
   python tools\import_reviewed.py --export "%USERPROFILE%\Downloads\<the exported file>.zip"
   ```
3. Read the summary: frames kept or deleted per session, and boxes per class for train, val and test.
   **Every class needs boxes in train, val and test.** It also writes **`dataset/bas_reviewed_colab.zip`** for step 4.

> **Roboflow instead of CVAT?** Only if you accept that the free plan makes the dataset public. If you do, upload the
> `dataset/bas` folder, mark frames with no objects as **Null**, and generate a version with **no augmentation** and
> Resize set to **Stretch** (or off). Export as **YOLOv11**. Use the same `import_reviewed.py` command: it undoes
> Roboflow's alphabetical class order, renamed files and re-split, and it refuses augmented or letterboxed exports.

---

## 4. Train on a free GPU (45 min)

1. Open **colab.research.google.com → File → Upload notebook** → `tools\colab_train.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. In cell 2, set `ULTRALYTICS_VERSION` to the version from step 0. The weights then load in your backend without surprises.
4. **Runtime → Run all.** When cell 3 shows *Choose Files*, pick `dataset\bas_reviewed_colab.zip` (about 100 MB,
   so it takes a few minutes).
5. Training runs `tools/train.py` for 60 epochs at `--imgsz 480`, with early stopping (about 10–20 min on a T4).
   480 is the size the backend runs at (`detector.imgsz`) and the size step 5 evaluates at. If you change one, change all
   three, or the before/after comparison also measures the size change. In the log, the
   `all` row's **mAP50** should rise and then level off.
6. The last cell downloads **`bas_custom_results.zip`**. Unpack it into the project folder:
   ```
   tar -xf "%USERPROFILE%\Downloads\bas_custom_results.zip"
   ```
   You now have `backend\models\bas_custom.pt` and the training curves in `runs\bas\` (`results.png`, `confusion_matrix.png`).

**No GPU available?**

- **Kaggle** (about 30 GPU hours a week; may ask for phone verification): upload `bas_reviewed_colab.zip` as a Dataset
  (Kaggle unpacks it), create a notebook with a T4 GPU and that dataset attached, copy it to `/kaggle/working/BAS-HAR`, then
  run the same `python tools/train.py ...` line as in cell 4.
- **Laptop CPU** (slow, expect 1–2 h). Start it and prepare step 6 meanwhile:
  ```
  python tools\train.py --data dataset/bas_reviewed/data.yaml --epochs 30 --imgsz 480 --batch 8 --device cpu
  ```

---

## 5. Evaluate and switch the model (45 min)

### 5a. Measure (10 min)

Score the **pretrained COCO model** and the **fine-tuned model** on the **same held-out test session** (bench6), at
`--imgsz 480`, the size the backend uses (`detector.imgsz`). COCO has no tray, so the fair comparison uses the five
shared classes. A third run adds the tray.

```
python evaluation\evaluate_detector.py --data dataset/bas_reviewed/data.yaml --split test --imgsz 480 ^
  --model backend/models/yolo11n.pt --classes person bottle cup "cell phone" scissors ^
  --output-dir evaluation/results/coco_baseline

python evaluation\evaluate_detector.py --data dataset/bas_reviewed/data.yaml --split test --imgsz 480 ^
  --model backend/models/bas_custom.pt --classes person bottle cup "cell phone" scissors ^
  --output-dir evaluation/results/finetuned_5cls

python evaluation\evaluate_detector.py --data dataset/bas_reviewed/data.yaml --split test --imgsz 480 ^
  --model backend/models/bas_custom.pt --output-dir evaluation/results/finetuned_all

python evaluation\compare_results.py ^
  --run "Pretrained YOLO11n (COCO)=evaluation/results/coco_baseline" ^
  --run "Fine-tuned (same 5 classes)=evaluation/results/finetuned_5cls" ^
  --run "Fine-tuned (all 6 classes)=evaluation/results/finetuned_all" ^
  --out docs/finetune-results.md
```

`docs/finetune-results.md` now holds the table, copied from the JSON results (no hand-typed numbers), plus the
dataset facts and caveats.

### 5b. Decide

Switch only if **all** of these hold:

1. On the five shared classes, the fine-tuned **mAP50 is higher** than the pretrained model's.
2. No shared class is clearly worse than with the pretrained model (as a rule of thumb, more than 0.05 lower mAP50).
   Check **person** (presence) and **scissors** (the critical restricted-tool alert) especially.
3. The live check in 5c passes.

The tray is display-only (see 5c), so a weak tray score doesn't block the switch. It does mean you must not claim tray
detection in the docs. If 1 or 2 fails, keep the pretrained model and report the result honestly. That is a valid outcome.

### 5c. Switch and check live (25 min)

In `backend\config.yaml`, under `detector:`, change two lines (the shipped config runs the pretrained model, `mode: coco`):

```yaml
  mode: custom
  model_path: models/bas_custom.pt
```

Then:

1. `verify.bat live` should show *YOLO loads + runs: bas_custom.pt, 6 mapped classes*. During the 15 s, show your
   hands and every object. "Experimental Workstation Stand-in" should be among the detected stand-ins.
2. `run.bat`. The first timeline event should read *YOLO: ready [CUSTOM MODEL]*.
3. Rehearse steps 5–10 of the demo script ([demo-guide.md §3](demo-guide.md)) and check:
   - The **Detections** panel shows the tray as *Experimental Workstation Stand-in* and it doesn't flicker.
   - A hand over the tray gives **APPROACHING** (from the workstation zone). It must **not** give "Handled Experimental
     Workstation Stand-in". The tray is kept out of the interaction logic by `detector.surface_objects`, so keep drawing
     the workstation zone.
   - Bottle → INTERACTING → COMPLETED, cup move, phone, restricted zone WARNING → CRITICAL, and scissors CRITICAL all behave as before.
   - A bare hand with no phone doesn't show *Data Tablet Stand-in*. If it does, raise `class_conf: cell phone`.
     If an object flickers, lower `detector.conf` a little (not below 0.25).
   - `class_conf: cell phone: 0.55` was tuned for the pretrained model, which mistakes hands for phones. The
     fine-tuned model has seen hands without phones, so if the real phone flickers or is missed, try 0.45, then
     0.35 (`detector.conf`), while re-checking the bare-hand case above. `class_conf: remote` does nothing for the
     fine-tuned model (it has no remote class) and can stay for rollback.
   - Switch the lamp off once and check again.
4. `run_tests.bat`. The unit tests must pass. The integration and smoke tests run the **sample video**
   (`hand_demo.mp4`, not your table) through the new model. If *person detected* now fails there, the fine-tuned model
   has specialised to your table. **Don't weaken the test.** Write it down as a limitation in step 6. For demo day,
   `verify.bat live` at your table is the check that matters.

**Rollback** (takes seconds): set `mode: coco` in `backend\config.yaml` and restart.

---

## 6. Update the docs honestly (15 min)

`docs/finetune-results.md` already exists (step 5a). Then edit these places, filling `{…}` from that file.
If you did **not** switch, only add a line to README §11 saying what you tried and why the demo keeps the pretrained model.

| File | Where | Change to |
|---|---|---|
| `README.md` | §5 table, "Pretrained YOLO11n (COCO)…" | "YOLO11n (COCO-pretrained) **fine-tuned on our own demo-table recordings** (person, bottle, cup, cell phone, scissors, tray)" |
| `README.md` | §6, YOLO11n row, "Trained by us?" | "Fine-tuned by us: transfer learning from COCO on {N} frames of our own demo table ([results](docs/finetune-results.md)). Not trained on BAS data." |
| `README.md` | §9, Tray row | Detected class `tray` (fine-tuned model). Shown as a detection only; "approaching" still uses the workstation zone. |
| `README.md` | §11, "No accuracy figures" | "**Detection measured only on our own table.** On one held-out recording session ({n} frames), mAP50 {x} (fine-tuned) vs {y} (pretrained COCO) on the same five classes; tray mAP50 {t}. Same room, table and objects as training; that session was not used for training or for picking the epoch; frame-level detection only. Activity-recognition accuracy is still not measured." Add the sample-video limitation from 5c if it applies. |
| `README.md` | §13, Privacy line | Keep it, and add: "For the detector fine-tune we deliberately recorded {N} webcam frames of our table, with the consent of the people shown. They were labelled in a private CVAT project, trained on Google Colab, and are not in this repository." |
| `docs/dataset-strategy.md` | "WHAT EXISTS NOW" | YOLO11n bullet: fine-tuned on our own mock-bench recordings (interim data, see *Optional interim data*). Evaluation bullet: the measured number above, with the same caveats. The "No ISRO/BAS data" bullet stays true. |
| `docs/judge-qa.md` | Q1 ("we have not trained a custom model") and "No measured accuracy figures yet" | Say what we fine-tuned, on what, and the one number with its caveats. |
| `docs/demo-guide.md` | Tray row + "What to say" | The tray is now detected; the workstation zone is still drawn. Say: "the detector was fine-tuned on our own recordings of this table; on a separate recording session it scored {x} mAP50 vs {y} for the pretrained model; it has never seen BAS hardware." |
| `docs/custom-training.md` | top | Note that the demo-table fine-tune follows `docs/finetune-runbook.md` with COCO class names plus `tray`; the five-class names in that guide are an alternative scheme that was not used. |
| `backend/app/detector.py` | module docstring, lines 3–4 | "Pretrained on COCO; in `custom` mode, fine-tuned on our demo-table recordings. Never trained on BAS hardware." |

Don't claim what the numbers don't show: not BAS hardware, not other rooms or cameras, not activity accuracy.

---

## What was changed in the code for this

- `tools/auto_label.py`: train/val/test split **by session folder** (`--val-session`, `--test-session`); class order kept exactly; `data.yaml`
  without an absolute path (works in Colab) and without `nc` (CVAT rejects it); `split.json`; CVAT upload zip;
  cross-class duplicate suppression (`--agnostic-nms`); refuses to overwrite a dataset unless `--overwrite` is given.
- `tools/import_reviewed.py` (new): brings reviewed CVAT/Roboflow exports back safely (see 3d) and writes the Colab zip.
- `tools/colab_train.ipynb` (new): the free-GPU training notebook.
- `evaluation/evaluate_detector.py`: the split is read from `data.yaml` (it used to look only in `dataset/images/`); classes
  are matched **by name** so the pretrained COCO model can be scored on the same split; `--classes`; fixed per-class
  metrics, which were indexed by class id instead of Ultralytics' `ap_class_index`.
- `evaluation/compare_results.py` (new): side-by-side results table from the JSON files.
- `backend/config.yaml`: `class_map.tray` and the new `detector.surface_objects` (validated in `app/config.py`).
- `backend/app/pipeline.py`, `app/interaction.py`: surface objects (the tray) are drawn and listed, but kept out of the
  hand-object interaction, activity, safety and scene logic. Covered by a new unit test.
- `.gitignore`: recorded frames and datasets stay out of git (they show real people).
