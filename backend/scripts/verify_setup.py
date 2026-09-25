"""Pre-demo verification. Run this on the demo laptop BEFORE the professor arrives.

    python backend/scripts/verify_setup.py            # environment, models, config, cameras, TTS
    python backend/scripts/verify_setup.py --live     # + 15 s of REAL webcam frames through YOLO + MediaPipe

Everything is checked for real; nothing is assumed. Exit code 0 = all required checks passed.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

RESULTS: list[tuple[str, bool, str, bool]] = []   # (name, ok, detail, required)


def check(name: str, ok: bool, detail: str = "", required: bool = True):
    RESULTS.append((name, ok, detail, required))
    mark = "PASS" if ok else ("FAIL" if required else "WARN")
    print(f"  [{mark}] {name}{' - ' + detail if detail else ''}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="run real webcam frames through the models")
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=15)
    args = ap.parse_args()

    print("\n== Environment")
    v = sys.version_info
    check("Python version", (3, 10) <= (v.major, v.minor) <= (3, 12), f"{platform.python_version()} (3.10-3.12 supported)")
    mods = {}
    for mod in ("cv2", "numpy", "ultralytics", "mediapipe", "fastapi", "uvicorn", "yaml", "multipart"):
        try:
            m = importlib.import_module(mod)
            mods[mod] = m
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, str(e))
    try:
        importlib.import_module("pyttsx3")
        check("import pyttsx3 (voice)", True, required=False)
    except Exception as e:  # noqa: BLE001
        check("import pyttsx3 (voice)", False, str(e), required=False)

    print("\n== Configuration")
    cfg = None
    try:
        from app.config import load_config, resolve_path
        cfg = load_config()
        check("config.yaml valid", True, f"{len(cfg['safety']['zones'])} zone(s), mode={cfg['video']['mode']}")
        for w in cfg["_warnings"]:
            check("config warning", False, w, required=False)
    except Exception as e:  # noqa: BLE001
        check("config.yaml valid", False, str(e))

    print("\n== Model files (must be local - no download during the demo)")
    yolo_ok = hands_ok = False
    if cfg:
        mp_path = resolve_path(cfg["hands"]["model"])
        y_path = resolve_path(cfg["detector"]["model"])
        check("YOLO weights", y_path.exists(), str(y_path))
        check("MediaPipe hand model", mp_path.exists(),
              str(mp_path) if mp_path.exists() else f"missing {mp_path} - run python backend/scripts/download_models.py")
        if "ultralytics" in mods and y_path.exists():
            try:
                from app.detector import ObjectDetector
                t0 = time.time()
                det = ObjectDetector(cfg)
                check("YOLO loads + runs", True, f"{det.model_name}, {len(det.class_ids)} mapped classes, {time.time() - t0:.1f}s")
                yolo_ok = True
            except Exception as e:  # noqa: BLE001
                check("YOLO loads + runs", False, str(e))
        if "mediapipe" in mods:
            from app.hands import HandTracker
            ht = HandTracker(cfg)
            hands_ok = check("MediaPipe HandLandmarker loads", ht.available, ht.backend if ht.available else ht.error)
            ht.close()

    print("\n== Dashboard build")
    dist = BACKEND.parent / "frontend" / "dist" / "index.html"
    check("frontend/dist present", dist.exists(), str(dist))

    print("\n== Cameras")
    cams = []
    if "cv2" in mods:
        from app.pipeline import probe_cameras
        cams = probe_cameras(4)
        found = [c for c in cams if c["available"]]
        for c in cams:
            if c["available"]:
                print(f"         webcam {c['index']}: {c['detail']}")
        want = args.camera if args.camera is not None else (cfg["video"]["camera_index"] if cfg else 0)
        check("at least one webcam opens and delivers frames", bool(found),
              "none found - close Teams/Zoom/Camera app; check Windows camera privacy settings" if not found else "")
        if found:
            check(f"configured webcam {want} works", any(c["index"] == want for c in found),
                  "pick another index with run.bat <index>" if not any(c["index"] == want for c in found) else "")

    print("\n== Offline voice")
    if cfg:
        from app.voice import VoiceAlerter
        va = VoiceAlerter(cfg)
        va.wait_ready(8)
        check("offline TTS initialises", va.available, va.error or "pyttsx3 ready", required=False)
        if va.available:
            va.say("Voice check.", key="verify")
            time.sleep(2)

    if args.live:
        print(f"\n== LIVE webcam test ({args.seconds:.0f}s) - stand in front of the camera and show a hand")
        if not (cams and yolo_ok and hands_ok and cfg):
            check("live test", False, "skipped - fix the failures above first")
        else:
            live_test(cfg, args.camera if args.camera is not None else cfg["video"]["camera_index"], args.seconds)

    failed = [r for r in RESULTS if r[3] and not r[1]]
    warns = [r for r in RESULTS if not r[3] and not r[1]]
    print(f"\n== RESULT: {len(RESULTS) - len(failed) - len(warns)} passed, {len(failed)} failed, {len(warns)} warnings")
    if failed:
        print("   Fix the FAIL items before the demo.")
    return 1 if failed else 0


def live_test(cfg, index: int, seconds: float):
    import cv2

    from app.detector import ObjectDetector
    from app.hands import HandTracker
    from app.pipeline import open_camera

    cap, backend, err = open_camera(index, cfg["video"]["frame_width"], cfg["video"]["frame_height"], 5)
    if cap is None:
        check("webcam opens", False, err)
        return
    check("webcam opens", True, f"webcam {index} via {backend}")
    det, ht = ObjectDetector(cfg), HandTracker(cfg)
    frames = persons = hand_frames = 0
    labels: dict[str, int] = {}
    t0 = time.time()
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1
        h, w = frame.shape[:2]
        pw = cfg["video"]["process_width"]
        if w > pw:
            frame = cv2.resize(frame, (pw, int(h * pw / w)))
        dets = det.detect(frame)
        for d in dets:
            labels[d.label] = labels.get(d.label, 0) + 1
        persons += any(d.label == "Person" for d in dets)
        hand_frames += bool(ht.process(frame))
    cap.release()
    ht.close()
    elapsed = time.time() - t0
    check("frames processed", frames > 0, f"{frames} frames in {elapsed:.0f}s = {frames / elapsed:.1f} FPS (YOLO every frame here)")
    check("YOLO detected a person", persons > 0, f"in {persons}/{frames} frames")
    check("MediaPipe detected a hand", hand_frames > 0, f"in {hand_frames}/{frames} frames - show your hand to the camera")
    stand_ins = {k: v for k, v in labels.items() if k != "Person"}
    check("stand-in objects detected", bool(stand_ins),
          ", ".join(f"{k}: {v}" for k, v in stand_ins.items()) or "none - put the bottle/cup/phone in view",
          required=False)


if __name__ == "__main__":
    sys.exit(main())
