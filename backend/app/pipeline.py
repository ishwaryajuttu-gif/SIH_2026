"""The real-time processing pipeline.

Webcam/file -> OpenCV -> YOLO + tracker -> MediaPipe hands -> 2D interaction estimation
-> temporal state machine -> workflow + safety rules -> overlay / events / alerts

LIVE mode processes real webcam frames only. There is no simulated data path: if the
camera or a model is unavailable the dashboard shows an ERROR with the reason instead.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .activity import ActivityRecognizer
from .config import resolve_path
from .detector import ObjectDetector, SimpleTracker
from .events import EventLog
from .hands import HandTracker
from .interaction import HUMAN_LABELS, InteractionAnalyzer
from .overlay import draw
from .procedure import WorkflowTracker
from .safety import SafetyMonitor
from .scene import derive_scene
from .voice import VoiceAlerter

log = logging.getLogger("bas.pipeline")
IS_WINDOWS = platform.system() == "Windows"


def _backends() -> list[tuple[str, int]]:
    """Capture back-ends to try, in order. DirectShow opens fastest on most Windows laptops."""
    if IS_WINDOWS:
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def open_camera(index: int, width: int, height: int, timeout_s: float = 3.0):
    """Open a webcam and prove it works by reading a real frame.
    Returns (cap, backend_name, error). cap is None on failure."""
    errors = []
    for name, api in _backends():
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            cap.release()
            errors.append(f"{name}: cannot open")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return cap, name, ""
            time.sleep(0.05)
        cap.release()
        errors.append(f"{name}: opened but delivered no frames")
    hint = ("Check that no other app (Teams, Zoom, Camera) is using it and that camera access is allowed "
            "in Windows Settings > Privacy & security > Camera." if IS_WINDOWS else "Check the device permissions.")
    return None, "", f"Camera {index} unavailable ({'; '.join(errors)}). {hint}"


def probe_cameras(max_index: int = 4, skip: set[int] | None = None, timeout_s: float = 1.5) -> list[dict]:
    """Try camera indices 0..max_index. Indices in `skip` (already in use by us) are reported as in use."""
    out = []
    for i in range(max_index + 1):
        if skip and i in skip:
            out.append({"index": i, "available": True, "in_use": True, "detail": "in use by BAS-HAR"})
            continue
        cap, backend, err = open_camera(i, 640, 480, timeout_s)
        if cap is not None:
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            out.append({"index": i, "available": True, "in_use": False, "detail": f"{w}x{h} via {backend}"})
        else:
            out.append({"index": i, "available": False, "in_use": False, "detail": err.split(". ")[0]})
    return out


class VideoSource:
    """Grabs frames in a background thread and always serves the newest one (no lag build-up)."""

    def __init__(self, kind: str, value, width=1280, height=720, loop=True, open_timeout=5.0, stall_s=3.0):
        self.kind = kind                   # "webcam" | "file"
        self.value = value                 # camera index or file path
        self.loop = loop
        self.width, self.height = width, height
        self.open_timeout, self.stall_s = open_timeout, stall_s
        self.frame: np.ndarray | None = None
        self.frame_id = 0
        self.last_frame_ts = 0.0
        self.opened = False
        self.ended = False
        self.error = ""
        self.backend = ""
        self.resolution = (0, 0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    @property
    def is_file(self) -> bool:
        return self.kind == "file"

    @property
    def state(self) -> str:
        if self.error and not self.opened:
            return "error"
        if self.ended:
            return "ended"
        if not self.opened:
            return "connecting"
        if time.time() - self.last_frame_ts > self.stall_s:
            return "stalled"
        return "connected"

    def _open(self):
        if self.kind == "webcam":
            cap, backend, err = open_camera(int(self.value), self.width, self.height, self.open_timeout)
            self.backend, self.error = backend, err
            return cap
        path = resolve_path(self.value)
        if not path.exists():
            self.error = f"Video file not found: {path}"
            return None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self.error = f"Cannot decode video file: {path}"
            return None
        self.backend = "file"
        return cap

    def _run(self):
        cap = self._open()
        if cap is None:
            log.error(self.error)
            return
        self.opened, self.error = True, ""
        self.resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps if self.is_file and fps > 0 else 0
        fails = 0
        while not self._stop.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.is_file:
                    if self.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.ended = True
                    break
                fails += 1
                if fails >= 30:  # webcam unplugged / taken by another app -> try to reopen
                    self.error = "Camera stopped delivering frames - reconnecting"
                    log.warning(self.error)
                    cap.release()
                    time.sleep(1.0)
                    cap = self._open()
                    if cap is None:
                        self.opened = False
                        log.error(self.error)
                        return
                    self.error, fails = "", 0
                time.sleep(0.02)
                continue
            fails = 0
            with self._lock:
                self.frame = frame
                self.frame_id += 1
                self.last_frame_ts = time.time()
            if delay:
                time.sleep(max(0.0, delay - (time.time() - t0)))
        cap.release()

    def read(self):
        with self._lock:
            return self.frame, self.frame_id

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)


class Pipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        v = cfg["video"]
        self.process_width = int(v["process_width"])
        self.jpeg_quality = int(v["jpeg_quality"])
        self.min_period = 1.0 / float(v["max_processing_fps"])
        self.detect_every = int(cfg["detector"]["detect_every"])

        self.events = EventLog(cfg)
        self.voice = VoiceAlerter(cfg)
        self.events.subscribe(self.voice.on_event)

        # ---- perception models: failures are reported, never hidden or simulated ----
        self.detector: ObjectDetector | None = None
        self.detector_error = ""
        try:
            self.detector = ObjectDetector(cfg)
        except Exception as e:  # noqa: BLE001
            self.detector_error = str(e)
            log.error("YOLO unavailable: %s", e)
            self.events.emit("COMPONENT_ERROR", "critical", f"YOLO unavailable: {e}", activity="SYSTEM", state="ERROR",
                             subject="YOLO")
        self.hands = HandTracker(cfg) if cfg["hands"]["enabled"] else None
        if self.hands is not None and not self.hands.available:
            self.events.emit("COMPONENT_ERROR", "critical", f"MediaPipe unavailable: {self.hands.error}",
                             activity="SYSTEM", state="ERROR", subject="MediaPipe")

        tr = cfg["tracker"]
        self.tracker = SimpleTracker(tr["iou_match"], tr["max_missed"], tr["smoothing"], tr["min_hits"],
                                     tr["center_match"])
        self.analyzer = InteractionAnalyzer(cfg)
        self.activity = ActivityRecognizer(cfg, self.events)
        self.safety = SafetyMonitor(cfg, self.events)
        self.workflow = WorkflowTracker(cfg, self.events)
        self.restricted = set(cfg["detector"]["restricted_objects"])

        self.source: VideoSource | None = None
        self.paused = False
        self.started_at = time.time()
        self.fps = 0.0
        self.latency = {"detect": 0.0, "hands": 0.0, "logic": 0.0, "total": 0.0}
        self.frames_processed = 0
        self.frame_size = (0, 0)
        self.last_process_ts = 0.0
        self.last_detect_ts = 0.0
        self.last_hands_ts = 0.0
        self.errors = 0
        self.last_error = ""
        self.cameras: list[dict] = []

        self._lock = threading.RLock()     # guards all state shared with API threads
        self._jpeg: bytes | None = None
        self._jpeg_cond = threading.Condition()
        self._last_viewer = 0.0
        self._snapshot: dict = {}
        self._snap_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)

        if v["mode"] == "file":
            self.set_source("file", v["file"])
        else:
            self.set_source("webcam", int(v["camera_index"]))
        for w in cfg.get("_warnings", []):
            log.warning("config: %s", w)
        self.events.emit(
            "SYSTEM_START", "info",
            f"Pipeline started - YOLO: {'ready' if self.detector else 'ERROR'}, "
            f"MediaPipe: {self.hands.backend if self.hands else 'disabled'}",
            activity="SYSTEM_START", state="IDLE",
        )

    # ------------------------------------------------------------------ control
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)
        if self.source:
            self.source.stop()
        if self.hands:
            self.hands.close()

    def set_source(self, kind: str, value):
        if kind not in ("webcam", "file"):
            raise ValueError("source kind must be webcam or file")
        with self._lock:
            if self.source:
                self.source.stop()
            v = self.cfg["video"]
            self.source = VideoSource(kind, value, v["frame_width"], v["frame_height"], v["loop_video"],
                                      v["camera_open_timeout_s"], v["camera_stall_s"])
            self.tracker.reset()
            self.activity.reset()
            self.safety.reset()
            self._source_announced = False
        name = f"webcam {value}" if kind == "webcam" else Path(str(value)).name
        self.events.emit("SOURCE_SELECTED", "info", f"Video source: {name} ({'LIVE' if kind == 'webcam' else 'file fallback'})",
                         activity="SOURCE_SELECTED", state="CONNECTING", subject=name)

    def rescan_cameras(self) -> list[dict]:
        src = self.source
        skip = {int(src.value)} if src and src.kind == "webcam" and src.opened else set()
        self.cameras = probe_cameras(4, skip)
        return self.cameras

    def set_zones(self, zones: list[dict]):
        with self._lock:
            self.safety.set_zones(zones)

    def reset_workflow(self):
        with self._lock:
            self.workflow.reset()

    # ------------------------------------------------------------------ main loop
    def _loop(self):
        last_id = -1
        idx = 0
        tracks: list = []
        last_start = 0.0
        while not self._stop.is_set():
            src = self.source
            if self.paused or src is None:
                time.sleep(0.05)
                continue
            if not getattr(self, "_source_announced", True) and src.state in ("connected", "error"):
                self._source_announced = True
                if src.state == "connected":
                    self.events.emit("CAMERA_CONNECTED" if not src.is_file else "VIDEO_FILE_OPENED", "info",
                                     f"{'Camera' if not src.is_file else 'Video'} connected "
                                     f"({src.resolution[0]}x{src.resolution[1]}, {src.backend})",
                                     activity="SOURCE_CONNECTED", state="CONNECTED")
                else:
                    self.events.emit("CAMERA_ERROR", "critical", src.error, activity="SOURCE_ERROR", state="ERROR")
            frame, fid = src.read()
            if frame is None or fid == last_id:
                time.sleep(0.005)
                continue
            wait = self.min_period - (time.perf_counter() - last_start)
            if wait > 0:  # processing-frequency cap: newest frame is picked up after the wait
                time.sleep(wait)
                frame, fid = src.read()
            last_id = fid
            last_start = time.perf_counter()
            try:
                with self._lock:
                    tracks = self._process(frame, idx, tracks, last_start)
            except Exception as e:  # noqa: BLE001 - keep the live demo running, but report it
                self.errors += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("Frame processing failed: %s", e)
                time.sleep(0.1)
            idx += 1

    def _process(self, frame, idx, tracks, t0):
        h0, w0 = frame.shape[:2]
        if w0 > self.process_width:
            scale = self.process_width / w0
            frame = cv2.resize(frame, (self.process_width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        self.frame_size = (w, h)
        diag = (w * w + h * h) ** 0.5

        t1 = time.perf_counter()
        if self.detector is not None and idx % self.detect_every == 0:
            tracks = self.tracker.update(self.detector.detect(frame))
            self.last_detect_ts = time.time()
        t2 = time.perf_counter()
        hands = []
        if self.hands is not None and self.hands.available:
            hands = self.hands.process(frame)
            self.last_hands_ts = time.time()
        t3 = time.perf_counter()

        inters = self.analyzer.analyze(hands, tracks)
        best = self.analyzer.per_object(inters)
        now = time.time()
        for done in self.activity.update(tracks, best, len(hands), diag, now):
            self.workflow.on_activity(done)
        self.safety.update(hands, tracks, self.activity, w, h, now)
        zones = self.safety.zone_state()
        scene = derive_scene(self.activity, self.safety, tracks, now)
        t4 = time.perf_counter()

        if time.time() - self._last_viewer < 3.0 or self._jpeg is None:  # only encode when someone watches
            vis = draw(frame, tracks, hands, best, self.activity, zones, self.restricted, scene)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if ok:
                with self._jpeg_cond:
                    self._jpeg = buf.tobytes()
                    self._jpeg_cond.notify_all()
        t5 = time.perf_counter()

        a = 0.15
        for k, val in (("detect", t2 - t1), ("hands", t3 - t2), ("logic", t4 - t3), ("total", t5 - t0)):
            if k == "detect" and not (idx % self.detect_every == 0):
                continue  # average YOLO latency over frames where YOLO actually ran
            self.latency[k] = (1 - a) * self.latency[k] + a * val * 1000 if self.latency[k] else val * 1000
        now_p = time.perf_counter()
        dt = now_p - getattr(self, "_last_t", now_p - 0.1)
        self._last_t = now_p
        if dt > 0:
            self.fps = (1 - a) * self.fps + a * (1.0 / dt) if self.fps else 1.0 / dt
        self.frames_processed += 1
        self.last_process_ts = time.time()

        snap = self._build_snapshot(tracks, hands, best, zones, scene, w, h)
        with self._snap_lock:
            self._snapshot = snap
        return tracks

    # ------------------------------------------------------------------ outputs
    def _build_snapshot(self, tracks, hands, best, zones, scene, w, h):
        objects = []
        for t in tracks:
            oa = self.activity.objects.get(t.track_id)
            it = best.get(t.track_id)
            x1, y1, x2, y2 = t.box
            objects.append({
                "track_id": t.track_id, "label": t.label, "cls": t.cls_name,
                "conf": round(t.conf, 3), "visible": t.visible,
                "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
                "state": oa.state.value if oa else ("HUMAN" if t.label in HUMAN_LABELS else "IDLE"),
                "stability": round(oa.stability(), 3) if oa else None,
                "proximity": it.level.name if it else "FAR",
                "restricted": t.label in self.restricted, "is_human": t.label in HUMAN_LABELS,
            })
        return {
            "ts": time.time(),
            "activity": scene,
            "objects": objects,
            "hands": [
                {"name": hd.name, "score": round(hd.score, 3),
                 "center": [round(hd.palm_center[0] / w, 4), round(hd.palm_center[1] / h, 4)]}
                for hd in hands
            ],
            "interactions": [
                {"hand": it.hand.name, "object": it.track.label, "track_id": it.track.track_id,
                 "level": it.level.name, "distance": round(it.norm_distance, 2)}
                for it in best.values()
            ],
            "zones": zones,
            "human_present": self.activity.human_present,
        }

    def health(self) -> dict:
        """Per-component status. Green only when there is evidence the component works."""
        now = time.time()
        src = self.source
        processing = now - self.last_process_ts < 2.0 and not self.paused

        def comp(state, detail):
            return {"state": state, "detail": detail}

        if src is None:
            cam = comp("error", "no source")
        else:
            st = src.state
            label = f"webcam {src.value}" if src.kind == "webcam" else Path(str(src.value)).name
            detail = {"connected": f"{label} - {src.resolution[0]}x{src.resolution[1]} ({src.backend})",
                      "connecting": f"opening {label}...", "stalled": f"{label}: no new frames for >{src.stall_s:.0f}s",
                      "ended": f"{label}: end of file", "error": src.error}.get(st, st)
            cam = comp(st, detail)

        if self.detector is None:
            yolo = comp("error", self.detector_error)
        else:
            yolo = comp("active" if now - self.last_detect_ts < 2.0 else "ready",
                        f"{self.detector.model_name} ({self.detector.mode}), every {self.detect_every} frame(s)")
        if self.hands is None:
            mp = comp("disabled", "hands.enabled = false")
        elif not self.hands.available:
            mp = comp("error", self.hands.error)
        else:
            mp = comp("active" if now - self.last_hands_ts < 2.0 else "ready", self.hands.backend)
        n_zones = sum(1 for z in self.safety.zones if z.enabled and z.type == "restricted")
        if self.voice.state == "error":
            voice = comp("error", self.voice.error)
        elif not self.voice.enabled:
            voice = comp("muted", "muted by operator")
        else:
            voice = comp(self.voice.state, "offline TTS (pyttsx3)" if self.voice.available else "initialising")
        return {
            "camera": cam,
            "yolo": yolo,
            "mediapipe": mp,
            "activity": comp("active" if processing else "idle",
                             "temporal state machine running" if processing else "waiting for frames"),
            "safety": comp("active" if processing else "idle", f"{n_zones} restricted zone(s) enabled"),
            "voice": voice,
        }

    def status(self) -> dict:
        src = self.source
        return {
            "running": not self._stop.is_set(),
            "paused": self.paused,
            "mode": ("file" if src and src.is_file else "webcam"),
            "source": (str(src.value) if src else ""),
            "camera_ok": bool(src and src.state == "connected"),
            "camera_error": src.error if src else "",
            "is_file": bool(src and src.is_file),
            "fps": round(self.fps, 1),
            "latency_ms": {k: round(v, 1) for k, v in self.latency.items()},
            "frame_size": list(self.frame_size),
            "frames_processed": self.frames_processed,
            "last_frame_age_s": round(time.time() - self.last_process_ts, 2) if self.last_process_ts else None,
            "processing_errors": self.errors,
            "last_error": self.last_error,
            "max_processing_fps": round(1.0 / self.min_period, 1),
            "model": self.detector.model_name if self.detector else "unavailable",
            "detector_mode": self.cfg["detector"]["mode"],
            "hand_backend": self.hands.backend if self.hands else "disabled",
            "voice_enabled": self.voice.enabled,
            "voice_available": self.voice.available,
            "uptime_s": round(time.time() - self.started_at),
            "session_start": self.events.session_start,
            "log_file": self.events.file.name,
            "health": self.health(),
            "cameras": self.cameras,
            "config_warnings": self.cfg.get("_warnings", []),
        }

    def snapshot(self) -> dict:
        with self._snap_lock:
            snap = dict(self._snapshot)
        if "activity" not in snap:
            snap.update(activity=derive_scene(self.activity, self.safety, []), objects=[], hands=[],
                        interactions=[], zones=self.safety.zone_state(), human_present=False)
        snap["status"] = self.status()
        snap["workflow"] = self.workflow.as_dict()
        snap["alerts"] = [
            {"id": e.id, "time": e.time, "ts": e.ts, "type": e.type, "severity": e.severity, "message": e.message,
             "subject": e.subject}
            for e in self.events.active_alerts()
        ]
        snap["counts"] = self.events.counts()
        snap["app"] = self.cfg["app"]
        return snap

    def wait_jpeg(self, timeout=1.0) -> bytes | None:
        self._last_viewer = time.time()
        with self._jpeg_cond:
            self._jpeg_cond.wait(timeout)
            return self._jpeg

    @property
    def latest_jpeg(self) -> bytes | None:
        return self._jpeg
