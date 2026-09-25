"""Object/person perception (pretrained YOLO) + lightweight multi-object tracking.

YOLO answers: "What objects are present?" It is a PRETRAINED model (COCO classes). It has
not been trained on BAS hardware; everyday objects are mapped to "stand-in" labels.

The tracker is a deliberately simple IoU tracker with a centre-distance fallback for fast
motion. It is adequate for a controlled single-camera demo; it is not mission-grade
tracking (see docs/architecture.md - limitations and future path, e.g. ByteTrack/OC-SORT).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import resolve_path
from .geometry import Box, center, dist, iou

log = logging.getLogger("bas.detector")


class ModelMissingError(FileNotFoundError):
    pass


@dataclass
class Detection:
    cls_name: str   # raw model class, e.g. "bottle"
    label: str      # dashboard label, e.g. "Sample Container Stand-in"
    conf: float
    box: Box


@dataclass
class Track:
    track_id: int
    cls_name: str
    label: str
    conf: float
    box: Box
    age: int = 0            # detector runs since creation
    missed: int = 0         # consecutive detector runs without a matching detection
    visible_frames: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=90))  # recent centres

    @property
    def visible(self) -> bool:
        return self.missed == 0

    @property
    def center(self):
        return center(self.box)

    @property
    def size(self) -> float:
        return max(self.box[2] - self.box[0], self.box[3] - self.box[1], 1.0)


class SimpleTracker:
    """Greedy IoU tracker + same-label centre-distance fallback. Dependency-free and CPU-cheap."""

    def __init__(self, iou_match: float = 0.3, max_missed: int = 20, smoothing: float = 0.5,
                 min_hits: int = 1, center_match: float = 0.75):
        self.iou_match = iou_match
        self.center_match = center_match
        self.max_missed = max_missed
        self.smoothing = smoothing
        self.min_hits = min_hits      # a new object must be seen in N detector runs before it is trusted
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def reset(self):
        self.tracks.clear()
        self._next_id = 1

    def _assign(self, t: Track, d: Detection):
        a = self.smoothing
        t.box = tuple(a * o + (1 - a) * n for o, n in zip(t.box, d.box))  # type: ignore
        t.conf = d.conf
        t.missed = 0
        t.visible_frames += 1

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = set(range(len(detections)))
        used_t: set[int] = set()

        # 1) IoU matching (same label), best first
        pairs = []
        for tid, t in self.tracks.items():
            for di, d in enumerate(detections):
                if d.label == t.label:
                    score = iou(t.box, d.box)
                    if score >= self.iou_match:
                        pairs.append((score, tid, di))
        for _, tid, di in sorted(pairs, reverse=True):
            if tid in used_t or di not in unmatched:
                continue
            used_t.add(tid)
            unmatched.discard(di)
            self._assign(self.tracks[tid], detections[di])

        # 2) fallback: centre distance for fast-moving objects (IoU drops to 0 between frames)
        if self.center_match > 0:
            pairs = []
            for tid, t in self.tracks.items():
                if tid in used_t:
                    continue
                for di in unmatched:
                    d = detections[di]
                    if d.label != t.label:
                        continue
                    gap = dist(t.center, center(d.box)) / t.size
                    if gap <= self.center_match:
                        pairs.append((gap, tid, di))
            for _, tid, di in sorted(pairs):
                if tid in used_t or di not in unmatched:
                    continue
                used_t.add(tid)
                unmatched.discard(di)
                t = self.tracks[tid]
                t.box = detections[di].box        # jump: no smoothing across a large move
                t.conf, t.missed = detections[di].conf, 0
                t.visible_frames += 1

        for tid, t in list(self.tracks.items()):
            t.age += 1
            if tid not in used_t:
                t.missed += 1
                if t.missed > self.max_missed:
                    del self.tracks[tid]
                    continue
            t.history.append(t.center)

        for di in unmatched:
            d = detections[di]
            t = Track(self._next_id, d.cls_name, d.label, d.conf, d.box, visible_frames=1)
            t.history.append(t.center)
            self.tracks[self._next_id] = t
            self._next_id += 1
        # only confirmed tracks go downstream -> one-frame false detections are ignored
        return [t for t in self.tracks.values() if t.visible_frames >= self.min_hits]


class ObjectDetector:
    """Loads LOCAL weights only. A missing file is an error - nothing is downloaded at run time."""

    def __init__(self, cfg: dict):
        dcfg = cfg["detector"]
        self.mode = dcfg.get("mode", "coco")
        self.conf = float(dcfg.get("conf", 0.35))
        # optional stricter confidence for classes that cause false positives
        self.class_conf: dict[str, float] = {k: float(v) for k, v in (dcfg.get("class_conf") or {}).items()}
        self.imgsz = int(dcfg.get("imgsz", 640))
        self.class_map: dict[str, str] = dict(dcfg.get("class_map", {}))
        key = {"world": "world_model", "custom": "custom_model"}.get(self.mode, "model")
        self.model_path = resolve_path(dcfg.get(key, "models/yolo11n.pt"))
        self.model_name = self.model_path.name
        if not self.model_path.exists():
            raise ModelMissingError(
                f"YOLO weights not found: {self.model_path}. Run setup.bat (or "
                f"python backend/scripts/download_models.py) before the demo."
            )

        from ultralytics import YOLO  # imported lazily so unit tests don't need torch

        self.model = YOLO(str(self.model_path))
        if self.mode == "world":
            prompts: dict[str, str] = dict(dcfg.get("world_prompts", {}))
            self.model.set_classes(list(prompts.keys()) + ["person"])
            self.class_map = {**prompts, "person": self.class_map.get("person", "Person")}
        elif self.mode == "custom":
            for name in self.model.names.values():  # a custom model's classes are used as-is unless mapped
                self.class_map.setdefault(name, name.replace("_", " ").title())

        names = self.model.names  # id -> name
        self.class_ids = [i for i, n in names.items() if n in self.class_map]
        log.info("YOLO loaded: %s (%s), %d mapped classes", self.model_name, self.mode, len(self.class_ids))
        # warm-up so the first real frame is not slow
        self.model.predict(np.zeros((360, 640, 3), np.uint8), imgsz=self.imgsz, verbose=False)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, classes=self.class_ids or None, verbose=False
        )[0]
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        names = res.names
        boxes = res.boxes
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            name = names[int(cls)]
            label = self.class_map.get(name)
            if label is None or conf < self.class_conf.get(name, 0.0):
                continue
            out.append(Detection(name, label, float(conf), tuple(float(v) for v in xyxy)))  # type: ignore
        return out
