"""Hand landmark perception (pretrained MediaPipe HandLandmarker).

MediaPipe answers: "Where are the person's hands and fingertips?" It is a pretrained
model and has not been trained on BAS data. Uses the MediaPipe Tasks API; falls back to
the legacy `mp.solutions.hands` API only on old MediaPipe versions that still ship it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .config import resolve_path
from .geometry import Point, dist

log = logging.getLogger("bas.hands")

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, RING_TIP, PINKY_MCP, PINKY_TIP = (
    0, 4, 5, 8, 9, 12, 16, 17, 20,
)
FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class Hand:
    index: int
    handedness: str               # "Left" / "Right" (as seen by the camera)
    score: float
    landmarks: list[Point]        # 21 points in pixels

    @property
    def palm_center(self) -> Point:
        pts = [self.landmarks[i] for i in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)]
        return (sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4)

    @property
    def key_points(self) -> list[Point]:
        """Points used for contact testing: fingertips + palm centre."""
        return [self.landmarks[i] for i in (THUMB_TIP, INDEX_TIP, MIDDLE_TIP)] + [self.palm_center]

    @property
    def size(self) -> float:
        """Hand scale in pixels (wrist -> middle MCP). Makes thresholds distance-invariant."""
        return max(dist(self.landmarks[WRIST], self.landmarks[MIDDLE_MCP]), 1.0)

    @property
    def name(self) -> str:
        return f"{self.handedness} hand"


class HandTracker:
    """Pretrained MediaPipe HandLandmarker. The model file must exist locally (prepared by setup.bat)."""

    def __init__(self, cfg: dict):
        hcfg = cfg.get("hands", {})
        self.max_hands = int(hcfg.get("max_hands", 2))
        self.backend = "none"
        self.error = ""
        self._landmarker = None
        self._legacy = None
        self._t0 = time.monotonic()
        self._last_ts = -1

        self.model_path = resolve_path(hcfg.get("model", "models/hand_landmarker.task"))
        det_conf = float(hcfg.get("min_detection_conf", 0.5))
        trk_conf = float(hcfg.get("min_tracking_conf", 0.5))

        try:
            import mediapipe as mp
        except Exception as e:  # noqa: BLE001
            self.error = f"mediapipe is not installed or failed to import: {e}"
            log.error(self.error)
            return

        if self.model_path.exists():
            try:
                from mediapipe.tasks.python import BaseOptions, vision

                opts = vision.HandLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(self.model_path)),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=self.max_hands,
                    min_hand_detection_confidence=det_conf,
                    min_hand_presence_confidence=det_conf,
                    min_tracking_confidence=trk_conf,
                )
                self._landmarker = vision.HandLandmarker.create_from_options(opts)
                self.backend = "mediapipe-tasks"
            except Exception as e:  # noqa: BLE001
                self.error = f"HandLandmarker failed to load {self.model_path.name}: {e}"
        else:
            self.error = (f"Hand model not found: {self.model_path}. Run setup.bat "
                          f"(python backend/scripts/download_models.py) before the demo.")

        if self._landmarker is None and hasattr(mp, "solutions"):
            # very old MediaPipe versions ship the legacy solution with a bundled model
            try:
                self._legacy = mp.solutions.hands.Hands(
                    static_image_mode=False, max_num_hands=self.max_hands,
                    min_detection_confidence=det_conf, min_tracking_confidence=trk_conf,
                )
                self.backend = "mediapipe-legacy"
                self.error = ""
            except Exception as e:  # noqa: BLE001
                self.error += f" | legacy API failed: {e}"
        if self.error:
            log.error("Hand tracking unavailable: %s", self.error)
        log.info("Hand tracker backend: %s", self.backend)

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def process(self, frame_bgr: np.ndarray) -> list[Hand]:
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands: list[Hand] = []

        if self._landmarker is not None:
            import mediapipe as mp

            ts = int((time.monotonic() - self._t0) * 1000)
            if ts <= self._last_ts:  # timestamps must be strictly increasing
                ts = self._last_ts + 1
            self._last_ts = ts
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self._landmarker.detect_for_video(img, ts)
            for i, lms in enumerate(res.hand_landmarks):
                cat = res.handedness[i][0] if res.handedness and res.handedness[i] else None
                hands.append(
                    Hand(
                        index=i,
                        handedness=(cat.category_name if cat else "Unknown"),
                        score=float(cat.score if cat else 1.0),
                        landmarks=[(lm.x * w, lm.y * h) for lm in lms],
                    )
                )
        else:
            res = self._legacy.process(rgb)
            if res.multi_hand_landmarks:
                for i, lms in enumerate(res.multi_hand_landmarks):
                    cls = res.multi_handedness[i].classification[0]
                    hands.append(
                        Hand(i, cls.label, float(cls.score), [(lm.x * w, lm.y * h) for lm in lms.landmark])
                    )
        return self._dedupe(hands)

    @staticmethod
    def _dedupe(hands: list[Hand]) -> list[Hand]:
        """MediaPipe occasionally reports the same physical hand twice - keep the best one."""
        kept: list[Hand] = []
        for hd in sorted(hands, key=lambda x: -x.score):
            if all(dist(hd.palm_center, k.palm_center) > 0.6 * max(hd.size, k.size) for k in kept):
                kept.append(hd)
        names: dict[str, int] = {}
        for hd in kept:  # two hands with the same side label -> make the names unique
            n = names.get(hd.handedness, 0) + 1
            names[hd.handedness] = n
            if n > 1:
                hd.handedness = f"{hd.handedness} #{n}"
        return kept

    def close(self):
        try:
            if self._landmarker is not None:
                self._landmarker.close()
            if self._legacy is not None:
                self._legacy.close()
        except Exception:  # noqa: BLE001
            pass
