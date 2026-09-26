"""Human-Object Interaction - 2D vision-based interaction estimation.

Connects YOLO object boxes with MediaPipe hand landmarks and estimates, per frame,
whether a hand is FAR from, NEAR, or in CONTACT with each object. "CONTACT" means a
fingertip/palm key point lies inside the (slightly expanded) 2D bounding box - it is an
image-plane estimate, NOT true physical contact detection (no depth is available).
Thresholds are scaled by the apparent hand size, so they work at different distances.
A single frame never confirms an interaction - see activity.py for temporal confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .detector import Track
from .geometry import expand, point_box_distance, point_in_box
from .hands import Hand

HUMAN_LABELS = {"Person"}   # dashboard label of the COCO "person" class


def reasoning_tracks(tracks: list[Track], surface_labels: set[str]) -> list[Track]:
    """Tracks the reasoning layers (interaction, activity, safety, scene) work on.

    Surface stand-ins such as the tray are detected and drawn, but a hand over the tray is not an
    equipment interaction - everything else stands on it. Approaching the work area is still
    covered by the workstation zone (safety.py)."""
    return [t for t in tracks if t.label not in surface_labels] if surface_labels else list(tracks)


class Proximity(IntEnum):
    FAR = 0
    NEAR = 1
    CONTACT = 2


@dataclass
class Interaction:
    hand: Hand
    track: Track
    level: Proximity
    distance: float          # pixels from nearest key point to the box
    norm_distance: float     # distance / hand size


class InteractionAnalyzer:
    def __init__(self, cfg: dict):
        icfg = cfg["interaction"]
        self.contact_margin = float(icfg["contact_margin"])
        self.near_factor = float(icfg["near_factor"])

    def analyze(self, hands: list[Hand], tracks: list[Track]) -> list[Interaction]:
        results: list[Interaction] = []
        objects = [t for t in tracks if t.label not in HUMAN_LABELS]
        for hand in hands:
            kps = hand.key_points
            for t in objects:
                grow = expand(t.box, self.contact_margin)
                d = min(point_box_distance(p, t.box) for p in kps)
                nd = d / hand.size
                if any(point_in_box(p, grow) for p in kps):
                    level = Proximity.CONTACT
                elif nd <= self.near_factor:
                    level = Proximity.NEAR
                else:
                    level = Proximity.FAR
                results.append(Interaction(hand, t, level, d, nd))
        return results

    @staticmethod
    def per_object(interactions: list[Interaction]) -> dict[int, Interaction]:
        """Strongest interaction for each object track (closest hand wins)."""
        best: dict[int, Interaction] = {}
        for it in interactions:
            cur = best.get(it.track.track_id)
            if cur is None or (it.level, -it.norm_distance) > (cur.level, -cur.norm_distance):
                best[it.track.track_id] = it
        return best
