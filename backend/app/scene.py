"""Scene-level state shown on the dashboard.

Combines the per-object activity state machines with presence, the workstation zone and
the safety engine into one headline state:

  IDLE -> PERSON_DETECTED -> APPROACHING (workstation) -> HAND_NEAR_EQUIPMENT
       -> INTERACTING -> MANIPULATING -> COMPLETED            (activity progression)
  WARNING / CRITICAL                                         (safety overrides the headline)

Everything here is derived from real, already-debounced pipeline state - nothing is simulated.
"""
from __future__ import annotations

import time

from .activity import ActivityRecognizer
from .detector import Track
from .interaction import HUMAN_LABELS
from .safety import SafetyMonitor

ACTIVITY_STATES = ["IDLE", "PERSON_DETECTED", "APPROACHING", "HAND_NEAR_EQUIPMENT",
                   "INTERACTING", "MANIPULATING", "COMPLETED"]


def derive_scene(activity: ActivityRecognizer, safety: SafetyMonitor, tracks: list[Track],
                 now: float | None = None) -> dict:
    now = now or time.time()
    focus = activity.focus()
    conf_by_id = {t.track_id: t.conf for t in tracks}
    lost_objects = sorted({t.label for t in tracks if t.label not in HUMAN_LABELS and not t.visible})

    out = {
        "activity_state": "IDLE", "label": "Monitoring - no person in view", "object": None, "hand": None,
        "stability": None, "detection_confidence": None, "since": None,
        "hand_tracking_lost": False, "objects_temporarily_lost": lost_objects,
    }
    if focus is not None:
        out.update(
            activity_state=focus.state.value, label=focus.description(), object=focus.label,
            hand=focus.hand or None, stability=round(focus.stability(), 3),
            detection_confidence=round(conf_by_id[focus.track_id], 3) if focus.track_id in conf_by_id else None,
            since=focus.state_since, hand_tracking_lost=focus.hand_lost,
        )
    elif safety.workstation_active():
        out.update(activity_state="APPROACHING", label="Approaching workstation")
    elif activity.human_present:
        out.update(activity_state="PERSON_DETECTED", label="Person detected - no equipment interaction")

    level = safety.level(now)
    out["safety_level"] = level
    out["state"] = level if level != "NORMAL" else out["activity_state"]
    zone = next((z for z in safety.zones if z.type == "restricted" and z.active), None)
    if zone is not None:
        out["label"] = ("CRITICAL - hand remained in " if zone.level == "CRITICAL" else "Hand entered ") + zone.name
    return out
