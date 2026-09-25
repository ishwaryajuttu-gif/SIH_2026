"""Rule-based safety engine on top of the activity layer.

Zones (drawn in the dashboard, normalised coordinates):
  restricted  : hand inside for restricted_confirm_frames -> WARNING (RESTRICTED_ZONE_ENTRY)
                still inside after critical_after_s        -> CRITICAL (RESTRICTED_ZONE_CRITICAL)
                outside / not tracked for zone_exit_s      -> RESTRICTED_ZONE_EXIT
  workstation : hand inside (debounced)                    -> APPROACHING_WORKSTATION (info)

Object rules:
  R3 interaction with restricted equipment stand-in                     -> CRITICAL
  R4 object moves while hands are tracked and none is near it           -> WARNING
  R5 all tracked objects shift together (camera bumped)                 -> WARNING
  R6 stable object disappears without having been handled               -> WARNING

Every alert has a per-key cooldown so nothing is raised (or spoken) every frame.
Rule-based logic is used deliberately: it is transparent, testable and needs no
training data. It is a demonstration safety layer, not a certified safety system.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .activity import ACTIVE, ActivityRecognizer, State
from .detector import Track
from .events import SEV_RANK, EventLog, activity_code
from .geometry import any_point_in_polygon, dist
from .hands import Hand
from .interaction import HUMAN_LABELS


@dataclass
class ZoneStatus:
    id: str
    name: str
    type: str                           # restricted | workstation
    enabled: bool
    points: list[list[float]]           # normalised polygon
    inside_frames: int = 0
    last_inside: float = 0.0
    active: bool = False                # hand confirmed inside
    level: str = "NORMAL"               # NORMAL | WARNING | CRITICAL (restricted zones)
    entered_at: float = 0.0
    hands_inside: list[str] = field(default_factory=list)


class SafetyMonitor:
    def __init__(self, cfg: dict, events: EventLog):
        s = cfg["safety"]
        self.zone_frames = int(s["restricted_confirm_frames"])
        self.zone_exit_s = float(s["zone_exit_s"])
        self.critical_after_s = float(s["critical_after_s"])
        self.cooldown = float(s["alert_cooldown_s"])
        self.move_thr = float(s["unexpected_move_threshold"])
        self.move_win = int(s["unexpected_move_window"])
        self.stable_frames = int(s["stable_object_frames"])
        self.missing_s = float(s["missing_object_s"])
        self.recent_alert_s = float(s["recent_alert_s"])
        self.restricted_objects = set(cfg["detector"]["restricted_objects"])
        self.events = events
        self.zones: list[ZoneStatus] = []
        self.set_zones(s.get("zones", []))
        self._last_alert: dict[str, float] = {}
        self._label_last_seen: dict[str, float] = {}
        self._label_stable: dict[str, bool] = {}
        self._label_handled: dict[str, float] = {}
        self._label_missing: set[str] = set()
        self._hands_seen: deque[bool] = deque(maxlen=self.move_win)
        self._recent: deque[tuple[float, str]] = deque(maxlen=50)   # (ts, severity) of raised alerts

    # ------------------------------------------------------------------ zones config
    def set_zones(self, zones: list[dict]):
        self.zones = [
            ZoneStatus(z.get("id", f"zone-{i + 1}"), z.get("name", f"Zone {i + 1}"),
                       z.get("type", "restricted"), bool(z.get("enabled", True)),
                       [list(map(float, p)) for p in z.get("points", [])])
            for i, z in enumerate(zones)
        ]

    def zones_as_dicts(self) -> list[dict]:
        return [{"id": z.id, "name": z.name, "type": z.type, "enabled": z.enabled, "points": z.points}
                for z in self.zones]

    def reset(self):
        self._last_alert.clear()
        self._label_last_seen.clear()
        self._label_stable.clear()
        self._label_handled.clear()
        self._label_missing.clear()
        self._hands_seen.clear()
        self._recent.clear()
        for z in self.zones:
            z.inside_frames = 0
            z.active, z.level = False, "NORMAL"

    def _alert(self, key: str, type_: str, severity: str, msg: str, now: float, **kw) -> bool:
        """Emit an alert unless the same key fired within the cooldown."""
        if now - self._last_alert.get(key, -1e9) < self.cooldown:
            return False
        self._last_alert[key] = now
        self._recent.append((now, severity))
        self.events.emit(type_, severity, msg, **kw)
        return True

    # ------------------------------------------------------------------ main entry
    def update(self, hands: list[Hand], tracks: list[Track], activity: ActivityRecognizer,
               w: int, h: int, now: float | None = None):
        now = now or time.time()
        diag = (w * w + h * h) ** 0.5
        self._hands_seen.append(bool(hands))
        self._check_zones(hands, w, h, now)
        self._check_restricted_objects(activity, now)
        self._check_unexpected_movement(tracks, activity, diag, now)
        self._check_missing(tracks, activity, now)

    # ------------------------------------------------------------------ zones
    def _check_zones(self, hands: list[Hand], w: int, h: int, now: float):
        for z in self.zones:
            if not z.enabled or len(z.points) < 3:
                z.active, z.level, z.hands_inside, z.inside_frames = False, "NORMAL", [], 0
                continue
            poly = [(p[0] * w, p[1] * h) for p in z.points]
            inside = [hd.name for hd in hands if any_point_in_polygon(hd.key_points, poly)]
            z.hands_inside = inside
            if inside:
                z.inside_frames += 1
                z.last_inside = now
            else:
                z.inside_frames = 0

            if not z.active and z.inside_frames >= self.zone_frames:
                z.active, z.entered_at = True, now
                if z.type == "restricted":
                    z.level = "WARNING"
                    self._alert(
                        f"zone:{z.id}", "RESTRICTED_ZONE_ENTRY", "warning",
                        f"Hand entered restricted zone '{z.name}' ({inside[0]})", now,
                        activity="RESTRICTED_ZONE_ENTRY", state="WARNING", subject=z.name, hand=inside[0],
                    )
                else:
                    self.events.emit(
                        "APPROACHING_WORKSTATION", "info", f"Hand entered '{z.name}'",
                        activity="APPROACHING_WORKSTATION", state="APPROACHING", subject=z.name, hand=inside[0],
                    )
            elif z.active and z.type == "restricted" and z.level == "WARNING" \
                    and inside and now - z.entered_at >= self.critical_after_s:
                z.level = "CRITICAL"
                self._alert(
                    f"zone-crit:{z.id}", "RESTRICTED_ZONE_CRITICAL", "critical",
                    f"Critical: hand has remained in restricted zone '{z.name}' for {now - z.entered_at:.1f}s",
                    now, activity="RESTRICTED_ZONE_INTERACTION", state="CRITICAL", subject=z.name,
                    dwell=round(now - z.entered_at, 2),
                )
            elif z.active and not inside and now - z.last_inside >= self.zone_exit_s:
                # hand outside OR not tracked for zone_exit_s -> a brief tracking drop does not clear/re-trigger
                dwell = z.last_inside - z.entered_at
                z.active, z.level = False, "NORMAL"
                self.events.emit(
                    "RESTRICTED_ZONE_EXIT" if z.type == "restricted" else "LEFT_WORKSTATION", "info",
                    f"Hand left '{z.name}' (dwell {dwell:.1f}s)",
                    activity="ZONE_EXIT", state="NORMAL", subject=z.name, dwell=round(dwell, 2),
                )

    # ------------------------------------------------------------------ R3
    def _check_restricted_objects(self, activity: ActivityRecognizer, now: float):
        for oa in activity.objects.values():
            if oa.state != State.IDLE:
                self._label_handled[oa.label] = now
            if oa.label in self.restricted_objects and oa.state in ACTIVE:
                self._alert(
                    f"robj:{oa.label}", "RESTRICTED_OBJECT", "critical",
                    f"Critical: interaction with restricted equipment stand-in '{oa.label}'",
                    now, activity=activity_code(oa.label, "INTERACTION"), state="CRITICAL",
                    subject=oa.label, stability=oa.stability(), hand=oa.hand,
                )

    # ------------------------------------------------------------------ R4 + R5
    def _check_unexpected_movement(self, tracks, activity: ActivityRecognizer, diag, now):
        # Only judge "moved without contact" when hand tracking was actually working during the
        # window; otherwise a lost hand could make a normal pick-up look like an anomaly.
        hands_reliable = len(self._hands_seen) == self.move_win and sum(self._hands_seen) >= 0.6 * self.move_win
        nobody = not activity.human_present
        moved, candidates = [], 0
        for t in tracks:
            if t.label in HUMAN_LABELS or len(t.history) < self.move_win or t.visible_frames < self.move_win:
                continue
            candidates += 1
            d = dist(t.history[-1], t.history[-self.move_win]) / diag
            oa = activity.objects.get(t.track_id)
            untouched = oa is None or (oa.state == State.IDLE and oa.far_count >= self.move_win // 2)
            if d > self.move_thr and untouched and t.visible:
                moved.append(t)
        if not moved:
            return
        if len(moved) >= 2 and len(moved) == candidates:
            self._alert("camera-shift", "CAMERA_SHIFT", "warning",
                        "Camera shift suspected - all tracked objects moved together", now,
                        activity="CAMERA_SHIFT", state="WARNING")
        elif hands_reliable or nobody:
            for t in moved:
                self._alert(
                    f"drift:{t.label}", "UNEXPECTED_MOVEMENT", "warning",
                    f"Unexpected movement: {t.label} moved with no hand near it", now,
                    activity=activity_code(t.label, "UNEXPECTED_MOVEMENT"), state="WARNING", subject=t.label,
                )

    # ------------------------------------------------------------------ R6
    def _check_missing(self, tracks: list[Track], activity: ActivityRecognizer, now: float):
        visible_labels = {t.label for t in tracks if t.visible and t.label not in HUMAN_LABELS}
        for t in tracks:
            if t.label not in HUMAN_LABELS and t.visible_frames >= self.stable_frames:
                self._label_stable[t.label] = True
        for label in visible_labels:
            self._label_last_seen[label] = now
            if label in self._label_missing:
                self._label_missing.discard(label)
                self.events.emit("OBJECT_RETURNED", "info", f"{label} back in view",
                                 activity="OBJECT_RETURNED", state="NORMAL", subject=label)
        for label, seen in self._label_last_seen.items():
            if label in visible_labels or not self._label_stable.get(label) or label in self._label_missing:
                continue
            if now - seen <= self.missing_s:
                continue
            self._label_missing.add(label)
            # handled around the moment it vanished -> the person took it away, not an anomaly
            if self._label_handled.get(label, 0) < seen - 2:
                self._alert(
                    f"missing:{label}", "OBJECT_MISSING", "warning",
                    f"{label} no longer visible and was not handled", now,
                    activity=activity_code(label, "MISSING"), state="WARNING", subject=label,
                )

    # ------------------------------------------------------------------ outputs
    def level(self, now: float | None = None) -> str:
        """Current safety level for the dashboard: NORMAL | WARNING | CRITICAL."""
        now = now or time.time()
        rank = 0
        for z in self.zones:
            if z.type == "restricted" and z.active:
                rank = max(rank, 2 if z.level == "CRITICAL" else 1)
        for ts, sev in self._recent:
            if now - ts <= self.recent_alert_s:
                rank = max(rank, 2 if SEV_RANK[sev] >= SEV_RANK["critical"] else 1)
        return ("NORMAL", "WARNING", "CRITICAL")[rank]

    def workstation_active(self) -> bool:
        return any(z.type == "workstation" and z.active for z in self.zones)

    def zone_state(self) -> list[dict]:
        return [
            {"id": z.id, "name": z.name, "type": z.type, "enabled": z.enabled, "points": z.points,
             "active": z.active, "level": z.level, "hands": z.hands_inside}
            for z in self.zones
        ]
