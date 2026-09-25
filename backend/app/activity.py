"""Temporal reasoning + per-object activity state machine.

A single frame is never trusted. Each tracked object owns a small state machine

    IDLE -> HAND_NEAR_EQUIPMENT -> INTERACTING -> MANIPULATING -> COMPLETED -> IDLE

and every transition needs temporal evidence (N consecutive frames, a minimum duration,
a release period, a cooldown). Perception drop-outs are handled with grace periods:

  * hand tracking lost during an interaction  -> state is HELD (shown as HAND TRACKING LOST);
    after hand_lost_grace_s the interaction ends as "tracking lost", never as "completed"
  * object temporarily not detected           -> tracker keeps the box (OBJECT TEMPORARILY LOST);
    if the track is dropped mid-interaction the activity is NOT counted as completed

This is rule-based activity recognition on top of pretrained perception. It is not a
trained activity-classification network.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from .detector import Track
from .events import EventLog, activity_code
from .geometry import dist
from .interaction import HUMAN_LABELS, Interaction, Proximity


class State(str, Enum):
    IDLE = "IDLE"
    HAND_NEAR = "HAND_NEAR_EQUIPMENT"
    INTERACTING = "INTERACTING"
    MANIPULATING = "MANIPULATING"
    COMPLETED = "COMPLETED"


PRIORITY = {State.MANIPULATING: 4, State.INTERACTING: 3, State.HAND_NEAR: 2, State.COMPLETED: 1, State.IDLE: 0}
ACTIVE = (State.INTERACTING, State.MANIPULATING)


@dataclass
class CompletedActivity:
    track_id: int
    object: str
    action: str          # "interact" | "move"
    duration: float
    displacement: float  # fraction of frame diagonal
    hand: str
    ts: float = field(default_factory=time.time)


@dataclass
class ObjectActivity:
    track_id: int
    label: str
    window: int = 15
    state: State = State.IDLE
    state_since: float = field(default_factory=time.time)
    frames_in_state: int = 0
    near_count: int = 0
    contact_count: int = 0
    release_count: int = 0          # consecutive frames with a visible hand that is NOT in contact
    far_count: int = 0
    missing_in_contact: int = 0
    anchor: tuple[float, float] | None = None
    started_at: float = 0.0
    max_disp: float = 0.0
    hand: str = ""
    prolonged_warned: bool = False
    last_action: str = ""
    level: Proximity = Proximity.FAR
    hand_lost_since: float | None = None
    announced: bool = False
    evidence: deque = field(default_factory=deque)   # recent proximity levels (None = hand not tracked)

    def __post_init__(self):
        self.evidence = deque(maxlen=self.window)

    def set(self, s: State, now: float):
        self.state, self.state_since, self.frames_in_state = s, now, 0

    @property
    def hand_lost(self) -> bool:
        return self.hand_lost_since is not None

    def stability(self) -> float:
        """Share of recent frames whose evidence agrees with the current state (temporal consistency).
        Heuristic - NOT a trained activity-classifier probability."""
        if not self.evidence:
            return 0.0
        if self.state in ACTIVE:
            ok = sum(1 for e in self.evidence if e == Proximity.CONTACT or (e is None and self.hand_lost))
        elif self.state == State.HAND_NEAR:
            ok = sum(1 for e in self.evidence if e is not None and e >= Proximity.NEAR)
        else:
            ok = sum(1 for e in self.evidence if e in (Proximity.FAR, None))
        return ok / len(self.evidence)

    def description(self) -> str:
        if self.hand_lost and self.state in ACTIVE:
            return f"Hand tracking lost - holding '{self.label}' interaction"
        return {
            State.IDLE: f"{self.label} idle",
            State.HAND_NEAR: f"Hand near {self.label}",
            State.INTERACTING: f"Interacting with {self.label}",
            State.MANIPULATING: f"Moving {self.label}",
            State.COMPLETED: f"{'Moved' if self.last_action == 'move' else 'Handled'} {self.label}",
        }[self.state]


class ActivityRecognizer:
    def __init__(self, cfg: dict, events: EventLog):
        a = cfg["activity"]
        self.approach_frames = int(a["approach_confirm_frames"])
        self.contact_frames = int(a["interaction_confirm_frames"])
        self.release_frames = int(a["release_confirm_frames"])
        self.min_interaction_s = float(a["min_interaction_s"])
        self.cooldown_s = float(a["interaction_cooldown_s"])
        self.move_threshold = float(a["move_threshold"])
        self.missing_as_held = int(a["missing_as_held_frames"])
        self.hand_lost_grace_s = float(a["hand_lost_grace_s"])
        self.max_interaction_s = float(a["max_interaction_s"])
        self.person_confirm = int(a["person_confirm_frames"])
        self.person_lost = int(a["person_lost_frames"])
        self.window = int(a["stability_window"])
        self.events = events
        self.objects: dict[int, ObjectActivity] = {}

        self.human_present = False
        self._presence_on = 0
        self._presence_off = 0
        self._announced_labels: dict[str, float] = {}

    def reset(self):
        self.objects.clear()
        self.human_present = False
        self._presence_on = self._presence_off = 0
        self._announced_labels.clear()

    # ------------------------------------------------------------------
    def update(self, tracks: list[Track], best: dict[int, Interaction], n_hands: int,
               frame_diag: float, now: float | None = None) -> list[CompletedActivity]:
        now = now or time.time()
        completed: list[CompletedActivity] = []
        self._update_presence(tracks, n_hands)
        hands_tracked = n_hands > 0

        live_ids = set()
        for t in tracks:
            if t.label in HUMAN_LABELS:
                continue
            live_ids.add(t.track_id)
            oa = self.objects.get(t.track_id)
            if oa is None:
                oa = self.objects[t.track_id] = ObjectActivity(t.track_id, t.label, window=self.window)
            if not oa.announced and t.visible_frames >= self.approach_frames + 3:
                oa.announced = True
                self._announce_object(t, now)
            c = self._step(oa, t, best.get(t.track_id), hands_tracked, frame_diag, now)
            if c:
                completed.append(c)

        # track dropped by the tracker (object not detected for too long)
        for tid in list(self.objects):
            if tid not in live_ids:
                oa = self.objects.pop(tid)
                if oa.state in ACTIVE:
                    self.events.emit(
                        "OBJECT_LOST_DURING_INTERACTION", "info",
                        f"{oa.label} no longer detected during interaction - activity not counted as completed",
                        activity=activity_code(oa.label, "INTERACTION_INTERRUPTED"), state="IDLE",
                        subject=oa.label, track_id=oa.track_id,
                    )
        return completed

    # ------------------------------------------------------------------
    def _step(self, oa: ObjectActivity, t: Track, it: Interaction | None, hands_tracked: bool,
              diag: float, now: float) -> CompletedActivity | None:
        oa.frames_in_state += 1
        if not hands_tracked:
            # No hand landmarks at all this frame: this is a perception drop-out, not evidence of release.
            oa.evidence.append(None)
            oa.near_count = oa.contact_count = 0
            if oa.state in ACTIVE:
                if oa.hand_lost_since is None:
                    oa.hand_lost_since = now
                    self.events.emit(
                        "HAND_TRACKING_LOST", "info", f"Hand tracking lost while interacting with {oa.label} - holding state",
                        activity=activity_code(oa.label, "INTERACTION"), state=oa.state.value, subject=oa.label,
                    )
                elif now - oa.hand_lost_since > self.hand_lost_grace_s:
                    oa.hand_lost_since = None
                    oa.set(State.IDLE, now)
                    self.events.emit(
                        "INTERACTION_ENDED_TRACKING_LOST", "info",
                        f"Interaction with {oa.label} ended because hand tracking was lost - not counted as completed",
                        activity=activity_code(oa.label, "INTERACTION_INTERRUPTED"), state="IDLE", subject=oa.label,
                    )
                return None
            if oa.state == State.HAND_NEAR:
                oa.far_count += 1
                if oa.far_count >= self.release_frames:
                    oa.set(State.IDLE, now)
            elif oa.state == State.COMPLETED and now - oa.state_since >= self.cooldown_s:
                oa.set(State.IDLE, now)
            return None

        if oa.hand_lost_since is not None:
            oa.hand_lost_since = None  # hand re-acquired within the grace period -> continue as before

        lv = it.level if it else Proximity.FAR
        oa.level = lv
        oa.evidence.append(lv)
        oa.near_count = oa.near_count + 1 if lv >= Proximity.NEAR else 0
        oa.far_count = oa.far_count + 1 if lv == Proximity.FAR else 0
        if lv == Proximity.CONTACT:
            oa.contact_count += 1
            oa.release_count = 0
        else:
            oa.contact_count = 0
            oa.release_count += 1
        if it and lv >= Proximity.NEAR:
            oa.hand = it.hand.name

        s = oa.state
        if s == State.COMPLETED:
            if now - oa.state_since >= self.cooldown_s:
                oa.set(State.IDLE, now)
                s = State.IDLE
            else:
                return None  # cooldown: no new interaction on this object yet

        if s == State.IDLE:
            if oa.contact_count >= self.contact_frames:
                self._start_interaction(oa, t, now)
            elif oa.near_count >= self.approach_frames:
                oa.set(State.HAND_NEAR, now)
                self.events.emit(
                    "HAND_NEAR_EQUIPMENT", "info", f"{oa.hand or 'Hand'} near {oa.label}",
                    activity=activity_code(oa.label, "APPROACH"), state=State.HAND_NEAR.value,
                    subject=oa.label, stability=oa.stability(), track_id=oa.track_id,
                )

        elif s == State.HAND_NEAR:
            if oa.contact_count >= self.contact_frames:
                self._start_interaction(oa, t, now)
            elif oa.far_count >= self.release_frames:
                oa.set(State.IDLE, now)
                self.events.emit(
                    "HAND_WITHDRAWN", "info", f"Hand withdrew from {oa.label} without interaction",
                    activity=activity_code(oa.label, "APPROACH_ABORTED"), state="IDLE", subject=oa.label,
                )

        elif s in ACTIVE:
            disp = dist(t.center, oa.anchor) / diag if oa.anchor else 0.0
            oa.max_disp = max(oa.max_disp, disp)
            oa.missing_in_contact = oa.missing_in_contact + 1 if (not t.visible and lv >= Proximity.NEAR) else 0

            if s == State.INTERACTING:
                if disp >= self.move_threshold:
                    oa.set(State.MANIPULATING, now)
                    self.events.emit(
                        "OBJECT_MOVED", "info", f"{oa.label} is being moved",
                        activity=activity_code(oa.label, "MOVE"), state=State.MANIPULATING.value,
                        subject=oa.label, stability=oa.stability(), displacement=round(disp, 3),
                    )
                elif oa.missing_in_contact >= self.missing_as_held:
                    oa.set(State.MANIPULATING, now)
                    self.events.emit(
                        "OBJECT_PICKED_UP", "info", f"{oa.label} picked up (occluded by the hand)",
                        activity=activity_code(oa.label, "PICK_UP"), state=State.MANIPULATING.value,
                        subject=oa.label, stability=oa.stability(),
                    )

            if not oa.prolonged_warned and now - oa.started_at > self.max_interaction_s:
                oa.prolonged_warned = True
                self.events.emit(
                    "PROLONGED_INTERACTION", "warning",
                    f"Prolonged interaction with {oa.label} ({now - oa.started_at:.0f}s)",
                    activity=activity_code(oa.label, "INTERACTION"), state="WARNING", subject=oa.label,
                )

            # release: the hand is visible and has been away from the object for N frames,
            # and the object itself is visible again (not just hidden behind the hand)
            if oa.release_count >= self.release_frames and t.visible:
                duration = now - oa.started_at
                if duration < self.min_interaction_s:
                    oa.set(State.IDLE, now)
                    self.events.emit(
                        "BRIEF_CONTACT_IGNORED", "info",
                        f"Brief contact with {oa.label} ({duration:.1f}s) - below minimum interaction duration",
                        activity=activity_code(oa.label, "BRIEF_CONTACT"), state="IDLE", subject=oa.label,
                    )
                    return None
                action = "move" if oa.state == State.MANIPULATING else "interact"
                return self._complete(oa, action, now)
        return None

    def _start_interaction(self, oa: ObjectActivity, t: Track, now: float):
        oa.set(State.INTERACTING, now)
        oa.anchor = t.center
        oa.started_at = now
        oa.max_disp = 0.0
        oa.missing_in_contact = 0
        oa.prolonged_warned = False
        oa.release_count = 0
        self.events.emit(
            "INTERACTION_STARTED", "info", f"Interaction confirmed: {oa.hand or 'hand'} with {oa.label}",
            activity=activity_code(oa.label, "INTERACTION"), state=State.INTERACTING.value,
            subject=oa.label, stability=oa.stability(), hand=oa.hand, track_id=oa.track_id,
        )

    def _complete(self, oa: ObjectActivity, action: str, now: float) -> CompletedActivity:
        duration = now - oa.started_at if oa.started_at else 0.0
        stab = oa.stability()
        oa.last_action = action
        oa.set(State.COMPLETED, now)
        if action == "move":
            msg = f"Activity completed: {oa.label} moved and released ({oa.max_disp * 100:.0f}% of view, {duration:.1f}s)"
        else:
            msg = f"Activity completed: {oa.label} handled for {duration:.1f}s"
        self.events.emit(
            "ACTIVITY_COMPLETED", "success", msg,
            activity=activity_code(oa.label, "MOVE" if action == "move" else "INTERACTION"),
            state=State.COMPLETED.value, subject=oa.label, stability=stab,
            action=action, duration=round(duration, 2), displacement=round(oa.max_disp, 3), hand=oa.hand,
        )
        return CompletedActivity(oa.track_id, oa.label, action, duration, oa.max_disp, oa.hand, now)

    # ------------------------------------------------------------------
    def _update_presence(self, tracks: list[Track], n_hands: int):
        person = any(t.label in HUMAN_LABELS and t.visible for t in tracks) or n_hands > 0
        if person:
            self._presence_on += 1
            self._presence_off = 0
        else:
            self._presence_off += 1
            self._presence_on = 0
        if not self.human_present and self._presence_on >= self.person_confirm:
            self.human_present = True
            self.events.emit("PERSON_DETECTED", "info", "Person detected in the monitored area",
                             activity="PERSON_DETECTED", state="PERSON_DETECTED")
        elif self.human_present and self._presence_off >= self.person_lost:
            self.human_present = False
            self.events.emit("PERSON_LEFT", "info", "Person left the monitored area",
                             activity="PERSON_LEFT", state="IDLE")

    def _announce_object(self, t: Track, now: float):
        if now - self._announced_labels.get(t.label, 0) > 60:
            self._announced_labels[t.label] = now
            self.events.emit(
                "OBJECT_DETECTED", "info", f"{t.label} detected (YOLO '{t.cls_name}', {t.conf * 100:.0f}% detection confidence)",
                activity="OBJECT_DETECTED", state="IDLE", subject=t.label, track_id=t.track_id, cls=t.cls_name,
                detection_confidence=round(t.conf, 3),
            )

    # ------------------------------------------------------------------
    def focus(self) -> ObjectActivity | None:
        """The object whose activity is currently most advanced (for the dashboard)."""
        if not self.objects:
            return None
        oa = max(self.objects.values(), key=lambda o: (PRIORITY[o.state], o.state_since))
        return None if oa.state == State.IDLE else oa
