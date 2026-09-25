"""Unit tests for the reasoning layers (no camera / GPU / model files needed).

    cd backend && python -m pytest -q

Frames are simulated at 15 FPS with synthetic hands/boxes so every rule can be exercised
deterministically. These tests check the LOGIC; they say nothing about perception accuracy.
"""
import copy
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.activity import ActivityRecognizer, State  # noqa: E402
from app.config import SCHEMA, ConfigError, load_config, validate_config  # noqa: E402
from app.detector import Detection, SimpleTracker  # noqa: E402
from app.events import EventLog  # noqa: E402
from app.hands import Hand  # noqa: E402
from app.interaction import InteractionAnalyzer, Proximity  # noqa: E402
from app.procedure import WorkflowTracker  # noqa: E402
from app.safety import SafetyMonitor  # noqa: E402
from app.scene import derive_scene  # noqa: E402
from app.voice import VoiceAlerter  # noqa: E402

W, H = 640, 360
DIAG = (W * W + H * H) ** 0.5
FPS = 15
RESTRICTED = {"id": "z", "name": "RESTRICTED ZONE", "type": "restricted", "enabled": True,
              "points": [[0.8, 0.0], [1.0, 0.0], [1.0, 0.4], [0.8, 0.4]]}
WORKSTATION = {"id": "w", "name": "Workstation", "type": "workstation", "enabled": True,
               "points": [[0.0, 0.4], [0.45, 0.4], [0.45, 1.0], [0.0, 1.0]]}
IN_ZONE = (590, 80)       # a hand position inside RESTRICTED
FAR = (620, 340)          # visible hand far from everything


# ------------------------------------------------------------------ helpers
def make_hand(cx, cy, size=40, name="Right"):
    """Synthetic 21-point hand whose fingertips point up from (cx, cy)."""
    pts = [(cx, cy + size)] * 21
    for i in (5, 9, 13, 17):
        pts[i] = (cx + (i - 11) * 2, cy)               # MCPs -> middle MCP ~ (cx, cy)
    for i in (4, 8, 12, 16, 20):
        pts[i] = (cx + (i - 12) * 2, cy - size * 0.8)  # fingertips
    return Hand(0, name, 0.95, pts)


def base_cfg(tmp_path, zones=None):
    cfg = load_config(use_env=False, use_zone_file=False)
    cfg["logging"]["dir"] = str(tmp_path)
    cfg["safety"]["zones"] = zones if zones is not None else [RESTRICTED]
    return cfg


class Sim:
    """Runs the reasoning pipeline exactly as pipeline.py does, on synthetic frames."""

    def __init__(self, cfg):
        self.ev = EventLog(cfg)
        self.tracker = SimpleTracker(min_hits=1, center_match=cfg["tracker"]["center_match"],
                                     max_missed=cfg["tracker"]["max_missed"])
        self.ana = InteractionAnalyzer(cfg)
        self.act = ActivityRecognizer(cfg, self.ev)
        self.saf = SafetyMonitor(cfg, self.ev)
        self.wf = WorkflowTracker(cfg, self.ev)
        self.t = 1000.0
        self.tracks = []

    def run(self, frames):
        for dets, hands in frames:
            self.t += 1 / FPS
            self.tracks = self.tracker.update(dets)
            best = self.ana.per_object(self.ana.analyze(hands, self.tracks))
            for c in self.act.update(self.tracks, best, len(hands), DIAG, now=self.t):
                self.wf.on_activity(c)
            self.saf.update(hands, self.tracks, self.act, W, H, now=self.t)
        return self

    def types(self):
        return [e.type for e in self.ev.events]

    def of(self, type_):
        return [e for e in self.ev.events if e.type == type_]

    def scene(self):
        return derive_scene(self.act, self.saf, self.tracks, now=self.t)


def cup(x=300, y=150):
    return Detection("cup", "Culture Vessel Stand-in", 0.9, (x, y, x + 60, y + 80))


def bottle(x=100, y=150):
    return Detection("bottle", "Sample Container Stand-in", 0.9, (x, y, x + 50, y + 100))


def tablet(x=300, y=150):
    return Detection("cell phone", "Data Tablet Stand-in", 0.9, (x, y, x + 60, y + 100))


def sharp(x=200, y=150):
    return Detection("scissors", "Sharp Tool Stand-in", 0.8, (x, y, x + 60, y + 80))


def touch(objs, hand_xy, n):
    return [(objs, [make_hand(*hand_xy)])] * n


# ------------------------------------------------------------------ interaction + temporal logic
def test_interaction_levels(tmp_path):
    cfg = base_cfg(tmp_path)
    ana = InteractionAnalyzer(cfg)
    tr = SimpleTracker().update([cup()])
    assert ana.analyze([make_hand(330, 190)], tr)[0].level == Proximity.CONTACT
    assert ana.analyze([make_hand(420, 200)], tr)[0].level == Proximity.NEAR
    assert ana.analyze([make_hand(600, 330)], tr)[0].level == Proximity.FAR


def test_single_frame_contact_is_ignored(tmp_path):
    frames = touch([cup()], FAR, 5) + touch([cup()], (330, 190), 1) + touch([cup()], FAR, 10)
    sim = Sim(base_cfg(tmp_path)).run(frames)
    assert "INTERACTION_STARTED" not in sim.types()


def test_sustained_contact_confirms_interaction(tmp_path):
    cfg = base_cfg(tmp_path)
    n = cfg["activity"]["interaction_confirm_frames"]
    sim = Sim(cfg).run(touch([cup()], FAR, 3) + touch([cup()], (330, 190), n - 1))
    assert "INTERACTION_STARTED" not in sim.types()          # N-1 frames: not yet
    sim.run(touch([cup()], (330, 190), 1))
    assert "INTERACTION_STARTED" in sim.types()              # N-th frame: confirmed
    assert sim.act.objects[1].state == State.INTERACTING


def test_touch_sequence_completes(tmp_path):
    frames = (touch([cup()], FAR, 5) + touch([cup()], (420, 200), 4)
              + touch([cup()], (330, 190), 12) + touch([cup()], FAR, 12))
    sim = Sim(base_cfg(tmp_path)).run(frames)
    t = sim.types()
    assert t.index("HAND_NEAR_EQUIPMENT") < t.index("INTERACTION_STARTED") < t.index("ACTIVITY_COMPLETED")
    done = sim.of("ACTIVITY_COMPLETED")[0]
    assert done.data["action"] == "interact"
    assert done.activity == "CULTURE_VESSEL_INTERACTION"
    assert done.subject == "Culture Vessel Stand-in" and 0 <= done.stability <= 1


def test_release_requires_persistence(tmp_path):
    cfg = base_cfg(tmp_path)
    rel = cfg["activity"]["release_confirm_frames"]
    frames = (touch([cup()], FAR, 3) + touch([cup()], (330, 190), 12)
              + touch([cup()], FAR, rel - 1)                  # short gap: not a release
              + touch([cup()], (330, 190), 5))
    sim = Sim(cfg).run(frames)
    assert "ACTIVITY_COMPLETED" not in sim.types()
    assert sim.act.objects[1].state == State.INTERACTING
    sim.run(touch([cup()], FAR, rel))                         # sustained: release accepted
    assert sim.types().count("ACTIVITY_COMPLETED") == 1


def test_brief_contact_below_min_duration_is_not_an_activity(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["activity"]["min_interaction_s"] = 3.0
    frames = touch([cup()], FAR, 3) + touch([cup()], (330, 190), 6) + touch([cup()], FAR, 12)
    sim = Sim(cfg).run(frames)
    assert "INTERACTION_STARTED" in sim.types()
    assert "BRIEF_CONTACT_IGNORED" in sim.types()
    assert "ACTIVITY_COMPLETED" not in sim.types()


def test_interaction_cooldown_after_completion(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["activity"]["interaction_cooldown_s"] = 2.0
    first = touch([cup()], FAR, 3) + touch([cup()], (330, 190), 10) + touch([cup()], FAR, 8)
    sim = Sim(cfg).run(first)
    assert sim.types().count("INTERACTION_STARTED") == 1
    sim.run(touch([cup()], (330, 190), 10))                   # immediately again: inside cooldown
    assert sim.types().count("INTERACTION_STARTED") == 1
    sim.run(touch([cup()], FAR, 30) + touch([cup()], (330, 190), 10))   # after cooldown: allowed
    assert sim.types().count("INTERACTION_STARTED") == 2


def test_move_sequence_and_workflow(tmp_path):
    frames = [([bottle(), cup()], [make_hand(*FAR)])] * 5
    frames += touch([bottle(), cup()], (125, 190), 10) + touch([bottle(), cup()], FAR, 12)   # S1 touch bottle
    for k in range(20):                                                                      # S2 move cup
        x = 300 - k * 6
        frames.append(([bottle(), cup(x)], [make_hand(x + 30, 190)]))
    frames += touch([bottle(), cup(180)], FAR, 12)
    sim = Sim(base_cfg(tmp_path)).run(frames)
    assert "OBJECT_MOVED" in sim.types()
    steps = sim.wf.as_dict()["steps"]
    assert [s["status"] for s in steps] == ["done", "done", "active"]
    assert "OUT_OF_SEQUENCE" not in sim.types()


def test_workflow_completes_and_resets(tmp_path):
    cfg = base_cfg(tmp_path)
    sim = Sim(cfg)
    wf = sim.wf
    from app.activity import CompletedActivity
    wf.on_activity(CompletedActivity(1, "Sample Container Stand-in", "interact", 1, 0, "R"))
    wf.on_activity(CompletedActivity(2, "Culture Vessel Stand-in", "interact", 1, 0, "R"))   # not a move
    assert wf.index == 1                                             # S2 requires a MOVE
    wf.on_activity(CompletedActivity(2, "Culture Vessel Stand-in", "move", 1, 0.2, "R"))
    wf.on_activity(CompletedActivity(3, "Data Tablet Stand-in", "interact", 1, 0, "R"))
    assert wf.completed and "WORKFLOW_COMPLETED" in sim.types()
    wf.reset()
    assert wf.index == 0 and not wf.completed and wf.steps[0].status == "active"


def test_out_of_sequence(tmp_path):
    frames = touch([tablet()], FAR, 3) + touch([tablet()], (330, 200), 10) + touch([tablet()], FAR, 12)
    sim = Sim(base_cfg(tmp_path)).run(frames)
    assert "OUT_OF_SEQUENCE" in sim.types()
    assert sim.of("OUT_OF_SEQUENCE")[0].severity == "warning"
    assert sim.wf.as_dict()["deviations"] == 1


# ------------------------------------------------------------------ perception drop-outs
def test_hand_tracking_loss_holds_state_then_ends_without_completion(tmp_path):
    cfg = base_cfg(tmp_path)
    grace_frames = int(cfg["activity"]["hand_lost_grace_s"] * FPS)
    sim = Sim(cfg).run(touch([cup()], FAR, 3) + touch([cup()], (330, 190), 10))
    assert sim.act.objects[1].state == State.INTERACTING
    sim.run([([cup()], [])] * (grace_frames - 3))               # brief loss: state is held
    assert sim.act.objects[1].state == State.INTERACTING
    assert sim.scene()["hand_tracking_lost"] is True
    assert "HAND_TRACKING_LOST" in sim.types()
    sim.run(touch([cup()], (330, 190), 3))                      # hand re-acquired: continues
    assert sim.act.objects[1].state == State.INTERACTING and not sim.scene()["hand_tracking_lost"]
    sim.run([([cup()], [])] * (grace_frames + 5))               # long loss: ends, NOT completed
    assert "INTERACTION_ENDED_TRACKING_LOST" in sim.types()
    assert "ACTIVITY_COMPLETED" not in sim.types()
    assert sim.wf.index == 0


def test_object_track_loss_is_not_counted_as_completed(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["tracker"]["max_missed"] = 5
    frames = touch([bottle()], FAR, 3) + touch([bottle()], (125, 190), 10) + touch([], (125, 190), 10)
    sim = Sim(cfg).run(frames)
    assert "OBJECT_LOST_DURING_INTERACTION" in sim.types()
    assert "ACTIVITY_COMPLETED" not in sim.types()
    assert sim.wf.index == 0


def test_object_temporarily_lost_is_reported(tmp_path):
    sim = Sim(base_cfg(tmp_path)).run(touch([cup()], FAR, 5) + touch([], FAR, 3))
    assert sim.scene()["objects_temporarily_lost"] == ["Culture Vessel Stand-in"]


# ------------------------------------------------------------------ safety engine
def test_restricted_zone_warning_then_critical(tmp_path):
    cfg = base_cfg(tmp_path)
    n = cfg["safety"]["restricted_confirm_frames"]
    sim = Sim(cfg).run([([], [make_hand(*IN_ZONE)])] * (n - 1))
    assert "RESTRICTED_ZONE_ENTRY" not in sim.types()              # debounced
    sim.run([([], [make_hand(*IN_ZONE)])])
    entry = sim.of("RESTRICTED_ZONE_ENTRY")
    assert len(entry) == 1 and entry[0].severity == "warning"
    assert sim.scene()["state"] == "WARNING"
    sim.run([([], [make_hand(*IN_ZONE)])] * int(cfg["safety"]["critical_after_s"] * FPS + 2))
    crit = sim.of("RESTRICTED_ZONE_CRITICAL")
    assert len(crit) == 1 and crit[0].severity == "critical"
    assert sim.scene()["state"] == "CRITICAL"
    assert len(sim.of("RESTRICTED_ZONE_ENTRY")) == 1               # not re-raised every frame


def test_restricted_zone_exit_is_debounced(tmp_path):
    cfg = base_cfg(tmp_path)
    exit_frames = int(cfg["safety"]["zone_exit_s"] * FPS)
    sim = Sim(cfg).run([([], [make_hand(*IN_ZONE)])] * 5)
    sim.run([([], [])] * (exit_frames - 3))                         # hand not tracked briefly
    assert "RESTRICTED_ZONE_EXIT" not in sim.types() and sim.saf.zones[0].active
    sim.run([([], [make_hand(*IN_ZONE)])] * 2)                      # back in: no second entry alert
    assert len(sim.of("RESTRICTED_ZONE_ENTRY")) == 1
    sim.run([([], [make_hand(*FAR)])] * (exit_frames + 2))
    assert "RESTRICTED_ZONE_EXIT" in sim.types() and not sim.saf.zones[0].active


def test_disabled_zone_does_not_alert(tmp_path):
    zone = dict(RESTRICTED, enabled=False)
    sim = Sim(base_cfg(tmp_path, [zone])).run([([], [make_hand(*IN_ZONE)])] * 20)
    assert "RESTRICTED_ZONE_ENTRY" not in sim.types()


def test_alert_cooldown(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["safety"]["zone_exit_s"] = 0.2
    cooldown_frames = int(cfg["safety"]["alert_cooldown_s"] * FPS)
    enter_leave = [([], [make_hand(*IN_ZONE)])] * 5 + [([], [make_hand(*FAR)])] * 6
    sim = Sim(cfg).run(enter_leave * 3)                             # 3 entries within the cooldown
    assert len(sim.of("RESTRICTED_ZONE_ENTRY")) == 1
    assert sim.types().count("RESTRICTED_ZONE_EXIT") == 3          # the visual/zone state still updates
    sim.run([([], [make_hand(*FAR)])] * cooldown_frames + enter_leave)
    assert len(sim.of("RESTRICTED_ZONE_ENTRY")) == 2


def test_workstation_zone_approach(tmp_path):
    sim = Sim(base_cfg(tmp_path, [RESTRICTED, WORKSTATION]))
    sim.run([([], [make_hand(*FAR)])] * 6)
    assert sim.scene()["activity_state"] == "PERSON_DETECTED"
    sim.run([([], [make_hand(100, 250)])] * 4)
    assert "APPROACHING_WORKSTATION" in sim.types()
    assert sim.scene()["activity_state"] == "APPROACHING"


def test_restricted_object_alert(tmp_path):
    frames = touch([sharp()], FAR, 3) + touch([sharp()], (230, 190), 8)
    sim = Sim(base_cfg(tmp_path)).run(frames)
    alerts = sim.of("RESTRICTED_OBJECT")
    assert len(alerts) == 1 and alerts[0].severity == "critical"


def test_unexpected_movement(tmp_path):
    frames = [([bottle(), cup()], [make_hand(*FAR)])] * 25
    for k in range(25):                    # cup moves ~8 px/frame on its own, bottle stays, hand visible but far
        frames.append(([bottle(), cup(300 - k * 8, 150)], [make_hand(*FAR)]))
    sim = Sim(base_cfg(tmp_path)).run(frames)
    assert "UNEXPECTED_MOVEMENT" in sim.types()
    assert "CAMERA_SHIFT" not in sim.types()


def test_no_unexpected_movement_when_hand_tracking_is_lost(tmp_path):
    frames = [([bottle(), cup()], [make_hand(*FAR)])] * 10
    for k in range(25):   # a person is present but hand landmarks are lost while the cup moves
        frames.append(([bottle(), cup(300 - k * 5, 150),
                        Detection("person", "Person", 0.9, (0, 0, 640, 360))], []))
    sim = Sim(base_cfg(tmp_path)).run(frames)
    assert "UNEXPECTED_MOVEMENT" not in sim.types()


def test_missing_object_warning(tmp_path):
    cfg = base_cfg(tmp_path)
    cfg["tracker"]["max_missed"] = 3
    stable = cfg["safety"]["stable_object_frames"]
    frames = [([cup()], [make_hand(*FAR)])] * (stable + 2)
    frames += [([], [make_hand(*FAR)])] * int((cfg["safety"]["missing_object_s"] + 1) * FPS)
    sim = Sim(cfg).run(frames)
    miss = sim.of("OBJECT_MISSING")
    assert len(miss) == 1 and miss[0].severity == "warning"


# ------------------------------------------------------------------ scene state + stability
def test_scene_state_progression(tmp_path):
    sim = Sim(base_cfg(tmp_path, [RESTRICTED, WORKSTATION]))
    assert sim.scene()["state"] == "IDLE"
    seen = []
    for frames in ([([cup()], [make_hand(*FAR)])] * 6,            # person (hand) visible
                   touch([cup()], (420, 200), 4),                  # hand near cup
                   touch([cup()], (330, 190), 6)):                 # contact
        sim.run(frames)
        seen.append(sim.scene()["state"])
    assert seen == ["PERSON_DETECTED", "HAND_NEAR_EQUIPMENT", "INTERACTING"]


def test_stability_is_temporal_consistency(tmp_path):
    sim = Sim(base_cfg(tmp_path)).run(touch([cup()], FAR, 3) + touch([cup()], (330, 190), 20))
    assert sim.scene()["stability"] == 1.0                         # every recent frame agreed
    sim.run(touch([cup()], FAR, 3))                                 # 3 disagreeing frames (not yet released)
    s = sim.scene()["stability"]
    assert 0.5 < s < 1.0
    assert sim.scene()["detection_confidence"] == 0.9              # reported separately


# ------------------------------------------------------------------ tracker
def test_tracker_keeps_identity_on_fast_motion():
    tr = SimpleTracker(min_hits=1, center_match=0.75)
    tr.update([cup(300)])
    tracks = tr.update([cup(340)])       # moved 40 px on a 60 px box: IoU ~0.2 < 0.3
    assert len(tracks) == 1 and tracks[0].track_id == 1
    tracks = tr.update([cup(600)])       # far jump: a different object
    assert {t.track_id for t in tracks} == {1, 2}


def test_single_frame_false_detection_is_not_trusted():
    tr = SimpleTracker(min_hits=3)
    assert tr.update([cup()]) == []
    assert tr.update([cup()]) == []
    assert len(tr.update([cup()])) == 1


# ------------------------------------------------------------------ events
def test_event_fields_and_exports(tmp_path):
    cfg = base_cfg(tmp_path)
    ev = EventLog(cfg)
    ev.emit("RESTRICTED_ZONE_ENTRY", "warning", "Hand entered zone", activity="RESTRICTED_ZONE_ENTRY",
            state="WARNING", subject="RESTRICTED ZONE", stability=0.8123)
    csv_text = ev.export_csv()
    header = csv_text.splitlines()[0].split(",")
    for col in ("timestamp", "event_type", "activity", "state", "severity", "subject", "stability", "description"):
        assert col in header
    row = ev.export_json()[0]
    assert row["event_type"] == "RESTRICTED_ZONE_ENTRY" and row["stability"] == 0.812
    assert row["subject"] == "RESTRICTED ZONE" and row["timestamp"]
    assert Path(ev.file).read_text(encoding="utf-8").count("\n") == 1   # persisted as JSON lines
    with pytest.raises(ValueError):
        ev.emit("X", "fatal", "bad severity")


# ------------------------------------------------------------------ voice
def test_voice_phrases_and_cooldown(tmp_path):
    cfg = base_cfg(tmp_path)
    v = VoiceAlerter(cfg, start_thread=False)
    ev = EventLog(cfg)
    e1 = ev.emit("RESTRICTED_ZONE_ENTRY", "warning", "x")
    e2 = ev.emit("RESTRICTED_ZONE_CRITICAL", "critical", "x")
    e3 = ev.emit("UNEXPECTED_MOVEMENT", "warning", "x")
    e4 = ev.emit("INTERACTION_STARTED", "info", "x")
    assert v.phrase_for(e1) == "Warning. Restricted zone interaction detected."
    assert v.phrase_for(e2) == "Critical alert. Unsafe interaction detected."
    assert v.phrase_for(e3) == "Warning. Unexpected activity detected."
    assert v.phrase_for(e4) is None
    assert v.say("hello", now=100.0) is True
    assert v.say("hello", now=101.0) is False                       # cooldown
    assert v.say("hello", now=100.0 + cfg["voice"]["cooldown_s"] + 0.1) is True
    v.enabled = False
    assert v.say("other", now=500.0) is False                       # muted


# ------------------------------------------------------------------ configuration
def test_shipped_config_is_valid_without_warnings():
    cfg = load_config(use_env=False, use_zone_file=False)
    assert cfg["_warnings"] == []


@pytest.mark.parametrize("section,key,value", [
    ("activity", "interaction_confirm_frames", 1),      # single-frame confirmation is not allowed
    ("activity", "release_confirm_frames", "8"),        # wrong type
    ("safety", "alert_cooldown_s", -1),
    ("video", "mode", "simulation"),                    # no fake/simulated mode exists
    ("detector", "conf", 1.5),
    ("voice", "enabled", "yes"),
])
def test_config_rejects_bad_values(section, key, value):
    cfg = load_config(use_env=False, use_zone_file=False)
    bad = copy.deepcopy(cfg)
    bad[section][key] = value
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_config_rejects_bad_zone():
    cfg = load_config(use_env=False, use_zone_file=False)
    bad = copy.deepcopy(cfg)
    bad["safety"]["zones"] = [{"name": "z", "points": [[0, 0], [2, 0], [1, 1]]}]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_unknown_config_key_is_reported():
    cfg = load_config(use_env=False, use_zone_file=False)
    cfg["activity"]["typo_frames"] = 3
    assert any("typo_frames" in w for w in validate_config(cfg))


def test_every_config_key_is_read_by_the_runtime():
    """Guards against dead configuration: each schema key must appear in the app source."""
    root = Path(__file__).resolve().parent.parent
    src = "\n".join(p.read_text(encoding="utf-8") for p in (root / "app").glob("*.py") if p.name != "config.py")
    ui = root.parent / "frontend" / "src"   # the "app" section is consumed by the dashboard
    ui_src = "\n".join(p.read_text(encoding="utf-8") for p in ui.rglob("*.ts*")) if ui.exists() else ""
    missing = [f"{s}.{k}" for s, keys in SCHEMA.items() for k in keys
               if not re.search(rf"['\"]{re.escape(k)}['\"]", src)
               and not (s == "app" and re.search(rf"app\??\.{k}\b", ui_src))]
    assert missing == []


# ------------------------------------------------------------------ video sources (no fake fallback)
def test_unavailable_camera_reports_error_instead_of_fake_frames():
    import time as _t

    from app.pipeline import VideoSource
    src = VideoSource("webcam", 9, open_timeout=1.0)
    deadline = _t.time() + 15
    while src.state == "connecting" and _t.time() < deadline:
        _t.sleep(0.1)
    assert src.state == "error" and "Camera 9 unavailable" in src.error
    assert src.read() == (None, 0)                                  # no frames are ever invented
    src.stop()
    missing = VideoSource("file", "data/videos/does_not_exist.mp4")
    missing._thread.join(3)
    assert missing.state == "error" and "not found" in missing.error
