"""End-to-end API test: real YOLO + real MediaPipe on the bundled sample video (file mode).

Skipped automatically when the model files are not present (run setup.bat first).
This exercises the real pipeline and API; it does NOT test a physical webcam - that must
be checked manually on the demo laptop (docs/validation-checklist.md).
"""
import importlib
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_OK = (ROOT / "models" / "yolo11n.pt").exists() and (ROOT / "models" / "hand_landmarker.task").exists()
VIDEO = ROOT / "data" / "videos" / "hand_demo.mp4"
pytestmark = pytest.mark.skipif(not (MODELS_OK and VIDEO.exists()),
                                reason="model files or sample video missing - run setup.bat")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["BAS_VIDEO"] = "data/videos/hand_demo.mp4"
    # never touch the operator's saved zones (data/zones.json) from tests
    os.environ["BAS_ZONES_FILE"] = str(tmp_path_factory.mktemp("zones") / "zones.json")
    os.environ.pop("BAS_CAMERA", None)
    from fastapi.testclient import TestClient

    import app.main as main
    main = importlib.reload(main)
    main.CFG["voice"]["enabled"] = False
    main.CFG["logging"]["dir"] = str(tmp_path_factory.mktemp("logs"))
    with TestClient(main.app) as c:
        deadline = time.time() + 60
        while time.time() < deadline:
            if c.get("/api/state").json()["status"]["frames_processed"] > 20:
                break
            time.sleep(0.5)
        yield c
    os.environ.pop("BAS_VIDEO", None)
    os.environ.pop("BAS_ZONES_FILE", None)


def test_health_reports_real_components(client):
    h = client.get("/api/health").json()
    comps = h["components"]
    assert comps["camera"]["state"] == "connected"
    assert comps["yolo"]["state"] in ("ready", "active")
    assert comps["mediapipe"]["state"] in ("ready", "active")
    assert comps["activity"]["state"] == "active"
    assert h["status"]["mode"] == "file" and h["status"]["fps"] > 0


def test_state_contains_real_perception(client):
    s = client.get("/api/state").json()
    assert any(o["label"] == "Person" for o in s["objects"])      # YOLO saw the person in the clip
    assert s["hands"], "MediaPipe should detect the hand in the sample clip"
    assert s["activity"]["state"] in ("PERSON_DETECTED", "APPROACHING", "WARNING", "CRITICAL",
                                      "HAND_NEAR_EQUIPMENT", "INTERACTING", "MANIPULATING", "COMPLETED")
    assert s["workflow"]["name"] == "Representative Demo Workflow"


def test_annotated_frames_and_snapshot(client):
    # The infinite MJPEG stream itself is checked with a real server in scripts/smoke_test.py
    # (TestClient cannot close an endless streaming response).
    import app.main as main
    jpg = main.PIPE.wait_jpeg(2.0)
    assert jpg and jpg[:2] == b"\xff\xd8"
    assert client.get("/api/snapshot.jpg").content[:2] == b"\xff\xd8"


def test_websocket_pushes_state(client):
    with client.websocket_connect("/ws/state") as ws:
        assert ws.receive_json()["kind"] == "history"
        msg = ws.receive_json()
        assert msg["kind"] == "state" and "status" in msg["state"]


def test_zone_update_triggers_restricted_alert(client):
    zones = [{"id": "z1", "name": "RESTRICTED ZONE", "type": "restricted", "enabled": True,
              "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}]   # whole frame
    assert client.put("/api/zones", json=zones).status_code == 200
    deadline = time.time() + 20
    types = []
    while time.time() < deadline:
        types = [e["type"] for e in client.get("/api/events?limit=500").json()]
        if "RESTRICTED_ZONE_ENTRY" in types:
            break
        time.sleep(0.5)
    assert "RESTRICTED_ZONE_ENTRY" in types
    bad = [{"id": "z", "name": "bad", "points": [[0, 0], [5, 5], [1, 0]]}]
    assert client.put("/api/zones", json=bad).status_code == 422
    client.put("/api/zones", json=[])


def test_exports(client):
    csv_text = client.get("/api/events/export?format=csv").text
    assert csv_text.startswith("id,timestamp,time,event_type,activity,state,severity,subject,stability,description")
    data = client.get("/api/events/export?format=json").json()
    assert data and {"timestamp", "event_type", "severity", "description"} <= set(data[0])
    assert client.get("/api/events/export?format=xml").status_code == 400


def test_bad_sources_are_rejected_or_reported(client):
    assert client.post("/api/source", json={"kind": "file", "value": "data/videos/nope.mp4"}).status_code == 404
    assert client.post("/api/source", json={"kind": "sim", "value": "0"}).status_code == 400
    # restore
    client.post("/api/source", json={"kind": "file", "value": "data/videos/hand_demo.mp4"})
