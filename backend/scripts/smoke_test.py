"""Server smoke test: starts the real backend, checks the HTTP API, the MJPEG stream and the
exports, then stops it.

    python backend/scripts/smoke_test.py                      # uses the sample video (file mode)
    python backend/scripts/smoke_test.py --camera 0           # uses the real webcam
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers, r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    env = dict(os.environ)
    if args.camera is not None:
        env["BAS_CAMERA"] = str(args.camera)
        env.pop("BAS_VIDEO", None)
    else:
        env["BAS_VIDEO"] = "data/videos/hand_demo.mp4"
    import tempfile
    env["BAS_ZONES_FILE"] = str(Path(tempfile.mkdtemp()) / "zones.json")   # keep the operator's zones untouched
    base = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                             "--port", str(args.port), "--log-level", "warning"], cwd=BACKEND, env=env)
    failures = []

    def ok(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    try:
        t0 = time.time()
        health = None
        while time.time() - t0 < 120:
            try:
                health = json.loads(get(f"{base}/api/health")[2])
                if health["status"]["frames_processed"] > 20:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        ok("backend ready", health is not None, f"after {time.time() - t0:.0f}s")
        if health is None:
            return 1
        for name, c in health["components"].items():
            print(f"         {name:10s} {c['state'].upper():10s} {c['detail']}")
        ok("health ok (camera + YOLO + MediaPipe ready)", health["ok"])
        st = json.loads(get(f"{base}/api/state")[2])
        ok("frames processed", st["status"]["frames_processed"] > 0, f"{st['status']['frames_processed']} @ {st['status']['fps']} FPS")
        ok("person detected by YOLO", any(o["label"] == "Person" for o in st["objects"]))
        ok("hand detected by MediaPipe", bool(st["hands"]))

        req = urllib.request.urlopen(f"{base}/video_feed", timeout=10)
        chunk = req.read(4096)
        req.close()
        ok("MJPEG stream delivers JPEG frames", b"--frame" in chunk and b"\xff\xd8" in chunk,
           req.headers.get("Content-Type", ""))
        code, hdr, body = get(f"{base}/api/events/export?format=csv")
        ok("CSV export", code == 200 and body.startswith(b"id,timestamp"), hdr.get("Content-Disposition", ""))
        code, hdr, body = get(f"{base}/api/events/export?format=json")
        ok("JSON export", code == 200 and json.loads(body)[0]["event_type"] == "SOURCE_SELECTED")
        code, _, body = get(f"{base}/")
        ok("dashboard served", code == 200 and b"<div id=\"root\">" in body)
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"\nRESULT: {'ALL PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
