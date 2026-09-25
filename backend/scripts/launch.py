"""Start the backend, WAIT until it is ready, then open the dashboard.

    python backend/scripts/launch.py              # webcam from config.yaml (default 0)
    python backend/scripts/launch.py --camera 1   # a different webcam
    python backend/scripts/launch.py --video data/videos/hand_demo.mp4   # file fallback
    python backend/scripts/launch.py --no-browser

Order: start server -> poll /api/health until the pipeline is up -> open browser.
The server binds to 127.0.0.1 only (webcam frames stay on this computer).
Ctrl+C stops everything.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, help="webcam index (overrides config.yaml)")
    ap.add_argument("--video", help="video file instead of the webcam (fallback mode)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=120, help="seconds to wait for the backend")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    env = dict(os.environ)
    env.pop("BAS_CAMERA", None)
    env.pop("BAS_VIDEO", None)
    if args.video:
        env["BAS_VIDEO"] = args.video
    elif args.camera is not None:
        env["BAS_CAMERA"] = str(args.camera)

    url = f"http://127.0.0.1:{args.port}"
    print(f"[launch] starting backend on {url} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port)],
        cwd=BACKEND, env=env,
    )
    try:
        deadline = time.time() + args.timeout
        health = None
        while time.time() < deadline:
            if proc.poll() is not None:
                print(f"[launch] backend exited with code {proc.returncode} - see the messages above.")
                return proc.returncode or 1
            try:
                with urllib.request.urlopen(f"{url}/api/health", timeout=2) as r:
                    health = json.loads(r.read().decode())
                    break
            except Exception:  # noqa: BLE001 - not up yet
                time.sleep(1)
        if health is None:
            print("[launch] backend did not become ready in time.")
            proc.terminate()
            return 1

        # give the camera a moment to deliver its first frame, then report component status
        for _ in range(10):
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as r:
                health = json.loads(r.read().decode())
            if health["components"]["camera"]["state"] != "connecting":
                break
            time.sleep(1)
        print("[launch] component status:")
        for name, c in health["components"].items():
            print(f"    {name:10s} {c['state'].upper():11s} {c['detail']}")
        if not health["ok"]:
            print("[launch] WARNING: not every component is ready - the dashboard shows the reason.")
        if not args.no_browser:
            webbrowser.open(url)
        print(f"[launch] dashboard: {url}   (press Ctrl+C here to stop)")
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[launch] stopping ...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 0


if __name__ == "__main__":
    sys.exit(main())
