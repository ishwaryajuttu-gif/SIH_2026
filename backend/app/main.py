"""FastAPI backend - bridge between the AI pipeline and the React dashboard.

Run (from backend/):  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
The server binds to localhost by default: webcam frames never leave this computer.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import BACKEND_DIR, ConfigError, load_config, resolve_path, save_zones
from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("bas.api")

CFG = load_config()
PIPE: Pipeline | None = None
VIDEO_DIR = resolve_path("data/videos")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRONTEND_DIST = (BACKEND_DIR.parent / "frontend" / "dist").resolve()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global PIPE
    PIPE = Pipeline(CFG)
    PIPE.start()
    # camera discovery runs in the background so start-up is not delayed
    asyncio.get_running_loop().run_in_executor(None, PIPE.rescan_cameras)
    log.info("Dashboard: http://localhost:8000  (API docs: /docs)")
    yield
    PIPE.stop()


app = FastAPI(title="BAS-HAR prototype API", version="0.2.0", lifespan=lifespan)
app.add_middleware(  # only the local React dev server needs cross-origin access
    CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


def pipe() -> Pipeline:
    if PIPE is None:
        raise HTTPException(503, "Pipeline not ready")
    return PIPE


# ------------------------------------------------------------------ live data
@app.get("/api/health")
def health():
    p = pipe()
    h = p.health()
    ok = h["camera"]["state"] == "connected" and h["yolo"]["state"] in ("ready", "active") \
        and h["mediapipe"]["state"] in ("ready", "active")
    return {"ok": ok, "components": h, "status": p.status()}


@app.get("/api/state")
def state():
    return pipe().snapshot()


@app.websocket("/ws/state")
async def ws_state(ws: WebSocket):
    """Pushes the full state ~8x per second, plus every new event exactly once."""
    await ws.accept()
    p = pipe()
    history = p.events.recent(150)
    last_event = history[-1].id if history else 0
    try:
        await ws.send_json({"kind": "history", "events": [asdict(e) for e in history]})
        while True:
            new = p.events.since(last_event)
            if new:
                last_event = new[-1].id
            await ws.send_json({"kind": "state", "state": p.snapshot(), "events": [asdict(e) for e in new]})
            await asyncio.sleep(0.125)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.get("/video_feed")
async def video_feed():
    """MJPEG stream of the annotated video - use directly as <img src>."""
    p = pipe()

    async def gen():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        last = None
        while True:
            jpg = await asyncio.to_thread(p.wait_jpeg, 1.0)
            if jpg is None:
                await asyncio.sleep(0.05)
                continue
            if jpg is last:  # paused / no new frame: re-send at most once per second to keep the stream alive
                await asyncio.sleep(0.2)
            last = jpg
            yield boundary + jpg + b"\r\n"

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/snapshot.jpg")
def snapshot_jpg():
    jpg = pipe().latest_jpeg
    if not jpg:
        raise HTTPException(404, "No frame yet")
    return StreamingResponse(iter([jpg]), media_type="image/jpeg")


# ------------------------------------------------------------------ events / alerts
@app.get("/api/events")
def events(limit: int = 200):
    return [asdict(e) for e in pipe().events.recent(limit)]


@app.get("/api/events/export")
def export_events(format: str = "csv"):
    p = pipe()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if format == "json":
        return JSONResponse(p.events.export_json(),
                            headers={"Content-Disposition": f'attachment; filename="bas_events_{stamp}.json"'})
    if format != "csv":
        raise HTTPException(400, "format must be csv or json")
    return PlainTextResponse(p.events.export_csv(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="bas_events_{stamp}.csv"'})


@app.post("/api/alerts/ack")
def ack_all():
    return {"acknowledged": pipe().events.acknowledge(None)}


@app.post("/api/alerts/{event_id}/ack")
def ack(event_id: int):
    return {"acknowledged": pipe().events.acknowledge(event_id)}


# ------------------------------------------------------------------ zones / workflow
class Zone(BaseModel):
    id: str
    name: str
    type: str = "restricted"
    enabled: bool = True
    points: list[list[float]]


@app.get("/api/zones")
def get_zones():
    return pipe().safety.zones_as_dicts()


@app.put("/api/zones")
def put_zones(zones: list[Zone]):
    p = pipe()
    try:
        data = save_zones(CFG, [z.model_dump() for z in zones])
    except ConfigError as e:
        raise HTTPException(422, str(e)) from e
    p.set_zones(data)
    p.events.emit("ZONES_UPDATED", "info", f"Zones updated ({len(data)} zone(s))", activity="ZONES_UPDATED")
    return data


@app.get("/api/workflow")
def get_workflow():
    return pipe().workflow.as_dict()


@app.post("/api/workflow/reset")
def reset_workflow():
    p = pipe()
    p.reset_workflow()
    return p.workflow.as_dict()


# ------------------------------------------------------------------ sources / cameras
@app.get("/api/cameras")
def cameras():
    return pipe().cameras


@app.post("/api/cameras/rescan")
async def rescan():
    return await asyncio.to_thread(pipe().rescan_cameras)


class SourceReq(BaseModel):
    kind: str            # webcam | file
    value: str           # camera index or file path under data/videos


@app.post("/api/source")
def set_source(req: SourceReq):
    p = pipe()
    if req.kind == "webcam":
        if not req.value.isdigit():
            raise HTTPException(400, "camera index must be a number")
        p.set_source("webcam", int(req.value))
    elif req.kind == "file":
        path = resolve_path(req.value)
        if not path.exists() or path.suffix.lower() not in VIDEO_EXTS:
            raise HTTPException(404, f"video not found: {req.value}")
        p.set_source("file", req.value)
    else:
        raise HTTPException(400, "kind must be webcam or file")
    return p.status()


@app.get("/api/videos")
def list_videos():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return [f"data/videos/{f.name}" for f in sorted(VIDEO_DIR.iterdir()) if f.suffix.lower() in VIDEO_EXTS]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload.mp4").name
    if Path(name).suffix.lower() not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported video type; use one of {sorted(VIDEO_EXTS)}")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    with open(VIDEO_DIR / name, "wb") as f:
        shutil.copyfileobj(file.file, f)
    rel = f"data/videos/{name}"
    pipe().set_source("file", rel)
    return {"source": rel}


class ControlReq(BaseModel):
    action: str  # pause | resume


@app.post("/api/control")
def control(req: ControlReq):
    p = pipe()
    if req.action not in ("pause", "resume"):
        raise HTTPException(400, "action must be pause or resume")
    p.paused = req.action == "pause"
    p.events.emit("MONITORING_PAUSED" if p.paused else "MONITORING_RESUMED", "info",
                  "Monitoring paused by operator" if p.paused else "Monitoring resumed", activity="OPERATOR")
    return p.status()


class VoiceReq(BaseModel):
    enabled: bool


@app.put("/api/voice")
def voice(req: VoiceReq):
    p = pipe()
    p.voice.enabled = req.enabled
    return {"enabled": p.voice.enabled, "available": p.voice.available, "error": p.voice.error}


@app.post("/api/voice/test")
def voice_test():
    p = pipe()
    queued = p.voice.say("Voice alert system ready.", key=f"test-{time.time()}")
    return {"queued": queued, "available": p.voice.available, "state": p.voice.state, "error": p.voice.error}


@app.get("/api/config")
def config():
    return {k: v for k, v in CFG.items() if not k.startswith("_")}


# ------------------------------------------------------------------ dashboard (built React app)
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = (FRONTEND_DIST / path).resolve()
        if f.is_file() and FRONTEND_DIST in f.parents:
            return FileResponse(f)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def no_ui():
        return {"message": "BAS-HAR API running. Frontend not built - run 'npm run build' in frontend/."}
