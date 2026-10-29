"""Crime Weapon Detection & Tracking — FastAPI service around the best.pt model.

One model, one class ("Real Gun"). Still images are a plain predict; video and
webcam frames run through ByteTrack so each weapon keeps a stable #id while it
stays in view.
"""

import asyncio
import base64
import binascii
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"
STATIC_DIR = BASE_DIR / "static"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DEFAULT_CONF = 0.35
INFER_SIZE = 640
# ByteTrack settings live in a YAML next to this file so the buffer and the
# match thresholds can be tuned without touching code.
TRACKER_CFG = BASE_DIR / "bytetrack.yaml"

DETECTION_COLOR = "#ef4444"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available(): 
        return "mps"
    return "cpu"


DEVICE = pick_device()

model = YOLO(str(MODEL_PATH))
CLASS_NAMES = model.names

with TRACKER_CFG.open() as handle:
    TRACKER_SETTINGS = yaml.safe_load(handle)


def run_model(frame, conf: float, track: bool):
    """Infer on a BGR frame. Returns (detections, inference_ms)."""
    started = time.perf_counter()
    if track:
        result = model.track(
            frame, conf=conf, imgsz=INFER_SIZE, device=DEVICE,
            persist=True, tracker=str(TRACKER_CFG), verbose=False,
        )[0]
    else:
        result = model.predict(
            frame, conf=conf, imgsz=INFER_SIZE, device=DEVICE, verbose=False
        )[0]
    elapsed_ms = (time.perf_counter() - started) * 1000

    detections = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class_id": cls_id,
                "label": CLASS_NAMES.get(cls_id, str(cls_id)),
                "track_id": None if box.id is None else int(box.id[0]),
                "confidence": round(float(box.conf[0]), 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "color": DETECTION_COLOR,
            }
        )
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, round(elapsed_ms, 1)


def reset_tracker():
    """Drop ByteTrack state so a new clip starts numbering from #1 again."""
    model.predictor = None


def decode_image(data: bytes):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode image data")
    return frame


@asynccontextmanager
async def lifespan(_: FastAPI):
    """First inference pays the lazy-init cost; spend it before anyone connects."""
    blank = np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)
    for _ in range(2):
        run_model(blank, DEFAULT_CONF, track=False)
    print(f"[weapon-detect] model ready on '{DEVICE}' — classes: {list(CLASS_NAMES.values())}")
    print(f"[weapon-detect] bytetrack {TRACKER_CFG.name}: track_buffer={TRACKER_SETTINGS['track_buffer']} frames")
    yield


app = FastAPI(title="Crime Weapon Detection & Tracking", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/meta")
def meta():
    return {
        "model": MODEL_PATH.name,
        "device": DEVICE,
        "classes": {str(i): name for i, name in CLASS_NAMES.items()},
        "color": DETECTION_COLOR,
        "default_conf": DEFAULT_CONF,
        "tracker": TRACKER_SETTINGS,
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=0.95),
):
    data = await file.read()
    if not data:
        return JSONResponse({"error": "empty upload"}, status_code=400)
    if len(data) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": "file too large (50 MB max)"}, status_code=413)

    try:
        frame = decode_image(data)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    detections, inference_ms = await asyncio.to_thread(run_model, frame, conf, False)
    height, width = frame.shape[:2]
    return {
        "source": "image",
        "filename": file.filename,
        "width": width,
        "height": height,
        "inference_ms": inference_ms,
        "count": len(detections),
        "active_tracks": 0,
        "detections": detections,
    }


@app.websocket("/ws/detect")
async def detect_stream(websocket: WebSocket):
    """Tracked, frame-by-frame detection for webcam and uploaded-video playback.

    The client sends {"frame": "<base64 jpeg>", "conf": 0.35, "seq": n} and gets
    one reply per frame, so it throttles itself to the model's real pace.
    A {"reset": true} message clears tracker state between sources.
    """
    await websocket.accept()
    await asyncio.to_thread(reset_tracker)
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
            except ValueError:
                await websocket.send_json({"error": "malformed message"})
                continue

            if payload.get("reset"):
                await asyncio.to_thread(reset_tracker)
                await websocket.send_json({"reset": True})
                continue

            try:
                raw = payload["frame"]
                if "," in raw:  # strip a data: URL prefix if present
                    raw = raw.split(",", 1)[1]
                frame = decode_image(base64.b64decode(raw))
            except (ValueError, KeyError, TypeError, binascii.Error) as exc:
                await websocket.send_json({"error": f"bad frame: {exc}"})
                continue

            conf = min(max(float(payload.get("conf", DEFAULT_CONF)), 0.01), 0.95)
            detections, inference_ms = await asyncio.to_thread(run_model, frame, conf, True)
            height, width = frame.shape[:2]
            await websocket.send_json(
                {
                    "source": payload.get("source", "stream"),
                    "seq": payload.get("seq"),
                    "width": width,
                    "height": height,
                    "inference_ms": inference_ms,
                    "count": len(detections),
                    "active_tracks": len({d["track_id"] for d in detections if d["track_id"]}),
                    "detections": detections,
                }
            )
    except WebSocketDisconnect:
        pass


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
