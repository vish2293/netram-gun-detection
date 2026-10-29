# Crime Weapon Detection & Tracking

A single FastAPI service that serves a web UI and runs the `best.pt` YOLO model
over images, uploaded video, and a live webcam. Video and webcam frames also go
through ByteTrack, so each weapon keeps a stable `#id` while it stays in view.
No login — it's a demo.

## Run

```bash
./run.sh
```

Then open <http://127.0.0.1:8000>. The script creates `venv/` and installs
`requirements.txt` on first run. To do it by hand:

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```