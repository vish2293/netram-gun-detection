/* Crime Weapon Detection & Tracking — drives three sources (image, video,
   webcam) into one FastAPI backend and paints the raw frame on the left
   canvas and the annotated frame on the right. */

const el = (id) => document.getElementById(id);

const rawCanvas = el("rawCanvas");
const outCanvas = el("outCanvas");
const rawCtx = rawCanvas.getContext("2d");
const outCtx = outCanvas.getContext("2d");
const video = el("video");

const CAPTURE_WIDTH = 640;        // frames are downscaled before upload
const capture = document.createElement("canvas");
const captureCtx = capture.getContext("2d");

// What the UI calls each model class. The backend keeps sending the model's own
// name ("Real Gun"), which is what the raw response panel shows.
const DISPLAY_LABELS = { "Real Gun": "Weapon" };
const labelFor = (name) => DISPLAY_LABELS[name] || name;

const state = {
  mode: null,
  conf: 0.35,
  socket: null,
  streaming: false,
  inFlight: false,
  seq: 0,
  frames: 0,
  detections: [],
  scale: { x: 1, y: 1 },
  mediaStream: null,
  objectUrl: null,
  rafId: null,
  fpsWindow: [],
};

/* ---------------- small helpers ---------------- */

function log(message, kind = "") {
  const box = el("log");
  const line = document.createElement("div");
  if (kind) line.className = kind;
  line.innerHTML = `<time>${new Date().toLocaleTimeString([], { hour12: false })}</time>${message}`;
  box.appendChild(line);
  box.scrollTop = box.scrollHeight;
  while (box.childElementCount > 300) box.removeChild(box.firstChild);
}

function setStatus(text, kind = "") {
  const node = el("toolbarStatus");
  node.textContent = text;
  node.className = `toolbar-status ${kind}`;
}

function setActiveButton(mode) {
  for (const [id, name] of [["webcamBtn", "webcam"], ["videoBtn", "video"], ["imageBtn", "image"]]) {
    el(id).classList.toggle("active", name === mode);
  }
}

function resizeCanvases(width, height) {
  for (const canvas of [rawCanvas, outCanvas]) {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }
}

function showStages() {
  el("rawPlaceholder").hidden = true;
  el("outPlaceholder").hidden = true;
}

function resetStats() {
  state.frames = 0;
  state.seq = 0;
  state.fpsWindow = [];
  state.detections = [];
  state.scale = { x: 1, y: 1 };
  el("statTracks").textContent = "0";
  el("statCount").textContent = "0";
  el("statFps").textContent = "0.0";
  el("statFrames").textContent = "0";
  renderDetections([]);
  el("rawJson").textContent = "{}";
}

/* ---------------- drawing ---------------- */

function drawBoxes(ctx, detections, scaleX, scaleY) {
  const base = Math.max(ctx.canvas.width, ctx.canvas.height);
  const lineWidth = Math.max(1.5, base * 0.0022);
  const fontSize = Math.max(11, base * 0.016);
  ctx.lineWidth = lineWidth;
  ctx.font = `600 ${fontSize}px -apple-system, Segoe UI, Roboto, sans-serif`;
  ctx.textBaseline = "top";

  for (const det of detections) {
    const [x1, y1, x2, y2] = det.box;
    const x = x1 * scaleX;
    const y = y1 * scaleY;
    const w = (x2 - x1) * scaleX;
    const h = (y2 - y1) * scaleY;

    ctx.strokeStyle = det.color;
    ctx.strokeRect(x, y, w, h);

    const id = det.track_id === null ? "" : ` #${det.track_id}`;
    const text = `${labelFor(det.label)}${id} ${Math.round(det.confidence * 100)}%`;
    const padX = fontSize * 0.35;
    const labelW = ctx.measureText(text).width + padX * 2;
    const labelH = fontSize * 1.5;
    const labelY = y - labelH >= 0 ? y - labelH : y;

    ctx.fillStyle = det.color;
    ctx.fillRect(x - lineWidth / 2, labelY, labelW, labelH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(text, x - lineWidth / 2 + padX, labelY + (labelH - fontSize) / 2);
  }
}

function renderDetections(detections) {
  const list = el("detectionList");
  list.innerHTML = "";
  if (!detections.length) {
    list.innerHTML = '<li class="empty">No detections in the current frame.</li>';
    return;
  }
  for (const det of detections) {
    const item = document.createElement("li");
    item.className = "det";
    item.style.borderLeftColor = det.color;
    const pct = (det.confidence * 100).toFixed(1);
    const [x1, y1, x2, y2] = det.box;
    const id = det.track_id === null ? "" : ` #${det.track_id}`;
    item.innerHTML = `
      <div class="det-head">
        <span class="det-label">${labelFor(det.label)}${id}</span>
        <span class="det-conf">${pct}%</span>
      </div>
      <div class="det-bar"><span style="width:${pct}%;background:${det.color}"></span></div>
      <div class="det-box">x1 ${x1} · y1 ${y1} · x2 ${x2} · y2 ${y2} · ${Math.round(x2 - x1)}×${Math.round(y2 - y1)} px</div>`;
    list.appendChild(item);
  }
}

function applyResult(data) {
  state.detections = data.detections;
  state.frames += 1;
  state.fpsWindow.push(performance.now());
  if (state.fpsWindow.length > 15) state.fpsWindow.shift();

  let fps = 0;
  if (state.fpsWindow.length > 1) {
    const span = (state.fpsWindow.at(-1) - state.fpsWindow[0]) / 1000;
    if (span > 0) fps = (state.fpsWindow.length - 1) / span;
  }

  el("statTracks").textContent = data.active_tracks;
  el("statCount").textContent = data.count;
  el("statFps").textContent = fps.toFixed(1);
  el("statFrames").textContent = state.frames;
  el("rawJson").textContent = JSON.stringify(data, null, 2);
  renderDetections(data.detections);
}

/* ---------------- image mode ---------------- */

async function handleImage(file) {
  stopStreaming("Idle", { quiet: true });
  state.mode = "image";
  setActiveButton("image");
  resetStats();
  setStatus(`Processing ${file.name}`);

  const bitmap = await createImageBitmap(file);
  resizeCanvases(bitmap.width, bitmap.height);
  rawCtx.drawImage(bitmap, 0, 0);
  outCtx.drawImage(bitmap, 0, 0);
  showStages();

  const body = new FormData();
  body.append("file", file);

  try {
    const response = await fetch(`/api/detect/image?conf=${state.conf}`, { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.statusText);

    applyResult(data);
    outCtx.drawImage(bitmap, 0, 0);
    drawBoxes(outCtx, data.detections, outCanvas.width / data.width, outCanvas.height / data.height);

    setStatus(`${file.name} — ${data.count} detection(s) in ${data.inference_ms} ms`, data.count ? "alert" : "");
    log(`${file.name} → ${data.count} detection(s) in ${data.inference_ms} ms`, data.count ? "hit" : "");
  } catch (err) {
    setStatus(`Detection failed: ${err.message}`, "err");
    log(`Image detection failed: ${err.message}`, "err");
  }
}

/* ---------------- streaming (video + webcam) ---------------- */

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws/detect`);
  state.socket = socket;

  socket.addEventListener("open", () => {
    log("Detection socket connected.");
    socket.send(JSON.stringify({ reset: true }));
  });
  socket.addEventListener("close", () => {
    log("Detection socket closed.");
    state.inFlight = false;
  });
  socket.addEventListener("error", () => {
    setStatus("Detection socket error", "err");
    log("Detection socket error.", "err");
  });
  socket.addEventListener("message", (event) => {
    const data = JSON.parse(event.data);
    if (data.reset) return;

    state.inFlight = false;
    if (data.error) {
      log(data.error, "err");
      return;
    }

    const hadDetections = state.detections.length;
    // Frames go up downscaled, so map the boxes back onto the full canvas.
    state.scale = { x: outCanvas.width / data.width, y: outCanvas.height / data.height };
    applyResult(data);

    if (data.count && !hadDetections) {
      log(`Weapon detected (${data.count}) at ${data.inference_ms} ms`, "hit");
    }
    setStatus(
      data.count
        ? `${state.sourceLabel} — ${data.count} weapon${data.count > 1 ? "s" : ""} tracked`
        : `Processing ${state.sourceLabel}`,
      data.count ? "alert" : ""
    );
  });

  return socket;
}

function sendFrame() {
  const socket = state.socket;
  if (!socket || socket.readyState !== WebSocket.OPEN || state.inFlight) return;
  if (!video.videoWidth) return;

  const ratio = Math.min(1, CAPTURE_WIDTH / video.videoWidth);
  capture.width = Math.round(video.videoWidth * ratio);
  capture.height = Math.round(video.videoHeight * ratio);
  captureCtx.drawImage(video, 0, 0, capture.width, capture.height);

  state.inFlight = true;
  state.seq += 1;
  socket.send(JSON.stringify({
    frame: capture.toDataURL("image/jpeg", 0.7),
    conf: state.conf,
    seq: state.seq,
    source: state.mode,
  }));
}

function renderLoop() {
  if (!state.streaming) return;

  if (video.videoWidth) {
    resizeCanvases(video.videoWidth, video.videoHeight);
    rawCtx.drawImage(video, 0, 0, rawCanvas.width, rawCanvas.height);
    outCtx.drawImage(video, 0, 0, outCanvas.width, outCanvas.height);
    drawBoxes(outCtx, state.detections, state.scale.x, state.scale.y);
  }

  sendFrame();
  state.rafId = requestAnimationFrame(renderLoop);
}

function startStreaming(label) {
  resetStats();
  state.sourceLabel = label;
  state.streaming = true;
  showStages();
  setStatus(`Processing ${label}`);
  connectSocket();
  cancelAnimationFrame(state.rafId);
  renderLoop();
}

function stopStreaming(message = "Stopped", { quiet = false } = {}) {
  const wasRunning = state.streaming || state.socket || state.mediaStream;
  state.streaming = false;
  state.inFlight = false;
  cancelAnimationFrame(state.rafId);

  video.pause();
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  if (state.socket && state.socket.readyState <= WebSocket.OPEN) state.socket.close();
  state.socket = null;

  if (!wasRunning) return;
  if (!quiet) {
    setStatus(message);
    log(message);
  }
}

async function handleVideo(file) {
  stopStreaming("Idle", { quiet: true });
  state.mode = "video";
  setActiveButton("video");

  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = state.objectUrl;
  video.loop = true;

  try {
    await video.play();
  } catch (err) {
    setStatus(`Could not play video: ${err.message}`, "err");
    log(`Video playback failed: ${err.message}`, "err");
    return;
  }

  startStreaming(file.name);
  log(`Playing ${file.name} (${video.videoWidth}×${video.videoHeight}) on loop.`);
}

async function startWebcam() {
  stopStreaming("Idle", { quiet: true });
  state.mode = "webcam";
  setActiveButton("webcam");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    state.mediaStream = stream;
    video.src = "";
    video.srcObject = stream;
    video.loop = false;
    await video.play();
    startStreaming("webcam");
    log("Webcam started.");
  } catch (err) {
    setStatus(`Webcam unavailable: ${err.message}`, "err");
    log(`Webcam error: ${err.message}`, "err");
    setActiveButton(null);
  }
}

/* ---------------- wiring ---------------- */

el("webcamBtn").addEventListener("click", startWebcam);

el("videoInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) handleVideo(file);
  event.target.value = "";
});

el("imageInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) handleImage(file);
  event.target.value = "";
});

el("logsBtn").addEventListener("click", () => {
  const panel = el("logs");
  panel.hidden = !panel.hidden;
  el("logsBtn").textContent = panel.hidden ? "Show Logs" : "Hide Logs";
  el("logsBtn").classList.toggle("active", !panel.hidden);
});

el("stopBtn").addEventListener("click", () => {
  stopStreaming("Stopped");
  state.mode = null;
  setActiveButton(null);
  setStatus("Stopped");
});

window.addEventListener("beforeunload", () => stopStreaming("", { quiet: true }));

/* ---------------- boot ---------------- */

(async function init() {
  try {
    const meta = await (await fetch("/api/meta")).json();
    state.conf = meta.default_conf;
    const classes = Object.values(meta.classes).map(labelFor).join(", ");
    log(`Model ${meta.model} ready on ${meta.device} — classes: ${classes}`);
  } catch (err) {
    log(`Could not load model metadata: ${err.message}`, "err");
  }
  setStatus("Idle — choose a source");
})();
