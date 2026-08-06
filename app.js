/* ============================================================
   오늘의 발표자 · AI 얼굴 인식 추첨기
   - 카메라 모드: MediaPipe Tasks Vision(FaceDetector)으로 실시간 얼굴 인식
   - 번호 추첨 모드: 카메라 없이 출석번호로 추첨 (예비/대체 수단)
   ============================================================ */

import {
  FaceDetector,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20";

/* ----------------------------------------------------------
   0. 공통 상수 / 상태
   ---------------------------------------------------------- */
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";

const RESOLUTION_PRESETS = {
  close: { width: 1280, height: 720, label: "720p" },
  standard: { width: 1920, height: 1080, label: "1080p" },
  far: { width: 2560, height: 1440, label: "1440p" },
  max: { width: 3840, height: 2160, label: "4K" },
};

/* ── 트래킹·스무딩 상수 (개선) ── */
// 위치(x,y) 변화 스무딩. 0에 가까울수록 부드럽고, 1에 가까울수록 원본에 가깝습니다.
const SMOOTHING_POS = 0.50;
// 크기(w,h) 변화 스무딩. 위치보다 훨씬 강하게 걸어 박스 크기가 출렁이지 않게 합니다.
const SMOOTHING_SIZE = 0.75;
// 이전 프레임과 현재 프레임을 같은 얼굴로 인정할 최대 중심점 이동 거리(정규화 좌표)
const MATCH_THRESHOLD = 0.09;
// 일시적으로 인식이 끊겨도 같은 얼굴로 유지할 최대 시간(ms)
const MISS_TIMEOUT_MS = 5000;
// MediaPipe BlazeFace boundingBox는 얼굴 주변에 넉넉한 여백을 포함하므로,
// 중심을 유지한 채 박스를 축소해 실제 얼굴 크기에 맞춥니다.
const BOX_SCALE = 0.82;
// 너무 작은 얼굴(화면 대비 0.6% 미만)은 교실 후방 노이즈나 잔상으로 보고 무시합니다.
const MIN_FACE_SIZE_RATIO = 0.006;

const state = {
  mode: "camera",
  faceDetector: null,
  stream: null,
  running: false,
  tracks: [],
  nextTrackId: 1,
  isDrawing: false,
  coverMap: null,
};

/* ----------------------------------------------------------
   1. DOM 참조
   ---------------------------------------------------------- */
const $ = (id) => document.getElementById(id);

const video = $("video");
const overlay = $("overlay");
const overlayCtx = overlay.getContext("2d");
const cameraStatus = $("cameraStatus");
const cameraStatusText = $("cameraStatusText");
const faceCountNum = $("faceCountNum");
const resultPanel = $("resultPanel");
const resultList = $("resultList");
const deviceSelect = $("deviceSelect");
const resolutionSelect = $("resolutionSelect");
const brightnessSlider = $("brightnessSlider");
const brightnessValue = $("brightnessValue");
const pickCountInput = $("pickCount");
const drawBtn = $("drawBtn");
const resetBtn = $("resetBtn");

const manualGrid = $("manualGrid");
const manualResultPanel = $("manualResultPanel");
const manualResultList = $("manualResultList");
const totalCountInput = $("totalCount");
const pickCountManualInput = $("pickCountManual");
const drawBtnManual = $("drawBtnManual");
const resetBtnManual = $("resetBtnManual");

const confettiCanvas = $("confetti");
const shutter = $("shutter");

/* ----------------------------------------------------------
   2. 인트로 셔터 애니메이션
   ---------------------------------------------------------- */
function playShutterOpen() {
  requestAnimationFrame(() => {
    shutter.classList.add("is-open");
    setTimeout(() => shutter.classList.add("is-hidden"), 750);
  });
}

/* ----------------------------------------------------------
   3. 모드 전환
   ---------------------------------------------------------- */
function setMode(mode) {
  state.mode = mode;
  const isCamera = mode === "camera";

  $("cameraMode").classList.toggle("is-active", isCamera);
  $("manualMode").classList.toggle("is-active", !isCamera);
  $("modeCameraBtn").classList.toggle("is-active", isCamera);
  $("modeCameraBtn").setAttribute("aria-selected", String(isCamera));
  $("modeManualBtn").classList.toggle("is-active", !isCamera);
  $("modeManualBtn").setAttribute("aria-selected", String(!isCamera));

  if (isCamera && !state.stream) {
    initCamera();
  }
  if (!isCamera) {
    buildManualGrid();
  }
}

$("modeCameraBtn").addEventListener("click", () => setMode("camera"));
$("modeManualBtn").addEventListener("click", () => setMode("manual"));

/* ----------------------------------------------------------
   4. 전체 화면 토글
   ---------------------------------------------------------- */
$("fullscreenBtn").addEventListener("click", () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen?.();
  } else {
    document.exitFullscreen?.();
  }
});

/* ----------------------------------------------------------
   5. 카메라 장치 목록
   ---------------------------------------------------------- */
async function populateDeviceList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    deviceSelect.innerHTML = "";
    cams.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `카메라 ${i + 1}`;
      deviceSelect.appendChild(opt);
    });
    if (cams.length === 0) {
      const opt = document.createElement("option");
      opt.textContent = "카메라를 찾을 수 없습니다";
      deviceSelect.appendChild(opt);
    }
  } catch (err) {
    console.error("장치 목록을 가져오지 못했습니다.", err);
  }
}

deviceSelect.addEventListener("change", () => {
  startVideoStream(deviceSelect.value);
});

resolutionSelect.addEventListener("change", () => {
  startVideoStream(deviceSelect.value || undefined);
});

/* ----------------------------------------------------------
   5-1. 밝기 보정
   ---------------------------------------------------------- */
function applyBrightness(value) {
  const brightness = Number(value);
  const contrast = 1 + (brightness - 1) * 0.35;
  video.style.filter = `brightness(${brightness}) contrast(${contrast})`;
  brightnessValue.textContent = `${brightness.toFixed(2)}×`;
}

brightnessSlider.addEventListener("input", () => applyBrightness(brightnessSlider.value));
applyBrightness(brightnessSlider.value);

/* ----------------------------------------------------------
   6. 카메라 스트림 시작
   ---------------------------------------------------------- */
async function startVideoStream(deviceId) {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
  }

  const preset = RESOLUTION_PRESETS[resolutionSelect.value] || RESOLUTION_PRESETS.standard;

  const constraints = {
    audio: false,
    video: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      width: { ideal: preset.width },
      height: { ideal: preset.height },
      facingMode: deviceId ? undefined : "environment",
    },
  };

  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    video.srcObject = stream;
    await video.play();
    resizeOverlay();

    const actualW = video.videoWidth;
    const actualH = video.videoHeight;
    setCameraStatus(true, `인식 준비 완료 · ${actualW}×${actualH} (요청: ${preset.label})`);

    await populateDeviceList();
  } catch (err) {
    console.error(err);
    setCameraStatus(
      false,
      "카메라를 열 수 없습니다. 해상도를 낮추거나 권한을 확인해주세요."
    );
  }
}

function setCameraStatus(ready, text) {
  cameraStatus.classList.toggle("is-ready", ready);
  cameraStatusText.textContent = text;
}

function resizeOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const cssW = video.clientWidth || overlay.clientWidth;
  const cssH = video.clientHeight || overlay.clientHeight;

  if (!cssW || !cssH) return;

  overlay.width = Math.round(cssW * dpr);
  overlay.height = Math.round(cssH * dpr);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  confettiCanvas.width = Math.round(window.innerWidth * dpr);
  confettiCanvas.height = Math.round(window.innerHeight * dpr);
  confettiCanvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);

  if (video.videoWidth && video.videoHeight) {
    const videoAR = video.videoWidth / video.videoHeight;
    const boxAR = cssW / cssH;
    if (videoAR > boxAR) {
      const srcH = video.videoHeight;
      const srcW = srcH * boxAR;
      state.coverMap = { srcX: (video.videoWidth - srcW) / 2, srcY: 0, srcW, srcH };
    } else {
      const srcW = video.videoWidth;
      const srcH = srcW / boxAR;
      state.coverMap = { srcX: 0, srcY: (video.videoHeight - srcH) / 2, srcW, srcH };
    }
  }
}
window.addEventListener("resize", resizeOverlay);
video.addEventListener("loadedmetadata", resizeOverlay);
video.addEventListener("resize", resizeOverlay);
new ResizeObserver(() => resizeOverlay()).observe(video.closest(".viewport"));

/* ----------------------------------------------------------
   7. MediaPipe FaceDetector 초기화
   ---------------------------------------------------------- */
async function initFaceDetector() {
  setCameraStatus(false, "AI 인식 모델을 불러오는 중…");
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  try {
    state.faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.65,
    });
  } catch (err) {
    console.warn("GPU 델리게이트 실패, CPU로 재시도합니다.", err);
    state.faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: 0.65,
    });
  }
}

/* ----------------------------------------------------------
   8. 카메라 초기화 전체 흐름
   ---------------------------------------------------------- */
async function initCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraStatus(false, "이 브라우저는 카메라를 지원하지 않습니다.");
    return;
  }
  try {
    await startVideoStream(undefined);

    await Promise.race([
      initFaceDetector(),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error("모델 로딩 시간 초과")), 12000)
      ),
    ]);

    state.running = true;
    requestAnimationFrame(detectionLoop);
  } catch (err) {
    console.error(err);
    setCameraStatus(
      false,
      "AI 인식 모델을 불러오지 못했습니다. 네트워크에서 구글 도메인이 차단되어 있을 수 있습니다 — 번호 추첨 모드를 이용해주세요."
    );
  }
}

/* ----------------------------------------------------------
   9. 실시간 인식 루프 + 트래킹 스무딩
   ---------------------------------------------------------- */
function detectionLoop() {
  if (!state.running) return;

  if (video.readyState >= 2 && video.videoWidth > 0 && state.faceDetector && !state.isDrawing) {
    const now = performance.now();
    const result = state.faceDetector.detectForVideo(video, now);
    updateTracks(result.detections, now);
    drawViewfinderBoxes(state.tracks);
    faceCountNum.textContent = state.tracks.length;
    updateDrawButtonState();
  }

  requestAnimationFrame(detectionLoop);
}

/* ── 개선: boundingBox를 정규화 + 축소 + 노이즈 필터링 ── */
function toNormalizedBox(detection) {
  const bb = detection.boundingBox;
  const map = state.coverMap || {
    srcX: 0,
    srcY: 0,
    srcW: video.videoWidth,
    srcH: video.videoHeight,
  };

  let x = (bb.originX - map.srcX) / map.srcW;
  let y = (bb.originY - map.srcY) / map.srcH;
  let w = bb.width / map.srcW;
  let h = bb.height / map.srcH;

  // BlazeFace는 얼굴 주변에 여백이 많으므로 중심 유지하며 축소
  const cx = x + w / 2;
  const cy = y + h / 2;
  w *= BOX_SCALE;
  h *= BOX_SCALE;
  x = cx - w / 2;
  y = cy - h / 2;

  // 화면 대비 너무 작은 영역은 노이즈로 간주하고 무시
  if (w * h < MIN_FACE_SIZE_RATIO) return null;

  return { x, y, w, h };
}

/* ── 개선: 위치/크기 스무딩 분리, 매칭 엄격화 ── */
function updateTracks(detections, now) {
  const boxes = detections.map(toNormalizedBox).filter((b) => b !== null);
  const usedBoxIdx = new Set();

  // 기존 트랙과 매칭
  state.tracks.forEach((track) => {
    let bestIdx = -1;
    let bestDist = Infinity;
    boxes.forEach((box, i) => {
      if (usedBoxIdx.has(i)) return;
      const dist = Math.hypot(
        box.x + box.w / 2 - (track.box.x + track.box.w / 2),
        box.y + box.h / 2 - (track.box.y + track.box.h / 2)
      );
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    });

    if (bestIdx !== -1 && bestDist < MATCH_THRESHOLD) {
      const box = boxes[bestIdx];
      // 위치는 부드럽게
      track.box.x += (box.x - track.box.x) * SMOOTHING_POS;
      track.box.y += (box.y - track.box.y) * SMOOTHING_POS;
      // 크기는 더욱 부드럽게 (출렁임 방지)
      track.box.w += (box.w - track.box.w) * SMOOTHING_SIZE;
      track.box.h += (box.h - track.box.h) * SMOOTHING_SIZE;
      track.lastSeen = now;
      usedBoxIdx.add(bestIdx);
    }
  });

  // 새 얼굴 추가
  boxes.forEach((box, i) => {
    if (usedBoxIdx.has(i)) return;
    state.tracks.push({ id: state.nextTrackId++, box: { ...box }, lastSeen: now });
  });

  // 오래 사라진 트랙 제거
  state.tracks = state.tracks.filter((t) => now - t.lastSeen <= MISS_TIMEOUT_MS);
}

/* ----------------------------------------------------------
   10. 뷰파인더 스타일 박스 그리기
   ---------------------------------------------------------- */
function drawViewfinderBoxes(tracks, { activeId = null, activeColor = "#F5B942", lockedIds = null } = {}) {
  const cssW = overlay.clientWidth;
  const cssH = overlay.clientHeight;
  overlayCtx.clearRect(0, 0, cssW, cssH);

  tracks.forEach((track) => {
    const x = track.box.x * cssW;
    const y = track.box.y * cssH;
    const w = track.box.w * cssW;
    const h = track.box.h * cssH;

    let color = "#4FD1C5";
    let lineWidth = 2;

    if (lockedIds?.has(track.id)) {
      color = "#F5B942";
      lineWidth = 4;
    } else if (track.id === activeId) {
      color = activeColor;
      lineWidth = 5;
    }

    drawCornerBrackets(overlayCtx, x, y, w, h, color, lineWidth);
    drawFaceNumberBadge(overlayCtx, x, y, w, h, track.id, color);
  });
}

function drawFaceNumberBadge(ctx, x, y, w, h, id, color) {
  const label = String(id);
  const cx = x + Math.min(22, w * 0.18);
  const cy = Math.max(16, y - 14);

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(-1, 1);

  ctx.font = "700 15px 'JetBrains Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  const paddingX = 7;
  const textWidth = ctx.measureText(label).width;
  const boxW = Math.max(24, textWidth + paddingX * 2);
  const boxH = 22;

  ctx.fillStyle = color;
  ctx.fillRect(-boxW / 2, -boxH / 2, boxW, boxH);
  ctx.fillStyle = "#0B0E14";
  ctx.fillText(label, 0, 1);
  ctx.restore();
}

function drawCornerBrackets(ctx, x, y, w, h, color, lineWidth) {
  const len = Math.min(w, h) * 0.28;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = "round";

  const corners = [
    [x, y, 1, 1],
    [x + w, y, -1, 1],
    [x, y + h, 1, -1],
    [x + w, y + h, -1, -1],
  ];

  corners.forEach(([cx, cy, dx, dy]) => {
    ctx.beginPath();
    ctx.moveTo(cx, cy + len * dy);
    ctx.lineTo(cx, cy);
    ctx.lineTo(cx + len * dx, cy);
    ctx.stroke();
  });
}

function captureFrameCanvas() {
  const map = state.coverMap || {
    srcX: 0,
    srcY: 0,
    srcW: video.videoWidth,
    srcH: video.videoHeight,
  };
  const cssW = overlay.clientWidth;
  const cssH = overlay.clientHeight;
  const canvas = document.createElement("canvas");
  canvas.width = cssW;
  canvas.height = cssH;
  canvas
    .getContext("2d")
    .drawImage(video, map.srcX, map.srcY, map.srcW, map.srcH, 0, 0, cssW, cssH);
  return canvas;
}

function cropFaceThumbnail(frameCanvas, track, outSize = 160) {
  const cssW = frameCanvas.width;
  const cssH = frameCanvas.height;
  const boxW = track.box.w * cssW;
  const boxH = track.box.h * cssH;
  const cx = track.box.x * cssW + boxW / 2;
  const cy = track.box.y * cssH + boxH / 2;
  const cropSize = Math.max(boxW, boxH) * 1.8;

  const sx = Math.max(0, Math.min(cx - cropSize / 2, cssW - cropSize));
  const sy = Math.max(0, Math.min(cy - cropSize / 2, cssH - cropSize));
  const sw = Math.min(cropSize, cssW);
  const sh = Math.min(cropSize, cssH);

  const out = document.createElement("canvas");
  out.width = outSize;
  out.height = outSize;
  const outCtx = out.getContext("2d");
  outCtx.fillStyle = "#12161F";
  outCtx.fillRect(0, 0, outSize, outSize);
  outCtx.drawImage(frameCanvas, sx, sy, sw, sh, 0, 0, outSize, outSize);
  return out.toDataURL("image/png");
}

/* ----------------------------------------------------------
   11. 추첨 버튼 상태 관리
   ---------------------------------------------------------- */
function updateDrawButtonState() {
  if (state.isDrawing) return;
  const total = state.tracks.length;
  pickCountInput.max = Math.max(total, 1);
  drawBtn.disabled = total < 1 || Number(pickCountInput.value) > total;
}

pickCountInput.addEventListener("input", updateDrawButtonState);
$("countMinus").addEventListener("click", () => stepInput(pickCountInput, -1));
$("countPlus").addEventListener("click", () => stepInput(pickCountInput, 1));

function stepInput(input, delta) {
  const min = Number(input.min || 1);
  const max = input.max ? Number(input.max) : Infinity;
  input.value = Math.min(max, Math.max(min, Number(input.value) + delta));
  input.dispatchEvent(new Event("input"));
}

const SPIN_DELAYS = [70, 70, 80, 90, 100, 115, 135, 160, 190, 225, 265, 310, 360, 420];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildSpinSteps(n, targetIndex) {
  const steps = SPIN_DELAYS.length;
  const indices = [];
  for (let k = 0; k < steps; k++) {
    const stepsFromEnd = steps - 1 - k;
    const idx = ((targetIndex - stepsFromEnd) % n + n) % n;
    indices.push(idx);
  }
  return indices;
}

async function runSpin(n, targetIndex, onStep) {
  const indices = buildSpinSteps(n, targetIndex);
  for (let k = 0; k < indices.length; k++) {
    onStep(indices[k], k);
    await sleep(SPIN_DELAYS[k]);
  }
}

/* ----------------------------------------------------------
   12. 카메라 모드 추첨 시퀀스
   ---------------------------------------------------------- */
drawBtn.addEventListener("click", runCameraDraw);
resetBtn.addEventListener("click", resetCameraMode);

async function runCameraDraw() {
  const pickCount = Number(pickCountInput.value);
  const candidates = [...state.tracks].sort((a, b) => a.box.x - b.box.x);
  if (candidates.length < 1 || pickCount > candidates.length) return;

  state.isDrawing = true;
  drawBtn.disabled = true;
  drawBtn.classList.add("is-drawing");
  drawBtn.textContent = "추첨 중…";

  const winners = pickRandomUnique(candidates, pickCount);
  const lockedIds = new Set();
  let pool = [...candidates];

  for (const winner of winners) {
    const targetIndex = pool.findIndex((c) => c.id === winner.id);
    await runSpin(pool.length, targetIndex, (idx, step) => {
      const flashColor = step % 2 === 0 ? "#F5B942" : "#FFFFFF";
      drawViewfinderBoxes(candidates, { activeId: pool[idx].id, activeColor: flashColor, lockedIds });
    });
    lockedIds.add(winner.id);
    pool = pool.filter((c) => c.id !== winner.id);
    drawViewfinderBoxes(candidates, { lockedIds });
  }

  video.pause();
  const frameSnapshot = captureFrameCanvas();

  showResult(
    winners.map((winner, i) => ({
      label: `발표자 ${i + 1} · ${winner.id}번`,
      thumb: cropFaceThumbnail(frameSnapshot, winner),
    })),
    resultPanel,
    resultList
  );
  fireConfetti();

  drawBtn.classList.remove("is-drawing");
  drawBtn.textContent = "추첨 시작";
  drawBtn.hidden = true;
  resetBtn.hidden = false;
}

function resetCameraMode() {
  resultPanel.hidden = true;
  resultList.innerHTML = "";
  drawBtn.hidden = false;
  resetBtn.hidden = true;
  drawBtn.disabled = false;
  state.isDrawing = false;
  video.play();
}

/* ----------------------------------------------------------
   13. 번호 추첨 모드
   ---------------------------------------------------------- */
function buildManualGrid() {
  const total = Math.max(1, Number(totalCountInput.value) || 30);
  manualGrid.innerHTML = "";
  for (let i = 1; i <= total; i++) {
    const cell = document.createElement("div");
    cell.className = "manual-num";
    cell.textContent = i;
    cell.dataset.num = String(i);
    manualGrid.appendChild(cell);
  }
  pickCountManualInput.max = total;
}

totalCountInput.addEventListener("change", buildManualGrid);
$("countMinusM").addEventListener("click", () => stepInput(pickCountManualInput, -1));
$("countPlusM").addEventListener("click", () => stepInput(pickCountManualInput, 1));

drawBtnManual.addEventListener("click", runManualDraw);
resetBtnManual.addEventListener("click", resetManualMode);

async function runManualDraw() {
  const total = Math.max(1, Number(totalCountInput.value) || 30);
  const pickCount = Number(pickCountManualInput.value);
  if (pickCount > total) return;

  drawBtnManual.disabled = true;
  drawBtnManual.classList.add("is-drawing");
  drawBtnManual.textContent = "추첨 중…";

  const numberPool = Array.from({ length: total }, (_, i) => i + 1);
  const winners = pickRandomUnique(numberPool, pickCount);

  let pool = Array.from(manualGrid.children);

  for (const winnerNum of winners) {
    const targetIndex = pool.findIndex((c) => Number(c.dataset.num) === winnerNum);
    await runSpin(pool.length, targetIndex, (idx, step) => {
      pool.forEach((c) => c.classList.remove("is-flash-a", "is-flash-b"));
      pool[idx].classList.add(step % 2 === 0 ? "is-flash-a" : "is-flash-b");
    });
    pool[targetIndex].classList.remove("is-flash-a", "is-flash-b");
    pool[targetIndex].classList.add("is-winner");
    pool = pool.filter((_, i) => i !== targetIndex);
  }

  showResult(
    winners.map((n) => ({ label: `${n}번` })),
    manualResultPanel,
    manualResultList
  );
  fireConfetti();

  drawBtnManual.classList.remove("is-drawing");
  drawBtnManual.textContent = "추첨 시작";
  drawBtnManual.hidden = true;
  resetBtnManual.hidden = false;
}

function resetManualMode() {
  manualResultPanel.hidden = true;
  manualResultList.innerHTML = "";
  drawBtnManual.hidden = false;
  resetBtnManual.hidden = true;
  drawBtnManual.disabled = false;
  buildManualGrid();
}

/* ----------------------------------------------------------
   14. 공통 유틸
   ---------------------------------------------------------- */
function pickRandomUnique(list, count) {
  const pool = [...list];
  const picked = [];
  const n = Math.min(count, pool.length);

  for (let i = 0; i < n; i++) {
    const randBuf = new Uint32Array(1);
    crypto.getRandomValues(randBuf);
    const idx = randBuf[0] % pool.length;
    picked.push(pool[idx]);
    pool.splice(idx, 1);
  }
  return picked;
}

function showResult(items, panelEl, listEl) {
  listEl.innerHTML = "";
  items.forEach((item, i) => {
    const chip = document.createElement("div");
    chip.className = "result-chip" + (item.thumb ? " has-thumb" : "");
    chip.style.animationDelay = `${i * 90}ms`;

    if (item.thumb) {
      const img = document.createElement("img");
      img.src = item.thumb;
      img.alt = item.label;
      chip.appendChild(img);
    }

    const labelEl = document.createElement("span");
    labelEl.className = "result-chip-label";
    labelEl.textContent = item.label;
    chip.appendChild(labelEl);

    listEl.appendChild(chip);
  });
  panelEl.hidden = false;
}

function fireConfetti() {
  resizeOverlay();
  const ctx = confettiCanvas.getContext("2d");
  const cssW = window.innerWidth;
  const cssH = window.innerHeight;

  const colors = ["#F5B942", "#4FD1C5", "#F1EFE7", "#E1523D"];
  const particles = Array.from({ length: 140 }, () => ({
    x: Math.random() * cssW,
    y: -20 - Math.random() * 200,
    size: 4 + Math.random() * 6,
    speedY: 2 + Math.random() * 3,
    speedX: -1.5 + Math.random() * 3,
    rotation: Math.random() * 360,
    spin: -6 + Math.random() * 12,
    color: colors[Math.floor(Math.random() * colors.length)],
  }));

  const start = performance.now();
  function tick(now) {
    const elapsed = now - start;
    ctx.clearRect(0, 0, cssW, cssH);

    particles.forEach((p) => {
      p.x += p.speedX;
      p.y += p.speedY;
      p.rotation += p.spin;

      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate((p.rotation * Math.PI) / 180);
      ctx.fillStyle = p.color;
      ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
      ctx.restore();
    });

    if (elapsed < 2600) {
      requestAnimationFrame(tick);
    } else {
      ctx.clearRect(0, 0, cssW, cssH);
    }
  }
  requestAnimationFrame(tick);
}

/* ----------------------------------------------------------
   15. 시작
   ---------------------------------------------------------- */
window.addEventListener("DOMContentLoaded", () => {
  playShutterOpen();
  buildManualGrid();
  initCamera();
});
