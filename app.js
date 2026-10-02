/* ============================================================
   오늘의 발표자 · AI 얼굴 인식 추첨기
   - 카메라 모드: MediaPipe Tasks Vision(FaceDetector)으로 실시간 얼굴 인식
   - 번호 추첨 모드: 카메라 없이 출석번호로 추첨 (예비/대체 수단)
   ※ 외부 라이브러리는 MediaPipe(@mediapipe/tasks-vision) 하나만 CDN에서 로드합니다.
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

// 감도 모드별 임계치 (초고감도 교실 모드: 30명 다인원·원거리 학생까지 감지)
const SENSITIVITY_CONFIG = {
  ultra: { minConfidence: 0.18, minSuppression: 0.25 }, // 30명 교실 추천 기본값
  high: { minConfidence: 0.30, minSuppression: 0.30 },
  normal: { minConfidence: 0.48, minSuppression: 0.40 },
};

// 해상도 모드별 프리셋. 교실 크기·카메라 거리에 따라 사용자가 직접 선택합니다.
const RESOLUTION_PRESETS = {
  close: { width: 1280, height: 720, label: "720p" },
  standard: { width: 1920, height: 1080, label: "1080p" },
  far: { width: 2560, height: 1440, label: "1440p" },
  max: { width: 3840, height: 2160, label: "4K" },
};

// 트래킹 스무딩 계수
const SMOOTHING = 0.35;
const MATCH_THRESHOLD = 0.14;
// 일시적 가림·움직임에도 박스를 유지할 시간 (ms) - 교실 환경에 맞게 8초로 넉넉히 연장
const MISS_TIMEOUT_MS = 8000;

// 교실 원거리/다인원 감지를 위한 타일 분할 정의 (정규화 좌표)
// 전체 화면 1회 + 영역별 타일 크롭 확대를 번갈아 수행해 작은 얼굴 감지율을 극대화합니다.
const TILES = [
  { name: "left", x: 0.0, y: 0.05, w: 0.55, h: 0.8 },
  { name: "center", x: 0.22, y: 0.05, w: 0.56, h: 0.8 },
  { name: "right", x: 0.45, y: 0.05, w: 0.55, h: 0.8 },
];

const state = {
  mode: "camera", // "camera" | "manual"
  faceDetector: null,
  visionResolver: null,
  stream: null,
  running: false,
  tracks: [], // [{id, box:{x,y,w,h}, lastSeen, isManual, isLocked}]
  nextTrackId: 1,
  isDrawing: false, // 추첨 애니메이션 진행 중 여부
  coverMap: null, // object-fit:cover 표시 영역
  isMirror: true, // 기본 좌우 반전
  isLocked: false, // 학생 위치 고정(잠금) 여부
  sensitivity: "ultra", // ultra | high | normal
  currentTileIdx: 0,
  tileCanvas: null,
  tileCtx: null,
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

// 신규 교실 도구 DOM 참조
const mirrorToggleBtn = $("mirrorToggleBtn");
const lockPositionsBtn = $("lockPositionsBtn");
const fill30Btn = $("fill30Btn");
const grid30Btn = $("grid30Btn");
const clearAllFacesBtn = $("clearAllFacesBtn");
const sensitivitySelect = $("sensitivitySelect");
const touchNotice = $("touchNotice");

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
   3. 모드 전환 (카메라 인식 / 번호 추첨)
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
  // 해상도 모드를 바꾸면 현재 선택된 카메라로 스트림을 다시 시작합니다.
  startVideoStream(deviceSelect.value || undefined);
});

/* ----------------------------------------------------------
   5-1. 밝기 보정 (역광·저조도 보정)
   ---------------------------------------------------------- */
// 카메라 자체의 노출(exposure)을 브라우저에서 직접 제어하기는 어려우므로,
// 화면에 표시되는 영상에 밝기/대비 필터를 입혀 어둡게 찍힌 화면을 보정합니다.
function applyBrightness(value) {
  const brightness = Number(value);
  // 밝기를 올릴수록 대비도 살짝 함께 올려 뿌옇게 날리는 느낌을 줄입니다.
  const contrast = 1 + (brightness - 1) * 0.35;
  video.style.filter = `brightness(${brightness}) contrast(${contrast})`;
  brightnessValue.textContent = `${brightness.toFixed(2)}×`;
}

brightnessSlider.addEventListener("input", () => applyBrightness(brightnessSlider.value));
applyBrightness(brightnessSlider.value); // 초기값 즉시 적용 (역광 환경을 감안해 기본값을 살짝 밝게 설정)

/* ----------------------------------------------------------
   6. 카메라 스트림 시작
   ---------------------------------------------------------- */
async function startVideoStream(deviceId) {
  // 기존 스트림이 있다면 정리
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

    // 요청한 해상도와 실제로 카메라가 제공한 해상도가 다를 수 있으므로 실측값을 표시합니다.
    const actualW = video.videoWidth;
    const actualH = video.videoHeight;
    setCameraStatus(true, `인식 준비 완료 · ${actualW}×${actualH} (요청: ${preset.label})`);

    // 라벨을 보려면 최초 권한 허용 이후 장치 목록을 다시 채워야 합니다.
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

  // 오버레이 캔버스: 화면에 표시되는 CSS 크기 × 디바이스 픽셀비로 내부 해상도를 맞춰
  // 고해상도(레티나 등) 화면에서도 박스 선이 흐려지지 않게 합니다.
  const cssW = video.clientWidth || overlay.clientWidth;
  const cssH = video.clientHeight || overlay.clientHeight;

  // 레이아웃이 아직 확정되지 않아 크기가 0인 순간에는 건너뜁니다.
  // (이 시점에 0×0으로 캔버스를 설정하면 이후 다시 리사이즈 이벤트가 없을 경우
  //  박스가 영원히 그려지지 않는 문제가 있었습니다.)
  if (!cssW || !cssH) return;

  overlay.width = Math.round(cssW * dpr);
  overlay.height = Math.round(cssH * dpr);
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // 컨페티 캔버스도 동일하게 HiDPI 대응
  confettiCanvas.width = Math.round(window.innerWidth * dpr);
  confettiCanvas.height = Math.round(window.innerHeight * dpr);
  confettiCanvas.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);

  // video는 object-fit:cover로 표시되므로, 실제 화면에 "보이는" 영상 영역만
  // 원본 픽셀 좌표계에서 계산해 둡니다. 이 영역 기준으로 얼굴 좌표를 매핑해야
  // 좌우 화면 비율이 다른 카메라에서도 박스가 얼굴 위치와 어긋나지 않습니다.
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
// 카메라/해상도 전환 시 영상의 실제 프레임 크기가 확정되는 시점에도 다시 계산합니다.
video.addEventListener("loadedmetadata", resizeOverlay);
video.addEventListener("resize", resizeOverlay);
// ResizeObserver: 위 이벤트들만으로는 "레이아웃이 실제로 몇 px로 확정됐는지"를
// 정확한 타이밍에 알 수 없어 0×0으로 잘못 설정되는 경우가 있었습니다.
// .viewport 박스의 실제 렌더링 크기가 바뀔 때마다(0→실제 크기가 되는 최초 순간 포함)
// 확실하게 다시 계산하도록 보강합니다.
new ResizeObserver(() => resizeOverlay()).observe(video.closest(".viewport"));

/* ----------------------------------------------------------
   7. MediaPipe FaceDetector 초기화 및 감도 재설정
   ---------------------------------------------------------- */
async function initFaceDetector() {
  setCameraStatus(false, "AI 인식 모델을 준비하는 중…");
  if (!state.visionResolver) {
    state.visionResolver = await FilesetResolver.forVisionTasks(WASM_URL);
  }

  // 오프스크린 타일 캔버스 초기화 (작은 얼굴 확대를 위한 타일 버퍼)
  if (!state.tileCanvas) {
    state.tileCanvas = document.createElement("canvas");
    state.tileCanvas.width = 640;
    state.tileCanvas.height = 480;
    state.tileCtx = state.tileCanvas.getContext("2d", { willReadFrequently: true });
  }

  const sens = SENSITIVITY_CONFIG[state.sensitivity] || SENSITIVITY_CONFIG.ultra;

  try {
    state.faceDetector = await FaceDetector.createFromOptions(state.visionResolver, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: sens.minConfidence,
      minSuppressionThreshold: sens.minSuppression,
    });
  } catch (err) {
    console.warn("GPU 델리게이트 실패, CPU로 재시도합니다.", err);
    state.faceDetector = await FaceDetector.createFromOptions(state.visionResolver, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      minDetectionConfidence: sens.minConfidence,
      minSuppressionThreshold: sens.minSuppression,
    });
  }
}

async function reinitFaceDetector() {
  if (!state.stream) return;
  setCameraStatus(true, "AI 인식 감도 재설정 중…");
  await initFaceDetector();
  setCameraStatus(true, "인식 준비 완료");
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
      "AI 인식 모델을 불러오지 못했습니다. 네트워크 연결을 확인하거나 번호 추첨 모드를 이용해주세요."
    );
  }
}

/* ----------------------------------------------------------
   9. 실시간 인식 루프 (전체 프레임 + 멀티 타일 스캔 + NMS 병합)
   ---------------------------------------------------------- */
let frameCounter = 0;

function detectionLoop() {
  if (!state.running) return;

  if (video.readyState >= 2 && state.faceDetector && !state.isDrawing) {
    const now = performance.now();

    // 위치 고정(Lock) 상태일 때는 학생들의 기존 박스를 자리에 그대로 유지합니다.
    if (state.isLocked) {
      state.tracks.forEach((t) => (t.lastSeen = now));
      drawViewfinderBoxes(state.tracks);
      faceCountNum.textContent = state.tracks.length;
      updateDrawButtonState();
      requestAnimationFrame(detectionLoop);
      return;
    }

    try {
      // 1. 전체 화면 검출 (전경 및 큰 얼굴)
      const fullResult = state.faceDetector.detectForVideo(video, now);
      let detectedBoxes = (fullResult.detections || []).map((d) => toNormalizedBox(d.boundingBox));

      // 2. 멀티 타일 확대 스캔 (교실 원거리/뒷자리 작은 얼굴을 2~3배 확대 검출)
      frameCounter++;
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        const tile = TILES[state.currentTileIdx % TILES.length];
        state.currentTileIdx++;

        const map = state.coverMap || {
          srcX: 0,
          srcY: 0,
          srcW: video.videoWidth,
          srcH: video.videoHeight,
        };

        // 타일 원본 영역을 오프스크린 캔버스(640x480)에 확대 렌더링
        const sx = map.srcX + tile.x * map.srcW;
        const sy = map.srcY + tile.y * map.srcH;
        const sw = tile.w * map.srcW;
        const sh = tile.h * map.srcH;

        state.tileCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 640, 480);

        // MediaPipe는 timestamp가 엄격히 증가해야 하므로 미세 오프셋 부여
        const tileResult = state.faceDetector.detectForVideo(state.tileCanvas, now + 0.003);
        if (tileResult.detections && tileResult.detections.length > 0) {
          const tileBoxes = tileResult.detections.map((d) => {
            const bb = d.boundingBox;
            // 타일 로컬 정규화 좌표 -> 화면 전체 정규화 좌표로 역변환
            const localX = bb.originX / 640;
            const localY = bb.originY / 480;
            const localW = bb.width / 640;
            const localH = bb.height / 480;
            return {
              x: tile.x + localX * tile.w,
              y: tile.y + localY * tile.h,
              w: localW * tile.w,
              h: localH * tile.h,
            };
          });
          detectedBoxes.push(...tileBoxes);
        }
      }

      // 3. 중복 박스 병합 (IoU 기반 Non-Maximum Suppression)
      const mergedBoxes = applyNMS(detectedBoxes, 0.35);

      // 4. 트랙 갱신 (위치 매칭 및 스무딩)
      updateTracks(mergedBoxes, now);
    } catch (err) {
      console.warn("Detection error in loop:", err);
    }

    drawViewfinderBoxes(state.tracks);
    faceCountNum.textContent = state.tracks.length;
    updateDrawButtonState();
  }

  requestAnimationFrame(detectionLoop);
}

// BoundingBox를 화면 표시 영역(coverMap) 기준 정규화 좌표(0~1)로 변환
function toNormalizedBox(bb) {
  const map = state.coverMap || {
    srcX: 0,
    srcY: 0,
    srcW: video.videoWidth,
    srcH: video.videoHeight,
  };
  return {
    x: (bb.originX - map.srcX) / map.srcW,
    y: (bb.originY - map.srcY) / map.srcH,
    w: bb.width / map.srcW,
    h: bb.height / map.srcH,
  };
}

// 두 정규화 박스 간의 IoU (Intersection over Union) 계산
function calculateIoU(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = a.w * a.h;
  const areaB = b.w * b.h;
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

// 중복 박스 병합 (NMS)
function applyNMS(boxes, iouThreshold = 0.35) {
  if (boxes.length <= 1) return boxes;
  // 박스 크기(넓이)가 적당한 것 우선 정렬
  const sorted = [...boxes].sort((a, b) => b.w * b.h - a.w * a.h);
  const result = [];

  for (const box of sorted) {
    let overlap = false;
    for (const chosen of result) {
      if (calculateIoU(box, chosen) > iouThreshold) {
        overlap = true;
        break;
      }
    }
    if (!overlap) {
      result.push(box);
    }
  }
  return result;
}

// 이전 프레임 트랙과 이번 프레임 인식 결과를 매칭하고 스무딩
function updateTracks(boxes, now) {
  const usedBoxIdx = new Set();

  // 기존 트랙들과 매칭
  state.tracks.forEach((track) => {
    // 수동 등록(isManual) 트랙은 타이머 만료 없이 영구 유지
    if (track.isManual) {
      track.lastSeen = now;
      return;
    }

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
      track.box.x += (box.x - track.box.x) * SMOOTHING;
      track.box.y += (box.y - track.box.y) * SMOOTHING;
      track.box.w += (box.w - track.box.w) * SMOOTHING;
      track.box.h += (box.h - track.box.h) * SMOOTHING;
      track.lastSeen = now;
      usedBoxIdx.add(bestIdx);
    }
  });

  // 매칭되지 않은 새 얼굴을 새 트랙으로 추가
  boxes.forEach((box, i) => {
    if (usedBoxIdx.has(i)) return;
    // 너무 작거나 터무니없는 박스 필터링 (가로세로 0.02 미만)
    if (box.w < 0.02 || box.h < 0.02) return;
    state.tracks.push({
      id: state.nextTrackId++,
      box: { ...box },
      lastSeen: now,
      isManual: false,
    });
  });

  // MISS_TIMEOUT_MS 동안 다시 인식되지 않은 트랙만 정리 (수동 트랙은 보존)
  state.tracks = state.tracks.filter(
    (t) => t.isManual || now - t.lastSeen <= MISS_TIMEOUT_MS
  );
}

/* ----------------------------------------------------------
   10. 뷰파인더 스타일 박스 그리기 (카메라 AF 브래킷 모티프)
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

    let color = track.isManual ? "#38BDF8" : "#4FD1C5"; // 수동 추가는 스카이블루, AI는 cyan
    let lineWidth = 2;

    if (lockedIds?.has(track.id)) {
      color = "#F5B942";
      lineWidth = 4;
    } else if (track.id === activeId) {
      color = activeColor;
      lineWidth = 5;
    } else if (state.isLocked) {
      lineWidth = 2.5;
    }

    drawCornerBrackets(overlayCtx, x, y, w, h, color, lineWidth);
    drawFaceNumberBadge(overlayCtx, x, y, w, h, track.id, color, track.isManual);
  });
}

// 각 얼굴 박스 위에 고유 번호 배지 렌더링
function drawFaceNumberBadge(ctx, x, y, w, h, id, color, isManual = false) {
  const label = String(id);
  const cx = x + Math.min(22, w * 0.18);
  const cy = Math.max(16, y - 14);

  ctx.save();
  ctx.translate(cx, cy);
  // 미러링 상태에 맞춰 텍스트를 반전시켜 항상 정방향으로 보이게 처리
  ctx.scale(state.isMirror ? -1 : 1, 1);

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

// 사각형 전체가 아닌 네 모서리만 그려주는 '오토포커스 브래킷' 스타일
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

// 추첨이 끝난 순간의 영상 프레임을 캔버스에 그대로 캡처해 둡니다.
// (video 자체를 pause()해서 화면도 함께 정지시키고, 이 캔버스에서 당첨자 얼굴만
//  잘라내 결과 화면에 작은 썸네일로 보여주는 데 사용합니다.)
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

// 캡처된 프레임에서 특정 얼굴 주변을 넉넉히 잘라 정사각형 썸네일(dataURL)로 반환합니다.
function cropFaceThumbnail(frameCanvas, track, outSize = 160) {
  const cssW = frameCanvas.width;
  const cssH = frameCanvas.height;
  const boxW = track.box.w * cssW;
  const boxH = track.box.h * cssH;
  const cx = track.box.x * cssW + boxW / 2;
  const cy = track.box.y * cssH + boxH / 2;
  const cropSize = Math.max(boxW, boxH) * 1.8; // 얼굴 주변 여유를 넉넉히 포함해 알아보기 쉽게

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

/* ----------------------------------------------------------
   11-1. 교실 30명 전용 강력 지원 도구 (강제 인식, 터치, 그리드, 잠금)
   ---------------------------------------------------------- */

// [도구 1] 화면 터치/클릭으로 얼굴 박스 강제 등록 / 삭제
overlay.addEventListener("pointerdown", handleOverlayPointerDown);

function handleOverlayPointerDown(e) {
  if (state.isDrawing) return;

  const rect = overlay.getBoundingClientRect();
  const visualX = (e.clientX - rect.left) / rect.width;
  const visualY = (e.clientY - rect.top) / rect.height;

  // 좌우 반전 상태(isMirror)에 맞추어 실제 트랙 정규화 좌표(0~1)로 변환
  const normX = state.isMirror ? 1 - visualX : visualX;
  const normY = visualY;

  // 이미 존재하는 박스를 터치했는지 확인 (박스 중심과의 거리)
  const hitIdx = state.tracks.findIndex((t) => {
    const cx = t.box.x + t.box.w / 2;
    const cy = t.box.y + t.box.h / 2;
    const dist = Math.hypot(normX - cx, normY - cy);
    return dist < Math.max(t.box.w * 0.8, t.box.h * 0.8, 0.05);
  });

  if (hitIdx !== -1) {
    // 이미 존재하는 박스 터치 시 삭제
    state.tracks.splice(hitIdx, 1);
  } else {
    // 빈 영역 터치 시 해당 위치에 수동 강제 얼굴 박스 등록
    const boxW = 0.065;
    const boxH = 0.085;
    state.tracks.push({
      id: state.nextTrackId++,
      box: {
        x: Math.max(0, Math.min(normX - boxW / 2, 1 - boxW)),
        y: Math.max(0, Math.min(normY - boxH / 2, 1 - boxH)),
        w: boxW,
        h: boxH,
      },
      lastSeen: performance.now(),
      isManual: true,
    });
  }

  drawViewfinderBoxes(state.tracks);
  faceCountNum.textContent = state.tracks.length;
  updateDrawButtonState();
}

// [도구 2] 위치 고정 (학생들이 움직이거나 고개를 숙여도 박스 유지)
lockPositionsBtn.addEventListener("click", toggleLockPositions);

function toggleLockPositions() {
  state.isLocked = !state.isLocked;
  lockPositionsBtn.classList.toggle("is-locked", state.isLocked);
  lockPositionsBtn.textContent = state.isLocked ? "🔒 위치 고정됨 (해제)" : "📌 위치 고정 (잠금)";
  drawViewfinderBoxes(state.tracks);
}

// [도구 3] 좌우 반전 (미러 모드) 토글
mirrorToggleBtn.addEventListener("click", toggleMirror);

function toggleMirror() {
  state.isMirror = !state.isMirror;
  video.classList.toggle("is-unmirrored", !state.isMirror);
  overlay.classList.toggle("is-unmirrored", !state.isMirror);
  mirrorToggleBtn.classList.toggle("is-active", state.isMirror);
  mirrorToggleBtn.textContent = state.isMirror ? "↔️ 좌우 반전 ON" : "➡️ 좌우 반전 OFF";
  drawViewfinderBoxes(state.tracks);
}

// [도구 4] 교실 30명 좌석 일괄 배치 (5행 6열 그리드)
grid30Btn.addEventListener("click", create30Grid);

function create30Grid() {
  state.tracks = [];
  state.nextTrackId = 1;

  const rows = 5; // 5행 (앞자리~뒷자리)
  const cols = 6; // 6열 (좌~우)
  const now = performance.now();

  for (let r = 0; r < rows; r++) {
    // 원근감: 뒷자리(r=0)일수록 상단이고 작으며, 앞자리(r=4)일수록 하단이고 큼
    const normY = 0.16 + (r / (rows - 1)) * 0.62;
    const boxW = 0.048 + (r / (rows - 1)) * 0.024;
    const boxH = 0.065 + (r / (rows - 1)) * 0.032;

    for (let c = 0; c < cols; c++) {
      const normX = 0.08 + (c / (cols - 1)) * 0.84 - boxW / 2;
      state.tracks.push({
        id: state.nextTrackId++,
        box: {
          x: Math.max(0, Math.min(normX, 1 - boxW)),
          y: Math.max(0, Math.min(normY, 1 - boxH)),
          w: boxW,
          h: boxH,
        },
        lastSeen: now,
        isManual: true,
      });
    }
  }

  // 자동 배치 후 학생 움직임에 박스가 흐트러지지 않도록 자동으로 고정 모드 활성화
  state.isLocked = true;
  lockPositionsBtn.classList.add("is-locked");
  lockPositionsBtn.textContent = "🔒 위치 고정됨 (해제)";

  drawViewfinderBoxes(state.tracks);
  faceCountNum.textContent = state.tracks.length;
  updateDrawButtonState();
}

// [도구 5] 미인식 빈자리 자동 채우기 (현재 인식 유지 + 부족분 자동 보충하여 정확히 30명 맞춤)
fill30Btn.addEventListener("click", fillRemainingTo30);

function fillRemainingTo30() {
  const targetTotal = 30;
  if (state.tracks.length >= targetTotal) {
    alert(`이미 ${state.tracks.length}명이 인식/등록되어 있습니다.`);
    return;
  }

  const rows = 5;
  const cols = 6;
  const candidateSlots = [];

  // 교실 30개 그리드 후보 위치 생성
  for (let r = 0; r < rows; r++) {
    const normY = 0.16 + (r / (rows - 1)) * 0.62;
    const boxW = 0.048 + (r / (rows - 1)) * 0.024;
    const boxH = 0.065 + (r / (rows - 1)) * 0.032;

    for (let c = 0; c < cols; c++) {
      const normX = 0.08 + (c / (cols - 1)) * 0.84 - boxW / 2;
      candidateSlots.push({ x: normX, y: normY, w: boxW, h: boxH });
    }
  }

  // 기존 트랙들과 가장 거리가 먼 빈 슬롯들을 찾아 우선 채움
  const scoredSlots = candidateSlots.map((slot) => {
    let minDist = Infinity;
    const cx = slot.x + slot.w / 2;
    const cy = slot.y + slot.h / 2;

    state.tracks.forEach((t) => {
      const tcx = t.box.x + t.box.w / 2;
      const tcy = t.box.y + t.box.h / 2;
      const dist = Math.hypot(cx - tcx, cy - tcy);
      if (dist < minDist) minDist = dist;
    });

    return { slot, minDist };
  });

  // 기존 트랙과 거리가 먼 빈 슬롯 순으로 정렬
  scoredSlots.sort((a, b) => b.minDist - a.minDist);

  const needed = targetTotal - state.tracks.length;
  const now = performance.now();

  for (let i = 0; i < needed && i < scoredSlots.length; i++) {
    const s = scoredSlots[i].slot;
    state.tracks.push({
      id: state.nextTrackId++,
      box: {
        x: Math.max(0, Math.min(s.x, 1 - s.w)),
        y: Math.max(0, Math.min(s.y, 1 - s.h)),
        w: s.w,
        h: s.h,
      },
      lastSeen: now,
      isManual: true,
    });
  }

  // 위치 고정 모드 활성화
  state.isLocked = true;
  lockPositionsBtn.classList.add("is-locked");
  lockPositionsBtn.textContent = "🔒 위치 고정됨 (해제)";

  drawViewfinderBoxes(state.tracks);
  faceCountNum.textContent = state.tracks.length;
  updateDrawButtonState();
}

// [도구 6] 등록된 모든 얼굴 트랙 초기화
clearAllFacesBtn.addEventListener("click", clearAllTracks);

function clearAllTracks() {
  state.tracks = [];
  state.nextTrackId = 1;
  state.isLocked = false;
  lockPositionsBtn.classList.remove("is-locked");
  lockPositionsBtn.textContent = "📌 위치 고정 (잠금)";
  drawViewfinderBoxes([]);
  faceCountNum.textContent = "0";
  updateDrawButtonState();
}

// [도구 7] AI 인식 감도 변경
sensitivitySelect.addEventListener("change", () => {
  state.sensitivity = sensitivitySelect.value;
  reinitFaceDetector();
});

function stepInput(input, delta) {
  const min = Number(input.min || 1);
  const max = input.max ? Number(input.max) : Infinity;
  input.value = Math.min(max, Math.max(min, Number(input.value) + delta));
  input.dispatchEvent(new Event("input"));
}

// ---- 룰렛 연출 공통 로직 -------------------------------------------------
// 고정된 지연시간 시퀀스: 처음엔 빠르게 지나가다 점점 느려지며(감속) 마지막에 목표 지점에 정확히 멈춥니다.
// 후보 수(n)와 무관하게 항상 같은 리듬으로 동작하도록 값을 고정해 두었습니다.
const SPIN_DELAYS = [70, 70, 80, 90, 100, 115, 135, 160, 190, 225, 265, 310, 360, 420];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// targetIndex에서 역산해, 원형으로 여러 바퀴를 돌다가 마지막 스텝에 정확히 target에서
// 멈추는 인덱스 시퀀스를 만듭니다.
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

// pool(길이 n) 안에서 targetIndex 하나를 향해 룰렛처럼 감속하며 도는 애니메이션.
// onStep(idx, stepNumber) 콜백으로 매 스텝마다 강조할 위치를 알려줍니다.
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
  // 화면상 왼쪽에서 오른쪽으로 자연스럽게 훑는 느낌을 주기 위해 x좌표 순으로 정렬
  const candidates = [...state.tracks].sort((a, b) => a.box.x - b.box.x);
  if (candidates.length < 1 || pickCount > candidates.length) return;

  state.isDrawing = true;
  drawBtn.disabled = true;
  drawBtn.classList.add("is-drawing");
  drawBtn.textContent = "추첨 중…";

  // 뽑힐 순서를 미리 정해 두고(공정한 비복원추출), 라운드마다 룰렛이 그 자리에서 멈추도록 연출합니다.
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

  // 화면을 정지시켜, 당첨자의 금색 박스가 실제 교실 화면 위에 그대로 남아있게 합니다.
  // (누가 몇 번 자리에 있었는지 박스 위치로 바로 확인 가능)
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
  // state.isDrawing은 여기서 false로 되돌리지 않습니다. 계속 true로 두어야
  // 실시간 인식 루프가 화면을 다시 덮어쓰지 않고, 정지된 화면 + 금색 강조 박스가
  // '다시 인식하기'를 누르기 전까지 그대로 유지됩니다.
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

  // 뽑힐 순서를 미리 정해 두고, 라운드마다 룰렛이 그 번호에서 멈추도록 연출합니다.
  const numberPool = Array.from({ length: total }, (_, i) => i + 1);
  const winners = pickRandomUnique(numberPool, pickCount);

  let pool = Array.from(manualGrid.children); // 그리드에 보이는 순서(1→N) 그대로 사용

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
   14. 공통 유틸: 무작위 비복원추출, 결과 표시, 컨페티
   ---------------------------------------------------------- */

// 암호학적 난수를 이용한 공정한 비복원추출(Fisher-Yates 부분 셔플)
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

// items: [{label, thumb?}] — thumb이 있으면(카메라 모드) 얼굴 썸네일을 함께 보여주고,
// 없으면(번호 추첨 모드) 기존처럼 큰 번호 칩만 보여줍니다.
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

// 외부 라이브러리 없이 캔버스로 그리는 가벼운 컨페티 연출
function fireConfetti() {
  resizeOverlay(); // 컨페티 캔버스도 최신 디바이스 픽셀비로 맞춥니다.
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
