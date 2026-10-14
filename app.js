/* ============================================================
   app.js · 메인 애플리케이션 오케스트레이터 (ES Module)
   - 모듈 결합: detector, tracker, raffle, audio
   - 뷰포트 좌표계 1:1 정합 렌더링
   - 박진감 넘치는 룰렛 애니메이션 화면 갱신
   - 전자칠판 터치 제어 및 현장 튜닝 연동
   ============================================================ */

import { soundFx } from "./js/audio.js";
import { FaceDetectionEngine } from "./js/detector.js";
import { FaceTracker } from "./js/tracker.js";
import { raffleEngine } from "./js/raffle.js";

// 해상도 모드별 프리셋
const RESOLUTION_PRESETS = {
  close: { width: 1280, height: 720, label: "720p" },
  standard: { width: 1920, height: 1080, label: "1080p" },
  far: { width: 2560, height: 1440, label: "1440p" },
  max: { width: 3840, height: 2160, label: "4K" }
};

class App {
  constructor() {
    this.detector = new FaceDetectionEngine();
    this.tracker = new FaceTracker();

    this.stream = null;
    this.animationFrameId = null;
    this.isRunning = false;
    this.isDrawing = false;

    // 화면 정합 좌표계
    this.videoWidth = 0;
    this.videoHeight = 0;
    this.displayWidth = 0;
    this.displayHeight = 0;
    this.scaleX = 1;
    this.scaleY = 1;
    this.isMirror = true;

    // 성능 및 프레임 제어
    this.frameCounter = 0;
    this.fpsFrames = 0;
    this.lastFpsTime = performance.now();

    // DOM 캐싱
    this.initDom();
    this.initEventListeners();
  }

  $(id) {
    return document.getElementById(id);
  }

  initDom() {
    this.video = this.$("video");
    this.overlay = this.$("overlay");
    this.overlayCtx = this.overlay.getContext("2d");
    this.viewportContainer = this.$("viewportContainer");
    this.flashOverlay = this.$("flashOverlay");

    this.cameraStatus = this.$("cameraStatus");
    this.cameraStatusText = this.$("cameraStatusText");
    this.faceCountNum = this.$("faceCountNum");
    this.fpsNum = this.$("fpsNum");
    this.lowCountAlert = this.$("lowCountAlert");

    this.deviceSelect = this.$("deviceSelect");
    this.resolutionSelect = this.$("resolutionSelect");
    this.mirrorToggleBtn = this.$("mirrorToggleBtn");
    this.lockPositionsBtn = this.$("lockPositionsBtn");
    this.muteToggleBtn = this.$("muteToggleBtn");

    this.fill30Btn = this.$("fill30Btn");
    this.grid30Btn = this.$("grid30Btn");
    this.clearAllFacesBtn = this.$("clearAllFacesBtn");

    this.confidenceSlider = this.$("confidenceSlider");
    this.confidenceVal = this.$("confidenceVal");
    this.tileSelect = this.$("tileSelect");
    this.speedSelect = this.$("speedSelect");
    this.preprocessBright = this.$("preprocessBright");

    this.pickCountInput = this.$("pickCount");
    this.drawBtn = this.$("drawBtn");
    this.resetBtn = this.$("resetBtn");
    this.resultPanel = this.$("resultPanel");
    this.resultList = this.$("resultList");
    this.historyTags = this.$("historyTags");

    this.confettiCanvas = this.$("confetti");
  }

  /* ---------- [1] 좌표계 1:1 정합 (허공 박스 원천 방지) ---------- */
  alignCanvas() {
    if (!this.video.videoWidth || !this.video.videoHeight) return;

    this.videoWidth = this.video.videoWidth;
    this.videoHeight = this.video.videoHeight;

    const containerW = this.viewportContainer.clientWidth;
    const containerH = this.viewportContainer.clientHeight;
    const videoRatio = this.videoWidth / this.videoHeight;
    const containerRatio = containerW / containerH;

    let displayW, displayH;
    if (videoRatio > containerRatio) {
      displayW = containerW;
      displayH = containerW / videoRatio;
    } else {
      displayH = containerH;
      displayW = containerH * videoRatio;
    }

    this.displayWidth = displayW;
    this.displayHeight = displayH;
    this.scaleX = displayW / this.videoWidth;
    this.scaleY = displayH / this.videoHeight;

    const dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.round(displayW * dpr);
    this.overlay.height = Math.round(displayH * dpr);
    this.overlay.style.width = `${displayW}px`;
    this.overlay.style.height = `${displayH}px`;
    this.overlay.style.left = `${(containerW - displayW) / 2}px`;
    this.overlay.style.top = `${(containerH - displayH) / 2}px`;

    this.overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

    this.confettiCanvas.width = window.innerWidth;
    this.confettiCanvas.height = window.innerHeight;
  }

  /* ---------- [2] 카메라 시작 & 제어 ---------- */
  async startCamera(deviceId) {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
    }
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }

    const preset = RESOLUTION_PRESETS[this.resolutionSelect.value] || RESOLUTION_PRESETS.standard;
    const constraints = {
      audio: false,
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: preset.width },
        height: { ideal: preset.height },
        frameRate: { ideal: 30 },
        facingMode: deviceId ? undefined : "environment"
      }
    };

    try {
      this.setCameraStatus(false, "카메라 연결 중…");
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      this.stream = stream;
      this.video.srcObject = stream;

      await this.video.play();
      this.alignCanvas();

      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      this.setCameraStatus(
        true,
        `준비 완료 · ${settings.width || this.video.videoWidth}×${settings.height || this.video.videoHeight} (${preset.label})`
      );

      await this.populateDevices();
      this.isRunning = true;
      this.animationFrameId = requestAnimationFrame(() => this.mainLoop());
    } catch (err) {
      console.error(err);
      this.setCameraStatus(false, "카메라를 시작할 수 없습니다. 권한을 확인해주세요.");
    }
  }

  setCameraStatus(ready, text) {
    this.cameraStatus.classList.toggle("is-ready", ready);
    this.cameraStatusText.textContent = text;
  }

  async populateDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cams = devices.filter((d) => d.kind === "videoinput");
      this.deviceSelect.innerHTML = "";
      cams.forEach((cam, i) => {
        const opt = document.createElement("option");
        opt.value = cam.deviceId;
        opt.textContent = cam.label || `카메라 ${i + 1}`;
        this.deviceSelect.appendChild(opt);
      });
    } catch (e) {}
  }

  /* ---------- [3] 메인 루프 (탐지 + 보간 렌더링) ---------- */
  mainLoop() {
    if (!this.isRunning) return;

    // FPS 계산
    this.fpsFrames++;
    const now = performance.now();
    if (now - this.lastFpsTime >= 1000) {
      this.fpsNum.textContent = String(this.fpsFrames);
      this.fpsFrames = 0;
      this.lastFpsTime = now;
    }

    // 추첨 중(isDrawing)이 아닐 때만 AI 탐지 수행
    if (!this.isDrawing) {
      this.frameCounter++;
      // 3프레임마다 1회 탐지, 나머지 프레임은 부드러운 위치 보간 렌더링
      if (this.frameCounter % 3 === 0) {
        const detections = this.detector.detect(this.video);
        this.tracker.update(detections);
      }
      this.renderViewfinder();
    }

    const confirmed = this.tracker.getConfirmedTracks();
    this.faceCountNum.textContent = String(confirmed.length);
    this.lowCountAlert.style.display = confirmed.length > 0 && confirmed.length < 5 ? "block" : "none";
    this.updateDrawButtonState();

    this.animationFrameId = requestAnimationFrame(() => this.mainLoop());
  }

  /* ---------- [4] 뷰파인더 렌더링 (룰렛 하이라이트 & 펄스) ---------- */
  renderViewfinder({ activeId = null, flashColor = "#F5B942", lockedIds = null } = {}) {
    this.overlayCtx.clearRect(0, 0, this.overlay.width, this.overlay.height);

    const tracks = this.tracker.getConfirmedTracks();

    tracks.forEach((track) => {
      let rx = track.box.x * this.scaleX;
      let ry = track.box.y * this.scaleY;
      const rw = track.box.w * this.scaleX;
      const rh = track.box.h * this.scaleY;

      // 좌우 반전 시 보정
      if (this.isMirror) {
        rx = this.displayWidth - (rx + rw);
      }

      // 색상 및 선 두께 결정
      let color = track.isManual ? "#38BDF8" : "#4FD1C5";
      let lineWidth = 2;
      let isHighlight = false;

      if (track.isExcluded) {
        color = "#64748B"; // 결석자: 어두운 회색
        lineWidth = 1.5;
      } else if (lockedIds?.has(track.id)) {
        color = "#F5B942"; // 당첨 확정자: 진한 금색
        lineWidth = 4;
      } else if (track.id === activeId) {
        color = flashColor; // 현재 룰렛 커서: 번쩍이는 하이라이트
        lineWidth = 6;
        isHighlight = true;
      }

      this.drawCornerBrackets(rx, ry, rw, rh, color, lineWidth, isHighlight);
      this.drawNumberBadge(rx, ry, rw, rh, track.id, color, isHighlight, track.isExcluded);
    });
  }

  drawCornerBrackets(x, y, w, h, color, lineWidth, isHighlight) {
    const ctx = this.overlayCtx;
    const len = Math.min(w, h) * 0.28;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";

    if (isHighlight) {
      // 룰렛 순회 중인 박스는 외곽 네온 글로우 효과
      ctx.shadowColor = color;
      ctx.shadowBlur = 18;
    }

    const corners = [
      [x, y, 1, 1],
      [x + w, y, -1, 1],
      [x, y + h, 1, -1],
      [x + w, y + h, -1, -1]
    ];
    corners.forEach(([cx, cy, dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(cx, cy + len * dy);
      ctx.lineTo(cx, cy);
      ctx.lineTo(cx + len * dx, cy);
      ctx.stroke();
    });
    ctx.restore();
  }

  drawNumberBadge(x, y, w, h, id, color, isHighlight, isExcluded) {
    const ctx = this.overlayCtx;
    const label = String(id) + (isExcluded ? " (결석)" : "");
    const cx = x + Math.min(22, w * 0.2);
    const cy = Math.max(14, y - 12);

    ctx.save();
    ctx.translate(cx, cy);
    if (isHighlight) {
      ctx.scale(1.15, 1.15); // 하이라이트 시 배지 15% 바운스 확대
    }

    ctx.font = "700 14px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    const textW = ctx.measureText(label).width;
    const boxW = Math.max(26, textW + 12);
    const boxH = 22;

    ctx.fillStyle = color;
    ctx.fillRect(-boxW / 2, -boxH / 2, boxW, boxH);
    ctx.fillStyle = "#0B0E14";
    ctx.fillText(label, 0, 1);
    ctx.restore();
  }

  /* ---------- [5] 박진감 넘치는 룰렛 추첨 실행 ---------- */
  async startRaffle() {
    const candidates = this.tracker.getRaffleCandidates();
    const pickCount = Number(this.pickCountInput.value);

    if (candidates.length < 1 || pickCount > candidates.length) {
      alert("추첨 가능한 학생 수가 부족합니다.");
      return;
    }

    this.isDrawing = true;
    this.drawBtn.disabled = true;
    this.drawBtn.classList.add("is-drawing");
    this.drawBtn.textContent = "추첨 진행 중…";

    // 룰렛 추첨 엔진 가동 (onStep, onWinnerLocked, onComplete 콜백)
    const winners = await raffleEngine.runCameraDraw({
      candidates,
      pickCount,
      onStep: ({ activeId, flashColor, lockedIds }) => {
        // 매 스텝마다 화면 캔버스에 룰렛 하이라이트를 즉각 렌더링!
        this.renderViewfinder({ activeId, flashColor, lockedIds });
      },
      onWinnerLocked: ({ winner, lockedIds }) => {
        // 당첨자 1명 확정 순간 화면 셔터 플래시 번쩍!
        this.triggerScreenFlash();
        this.renderViewfinder({ lockedIds });
      },
      onComplete: ({ winners, lockedIds }) => {
        this.renderViewfinder({ lockedIds });
        this.showRaffleResults(winners);
      }
    });
  }

  triggerScreenFlash() {
    this.flashOverlay.classList.add("is-active");
    setTimeout(() => {
      this.flashOverlay.classList.remove("is-active");
    }, 250);
  }

  showRaffleResults(winners) {
    // 썸네일 생성을 위한 클린 프레임 캡처
    const cleanCanvas = document.createElement("canvas");
    cleanCanvas.width = this.videoWidth;
    cleanCanvas.height = this.videoHeight;
    cleanCanvas.getContext("2d").drawImage(this.video, 0, 0);

    this.resultList.innerHTML = "";
    winners.forEach((w, i) => {
      const chip = document.createElement("div");
      chip.className = "result-chip";
      chip.style.animationDelay = `${i * 120}ms`;

      const thumbUrl = raffleEngine.cropThumbnail(cleanCanvas, w);
      const img = document.createElement("img");
      img.src = thumbUrl;
      img.alt = `발표자 ${w.id}번`;
      chip.appendChild(img);

      const label = document.createElement("span");
      label.className = "result-chip-label";
      label.textContent = `발표자 ${i + 1} · ${w.id}번`;
      chip.appendChild(label);

      this.resultList.appendChild(chip);
    });

    this.resultPanel.hidden = false;
    this.fireConfetti();
    this.updateHistoryUI();

    this.drawBtn.classList.remove("is-drawing");
    this.drawBtn.hidden = true;
    this.resetBtn.hidden = false;
  }

  resetRaffle() {
    this.resultPanel.hidden = true;
    this.resultList.innerHTML = "";
    this.drawBtn.hidden = false;
    this.resetBtn.hidden = true;
    this.drawBtn.textContent = "추첨 시작";
    this.drawBtn.disabled = false;
    this.isDrawing = false;
  }

  updateHistoryUI() {
    this.historyTags.innerHTML = "";
    raffleEngine.history.forEach((h) => {
      const tag = document.createElement("span");
      tag.className = "history-badge";
      tag.textContent = `${h.id}번`;
      this.historyTags.appendChild(tag);
    });
  }

  /* ---------- [6] 컨페티 축하 폭죽 연출 ---------- */
  fireConfetti() {
    const ctx = this.confettiCanvas.getContext("2d");
    const w = window.innerWidth;
    const h = window.innerHeight;
    const colors = ["#F5B942", "#4FD1C5", "#F1EFE7", "#E1523D", "#38BDF8"];
    const particles = Array.from({ length: 140 }, () => ({
      x: Math.random() * w,
      y: -20 - Math.random() * 200,
      size: 5 + Math.random() * 6,
      speedY: 2.5 + Math.random() * 3.5,
      speedX: -1.5 + Math.random() * 3,
      rot: Math.random() * 360,
      spin: -5 + Math.random() * 10,
      color: colors[Math.floor(Math.random() * colors.length)]
    }));

    const start = performance.now();
    const tick = (now) => {
      ctx.clearRect(0, 0, w, h);
      particles.forEach((p) => {
        p.x += p.speedX;
        p.y += p.speedY;
        p.rot += p.spin;
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate((p.rot * Math.PI) / 180);
        ctx.fillStyle = p.color;
        ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
        ctx.restore();
      });
      if (now - start < 2800) requestAnimationFrame(tick);
      else ctx.clearRect(0, 0, w, h);
    };
    requestAnimationFrame(tick);
  }

  /* ---------- [7] UI 이벤트 리스너 바인딩 ---------- */
  initEventListeners() {
    window.addEventListener("resize", () => this.alignCanvas());
    this.video.addEventListener("loadedmetadata", () => this.alignCanvas());
    new ResizeObserver(() => this.alignCanvas()).observe(this.viewportContainer);

    this.deviceSelect.addEventListener("change", () => this.startCamera(this.deviceSelect.value));
    this.resolutionSelect.addEventListener("change", () => this.startCamera(this.deviceSelect.value || undefined));

    // 미러링 토글
    this.mirrorToggleBtn.addEventListener("click", () => {
      this.isMirror = !this.isMirror;
      this.video.classList.toggle("is-unmirrored", !this.isMirror);
      this.mirrorToggleBtn.classList.toggle("is-active", this.isMirror);
      this.mirrorToggleBtn.textContent = this.isMirror ? "↔️ 좌우 반전 ON" : "➡️ 좌우 반전 OFF";
      this.renderViewfinder();
    });

    // 위치 고정 토글
    this.lockPositionsBtn.addEventListener("click", () => {
      this.tracker.isLocked = !this.tracker.isLocked;
      this.lockPositionsBtn.classList.toggle("is-locked", this.tracker.isLocked);
      this.lockPositionsBtn.textContent = this.tracker.isLocked ? "🔒 위치 고정됨 (해제)" : "📌 위치 고정 (잠금)";
    });

    // 음소거 토글
    this.muteToggleBtn.addEventListener("click", () => {
      const isMuted = soundFx.toggleMute();
      this.muteToggleBtn.classList.toggle("is-muted", isMuted);
      this.muteToggleBtn.textContent = isMuted ? "🔇 소리 OFF" : "🔊 소리 ON";
    });

    // 30명 일괄 배치
    this.grid30Btn.addEventListener("click", () => {
      this.tracker.create30Grid(this.videoWidth || 1920, this.videoHeight || 1080);
      this.lockPositionsBtn.classList.add("is-locked");
      this.lockPositionsBtn.textContent = "🔒 위치 고정됨 (해제)";
      this.renderViewfinder();
      this.faceCountNum.textContent = "30";
      this.updateDrawButtonState();
    });

    // 30명 자동 채우기
    this.fill30Btn.addEventListener("click", () => {
      const added = this.tracker.fillTo30(this.videoWidth || 1920, this.videoHeight || 1080);
      if (!added) alert("이미 30명 이상 등록되어 있습니다.");
      this.lockPositionsBtn.classList.add("is-locked");
      this.lockPositionsBtn.textContent = "🔒 위치 고정됨 (해제)";
      this.renderViewfinder();
      this.faceCountNum.textContent = String(this.tracker.getConfirmedTracks().length);
      this.updateDrawButtonState();
    });

    // 전체 초기화
    this.clearAllFacesBtn.addEventListener("click", () => {
      this.tracker.clear();
      this.lockPositionsBtn.classList.remove("is-locked");
      this.lockPositionsBtn.textContent = "📌 위치 고정 (잠금)";
      this.renderViewfinder();
      this.faceCountNum.textContent = "0";
      this.updateDrawButtonState();
    });

    // 뷰포트 터치/클릭 (추가, 삭제, 결석자 토글)
    this.viewportContainer.addEventListener("pointerdown", (e) => {
      if (this.isDrawing) return;

      const rect = this.overlay.getBoundingClientRect();
      const clickX = e.clientX - rect.left;
      const clickY = e.clientY - rect.top;

      if (clickX < 0 || clickX > rect.width || clickY < 0 || clickY > rect.height) return;

      let targetX = this.isMirror ? (this.displayWidth - clickX) / this.scaleX : clickX / this.scaleX;
      let targetY = clickY / this.scaleY;

      // Shift+클릭 시 결석/추첨 제외 토글
      if (e.shiftKey) {
        const hit = this.tracker.tracks.find((t) => {
          const cx = t.box.x + t.box.w / 2;
          const cy = t.box.y + t.box.h / 2;
          return Math.hypot(targetX - cx, targetY - cy) < Math.max(t.box.w, t.box.h, 40);
        });
        if (hit) {
          this.tracker.toggleExclusion(hit.id);
          this.renderViewfinder();
          return;
        }
      }

      // 일반 터치: 기존 박스 삭제 or 새 박스 생성
      const removed = this.tracker.removeAt(targetX, targetY);
      if (!removed) {
        const defW = (this.videoWidth || 1920) * 0.065;
        const defH = (this.videoHeight || 1080) * 0.09;
        this.tracker.addManualTrack(
          Math.max(0, Math.min(targetX - defW / 2, this.videoWidth - defW)),
          Math.max(0, Math.min(targetY - defH / 2, this.videoHeight - defH)),
          defW,
          defH
        );
      }

      this.renderViewfinder();
      this.faceCountNum.textContent = String(this.tracker.getConfirmedTracks().length);
      this.updateDrawButtonState();
    });

    // 튜닝 파라미터 리스너
    this.confidenceSlider.addEventListener("input", () => {
      const val = parseFloat(this.confidenceSlider.value);
      this.confidenceVal.textContent = val.toFixed(2);
      this.detector.setConfidence(val);
      localStorage.setItem("cfg_confidence", String(val));
    });

    this.tileSelect.addEventListener("change", () => {
      this.detector.tileMode = this.tileSelect.value;
      localStorage.setItem("cfg_tile", this.tileSelect.value);
    });

    this.speedSelect.addEventListener("change", () => {
      raffleEngine.setSpeed(this.speedSelect.value);
      localStorage.setItem("cfg_speed", this.speedSelect.value);
    });

    this.preprocessBright.addEventListener("change", () => {
      this.detector.usePreprocess = this.preprocessBright.checked;
      localStorage.setItem("cfg_preprocess", String(this.preprocessBright.checked));
    });

    // 인원 스테퍼
    this.pickCountInput.addEventListener("input", () => this.updateDrawButtonState());
    this.$("countMinus").addEventListener("click", () => this.stepInput(this.pickCountInput, -1));
    this.$("countPlus").addEventListener("click", () => this.stepInput(this.pickCountInput, 1));

    // 추첨 실행 & 초기화 버튼
    this.drawBtn.addEventListener("click", () => this.startRaffle());
    this.resetBtn.addEventListener("click", () => this.resetRaffle());

    // 전체화면
    this.$("fullscreenBtn").addEventListener("click", () => {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
      else document.exitFullscreen?.();
    });

    // 모드 전환
    this.$("modeCameraBtn").addEventListener("click", () => this.setMode("camera"));
    this.$("modeManualBtn").addEventListener("click", () => this.setMode("manual"));
  }

  stepInput(input, delta) {
    const min = Number(input.min || 1);
    const max = input.max ? Number(input.max) : Infinity;
    input.value = Math.min(max, Math.max(min, Number(input.value) + delta));
    input.dispatchEvent(new Event("input"));
  }

  updateDrawButtonState() {
    if (this.isDrawing) return;
    const candidates = this.tracker.getRaffleCandidates();
    this.pickCountInput.max = Math.max(1, candidates.length);
    this.drawBtn.disabled = candidates.length < 1 || Number(this.pickCountInput.value) > candidates.length;
  }

  setMode(mode) {
    const isCam = mode === "camera";
    this.$("cameraMode").classList.toggle("is-active", isCam);
    this.$("manualMode").classList.toggle("is-active", !isCam);
    this.$("modeCameraBtn").classList.toggle("is-active", isCam);
    this.$("modeManualBtn").classList.toggle("is-active", !isCam);

    if (isCam && !this.stream) this.startCamera();
  }

  async init() {
    const savedConf = parseFloat(localStorage.getItem("cfg_confidence") || "0.35");
    this.confidenceSlider.value = String(savedConf);
    this.confidenceVal.textContent = savedConf.toFixed(2);

    const savedTile = localStorage.getItem("cfg_tile") || "2x2";
    this.tileSelect.value = savedTile;
    this.detector.tileMode = savedTile;

    const savedSpeed = localStorage.getItem("cfg_speed") || "normal";
    this.speedSelect.value = savedSpeed;
    raffleEngine.setSpeed(savedSpeed);

    const savedPre = localStorage.getItem("cfg_preprocess") !== "false";
    this.preprocessBright.checked = savedPre;
    this.detector.usePreprocess = savedPre;

    await this.detector.init(savedConf);
    await this.startCamera();
  }
}

window.addEventListener("DOMContentLoaded", () => {
  const app = new App();
  app.init();
});
