/* ============================================================
   detector.js · MediaPipe FaceDetector 엔진 및 타일 탐지/NMS
   - 공식 CDN: jsdelivr & Google Storage
   - 오프라인 상대경로 변경 시:
     MODEL_URL = "./models/blaze_face_short_range.tflite";
     WASM_URL = "./wasm";
   ============================================================ */

import {
  FaceDetector,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20";

const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.20/wasm";

export class FaceDetectionEngine {
  constructor() {
    this.detector = null;
    this.minConfidence = 0.35;
    this.tileMode = "2x2"; // "1x1" | "2x2" | "3x2"
    this.usePreprocess = true;

    // 탐지 전용 오프스크린 캔버스
    this.detectCanvas = document.createElement("canvas");
    this.detectCtx = this.detectCanvas.getContext("2d", { willReadFrequently: true });
  }

  async init(minConfidence = 0.35) {
    this.minConfidence = minConfidence;
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    try {
      this.detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: this.minConfidence,
        minSuppressionThreshold: 0.3
      });
    } catch (gpuErr) {
      console.warn("GPU Delegate 초기화 실패, CPU로 폴백합니다.", gpuErr);
      this.detector = await FaceDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "IMAGE",
        minDetectionConfidence: this.minConfidence,
        minSuppressionThreshold: 0.3
      });
    }
  }

  setConfidence(val) {
    this.minConfidence = val;
    // 인스턴스 옵션 갱신
    if (this.detector && this.detector.setOptions) {
      this.detector.setOptions({ minDetectionConfidence: val });
    }
  }

  // 20% 오버랩 타일 좌표 생성
  generateTiles(videoW, videoH) {
    if (this.tileMode === "1x1") {
      return [{ x: 0, y: 0, w: videoW, h: videoH }];
    }

    const tiles = [];
    const cols = this.tileMode === "3x2" ? 3 : 2;
    const rows = 2;
    const overlap = 0.20;

    const stepX = (videoW * (1 - overlap)) / (cols - overlap);
    const tileW = stepX / (1 - overlap);
    const stepY = (videoH * (1 - overlap)) / (rows - overlap);
    const tileH = stepY / (1 - overlap);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const tx = Math.min(c * (tileW * (1 - overlap)), videoW - tileW);
        const ty = Math.min(r * (tileH * (1 - overlap)), videoH - tileH);
        tiles.push({
          x: Math.round(tx),
          y: Math.round(ty),
          w: Math.round(tileW),
          h: Math.round(tileH)
        });
      }
    }
    return tiles;
  }

  // IoU 계산
  calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
  }

  // NMS (Non-Maximum Suppression)
  applyNMS(boxes, iouThreshold = 0.4) {
    if (boxes.length <= 1) return boxes;
    const sorted = [...boxes].sort((a, b) => (b.score || 0) - (a.score || 0));
    const selected = [];

    for (const box of sorted) {
      let overlap = false;
      for (const chosen of selected) {
        if (this.calculateIoU(box, chosen) > iouThreshold) {
          overlap = true;
          break;
        }
      }
      if (!overlap) selected.push(box);
    }
    return selected;
  }

  // 허공/사물 오탐지 방지 기하 필터
  isValidFaceBox(box, videoW, videoH, keypoints = []) {
    // 1. 최소 크기: 영상 폭의 2% 미만 제외
    if (box.w < videoW * 0.02 || box.h < videoH * 0.02) return false;

    // 2. 종횡비: 0.6 ~ 1.6 이외 제외
    const aspect = box.w / box.h;
    if (aspect < 0.6 || aspect > 1.6) return false;

    // 3. 키포인트(눈 2개, 코, 입) 검증: 4개 중 2개 이상이 박스 내부에 있어야 함
    if (keypoints && keypoints.length >= 4) {
      let inCount = 0;
      for (let i = 0; i < 4; i++) {
        const kp = keypoints[i];
        if (!kp) continue;
        const kx = kp.x * videoW;
        const ky = kp.y * videoH;
        if (kx >= box.x && kx <= box.x + box.w && ky >= box.y && ky <= box.y + box.h) {
          inCount++;
        }
      }
      if (inCount < 2) return false;
    }

    return true;
  }

  // 전체 프레임 타일 탐지 실행
  detect(videoElement) {
    if (!this.detector || videoElement.readyState < 2) return [];

    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;
    const tiles = this.generateTiles(vw, vh);
    const candidates = [];

    tiles.forEach((tile) => {
      this.detectCanvas.width = tile.w;
      this.detectCanvas.height = tile.h;

      // 탐지 전용 버퍼 밝기/대비 전처리
      if (this.usePreprocess) {
        this.detectCtx.filter = "brightness(1.2) contrast(1.15)";
      } else {
        this.detectCtx.filter = "none";
      }

      this.detectCtx.drawImage(videoElement, tile.x, tile.y, tile.w, tile.h, 0, 0, tile.w, tile.h);

      try {
        const res = this.detector.detect(this.detectCanvas);
        if (res.detections && res.detections.length > 0) {
          res.detections.forEach((det) => {
            const bb = det.boundingBox;
            const globalBox = {
              x: tile.x + bb.originX,
              y: tile.y + bb.originY,
              w: bb.width,
              h: bb.height,
              score: det.categories?.[0]?.score || 0.5
            };

            const keypoints = (det.keypoints || []).map((kp) => ({
              x: (tile.x + kp.x * tile.w) / vw,
              y: (tile.y + kp.y * tile.h) / vh
            }));

            if (this.isValidFaceBox(globalBox, vw, vh, keypoints)) {
              candidates.push({ ...globalBox, keypoints });
            }
          });
        }
      } catch (err) {
        console.warn("Tile detection error:", err);
      }
    });

    return this.applyNMS(candidates, 0.4);
  }
}
