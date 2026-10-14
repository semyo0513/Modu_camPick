/* ============================================================
   tracker.js · 얼굴 트래킹, 지수평활 필터 및 상태 머신
   - 연속 3프레임 이상 탐지 시 "확정 얼굴"로 승격
   - 6프레임(약 0.1~0.2초) 매칭 실패 시 즉시 소멸하여 잔상 제거
   - 지수평활 필터 (새 값 0.4 : 이전 값 0.6)
   - 결석자/추첨 제외(isExcluded) 토글 지원
   - 교실 30명 좌석 일괄 배치 & 빈자리 자동 채우기 지원
   ============================================================ */

export class FaceTracker {
  constructor() {
    this.tracks = [];
    this.nextId = 1;
    this.isLocked = false;
  }

  // AI 탐지 결과와 기존 트랙 매칭 및 갱신
  update(detections) {
    if (this.isLocked) return;

    const usedIndices = new Set();

    // 1. 기존 트랙 위치 보정
    this.tracks.forEach((track) => {
      if (track.isManual) return; // 수동 박스는 보존

      let bestIdx = -1;
      let bestDist = Infinity;

      detections.forEach((det, idx) => {
        if (usedIndices.has(idx)) return;
        const dist = Math.hypot(
          (det.x + det.w / 2) - (track.box.x + track.box.w / 2),
          (det.y + det.h / 2) - (track.box.y + track.box.h / 2)
        );
        const maxDist = Math.max(track.box.w, track.box.h) * 0.9;
        if (dist < maxDist && dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      });

      if (bestIdx !== -1) {
        const matched = detections[bestIdx];
        // 지수평활 필터
        track.box.x = track.box.x * 0.6 + matched.x * 0.4;
        track.box.y = track.box.y * 0.6 + matched.y * 0.4;
        track.box.w = track.box.w * 0.6 + matched.w * 0.4;
        track.box.h = track.box.h * 0.6 + matched.h * 0.4;
        track.keypoints = matched.keypoints;
        track.matchedFrames++;
        track.missedFrames = 0;
        usedIndices.add(bestIdx);
      } else {
        track.missedFrames++;
      }
    });

    // 2. 새로운 얼굴 트랙 등록
    detections.forEach((det, idx) => {
      if (usedIndices.has(idx)) return;
      this.tracks.push({
        id: this.nextId++,
        box: { x: det.x, y: det.y, w: det.w, h: det.h },
        matchedFrames: 1,
        missedFrames: 0,
        isManual: false,
        isExcluded: false,
        keypoints: det.keypoints
      });
    });

    // 3. 소멸 조건: 6프레임 매칭 실패 시 즉각 제거 (잔상 제거)
    this.tracks = this.tracks.filter(
      (t) => t.isManual || t.missedFrames <= 6
    );
  }

  // 연속 3프레임 이상 탐지된 확정 얼굴 목록 반환
  getConfirmedTracks() {
    return this.tracks.filter(
      (t) => (t.isManual || t.matchedFrames >= 3)
    );
  }

  // 추첨 가능한 후보 목록 (확정 얼굴 중 결석/제외자 제외)
  getRaffleCandidates() {
    return this.getConfirmedTracks().filter((t) => !t.isExcluded);
  }

  // 특정 ID의 결석/제외 상태 토글
  toggleExclusion(id) {
    const track = this.tracks.find((t) => t.id === id);
    if (track) {
      track.isExcluded = !track.isExcluded;
      return track.isExcluded;
    }
    return false;
  }

  // 수동 얼굴 박스 추가
  addManualTrack(x, y, w, h) {
    this.tracks.push({
      id: this.nextId++,
      box: { x, y, w, h },
      matchedFrames: 10,
      missedFrames: 0,
      isManual: true,
      isExcluded: false,
      keypoints: []
    });
  }

  // 특정 위치에 근접한 박스 삭제
  removeAt(x, y, radius = 40) {
    const idx = this.tracks.findIndex((t) => {
      const cx = t.box.x + t.box.w / 2;
      const cy = t.box.y + t.box.h / 2;
      return Math.hypot(x - cx, y - cy) < Math.max(t.box.w, t.box.h, radius);
    });
    if (idx !== -1) {
      this.tracks.splice(idx, 1);
      return true;
    }
    return false;
  }

  // 교실 30명 좌석 일괄 배치 (5행 6열 원근 그리드)
  create30Grid(videoW, videoH) {
    this.tracks = [];
    this.nextId = 1;
    const rows = 5;
    const cols = 6;

    for (let r = 0; r < rows; r++) {
      const boxW = videoW * (0.045 + (r / 4) * 0.025);
      const boxH = videoH * (0.065 + (r / 4) * 0.035);
      const y = videoH * 0.16 + (r / 4) * (videoH * 0.64) - boxH / 2;

      for (let c = 0; c < cols; c++) {
        const x = videoW * 0.08 + (c / 5) * (videoW * 0.84) - boxW / 2;
        this.addManualTrack(x, y, boxW, boxH);
      }
    }
    this.isLocked = true;
  }

  // 미인식 빈자리 30명 채우기
  fillTo30(videoW, videoH) {
    const confirmed = this.getConfirmedTracks();
    if (confirmed.length >= 30) return false;

    const rows = 5;
    const cols = 6;
    const candidates = [];

    for (let r = 0; r < rows; r++) {
      const boxW = videoW * (0.045 + (r / 4) * 0.025);
      const boxH = videoH * (0.065 + (r / 4) * 0.035);
      const y = videoH * 0.16 + (r / 4) * (videoH * 0.64) - boxH / 2;
      for (let c = 0; c < cols; c++) {
        const x = videoW * 0.08 + (c / 5) * (videoW * 0.84) - boxW / 2;
        candidates.push({ x, y, w: boxW, h: boxH });
      }
    }

    // 기존 박스와 거리가 먼 빈 슬롯 우선 채우기
    const scored = candidates.map((cand) => {
      let minDist = Infinity;
      const cx = cand.x + cand.w / 2;
      const cy = cand.y + cand.h / 2;
      confirmed.forEach((t) => {
        const dist = Math.hypot(cx - (t.box.x + t.box.w / 2), cy - (t.box.y + t.box.h / 2));
        if (dist < minDist) minDist = dist;
      });
      return { cand, minDist };
    });
    scored.sort((a, b) => b.minDist - a.minDist);

    const needed = 30 - confirmed.length;
    for (let i = 0; i < needed && i < scored.length; i++) {
      const c = scored[i].cand;
      this.addManualTrack(c.x, c.y, c.w, c.h);
    }
    this.isLocked = true;
    return true;
  }

  clear() {
    this.tracks = [];
    this.nextId = 1;
    this.isLocked = false;
  }
}
