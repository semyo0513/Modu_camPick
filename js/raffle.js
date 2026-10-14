/* ============================================================
   raffle.js · 박진감 넘치는 룰렛 추첨 엔진 및 연출
   - 지수 감속 룰렛 순회 애니메이션
   - 실시간 오디오 틱 & 당첨 팡파레/셔터음 동기화
   - 당첨자 썸네일 크롭 및 화면 플래시
   - 다인원 연속 추첨 시 각 라운드별 당첨자 확정 연출
   ============================================================ */

import { soundFx } from "./audio.js";

// 추첨 속도 프로파일 (ms 단위 딜레이)
export const SPEED_PRESETS = {
  fast: [45, 45, 55, 65, 80, 105, 140, 190, 260, 350],
  normal: [60, 60, 70, 80, 95, 115, 140, 175, 220, 280, 360, 460],
  thrill: [50, 50, 55, 65, 75, 90, 110, 135, 170, 215, 270, 340, 430, 540, 680]
};

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// 암호학적 난수 비복원추출 (Fisher-Yates)
export function pickRandomUnique(list, count) {
  const pool = [...list];
  const picked = [];
  const n = Math.min(count, pool.length);
  for (let i = 0; i < n; i++) {
    const rand = new Uint32Array(1);
    crypto.getRandomValues(rand);
    const idx = rand[0] % pool.length;
    picked.push(pool[idx]);
    pool.splice(idx, 1);
  }
  return picked;
}

export class RaffleEngine {
  constructor() {
    this.speedMode = "normal";
    this.history = []; // 역대 당첨자 ID 목록
  }

  setSpeed(mode) {
    if (SPEED_PRESETS[mode]) this.speedMode = mode;
  }

  // 카메라 모드 룰렛 추첨 실행
  async runCameraDraw({
    candidates,
    pickCount,
    onStep,
    onWinnerLocked,
    onComplete
  }) {
    if (!candidates || candidates.length === 0 || pickCount > candidates.length) {
      return null;
    }

    const delays = SPEED_PRESETS[this.speedMode] || SPEED_PRESETS.normal;
    // 화면상 왼쪽에서 오른쪽 순으로 후보 정렬
    const sortedCandidates = [...candidates].sort((a, b) => a.box.x - b.box.x);
    const winners = pickRandomUnique(sortedCandidates, pickCount);

    const lockedIds = new Set();
    let currentPool = [...sortedCandidates];

    // 라운드별로 당첨자 한 명씩 순차 룰렛 연출
    for (let round = 0; round < winners.length; round++) {
      const winner = winners[round];
      const targetIdx = currentPool.findIndex((c) => c.id === winner.id);
      const totalSteps = delays.length;

      for (let k = 0; k < totalSteps; k++) {
        const stepRatio = k / (totalSteps - 1);
        const stepsFromEnd = totalSteps - 1 - k;
        const curIdx = ((targetIdx - stepsFromEnd) % currentPool.length + currentPool.length) % currentPool.length;
        const activeTrack = currentPool[curIdx];

        // 룰렛 틱 사운드 재생
        soundFx.playTick(stepRatio);

        // 번쩍이는 하이라이트 색상 교차
        const flashColor = k % 2 === 0 ? "#F5B942" : "#FFFFFF";

        // 화면 갱신 콜백 호출
        if (onStep) {
          onStep({
            activeId: activeTrack.id,
            flashColor,
            lockedIds,
            stepIndex: k,
            totalSteps
          });
        }

        // 지연 시간 대기 + 화면 리프레시 보장
        await sleep(delays[k]);
      }

      // 당첨자 1명 확정 연출
      lockedIds.add(winner.id);
      this.history.push({ id: winner.id, time: new Date() });
      currentPool = currentPool.filter((c) => c.id !== winner.id);

      soundFx.playShutter();

      if (onWinnerLocked) {
        onWinnerLocked({ winner, lockedIds, roundIndex: round, totalWinners: winners.length });
      }

      // 다음 라운드 전 잠시 정적 (긴장감)
      if (round < winners.length - 1) {
        await sleep(500);
      }
    }

    // 최종 축하 팡파레
    soundFx.playFanfare();

    if (onComplete) {
      onComplete({ winners, lockedIds });
    }

    return winners;
  }

  // 캡처된 원본 캔버스에서 특정 얼굴 썸네일 잘라내기
  cropThumbnail(frameCanvas, track, outSize = 160) {
    const out = document.createElement("canvas");
    out.width = outSize;
    out.height = outSize;
    const ctx = out.getContext("2d");

    // 얼굴 주변 40% 여백 포함
    const padX = track.box.w * 0.4;
    const padY = track.box.h * 0.4;
    const sx = Math.max(0, track.box.x - padX);
    const sy = Math.max(0, track.box.y - padY);
    const sw = Math.min(frameCanvas.width - sx, track.box.w + padX * 2);
    const sh = Math.min(frameCanvas.height - sy, track.box.h + padY * 2);

    ctx.fillStyle = "#12161F";
    ctx.fillRect(0, 0, outSize, outSize);
    ctx.drawImage(frameCanvas, sx, sy, sw, sh, 0, 0, outSize, outSize);
    return out.toDataURL("image/png");
  }
}

export const raffleEngine = new RaffleEngine();
