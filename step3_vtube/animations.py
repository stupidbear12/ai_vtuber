"""
animations.py - emeth 캐릭터 전체 리깅 애니메이션 엔진

6가지 애니메이션 세트:
  idle      - 평상시 대기 (호흡, 눈깜빡임, 미세 머리 흔들림)
  opening   - 오프닝 연출 (고개 들어올리기, 눈 뜨기)
  speaking  - 말하기 (립싱크 + 자연스러운 움직임)
  bible     - 성경 구절 읽기 (눈 감고 고개 숙임)
  emotional - 감동/감사 표현 (눈 웃음, 고개 기울임)
  closing   - 클로징 (고개 숙이고 눈 감은 뒤 인사)
"""

import asyncio
import math
import random
import time
from typing import Optional

from config_vtube import (
    PARAM_FACE_ANGLE_X, PARAM_FACE_ANGLE_Y, PARAM_FACE_ANGLE_Z,
    PARAM_EYE_OPEN_L, PARAM_EYE_OPEN_R,
    PARAM_EYE_SMILE_L, PARAM_EYE_SMILE_R,
    PARAM_BROW_L, PARAM_BROW_R,
    PARAM_BROW_FORM_L, PARAM_BROW_FORM_R,
    PARAM_MOUTH_OPEN, PARAM_MOUTH_FORM,
    PARAM_BODY_ANGLE_X, PARAM_BODY_ANGLE_Y, PARAM_BREATH,
    IDLE_BREATH_PERIOD, IDLE_BLINK_MIN, IDLE_BLINK_MAX,
    IDLE_BLINK_DURATION, IDLE_HEAD_AMPLITUDE, IDLE_HEAD_PERIOD,
    IDLE_BODY_AMPLITUDE, SPEAKING_EYE_SQUINT, SPEAKING_BODY_LEAN,
    BIBLE_BREATH_PERIOD, BIBLE_EYE_CLOSE, BIBLE_HEAD_DOWN, BIBLE_MOUTH_MAX,
    CLOSING_HEAD_DOWN, CLOSING_DURATION,
)


# ─────────────────────────────────────────────
# 보간 유틸리티
# ─────────────────────────────────────────────

def lerp(start: float, end: float, t: float) -> float:
    """선형 보간"""
    return start + (end - start) * max(0.0, min(1.0, t))


def ease_in_out(t: float) -> float:
    """부드러운 ease in/out 곡선 (3차 에르미트)"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    """ease out (감속)"""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 2


# ─────────────────────────────────────────────
# AnimationEngine
# ─────────────────────────────────────────────

class AnimationEngine:
    def __init__(self, vts_api):
        """
        vts_api: pyvts VTSPlugin 인스턴스
                 (inject_parameter_values 메서드를 가진 객체)
        """
        self.api = vts_api
        self._idle_task: Optional[asyncio.Task] = None
        self._current_mode = "idle"
        self._stop_idle = False

    # ── 파라미터 주입 헬퍼 ─────────────────────────────────

    async def _set(self, param: str, value: float):
        """단일 파라미터 즉시 설정"""
        try:
            await self.api.request(
                self.api.vts_request.requestSetParameterValue(
                    param, value, face_found=False
                )
            )
        except Exception:
            pass

    async def _set_multi(self, params: dict):
        """여러 파라미터 동시 설정"""
        try:
            keys   = list(params.keys())
            values = list(params.values())
            await self.api.request(
                self.api.vts_request.requestSetMultiParameterValue(
                    keys, values, face_found=False
                )
            )
        except Exception:
            pass

    async def _animate(self, param: str, start: float, end: float,
                       duration: float, easing=ease_in_out):
        """단일 파라미터를 duration초 동안 부드럽게 변화"""
        steps = max(int(duration * 30), 2)
        for i in range(steps + 1):
            t = easing(i / steps)
            await self._set(param, lerp(start, end, t))
            await asyncio.sleep(duration / steps)

    async def _animate_multi(self, keyframes: list[dict]):
        """
        여러 파라미터를 동시에 애니메이션
        keyframes: [{"param": str, "start": float, "end": float, "duration": float}, ...]
        """
        tasks = [
            self._animate(kf["param"], kf["start"], kf["end"], kf["duration"],
                          kf.get("easing", ease_in_out))
            for kf in keyframes
        ]
        await asyncio.gather(*tasks)

    # ── 눈 깜빡임 ─────────────────────────────────────────

    async def _blink_once(self):
        """눈 한 번 깜빡임"""
        d = IDLE_BLINK_DURATION
        await self._animate_multi([
            {"param": PARAM_EYE_OPEN_L, "start": 1.0, "end": 0.0, "duration": d * 0.4, "easing": ease_in_out},
            {"param": PARAM_EYE_OPEN_R, "start": 1.0, "end": 0.0, "duration": d * 0.4, "easing": ease_in_out},
        ])
        await asyncio.sleep(d * 0.2)
        await self._animate_multi([
            {"param": PARAM_EYE_OPEN_L, "start": 0.0, "end": 1.0, "duration": d * 0.4, "easing": ease_out},
            {"param": PARAM_EYE_OPEN_R, "start": 0.0, "end": 1.0, "duration": d * 0.4, "easing": ease_out},
        ])

    # ── 아이들 루프 ───────────────────────────────────────

    async def _idle_loop(self, breath_period: float = None):
        """아이들 상태: 호흡 + 눈깜빡임 + 머리 흔들림 루프"""
        bp = breath_period or IDLE_BREATH_PERIOD
        self._stop_idle = False
        t_start = time.time()
        next_blink = time.time() + random.uniform(IDLE_BLINK_MIN, IDLE_BLINK_MAX)

        while not self._stop_idle:
            now = time.time()
            elapsed = now - t_start

            # 호흡 (사인파)
            breath_val = (math.sin(2 * math.pi * elapsed / bp) + 1) / 2
            # 머리 미세 흔들림
            head_x = IDLE_HEAD_AMPLITUDE * math.sin(2 * math.pi * elapsed / IDLE_HEAD_PERIOD)
            head_y = IDLE_HEAD_AMPLITUDE * 0.5 * math.sin(2 * math.pi * elapsed / (IDLE_HEAD_PERIOD * 1.3) + 1.0)
            # 몸 호흡 연동
            body_y = IDLE_BODY_AMPLITUDE * breath_val

            await self._set_multi({
                PARAM_BREATH:       breath_val,
                PARAM_FACE_ANGLE_X: head_x,
                PARAM_FACE_ANGLE_Y: head_y,
                PARAM_BODY_ANGLE_Y: body_y,
            })

            # 눈 깜빡임
            if now >= next_blink:
                await self._blink_once()
                next_blink = time.time() + random.uniform(IDLE_BLINK_MIN, IDLE_BLINK_MAX)

            await asyncio.sleep(1 / 30)

    def _start_idle(self, breath_period: float = None):
        """아이들 루프 백그라운드 시작"""
        self.stop_idle()
        self._idle_task = asyncio.get_running_loop().create_task(
            self._idle_loop(breath_period)
        )

    def stop_idle(self):
        """아이들 루프 정지"""
        self._stop_idle = True
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = None

    # ── 6가지 애니메이션 세트 ────────────────────────────

    async def play_idle(self):
        """평상시 대기 - 자연스러운 루프"""
        self._start_idle()

    async def play_opening(self):
        """오프닝 연출 - 고개 들어올리며 눈 뜨기"""
        self.stop_idle()
        # 초기 상태: 고개 숙이고 눈 감음
        await self._set_multi({
            PARAM_FACE_ANGLE_Y: -10.0,
            PARAM_EYE_OPEN_L: 0.0,
            PARAM_EYE_OPEN_R: 0.0,
            PARAM_MOUTH_FORM: 0.0,
        })
        await asyncio.sleep(0.5)

        # 고개 천천히 들어올리기 + 눈 뜨기 (동시)
        await self._animate_multi([
            {"param": PARAM_FACE_ANGLE_Y, "start": -10.0, "end": 0.0,  "duration": 2.0},
            {"param": PARAM_EYE_OPEN_L,   "start": 0.0,   "end": 1.0,  "duration": 1.5},
            {"param": PARAM_EYE_OPEN_R,   "start": 0.0,   "end": 1.0,  "duration": 1.5},
        ])
        await asyncio.sleep(0.3)

        # 살짝 미소
        await self._animate(PARAM_MOUTH_FORM, 0.0, 0.3, 0.8)
        await asyncio.sleep(0.5)

        # 아이들로 전환
        self._start_idle()

    async def play_speaking(self, mouth_value: float = 0.0):
        """말하기 모드 - 립싱크 파라미터 설정 (외부에서 반복 호출)"""
        squint = SPEAKING_EYE_SQUINT
        await self._set_multi({
            PARAM_MOUTH_OPEN:   mouth_value,
            PARAM_EYE_OPEN_L:   squint,
            PARAM_EYE_OPEN_R:   squint,
            PARAM_BODY_ANGLE_Y: SPEAKING_BODY_LEAN if mouth_value > 0.2 else 0.0,
        })

    async def play_speaking_emphasis(self):
        """강조 표현 - 눈썹 살짝 올라감"""
        await self._animate_multi([
            {"param": PARAM_BROW_L, "start": 0.0, "end": 0.2, "duration": 0.3},
            {"param": PARAM_BROW_R, "start": 0.0, "end": 0.2, "duration": 0.3},
        ])
        await asyncio.sleep(0.5)
        await self._animate_multi([
            {"param": PARAM_BROW_L, "start": 0.2, "end": 0.0, "duration": 0.4},
            {"param": PARAM_BROW_R, "start": 0.2, "end": 0.0, "duration": 0.4},
        ])

    async def play_bible_mode(self):
        """성경 구절 읽기 - 눈 감고 고개 숙임"""
        self.stop_idle()
        await self._animate_multi([
            {"param": PARAM_EYE_OPEN_L,   "start": 1.0, "end": BIBLE_EYE_CLOSE, "duration": 1.5},
            {"param": PARAM_EYE_OPEN_R,   "start": 1.0, "end": BIBLE_EYE_CLOSE, "duration": 1.5},
            {"param": PARAM_FACE_ANGLE_Y, "start": 0.0, "end": BIBLE_HEAD_DOWN,  "duration": 1.5},
        ])
        # 느린 호흡 아이들로 전환
        self._start_idle(breath_period=BIBLE_BREATH_PERIOD)

    async def play_emotional(self):
        """감동/감사 표현 - 눈 웃음 + 고개 기울임"""
        self.stop_idle()
        await self._animate_multi([
            {"param": PARAM_EYE_SMILE_L,  "start": 0.0, "end": 0.7, "duration": 1.0},
            {"param": PARAM_EYE_SMILE_R,  "start": 0.0, "end": 0.7, "duration": 1.0},
            {"param": PARAM_BROW_L,       "start": 0.0, "end": 0.15,"duration": 0.8},
            {"param": PARAM_BROW_R,       "start": 0.0, "end": 0.15,"duration": 0.8},
            {"param": PARAM_FACE_ANGLE_Z, "start": 0.0, "end": 5.0, "duration": 1.0},
            {"param": PARAM_MOUTH_FORM,   "start": 0.0, "end": 0.4, "duration": 0.8},
        ])
        await asyncio.sleep(1.5)
        # 원상복구 + 아이들
        await self._animate_multi([
            {"param": PARAM_EYE_SMILE_L,  "start": 0.7, "end": 0.0, "duration": 1.0},
            {"param": PARAM_EYE_SMILE_R,  "start": 0.7, "end": 0.0, "duration": 1.0},
            {"param": PARAM_FACE_ANGLE_Z, "start": 5.0, "end": 0.0, "duration": 1.0},
            {"param": PARAM_MOUTH_FORM,   "start": 0.4, "end": 0.0, "duration": 0.8},
        ])
        self._start_idle()

    async def play_closing(self):
        """클로징 - 고개 숙이고 눈 감기 → 마지막 인사"""
        self.stop_idle()
        # 고개 숙이며 눈 감기
        await self._animate_multi([
            {"param": PARAM_FACE_ANGLE_Y, "start": 0.0,  "end": CLOSING_HEAD_DOWN, "duration": CLOSING_DURATION},
            {"param": PARAM_EYE_OPEN_L,   "start": 1.0,  "end": 0.0,               "duration": CLOSING_DURATION},
            {"param": PARAM_EYE_OPEN_R,   "start": 1.0,  "end": 0.0,               "duration": CLOSING_DURATION},
        ])
        await asyncio.sleep(1.0)
        # 마지막 인사: 눈 뜨고 미소
        await self._animate_multi([
            {"param": PARAM_FACE_ANGLE_Y, "start": CLOSING_HEAD_DOWN, "end": 0.0,  "duration": 1.2},
            {"param": PARAM_EYE_OPEN_L,   "start": 0.0,               "end": 1.0,  "duration": 1.0},
            {"param": PARAM_EYE_OPEN_R,   "start": 0.0,               "end": 1.0,  "duration": 1.0},
            {"param": PARAM_EYE_SMILE_L,  "start": 0.0,               "end": 0.5,  "duration": 1.2},
            {"param": PARAM_EYE_SMILE_R,  "start": 0.0,               "end": 0.5,  "duration": 1.2},
            {"param": PARAM_MOUTH_FORM,   "start": 0.0,               "end": 0.35, "duration": 1.0},
        ])
        await asyncio.sleep(2.0)
        self._start_idle()

    async def play_animation(self, name: str, **kwargs):
        """이름으로 애니메이션 실행"""
        mapping = {
            "idle":      self.play_idle,
            "opening":   self.play_opening,
            "bible":     self.play_bible_mode,
            "emotional": self.play_emotional,
            "closing":   self.play_closing,
        }
        fn = mapping.get(name)
        if fn:
            await fn(**kwargs)
        else:
            print(f"[경고] 알 수 없는 애니메이션: {name}")
