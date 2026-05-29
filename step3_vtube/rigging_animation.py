"""
rigging_animation.py - emeth 전체 리깅 애니메이션 레이어 모듈

각 애니메이션 클래스는 독립적인 asyncio.Task로 실행되며,
공유 LayeredParamState에 파라미터를 기록합니다.
animation_controller.py의 write loop가 이 상태를 읽어
VTubeStudio에 30fps로 주입합니다.

클래스 목록:
  LayeredParamState     - 우선순위 기반 파라미터 병합 상태
  BreathingAnimation    - 흉부 사인파 호흡 (1회/4초)
  NaturalBlinkAnimation - 자연스러운 눈 깜빡임 (랜덤 간격 + 연속 깜빡임)
  HeadMovement          - 다중 주파수 합성 머리 흔들림
  HairPhysics           - 머리 이동 반응 스프링 물리
  EmotionAnimation      - 감정 상태별 표정 보간 전환
  TalkingAnimation      - 말하기 중 눈썹/눈 세부 움직임
"""

import asyncio
import math
import random
import time
from typing import Dict, Optional

from config_vtube import (
    PARAM_FACE_ANGLE_X, PARAM_FACE_ANGLE_Y, PARAM_FACE_ANGLE_Z,
    PARAM_EYE_OPEN_L, PARAM_EYE_OPEN_R,
    PARAM_EYE_SMILE_L, PARAM_EYE_SMILE_R,
    PARAM_BROW_L, PARAM_BROW_R,
    PARAM_BROW_FORM_L, PARAM_BROW_FORM_R,
    PARAM_MOUTH_OPEN, PARAM_MOUTH_FORM,
    PARAM_BODY_ANGLE_X, PARAM_BODY_ANGLE_Y, PARAM_BREATH,
    PARAM_HAIR_SIDE, PARAM_HAIR_BACK,
    IDLE_BREATH_PERIOD, IDLE_BLINK_MIN, IDLE_BLINK_MAX,
    IDLE_BLINK_DURATION, IDLE_HEAD_AMPLITUDE, IDLE_HEAD_PERIOD,
    IDLE_BODY_AMPLITUDE, SPEAKING_EYE_SQUINT,
)

# ─── 상수 ────────────────────────────────────────────────────
FPS        = 30
FRAME_TIME = 1.0 / FPS

# 레이어 우선순위 (높을수록 나중에 적용되어 하위 레이어를 덮어씀)
PRIORITY_BREATH   = 10
PRIORITY_HEAD     = 15
PRIORITY_HAIR     = 12
PRIORITY_EMOTION  = 20
PRIORITY_TALKING  = 25
PRIORITY_BLINK    = 30   # 깜빡임은 말하기 레이어보다 높아야 함
PRIORITY_REACTION = 40   # 반응 애니메이션은 최우선


# ─── 보간 유틸 ───────────────────────────────────────────────

def lerp(a: float, b: float, t: float) -> float:
    """선형 보간 (t 범위: 0.0 ~ 1.0)"""
    return a + (b - a) * max(0.0, min(1.0, t))


def ease_in_out(t: float) -> float:
    """3차 에르미트 ease in/out"""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def ease_out(t: float) -> float:
    """2차 ease out (감속)"""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 2


# ─── 공유 파라미터 상태 ──────────────────────────────────────

class LayeredParamState:
    """
    여러 애니메이션 레이어의 파라미터를 우선순위 기반으로 병합합니다.

    같은 파라미터를 여러 레이어가 설정하면 priority가 높은 레이어가 최종값을 결정합니다.
    (낮은 priority → 먼저 적용, 높은 priority → 덮어씀)
    """

    def __init__(self):
        self._layers: Dict[str, Dict[str, float]] = {}
        self._priorities: Dict[str, int] = {}

    def set_layer(self, layer_id: str, params: Dict[str, float], priority: int = 0):
        """레이어 파라미터 설정 또는 업데이트"""
        self._layers[layer_id] = params
        self._priorities[layer_id] = priority

    def remove_layer(self, layer_id: str):
        """레이어 제거"""
        self._layers.pop(layer_id, None)
        self._priorities.pop(layer_id, None)

    def get_layer_values(self, layer_id: str) -> Dict[str, float]:
        """특정 레이어의 현재 파라미터 값 반환 (없으면 빈 dict)"""
        return dict(self._layers.get(layer_id, {}))

    def get_merged(self) -> Dict[str, float]:
        """
        모든 레이어를 priority 순서대로 병합해 최종 파라미터 값 반환.
        낮은 priority를 먼저 적용하고, 높은 priority가 같은 키를 덮어씁니다.
        """
        sorted_ids = sorted(
            self._layers.keys(),
            key=lambda k: self._priorities.get(k, 0),
        )
        result: Dict[str, float] = {}
        for lid in sorted_ids:
            result.update(self._layers[lid])
        return result

    def clear(self):
        """모든 레이어 초기화"""
        self._layers.clear()
        self._priorities.clear()


# ─── 기반 클래스 ─────────────────────────────────────────────

class BaseAnimation:
    """모든 애니메이션 레이어의 기반 클래스"""
    LAYER_ID: str = "base"
    PRIORITY: int = 0

    def __init__(self, state: LayeredParamState):
        self.state = state
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def start(self):
        """애니메이션 태스크 시작 (이미 실행 중이면 무시)"""
        if self._task and not self._task.done():
            return
        self._running = True
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())

    def stop(self):
        """애니메이션 태스크 정지 및 레이어 제거"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        self.state.remove_layer(self.LAYER_ID)

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self):
        raise NotImplementedError


# ─── BreathingAnimation ──────────────────────────────────────

class BreathingAnimation(BaseAnimation):
    """
    흉부(ParamBreath) + 몸(BodyAngleY) 사인파 주기적 호흡
    기본 주기: IDLE_BREATH_PERIOD (4초)
    """
    LAYER_ID = "breath"
    PRIORITY = PRIORITY_BREATH

    def __init__(self, state: LayeredParamState, period: Optional[float] = None):
        super().__init__(state)
        self.period = period if period is not None else IDLE_BREATH_PERIOD

    def set_period(self, period: float):
        """호흡 주기 변경 (실행 중에도 적용됨)"""
        self.period = period

    async def _run(self):
        t_start = time.time()
        try:
            while self._running:
                elapsed = time.time() - t_start
                breath_val = (math.sin(2 * math.pi * elapsed / self.period) + 1) / 2
                body_y = IDLE_BODY_AMPLITUDE * breath_val
                self.state.set_layer(
                    self.LAYER_ID,
                    {
                        PARAM_BREATH:       breath_val,
                        PARAM_BODY_ANGLE_Y: body_y,
                    },
                    self.PRIORITY,
                )
                await asyncio.sleep(FRAME_TIME)
        except asyncio.CancelledError:
            pass


# ─── NaturalBlinkAnimation ───────────────────────────────────

class NaturalBlinkAnimation(BaseAnimation):
    """
    자연스러운 눈 깜빡임 레이어
    - 랜덤 2.5~6초 간격 (IDLE_BLINK_MIN/MAX)
    - 10% 확률로 연속 2회 깜빡임
    """
    LAYER_ID = "blink"
    PRIORITY = PRIORITY_BLINK

    def __init__(self, state: LayeredParamState):
        super().__init__(state)

    async def _run(self):
        # 초기 상태: 눈 열림
        self.state.set_layer(
            self.LAYER_ID,
            {PARAM_EYE_OPEN_L: 1.0, PARAM_EYE_OPEN_R: 1.0},
            self.PRIORITY,
        )
        try:
            while self._running:
                interval = random.uniform(IDLE_BLINK_MIN, IDLE_BLINK_MAX)
                await asyncio.sleep(interval)
                if not self._running:
                    break
                await self._blink_once()
                # 10% 확률로 연속 깜빡임
                if random.random() < 0.10 and self._running:
                    await asyncio.sleep(0.07)
                    await self._blink_once()
        except asyncio.CancelledError:
            pass
        finally:
            # 정지 시 눈 열린 상태로 복귀
            self.state.set_layer(
                self.LAYER_ID,
                {PARAM_EYE_OPEN_L: 1.0, PARAM_EYE_OPEN_R: 1.0},
                self.PRIORITY,
            )

    async def _blink_once(self):
        """눈 한 번 깜빡임 (감기 → 홀드 → 뜨기)"""
        d = IDLE_BLINK_DURATION
        close_steps = max(int(d * 0.4 * FPS), 2)
        open_steps  = max(int(d * 0.4 * FPS), 2)

        # 눈 감기
        for i in range(close_steps + 1):
            t = ease_in_out(i / close_steps)
            val = lerp(1.0, 0.0, t)
            self.state.set_layer(
                self.LAYER_ID,
                {PARAM_EYE_OPEN_L: val, PARAM_EYE_OPEN_R: val},
                self.PRIORITY,
            )
            await asyncio.sleep(d * 0.4 / close_steps)

        # 잠깐 감은 상태 유지
        await asyncio.sleep(d * 0.2)

        # 눈 뜨기 (ease_out으로 부드럽게)
        for i in range(open_steps + 1):
            t = ease_out(i / open_steps)
            val = lerp(0.0, 1.0, t)
            self.state.set_layer(
                self.LAYER_ID,
                {PARAM_EYE_OPEN_L: val, PARAM_EYE_OPEN_R: val},
                self.PRIORITY,
            )
            await asyncio.sleep(d * 0.4 / open_steps)

        # 완전히 열린 상태 확정
        self.state.set_layer(
            self.LAYER_ID,
            {PARAM_EYE_OPEN_L: 1.0, PARAM_EYE_OPEN_R: 1.0},
            self.PRIORITY,
        )


# ─── HeadMovement ────────────────────────────────────────────

class HeadMovement(BaseAnimation):
    """
    미세한 머리 흔들림 — 여러 주파수의 사인파 합성으로 자연스럽게 만듦.

    X축: 주기 8초 (70%) + 보조 주기 4.8초 (30%) 합성
    Y축: 주기 10.4초
    Z축: 주기 16.8초 (아주 미세한 기울임)

    각 인스턴스마다 랜덤 위상 오프셋을 적용해 매번 다른 패턴 생성.
    """
    LAYER_ID = "head"
    PRIORITY = PRIORITY_HEAD

    def __init__(
        self,
        state: LayeredParamState,
        amplitude_x: Optional[float] = None,
        amplitude_y: Optional[float] = None,
    ):
        super().__init__(state)
        self.amp_x = amplitude_x if amplitude_x is not None else IDLE_HEAD_AMPLITUDE
        self.amp_y = amplitude_y if amplitude_y is not None else IDLE_HEAD_AMPLITUDE * 0.5
        # 랜덤 위상 — 매 시작마다 다른 움직임
        self._phase_x1 = random.uniform(0, 2 * math.pi)
        self._phase_x2 = random.uniform(0, 2 * math.pi)
        self._phase_y  = random.uniform(0, 2 * math.pi)
        self._phase_z  = random.uniform(0, 2 * math.pi)

    async def _run(self):
        t_start = time.time()
        try:
            while self._running:
                elapsed = time.time() - t_start
                T = IDLE_HEAD_PERIOD  # 기준 주기 8초

                # X축: 두 주파수 합성 (좌우)
                x = (
                    self.amp_x * 0.70 * math.sin(
                        2 * math.pi * elapsed / T + self._phase_x1
                    )
                    + self.amp_x * 0.30 * math.sin(
                        2 * math.pi * elapsed / (T * 0.6) + self._phase_x2
                    )
                )

                # Y축: 단일 주파수 (상하)
                y = self.amp_y * math.sin(
                    2 * math.pi * elapsed / (T * 1.3) + self._phase_y
                )

                # Z축: 매우 느린 미세 기울임
                z = (self.amp_x * 0.25) * math.sin(
                    2 * math.pi * elapsed / (T * 2.1) + self._phase_z
                )

                self.state.set_layer(
                    self.LAYER_ID,
                    {
                        PARAM_FACE_ANGLE_X: x,
                        PARAM_FACE_ANGLE_Y: y,
                        PARAM_FACE_ANGLE_Z: z,
                    },
                    self.PRIORITY,
                )
                await asyncio.sleep(FRAME_TIME)
        except asyncio.CancelledError:
            pass


# ─── HairPhysics ─────────────────────────────────────────────

class HairPhysics(BaseAnimation):
    """
    머리카락/리본 파라미터 흔들림 — 스프링-댐퍼 물리 시뮬레이션.

    머리(FaceAngleX) 움직임을 타겟으로 삼아 지연 반응합니다.
    - HAIR_SIDE: 앞 사이드 헤어 — 빠른 반응
    - HAIR_BACK: 뒷 헤어 — 느린 반응 (더 크게 흔들림)
    """
    LAYER_ID = "hair"
    PRIORITY = PRIORITY_HAIR

    SPRING_K_SIDE  = 8.0    # 사이드 헤어 스프링 강도
    SPRING_K_BACK  = 4.0    # 백 헤어 스프링 강도 (더 느리게)
    DAMPING        = 0.65   # 감쇠 계수 (0=무한 진동, 1=즉시 정지)
    SCALE_SIDE     = 0.45   # FaceAngleX 대비 사이드 헤어 반응 배율
    SCALE_BACK     = 0.70   # FaceAngleX 대비 백 헤어 반응 배율

    def __init__(self, state: LayeredParamState):
        super().__init__(state)
        self._pos_side = 0.0
        self._vel_side = 0.0
        self._pos_back = 0.0
        self._vel_back = 0.0

    async def _run(self):
        t_prev = time.time()
        try:
            while self._running:
                now = time.time()
                dt = min(now - t_prev, 0.05)   # 최대 50ms (이상값 방지)
                t_prev = now

                # 현재 머리 X값을 타겟으로 읽기
                merged = self.state.get_merged()
                head_x = merged.get(PARAM_FACE_ANGLE_X, 0.0)

                # 사이드 헤어 스프링 업데이트
                target_side = head_x * self.SCALE_SIDE
                force_side  = self.SPRING_K_SIDE * (target_side - self._pos_side)
                self._vel_side += force_side * dt
                self._vel_side *= max(0.0, 1.0 - self.DAMPING * dt * 60)
                self._pos_side += self._vel_side * dt
                self._pos_side  = max(-8.0, min(8.0, self._pos_side))

                # 백 헤어 스프링 업데이트 (더 느리고 크게)
                target_back = head_x * self.SCALE_BACK
                force_back  = self.SPRING_K_BACK * (target_back - self._pos_back)
                self._vel_back += force_back * dt
                self._vel_back *= max(0.0, 1.0 - self.DAMPING * 0.75 * dt * 60)
                self._pos_back += self._vel_back * dt
                self._pos_back  = max(-8.0, min(8.0, self._pos_back))

                self.state.set_layer(
                    self.LAYER_ID,
                    {
                        PARAM_HAIR_SIDE: self._pos_side,
                        PARAM_HAIR_BACK: self._pos_back,
                    },
                    self.PRIORITY,
                )
                await asyncio.sleep(FRAME_TIME)
        except asyncio.CancelledError:
            pass


# ─── EmotionAnimation ─────────────────────────────────────────

# 감정 상태별 표정 파라미터 프리셋
# PARAM_MOUTH_FORM: 양수 = 웃음(smile) 방향 (animations.py 사용 패턴 기준)
EMOTION_PRESETS: Dict[str, Dict[str, float]] = {
    "calm": {
        PARAM_EYE_SMILE_L: 0.0,
        PARAM_EYE_SMILE_R: 0.0,
        PARAM_BROW_L:      0.0,
        PARAM_BROW_R:      0.0,
        PARAM_BROW_FORM_L: 0.0,
        PARAM_BROW_FORM_R: 0.0,
        PARAM_MOUTH_FORM:  0.0,
    },
    "happy": {
        PARAM_EYE_SMILE_L: 0.55,
        PARAM_EYE_SMILE_R: 0.55,
        PARAM_BROW_L:      0.10,
        PARAM_BROW_R:      0.10,
        PARAM_BROW_FORM_L: 0.0,
        PARAM_BROW_FORM_R: 0.0,
        PARAM_MOUTH_FORM:  0.35,
    },
    "surprised": {
        PARAM_EYE_SMILE_L: 0.0,
        PARAM_EYE_SMILE_R: 0.0,
        PARAM_BROW_L:      0.45,
        PARAM_BROW_R:      0.45,
        PARAM_BROW_FORM_L: -0.30,
        PARAM_BROW_FORM_R: -0.30,
        PARAM_MOUTH_FORM:  0.0,
    },
    "thinking": {
        PARAM_EYE_SMILE_L: 0.0,
        PARAM_EYE_SMILE_R: 0.0,
        PARAM_BROW_L:      -0.10,
        PARAM_BROW_R:       0.20,
        PARAM_BROW_FORM_L: -0.10,
        PARAM_BROW_FORM_R:  0.0,
        PARAM_MOUTH_FORM:   0.10,
    },
}


class EmotionAnimation(BaseAnimation):
    """
    감정 상태별 표정 파라미터 보간 전환.
    set_emotion() 호출 시 현재 감정값 → 목표 감정값으로 INTERP_DURATION초 동안 부드럽게 전환.
    """
    LAYER_ID        = "emotion"
    PRIORITY        = PRIORITY_EMOTION
    INTERP_DURATION = 0.8   # 보간 시간 (초)

    def __init__(self, state: LayeredParamState, initial: str = "calm"):
        super().__init__(state)
        preset = EMOTION_PRESETS.get(initial, EMOTION_PRESETS["calm"])
        self._current_emotion  = initial
        self._settled_values   = dict(preset)   # 마지막으로 안정된 값
        self._interp_from      = dict(preset)   # 보간 시작값
        self._target_values    = dict(preset)   # 보간 목표값
        self._interp_start     = 0.0
        self._interp_active    = False

    def set_emotion(self, emotion: str):
        """감정 상태 변경 (즉시 보간 시작). 없는 감정이면 무시."""
        if emotion not in EMOTION_PRESETS:
            return
        if emotion == self._current_emotion and not self._interp_active:
            return
        # 현재 실제 표시값에서 출발 (보간 중이라면 그 중간값에서 시작)
        if self._interp_active:
            elapsed = time.time() - self._interp_start
            t = ease_in_out(min(elapsed / self.INTERP_DURATION, 1.0))
            self._interp_from = {
                k: lerp(
                    self._interp_from.get(k, 0.0),
                    self._target_values.get(k, 0.0),
                    t,
                )
                for k in set(self._interp_from) | set(self._target_values)
            }
        else:
            self._interp_from = dict(self._settled_values)

        self._target_values   = dict(EMOTION_PRESETS[emotion])
        self._current_emotion = emotion
        self._interp_start    = time.time()
        self._interp_active   = True

    async def _run(self):
        # 초기 상태 적용
        self.state.set_layer(self.LAYER_ID, dict(self._settled_values), self.PRIORITY)
        try:
            while self._running:
                if self._interp_active:
                    elapsed = time.time() - self._interp_start
                    t = min(elapsed / self.INTERP_DURATION, 1.0)
                    t_eased = ease_in_out(t)

                    interped = {
                        k: lerp(
                            self._interp_from.get(k, 0.0),
                            self._target_values.get(k, 0.0),
                            t_eased,
                        )
                        for k in set(self._interp_from) | set(self._target_values)
                    }
                    self.state.set_layer(self.LAYER_ID, interped, self.PRIORITY)

                    if t >= 1.0:
                        self._interp_active = False
                        self._settled_values = dict(self._target_values)

                await asyncio.sleep(FRAME_TIME)
        except asyncio.CancelledError:
            pass


# ─── TalkingAnimation ─────────────────────────────────────────

class TalkingAnimation(BaseAnimation):
    """
    말하기 중 눈썹 · 눈 세부 움직임 레이어.
    립싱크(입 벌림)와 분리되어 독립적으로 동작합니다.

    활성화 시:
    - 눈 살짝 좁아짐 (SPEAKING_EYE_SQUINT = 0.85)
    - 눈썹 미세 사인파 움직임 (억양 강조 느낌)
    """
    LAYER_ID = "talking"
    PRIORITY = PRIORITY_TALKING

    BROW_MICRO_AMP  = 0.06   # 눈썹 미세 흔들림 진폭
    BROW_MICRO_FREQ = 0.9    # 눈썹 미세 흔들림 주기 (초)

    def __init__(self, state: LayeredParamState):
        super().__init__(state)
        self._active = False

    def set_active(self, active: bool):
        """말하기 레이어 활성/비활성 전환"""
        self._active = active
        if not active:
            self.state.remove_layer(self.LAYER_ID)

    async def _run(self):
        t_start = time.time()
        try:
            while self._running:
                if self._active:
                    elapsed = time.time() - t_start
                    # 눈 좁아짐
                    eye_val = SPEAKING_EYE_SQUINT
                    # 눈썹 미세 사인파
                    brow_micro = self.BROW_MICRO_AMP * math.sin(
                        2 * math.pi * elapsed / self.BROW_MICRO_FREQ
                    )
                    self.state.set_layer(
                        self.LAYER_ID,
                        {
                            PARAM_EYE_OPEN_L: eye_val,
                            PARAM_EYE_OPEN_R: eye_val,
                            PARAM_BROW_L:     brow_micro,
                            PARAM_BROW_R:     brow_micro,
                        },
                        self.PRIORITY,
                    )
                await asyncio.sleep(FRAME_TIME)
        except asyncio.CancelledError:
            pass
        finally:
            self.state.remove_layer(self.LAYER_ID)
