# -*- coding: utf-8 -*-
"""
animation_controller.py - 전체 애니메이션 레이어 오케스트레이터

여러 애니메이션 레이어를 asyncio task로 동시 실행하고,
단일 write loop가 LayeredParamState를 읽어 VTubeStudio에 30fps 주입합니다.

우선순위 시스템 (높을수록 최종값 결정):
  PRIORITY_BREATH   = 10  (호흡)
  PRIORITY_HEAD     = 15  (머리 흔들림)
  PRIORITY_HAIR     = 12  (머리카락 물리)
  PRIORITY_EMOTION  = 20  (감정 표정)
  PRIORITY_TALKING  = 25  (말하기 표정)
  PRIORITY_BLINK    = 30  (눈 깜빡임)
  PRIORITY_REACTION = 40  (즉각 반응 — 최우선)

공개 API:
  await start_idle()             — 기본 대기 상태
  await stop_idle()              — 모든 애니메이션 정지
  await start_talking(audio)     — 립싱크 + 말하기 표정 (idle 위에 레이어)
  set_emotion(emotion)           — 감정 변경 (보간)
  await reaction(type)           — 즉각 반응 애니메이션
"""

import asyncio
import time
from typing import Dict, Optional

from rigging_animation import (
    LayeredParamState,
    BreathingAnimation,
    NaturalBlinkAnimation,
    HeadMovement,
    HairPhysics,
    EmotionAnimation,
    TalkingAnimation,
    PRIORITY_REACTION,
    FRAME_TIME,
    lerp,
    ease_in_out,
    ease_out,
)
from config_vtube import (
    PARAM_FACE_ANGLE_X, PARAM_FACE_ANGLE_Y, PARAM_FACE_ANGLE_Z,
    PARAM_FACE_ANGRY,
    PARAM_EYE_OPEN_L, PARAM_EYE_OPEN_R,
    PARAM_EYE_SMILE_L, PARAM_EYE_SMILE_R,
    PARAM_BROW_L, PARAM_BROW_R,
    PARAM_BROW_FORM_L, PARAM_BROW_FORM_R,
    PARAM_MOUTH_OPEN, PARAM_MOUTH_FORM,
    PARAM_BODY_ANGLE_X, PARAM_BODY_ANGLE_Y, PARAM_BREATH,
    BROW_VALUE_MIN, BROW_VALUE_MAX,
)

# 립싱크 레이어 우선순위: TalkingAnimation(25) 위, Reaction(40) 아래
PRIORITY_LIPSYNC = 28

# 눈썹 파라미터 집합 — write 시 clamp 적용 대상
_BROW_PARAMS = {PARAM_BROW_L, PARAM_BROW_R}


class AnimationController:
    """
    emeth 전체 리깅 애니메이션 오케스트레이터.

    사용법:
        ctrl = AnimationController(vts_api)
        await ctrl.start_idle()          # 기본 대기 시작
        ctrl.set_emotion("happy")        # 감정 변경
        await ctrl.start_talking(path)   # 립싱크 + 말하기
        await ctrl.reaction("nod")       # 반응 애니메이션
        await ctrl.stop_idle()           # 전체 정지
    """

    def __init__(self, vts_api):
        """
        Parameters
        ----------
        vts_api : pyvts VTSPlugin 인스턴스
                  vts_request.InjectParameterDataRequest 를 가진 객체
        """
        self.api = vts_api

        # 공유 파라미터 상태
        self._state = LayeredParamState()

        # 애니메이션 레이어 인스턴스
        self._breath  = BreathingAnimation(self._state)
        self._blink   = NaturalBlinkAnimation(self._state)
        self._head    = HeadMovement(self._state)
        self._hair    = HairPhysics(self._state)
        self._emotion = EmotionAnimation(self._state, initial="calm")
        self._talking = TalkingAnimation(self._state)

        # write loop 태스크
        self._write_task: Optional[asyncio.Task] = None
        self._running = False

        # 반응 애니메이션 태스크
        self._reaction_task: Optional[asyncio.Task] = None

        # 말하기 태스크 (동시에 하나만)
        self._lipsync_task: Optional[asyncio.Task] = None

    # ── 내부: VTubeStudio 파라미터 주입 ─────────────────────

    async def _write_params(self, params: Dict[str, float]):
        """
        requestSetMultiParameterValue로 파라미터 일괄 전송.

        추가 규칙 (매 프레임 강제 적용):
          - face_found=True  : VTubeStudio가 얼굴 추적 중으로 인식
          - FaceAngry=0.0    : 항상 0으로 고정 (화난 표정 방지)
          - BrowLeftY / BrowRightY : BROW_VALUE_MIN ~ BROW_VALUE_MAX 범위로 clamp
        """
        if not params:
            return
        try:
            sanitized = {}
            for k, v in params.items():
                if k in _BROW_PARAMS:
                    # 눈썹은 항상 양수 범위 유지
                    v = max(BROW_VALUE_MIN, min(BROW_VALUE_MAX, v))
                sanitized[k] = v
            # FaceAngry 항상 0.0 강제 주입
            sanitized[PARAM_FACE_ANGRY] = 0.0

            keys   = list(sanitized.keys())
            values = list(sanitized.values())
            await self.api.request(
                self.api.vts_request.requestSetMultiParameterValue(
                    keys, values, face_found=True   # ← True: 얼굴 추적 중으로 인식
                )
            )
        except Exception:
            pass  # 연결 끊김 등 일시적 오류 무시

    async def _write_loop(self):
        """30fps write loop — 모든 레이어 병합 후 VTubeStudio에 주입"""
        try:
            while self._running:
                merged = self._state.get_merged()
                if merged:
                    await self._write_params(merged)
                await asyncio.sleep(FRAME_TIME)
        except asyncio.CancelledError:
            pass

    # ── 내부: 루프 관리 ──────────────────────────────────────

    def _start_write_loop(self):
        if self._write_task and not self._write_task.done():
            return
        self._running = True
        self._write_task = asyncio.get_running_loop().create_task(self._write_loop())

    def _stop_write_loop(self):
        self._running = False
        if self._write_task and not self._write_task.done():
            self._write_task.cancel()
        self._write_task = None

    def _stop_all_layers(self):
        """모든 애니메이션 레이어 정지"""
        for anim in (
            self._breath, self._blink, self._head,
            self._hair, self._emotion, self._talking,
        ):
            anim.stop()
        if self._reaction_task and not self._reaction_task.done():
            self._reaction_task.cancel()
            self._reaction_task = None
        if self._lipsync_task and not self._lipsync_task.done():
            self._lipsync_task.cancel()
            self._lipsync_task = None
        self._state.clear()

    # ── 공개 API ─────────────────────────────────────────────

    async def start_idle(self):
        """
        기본 대기 상태 시작.
        호흡 + 자연 깜빡임 + 미세 머리 흔들림 + 머리카락 물리 + 감정 레이어
        를 동시에 asyncio task로 실행합니다.
        """
        self._stop_all_layers()
        self._talking.set_active(False)

        self._breath.start()
        self._blink.start()
        self._head.start()
        self._hair.start()
        self._emotion.start()
        self._talking.start()   # TalkingAnimation 태스크 시작 (비활성 상태)

        self._start_write_loop()
        print("[AnimCtrl] idle 시작 — 호흡·깜빡임·머리·헤어·감정 레이어 활성")

    async def stop_idle(self):
        """모든 애니메이션 레이어 및 write loop 정지"""
        self._stop_all_layers()
        self._stop_write_loop()
        print("[AnimCtrl] 전체 애니메이션 정지")

    async def start_talking(self, audio_path: str):
        """
        말하기 상태 진입.
        idle 레이어 위에 립싱크(입 벌림)와 TalkingAnimation(눈/눈썹)을 추가합니다.
        오디오가 끝나면 자동으로 말하기 레이어를 해제합니다.

        Parameters
        ----------
        audio_path : str   분석할 오디오 파일 경로
        """
        if not self._running:
            await self.start_idle()

        # 이전 립싱크가 진행 중이면 취소
        if self._lipsync_task and not self._lipsync_task.done():
            self._lipsync_task.cancel()

        self._lipsync_task = asyncio.get_running_loop().create_task(
            self._run_lipsync(audio_path)
        )
        await self._lipsync_task

    async def _run_lipsync(self, audio_path: str):
        """립싱크 내부 루틴"""
        from lipsync import analyze_audio

        self._talking.set_active(True)
        print(f"[AnimCtrl] 립싱크 시작: {audio_path}")

        try:
            frames = await asyncio.get_running_loop().run_in_executor(
                None, analyze_audio, audio_path
            )
        except Exception as e:
            print(f"[AnimCtrl] 오디오 분석 실패: {e}")
            self._talking.set_active(False)
            return

        if not frames:
            print("[AnimCtrl] 분석 결과 없음 — 립싱크 건너뜀")
            self._talking.set_active(False)
            return

        loop = asyncio.get_running_loop()
        start_time = loop.time()
        try:
            for timestamp_ms, mouth_val in frames:
                target_time = start_time + timestamp_ms / 1000.0
                wait = target_time - loop.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                # 립싱크 레이어를 상태에 기록 (write loop가 VTS에 전송)
                self._state.set_layer(
                    "lipsync",
                    {PARAM_MOUTH_OPEN: mouth_val},
                    PRIORITY_LIPSYNC,
                )
        except asyncio.CancelledError:
            pass
        finally:
            # 입 닫기 후 레이어 제거
            self._state.set_layer("lipsync", {PARAM_MOUTH_OPEN: 0.0}, PRIORITY_LIPSYNC)
            await asyncio.sleep(0.1)
            self._state.remove_layer("lipsync")
            self._talking.set_active(False)
            print("[AnimCtrl] 립싱크 완료")

    def set_emotion(self, emotion: str):
        """
        감정 상태 변경. INTERP_DURATION 초 동안 보간해 자연스럽게 전환합니다.
        지원 감정: calm / happy / surprised / thinking
        """
        self._emotion.set_emotion(emotion)
        print(f"[AnimCtrl] 감정 전환: {emotion}")

    async def reaction(self, reaction_type: str):
        """
        즉각 반응 애니메이션 (PRIORITY_REACTION = 40, 최우선 레이어).
        이전 반응이 진행 중이면 취소하고 새 반응으로 교체합니다.

        Parameters
        ----------
        reaction_type : str
            "chat_superchat" — 슈퍼챗 반응 (기쁨, 눈 웃음)
            "surprised"      — 놀람 (눈 크게, 눈썹 올라감)
            "nod"            — 끄덕임 (고개 2번 아래위)
            "shake"          — 고개 젓기 (좌우 흔들림)
        """
        if self._reaction_task and not self._reaction_task.done():
            self._reaction_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self._reaction_task), timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        _reaction_map = {
            "chat_superchat": self._reaction_superchat,
            "surprised":      self._reaction_surprised,
            "nod":            self._reaction_nod,
            "shake":          self._reaction_shake,
        }
        fn = _reaction_map.get(reaction_type)
        if fn is None:
            print(f"[AnimCtrl] 알 수 없는 reaction 타입: {reaction_type}")
            return

        self._reaction_task = asyncio.get_running_loop().create_task(fn())
        print(f"[AnimCtrl] reaction 시작: {reaction_type}")

    # ── 반응 애니메이션 구현 ─────────────────────────────────

    async def _reaction_superchat(self):
        """슈퍼챗 반응: 눈 웃음 + 눈썹 올라감 + 미소 → 1.5초 유지 → fade out"""
        LAYER = "rx_superchat"
        try:
            await self._interp_layer(
                LAYER,
                start={
                    PARAM_EYE_OPEN_L:  1.0,
                    PARAM_EYE_OPEN_R:  1.0,
                    PARAM_BROW_L:      0.0,
                    PARAM_BROW_R:      0.0,
                    PARAM_EYE_SMILE_L: 0.0,
                    PARAM_EYE_SMILE_R: 0.0,
                    PARAM_MOUTH_FORM:  0.0,
                },
                end={
                    PARAM_EYE_OPEN_L:  1.0,
                    PARAM_EYE_OPEN_R:  1.0,
                    PARAM_BROW_L:      0.30,
                    PARAM_BROW_R:      0.30,
                    PARAM_EYE_SMILE_L: 0.65,
                    PARAM_EYE_SMILE_R: 0.65,
                    PARAM_MOUTH_FORM:  0.40,
                },
                duration=0.45,
                priority=PRIORITY_REACTION,
            )
            await asyncio.sleep(1.8)
        finally:
            await self._fade_out_layer(LAYER, duration=0.5)

    async def _reaction_surprised(self):
        """놀람 반응: 눈 크게 뜨기 + 눈썹 급격히 올라감 → fade out"""
        LAYER = "rx_surprised"
        try:
            await self._interp_layer(
                LAYER,
                start={
                    PARAM_BROW_L:      0.0,
                    PARAM_BROW_R:      0.0,
                    PARAM_BROW_FORM_L: 0.0,
                    PARAM_BROW_FORM_R: 0.0,
                    PARAM_MOUTH_OPEN:  0.0,
                    PARAM_EYE_OPEN_L:  1.0,
                    PARAM_EYE_OPEN_R:  1.0,
                },
                end={
                    PARAM_BROW_L:      0.50,
                    PARAM_BROW_R:      0.50,
                    PARAM_BROW_FORM_L: -0.35,
                    PARAM_BROW_FORM_R: -0.35,
                    PARAM_MOUTH_OPEN:  0.25,
                    PARAM_EYE_OPEN_L:  1.0,
                    PARAM_EYE_OPEN_R:  1.0,
                },
                duration=0.20,
                priority=PRIORITY_REACTION,
            )
            await asyncio.sleep(1.0)
        finally:
            await self._fade_out_layer(LAYER, duration=0.45)

    async def _reaction_nod(self):
        """끄덕임: 고개를 2번 아래위로 흔들기"""
        LAYER = "rx_nod"
        try:
            for _ in range(2):
                await self._interp_layer(
                    LAYER,
                    start={PARAM_FACE_ANGLE_Y: 0.0},
                    end={PARAM_FACE_ANGLE_Y:   -9.0},
                    duration=0.18,
                    priority=PRIORITY_REACTION,
                )
                await self._interp_layer(
                    LAYER,
                    start={PARAM_FACE_ANGLE_Y: -9.0},
                    end={PARAM_FACE_ANGLE_Y:    0.0},
                    duration=0.18,
                    priority=PRIORITY_REACTION,
                )
                await asyncio.sleep(0.06)
        finally:
            self._state.remove_layer(LAYER)

    async def _reaction_shake(self):
        """고개 젓기: 좌우로 4회 흔들기"""
        LAYER = "rx_shake"
        try:
            angles = [0.0, 11.0, -11.0, 8.0, -8.0, 0.0]
            for i in range(len(angles) - 1):
                await self._interp_layer(
                    LAYER,
                    start={PARAM_FACE_ANGLE_X: angles[i]},
                    end={PARAM_FACE_ANGLE_X:   angles[i + 1]},
                    duration=0.14,
                    priority=PRIORITY_REACTION,
                )
        finally:
            self._state.remove_layer(LAYER)

    # ── 보간 헬퍼 ────────────────────────────────────────────

    async def _interp_layer(
        self,
        layer_id: str,
        start: Dict[str, float],
        end: Dict[str, float],
        duration: float,
        priority: int,
    ):
        """
        start → end 파라미터를 duration초 동안 ease_in_out으로 보간.
        매 프레임마다 LayeredParamState를 업데이트합니다.
        """
        steps = max(int(duration * 30), 2)
        all_keys = set(start) | set(end)
        for i in range(steps + 1):
            t = ease_in_out(i / steps)
            interped = {
                k: lerp(start.get(k, 0.0), end.get(k, 0.0), t)
                for k in all_keys
            }
            self._state.set_layer(layer_id, interped, priority)
            await asyncio.sleep(duration / steps)

    async def _fade_out_layer(self, layer_id: str, duration: float = 0.35):
        """
        레이어의 현재 값들을 0으로 보간 후 레이어를 제거합니다.
        반응 애니메이션 종료 시 부드럽게 원래 레이어로 복귀.
        """
        current = self._state.get_layer_values(layer_id)
        if not current:
            self._state.remove_layer(layer_id)
            return
        zero = {k: 0.0 for k in current}
        await self._interp_layer(
            layer_id, current, zero, duration=duration, priority=PRIORITY_REACTION
        )
        self._state.remove_layer(layer_id)
