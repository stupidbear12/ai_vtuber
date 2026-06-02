"""
vtube_controller.py - VTube Studio 통합 컨트롤러 (전체 리깅 버전)

v2.0 업데이트:
  - AnimationController (rigging_animation 기반) 통합
  - start_full_animation() / stop_full_animation() 추가
  - speak(audio_path, emotion) 추가 — 말하기 + 감정 한번에
  - 기존 AnimationEngine(legacy 호환) 유지
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import pyvts
except ImportError:
    print("[오류] pyvts가 설치되지 않았습니다: pip install pyvts")
    sys.exit(1)

from config_vtube import (
    VTS_HOST, VTS_PORT, PLUGIN_NAME, PLUGIN_DEVELOPER, TOKEN_FILE,
    EXPRESSIONS, PARAM_MOUTH_OPEN, BIBLE_MOUTH_MAX,
)
from animations import AnimationEngine
from animation_controller import AnimationController
from lipsync import analyze_audio


class VTubeController:
    def __init__(self):
        self.vts = pyvts.vts(
            plugin_info={
                "plugin_name": PLUGIN_NAME,
                "developer":   PLUGIN_DEVELOPER,
                "authentication_token_path": TOKEN_FILE,
            },
            vts_api_info={"version": "1.0", "name": "VTubeStudioPublicAPI",
                          "port": VTS_PORT}
        )
        self.anim: Optional[AnimationEngine] = None          # 레거시 호환용
        self.anim_ctrl: Optional[AnimationController] = None # 전체 리깅 컨트롤러
        self._connected = False
        self._full_anim_active = False  # 전체 애니메이션 시스템 활성 여부

    # ── 연결/해제 ──────────────────────────────────────────

    async def connect(self):
        """VTube Studio 연결 및 플러그인 인증"""
        await self.vts.connect()
        await self.vts.request_authenticate_token()
        await self.vts.request_authenticate()
        self.anim      = AnimationEngine(self.vts)
        self.anim_ctrl = AnimationController(self.vts)
        self._connected = True
        print(f"[VTube] 연결 완료 — {VTS_HOST}:{VTS_PORT}")

    async def disconnect(self):
        """안전하게 연결 종료"""
        if self._full_anim_active and self.anim_ctrl:
            await self.anim_ctrl.stop_idle()
            self._full_anim_active = False
        elif self.anim:
            self.anim.stop_idle()
        await self.vts.close()
        self._connected = False
        print("[VTube] 연결 종료")

    def _check(self):
        if not self._connected:
            raise RuntimeError("VTube Studio에 연결되지 않았습니다. connect()를 먼저 호출하세요.")

    # ── 모델 정보 ──────────────────────────────────────────

    async def get_model_info(self) -> dict:
        """현재 로드된 모델 원시 응답 반환"""
        self._check()
        resp = await self.vts.request(
            self.vts.vts_request.BaseRequest("CurrentModelRequest")
        )
        return resp

    async def get_current_model(self) -> dict:
        """
        현재 로드된 모델 정보를 정리된 dict로 반환.
        test_vtube.py 호환용.
        """
        self._check()
        try:
            resp = await self.vts.request(
                self.vts.vts_request.BaseRequest("CurrentModelRequest")
            )
            data = resp.get("data", {})
            return {
                "model_loaded":   data.get("modelLoaded", False),
                "model_name":     data.get("modelName", ""),
                "model_id":       data.get("modelID", ""),
                "vts_model_name": data.get("vtsModelName", ""),
                "live2d_file":    data.get("live2DModelPath", ""),
            }
        except Exception as e:
            return {"model_loaded": False, "error": str(e)}

    async def set_mouth_open(self, value: float):
        """
        입 벌림 파라미터 직접 설정 (0.0 ~ 1.0).
        test_vtube.py 및 레거시 코드 호환용.
        """
        self._check()
        value = max(0.0, min(1.0, value))
        try:
            await self.vts.request(
                self.vts.vts_request.requestSetParameterValue(
                    PARAM_MOUTH_OPEN, value, face_found=False
                )
            )
        except Exception:
            pass

    async def idle_animation(self, duration: float = 5.0):
        """
        레거시 아이들 애니메이션 — duration초 동안 AnimationEngine의
        호흡 + 눈 깜빡임 + 머리 흔들림을 실행 후 정지.
        test_vtube.py 호환용.
        """
        self._check()
        self.anim._start_idle()
        await asyncio.sleep(duration)
        self.anim.stop_idle()

    async def get_parameters(self) -> list:
        """모델의 Live2D 파라미터 목록 반환"""
        self._check()
        resp = await self.vts.request(
            self.vts.vts_request.BaseRequest("ParameterListRequest")
        )
        return resp.get("data", {}).get("parameters", [])

    # ── 표정 제어 ──────────────────────────────────────────

    async def set_expression(self, name: str):
        """
        표정 전환 — VTube Studio 핫키 트리거 방식.

        pyvts 최신 버전에서 ExpressionActivationRequest 가 제거되었으므로
        requestTriggerHotKey(hotkeyID) 를 사용합니다.
        hotkeyID 는 VTube Studio 에서 설정한 핫키 이름(또는 ID)과 일치해야 합니다.
        EXPRESSIONS 딕셔너리의 값이 핫키 이름으로 사용됩니다.
        """
        self._check()
        hotkey_name = EXPRESSIONS.get(name, name)
        try:
            await self.vts.request(
                self.vts.vts_request.requestTriggerHotKey(hotkey_name)
            )
        except Exception as e:
            print(f"[경고] 표정 전환 실패 ({hotkey_name}): {e}")

    # ── 애니메이션 제어 ────────────────────────────────────

    async def play_animation(self, name: str, **kwargs):
        """이름으로 애니메이션 실행"""
        self._check()
        await self.anim.play_animation(name, **kwargs)

    async def set_broadcast_mode(self, mode: str):
        """
        방송 모드 전환
        mode: idle | opening | speaking | bible | emotional | closing
        """
        self._check()
        mode_map = {
            "idle":      "idle",
            "opening":   "opening",
            "bible":     "bible",
            "emotional": "emotional",
            "closing":   "closing",
        }
        anim_name = mode_map.get(mode, "idle")
        await self.anim.play_animation(anim_name)
        print(f"[VTube] 모드 전환: {mode}")

    # ── 전체 리깅 애니메이션 시스템 ───────────────────────────

    async def start_full_animation(self):
        """
        전체 리깅 애니메이션 시스템 시작.
        호흡 · 눈 깜빡임 · 머리 흔들림 · 머리카락 물리 · 감정 레이어를
        모두 asyncio task로 동시 실행합니다.
        """
        self._check()
        await self.anim_ctrl.start_idle()
        self._full_anim_active = True
        print("[VTube] 전체 애니메이션 시스템 시작")

    async def stop_full_animation(self):
        """전체 리깅 애니메이션 시스템 정지"""
        if self.anim_ctrl and self._full_anim_active:
            await self.anim_ctrl.stop_idle()
            self._full_anim_active = False
            print("[VTube] 전체 애니메이션 시스템 정지")

    async def speak(self, audio_path: str, emotion: str = "calm"):
        """
        말하기 + 감정 변경을 한번에 처리합니다.

        전체 애니메이션 시스템(start_full_animation)이 활성화된 상태에서
        사용하는 것을 권장합니다.

        Parameters
        ----------
        audio_path : str   립싱크에 사용할 오디오 파일 경로
        emotion    : str   말하기 중 표정 감정 (calm/happy/surprised/thinking)
        """
        self._check()
        if not self._full_anim_active:
            await self.start_full_animation()
        self.anim_ctrl.set_emotion(emotion)
        await self.anim_ctrl.start_talking(audio_path)

    # ── 감정 / 반응 ────────────────────────────────────────

    def set_emotion_layer(self, emotion: str):
        """
        전체 애니메이션 시스템의 감정 레이어 변경.
        start_full_animation() 이후에 호출해야 합니다.
        """
        self._check()
        if self.anim_ctrl:
            self.anim_ctrl.set_emotion(emotion)

    async def trigger_reaction(self, reaction_type: str):
        """
        즉각 반응 애니메이션 트리거.
        지원 타입: chat_superchat / surprised / nod / shake
        """
        self._check()
        if self.anim_ctrl:
            await self.anim_ctrl.reaction(reaction_type)

    # ── 립싱크 ─────────────────────────────────────────────

    async def lipsync_from_audio(self, audio_path: str, bible_mode: bool = False):
        """
        오디오 파일 분석 → 실시간 립싱크 + 말하기 애니메이션
        bible_mode: True면 입 최대 개방량 제한
        """
        self._check()
        max_mouth = BIBLE_MOUTH_MAX if bible_mode else 1.0
        frames = analyze_audio(audio_path)

        if not frames:
            print("[경고] 오디오 분석 결과 없음")
            return

        # 전체 애니메이션 시스템이 활성화된 경우 AnimationController에 위임
        if self._full_anim_active and self.anim_ctrl:
            await self.anim_ctrl.start_talking(audio_path)
            return

        # 레거시 방식: AnimationEngine 직접 사용
        _loop = asyncio.get_running_loop()
        start_time = _loop.time()
        for timestamp_ms, mouth_val in frames:
            target_time = start_time + timestamp_ms / 1000.0
            now = _loop.time()
            wait = target_time - now
            if wait > 0:
                await asyncio.sleep(wait)
            clamped = min(mouth_val, max_mouth)
            await self.anim.play_speaking(mouth_value=clamped)

        # 립싱크 완료 후 입 닫기
        await self.anim.play_speaking(mouth_value=0.0)

    # ── 방송 시퀀스 ────────────────────────────────────────

    async def run_broadcast_sequence(self, sections: list[dict]):
        """
        대본 섹션에 맞춰 자동으로 모드 전환
        sections: [{"type": "opening"|"body"|"bible"|"closing", "audio": "path.mp3"}, ...]
        """
        self._check()
        for section in sections:
            stype = section.get("type", "body")
            audio = section.get("audio")

            if stype == "opening":
                await self.set_broadcast_mode("opening")
                await asyncio.sleep(3.0)
            elif stype == "bible":
                await self.set_broadcast_mode("bible")
                if audio:
                    await self.lipsync_from_audio(audio, bible_mode=True)
            elif stype == "closing":
                await self.set_broadcast_mode("emotional")
                await asyncio.sleep(2.0)
                await self.set_broadcast_mode("closing")
                if audio:
                    await self.lipsync_from_audio(audio)
            else:
                # 일반 말하기
                if audio:
                    await self.lipsync_from_audio(audio)

        print("[VTube] 방송 시퀀스 완료")
