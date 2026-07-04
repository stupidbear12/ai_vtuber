# -*- coding: utf-8 -*-
"""
app/obs_controller.py — OBS WebSocket 연결을 통한 BGM 제어 유틸리티

TTS 음성 재생 시 BGM을 자동으로 음소거하고, 재생 완료 후 복원한다.
OBS 28+ 내장 WebSocket 서버 (기본 ws://localhost:4455)에 연결한다.

환경변수:
  OBS_WS_HOST          — OBS WebSocket 호스트 (기본: localhost)
  OBS_WS_PORT          — OBS WebSocket 포트 (기본: 4455)
  OBS_WS_PASSWORD      — OBS WebSocket 비밀번호 (없으면 인증 없이 연결)
  OBS_BGM_SOURCE_NAME  — BGM 미디어 소스 이름 (기본: BGM)
"""

import asyncio
import io
import logging
import os
import wave
from typing import Optional

logger = logging.getLogger(__name__)

# obsws-python은 선택 의존성 — 설치 안 되어 있으면 BGM 제어만 비활성
try:
    import obsws_python as obs
except ImportError:
    obs = None
    logger.info("[OBS] obsws-python 미설치 — BGM 자동 제어 비활성")


class OBSController:
    """OBS WebSocket을 통한 소스 제어 (음소거, 볼륨 등).

    연결 실패 시 경고만 출력하고 모든 메서드가 False를 반환한다.
    TTS 파이프라인은 OBS 연결 여부와 무관하게 정상 동작한다.
    """

    def __init__(self):
        self._host = os.environ.get("OBS_WS_HOST", "localhost")
        self._port = int(os.environ.get("OBS_WS_PORT", "4455"))
        self._password = os.environ.get("OBS_WS_PASSWORD", "")
        self._bgm_source = os.environ.get("OBS_BGM_SOURCE_NAME", "BGM")
        self._client = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> bool:
        """OBS WebSocket 연결을 보장한다. 실패 시 False 반환."""
        if obs is None:
            return False

        if self._connected and self._client:
            return True

        async with self._lock:
            # double-check after acquiring lock
            if self._connected and self._client:
                return True
            try:
                kwargs = {"host": self._host, "port": self._port}
                if self._password:
                    kwargs["password"] = self._password

                self._client = await asyncio.to_thread(
                    lambda: obs.ReqClient(**kwargs)
                )
                self._connected = True
                logger.info(
                    f"[OBS] WebSocket 연결 성공: {self._host}:{self._port} "
                    f"(BGM 소스: {self._bgm_source!r})"
                )
                return True
            except Exception as e:
                logger.warning(f"[OBS] WebSocket 연결 실패 (BGM 제어 비활성): {e}")
                self._client = None
                self._connected = False
                return False

    async def mute_source(self, source_name: Optional[str] = None) -> bool:
        """소스를 음소거한다.

        Args:
            source_name: 대상 소스 이름 (None이면 BGM 소스 사용)

        Returns:
            성공 시 True, 실패 시 False
        """
        name = source_name or self._bgm_source
        if not await self._ensure_connected():
            return False
        try:
            await asyncio.to_thread(
                lambda: self._client.set_input_mute(name=name, input_muted=True)
            )
            logger.debug(f"[OBS] 소스 음소거: {name!r}")
            return True
        except Exception as e:
            logger.warning(f"[OBS] 음소거 실패 ({name!r}): {e}")
            self._connected = False
            return False

    async def unmute_source(self, source_name: Optional[str] = None) -> bool:
        """소스 음소거를 해제한다.

        Args:
            source_name: 대상 소스 이름 (None이면 BGM 소스 사용)

        Returns:
            성공 시 True, 실패 시 False
        """
        name = source_name or self._bgm_source
        if not await self._ensure_connected():
            return False
        try:
            await asyncio.to_thread(
                lambda: self._client.set_input_mute(name=name, input_muted=False)
            )
            logger.debug(f"[OBS] 음소거 해제: {name!r}")
            return True
        except Exception as e:
            logger.warning(f"[OBS] 음소거 해제 실패 ({name!r}): {e}")
            self._connected = False
            return False

    async def set_source_volume(
        self, source_name: Optional[str] = None, volume_db: float = 0.0
    ) -> bool:
        """소스 볼륨을 dB 단위로 설정한다.

        Args:
            source_name: 대상 소스 이름 (None이면 BGM 소스 사용)
            volume_db: 볼륨 (dB). 0 = 원래 볼륨, 음수 = 감소.

        Returns:
            성공 시 True, 실패 시 False
        """
        name = source_name or self._bgm_source
        if not await self._ensure_connected():
            return False
        try:
            await asyncio.to_thread(
                lambda: self._client.set_input_volume(
                    name=name, input_volume_db=volume_db
                )
            )
            logger.debug(f"[OBS] 볼륨 설정: {name!r} = {volume_db}dB")
            return True
        except Exception as e:
            logger.warning(f"[OBS] 볼륨 설정 실패 ({name!r}): {e}")
            self._connected = False
            return False

    async def mute_bgm(self) -> bool:
        """BGM 소스를 음소거한다."""
        return await self.mute_source()

    async def unmute_bgm(self) -> bool:
        """BGM 소스 음소거를 해제한다."""
        return await self.unmute_source()

    def close(self):
        """연결을 종료한다."""
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
            self._connected = False
            logger.info("[OBS] WebSocket 연결 종료")


def get_wav_duration(audio_bytes: bytes) -> float:
    """WAV 바이트에서 오디오 재생 시간(초)을 계산한다.

    Args:
        audio_bytes: WAV 형식 바이트 데이터

    Returns:
        재생 시간 (초). 파싱 실패 시 대략적 추정값 반환.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0.0
            duration = frames / rate
            logger.debug(
                f"[OBS] WAV 분석: {rate}Hz, {wf.getnchannels()}ch, "
                f"{wf.getsampwidth()*8}bit, {duration:.2f}s"
            )
            return duration
    except Exception as e:
        logger.warning(f"[OBS] WAV 헤더 파싱 실패, 대략 추정: {e}")
        # 폴백: GPT-SoVITS 기본값 (32kHz mono 16bit)
        return len(audio_bytes) / (32000 * 1 * 2)


# ── 싱글톤 ──────────────────────────────────────────────────────

_obs_controller: Optional[OBSController] = None


def get_obs_controller() -> OBSController:
    """OBSController 싱글톤 인스턴스를 반환한다."""
    global _obs_controller
    if _obs_controller is None:
        _obs_controller = OBSController()
    return _obs_controller
