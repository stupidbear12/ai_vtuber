# -*- coding: utf-8 -*-
"""
app/tts_manager.py -- TTS 생성 + Live2D 브로드캐스트 믹스인

BroadcastChatManager에서 분리된 TTS 관련 공통 헬퍼 메서드.
"""

import base64
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class TTSMixin:
    """TTS 생성 + Live2D 브로드캐스트 공통 헬퍼 메서드.

    BroadcastChatManager가 이 믹스인을 상속하여 사용한다.
    self._tts_lock, self._voice_enabled, self._voice_url, self._live2d_url 등
    BroadcastChatManager.__init__에서 초기화된 속성에 의존한다.
    """

    async def _speak_and_broadcast(self, text: str, emotion: str = "calm") -> None:
        """TTS 생성 + Live2D 브로드캐스트 공통 헬퍼.

        자율 행동의 talk/react/topic_change 등에서 사용한다.
        _tts_lock으로 직렬화하여 다른 TTS와 겹치지 않도록 보장.

        Args:
            text: 말할 텍스트
            emotion: 감정 태그 (calm, happy, excited, sad 등)
        """
        async with self._tts_lock:
            audio_base64: Optional[str] = None
            if self._voice_enabled and text:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"{self._voice_url}/voice/tts",
                            json={"text": text, "emotion": emotion},
                            timeout=aiohttp.ClientTimeout(total=30.0),
                        ) as resp:
                            if resp.status == 200:
                                audio_bytes = await resp.read()
                                audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
                            else:
                                logger.warning("[SpeakBroadcast] TTS 실패: HTTP %s", resp.status)
                except Exception as e:
                    logger.warning("[SpeakBroadcast] ai_voice 연결 실패 (무시): %s", e)

            try:
                async with aiohttp.ClientSession() as session:
                    await session.post(
                        f"{self._live2d_url}/live2d/emotion",
                        json={"emotion": emotion},
                        timeout=aiohttp.ClientTimeout(total=3.0),
                    )
                    broadcast_payload: dict = {
                        "cmd": "speak",
                        "text": text,
                        "emotion": emotion,
                        "author": "시온",
                        "platform": "autonomous",
                        "is_donation": False,
                    }
                    if audio_base64:
                        broadcast_payload["audio_base64"] = audio_base64
                    await session.post(
                        f"{self._live2d_url}/live2d/broadcast",
                        json=broadcast_payload,
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    )
            except Exception as e:
                logger.warning("[SpeakBroadcast] ai_live2d 연결 실패 (무시): %s", e)
