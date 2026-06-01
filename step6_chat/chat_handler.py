"""
chat_handler.py - 채팅 메시지 처리 및 emeth 응답 생성 모듈
"""

import json
import logging
import os
import sys
import time
from typing import Optional

import requests

from config_chat import (
    AUTO_TTS,
    AUTO_VTUBE,
    MAX_RESPONSE_LENGTH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RESPONSE_COOLDOWN,
    RESPONSE_LOG_PATH,
    SKIP_KEYWORDS,
)

logger = logging.getLogger(__name__)

# step2_tts 경로 추가 (TTS 연동)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "step2_tts"))
sys.path.insert(0, os.path.join(ROOT_DIR, "step3_vtube"))


class ChatHandler:
    """채팅 메시지 처리 및 emeth 응답 생성 클래스."""

    def __init__(self):
        # 유저별 마지막 응답 시간 기록 {channel_id: timestamp}
        self._cooldown_map: dict[str, float] = {}
        # 응답 로그 파일 초기화
        self._log_path = RESPONSE_LOG_PATH

    def should_respond(self, message: dict) -> bool:
        """
        해당 메시지에 응답할지 여부를 판단합니다.

        Args:
            message: {"author": str, "author_channel_id": str, "message": str, ...}

        Returns:
            True이면 응답 생성, False이면 건너뜀
        """
        text = message.get("message", "").strip()
        channel_id = message.get("author_channel_id", "")

        # 1. 빈 메시지 건너뜀
        if not text:
            return False

        # 2. 너무 짧은 메시지 건너뜀 (2글자 이하)
        if len(text) <= 2:
            logger.debug(f"메시지 너무 짧음 (건너뜀): '{text}'")
            return False

        # 3. SKIP_KEYWORDS 포함 시 건너뜀
        text_lower = text.lower()
        for keyword in SKIP_KEYWORDS:
            if keyword.lower() in text_lower:
                logger.debug(f"스킵 키워드 감지 (건너뜀): '{keyword}' in '{text}'")
                return False

        # 4. 쿨다운 체크 (같은 유저 중복 응답 방지)
        if channel_id:
            last_time = self._cooldown_map.get(channel_id, 0)
            elapsed = time.time() - last_time
            if elapsed < RESPONSE_COOLDOWN:
                remaining = round(RESPONSE_COOLDOWN - elapsed, 1)
                logger.debug(f"쿨다운 중 (건너뜀): {channel_id} - {remaining}초 남음")
                return False

        return True

    def generate_response(self, username: str, message: str) -> Optional[str]:
        """
        Ollama emeth 모델로 채팅 응답을 생성합니다.

        Args:
            username: 시청자 닉네임
            message: 시청자가 보낸 채팅 메시지

        Returns:
            응답 문자열 (MAX_RESPONSE_LENGTH 이하), 실패 시 None
        """
        system_prompt = (
            "당신은 차분하고 따뜻한 lofi 음악 버튜버 emeth입니다. "
            "시청자의 채팅에 짧고 따뜻하게 1~2문장으로 답해주세요. "
            "존댓말을 사용하고, 잔잔한 분위기를 유지해주세요."
        )
        user_prompt = f"{username}님: {message}"

        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 150,
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            response_text = data["message"]["content"].strip()

            # 응답 길이 제한
            if len(response_text) > MAX_RESPONSE_LENGTH:
                # 마지막 완성된 문장 단위로 자르기
                truncated = response_text[:MAX_RESPONSE_LENGTH]
                last_period = max(
                    truncated.rfind("다."),
                    truncated.rfind("요."),
                    truncated.rfind("어요."),
                )
                if last_period > 0:
                    response_text = truncated[: last_period + 2]
                else:
                    response_text = truncated.rstrip() + "..."

            logger.info(f"응답 생성: [{username}] {message} → {response_text}")
            return response_text

        except requests.exceptions.ConnectionError:
            logger.error("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
            return None
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            return None

    def _save_log(self, username: str, message: str, response: str) -> None:
        """응답 이력을 JSONL 파일에 저장합니다."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "username": username,
            "message": message,
            "response": response,
        }
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"로그 저장 실패: {e}")

    async def process_message(self, chat_message: dict) -> Optional[str]:
        """
        채팅 메시지 전체 처리 파이프라인.

        1. should_respond() 체크
        2. generate_response()
        3. AUTO_TTS이면 TTS 변환
        4. 응답 로그 저장

        Args:
            chat_message: YouTubeChatReader가 반환하는 메시지 dict

        Returns:
            생성된 응답 문자열, 건너뛰거나 실패 시 None
        """
        username = chat_message.get("author", "시청자")
        message = chat_message.get("message", "").strip()
        channel_id = chat_message.get("author_channel_id", "")

        # 1. 응답 여부 판단
        if not self.should_respond(chat_message):
            return None

        # 2. 응답 생성
        response = self.generate_response(username, message)
        if not response:
            return None

        # 3. 쿨다운 갱신
        if channel_id:
            self._cooldown_map[channel_id] = time.time()

        # 4. 로그 저장
        self._save_log(username, message, response)

        # 5. TTS 변환 (AUTO_TTS=True인 경우)
        if AUTO_TTS:
            await self._tts_response(response)

        # 6. VTube Studio 연동 (AUTO_VTUBE=True인 경우)
        if AUTO_VTUBE:
            await self._vtube_response(response)

        return response

    async def _tts_response(self, text: str) -> None:
        """TTS 변환 (step2_tts 연동)."""
        try:
            from tts_engine import TTSEngine
            import asyncio
            engine = TTSEngine()
            loop = asyncio.get_event_loop()
            audio_path = await loop.run_in_executor(
                None,
                lambda: engine.synthesize(text=text)
            )
            logger.info(f"TTS 생성 완료: {audio_path}")
        except ImportError:
            logger.warning("tts_engine 모듈을 찾을 수 없습니다. step2_tts가 설정되어 있는지 확인하세요.")
        except Exception as e:
            logger.error(f"TTS 변환 실패: {e}")

    async def _vtube_response(self, text: str) -> None:
        """VTube Studio 립싱크 연동 (step3_vtube 연동)."""
        try:
            from vtube_controller import VTubeController
            ctrl = VTubeController()
            await ctrl.connect()
            await ctrl.set_expression("speaking")
            await ctrl.disconnect()
            logger.info("VTube Studio 표정 전환 완료")
        except ImportError:
            logger.warning("vtube_controller 모듈을 찾을 수 없습니다.")
        except Exception as e:
            logger.error(f"VTube Studio 연동 실패: {e}")
