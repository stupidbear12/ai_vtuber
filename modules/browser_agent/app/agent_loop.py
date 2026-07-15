# -*- coding: utf-8 -*-
"""
app/agent_loop.py — OBSERVE→THINK→ACT 메인 루프

역할:
  - BrowserController, vision, brain을 조합해 에이전트 루프 실행
  - 최대 MAX_STEPS회 반복 후 자동 종료 (무한루프 방지)
  - 각 스텝마다 코멘터리를 TTS→Live2D 파이프라인으로 전달
"""

import asyncio
import base64
import logging
import os
import time
from typing import Optional

import aiohttp

from .browser_controller import BrowserController
from .vision import observe_screenshot
from .brain import think

logger = logging.getLogger(__name__)

MAX_STEPS = int(os.environ.get("BROWSER_MAX_STEPS", "10"))
VOICE_URL = os.environ.get("AI_VOICE_URL", "http://localhost:8004")
LIVE2D_URL = os.environ.get("AI_LIVE2D_URL", "http://localhost:8001")


class AgentLoop:
    """OBSERVE→THINK→ACT 에이전트 루프 관리.

    상태: idle → running → idle
    동시에 하나의 명령만 실행 가능.
    """

    def __init__(self, browser: BrowserController):
        self._browser = browser
        self._status = "idle"       # idle, running
        self._current_command = ""
        self._current_step = 0
        self._task: Optional[asyncio.Task] = None
        self._cancel_requested = False

    @property
    def status(self) -> str:
        return self._status

    @property
    def current_command(self) -> str:
        return self._current_command

    @property
    def current_step(self) -> int:
        return self._current_step

    async def execute(self, command: str, requester: str = "") -> dict:
        """명령을 실행한다.

        Args:
            command: 시청자 명령 (예: "네이버에서 오늘 날씨 검색해줘")
            requester: 요청자 닉네임

        Returns:
            {"success", "steps", "final_comment", "error"} 결과 딕셔너리
        """
        if self._status == "running":
            return {"success": False, "error": "이미 명령 실행 중입니다."}

        if not self._browser.is_running:
            return {"success": False, "error": "브라우저가 실행 중이 아닙니다."}

        self._status = "running"
        self._current_command = command
        self._current_step = 0
        self._cancel_requested = False

        # 시작 코멘터리
        start_comment = f"{requester}님이 요청하신 '{command}' 한번 해볼게요!"
        await self._send_commentary(start_comment, "excited")

        history = []
        final_comment = ""
        steps_completed = 0

        try:
            for step in range(1, MAX_STEPS + 1):
                if self._cancel_requested:
                    final_comment = "명령이 취소되었어요."
                    break

                self._current_step = step
                logger.info("[AgentLoop] ── 스텝 %d/%d ──", step, MAX_STEPS)

                # ── OBSERVE ──
                try:
                    screenshot_b64 = await self._browser.screenshot_base64()
                except Exception as e:
                    logger.error("[AgentLoop] 스크린샷 실패: %s", e)
                    final_comment = "스크린샷을 찍는 데 실패했어요..."
                    break

                screen_description = await observe_screenshot(screenshot_b64)
                if screen_description.startswith("[OBSERVE 오류]"):
                    logger.warning("[AgentLoop] OBSERVE 실패: %s", screen_description)
                    # 에러가 있어도 계속 시도 (간단한 설명이라도 전달)

                # ── THINK ──
                action_data = await think(
                    screen_description=screen_description,
                    user_command=command,
                    step=step,
                    max_steps=MAX_STEPS,
                    history=history,
                )

                action = action_data["action"]
                target = action_data["target"]
                value = action_data["value"]
                comment = action_data["comment"]

                history.append(action_data)
                steps_completed = step

                # ── 코멘터리 전송 ──
                if comment:
                    emotion = _action_to_emotion(action)
                    await self._send_commentary(comment, emotion)

                # ── ACT ──
                if action == "done":
                    final_comment = comment or "작업 완료!"
                    logger.info("[AgentLoop] done — 루프 종료")
                    break

                try:
                    await self._execute_action(action, target, value)
                except Exception as e:
                    logger.warning("[AgentLoop] ACT 실패: %s", e)
                    # 액션 실패해도 루프는 계속 (다음 OBSERVE에서 상태 확인)

                # 액션 후 짧은 대기 (페이지 로딩)
                await asyncio.sleep(1.0)

            else:
                # MAX_STEPS 도달
                final_comment = f"최대 {MAX_STEPS}스텝까지 했는데 아직 끝나지 않았어요. 여기서 멈출게요!"
                await self._send_commentary(final_comment, "worried")

        except asyncio.CancelledError:
            final_comment = "명령이 취소되었어요."
        except Exception as e:
            logger.error("[AgentLoop] 루프 오류: %s", e, exc_info=True)
            final_comment = f"에러가 발생했어요: {e}"
        finally:
            self._status = "idle"
            self._current_command = ""
            self._current_step = 0

        return {
            "success": True,
            "steps": steps_completed,
            "final_comment": final_comment,
        }

    def cancel(self) -> None:
        """현재 실행 중인 명령을 취소 요청한다."""
        self._cancel_requested = True

    async def _execute_action(
        self, action: str, target: str, value: str,
    ) -> None:
        """THINK에서 결정한 액션을 실제 브라우저에서 실행한다."""
        if action == "navigate":
            await self._browser.navigate(value)
        elif action == "click":
            await self._browser.click(target)
        elif action == "type":
            await self._browser.type_text(target, value)
            # type 후 자동 Enter (검색 등)
            await self._browser.press_enter()
        elif action == "scroll":
            direction = value if value in ("up", "down") else "down"
            await self._browser.scroll(direction)
        else:
            logger.warning("[AgentLoop] 알 수 없는 action: %s", action)

    async def _send_commentary(self, text: str, emotion: str = "calm") -> None:
        """코멘터리를 TTS + Live2D 파이프라인으로 전달한다.

        broadcast 모듈의 _announce_music 패턴과 동일.
        """
        if not text:
            return

        # TTS 생성
        audio_base64: Optional[str] = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{VOICE_URL}/voice/tts",
                    json={"text": text, "emotion": emotion},
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.warning("[AgentLoop] TTS 실패 (무시): %s", e)

        # Live2D 표정 + 브로드캐스트
        try:
            async with aiohttp.ClientSession() as session:
                await session.post(
                    f"{LIVE2D_URL}/live2d/emotion",
                    json={"emotion": emotion},
                    timeout=aiohttp.ClientTimeout(total=3.0),
                )
                broadcast_payload: dict = {
                    "cmd": "speak",
                    "text": text,
                    "emotion": emotion,
                    "author": "시온",
                    "platform": "browser",
                    "is_donation": False,
                }
                if audio_base64:
                    broadcast_payload["audio_base64"] = audio_base64
                await session.post(
                    f"{LIVE2D_URL}/live2d/broadcast",
                    json=broadcast_payload,
                    timeout=aiohttp.ClientTimeout(total=5.0),
                )
        except Exception as e:
            logger.warning("[AgentLoop] Live2D 전송 실패 (무시): %s", e)


def _action_to_emotion(action: str) -> str:
    """액션에 따라 적절한 감정 태그를 반환한다."""
    return {
        "navigate": "excited",
        "click": "happy",
        "type": "thinking",
        "scroll": "calm",
        "done": "happy",
    }.get(action, "calm")
