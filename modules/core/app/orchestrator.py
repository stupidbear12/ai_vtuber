# -*- coding: utf-8 -*-
"""
app/orchestrator.py — ai_vtuber_core 모듈 간 연동 오케스트레이터

역할:
  - 각 모듈(ai_live2d, ai_chat, ai_broadcast, ai_voice) 상태 조회
  - 모듈 URL 환경변수 관리 및 기본값 제공
  - 통합 채팅 파이프라인 실행 (chat → live2d emotion → voice TTS)
  - 방송 채팅 시작/중지 위임 (→ ai_broadcast)

모듈 포트 기본값:
  ai_live2d     → 8001
  ai_chat       → 8002
  ai_broadcast  → 8003
  ai_voice      → 8004
"""

import asyncio
import os
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class ModuleConfig:
    """각 모듈의 URL을 환경변수 또는 기본값에서 로드하는 설정 클래스."""

    @staticmethod
    def live2d_url() -> str:
        return os.environ.get("AI_LIVE2D_URL", "http://localhost:8001")

    @staticmethod
    def chat_url() -> str:
        return os.environ.get("AI_CHAT_URL", "http://localhost:8002")

    @staticmethod
    def broadcast_url() -> str:
        return os.environ.get("AI_BROADCAST_URL", "http://localhost:8003")

    @staticmethod
    def voice_url() -> str:
        return os.environ.get("AI_VOICE_URL", "http://localhost:8004")


async def _get_module_health(session: aiohttp.ClientSession, name: str, url: str) -> dict:
    """단일 모듈에 /health 요청을 보내 상태를 조회한다.

    연결 실패나 타임아웃 시 오류 정보를 반환한다.

    Args:
        session: 재사용할 aiohttp 세션
        name: 모듈 이름 (표시용)
        url: 모듈 베이스 URL

    Returns:
        {"name": ..., "status": "ok" | "error", ...}
    """
    try:
        async with session.get(
            f"{url.rstrip('/')}/health",
            timeout=aiohttp.ClientTimeout(total=3.0),
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                return {"name": name, "url": url, **data}
            return {"name": name, "url": url, "status": "error", "http_status": resp.status}
    except aiohttp.ClientConnectorError:
        return {"name": name, "url": url, "status": "offline", "error": "연결 거부됨"}
    except asyncio.TimeoutError:
        return {"name": name, "url": url, "status": "timeout", "error": "응답 시간 초과"}
    except Exception as e:
        return {"name": name, "url": url, "status": "error", "error": str(e)}


async def check_all_modules() -> dict:
    """모든 모듈의 헬스 상태를 병렬로 조회한다.

    4개 모듈에 동시에 /health 요청을 보내 빠르게 전체 상태를 확인한다.

    Returns:
        {
            "modules": {
                "ai_live2d": {...},
                "ai_chat": {...},
                "ai_broadcast": {...},
                "ai_voice": {...},
            },
            "all_online": bool  # 모든 모듈이 ok 상태인지
        }
    """
    cfg = ModuleConfig()

    async with aiohttp.ClientSession() as session:
        # 4개 모듈에 동시에 헬스 체크 요청 (병렬)
        results = await asyncio.gather(
            _get_module_health(session, "ai_live2d",    cfg.live2d_url()),
            _get_module_health(session, "ai_chat",      cfg.chat_url()),
            _get_module_health(session, "ai_broadcast", cfg.broadcast_url()),
            _get_module_health(session, "ai_voice",     cfg.voice_url()),
        )

    modules = {r["name"]: r for r in results}
    all_online = all(m.get("status") == "ok" for m in modules.values())

    return {"modules": modules, "all_online": all_online}


async def run_chat_pipeline(
    message: str,
    mode: str = "pet",
    with_voice: bool = False,
) -> dict:
    """통합 채팅 파이프라인 실행.

    흐름:
      1. ai_chat POST /chat → {reply, emotion} 수신
      2. ai_live2d POST /live2d/emotion → 표정 변경
      3. (선택) ai_voice POST /voice/tts → MP3 음성 생성 후 base64 반환

    Args:
        message: 사용자 채팅 입력
        mode: "pet" (기본값) 또는 "broadcast"
        with_voice: True면 Step 3 TTS 변환을 추가 실행 (기본 False)

    Returns:
        {
            "reply": str,
            "emotion": str,
            "live2d_updated": bool,   # 표정 변경 성공 여부
            "audio_base64": str|None  # MP3 base64 (with_voice=True 시만 포함)
        }
    """
    cfg = ModuleConfig()

    # ── Step 1: ai_chat 호출 ──────────────────────────────────────
    reply = ""
    emotion = "calm"
    chat_error = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cfg.chat_url()}/chat",
                json={"message": message, "mode": mode},
                timeout=aiohttp.ClientTimeout(total=60.0),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"ai_chat 응답 오류: HTTP {resp.status}")
                data = await resp.json()
                reply = data.get("reply", "")
                emotion = data.get("emotion", "calm")
    except Exception as e:
        chat_error = str(e)
        logger.error(f"[Orchestrator] ai_chat 호출 실패: {e}")
        return {
            "reply": "채팅 엔진에 연결할 수 없습니다.",
            "emotion": "worried",
            "live2d_updated": False,
            "audio_base64": None,
            "error": chat_error,
        }

    # ── Step 2: ai_live2d 표정 변경 ──────────────────────────────
    live2d_updated = False
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cfg.live2d_url()}/live2d/emotion",
                json={"emotion": emotion},
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                live2d_updated = resp.status == 200
    except Exception as e:
        logger.warning(f"[Orchestrator] ai_live2d 표정 변경 실패 (무시): {e}")

    # ── Step 3: ai_voice TTS 변환 (선택적) ───────────────────────
    # with_voice=True 일 때만 실행. 실패해도 채팅 응답은 정상 반환.
    # 오디오는 base64 인코딩된 MP3로 반환 (짧은 응답 기준 ~100~200KB).
    audio_base64: Optional[str] = None
    if with_voice and reply:
        try:
            import base64
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{cfg.voice_url()}/voice/tts",
                    json={"text": reply, "emotion": emotion},
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status == 200:
                        audio_bytes = await resp.read()
                        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        except Exception as e:
            logger.warning(f"[Orchestrator] ai_voice TTS 변환 실패 (무시): {e}")

    return {
        "reply": reply,
        "emotion": emotion,
        "live2d_updated": live2d_updated,
        "audio_base64": audio_base64,  # None이면 TTS 미사용 또는 실패
    }


async def start_broadcast(platform: str, channel_id: str) -> dict:
    """ai_broadcast 모듈에 방송 채팅 수집 시작을 요청한다.

    Args:
        platform: "youtube" 또는 "chzzk"
        channel_id: 영상/채널 ID

    Returns:
        ai_broadcast의 응답 딕셔너리

    Raises:
        RuntimeError: ai_broadcast 연결 실패 시
    """
    cfg = ModuleConfig()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cfg.broadcast_url()}/broadcast/start",
                json={"platform": platform, "channel_id": channel_id},
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    raise RuntimeError(data.get("detail", f"HTTP {resp.status}"))
                return data
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"ai_broadcast 연결 실패: {e}")


async def stop_broadcast() -> dict:
    """ai_broadcast 모듈에 방송 채팅 수집 중지를 요청한다.

    Returns:
        ai_broadcast의 응답 딕셔너리 (최종 통계 포함)

    Raises:
        RuntimeError: ai_broadcast 연결 실패 시
    """
    cfg = ModuleConfig()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{cfg.broadcast_url()}/broadcast/stop",
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201):
                    raise RuntimeError(data.get("detail", f"HTTP {resp.status}"))
                return data
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"ai_broadcast 연결 실패: {e}")
