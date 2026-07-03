# -*- coding: utf-8 -*-
"""
app/llm_provider.py — LLM 프로바이더 (Ollama)

환경변수:
  OLLAMA_BASE_URL   — 기본 http://localhost:11434
  OLLAMA_MODEL      — 기본 sion (ollama list 로 확인)
"""

import logging
import os

import aiohttp

logger = logging.getLogger(__name__)


def get_provider_config() -> dict:
    """헬스 체크·디버그용 설정 요약."""
    return {
        "provider": "ollama",
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.environ.get("OLLAMA_MODEL", "sion"),
        },
    }


def _ollama_options(mode: str) -> dict:
    return {"temperature": 0.8, "num_predict": 300}


async def _generate_ollama(
    system_prompt: str,
    user_prompt: str,
    mode: str = "broadcast",
) -> str:
    """Ollama API로 텍스트 응답을 생성한다."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "sion")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": _ollama_options(mode),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120.0),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Ollama 응답 오류 HTTP {resp.status}: {body[:300]}"
                )
            data = await resp.json()
            return (data.get("message") or {}).get("content", "").strip()


async def generate_text(
    system_prompt: str,
    user_prompt: str,
    mode: str = "broadcast",
    is_donation: bool = False,
) -> str:
    """Ollama로 LLM 응답을 생성한다.

    Args:
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 입력
        mode: "broadcast"
        is_donation: API 호환용 (라우팅에는 사용하지 않음)

    Returns:
        생성된 텍스트 응답
    """
    _ = is_donation
    result = await _generate_ollama(system_prompt, user_prompt, mode)
    logger.info(f"[LLM] Ollama 응답 생성 완료 (mode={mode})")
    return result
