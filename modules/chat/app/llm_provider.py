# -*- coding: utf-8 -*-
"""
app/llm_provider.py — Ollama LLM 연동

환경변수:
  OLLAMA_BASE_URL  — 기본 http://localhost:11434
  OLLAMA_MODEL     — 기본 llama3.2
"""

import os

import aiohttp


def get_provider_config() -> dict:
    """헬스 체크·디버그용 Ollama 설정 요약."""
    return {
        "provider": "ollama",
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
    }


def _ollama_options(mode: str) -> dict:
    if mode == "broadcast":
        return {"temperature": 0.8, "num_predict": 300}
    return {"temperature": 0.7, "num_predict": 800}


async def generate_text(
    system_prompt: str,
    user_prompt: str,
    mode: str = "pet",
) -> str:
    """Ollama API로 텍스트 응답을 생성한다."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")

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
