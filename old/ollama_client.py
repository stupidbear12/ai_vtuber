# -*- coding: utf-8 -*-
"""old/ 레거시 코드용 Ollama HTTP 클라이언트."""

import os

import aiohttp


async def ollama_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    mode: str = "pet",
) -> str:
    """Ollama /api/chat 호출 후 응답 텍스트 반환."""
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    options = (
        {"temperature": 0.8, "num_predict": 300}
        if mode == "broadcast"
        else {"temperature": 0.7, "num_predict": 800}
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": options,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120.0),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Ollama HTTP {resp.status}: {body[:300]}")
            data = await resp.json()
            return (data.get("message") or {}).get("content", "").strip()
