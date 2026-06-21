# -*- coding: utf-8 -*-
"""
app/llm_provider.py — LLM 프로바이더 (Ollama + Gemini 하이브리드)

환경변수:
  OLLAMA_BASE_URL   — 기본 http://localhost:11434
  OLLAMA_MODEL      — 기본 sion (ollama list 로 확인)
  GEMINI_API_KEY    — Google Gemini API 키 (설정 시 하이브리드 모드 활성화)
  GEMINI_MODEL      — 기본 gemini-2.5-flash
  LLM_PROVIDER      — 강제 지정: "ollama" | "gemini" | "hybrid" (기본 hybrid)

하이브리드 모드:
  - is_donation=True 또는 is_important=True → Gemini API
  - 그 외 일반 채팅 → Ollama (로컬, 무료)
  - Gemini 실패 시 자동으로 Ollama 폴백
"""

import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── 설정 ────────────────────────────────────────────────────────────

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"


def get_provider_config() -> dict:
    """헬스 체크·디버그용 설정 요약."""
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    provider = os.environ.get("LLM_PROVIDER", "hybrid")
    return {
        "provider": provider,
        "ollama": {
            "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            "model": os.environ.get("OLLAMA_MODEL", "sion"),
        },
        "gemini": {
            "model": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            "configured": bool(gemini_key),
        },
    }


# ── Ollama ──────────────────────────────────────────────────────────

def _ollama_options(mode: str) -> dict:
    if mode == "broadcast":
        return {"temperature": 0.8, "num_predict": 300}
    return {"temperature": 0.7, "num_predict": 800}


async def _generate_ollama(
    system_prompt: str,
    user_prompt: str,
    mode: str = "pet",
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


# ── Gemini ──────────────────────────────────────────────────────────

def _gemini_generation_config(mode: str) -> dict:
    """모드별 Gemini 생성 파라미터."""
    if mode == "broadcast":
        return {"temperature": 0.8, "maxOutputTokens": 500}
    return {"temperature": 0.7, "maxOutputTokens": 1000}


async def _generate_gemini(
    system_prompt: str,
    user_prompt: str,
    mode: str = "pet",
) -> str:
    """Google Gemini API로 텍스트 응답을 생성한다."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 설정되지 않았습니다.")

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"{_GEMINI_BASE}/models/{model}:generateContent"

    gen_config = _gemini_generation_config(mode)
    # Gemini 2.5 Flash thinking 예산 제한 — 실제 응답에 집중
    gen_config["thinkingConfig"] = {"thinkingBudget": 128}

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
        "systemInstruction": {
            "parts": [{"text": system_prompt}],
        },
        "generationConfig": gen_config,
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60.0),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Gemini 응답 오류 HTTP {resp.status}: {body[:300]}"
                )
            data = await resp.json()

            # Gemini 응답 파싱
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini 응답에 candidates가 없습니다.")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise RuntimeError("Gemini 응답에 parts가 없습니다.")

            return parts[0].get("text", "").strip()


# ── 통합 인터페이스 ────────────────────────────────────────────────

def _select_provider(
    is_donation: bool = False,
    is_important: bool = False,
) -> str:
    """요청 특성에 따라 사용할 프로바이더를 결정한다."""
    forced = os.environ.get("LLM_PROVIDER", "").lower()
    if forced in ("ollama", "gemini"):
        return forced

    # hybrid 모드 (기본)
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return "ollama"  # Gemini 키 없으면 항상 Ollama

    if is_donation or is_important:
        return "gemini"

    return "ollama"


async def generate_text(
    system_prompt: str,
    user_prompt: str,
    mode: str = "pet",
    is_donation: bool = False,
    is_important: bool = False,
) -> str:
    """LLM 응답을 생성한다.

    하이브리드 모드에서는 후원/중요 채팅은 Gemini,
    일반 채팅은 Ollama로 라우팅한다.
    Gemini 실패 시 Ollama로 자동 폴백한다.

    Args:
        system_prompt: 시스템 프롬프트
        user_prompt: 사용자 입력
        mode: "pet" 또는 "broadcast"
        is_donation: 후원/도네이션 메시지 여부
        is_important: 중요 메시지 여부 (키워드 매칭 등)

    Returns:
        생성된 텍스트 응답
    """
    provider = _select_provider(is_donation, is_important)

    if provider == "gemini":
        try:
            result = await _generate_gemini(system_prompt, user_prompt, mode)
            logger.info(f"[LLM] Gemini 응답 생성 완료 (mode={mode}, donation={is_donation})")
            return result
        except Exception as e:
            logger.warning(f"[LLM] Gemini 실패, Ollama 폴백: {e}")
            # Gemini 실패 시 Ollama로 폴백
            return await _generate_ollama(system_prompt, user_prompt, mode)

    result = await _generate_ollama(system_prompt, user_prompt, mode)
    logger.info(f"[LLM] Ollama 응답 생성 완료 (mode={mode})")
    return result
