# -*- coding: utf-8 -*-
"""
app/vision.py — Gemini Flash 비전 API 호출 (OBSERVE 단계)

역할:
  - Playwright 스크린샷(base64)을 Gemini Flash API에 전달
  - 화면에 보이는 요소(텍스트, 버튼, 입력 필드 등)를 자연어로 설명받음

Gemini 무료 티어 제한: 하루 1500 RPD
"""

import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    f"/{GEMINI_MODEL}:generateContent"
)

OBSERVE_PROMPT = (
    "이 브라우저 스크린샷을 분석해줘. 다음 정보를 포함해서 설명해:\n"
    "1. 현재 어떤 페이지인지 (URL 바, 제목 등)\n"
    "2. 화면에 보이는 주요 텍스트 내용\n"
    "3. 클릭 가능한 요소들 (버튼, 링크, 메뉴 등)과 위치\n"
    "4. 입력 필드가 있다면 어떤 입력을 받는지\n"
    "5. 검색 결과가 있다면 주요 결과 요약\n"
    "한국어로 답변해줘."
)


async def observe_screenshot(screenshot_b64: str) -> str:
    """스크린샷을 Gemini Flash API에 보내 화면 상태를 텍스트로 설명받는다.

    Args:
        screenshot_b64: PNG 스크린샷의 base64 인코딩 문자열

    Returns:
        화면 상태 설명 텍스트. 실패 시 에러 메시지 문자열 반환.
    """
    if not GEMINI_API_KEY:
        return "[OBSERVE 오류] GEMINI_API_KEY가 설정되지 않았습니다."

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": OBSERVE_PROMPT},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": screenshot_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                GEMINI_API_URL,
                json=payload,
                params={"key": GEMINI_API_KEY},
                timeout=aiohttp.ClientTimeout(total=30.0),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "[OBSERVE] Gemini API 오류: HTTP %s: %s",
                        resp.status, body[:300],
                    )
                    return f"[OBSERVE 오류] Gemini API HTTP {resp.status}"

                data = await resp.json()

        # 응답에서 텍스트 추출
        candidates = data.get("candidates", [])
        if not candidates:
            return "[OBSERVE 오류] Gemini 응답에 candidates가 없습니다."

        parts = candidates[0].get("content", {}).get("parts", [])
        text_parts = [p["text"] for p in parts if "text" in p]
        description = "\n".join(text_parts).strip()

        if not description:
            return "[OBSERVE 오류] Gemini 응답이 비어 있습니다."

        logger.info("[OBSERVE] 화면 분석 완료 (%d자)", len(description))
        return description

    except aiohttp.ClientError as e:
        logger.error("[OBSERVE] Gemini API 연결 실패: %s", e)
        return f"[OBSERVE 오류] 네트워크 오류: {e}"
    except Exception as e:
        logger.error("[OBSERVE] 예상치 못한 오류: %s", e, exc_info=True)
        return f"[OBSERVE 오류] {e}"
