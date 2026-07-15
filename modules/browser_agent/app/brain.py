# -*- coding: utf-8 -*-
"""
app/brain.py — Ollama THINK 로직 (THINK 단계)

역할:
  - OBSERVE에서 받은 화면 설명 + 사용자 명령을 조합
  - Ollama(exagirl 모델)에 보내서 다음 행동 결정
  - JSON 형태의 액션 반환: {"action", "target", "value", "comment"}
"""

import json
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "exagirl")

THINK_SYSTEM_PROMPT = """\
너는 AI VTuber 시온의 브라우저 에이전트야.
시청자의 요청에 따라 브라우저를 조작해서 정보를 찾아주는 역할이야.

현재 브라우저 화면 설명과 사용자 명령을 보고, 다음에 할 행동을 JSON으로 답해줘.

반드시 아래 JSON 형식만 출력해:
{
  "action": "click|type|scroll|navigate|done",
  "target": "클릭할 요소의 텍스트 또는 CSS selector",
  "value": "type일 때 입력할 텍스트, navigate일 때 URL, scroll일 때 up/down",
  "comment": "시청자에게 할 말 (자연스럽게, 뭘 하고 있는지 설명)"
}

action 종류:
- navigate: URL로 이동. target은 비워두고 value에 URL 입력
- click: 요소 클릭. target에 클릭할 버튼/링크 텍스트 또는 selector
- type: 텍스트 입력. target에 입력 필드 selector, value에 입력할 텍스트
- scroll: 페이지 스크롤. value에 "up" 또는 "down"
- done: 작업 완료. comment에 최종 결과 요약

주의사항:
- 검색할 때는 보통 navigate로 검색 사이트에 가서 → click으로 검색창 클릭 → type으로 검색어 입력 → click으로 검색 버튼 클릭
- 네이버 검색: https://www.naver.com 으로 navigate 후 검색
- 구글 검색: https://www.google.com 으로 navigate 후 검색
- 결과를 찾았으면 done으로 마무리
- comment는 반드시 한국어로, 방송 시청자에게 말하는 것처럼 자연스럽게
- JSON만 출력하고 다른 텍스트는 절대 포함하지 마
"""


async def think(
    screen_description: str,
    user_command: str,
    step: int,
    max_steps: int,
    history: Optional[list] = None,
) -> dict:
    """화면 설명과 사용자 명령을 바탕으로 다음 행동을 결정한다.

    Args:
        screen_description: OBSERVE 단계에서 얻은 화면 상태 설명
        user_command: 사용자(시청자)의 원래 명령
        step: 현재 스텝 번호 (1부터)
        max_steps: 최대 스텝 수
        history: 이전 스텝들의 action 히스토리 (선택)

    Returns:
        {"action", "target", "value", "comment"} 딕셔너리.
        파싱 실패 시 action="done"으로 폴백.
    """
    # 히스토리 컨텍스트 구성
    history_text = ""
    if history:
        history_lines = []
        for i, h in enumerate(history, 1):
            history_lines.append(
                f"  스텝 {i}: {h.get('action', '?')} "
                f"→ {h.get('target', '')} {h.get('value', '')}"
            )
        history_text = "\n이전 행동:\n" + "\n".join(history_lines) + "\n"

    user_prompt = (
        f"사용자 명령: {user_command}\n"
        f"\n현재 스텝: {step}/{max_steps}\n"
        f"{history_text}"
        f"\n현재 브라우저 화면:\n{screen_description}\n"
        f"\n다음 행동을 JSON으로 답해줘."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": THINK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": 512,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120.0),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "[THINK] Ollama API 오류: HTTP %s: %s",
                        resp.status, body[:300],
                    )
                    return _fallback_action(
                        f"Ollama 서버에 문제가 있어요... (HTTP {resp.status})"
                    )

                data = await resp.json()

        raw_content = (
            data.get("message", {}).get("content", "").strip()
        )
        if not raw_content:
            return _fallback_action("생각을 정리하는 데 실패했어요...")

        # JSON 파싱
        action_data = _parse_action_json(raw_content)
        logger.info(
            "[THINK] 결정: action=%s target=%s value=%s",
            action_data.get("action"),
            action_data.get("target", "")[:30],
            action_data.get("value", "")[:30],
        )
        return action_data

    except aiohttp.ClientError as e:
        logger.error("[THINK] Ollama 연결 실패: %s", e)
        return _fallback_action("Ollama 서버에 연결할 수 없어요...")
    except Exception as e:
        logger.error("[THINK] 예상치 못한 오류: %s", e, exc_info=True)
        return _fallback_action(f"에러가 발생했어요: {e}")


def _parse_action_json(raw: str) -> dict:
    """LLM 응답에서 JSON을 파싱한다.

    LLM이 JSON 외 텍스트를 포함할 수 있으므로
    중괄호 블록을 추출해서 파싱을 시도한다.
    """
    # 1차: 전체 문자열 파싱
    try:
        result = json.loads(raw)
        return _validate_action(result)
    except json.JSONDecodeError:
        pass

    # 2차: 중괄호 블록 추출
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(raw[start:end + 1])
            return _validate_action(result)
        except json.JSONDecodeError:
            pass

    logger.warning("[THINK] JSON 파싱 실패: %s", raw[:200])
    return _fallback_action("응답을 이해할 수 없어서 멈출게요...")


def _validate_action(data: dict) -> dict:
    """액션 딕셔너리의 필수 필드를 검증하고 정규화한다."""
    valid_actions = {"click", "type", "scroll", "navigate", "done"}
    action = data.get("action", "done")
    if action not in valid_actions:
        action = "done"

    return {
        "action": action,
        "target": str(data.get("target", "")),
        "value": str(data.get("value", "")),
        "comment": str(data.get("comment", "")),
    }


def _fallback_action(comment: str) -> dict:
    """파싱 실패 시 안전한 done 액션을 반환한다."""
    return {
        "action": "done",
        "target": "",
        "value": "",
        "comment": comment,
    }
