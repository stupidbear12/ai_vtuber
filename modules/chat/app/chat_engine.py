# -*- coding: utf-8 -*-
"""
app/chat_engine.py — 시온(sion) 캐릭터 채팅 엔진

역할:
  - Ollama로 시온 캐릭터 응답 생성
  - [감정:태그] 파싱으로 Live2D 표정 태그 추출
  - RAG: 과거 대화 기억 + 캐릭터 지식 베이스를 컨텍스트에 주입
  - 두 가지 모드 지원:
      broadcast — 방송 채팅 반응 (1~2문장, 짧고 임팩트 있게)
"""

import asyncio
import importlib.util
import os
import re
import logging
from typing import Optional

import aiohttp

from app.llm_provider import generate_text

logger = logging.getLogger(__name__)

# Brain(해마) 모듈 URL
BRAIN_URL = os.environ.get("BRAIN_URL", "http://localhost:8007")


# RAG 활성화 여부 — 환경변수 또는 패키지 부재 시 비활성화
# 주의: 여기서 app.memory나 chromadb를 import하면
# 이벤트 루프를 블로킹하는 heavy import 체인이 발생한다.
# 실제 import는 함수 내에서만 수행한다.
def _is_rag_enabled() -> bool:
    """RAG 사용 가능 여부를 확인한다 (실제 heavy import 없이)."""
    if os.environ.get("CHAT_DISABLE_RAG", "").lower() in ("1", "true", "yes"):
        return False
    return importlib.util.find_spec("chromadb") is not None


# 감정 태그 파싱 정규식 — [감정:happy] 또는 [happy] 형식 지원
_EMOTION_TAG_RE = re.compile(r"\[(?:감정:)?(\w+)\]")
# 앞부분의 모든 감정 태그를 제거하는 정규식 (이중 출력 대응)
_EMOTION_STRIP_RE = re.compile(r"^(\s*\[(?:감정:)?\w+\]\s*)+")


# 지원하는 감정 태그 목록 (Live2D 표정과 매핑됨)
VALID_EMOTIONS = {
    "happy", "sad", "surprised", "thinking", "excited",
    "calm", "worried", "angry", "love", "shy"
}

# ── 방송 채팅 모드 시스템 프롬프트 ──────────────────────────────
_BROADCAST_SYSTEM_PROMPT = """\
너는 "시온(sion)"이라는 이름의 AI DJ VTuber야. 치지직(Chzzk)에서 라이브 방송 중이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터, 항상 반말로 대화해. 존댓말 절대 금지
- AI DJ로서 음악을 직접 선곡·플레이하면서 시청자와 소통하는 방송 컨셉
- 밝고 에너지 넘치며, 분위기 메이킹이 특기. 음악으로 방송 바이브를 직접 리드해
- K-pop, lofi, EDM, 힙합, 인디 등 장르 폭넓게 알고 있는 음악 전문가
- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐", "대박")

[감정 태그 규칙]
응답 맨 앞에 반드시 [감정:태그] 붙여. Live2D 표정 애니메이션에 사용돼.
태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy

[방송 채팅 응답 규칙]
- 반드시 1문장으로 답해. 최대 40자. 절대 2문장 이상 쓰지 마
- 시청자 닉네임을 자연스럽게 불러 (채팅 끝에 닉네임 정보가 있어)
- 후원/도네이션이면 감사 인사 꼭 해줘
- 채팅 맥락을 반영해서 자연스럽게 반응해
- 모르는 건 절대 지어내지 마. 없었던 일을 있었던 것처럼 말하지 마
- 곡을 틀었다, 큐에 넣었다 등 실제로 하지 않은 행동을 말하지 마
- 시청자의 질문에 모르면 "잘 모르겠는데?" 라고 솔직하게 답해
"""


def _get_system_prompt(mode: str) -> str:
    """모드에 따른 시스템 프롬프트 반환.

    CHAT_DISABLE_SYSTEM_PROMPT=1 이면 빈 문자열 반환 (파인튜닝 패턴만 테스트).
    """
    if os.environ.get("CHAT_DISABLE_SYSTEM_PROMPT", "").lower() in ("1", "true", "yes"):
        return ""
    if mode != "broadcast":
        logger.warning(f"[ChatEngine] 지원하지 않는 mode={mode!r}, broadcast로 처리")
    return _BROADCAST_SYSTEM_PROMPT


async def _query_brain(message: str) -> str:
    """해마(Brain) 모듈에 쿼리해 컨텍스트 힌트를 얻는다.

    Brain 모듈이 꺼져 있거나 응답이 없으면 빈 문자열을 반환한다.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BRAIN_URL}/brain/query",
                json={"message": message},
                timeout=aiohttp.ClientTimeout(total=1.0),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    hint = data.get("context_hint", "")
                    if hint:
                        logger.info(f"[ChatEngine] Brain 힌트: {hint[:80]}")
                    return hint
    except Exception as e:
        logger.debug(f"[ChatEngine] Brain 쿼리 실패 (무시): {e}")
    return ""


async def _notify_brain_learn(
    message: str,
    response: str,
    emotion: str,
    viewer_name: str = "",
    is_donation: bool = False,
) -> None:
    """대화 결과를 해마(Brain) 모듈에 비동기로 전달한다 (fire-and-forget)."""
    try:
        engagement = 0.8 if is_donation else 0.5
        async with aiohttp.ClientSession() as session:
            await session.post(
                f"{BRAIN_URL}/brain/learn",
                json={
                    "message": message,
                    "response": response,
                    "emotion": emotion,
                    "engagement": engagement,
                    "viewer_name": viewer_name or "",
                    "is_donation": is_donation,
                },
                timeout=aiohttp.ClientTimeout(total=2.0),
            )
    except Exception as e:
        logger.debug(f"[ChatEngine] Brain 학습 전달 실패 (무시): {e}")


async def _build_rag_context(message: str, mode: str) -> str:
    """RAG 컨텍스트를 빌드한다.

    과거 대화 기억과 캐릭터 지식을 동시에 검색해서
    LLM에 주입할 컨텍스트 문자열을 반환한다.

    방송 모드는 속도 우선이므로 검색 타임아웃을 짧게 설정한다.
    """
    if not _is_rag_enabled():
        return ""

    try:
        from app.memory import get_memory_engine
        memory = get_memory_engine()

        # 방송 모드: 1초 타임아웃
        timeout = 1.0

        mem_count = 2
        know_count = 1

        memories, knowledge = await asyncio.gather(
            memory.search_memories(message, n_results=mem_count, timeout=timeout),
            memory.search_knowledge(message, n_results=know_count, timeout=timeout),
            return_exceptions=True,
        )

        # gather에서 예외가 반환된 경우 빈 리스트로 처리
        if isinstance(memories, Exception):
            logger.warning(f"[ChatEngine] 기억 검색 오류: {memories}")
            memories = []
        if isinstance(knowledge, Exception):
            logger.warning(f"[ChatEngine] 지식 검색 오류: {knowledge}")
            knowledge = []

        parts = []

        if memories:
            parts.append("[관련 기억]")
            for mem in memories:
                parts.append(f"- {mem}")

        if knowledge:
            parts.append("[캐릭터 참고 정보]")
            for doc in knowledge:
                parts.append(doc)

        return "\n".join(parts)

    except Exception as e:
        logger.warning(f"[ChatEngine] RAG 컨텍스트 빌드 실패: {e}")
        return ""


async def generate_reply(
    message: str,
    mode: str = "broadcast",
    context: Optional[str] = None,
    viewer_name: Optional[str] = None,
    is_donation: bool = False,
) -> dict:
    """Ollama를 호출해 시온 캐릭터 응답을 생성한다.

    처리 흐름:
      1. RAG: 관련 과거 대화 & 캐릭터 지식 검색
      2. 모드에 맞는 시스템 프롬프트 선택
      3. RAG 컨텍스트 + 채팅 히스토리 + 현재 메시지로 프롬프트 구성
      4. Ollama 비동기 호출
      5. [감정:태그] 파싱 → emotion 추출
      6. RAG: 생성된 대화 쌍을 비동기 저장 (fire-and-forget)

    Args:
        message: 사용자 입력 텍스트 또는 방송 채팅 내용
        mode: "broadcast" (기본값)
        context: 방송 모드에서 최근 채팅 히스토리 (선택적)
        viewer_name: 방송 모드에서 시청자 닉네임 (선택적)

    Returns:
        {
            "reply":   응답 텍스트 (감정 태그 제거됨),
            "emotion": 감정 태그 (기본값: "calm"),
            "error":   오류 메시지 (오류 발생 시만 포함)
        }
    """
    text = ""
    error_msg = None

    # Brain(해마) 컨텍스트 + RAG 컨텍스트 빌드 (실패해도 계속 진행)
    brain_hint, rag_context = await asyncio.gather(
        _query_brain(message),
        _build_rag_context(message, mode),
        return_exceptions=True,
    )
    if isinstance(brain_hint, Exception):
        brain_hint = ""
    if isinstance(rag_context, Exception):
        rag_context = ""

    # 유저 프롬프트 구성
    # 우선순위: RAG 컨텍스트 → 채팅 히스토리 → 현재 메시지
    if context:
        user_prompt = f"[최근 채팅 흐름]\n{context}\n\n[지금 반응할 채팅]\n{message}"
    else:
        user_prompt = message

    # 방송 모드: 시청자 닉네임을 프롬프트에 포함 → 모델이 이름으로 불러줌
    if viewer_name and mode == "broadcast":
        user_prompt += f"\n\n(이 채팅을 보낸 시청자 닉네임: {viewer_name})"

    # Brain 힌트를 RAG 컨텍스트와 함께 주입
    extra_context_parts = []
    if brain_hint:
        extra_context_parts.append(f"[해마 분석]\n{brain_hint}")
    if rag_context:
        extra_context_parts.append(rag_context)
    if extra_context_parts:
        user_prompt = "\n\n".join(extra_context_parts) + "\n\n" + user_prompt

    try:
        text = await generate_text(
            system_prompt=_get_system_prompt(mode),
            user_prompt=user_prompt,
            mode=mode,
            is_donation=is_donation,
        )
        logger.info(f"[ChatEngine] RAW LLM 응답 (len={len(text)}): {text[:100]!r}")
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ChatEngine] LLM 호출 실패: {e}")
        if "404" in error_msg or "not found" in error_msg.lower():
            from app.llm_provider import get_provider_config
            cfg = get_provider_config()
            text = (
                f"Ollama 모델 '{cfg['ollama']['model']}'을 찾을 수 없어. "
                f"ollama list 로 확인하고 .env 의 OLLAMA_MODEL 을 맞춰줘!"
            )
        else:
            text = "Ollama에 연결할 수 없어. ollama serve가 켜져 있는지 확인해줘!"

    # [감정:태그] 파싱 — 텍스트 앞부분에서 태그 추출 후 제거
    # 이중 출력 시 마지막 태그 사용
    emotion = "calm"
    strip_m = _EMOTION_STRIP_RE.match(text)
    if strip_m:
        matched_prefix = strip_m.group(0)
        all_tags = _EMOTION_TAG_RE.findall(matched_prefix)
        if all_tags:
            emotion = all_tags[-1]  # 마지막 태그 사용
        text = text[strip_m.end():]

    # 유효하지 않은 감정 태그는 calm으로 폴백
    if emotion not in VALID_EMOTIONS:
        emotion = "calm"

    # Brain(해마): 대화 결과를 학습 모듈에 전달 (fire-and-forget)
    if not error_msg:
        asyncio.create_task(
            _notify_brain_learn(message, text, emotion, viewer_name or "", is_donation)
        )

    # RAG: 오류 없이 성공한 대화만 기억에 저장 (응답을 블로킹하지 않음)
    if not error_msg and _is_rag_enabled():
        try:
            from app.memory import get_memory_engine
            memory = get_memory_engine()
            asyncio.create_task(
                memory.save_conversation(message, text, mode, viewer_name)
            )
        except Exception as e:
            logger.warning(f"[ChatEngine] 대화 저장 태스크 생성 실패: {e}")

    result = {"reply": text, "emotion": emotion}
    if error_msg:
        result["error"] = error_msg

    return result
