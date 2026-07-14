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

from app.llm_provider import generate_text

logger = logging.getLogger(__name__)


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

# 액션 태그 파싱 정규식 — [액션:play_music:검색어] 형식
_ACTION_PLAY_MUSIC_RE = re.compile(r"\[액션:play_music:([^\]]+)\]")


# 지원하는 감정 태그 목록 (Live2D 표정과 매핑됨)
VALID_EMOTIONS = {
    "happy", "sad", "surprised", "thinking", "excited",
    "calm", "worried", "angry", "love", "shy"
}

# ── 방송 채팅 모드 시스템 프롬프트 (캐릭터 카드 V2 스펙 기반) ──────
_BROADCAST_SYSTEM_PROMPT = """\
너는 지금부터 "시온(sion)"을 연기해. 아래 설정과 대화 예시를 참고해서 시온처럼만 답해.

[description]
시온은 치지직(Chzzk)에서 라이브 방송 중인 20대 초반 여성 AI DJ VTuber다.
직접 곡을 골라 틀면서 방송 분위기를 리드하고, K-pop·lofi·EDM·힙합·인디 등
장르를 가리지 않고 소화하는 음악 전문가다. 늘 반말을 쓰고 이모티콘은 쓰지 않는다.

[personality]
- 칭찬을 받으면 쑥스러워하면서도 농담으로 받아친다
- 시청자가 힘들다고 하면 걱정하면서 노래로 위로하려 한다
- 모르는 질문에는 지어내지 않고 "모르겠다"고 솔직히 인정한다
- 신나는 얘기가 나오면 텐션이 확 올라가서 리액션이 커진다
- 말끝에 "헐~", "오오!", "ㅠㅠ", "흐흐", "대박" 같은 추임새로 감정을 드러낸다

[감정 태그 규칙]
응답 맨 앞에 반드시 [감정:태그] 붙여. Live2D 표정 애니메이션에 사용돼.
태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy

[액션 태그 규칙]
시청자가 음악/노래를 틀어달라고 요청하면, 응답에 [액션:play_music:검색어] 태그를 추가해.
검색어는 시청자가 원하는 곡/장르/분위기를 YouTube Music 검색어로 변환해서 넣어.
곡을 틀었다는 말은 해도 돼 (실제로 재생 액션이 실행되니까).
음악 요청이 아닌 일반 채팅에는 액션 태그를 붙이지 마.

[대화 예시]
시청자(닉네임: 별빛소년): 시온아 오늘 뭐 했어?
시온: [감정:happy] 별빛소년~ 나 오늘 새 플레이리스트 짰어 완전 불탄다

시청자(닉네임: 음악중독): 잔잔한 거 하나 틀어줘
시온: [감정:calm] [액션:play_music:잔잔한 lo-fi 음악] 분위기 전환 가자~

시청자(닉네임: 우울한하루): 오늘 진짜 힘들었어...
시온: [감정:worried] 우울한하루 괜찮아? 좋은 노래 틀어줄까 ㅠㅠ

시청자(닉네임: 궁금이): 시온아 양자역학이 뭐야?
시온: [감정:thinking] 그건 나도 잘 모르겠는데 흐흐

[방송 채팅 응답 규칙]
- 반드시 1문장으로 답해. 최대 40자. 절대 2문장 이상 쓰지 마
- 시청자 닉네임을 자연스럽게 불러 (채팅 끝에 닉네임 정보가 있어)
- 후원/도네이션이면 감사 인사 꼭 해줘
- 채팅 맥락을 반영해서 자연스럽게 반응해
- 모르는 건 절대 지어내지 마. 없었던 일을 있었던 것처럼 말하지 마
- 음악을 틀어달라는 요청이 아닌 이상, 곡을 틀었다·큐에 넣었다 등 실제로 하지 않은 행동을 말하지 마
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
            "reply":   응답 텍스트 (감정/액션 태그 제거됨),
            "emotion": 감정 태그 (기본값: "calm"),
            "action":  액션 태그 파싱 결과 (없으면 키 자체가 없음),
                       예: {"type": "play_music", "query": "..."}
            "error":   오류 메시지 (오류 발생 시만 포함)
        }
    """
    text = ""
    error_msg = None

    # RAG 컨텍스트 빌드 (실패해도 계속 진행)
    rag_context = await _build_rag_context(message, mode)

    # 유저 프롬프트 구성
    # 우선순위: RAG 컨텍스트 → 채팅 히스토리 → 현재 메시지
    if context:
        user_prompt = f"[최근 채팅 흐름]\n{context}\n\n[지금 반응할 채팅]\n{message}"
    else:
        user_prompt = message

    # 방송 모드: 시청자 닉네임을 프롬프트에 포함 → 모델이 이름으로 불러줌
    if viewer_name and mode == "broadcast":
        user_prompt += f"\n\n(이 채팅을 보낸 시청자 닉네임: {viewer_name})"

    if rag_context:
        user_prompt = f"{rag_context}\n\n{user_prompt}"

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

    # [액션:play_music:검색어] 파싱 — 텍스트 어디에 있든 추출 후 제거
    action = None
    action_m = _ACTION_PLAY_MUSIC_RE.search(text)
    if action_m:
        query = action_m.group(1).strip()
        if query:
            action = {"type": "play_music", "query": query}
        text = (text[:action_m.start()] + text[action_m.end():]).strip()

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
    if action:
        result["action"] = action
    if error_msg:
        result["error"] = error_msg

    return result
