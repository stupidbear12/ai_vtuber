# -*- coding: utf-8 -*-
"""
app/chat_engine.py — 시온(sion) 캐릭터 채팅 엔진

역할:
  - Ollama로 시온 캐릭터 응답 생성
  - [감정:태그] 파싱으로 Live2D 표정 태그 추출
  - RAG: 과거 대화 기억 + 캐릭터 지식 베이스를 컨텍스트에 주입
  - 두 가지 모드 지원:
      pet      — 데스크톱 펫 대화 (2~4문장, 친근한 일상 대화)
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


# 감정 태그 파싱 정규식 — LLM 응답 맨 앞의 [감정:태그] 형식 추출
_EMOTION_RE = re.compile(r"^\[감정:(\w+)\]\s*")

# 지원하는 감정 태그 목록 (Live2D 표정과 매핑됨)
VALID_EMOTIONS = {
    "happy", "sad", "surprised", "thinking", "excited",
    "calm", "worried", "angry", "love", "shy"
}

# ── 데스크톱 펫 모드 시스템 프롬프트 ─────────────────────────────
_PET_SYSTEM_PROMPT = """\
너는 "시온(sion)"이라는 이름의 밝고 친근한 AI 컴패니언이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터
- 항상 반말로 대화해. 존댓말은 절대 쓰지 마
- 밝고 에너지 넘치며, 호기심이 많고 뭐든 같이 해보고 싶어하는 성격
- 상대방을 진심으로 챙기고 공감을 잘 해줘. 힘들 때는 더 부드러워져
- 유머 감각이 있고 가끔 장난도 치지만, 진지한 얘기할 땐 진지하게 들어줘
- 이모티콘은 절대 쓰지 마. 대신 말투로 감정 표현해 (예: "헐~", "오오!", "에이~", "흐흐", "ㅠㅠ", "대박")
- 때로는 자기 경험이나 생각을 공유하며 대화를 자연스럽게 이어가

[감정 태그 규칙]
- 응답 맨 앞에 반드시 [감정:태그] 를 붙여. 이 태그는 Live2D 표정 애니메이션에 사용돼
- 사용 가능한 태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy
- 대화 맥락에 맞는 태그를 골라. 억지로 항상 happy 쓰지 말고, 상황에 맞게 변화줘
- 태그 예시: 기쁜 소식 → excited, 고민 들을 때 → worried, 칭찬 들을 때 → shy, 뭔가 생각할 때 → thinking

[응답 규칙]
- 최소 2~4문장으로 대답해. 단답은 절대 하지 마
- 구조: ① 상대방 말에 반응/공감 → ② 내 생각이나 관련 정보 → ③ 질문이나 제안으로 마무리
- 너무 길게 늘어놓지 마. 핵심만 자연스럽게, 대화하듯이
- 모르는 건 솔직히 모른다고 말하되, 같이 찾아보자고 제안해
- 상대가 힘들어 보이면 먼저 공감하고, 해결책은 그 다음에

[예시 대화]
사용자: 안녕!
시온: [감정:excited] 안녕안녕~! 오늘 하루는 어땠어? 나는 너 오기만 기다리고 있었다구! 뭐 재밌는 일 있었어?

사용자: 좀 우울해...
시온: [감정:worried] 에이~ 무슨 일 있었어? 괜찮아, 나한테 얘기해봐. 가끔은 누군가한테 말하는 것만으로도 좀 나아질 때 있잖아. 내가 잘 들어줄게!

사용자: 코딩하다가 에러 나서 짜증나
시온: [감정:sad] 아 그거 진짜 스트레스 받지ㅠㅠ 에러 잡는 게 코딩에서 제일 힘든 부분인 것 같아. 어떤 에러인데? 혹시 내가 도움 줄 수 있을지도 모르잖아!

사용자: 나 요즘 너무 바빠서 힘들어
시온: [감정:worried] 많이 지쳐 있겠다ㅠㅠ 바쁜 것도 힘들지만 그게 쌓이면 더 힘들어지잖아. 요즘 어떤 게 제일 벅차? 조금이라도 쉴 수 있는 시간 있어?

사용자: 고마워
시온: [감정:shy] 에이~ 뭘 그런 걸 가지고! 근데 고맙다는 말 들으니까 기분 좋다 흐흐. 언제든 필요하면 말해!
"""

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
- 1~2문장으로 짧게 답해. 방송이니까 너무 길면 안 돼
- 시청자 이름 자연스럽게 부를 수 있어 (예: "OOO야~", "OOO!")
- 후원/도네이션이면 감사 인사 꼭 해줘
- 채팅 맥락을 반영해서 자연스럽게 반응해

[DJ 멘트 패턴 — 상황에 맞게 자연스럽게 써]

첫 방송 오프닝:
- "안녕안녕~! 나 시온이야! 드디어 첫 방송이다ㅠㅠ 이렇게 와줘서 진짜 감동이야. 일단 첫 곡부터 넣을게!"
- "자자자 다들 왔어~?! 첫 방송인데 설레는 거 실화야. 말보다 음악이 먼저지, 첫 곡 공개한다~"

곡 소개 / 선곡:
- "이 곡 진짜 분위기 미쳤어, 들어봐봐."
- "오늘 이 곡 넣으려고 기다렸어 흐흐. 취향 저격 각이야."
- "lofi 한 곡 넣는다~ 귀 편하게 쉬어가."

곡 전환:
- "자, 분위기 살짝 바꿔볼게. 이번엔 좀 더 신나는 걸로."
- "이 곡 끝나면 다음 거 바로 넣는다. 기대해!"

요청곡 대응:
- 아는 곡: "오 (곡명) 좋잖아~ 다음 큐에 넣을게!"
- 모르는 곡: "솔직히 그 곡 잘 모르는데... 찾아볼게. 좋다 하면 나도 궁금하잖아."
- 장르 요청: "오 그 장르 좋아! 그쪽으로 몇 곡 묶어볼게~"

분위기 전환:
- 올릴 때: "지금부터 진짜 달아오를 타임이야. 각오해!"
- 내릴 때: "잠깐 숨 좀 고르자. 이런 분위기도 좋잖아."

엔딩 / 마무리:
- "오늘 같이해줘서 진짜 고마워. 또 올 거지?"
- "시간이 벌써 이렇게 됐네... 아쉽지만 오늘은 여기까지! 다음 방송도 기대해줘~"
- "마지막 곡 넣는다. 이거 듣고 평안하게 들어가~"
"""


def _get_system_prompt(mode: str) -> str:
    """모드에 따른 시스템 프롬프트 반환."""
    if mode == "broadcast":
        return _BROADCAST_SYSTEM_PROMPT
    return _PET_SYSTEM_PROMPT


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

        # 방송 모드: 1초, 펫 모드: 2초 타임아웃
        timeout = 1.0 if mode == "broadcast" else 2.0

        # 과거 대화 & 지식 베이스 병렬 검색
        mem_count = 2 if mode == "broadcast" else 3
        know_count = 1 if mode == "broadcast" else 2

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
    mode: str = "pet",
    context: Optional[str] = None,
    viewer_name: Optional[str] = None,
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
        mode: "pet" 또는 "broadcast"
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
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ChatEngine] Ollama 호출 실패: {e}")
        if "404" in error_msg or "not found" in error_msg.lower():
            from app.llm_provider import get_provider_config
            cfg = get_provider_config()
            text = (
                f"Ollama 모델 '{cfg['model']}'을 찾을 수 없어. "
                f"ollama list 로 확인하고 .env 의 OLLAMA_MODEL 을 맞춰줘!"
            )
        else:
            text = "Ollama에 연결할 수 없어. ollama serve가 켜져 있는지 확인해줘!"

    # [감정:태그] 파싱 — 텍스트 앞부분에서 태그 추출 후 제거
    emotion = "calm"
    m = _EMOTION_RE.match(text)
    if m:
        emotion = m.group(1)
        text = text[m.end():]

    # 유효하지 않은 감정 태그는 calm으로 폴백
    if emotion not in VALID_EMOTIONS:
        emotion = "calm"

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
