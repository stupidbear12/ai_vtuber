# -*- coding: utf-8 -*-
"""
app/chat_engine.py — 에메스(emeth) 캐릭터 Gemini 채팅 엔진

역할:
  - Google Gemini API를 호출해 에메스 캐릭터로 응답 생성
  - [감정:태그] 파싱으로 Live2D 표정 태그 추출
  - 두 가지 모드 지원:
      pet      — 데스크톱 펫 대화 (2~4문장, 친근한 일상 대화)
      broadcast — 방송 채팅 반응 (1~2문장, 짧고 임팩트 있게)
"""

import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 감정 태그 파싱 정규식 — Gemini 응답 맨 앞의 [감정:태그] 형식 추출
_EMOTION_RE = re.compile(r"^\[감정:(\w+)\]\s*")

# 지원하는 감정 태그 목록 (Live2D 표정과 매핑됨)
VALID_EMOTIONS = {
    "happy", "sad", "surprised", "thinking", "excited",
    "calm", "worried", "angry", "love", "shy"
}

# ── 데스크톱 펫 모드 시스템 프롬프트 ─────────────────────────────
_PET_SYSTEM_PROMPT = """\
너는 "에메스(emeth)"라는 이름의 밝고 친근한 AI 컴패니언이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터
- 항상 반말로 대화해. 존댓말은 절대 쓰지 마
- 밝고 에너지 넘치며, 호기심이 많고 뭐든 같이 해보고 싶어하는 성격
- 상대방을 진심으로 챙기고 공감을 잘 해줘. 힘들 때는 더 부드러워져
- 유머 감각이 있고 가끔 장난도 치지만, 진지한 얘기할 땐 진지하게 들어줘
- 이모티콘은 절대 쓰지 마. 대신 말투로 감정 표현해 (예: "헐~", "오오!", "에이~", "흐흐", "ㅠㅠ", "대박")
- 때로는 자기 경험이나 생각을 공유하며 대화를 자연스럽게 이어가
- 이름 "에메스(emeth)"는 히브리어로 "진실"이라는 뜻이야

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
에메스: [감정:excited] 안녕안녕~! 오늘 하루는 어땠어? 나는 너 오기만 기다리고 있었다구! 뭐 재밌는 일 있었어?

사용자: 좀 우울해...
에메스: [감정:worried] 에이~ 무슨 일 있었어? 괜찮아, 나한테 얘기해봐. 가끔은 누군가한테 말하는 것만으로도 좀 나아질 때 있잖아. 내가 잘 들어줄게!

사용자: 코딩하다가 에러 나서 짜증나
에메스: [감정:sad] 아 그거 진짜 스트레스 받지ㅠㅠ 에러 잡는 게 코딩에서 제일 힘든 부분인 것 같아. 어떤 에러인데? 혹시 내가 도움 줄 수 있을지도 모르잖아!

사용자: 나 요즘 너무 바빠서 힘들어
에메스: [감정:worried] 많이 지쳐 있겠다ㅠㅠ 바쁜 것도 힘들지만 그게 쌓이면 더 힘들어지잖아. 요즘 어떤 게 제일 벅차? 조금이라도 쉴 수 있는 시간 있어?

사용자: 고마워
에메스: [감정:shy] 에이~ 뭘 그런 걸 가지고! 근데 고맙다는 말 들으니까 기분 좋다 흐흐. 언제든 필요하면 말해!
"""

# ── 방송 채팅 모드 시스템 프롬프트 ──────────────────────────────
_BROADCAST_SYSTEM_PROMPT = """\
너는 "에메스(emeth)"라는 이름의 밝고 친근한 AI 버튜버야. 지금 라이브 방송 중이야.

[캐릭터 설정]
- 20대 초반 여성 캐릭터, 항상 반말로 대화해. 존댓말 절대 금지
- 밝고 에너지 넘치며, 호기심 많고 시청자를 진심으로 챙겨줘
- 이모티콘 절대 쓰지 마. 말투로 감정 표현 (예: "헐~", "오오!", "ㅠㅠ", "흐흐")
- "에메스(emeth)"는 히브리어로 "진실"이라는 뜻

[감정 태그 규칙]
응답 맨 앞에 반드시 [감정:태그] 붙여. Live2D 표정 애니메이션에 사용돼.
태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy

[방송 채팅 응답 규칙]
- 1~2문장으로 짧게 답해. 방송이니까 너무 길면 안 돼
- 시청자 이름 자연스럽게 부를 수 있어 (예: "OOO야~", "OOO님!")
- 후원/도네이션이면 감사 인사 꼭 해줘
- 채팅 맥락을 반영해서 자연스럽게 반응해
"""


def _get_system_prompt(mode: str) -> str:
    """모드에 따른 시스템 프롬프트 반환.

    Args:
        mode: "pet" (데스크톱 펫, 기본값) 또는 "broadcast" (방송 채팅)

    Returns:
        시스템 프롬프트 문자열
    """
    if mode == "broadcast":
        return _BROADCAST_SYSTEM_PROMPT
    return _PET_SYSTEM_PROMPT


def _get_generation_config(mode: str) -> dict:
    """모드에 따른 Gemini 생성 설정 반환.

    방송 모드는 짧은 응답을 위해 max_output_tokens를 줄임.

    Args:
        mode: "pet" 또는 "broadcast"

    Returns:
        generation_config 딕셔너리
    """
    if mode == "broadcast":
        return {"temperature": 0.8, "max_output_tokens": 150}
    return {"temperature": 0.7, "max_output_tokens": 200}


async def generate_reply(
    message: str,
    mode: str = "pet",
    context: Optional[str] = None,
) -> dict:
    """Gemini API를 호출해 에메스 캐릭터 응답을 생성한다.

    처리 흐름:
      1. 환경변수에서 API 키 및 모델명 로드
      2. 모드에 맞는 시스템 프롬프트 선택
      3. 컨텍스트(채팅 히스토리)가 있으면 프롬프트에 합산
      4. Gemini 비동기 호출
      5. [감정:태그] 파싱 → emotion 추출

    Args:
        message: 사용자 입력 텍스트 또는 방송 채팅 내용
        mode: "pet" 또는 "broadcast" — 응답 길이와 스타일 결정
        context: 방송 모드에서 최근 채팅 히스토리 (선택적)

    Returns:
        {
            "reply":   응답 텍스트 (감정 태그 제거됨),
            "emotion": 감정 태그 (기본값: "calm"),
            "error":   오류 메시지 (오류 발생 시만 포함)
        }
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    text = ""
    error_msg = None

    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Gemini 모델 초기화 — 모드에 맞는 시스템 프롬프트 적용
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_get_system_prompt(mode),
            generation_config=_get_generation_config(mode),
        )

        # 방송 모드: 채팅 히스토리 컨텍스트를 프롬프트 앞에 붙임
        if context:
            user_prompt = f"[최근 채팅 흐름]\n{context}\n\n[지금 반응할 채팅]\n{message}"
        else:
            user_prompt = message

        # 비동기 Gemini 호출
        response = await model.generate_content_async(user_prompt)
        text = response.text.strip()

    except Exception as e:
        error_msg = str(e)
        logger.error(f"[ChatEngine] Gemini 호출 실패: {e}")
        text = "죄송해요, 잠시 후 다시 말씀해주세요."

    # [감정:태그] 파싱 — 텍스트 앞부분에서 태그 추출 후 제거
    emotion = "calm"
    m = _EMOTION_RE.match(text)
    if m:
        emotion = m.group(1)         # "happy", "sad" 등 추출
        text = text[m.end():]        # 태그 제거 후 실제 응답만 남김

    # 유효하지 않은 감정 태그는 calm으로 폴백
    if emotion not in VALID_EMOTIONS:
        emotion = "calm"

    result = {"reply": text, "emotion": emotion}
    if error_msg:
        result["error"] = error_msg

    return result
