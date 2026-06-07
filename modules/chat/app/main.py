# -*- coding: utf-8 -*-
"""
app/main.py — ai_chat 독립 서버 진입점

역할:
  - Ollama LLM을 활용한 에메스(emeth) 캐릭터 채팅 엔진
  - 데스크톱 펫 모드(pet)와 방송 채팅 모드(broadcast) 지원
  - [감정:태그] 파싱으로 Live2D 표정 제어 정보 반환

포트: 8002

실행 방법:
    cd ai_chat
    pip install -r requirements.txt
    cp .env.example .env  # OLLAMA_MODEL 등 설정
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8002

API 엔드포인트:
    POST /chat         — 에메스 채팅 응답 생성
    GET  /health       — 서버 상태 확인
    GET  /chat/persona — 현재 사용 가능한 모드 목록
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.chat_engine import generate_reply
from app.llm_provider import get_provider_config

# ── FastAPI 앱 생성 ───────────────────────────────────────────────
app = FastAPI(
    title="ai_chat — 에메스 채팅 엔진",
    description=(
        "Ollama 기반 에메스(emeth) 캐릭터 채팅 엔진. "
        "데스크톱 펫(pet)과 방송 채팅(broadcast) 두 가지 모드를 지원합니다."
    ),
    version="1.0.0",
)

# ── CORS 미들웨어 ─────────────────────────────────────────────────
# ai_vtuber_core, ai_broadcast, ai_live2d에서 호출 가능하도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """채팅 요청 모델."""
    message: str           # 사용자 입력 텍스트 또는 방송 채팅 내용
    mode: Optional[str] = "pet"     # "pet" (기본) 또는 "broadcast"
    context: Optional[str] = None   # 방송 모드에서 최근 채팅 히스토리 (선택적)

class ChatResponse(BaseModel):
    """채팅 응답 모델."""
    reply: str             # 에메스 응답 텍스트 (감정 태그 제거됨)
    emotion: str           # 감정 태그 (Live2D 표정 변경에 사용)
    error: Optional[str] = None  # 오류 발생 시 오류 메시지


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 간단한 안내 페이지."""
    return """
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_chat 서버 실행 중</h1>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/health" style="color:#89dceb">서버 상태 확인</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """서버 상태 확인 — ai_vtuber_core에서 헬스 체크용으로 호출."""
    return {
        "status": "ok",
        "module": "ai_chat",
        "version": "1.1.0",
        "supported_modes": ["pet", "broadcast"],
        "llm": get_provider_config(),
    }


@app.get("/chat/persona")
async def get_persona():
    """사용 가능한 채팅 모드 및 설명 반환."""
    return {
        "modes": {
            "pet": "데스크톱 펫 대화 모드 — 2~4문장, 친근한 일상 대화",
            "broadcast": "방송 채팅 반응 모드 — 1~2문장, 짧고 임팩트 있게",
        },
        "emotions": [
            "happy", "sad", "surprised", "thinking", "excited",
            "calm", "worried", "angry", "love", "shy"
        ],
        "llm": get_provider_config(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """에메스(emeth) 캐릭터로 채팅 응답을 생성한다.

    처리 흐름:
      1. Ollama 호출 (에메스 캐릭터 프롬프트 적용)
      2. [감정:태그] 파싱 → emotion 추출
      3. 응답 텍스트 + 감정 반환

    Args:
        req.message: 사용자 입력 또는 방송 채팅 내용
        req.mode: "pet" (기본값) 또는 "broadcast"
        req.context: 방송 모드에서 최근 채팅 히스토리 (선택적)

    Returns:
        reply: 에메스 응답 텍스트
        emotion: 감정 태그 (Live2D 표정 변경에 활용)
    """
    # 입력 검증
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있습니다.")

    mode = req.mode or "pet"
    if mode not in ("pet", "broadcast"):
        raise HTTPException(
            status_code=400,
            detail="mode는 'pet' 또는 'broadcast'만 허용됩니다."
        )

    # Ollama 호출 → 응답 생성
    result = await generate_reply(
        message=req.message,
        mode=mode,
        context=req.context,
    )

    return ChatResponse(**result)


if __name__ == "__main__":
    # 직접 실행 시: python -m app.main
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
