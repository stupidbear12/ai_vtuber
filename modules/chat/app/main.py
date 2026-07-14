# -*- coding: utf-8 -*-
"""
app/main.py — ai_chat 독립 서버 진입점

역할:
  - Ollama LLM을 활용한 시온(sion) 캐릭터 채팅 엔진
  - 방송 채팅 모드(broadcast) 지원
  - [감정:태그] 파싱으로 Live2D 표정 제어 정보 반환
  - RAG: 과거 대화 기억 + 캐릭터 지식 베이스 연동

포트: 8002

실행 방법:
    cd modules/chat
    pip install -r requirements.txt
    cp .env.example .env  # OLLAMA_MODEL 등 설정
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8002

API 엔드포인트:
    POST /chat          — 시온 채팅 응답 생성
    GET  /health        — 서버 상태 확인
    GET  /chat/persona  — 현재 사용 가능한 모드 목록
    GET  /memory/stats  — RAG 메모리 현황 (디버그용)
"""

import importlib.util
import logging
import os
import uvicorn
from contextlib import asynccontextmanager
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

logger = logging.getLogger(__name__)


# ── 헬퍼 ─────────────────────────────────────────────────────────────

def _is_rag_available() -> bool:
    """RAG 기능 사용 가능 여부 확인.

    CHAT_DISABLE_RAG=1 이면 비활성화.
    chromadb 패키지가 설치되어 있어야 활성화 (HTTP 클라이언트로 Docker ChromaDB에 연결).
    """
    if os.environ.get("CHAT_DISABLE_RAG", "").lower() in ("1", "true", "yes"):
        return False
    return importlib.util.find_spec("chromadb") is not None


# ── 서버 수명주기 ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 훅.

    시작 시: RAG 메모리 엔진을 초기화하고 지식 베이스를 백그라운드로 로드한다.
    로드가 완료되기 전에도 서버는 정상 응답하며, 로드 후 RAG가 활성화된다.
    """
    if os.environ.get("CHAT_DISABLE_RAG", "").lower() in ("1", "true", "yes"):
        logger.info("[Main] CHAT_DISABLE_RAG=1 — RAG 메모리 비활성화")
    elif not _is_rag_available():
        logger.warning("[Main] RAG 패키지 없음 — 기억 기능 없이 실행")
    else:
        # 지식 베이스 로드를 별도 스레드에서 실행한다.
        # 중요: chromadb/sentence_transformers의 import는 GIL을 오래 점유하므로
        # asyncio 이벤트 루프(run_in_executor 포함)가 아닌
        # 순수 threading.Thread에서만 수행해야 한다.
        import threading

        def _load_knowledge_thread():
            """별도 스레드에서 RAG 지식 베이스를 로드한다 (이벤트 루프 미사용)."""
            try:
                from app.memory import get_memory_engine
                engine = get_memory_engine()
                engine.load_knowledge_docs()
                logger.info("[Main] RAG 지식 베이스 로드 완료")
            except Exception as e:
                logger.warning(f"[Main] RAG 초기화 실패 (기억 기능 비활성화): {e}")

        t = threading.Thread(target=_load_knowledge_thread, daemon=True)
        t.start()
        logger.info("[Main] RAG 메모리 엔진 초기화 시작 (백그라운드 스레드)")

    yield  # 서버 실행 중


# ── FastAPI 앱 생성 ───────────────────────────────────────────────────────
app = FastAPI(
    title="ai_chat — 시온 채팅 엔진",
    description=(
        "Ollama 기반 시온(sion) 캐릭터 채팅 엔진. "
        "방송 채팅(broadcast) 모드를 지원합니다."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS 미들웨어 ─────────────────────────────────────────────────────────
# ai_vtuber_core, ai_broadcast, ai_live2d에서 호출 가능하도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """채팅 요청 모델."""
    message: str
    mode: Optional[str] = "broadcast"     # "broadcast"
    context: Optional[str] = None         # 방송 모드 채팅 히스토리 (선택)
    viewer_name: Optional[str] = None     # 방송 모드 시청자 닉네임 (선택)
    is_donation: Optional[bool] = False   # 후원/도네이션 여부 (API 호환용)


class ChatResponse(BaseModel):
    """채팅 응답 모델."""
    reply: str
    emotion: str
    action: Optional[dict] = None  # {"type": "play_music", "query": "..."}
    error: Optional[str] = None


# ── 엔드포인트 ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 간단한 안내 페이지."""
    return """
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_chat 서버 실행 중</h1>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/health" style="color:#89dceb">서버 상태 확인</a></li>
      <li><a href="/memory/stats" style="color:#89dceb">RAG 메모리 현황</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """서버 상태 확인 — ai_vtuber_core에서 헬스 체크용으로 호출."""
    return {
        "status": "ok",
        "module": "ai_chat",
        "version": "2.0.0",
        "supported_modes": ["broadcast"],
        "rag_enabled": _is_rag_available(),
        "llm": get_provider_config(),
    }


@app.get("/chat/persona")
async def get_persona():
    """사용 가능한 채팅 모드 및 설명 반환."""
    return {
        "modes": {
            "broadcast": "방송 채팅 반응 모드 — 1~2문장, 짧고 임팩트 있게",
        },
        "emotions": list(sorted([
            "happy", "sad", "surprised", "thinking", "excited",
            "calm", "worried", "angry", "love", "shy"
        ])),
        "llm": get_provider_config(),
    }


@app.get("/memory/stats")
async def memory_stats():
    """RAG 메모리 현황 — 대화 기억 수, 지식 청크 수 등 반환 (디버그용)."""
    try:
        from app.memory import get_memory_engine
        engine = get_memory_engine()
        return {"rag_enabled": True, **engine.get_stats()}
    except ImportError:
        return {"rag_enabled": False, "reason": "chromadb 또는 sentence-transformers 미설치"}
    except Exception as e:
        return {"rag_enabled": False, "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """시온(sion) 캐릭터로 채팅 응답을 생성한다.

    처리 흐름:
      1. RAG: 관련 기억 & 지식 베이스 검색 (비동기, 타임아웃 적용)
      2. Ollama 호출 (시온 캐릭터 프롬프트 + RAG 컨텍스트)
      3. [감정:태그] 파싱 → emotion 추출
      4. RAG: 대화 기억 저장 (fire-and-forget)
      5. 응답 텍스트 + 감정 반환
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있습니다.")

    mode = req.mode or "broadcast"
    if mode != "broadcast":
        raise HTTPException(
            status_code=400,
            detail="mode는 'broadcast'만 허용됩니다.",
        )

    result = await generate_reply(
        message=req.message,
        mode=mode,
        context=req.context,
        viewer_name=req.viewer_name,
        is_donation=req.is_donation or False,
    )

    return ChatResponse(**result)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
