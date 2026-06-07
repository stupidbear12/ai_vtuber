# -*- coding: utf-8 -*-
"""
app/main.py — ai_broadcast 독립 서버 진입점

역할:
  - 치지직/유튜브 방송 채팅 수집
  - 선별된 채팅을 ai_chat 모듈로 전달해 시온 응답 생성
  - ai_live2d 모듈로 표정 변경 명령 전송

포트: 8003

실행 방법:
    cd ai_broadcast
    pip install -r requirements.txt
    cp .env.example .env
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8003

의존 서비스:
    ai_chat  (http://localhost:8002) — 시온 채팅 응답 생성
    ai_live2d (http://localhost:8001) — Live2D 표정 변경
"""

import logging
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.chat_collector import BroadcastChatManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 싱글턴 채팅 매니저 — 서버 전역에서 하나만 실행
_manager: Optional[BroadcastChatManager] = None

# ── FastAPI 앱 생성 ───────────────────────────────────────────────
app = FastAPI(
    title="ai_broadcast — 방송 채팅 수집 서버",
    description=(
        "치지직/유튜브 라이브 채팅을 수집하고 시온(sion) 캐릭터로 반응합니다. "
        "ai_chat(채팅 엔진)과 ai_live2d(아바타)와 연동합니다."
    ),
    version="1.0.0",
)

# ── CORS 미들웨어 ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 모델 ────────────────────────────────────────────────────

class BroadcastStartRequest(BaseModel):
    """방송 채팅 수집 시작 요청 모델."""
    platform: str     # "youtube" 또는 "chzzk"
    channel_id: str   # 유튜브: 영상 ID (예: dQw4w9WgXcQ), 치지직: 채널 해시 ID


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 간단한 안내 페이지."""
    return """
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_broadcast 서버 실행 중</h1>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/broadcast/status" style="color:#89dceb">채팅 수집 상태</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """서버 상태 확인 — ai_vtuber_core에서 헬스 체크용으로 호출."""
    return {
        "status": "ok",
        "module": "ai_broadcast",
        "version": "1.0.0",
        "supported_platforms": ["youtube", "chzzk"],
        "collecting": _manager.is_running if _manager else False,
    }


@app.post("/broadcast/start")
async def broadcast_start(req: BroadcastStartRequest):
    """방송 채팅 수집을 시작한다.

    치지직 또는 유튜브 채팅을 수집하고,
    선별된 채팅에 대해 ai_chat을 통해 시온 응답을 생성한다.

    Args:
        req.platform: "youtube" 또는 "chzzk"
        req.channel_id: 유튜브 영상 ID 또는 치지직 채널 해시 ID

    Returns:
        success, platform, channel_id, started_at
    """
    global _manager

    # 이미 실행 중인지 확인
    if _manager and _manager.is_running:
        raise HTTPException(
            status_code=400,
            detail="채팅 수집이 이미 실행 중입니다. 먼저 /broadcast/stop을 호출하세요."
        )

    if not req.channel_id.strip():
        raise HTTPException(status_code=400, detail="channel_id가 비어있습니다.")

    # 새 매니저 생성 (환경변수에서 서버 URL 자동 로드)
    _manager = BroadcastChatManager()

    try:
        await _manager.start(req.platform, req.channel_id)
    except (ValueError, RuntimeError) as e:
        _manager = None
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _manager = None
        raise HTTPException(status_code=500, detail=f"채팅 수집 시작 실패: {e}")

    return {
        "success": True,
        "message": "방송 채팅 수집이 시작되었습니다.",
        "platform": req.platform,
        "channel_id": req.channel_id,
        "started_at": datetime.now().isoformat(),
    }


@app.post("/broadcast/stop")
async def broadcast_stop():
    """방송 채팅 수집을 중지한다.

    Returns:
        success, stats (수집 중 받은 채팅/응답/건너뜀/오류 횟수)
    """
    global _manager

    if _manager is None or not _manager.is_running:
        return {"success": True, "message": "채팅 수집이 이미 정지 상태입니다."}

    try:
        final_stats = await _manager.stop()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"채팅 수집 중지 실패: {e}")

    _manager = None
    return {
        "success": True,
        "message": "방송 채팅 수집이 중지되었습니다.",
        "stats": final_stats,
    }


@app.get("/broadcast/status")
async def broadcast_status():
    """현재 방송 채팅 수집 상태를 반환한다.

    Returns:
        running, platform, channel_id, stats, buffer_size, queue_size
    """
    if _manager is None:
        return JSONResponse(content={
            "running": False,
            "platform": None,
            "channel_id": None,
        })
    return JSONResponse(content=_manager.get_status())


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)
