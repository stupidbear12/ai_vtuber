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
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.chat_collector import BroadcastChatManager
from app.chzzk_auth import ChzzkTokenManager
from app.chzzk_api import ChzzkClient

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# ── 전역 싱글턴 ──────────────────────────────────────────────────
_manager: Optional[BroadcastChatManager] = None
_token_manager: Optional[ChzzkTokenManager] = None
_chzzk_client: Optional[ChzzkClient] = None

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


# ── 앱 시작 시 초기화 ───────────────────────────────────────────

@app.on_event("startup")
async def _startup():
    global _token_manager, _chzzk_client
    _token_manager = ChzzkTokenManager()
    _chzzk_client = ChzzkClient(_token_manager)
    logging.getLogger(__name__).info(
        f"[startup] Chzzk 공식 API 클라이언트 초기화 완료 "
        f"(token={'있음' if _token_manager.has_token() else '없음'})"
    )


# ── 요청 모델 ────────────────────────────────────────────────────

class BroadcastStartRequest(BaseModel):
    """방송 채팅 수집 시작 요청 모델."""
    platform: str     # "youtube" 또는 "chzzk"
    channel_id: str   # 유튜브: 영상 ID (예: dQw4w9WgXcQ), 치지직: 채널 해시 ID


class LiveSettingRequest(BaseModel):
    """방송 설정 변경 요청 모델."""
    title: Optional[str] = None
    category_type: Optional[str] = None
    category_id: Optional[str] = None
    tags: Optional[List[str]] = None


class ChzzkChatSendRequest(BaseModel):
    """치지직 채팅 전송 요청 모델."""
    message: str


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

    # 새 매니저 생성 (환경변수에서 서버 URL 자동 로드, 공식 Chzzk 클라이언트 전달)
    _manager = BroadcastChatManager(chzzk_client=_chzzk_client)

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


# ═══════════════════════════════════════════════════════════════
# 치지직 공식 API 엔드포인트
# ═══════════════════════════════════════════════════════════════

# ── OAuth 인증 ──────────────────────────────────────────────────

@app.get("/chzzk/auth/url")
async def chzzk_auth_url():
    """OAuth 인증 URL 반환.

    사용자가 이 URL을 브라우저에서 열어 치지직 계정으로 로그인하면
    /chzzk/auth/callback으로 리다이렉트된다.

    CHZZK_CLIENT_ID 환경변수가 필요하다.
    """
    try:
        url, state = _token_manager.get_auth_url()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"auth_url": url, "state": state}


@app.get("/chzzk/auth/callback")
async def chzzk_auth_callback(
    code: str = Query(..., description="치지직에서 받은 인증 코드"),
    state: str = Query("", description="CSRF 방지용 state 값"),
):
    """OAuth 콜백 — 인증 코드를 Access Token으로 교환.

    치지직 개발자 콘솔의 리다이렉트 URI를
    http://localhost:8003/chzzk/auth/callback 으로 설정해야 한다.
    """
    try:
        content = await _token_manager.exchange_code(code, state)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "success": True,
        "message": "치지직 OAuth 인증이 완료되었습니다.",
        "expires_in": content.get("expiresIn"),
    }


@app.get("/chzzk/auth/status")
async def chzzk_auth_status():
    """현재 Chzzk Access Token 상태 확인."""
    return _token_manager.get_status()


@app.post("/chzzk/auth/revoke")
async def chzzk_auth_revoke():
    """Chzzk 토큰 취소 (로그아웃). 로컬 토큰 파일도 삭제된다."""
    ok = await _token_manager.revoke()
    return {"success": ok, "message": "토큰이 취소되었습니다."}


# ── Live API ────────────────────────────────────────────────────

@app.get("/chzzk/stream-key")
async def chzzk_stream_key():
    """방송 스트림키 조회 (Access Token 필요)."""
    try:
        key = await _chzzk_client.get_stream_key()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"stream_key": key}


@app.get("/chzzk/live-setting")
async def chzzk_get_live_setting():
    """방송 설정 조회 (제목, 카테고리, 태그 등)."""
    try:
        setting = await _chzzk_client.get_live_setting()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return setting


@app.patch("/chzzk/live-setting")
async def chzzk_update_live_setting(req: LiveSettingRequest):
    """방송 설정 변경 (제공한 항목만 변경됨).

    요청 예시:
        {"title": "시온의 AI DJ 방송 🎵", "tags": ["AI", "VTuber"]}
    """
    if not any([req.title, req.category_type, req.category_id, req.tags]):
        raise HTTPException(status_code=400, detail="변경할 항목이 없습니다.")
    try:
        result = await _chzzk_client.update_live_setting(
            title=req.title,
            category_type=req.category_type,
            category_id=req.category_id,
            tags=req.tags,
        )
    except (RuntimeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "setting": result}


# ── Chat API ────────────────────────────────────────────────────

@app.post("/chzzk/chat/send")
async def chzzk_send_chat(req: ChzzkChatSendRequest):
    """치지직 채팅창에 메시지 전송 (Access Token 필요).

    최대 100자. 초과 시 자동으로 잘린다.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="메시지가 비어있습니다.")
    try:
        result = await _chzzk_client.send_chat(req.message)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"success": True, "result": result}


@app.get("/chzzk/chat/settings")
async def chzzk_get_chat_settings():
    """채팅 설정 조회 (느린 모드, 팔로워 전용 등)."""
    try:
        settings = await _chzzk_client.get_chat_settings()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return settings


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8003, reload=True)
