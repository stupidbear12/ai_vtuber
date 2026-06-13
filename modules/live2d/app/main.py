# -*- coding: utf-8 -*-
"""
app/main.py — ai_live2d 독립 서버 진입점

역할:
  - Live2D 웹 뷰어 제공 (http://localhost:8001/live2d/static/)
  - WebSocket 채널 유지 (ws://localhost:8001/live2d/ws)
  - 표정/모션/립싱크/감정/반응 REST API
  - 데스크톱 펫 채팅 엔드포인트 (/live2d/chat → ai_chat 위임)

실행 방법:
    cd ai_live2d
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

Electron 펫 실행 (별도 터미널):
    cd ai_live2d/electron
    npm install
    npm start
"""

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse

from app.router import live2d_router, mount_static


class NoCacheMiddleware:
    """정적 파일 캐싱 방지 — ASGI 미들웨어 (BaseHTTPMiddleware 대신 순수 ASGI)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"cache-control", b"no-cache, no-store, must-revalidate"))
                headers.append((b"pragma", b"no-cache"))
                headers.append((b"expires", b"0"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)

# ── FastAPI 앱 생성 ───────────────────────────────────────────────
app = FastAPI(
    title="ai_live2d — Live2D 아바타 서버",
    description=(
        "Live2D 아바타 독립 모듈. "
        "웹 뷰어(Haru 모델), WebSocket 실시간 제어, "
        "표정/모션/립싱크 REST API, 데스크톱 Electron 펫을 제공합니다."
    ),
    version="1.0.0",
)

# ── CORS 미들웨어 ─────────────────────────────────────────────────
# ai_vtuber_core 오케스트레이터 및 외부 서비스에서 호출 가능하도록 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 운영 환경에서는 특정 도메인으로 제한 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 및 정적 파일 등록 ──────────────────────────────────────
app.include_router(live2d_router)  # /live2d/* 엔드포인트
mount_static(app)                  # /live2d/static/* 정적 파일



# ── 기본 엔드포인트 ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 간단한 안내 페이지 반환."""
    return """
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_live2d 서버 실행 중</h1>
    <ul>
      <li><a href="/live2d/static/" style="color:#89dceb">Live2D 웹 뷰어</a></li>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/live2d/status" style="color:#89dceb">WebSocket 클라이언트 수</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """서버 상태 확인 — ai_vtuber_core에서 헬스 체크용으로 호출."""
    return {
        "status": "ok",
        "module": "ai_live2d",
        "version": "1.0.0",
        "endpoints": {
            "web_viewer": "/live2d/static/",
            "websocket": "ws://localhost:8001/live2d/ws",
            "api_docs": "/docs",
        },
    }


# ── 캐시 방지 미들웨어 (모든 라우트 정의 후 래핑) ──────────────────
_fastapi_app = app
app = NoCacheMiddleware(_fastapi_app)


if __name__ == "__main__":
    # 직접 실행 시: python -m app.main
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True)
