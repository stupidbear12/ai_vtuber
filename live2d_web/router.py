# -*- coding: utf-8 -*-
"""
live2d_web/router.py - Live2D 웹 뷰어용 FastAPI 라우터

api/main.py에서 include_router로 통합:
    from live2d_web.router import live2d_router, mount_static
    app.include_router(live2d_router)
    mount_static(app)

엔드포인트:
    GET  /live2d/          - 뷰어 HTML 반환
    WS   /live2d/ws        - 브라우저 ↔ 서버 실시간 채널
    POST /live2d/params    - 파라미터 직접 주입
    POST /live2d/emotion   - 감정 변경
    POST /live2d/mouth     - 입 열림 값 설정
    POST /live2d/reaction  - 반응 애니메이션 트리거
    POST /live2d/idle/start
    POST /live2d/idle/stop
    GET  /live2d/status    - 연결된 클라이언트 수
"""

import json
from pathlib import Path
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

live2d_router = APIRouter(prefix="/live2d", tags=["live2d"])

_STATIC_DIR = Path(__file__).parent


# ── WebSocket 연결 관리자 ────────────────────────────────────────

class _WSManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)

    async def broadcast(self, msg: dict):
        if not self._clients:
            return
        payload = json.dumps(msg, ensure_ascii=False)
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    @property
    def count(self) -> int:
        return len(self._clients)


ws_manager = _WSManager()


# ── 요청 모델 ────────────────────────────────────────────────────

class ParamRequest(BaseModel):
    params: dict

class EmotionRequest(BaseModel):
    emotion: str       # calm | happy | surprised | thinking

class MouthRequest(BaseModel):
    value: float       # 0.0 ~ 1.0

class ReactionRequest(BaseModel):
    name: str          # nod | shake | surprised | superchat


# ── 라우트 ───────────────────────────────────────────────────────

@live2d_router.get("/")
async def viewer():
    # /live2d/static/ 로 리다이렉트 → 상대 경로(js/, models/, cubism/) 정상 해석
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/live2d/static/")


@live2d_router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            # 브라우저 → 서버 메시지 (현재는 로깅만)
            print(f"[Live2D WS] {raw[:120]}")
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@live2d_router.get("/status")
async def status():
    return {"clients": ws_manager.count}


@live2d_router.post("/params")
async def set_params(req: ParamRequest):
    await ws_manager.broadcast({"cmd": "set_params", "params": req.params})
    return {"ok": True, "clients": ws_manager.count}


@live2d_router.post("/emotion")
async def set_emotion(req: EmotionRequest):
    await ws_manager.broadcast({"cmd": "set_emotion", "emotion": req.emotion})
    return {"ok": True}


@live2d_router.post("/mouth")
async def set_mouth(req: MouthRequest):
    await ws_manager.broadcast({"cmd": "set_mouth", "value": max(0.0, min(1.0, req.value))})
    return {"ok": True}


@live2d_router.post("/mouth/clear")
async def clear_mouth():
    await ws_manager.broadcast({"cmd": "clear_mouth"})
    return {"ok": True}


@live2d_router.post("/reaction")
async def trigger_reaction(req: ReactionRequest):
    await ws_manager.broadcast({"cmd": "reaction", "name": req.name})
    return {"ok": True}


@live2d_router.post("/idle/start")
async def idle_start():
    await ws_manager.broadcast({"cmd": "idle_start"})
    return {"ok": True}


@live2d_router.post("/idle/stop")
async def idle_stop():
    await ws_manager.broadcast({"cmd": "idle_stop"})
    return {"ok": True}


# ── 정적 파일 마운트 헬퍼 (main.py에서 호출) ─────────────────────

def mount_static(app):
    """app.mount()로 live2d_web/ 정적 파일 서빙 등록.

    html=True: GET /live2d/static/ → index.html 자동 반환
    """
    app.mount(
        "/live2d/static",
        StaticFiles(directory=str(_STATIC_DIR), html=True),
        name="live2d_static",
    )
