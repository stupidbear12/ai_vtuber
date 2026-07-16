# -*- coding: utf-8 -*-
"""
체스 모듈 FastAPI 서버 (포트 8008)

시온 vs 시청자 체스 대국 관리
- POST /chess/new          — 새 대국 시작
- POST /chess/sion-move     — 시온(Stockfish) 수
- POST /chess/vote          — 시청자 투표
- POST /chess/close-vote    — 투표 마감 & 수 적용
- GET  /chess/state         — 현재 게임 상태
- GET  /chess/votes         — 투표 현황
- POST /chess/resign        — 기권
- GET  /chess/board         — 체스판 UI (HTML)
- WS   /chess/ws            — 실시간 상태 WebSocket
- GET  /health              — 헬스체크
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

_root = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env", override=True)
except ImportError:
    pass

from .chess_engine import ChessEngine, GamePhase

logger = logging.getLogger(__name__)

HOST = os.environ.get("AI_CHESS_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_CHESS_PORT", "8008"))

engine = ChessEngine()
ws_clients: list[WebSocket] = []


# ── Pydantic 스키마 ───────────────────────────────────────────────

class NewGameRequest(BaseModel):
    sion_color: str = Field(default="white", description="시온 색상 (white/black)")
    skill_level: int = Field(default=5, ge=0, le=20, description="Stockfish 난이도 0~20")
    vote_duration: int = Field(default=30, ge=10, le=120, description="투표 시간(초)")


class VoteRequest(BaseModel):
    user_id: str = Field(..., description="시청자 ID")
    move: str = Field(..., description="수 (SAN 또는 UCI, 예: 'e4', 'e2e4', 'Nf3')")


class ResignRequest(BaseModel):
    side: str = Field(default="sion", description="기권 측 (sion/viewer)")


# ── 라이프사이클 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start_engine()
    logger.info("[Chess] 모듈 초기화 완료")
    yield
    await engine.stop_engine()
    logger.info("[Chess] 모듈 종료")


app = FastAPI(
    title="AI Chess Module",
    description="시온 vs 시청자 체스 대국",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 정적 파일 서빙 (체스판 UI)
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/chess/static", StaticFiles(directory=str(_static_dir)), name="chess_static")


# ── WebSocket 브로드캐스트 ────────────────────────────────────────

async def _broadcast(data: dict):
    """모든 WebSocket 클라이언트에 상태를 전송한다."""
    msg = json.dumps(data, ensure_ascii=False)
    disconnected = []
    for ws in ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_clients.remove(ws)


# ── 엔드포인트 ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "module": "chess",
        "version": "1.0.0",
        "game_phase": engine.game.phase.value,
    }


@app.post("/chess/new")
async def new_game(req: NewGameRequest):
    """새 대국을 시작한다."""
    try:
        state = engine.new_game(
            sion_color=req.sion_color,
            skill_level=req.skill_level,
            vote_duration=req.vote_duration,
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    await _broadcast({"event": "new_game", "state": state})
    return {"success": True, "state": state}


@app.post("/chess/sion-move")
async def sion_move():
    """시온(Stockfish)이 수를 둔다."""
    try:
        result = await engine.sion_move()
    except ValueError as e:
        raise HTTPException(400, str(e))

    await _broadcast({
        "event": "sion_move",
        "move_uci": result["move_uci"],
        "move_san": result["move_san"],
        "state": result["state"],
    })
    return result


@app.post("/chess/vote")
async def vote(req: VoteRequest):
    """시청자가 수를 투표한다."""
    result = engine.submit_vote(req.user_id, req.move)
    if not result["success"]:
        raise HTTPException(400, result["reason"])

    await _broadcast({
        "event": "vote",
        "user_id": req.user_id,
        "move_san": result["move_san"],
        "total_votes": result["total_votes"],
    })
    return result


@app.post("/chess/close-vote")
async def close_vote():
    """투표를 마감하고 최다 득표 수를 적용한다."""
    try:
        result = engine.close_vote()
    except ValueError as e:
        raise HTTPException(400, str(e))

    await _broadcast({
        "event": "viewer_move",
        "move_uci": result["move_uci"],
        "move_san": result["move_san"],
        "state": result["state"],
    })
    return result


@app.get("/chess/state")
async def get_state():
    """현재 게임 상태를 반환한다."""
    return engine._state_dict()


@app.get("/chess/votes")
async def get_votes():
    """현재 투표 현황을 반환한다."""
    return engine.get_vote_tally()


@app.get("/chess/legal-moves")
async def get_legal_moves():
    """현재 합법 수 목록을 반환한다."""
    return {"moves": engine.get_legal_moves()}


@app.post("/chess/resign")
async def resign(req: ResignRequest):
    """기권한다."""
    state = engine.resign(req.side)
    await _broadcast({"event": "resign", "side": req.side, "state": state})
    return {"success": True, "state": state}


# ── LLM 해설 + TTS + Live2D 연동 ─────────────────────────────────

CHAT_URL = os.environ.get("AI_CHAT_URL", "http://localhost:8002")
CORE_URL = os.environ.get("AI_CORE_URL", "http://localhost:8000")


async def _generate_commentary(move_san: str, side: str, state: dict):
    """시온 LLM이 체스 해설을 생성하고 TTS로 말한다."""
    import aiohttp

    # 상황 설명 생성
    is_check = state.get("is_check", False)
    is_game_over = state.get("is_game_over", False)
    result = state.get("result")
    move_count = state.get("move_count", 0)

    if is_game_over:
        if result == "sion_win":
            prompt = "체스에서 시청자를 이겼다! 기뻐하면서 한마디 해줘. 짧게 1~2문장."
        elif result == "viewer_win":
            prompt = "체스에서 시청자에게 졌다. 아쉬워하면서 한마디 해줘. 짧게 1~2문장."
        else:
            prompt = "체스가 무승부로 끝났다. 한마디 해줘. 짧게 1~2문장."
    elif side == "sion":
        if is_check:
            prompt = f"내가 {move_san}으로 체크를 걸었다! 자신감 있게 한마디. 짧게 1문장."
        else:
            prompt = f"내가 체스에서 {move_san}을 뒀다. 왜 이 수를 뒀는지 짧게 한마디. 1문장."
    else:
        if is_check:
            prompt = f"시청자가 {move_san}으로 체크를 걸었다! 놀라면서 한마디. 짧게 1문장."
        else:
            prompt = f"시청자가 체스에서 {move_san}을 뒀다. 이 수에 대해 리액션. 짧게 1문장."

    # 감정 결정
    if is_game_over and result == "sion_win":
        emotion = "happy"
    elif is_game_over and result == "viewer_win":
        emotion = "sad"
    elif is_check and side == "sion":
        emotion = "excited"
    elif is_check and side != "sion":
        emotion = "worried"
    else:
        emotion = "calm"

    try:
        async with aiohttp.ClientSession() as session:
            # core orchestrator의 speak 엔드포인트 사용
            async with session.post(
                f"{CORE_URL}/speak",
                json={"text": prompt, "emotion": emotion},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    logger.info("[Chess Commentary] 해설 전달 완료: %s", prompt[:50])
                else:
                    logger.warning("[Chess Commentary] 실패 HTTP %s", resp.status)
    except Exception as e:
        logger.warning("[Chess Commentary] 연결 실패: %s", e)


@app.get("/chess/board", response_class=HTMLResponse)
async def board_page():
    """체스판 UI HTML을 반환한다."""
    html_path = _static_dir / "board.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>board.html not found</h1>", status_code=404)


@app.websocket("/chess/ws")
async def websocket_endpoint(ws: WebSocket):
    """실시간 게임 상태 WebSocket."""
    await ws.accept()
    ws_clients.append(ws)
    logger.info("[Chess WS] 클라이언트 연결 (총 %d명)", len(ws_clients))

    # 현재 상태 전송
    try:
        await ws.send_text(json.dumps({
            "event": "connected",
            "state": engine._state_dict(),
        }, ensure_ascii=False))
    except Exception:
        pass

    try:
        while True:
            # 클라이언트로부터의 메시지 (핑 등)
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in ws_clients:
            ws_clients.remove(ws)
        logger.info("[Chess WS] 클라이언트 해제 (총 %d명)", len(ws_clients))


# ── 자동 투표 마감 태스크 ─────────────────────────────────────────

async def _auto_vote_closer():
    """투표 시간이 지나면 자동으로 마감한다."""
    while True:
        await asyncio.sleep(1)
        if engine.game.phase == GamePhase.VOTE_OPEN:
            if time.time() >= engine.game.vote_deadline:
                logger.info("[Chess] 투표 자동 마감")
                try:
                    result = engine.close_vote()
                    await _broadcast({
                        "event": "viewer_move",
                        "move_uci": result["move_uci"],
                        "move_san": result["move_san"],
                        "state": result["state"],
                        "auto_closed": True,
                    })
                    # 시청자 수에 대한 해설
                    asyncio.create_task(
                        _generate_commentary(result["move_san"], "viewer", result["state"])
                    )
                    # 게임 끝이 아니면 시온 자동 수
                    if engine.game.phase == GamePhase.SION_TURN:
                        await asyncio.sleep(4)  # 해설 후 대기
                        sion_result = await engine.sion_move()
                        await _broadcast({
                            "event": "sion_move",
                            "move_uci": sion_result["move_uci"],
                            "move_san": sion_result["move_san"],
                            "state": sion_result["state"],
                        })
                        # 시온 수에 대한 해설
                        asyncio.create_task(
                            _generate_commentary(sion_result["move_san"], "sion", sion_result["state"])
                        )
                except Exception as e:
                    logger.error("[Chess] 자동 투표 처리 실패: %s", e)


@app.on_event("startup")
async def _start_auto_closer():
    asyncio.create_task(_auto_vote_closer())


# ── 실행 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)
