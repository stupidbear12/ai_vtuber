# -*- coding: utf-8 -*-
"""
ai_browser_agent FastAPI 서버 (포트 8007)

엔드포인트:
  POST /browser/start                — 브라우저 세션 시작
  POST /browser/command              — 명령 실행 (OBSERVE→THINK→ACT 루프)
  POST /browser/stop                 — 브라우저 세션 종료
  GET  /browser/status               — 현재 상태
  GET  /browser/screenshot           — 현재 스크린샷 (PNG)
  POST /browser/album-review/start   — 앨범 리뷰 시작 (백그라운드)
  GET  /browser/album-review/status  — 앨범 리뷰 진행 상태
  POST /browser/album-review/cancel  — 앨범 리뷰 취소
  GET  /health                       — 헬스 체크
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[3]
    load_dotenv(_root / ".env")
except ImportError:
    pass

from .browser_controller import BrowserController
from .agent_loop import AgentLoop
from .album_review import AlbumReviewer

logger = logging.getLogger(__name__)

HOST = os.environ.get("AI_BROWSER_AGENT_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_BROWSER_AGENT_PORT", "8007"))

browser: Optional[BrowserController] = None
agent: Optional[AgentLoop] = None
reviewer: Optional[AlbumReviewer] = None


# ── Pydantic 스키마 ───────────────────────────────────────────────

class CommandRequest(BaseModel):
    command: str = Field(..., description="시청자 명령 (예: '네이버에서 오늘 날씨 검색해줘')")
    requester: str = Field(default="", description="요청자 닉네임")


class AlbumReviewRequest(BaseModel):
    artist: str = Field(..., description="아티스트명 (예: 'IU', '아이유')")
    album: str = Field(..., description="앨범명 (예: 'LILAC')")
    highlight_sec: int = Field(
        default=90,
        ge=30,
        le=300,
        description="트랙당 하이라이트 재생 시간(초), 기본 90초",
    )


# ── 라이프사이클 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser, agent, reviewer
    browser = BrowserController()
    agent = AgentLoop(browser)
    reviewer = AlbumReviewer(browser)
    logger.info("[BrowserAgent] 모듈 초기화 완료 (브라우저 미시작)")
    yield
    if browser and browser.is_running:
        await browser.stop()
    logger.info("[BrowserAgent] 모듈 종료")


app = FastAPI(
    title="AI Browser Agent",
    description="OBSERVE→THINK→ACT 브라우저 에이전트",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 엔드포인트 ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "module": "ai_browser_agent",
        "version": "1.0.0",
        "browser_running": browser.is_running if browser else False,
        "agent_status": agent.status if agent else "not_initialized",
    }


@app.post("/browser/start")
async def browser_start():
    if not browser:
        raise HTTPException(500, "모듈 초기화 실패")
    if browser.is_running:
        return {"success": True, "message": "브라우저가 이미 실행 중입니다."}
    try:
        await browser.start()
        return {"success": True, "message": "브라우저가 시작되었습니다."}
    except Exception as e:
        raise HTTPException(500, f"브라우저 시작 실패: {e}")


@app.post("/browser/stop")
async def browser_stop():
    if not browser or not browser.is_running:
        return {"success": True, "message": "브라우저가 이미 종료되어 있습니다."}
    if agent and agent.status == "running":
        agent.cancel()
    await browser.stop()
    return {"success": True, "message": "브라우저가 종료되었습니다."}


@app.post("/browser/command")
async def browser_command(req: CommandRequest):
    if not browser or not browser.is_running:
        raise HTTPException(400, "브라우저가 실행 중이 아닙니다. /browser/start를 먼저 호출하세요.")
    if not agent:
        raise HTTPException(500, "에이전트 초기화 실패")
    if agent.status == "running":
        raise HTTPException(409, "이미 명령 실행 중입니다.")
    result = await agent.execute(req.command, req.requester)
    return result


@app.get("/browser/status")
async def browser_status():
    return {
        "browser_running": browser.is_running if browser else False,
        "agent_status": agent.status if agent else "not_initialized",
        "current_command": agent.current_command if agent else "",
        "current_step": agent.current_step if agent else 0,
    }


@app.get("/browser/screenshot")
async def browser_screenshot():
    if not browser or not browser.is_running:
        raise HTTPException(400, "브라우저가 실행 중이 아닙니다.")
    try:
        png_bytes = await browser.screenshot_bytes()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(500, f"스크린샷 실패: {e}")


# ── 앨범 리뷰 ────────────────────────────────────────────────────

@app.post("/browser/album-review/start")
async def album_review_start(req: AlbumReviewRequest):
    """앨범 리뷰를 시작한다 (백그라운드 실행)."""
    if not browser or not browser.is_running:
        raise HTTPException(400, "브라우저가 실행 중이 아닙니다. /browser/start를 먼저 호출하세요.")
    if not reviewer:
        raise HTTPException(500, "리뷰어 초기화 실패")
    if reviewer.status == "reviewing":
        raise HTTPException(409, "이미 앨범 리뷰가 진행 중입니다.")

    # 백그라운드 태스크로 실행
    asyncio.create_task(
        reviewer.review(req.artist, req.album, req.highlight_sec)
    )
    return {
        "success": True,
        "message": f"{req.artist} - {req.album} 앨범 리뷰를 시작합니다.",
        "highlight_sec": req.highlight_sec,
    }


@app.get("/browser/album-review/status")
async def album_review_status():
    """앨범 리뷰 진행 상태를 조회한다."""
    if not reviewer:
        return {"status": "not_initialized"}
    return reviewer.progress


@app.post("/browser/album-review/cancel")
async def album_review_cancel():
    """진행 중인 앨범 리뷰를 취소한다."""
    if not reviewer:
        raise HTTPException(500, "리뷰어 초기화 실패")
    if reviewer.status != "reviewing":
        return {"success": True, "message": "진행 중인 리뷰가 없습니다."}
    reviewer.cancel()
    return {"success": True, "message": "리뷰 취소를 요청했습니다."}


# ── 실행 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)
