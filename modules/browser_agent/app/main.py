# -*- coding: utf-8 -*-
"""
ai_browser_agent FastAPI 서버 (포트 8007)

엔드포인트:
  POST /browser/start       — 브라우저 세션 시작
  POST /browser/command      — 명령 실행 (OBSERVE→THINK→ACT 루프)
  POST /browser/stop         — 브라우저 세션 종료
  GET  /browser/status       — 현재 상태
  GET  /browser/screenshot   — 현재 스크린샷 (PNG)
  GET  /health               — 헬스 체크
"""

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

logger = logging.getLogger(__name__)

HOST = os.environ.get("AI_BROWSER_AGENT_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_BROWSER_AGENT_PORT", "8007"))

browser: Optional[BrowserController] = None
agent: Optional[AgentLoop] = None


# ── Pydantic 스키마 ───────────────────────────────────────────────

class CommandRequest(BaseModel):
    command: str = Field(..., description="시청자 명령 (예: '네이버에서 오늘 날씨 검색해줘')")
    requester: str = Field(default="", description="요청자 닉네임")


# ── 라이프사이클 ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser, agent
    browser = BrowserController()
    agent = AgentLoop(browser)
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


# ── 실행 ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)
