# -*- coding: utf-8 -*-
"""
ai_browser_agent FastAPI 서버 (포트 8007)

엔드포인트:
  POST /browser/start                — 브라우저 세션 시작
  POST /browser/stop                 — 브라우저 세션 종료
  POST /browser/navigate             — URL 이동
  POST /browser/click                — 요소 클릭
  POST /browser/show-page            — 임의 URL을 OBS browser_source에 표시
  POST /browser/play-video           — YouTube 영상 재생 (show-page 래퍼)
  POST /browser/stop-video           — YouTube 영상 정지
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
from pydantic import BaseModel, Field

_root = Path(__file__).resolve().parents[3]
try:
    from dotenv import load_dotenv
    load_dotenv(_root / ".env", override=True)
except ImportError:
    pass

from .browser_controller import BrowserController
from .album_review import AlbumReviewer

logger = logging.getLogger(__name__)

HOST = os.environ.get("AI_BROWSER_AGENT_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_BROWSER_AGENT_PORT", "8007"))

browser: Optional[BrowserController] = None
reviewer: Optional[AlbumReviewer] = None


# ── Pydantic 스키마 ───────────────────────────────────────────────

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
    global browser, reviewer
    browser = BrowserController()
    reviewer = AlbumReviewer(browser)
    logger.info("[BrowserAgent] 모듈 초기화 완료 (브라우저 미시작)")
    yield
    if browser and browser.is_running:
        await browser.stop()
    logger.info("[BrowserAgent] 모듈 종료")


app = FastAPI(
    title="AI Browser Agent",
    description="앨범 리뷰 브라우저 에이전트",
    version="2.0.0",
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
        "version": "2.0.0",
        "browser_running": browser.is_running if browser else False,
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


class NavigateRequest(BaseModel):
    url: str = Field(..., description="이동할 URL")


class ClickRequest(BaseModel):
    target: str = Field(..., description="클릭 대상 (CSS selector 또는 텍스트)")


class PlayVideoRequest(BaseModel):
    video_id: str = Field(..., description="YouTube 영상 ID (예: 'dQw4w9WgXcQ')")
    scene: str = Field(default="web_browser", description="OBS 씬 이름")
    source: str = Field(default="album_yt_browser", description="OBS 소스 이름")
    switch_scene: bool = Field(default=True, description="해당 씬으로 자동 전환 여부")


class StopVideoRequest(BaseModel):
    scene: str = Field(default="web_browser", description="OBS 씬 이름")
    source: str = Field(default="album_yt_browser", description="OBS 소스 이름")
    return_scene: str = Field(default="Radio Mode", description="전환할 씬 이름")


class ShowPageRequest(BaseModel):
    url: str = Field(..., description="표시할 URL (예: 'https://github.com/stupidbear12')")
    scene: str = Field(default="web_browser", description="OBS 씬 이름")
    source: str = Field(default="album_yt_browser", description="OBS browser_source 이름")
    switch_scene: bool = Field(default=True, description="해당 씬으로 자동 전환 여부")
    width: int = Field(default=1920, description="브라우저 소스 너비")
    height: int = Field(default=1080, description="브라우저 소스 높이")


@app.post("/browser/navigate")
async def browser_navigate(req: NavigateRequest):
    """브라우저를 지정한 URL로 이동시킨다."""
    if not browser or not browser.is_running:
        raise HTTPException(400, "브라우저가 실행 중이 아닙니다.")
    try:
        await browser.navigate(req.url)
        return {"success": True, "url": req.url}
    except Exception as e:
        raise HTTPException(500, f"이동 실패: {e}")


@app.post("/browser/click")
async def browser_click(req: ClickRequest):
    """브라우저에서 요소를 클릭한다."""
    if not browser or not browser.is_running:
        raise HTTPException(400, "브라우저가 실행 중이 아닙니다.")
    try:
        await browser.click(req.target)
        return {"success": True, "target": req.target}
    except Exception as e:
        raise HTTPException(500, f"클릭 실패: {e}")


# ── YouTube 영상 재생 (OBS browser_source) ───────────────────────

def _load_env_value(key: str) -> str:
    """환경 변수를 읽되, 없으면 .env 파일에서 직접 파싱한다."""
    val = os.environ.get(key, "")
    if val:
        return val
    env_path = _root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}="):
                return stripped.split("=", 1)[1].strip().strip("'\"")
    return ""


def _get_obs_client():
    """OBS WebSocket 클라이언트를 생성한다."""
    try:
        import obsws_python as obsws
    except ImportError:
        raise HTTPException(500, "obsws-python 패키지가 설치되어 있지 않습니다.")
    host = _load_env_value("OBS_WS_HOST") or "localhost"
    port = int(_load_env_value("OBS_WS_PORT") or "4455")
    password = _load_env_value("OBS_WS_PASSWORD")
    logger.info("[OBS] host=%s port=%d password=%s", host, port, "***" if password else "EMPTY")
    return obsws.ReqClient(host=host, port=port, password=password)


_YT_PLAYER_HTML = str(
    Path(__file__).resolve().parents[3] / "obs" / "yt_player.html"
)


@app.post("/browser/show-page")
async def show_page(req: ShowPageRequest):
    """임의 URL을 OBS browser_source에 표시한다."""
    try:
        cl = _get_obs_client()
    except Exception as e:
        raise HTTPException(500, f"OBS 연결 실패: {e}")

    try:
        cl.set_input_settings(req.source, {
            "is_local_file": False,
            "url": req.url,
            "width": req.width,
            "height": req.height,
            "reroute_audio": False,
            "css": "",
        }, overlay=True)
    except Exception as e:
        raise HTTPException(500, f"OBS 소스 업데이트 실패: {e}")

    if req.switch_scene:
        try:
            cl.set_current_program_scene(req.scene)
        except Exception:
            pass

    logger.info("[ShowPage] url=%s scene=%s", req.url, req.scene)
    return {"success": True, "url": req.url, "scene": req.scene}


@app.post("/browser/play-video")
async def play_video(req: PlayVideoRequest):
    """YouTube 영상을 OBS browser_source로 재생한다 (show-page 래퍼)."""
    yt_url = f"https://www.youtube.com/watch?v={req.video_id}"
    result = await show_page(ShowPageRequest(
        url=yt_url,
        scene=req.scene,
        source=req.source,
        switch_scene=req.switch_scene,
    ))
    result["video_id"] = req.video_id
    logger.info("[PlayVideo] video_id=%s scene=%s", req.video_id, req.scene)
    return result


@app.post("/browser/stop-video")
async def stop_video(req: StopVideoRequest):
    """OBS browser_source 영상을 정지하고 씬을 전환한다."""
    try:
        cl = _get_obs_client()
    except Exception as e:
        raise HTTPException(500, f"OBS 연결 실패: {e}")

    try:
        cl.set_input_settings(req.source, {
            "is_local_file": False,
            "url": "about:blank",
        }, overlay=True)
    except Exception as e:
        raise HTTPException(500, f"OBS 소스 정지 실패: {e}")

    try:
        cl.set_current_program_scene(req.return_scene)
    except Exception:
        pass

    logger.info("[StopVideo] stopped, scene → %s", req.return_scene)
    return {"success": True, "scene": req.return_scene}


@app.post("/browser/stop")
async def browser_stop():
    if not browser or not browser.is_running:
        return {"success": True, "message": "브라우저가 이미 종료되어 있습니다."}
    await browser.stop()
    return {"success": True, "message": "브라우저가 종료되었습니다."}


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
