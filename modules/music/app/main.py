# -*- coding: utf-8 -*-
"""
ai_music FastAPI 서버 (포트 8005)

엔드포인트:
  [YouTube Music — 방송 BGM / 신청곡]
  GET  /ymusic/search      — 곡 검색
  POST /ymusic/play        — 검색어/video_id 재생
  POST /ymusic/queue       — 큐 추가
  POST /ymusic/skip|pause|resume|stop
  GET  /ymusic/now-playing

  [공통]
  WS   /music/stream      — 실시간 오디오 PCM 스트림 (OBS 연동)
  GET  /music/status       — 믹서 상태
  GET  /health             — 헬스 체크
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    _root = Path(__file__).resolve().parents[3]
    load_dotenv(_root / ".env")
except ImportError:
    pass

from .audio_mixer import AudioMixer
from .youtube_music import YouTubeMusicPlayer

logger = logging.getLogger(__name__)

HOST = os.environ.get("AI_MUSIC_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_MUSIC_PORT", "8005"))
ENABLE_YTMUSIC = os.environ.get("YTMUSIC_ENABLED", "1") == "1"

mixer: Optional[AudioMixer] = None
ymusic: Optional[YouTubeMusicPlayer] = None


# ── Pydantic 스키마 ───────────────────────────────────────────────

class YTMusicPlayRequest(BaseModel):
    query: Optional[str] = Field(None, description="곡 제목·아티스트 검색어")
    video_id: Optional[str] = Field(None, description="YouTube video ID (11자)")
    requester: Optional[str] = Field(None, description="요청자 닉네임")


class YTMusicQueueRequest(BaseModel):
    video_id: str = Field(..., description="YouTube video ID")
    title: Optional[str] = None
    artist: Optional[str] = None
    requester: Optional[str] = None


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mixer, ymusic

    logger.info("ai_music 모듈 초기화 중...")

    mixer = AudioMixer()
    await mixer.initialize()

    if ENABLE_YTMUSIC:
        ymusic = YouTubeMusicPlayer(mixer)
        await ymusic.start()
        logger.info("YouTube Music 재생 활성화")
    else:
        ymusic = None

    logger.info("ai_music 모듈 준비 완료 (port %s, ytmusic=%s)", PORT, ENABLE_YTMUSIC)
    yield

    logger.info("ai_music 모듈 종료 중...")
    if ymusic:
        await ymusic.stop()
    await mixer.shutdown()


# ── FastAPI 앱 ────────────────────────────────────────────────────

app = FastAPI(title="ai_music", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "module": "ai_music",
        "ytmusic_enabled": ENABLE_YTMUSIC,
        "now_playing": ymusic.now_playing_info() if ymusic else None,
        "queue_size": len(ymusic.queue_list()) if ymusic else 0,
    }


@app.get("/music/status")
async def music_status():
    """믹서 상세 상태."""
    return {
        "ytmusic_enabled": ENABLE_YTMUSIC,
        "playback": mixer.get_playback_state() if mixer else None,
        "now_playing": ymusic.now_playing_info() if ymusic else None,
        "ymusic_queue": ymusic.queue_list() if ymusic else [],
        "queue_size": len(ymusic.queue_list()) if ymusic else 0,
    }


@app.post("/music/volume")
async def set_volume(volume: float):
    """볼륨 설정 (0.0~2.0). 1.0 초과 시 증폭."""
    if mixer is None:
        raise HTTPException(status_code=503, detail="Mixer not initialized")
    mixer.set_volume(volume)
    return {"success": True, "volume": volume}


# ── YouTube Music ─────────────────────────────────────────────────

def _require_ymusic() -> YouTubeMusicPlayer:
    if not ymusic:
        raise HTTPException(
            status_code=503,
            detail="YouTube Music가 비활성화되어 있습니다 (YTMUSIC_ENABLED=0).",
        )
    return ymusic


@app.get("/ymusic/search")
async def ymusic_search(q: str, limit: int = 10):
    """YouTube Music / YouTube 검색."""
    player = _require_ymusic()
    if not q.strip():
        raise HTTPException(status_code=400, detail="검색어(q)가 필요합니다.")
    results = await player.search(q.strip(), limit=min(limit, 25))
    return {"query": q.strip(), "results": results, "total": len(results)}


@app.post("/ymusic/play")
async def ymusic_play(req: YTMusicPlayRequest):
    """검색어 또는 video_id로 즉시 재생 (재생 중이면 큐에 추가)."""
    player = _require_ymusic()
    if req.video_id:
        track = await player.play_video(
            video_id=req.video_id.strip(),
            title=req.query or "",
            requester=req.requester,
        )
    elif req.query:
        track = await player.play_query(req.query.strip(), requester=req.requester)
    else:
        raise HTTPException(status_code=400, detail="query 또는 video_id 중 하나가 필요합니다.")
    return {"success": True, "track": track.to_dict(), "now_playing": player.now_playing_info()}


@app.post("/ymusic/queue")
async def ymusic_queue(req: YTMusicQueueRequest):
    """video_id를 재생 큐에 추가."""
    player = _require_ymusic()
    track = await player.enqueue(
        video_id=req.video_id.strip(),
        title=req.title or "",
        artist=req.artist or "",
        requester=req.requester,
    )
    return {
        "success": True,
        "track": track.to_dict(),
        "queue": player.queue_list(),
        "now_playing": player.now_playing_info(),
    }


@app.post("/ymusic/skip")
async def ymusic_skip():
    """현재 곡 스킵."""
    player = _require_ymusic()
    next_track = await player.skip()
    return {
        "success": True,
        "next": next_track.to_dict() if next_track else None,
        "now_playing": player.now_playing_info(),
    }


@app.post("/ymusic/pause")
async def ymusic_pause():
    player = _require_ymusic()
    await player.pause()
    return {"success": True, "now_playing": player.now_playing_info()}


@app.post("/ymusic/resume")
async def ymusic_resume():
    player = _require_ymusic()
    await player.resume()
    return {"success": True, "now_playing": player.now_playing_info()}


@app.post("/ymusic/stop")
async def ymusic_stop():
    player = _require_ymusic()
    await player.stop()
    return {"success": True, "now_playing": player.now_playing_info()}


@app.get("/ymusic/now-playing")
async def ymusic_now_playing():
    player = _require_ymusic()
    return {"now_playing": player.now_playing_info(), "queue": player.queue_list()}


# ── 오디오 스트림 ─────────────────────────────────────────────────

@app.websocket("/music/stream")
async def audio_stream(ws: WebSocket):
    """실시간 오디오 PCM 스트림 (int16 stereo 44100Hz)."""
    await ws.accept()
    if not mixer:
        await ws.close(code=1011)
        return

    await mixer.subscribe(ws)
    logger.info("Audio stream client connected")

    try:
        while True:
            msg = await ws.receive_text()
            if msg.strip().lower() == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        logger.info("Audio stream client disconnected")
    except Exception as exc:
        logger.warning("Audio stream error: %s", exc)
    finally:
        await mixer.unsubscribe(ws)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
