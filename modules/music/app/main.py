# -*- coding: utf-8 -*-
"""
ai_music FastAPI 서버 (포트 8005)

엔드포인트:
  POST /music/generate    — 즉시 1곡 생성 (동기 대기)
  POST /music/queue       — DJ 큐에 생성 요청 추가
  POST /music/skip        — 현재 곡 스킵 → 다음 곡 크로스페이드
  GET  /music/now-playing — 현재 재생 중인 트랙 정보
  GET  /music/queue-list  — 대기 큐 목록
  GET  /music/status      — DJ·믹서·GPU 상태
  WS   /music/stream      — 실시간 오디오 PCM 스트림
  GET  /health            — 헬스 체크
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

from .music_engine import MusicEngine, GenerationParams, TrackMeta
from .dj_controller import DJController
from .audio_mixer import AudioMixer
from .track_queue import TrackQueue, QueueFullError

logger = logging.getLogger(__name__)

HOST = os.environ.get("AI_MUSIC_HOST", "0.0.0.0")
PORT = int(os.environ.get("AI_MUSIC_PORT", "8005"))

engine: Optional[MusicEngine] = None
queue: Optional[TrackQueue] = None
mixer: Optional[AudioMixer] = None
dj: Optional[DJController] = None


# ── Pydantic 스키마 ───────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="음악 설명 프롬프트")
    lyrics: Optional[str] = Field(None, description="가사 (없으면 인스트루멘탈)")
    genre: Optional[str] = Field(None, description="장르 힌트")
    bpm: Optional[int] = Field(None, ge=30, le=300)
    duration: Optional[float] = Field(90.0, ge=10, le=600)
    key_scale: Optional[str] = Field(None, description="키/스케일 (예: C Major)")
    play: bool = Field(False, description="True면 생성 후 즉시 재생")


class QueueRequest(BaseModel):
    prompt: str
    lyrics: Optional[str] = None
    genre: Optional[str] = None
    bpm: Optional[int] = None
    duration: Optional[float] = Field(90.0, ge=10, le=600)
    requester: Optional[str] = Field(None, description="요청자 닉네임")
    priority: int = Field(0, ge=0, le=10)


class SkipRequest(BaseModel):
    crossfade_sec: float = Field(3.0, ge=0.5, le=10.0)


# ── 헬퍼 ──────────────────────────────────────────────────────────

def _build_params(
    prompt: str,
    lyrics: Optional[str] = None,
    genre: Optional[str] = None,
    bpm: Optional[int] = None,
    duration: Optional[float] = 90.0,
    key_scale: Optional[str] = None,
) -> GenerationParams:
    caption = prompt.strip()
    if genre:
        caption = f"{caption}. Genre: {genre}"

    return GenerationParams(
        prompt=caption,
        lyrics=lyrics or "",
        bpm=bpm,
        duration=float(duration or 90.0),
        key_scale=key_scale or "",
    )


def _serialize_track(track: Optional[TrackMeta]) -> Optional[dict]:
    if track is None:
        return None
    return {
        "track_id": track.track_id,
        "prompt": track.prompt,
        "lyrics": track.lyrics,
        "genre": track.genre,
        "bpm": track.bpm,
        "key_scale": track.key_scale,
        "duration_sec": track.duration_sec,
        "file_path": str(track.file_path) if track.file_path else None,
        "seed": track.seed,
        "generation_time_sec": track.generation_time_sec,
    }


async def _play_track(track: TrackMeta) -> None:
    """생성된 트랙을 믹서에 로드해 재생."""
    if not track.file_path or not Path(track.file_path).exists():
        raise HTTPException(status_code=500, detail="생성된 오디오 파일을 찾을 수 없습니다.")
    await mixer.load_track(Path(track.file_path))
    await mixer.play()


# ── Lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, queue, mixer, dj

    logger.info("ai_music 모듈 초기화 중...")

    engine = MusicEngine()
    await engine.initialize()

    queue = TrackQueue()
    mixer = AudioMixer()
    await mixer.initialize()

    dj = DJController(engine=engine, queue=queue, mixer=mixer)
    await dj.start()

    logger.info("ai_music 모듈 준비 완료 (port %s, engine_ready=%s)", PORT, engine.is_ready)
    yield

    logger.info("ai_music 모듈 종료 중...")
    await dj.stop()
    await mixer.shutdown()
    await engine.shutdown()


# ── FastAPI 앱 ────────────────────────────────────────────────────

app = FastAPI(title="ai_music", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    gpu = await engine.get_gpu_status() if engine else {}
    return {
        "status": "ok",
        "module": "ai_music",
        "engine_ready": engine.is_ready if engine else False,
        "now_playing": dj.now_playing_info() if dj else None,
        "queue_size": queue.size() if queue else 0,
        "buffer": dj.buffer_status() if dj else None,
        "gpu": gpu,
    }


@app.get("/music/status")
async def music_status():
    """DJ·믹서·GPU 상세 상태."""
    gpu = await engine.get_gpu_status() if engine else {}
    return {
        "engine_ready": engine.is_ready if engine else False,
        "playback": mixer.get_playback_state() if mixer else None,
        "now_playing": dj.now_playing_info() if dj else None,
        "buffer": dj.buffer_status() if dj else None,
        "queue_size": queue.size() if queue else 0,
        "gpu": gpu,
    }


@app.post("/music/generate")
async def generate(req: GenerateRequest):
    """단일 곡 즉시 생성."""
    if not engine or not engine.is_ready:
        raise HTTPException(
            status_code=503,
            detail="MusicEngine이 준비되지 않았습니다. ACESTEP_STUB=1 또는 acestep 설치를 확인하세요.",
        )

    params = _build_params(
        req.prompt, req.lyrics, req.genre, req.bpm, req.duration, req.key_scale,
    )

    try:
        track = await engine.generate(params)
    except Exception as exc:
        logger.error("Generate failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"음악 생성 실패: {exc}") from exc

    if req.genre:
        track.genre = req.genre

    if req.play:
        await _play_track(track)

    return {
        "success": True,
        "track": _serialize_track(track),
        "playing": req.play,
    }


@app.post("/music/queue")
async def add_to_queue(req: QueueRequest):
    """DJ 큐에 생성 요청 추가 (백그라운드에서 생성·재생)."""
    if not queue or not dj:
        raise HTTPException(status_code=503, detail="ai_music가 초기화되지 않았습니다.")

    try:
        if req.requester:
            item_id = await dj.handle_viewer_request(
                prompt=req.prompt,
                requester=req.requester,
                genre=req.genre,
                bpm=req.bpm,
                duration=req.duration,
                priority=req.priority,
            )
        else:
            params = _build_params(
                req.prompt, req.lyrics, req.genre, req.bpm, req.duration,
            )
            item_id = await queue.enqueue(
                params,
                requester=req.requester,
                priority=req.priority,
            )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc

    return {
        "success": True,
        "item_id": item_id,
        "queue_size": queue.size(),
    }


@app.post("/music/skip")
async def skip(req: SkipRequest):
    """현재 곡 스킵."""
    if not dj:
        raise HTTPException(status_code=503, detail="DJController가 초기화되지 않았습니다.")

    result = await dj.skip_current(crossfade_sec=req.crossfade_sec)
    return {
        "success": True,
        "skipped": _serialize_track(result.get("skipped")),
        "next": _serialize_track(result.get("next")),
    }


@app.get("/music/now-playing")
async def now_playing():
    """현재 재생 중인 트랙."""
    if not dj:
        return {"now_playing": None}
    return {"now_playing": dj.now_playing_info()}


@app.get("/music/queue-list")
async def queue_list():
    """대기 큐 목록."""
    if not queue:
        return {"items": [], "total": 0}
    items = queue.list_all()
    return {"items": items, "total": len(items)}


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
