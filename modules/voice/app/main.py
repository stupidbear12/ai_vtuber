# -*- coding: utf-8 -*-
"""
app/main.py — ai_voice TTS 서버 진입점

엔진:
  - Edge TTS (기본) — 일반 TTS, GPU 불필요
  - Chatterbox TTS — 데뷔곡 보컬용, 음성 클로닝

포트: 8004

엔드포인트 (Edge TTS — 기본):
  POST /voice/tts              — 텍스트 → MP3 반환
  POST /voice/tts/stream       — 텍스트 → MP3 스트리밍
  GET  /voice/voices           — 사용 가능한 Edge TTS 음성 목록
  POST /voice/set-voice        — Edge TTS 음성 변경
  GET  /voice/status           — 현재 엔진 상태

엔드포인트 (Chatterbox — 보컬 전용):
  POST /voice/chatterbox/tts   — Chatterbox로 WAV 생성
  GET  /voice/references       — 레퍼런스 오디오 목록
  POST /voice/references/upload — 레퍼런스 오디오 업로드
  POST /voice/set-default      — 기본 레퍼런스 오디오 설정

공통:
  GET  /health                 — 서버 상태 확인

실행 방법:
    cd modules/voice
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.voice_engine import EdgeTTSEngine, ChatterboxEngine, REFERENCE_AUDIO_DIR

logger = logging.getLogger(__name__)

# ── 음성 엔진 싱글톤 ──────────────────────────────────────────────
_edge_engine: Optional[EdgeTTSEngine] = None
_chatterbox_engine: Optional[ChatterboxEngine] = None


def get_edge_engine() -> EdgeTTSEngine:
    global _edge_engine
    if _edge_engine is None:
        _edge_engine = EdgeTTSEngine()
    return _edge_engine


def get_chatterbox_engine() -> ChatterboxEngine:
    global _chatterbox_engine
    if _chatterbox_engine is None:
        _chatterbox_engine = ChatterboxEngine()
    return _chatterbox_engine


# ── FastAPI 앱 ────────────────────────────────────────────────────
app = FastAPI(
    title="ai_voice — Edge TTS + Chatterbox 서버",
    description=(
        "Edge TTS 기반 일반 TTS + Chatterbox 보컬 클로닝. "
        "AI VTuber '시온'의 음성 합성 모듈."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ────────────────────────────────────────────────

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None             # Edge TTS 음성 이름
    # voice_id, reference_name은 하위 호환용
    reference_name: Optional[str] = None
    voice_id: Optional[str] = None
    emotion: Optional[str] = None
    rate: Optional[str] = None              # Edge TTS 속도 (예: "+10%")
    pitch: Optional[str] = None             # Edge TTS 피치 (예: "+5Hz")
    cfg_weight: float = 0.5                 # Chatterbox 전용


class SetVoiceRequest(BaseModel):
    voice: str                              # Edge TTS 음성 이름 (예: "ko-KR-SunHiNeural")


class SetDefaultRequest(BaseModel):
    reference_name: str


# ══════════════════════════════════════════════════════════════════
# Edge TTS 엔드포인트 (기본)
# ══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    edge = get_edge_engine()
    edge_status = edge.get_status()
    return f"""
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_voice — TTS 서버 v3 (Edge TTS + Chatterbox)</h1>
    <h2>기본 엔진: Edge TTS</h2>
    <p>현재 음성: <code style="color:#cba6f7">{edge_status['voice']}</code></p>
    <p>감정 태그: <code style="color:#89b4fa">{', '.join(edge_status['emotion_tags'])}</code></p>
    <h2>보컬 엔진: Chatterbox (데뷔곡 전용)</h2>
    <p><code>POST /voice/chatterbox/tts</code> 로 사용</p>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/health" style="color:#89dceb">서버 상태</a></li>
      <li><a href="/voice/status" style="color:#89dceb">엔진 상태</a></li>
      <li><a href="/voice/voices" style="color:#89dceb">Edge TTS 음성 목록</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    edge = get_edge_engine()
    return {
        "status": "ok",
        "module": "ai_voice",
        "version": "3.0.0",
        "primary_engine": "edge-tts",
        "edge_voice": edge.voice,
    }


@app.post("/voice/tts")
async def tts(req: TTSRequest):
    """텍스트를 Edge TTS로 음성 변환 (MP3).

    Returns:
        audio/mpeg 바이너리
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    engine = get_edge_engine()
    try:
        audio_bytes = await engine.synthesize_async(
            req.text,
            voice=req.voice,
            emotion=req.emotion,
            rate=req.rate,
            pitch=req.pitch,
        )
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except Exception as e:
        logger.error(f"Edge TTS 변환 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {e}")


@app.post("/voice/tts/stream")
async def tts_stream(req: TTSRequest):
    """텍스트를 Edge TTS로 스트리밍 변환 (MP3 청크)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    engine = get_edge_engine()

    async def audio_stream():
        async for chunk in engine.get_stream(
            req.text,
            voice=req.voice,
            emotion=req.emotion,
            rate=req.rate,
            pitch=req.pitch,
        ):
            yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="audio/mpeg",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.get("/voice/voices")
async def list_voices(locale: str = "ko-KR"):
    """사용 가능한 Edge TTS 음성 목록."""
    engine = get_edge_engine()
    try:
        voices = await engine.list_voices(locale=locale)
        return {"voices": voices, "total": len(voices), "locale": locale}
    except Exception as e:
        logger.error(f"음성 목록 조회 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"음성 목록 조회 실패: {e}")


@app.post("/voice/set-voice")
async def set_voice(req: SetVoiceRequest):
    """Edge TTS 기본 음성을 변경."""
    engine = get_edge_engine()
    engine.set_voice(req.voice)
    return {"success": True, "voice": req.voice}


@app.get("/voice/status")
async def voice_status():
    """현재 엔진 상태 정보."""
    edge = get_edge_engine()
    result = edge.get_status()
    result["chatterbox_available"] = True
    return result


# ══════════════════════════════════════════════════════════════════
# Chatterbox 엔드포인트 (데뷔곡 보컬 전용)
# ══════════════════════════════════════════════════════════════════

@app.post("/voice/chatterbox/tts")
async def chatterbox_tts(req: TTSRequest):
    """Chatterbox로 WAV 생성 (음성 클로닝, 데뷔곡 보컬용).

    Returns:
        audio/wav 바이너리
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    engine = get_chatterbox_engine()
    ref_name = req.reference_name or req.voice_id or None
    try:
        audio_bytes = engine.synthesize(
            req.text,
            reference_name=ref_name,
            emotion=req.emotion,
            cfg_weight=req.cfg_weight,
        )
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except Exception as e:
        logger.error(f"Chatterbox TTS 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatterbox TTS 실패: {e}")


@app.get("/voice/references")
async def list_references():
    """Chatterbox 레퍼런스 오디오 목록."""
    engine = get_chatterbox_engine()
    refs = engine.list_references()
    return {
        "references": refs,
        "total": len(refs),
        "default_reference": engine.default_reference,
        "reference_audio_dir": str(REFERENCE_AUDIO_DIR),
    }


@app.post("/voice/references/upload")
async def upload_reference(
    file: UploadFile = File(...),
    name: Optional[str] = None,
):
    """Chatterbox 레퍼런스 오디오 WAV 파일 업로드."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다.")

    ref_name = name or Path(file.filename).stem
    if not ref_name:
        raise HTTPException(status_code=400, detail="name을 지정해주세요.")

    REFERENCE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    save_path = REFERENCE_AUDIO_DIR / f"{ref_name}.wav"

    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 저장 실패: {e}")
    finally:
        file.file.close()

    engine = get_chatterbox_engine()
    engine._scan_references()

    return {
        "name": ref_name,
        "path": str(save_path),
        "message": f"레퍼런스 오디오 '{ref_name}' 등록 완료.",
    }


@app.post("/voice/set-default")
async def set_default(req: SetDefaultRequest):
    """Chatterbox 기본 레퍼런스 오디오 설정."""
    if not req.reference_name.strip():
        raise HTTPException(status_code=400, detail="reference_name이 비어있습니다.")

    engine = get_chatterbox_engine()
    engine._scan_references()
    if req.reference_name not in engine._references:
        raise HTTPException(
            status_code=404,
            detail=f"레퍼런스 오디오 '{req.reference_name}'이 없습니다. "
                   f"사용 가능: {list(engine._references.keys())}",
        )

    engine.set_default_reference(req.reference_name)
    return {
        "success": True,
        "default_reference": req.reference_name,
    }


# ── 구형 API 호환 엔드포인트 ─────────────────────────────────────

@app.get("/voice/list")
async def list_voices_compat():
    """구형 API 호환 — Edge TTS 음성 목록 반환."""
    engine = get_edge_engine()
    try:
        voices = await engine.list_voices()
        return {
            "voices": [
                {
                    "voice_id": v["short_name"],
                    "name": v["local_name"],
                    "gender": v["gender"],
                    "is_current": v["is_current"],
                    "category": "edge_tts",
                }
                for v in voices
            ],
            "total": len(voices),
            "note": "Edge TTS 엔진 사용 중. Chatterbox는 /voice/chatterbox/tts 로 접근.",
        }
    except Exception:
        return {"voices": [], "total": 0, "note": "Edge TTS 음성 목록 조회 실패"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)
