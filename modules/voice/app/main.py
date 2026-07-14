# -*- coding: utf-8 -*-
"""
app/main.py — ai_voice TTS 서버 진입점

엔진: GPT-SoVITS (시온 전용 파인튜닝 모델, sion_jfla_v3)

포트: 8004

사전 준비:
  GPT-SoVITS API 서버를 별도 프로세스로 먼저 실행해야 한다.
    cd C:\\Users\\thtgg\\workspace2\\GPT-SoVITS
    python api_v2.py -a 127.0.0.1 -p 9880

엔드포인트:
  POST /voice/tts    — 텍스트 → WAV 반환 (한국어/영어)
  GET  /voice/status — 현재 엔진 상태
  GET  /health        — 서버 상태 확인

실행 방법:
    cd modules/voice
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
"""

import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.voice_engine import GPTSoVITSEngine

logger = logging.getLogger(__name__)

# ── 음성 엔진 싱글톤 ──────────────────────────────────────────────
_engine: Optional[GPTSoVITSEngine] = None


def get_engine() -> GPTSoVITSEngine:
    global _engine
    if _engine is None:
        _engine = GPTSoVITSEngine()
    return _engine


# ── FastAPI 앱 ────────────────────────────────────────────────────
app = FastAPI(
    title="ai_voice — GPT-SoVITS 서버",
    description="GPT-SoVITS 파인튜닝 모델 기반 AI VTuber '시온'의 음성 합성 모듈.",
    version="4.0.0",
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
    language: Optional[str] = None   # "ko"|"en"|"ja"|"auto" (None이면 자동 감지)
    emotion: Optional[str] = None    # 하위 호환용 (현재 미사용)
    top_p: float = 0.8
    temperature: float = 0.8


# ══════════════════════════════════════════════════════════════════
# 엔드포인트
# ══════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    engine = get_engine()
    status = engine.get_status()
    return f"""
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_voice — GPT-SoVITS TTS 서버</h1>
    <p>GPT-SoVITS API: <code style="color:#cba6f7">{status['api_url']}</code></p>
    <p>지원 언어: <code style="color:#89b4fa">{', '.join(status['languages'])}</code></p>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/health" style="color:#89dceb">서버 상태</a></li>
      <li><a href="/voice/status" style="color:#89dceb">엔진 상태</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "module": "ai_voice",
        "version": "4.0.0",
        "engine": "gpt-sovits",
    }


@app.post("/voice/tts")
async def tts(req: TTSRequest):
    """텍스트를 GPT-SoVITS로 음성 변환 (WAV).

    Returns:
        audio/wav 바이너리
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    engine = get_engine()
    try:
        audio_bytes = await engine.synthesize_async(
            req.text,
            language=req.language,
            top_p=req.top_p,
            temperature=req.temperature,
        )
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"},
        )
    except Exception as e:
        logger.error(f"GPT-SoVITS TTS 변환 실패: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {e}")


@app.get("/voice/status")
async def voice_status():
    """현재 엔진 상태 정보."""
    engine = get_engine()
    return engine.get_status()


@app.post("/voice/reload")
async def voice_reload():
    """설정 파일을 다시 읽어 레퍼런스 오디오 등을 런타임에 갱신한다."""
    engine = get_engine()
    result = engine.reload_config()
    return result


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)
