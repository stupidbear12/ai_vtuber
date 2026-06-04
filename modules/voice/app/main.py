# -*- coding: utf-8 -*-
"""
app/main.py — ai_voice 독립 서버 진입점

역할:
  - 음성 합성 모듈 (현재 스텁 상태 — ElevenLabs 연동 예정)
  - 텍스트 → 음성 파일 변환 API 제공

포트: 8004

현재 상태:
  - POST /voice/synthesize 는 스텁으로 placeholder 응답 반환
  - ElevenLabs API 연동은 추후 구현 예정

실행 방법:
    cd ai_voice
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8004
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── FastAPI 앱 생성 ───────────────────────────────────────────────
app = FastAPI(
    title="ai_voice — 음성 합성 서버",
    description=(
        "텍스트 음성 합성 모듈. "
        "현재는 스텁 상태이며, ElevenLabs API 연동이 예정되어 있습니다."
    ),
    version="0.1.0",
)

# ── CORS 미들웨어 ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    """음성 합성 요청 모델."""
    text: str                          # 합성할 텍스트
    voice_id: Optional[str] = None     # ElevenLabs 보이스 ID (선택적)
    output_filename: Optional[str] = None  # 출력 파일명 (선택적)

class SynthesizeResponse(BaseModel):
    """음성 합성 응답 모델."""
    success: bool
    audio_path: Optional[str] = None   # 생성된 오디오 파일 경로
    message: str = ""                   # 상태 메시지


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 간단한 안내 페이지."""
    return """
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_voice 서버 실행 중</h1>
    <p style="color:#f9e2af">현재 스텁 상태입니다. ElevenLabs 연동 예정.</p>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/health" style="color:#89dceb">서버 상태 확인</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """서버 상태 확인 — ai_vtuber_core에서 헬스 체크용으로 호출."""
    return {
        "status": "ok",
        "module": "ai_voice",
        "version": "0.1.0",
        "stub": True,          # 스텁 상태임을 명시
        "planned": "ElevenLabs TTS 연동 예정",
    }


@app.post("/voice/synthesize", response_model=SynthesizeResponse)
async def synthesize(req: SynthesizeRequest):
    """텍스트를 음성으로 합성한다.

    현재 상태: 스텁 — 항상 "구현 예정" 응답 반환
    예정 기능: ElevenLabs API를 사용해 실제 음성 파일 생성

    Args:
        req.text: 합성할 텍스트
        req.voice_id: ElevenLabs 보이스 ID (선택적)
        req.output_filename: 출력 파일명 (선택적)

    Returns:
        success: False (스텁 상태)
        message: 구현 예정 안내
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    # TODO: ElevenLabs API 연동 구현
    # import elevenlabs
    # audio = elevenlabs.generate(text=req.text, voice=req.voice_id or "default")
    # audio_path = save_audio(audio, req.output_filename)
    # return SynthesizeResponse(success=True, audio_path=audio_path)

    return SynthesizeResponse(
        success=False,
        message="ElevenLabs TTS 연동 구현 예정입니다. 현재 스텁 상태.",
    )


@app.get("/voice/voices")
async def list_voices():
    """사용 가능한 보이스 목록 반환 (스텁).

    예정 기능: ElevenLabs API에서 보이스 목록 조회
    """
    return {
        "stub": True,
        "voices": [],
        "message": "ElevenLabs API 연동 후 보이스 목록이 표시됩니다.",
    }


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)
