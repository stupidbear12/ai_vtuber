# -*- coding: utf-8 -*-
"""
app/main.py — ai_voice ElevenLabs TTS 서버 진입점

역할:
  - ElevenLabs API를 사용한 텍스트 음성 합성 (TTS)
  - 음성 디자인 (Voice Design) — 텍스트 설명으로 새 목소리 생성
  - 음성 목록 조회 및 기본 음성(시온) 설정 관리
  - 실시간 스트리밍 TTS (StreamingResponse)

포트: 8004

엔드포인트:
  POST /voice/tts          — 텍스트 → MP3 파일 반환
  POST /voice/tts/stream   — 텍스트 → MP3 스트리밍
  POST /voice/design       — 텍스트 프롬프트로 새 음성 생성
  GET  /voice/list         — 사용 가능한 음성 목록
  POST /voice/set-default  — 기본 음성(시온) 설정
  GET  /voice/status       — 현재 음성 설정 정보
  GET  /health             — 서버 상태 확인

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
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.voice_engine import ElevenLabsEngine

logger = logging.getLogger(__name__)

# ── 음성 엔진 싱글톤 ──────────────────────────────────────────────
# 모듈 로드 시 1회 생성, 이후 재사용 (API 키·설정 재로드 없음)
_engine: Optional[ElevenLabsEngine] = None


def get_engine() -> ElevenLabsEngine:
    """ElevenLabsEngine 싱글톤을 반환한다."""
    global _engine
    if _engine is None:
        _engine = ElevenLabsEngine()
    return _engine


# ── FastAPI 앱 생성 ───────────────────────────────────────────────
app = FastAPI(
    title="ai_voice — ElevenLabs TTS 서버",
    description=(
        "ElevenLabs API를 사용한 텍스트 음성 합성 모듈. "
        "시온(AI 버튜버) 캐릭터 음성 합성 및 음성 디자인을 지원합니다."
    ),
    version="1.0.0",
)

# ── CORS 미들웨어 ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ────────────────────────────────────────────────

class TTSRequest(BaseModel):
    """TTS 요청 모델."""
    text: str                           # 변환할 텍스트
    voice_id: Optional[str] = None      # 보이스 ID (None이면 기본 음성)
    emotion: Optional[str] = None       # 감정 태그 (happy/sad/calm 등)


class VoiceDesignRequest(BaseModel):
    """음성 디자인 요청 모델."""
    name: str           # 생성할 음성 이름 (예: "시온")
    description: str    # 음성 설명 (예: "밝고 귀여운 20대 여성, 약간 높은 톤, 한국어")


class SetDefaultRequest(BaseModel):
    """기본 음성 설정 요청 모델."""
    voice_id: str
    voice_name: Optional[str] = ""


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 안내 페이지."""
    engine = get_engine()
    api_status = "설정됨" if engine.api_key else "미설정 (.env에 ELEVENLABS_API_KEY 추가 필요)"
    return f"""
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_voice — ElevenLabs TTS 서버</h1>
    <p>API 키: <span style="color:{'#a6e3a1' if engine.api_key else '#f38ba8'}">{api_status}</span></p>
    <p>기본 음성: <code style="color:#cba6f7">{engine.default_voice_id}</code></p>
    <ul>
      <li><a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a></li>
      <li><a href="/health" style="color:#89dceb">서버 상태</a></li>
      <li><a href="/voice/status" style="color:#89dceb">음성 설정</a></li>
      <li><a href="/voice/list" style="color:#89dceb">음성 목록</a></li>
    </ul>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """서버 상태 확인 — ai_vtuber_core에서 헬스 체크용으로 호출."""
    engine = get_engine()
    return {
        "status": "ok",
        "module": "ai_voice",
        "version": "1.0.0",
        "api_key_set": bool(engine.api_key),
        "default_voice_id": engine.default_voice_id,
    }


@app.post("/voice/tts")
async def tts(req: TTSRequest):
    """텍스트를 MP3 음성 파일로 변환하여 반환한다.

    ElevenLabs TTS API를 호출해 음성을 생성하고 audio/mpeg로 반환.
    감정(emotion)에 따라 stability/style 파라미터가 자동으로 조정된다.

    Args:
        req.text: 변환할 텍스트
        req.voice_id: ElevenLabs 보이스 ID (None이면 기본 음성 사용)
        req.emotion: 감정 태그 (happy/excited/sad/worried/calm/thinking/
                               angry/surprised/love/shy)

    Returns:
        audio/mpeg 바이너리 (MP3 파일)

    Raises:
        400: text가 비어있을 때
        503: API 키 미설정 시
        500: ElevenLabs API 오류
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    engine = get_engine()
    try:
        audio_bytes = engine.synthesize(req.text, req.voice_id, req.emotion)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"},
        )
    except ValueError as e:
        # API 키 미설정 등 설정 오류
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"TTS 변환 실패: {e}")
        raise HTTPException(status_code=500, detail=f"TTS 변환 실패: {e}")


@app.post("/voice/tts/stream")
async def tts_stream(req: TTSRequest):
    """텍스트를 MP3 스트림으로 변환한다 (실시간 대화용).

    청크 단위로 오디오를 스트리밍하여 낮은 지연 시간으로 재생 시작 가능.
    방송 응답 TTS에 사용한다.

    Args:
        req.text: 변환할 텍스트
        req.voice_id: 보이스 ID (None이면 기본 음성)
        req.emotion: 감정 태그

    Returns:
        StreamingResponse (audio/mpeg) — 청크 단위 스트리밍
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어있습니다.")

    engine = get_engine()
    try:
        # get_sync_stream()은 동기 제너레이터 반환
        # FastAPI StreamingResponse는 동기/비동기 제너레이터 모두 지원
        stream = engine.get_sync_stream(req.text, req.voice_id, req.emotion)
        return StreamingResponse(
            stream,
            media_type="audio/mpeg",
            headers={"X-Content-Type-Options": "nosniff"},
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"TTS 스트리밍 실패: {e}")
        raise HTTPException(status_code=500, detail=f"TTS 스트리밍 실패: {e}")


@app.post("/voice/design")
async def design_voice(req: VoiceDesignRequest):
    """텍스트 설명(프롬프트)으로 새 음성을 생성한다.

    ElevenLabs Voice Generation API 2단계 흐름:
      1. 설명 기반 미리보기 생성
      2. 미리보기에서 실제 음성 저장

    생성된 voice_id는 /voice/set-default로 시온 기본 음성으로 설정 가능.

    Args:
        req.name: 음성 이름 (예: "시온")
        req.description: 음성 설명
                         (예: "밝고 귀여운 20대 여성, 약간 높은 톤, 에너지 넘치는 한국어")

    Returns:
        {"voice_id": str, "name": str, "description": str}

    Raises:
        503: API 키 미설정 시
        500: 음성 생성 실패
    """
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="name이 비어있습니다.")
    if not req.description.strip():
        raise HTTPException(status_code=400, detail="description이 비어있습니다.")

    engine = get_engine()
    try:
        result = engine.design_voice(req.name, req.description)
        return result
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"음성 디자인 실패: {e}")
        raise HTTPException(status_code=500, detail=f"음성 디자인 실패: {e}")


@app.get("/voice/list")
async def list_voices():
    """ElevenLabs 계정의 음성 목록을 조회한다.

    빌트인 음성 + 클론 음성 + Voice Design으로 만든 커스텀 음성 포함.
    기본 음성(시온)이 목록 맨 앞에 표시된다.

    Returns:
        {
            "voices": [{"voice_id", "name", "category", "is_custom", "is_default"}, ...],
            "total": int,
            "default_voice_id": str
        }
    """
    engine = get_engine()
    try:
        voices = engine.list_voices()
        return {
            "voices": voices,
            "total": len(voices),
            "default_voice_id": engine.default_voice_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"음성 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"음성 목록 조회 실패: {e}")


@app.post("/voice/set-default")
async def set_default_voice(req: SetDefaultRequest):
    """기본 음성(시온)을 설정한다.

    설정된 voice_id는 data/custom_voices.json에 저장되어 재시작 후에도 유지된다.
    /voice/tts 요청 시 voice_id를 생략하면 이 기본값이 사용된다.

    Args:
        req.voice_id: ElevenLabs 보이스 ID
        req.voice_name: 음성 이름 (표시용, 선택적)

    Returns:
        {"success": True, "default_voice_id": str, "message": str}
    """
    if not req.voice_id.strip():
        raise HTTPException(status_code=400, detail="voice_id가 비어있습니다.")

    engine = get_engine()
    engine.set_default_voice(req.voice_id, req.voice_name or "")

    return {
        "success": True,
        "default_voice_id": req.voice_id,
        "message": f"기본 음성이 '{req.voice_name or req.voice_id}'로 설정되었습니다.",
    }


@app.get("/voice/status")
async def voice_status():
    """현재 음성 설정 정보를 반환한다.

    API 키 설정 여부, 기본 음성 ID, 모델, 커스텀 음성 목록, 지원 감정 태그 포함.

    Returns:
        {
            "api_key_set": bool,
            "default_voice_id": str,
            "model": str,
            "custom_voices_count": int,
            "custom_voices": [...],
            "emotion_tags": [...]
        }
    """
    engine = get_engine()
    return engine.get_status()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8004, reload=True)
