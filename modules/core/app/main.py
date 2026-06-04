# -*- coding: utf-8 -*-
"""
app/main.py — ai_vtuber_core 오케스트레이터 서버 진입점

역할:
  - 전체 시스템 상태 모니터링 (각 모듈 헬스 체크)
  - 통합 채팅 파이프라인 (ai_chat → ai_live2d 동시 제어)
  - 방송 채팅 수집 시작/중지 (ai_broadcast 위임)
  - 각 모듈의 단일 진입점 역할

포트: 8000 (메인 API)

모듈 구성:
  ai_live2d    → http://localhost:8001  (Live2D 아바타)
  ai_chat      → http://localhost:8002  (Gemini 채팅 엔진)
  ai_broadcast → http://localhost:8003  (방송 채팅 수집)
  ai_voice     → http://localhost:8004  (음성 합성, 예정)

실행 방법:
    cd ai_vtuber_core
    pip install -r requirements.txt
    cp .env.example .env
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.orchestrator import (
    check_all_modules,
    run_chat_pipeline,
    start_broadcast,
    stop_broadcast,
    ModuleConfig,
)

# ── FastAPI 앱 생성 ───────────────────────────────────────────────
app = FastAPI(
    title="ai_vtuber_core — 오케스트레이터 API",
    description=(
        "AI 버튜버 에메스(emeth) 전체 시스템의 오케스트레이터. "
        "ai_live2d(아바타), ai_chat(채팅 엔진), ai_broadcast(방송 연동), "
        "ai_voice(음성)를 통합 제어합니다."
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


# ── 요청 모델 ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """통합 채팅 파이프라인 요청 모델."""
    message: str                        # 사용자 채팅 입력
    mode: Optional[str] = "pet"         # "pet" 또는 "broadcast"

class BroadcastStartRequest(BaseModel):
    """방송 채팅 수집 시작 요청 모델."""
    platform: str    # "youtube" 또는 "chzzk"
    channel_id: str  # 유튜브 영상 ID 또는 치지직 채널 해시 ID


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """루트 접속 시 시스템 대시보드."""
    cfg = ModuleConfig()
    return f"""
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_vtuber_core 오케스트레이터</h1>
    <h2 style="color:#cba6f7">모듈 연결 정보</h2>
    <table style="border-collapse:collapse">
      <tr><th style="text-align:left;padding:8px;color:#89b4fa">모듈</th>
          <th style="text-align:left;padding:8px;color:#89b4fa">URL</th>
          <th style="text-align:left;padding:8px;color:#89b4fa">역할</th></tr>
      <tr><td style="padding:8px">ai_live2d</td>
          <td style="padding:8px"><a href="{cfg.live2d_url()}" style="color:#89dceb">{cfg.live2d_url()}</a></td>
          <td style="padding:8px">Live2D 아바타, WebSocket, Electron 펫</td></tr>
      <tr><td style="padding:8px">ai_chat</td>
          <td style="padding:8px"><a href="{cfg.chat_url()}" style="color:#89dceb">{cfg.chat_url()}</a></td>
          <td style="padding:8px">Gemini 채팅 엔진 (에메스 캐릭터)</td></tr>
      <tr><td style="padding:8px">ai_broadcast</td>
          <td style="padding:8px"><a href="{cfg.broadcast_url()}" style="color:#89dceb">{cfg.broadcast_url()}</a></td>
          <td style="padding:8px">치지직/유튜브 방송 채팅 수집</td></tr>
      <tr><td style="padding:8px">ai_voice</td>
          <td style="padding:8px"><a href="{cfg.voice_url()}" style="color:#89dceb">{cfg.voice_url()}</a></td>
          <td style="padding:8px">음성 합성 (ElevenLabs 예정)</td></tr>
    </table>
    <br>
    <a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a> &nbsp;
    <a href="/status" style="color:#89dceb">시스템 상태</a>
    </body></html>
    """


@app.get("/health")
async def health_check():
    """ai_vtuber_core 자체 상태 확인."""
    return {
        "status": "ok",
        "module": "ai_vtuber_core",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/status")
async def system_status():
    """전체 시스템 상태를 조회한다.

    모든 모듈(ai_live2d, ai_chat, ai_broadcast, ai_voice)에
    병렬로 /health 요청을 보내 상태를 수집한다.

    Returns:
        modules: 각 모듈 상태 딕셔너리
        all_online: 모든 모듈이 온라인인지 여부
        checked_at: 조회 시각
    """
    result = await check_all_modules()
    result["checked_at"] = datetime.now().isoformat()
    return JSONResponse(content=result)


@app.post("/pipeline/chat")
async def pipeline_chat(req: ChatRequest):
    """통합 채팅 파이프라인을 실행한다.

    흐름:
      1. ai_chat 모듈로 에메스 응답 생성
      2. ai_live2d 모듈로 표정 변경
      3. (예정) ai_voice 모듈로 음성 합성

    Args:
        req.message: 사용자 채팅 입력
        req.mode: "pet" (기본) 또는 "broadcast"

    Returns:
        reply: 에메스 응답 텍스트
        emotion: 감정 태그
        live2d_updated: 표정 변경 성공 여부
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있습니다.")

    mode = req.mode or "pet"
    if mode not in ("pet", "broadcast"):
        raise HTTPException(
            status_code=400,
            detail="mode는 'pet' 또는 'broadcast'만 허용됩니다."
        )

    result = await run_chat_pipeline(req.message, mode=mode)

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


@app.post("/broadcast/start")
async def broadcast_start(req: BroadcastStartRequest):
    """방송 채팅 수집을 시작한다 — ai_broadcast 모듈에 위임.

    Args:
        req.platform: "youtube" 또는 "chzzk"
        req.channel_id: 유튜브 영상 ID 또는 치지직 채널 해시 ID

    Returns:
        ai_broadcast 모듈의 응답
    """
    if not req.channel_id.strip():
        raise HTTPException(status_code=400, detail="channel_id가 비어있습니다.")

    try:
        result = await start_broadcast(req.platform, req.channel_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result


@app.post("/broadcast/stop")
async def broadcast_stop():
    """방송 채팅 수집을 중지한다 — ai_broadcast 모듈에 위임.

    Returns:
        ai_broadcast 모듈의 응답 (최종 통계 포함)
    """
    try:
        result = await stop_broadcast()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return result


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
