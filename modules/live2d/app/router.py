# -*- coding: utf-8 -*-
"""
app/router.py — Live2D 제어 FastAPI 라우터

엔드포인트 목록:
  GET  /live2d/              — 웹 뷰어로 리다이렉트
  WS   /live2d/ws            — 브라우저 ↔ 서버 실시간 WebSocket 채널
  GET  /live2d/status        — 연결된 클라이언트 수
  POST /live2d/params        — 파라미터 직접 주입 (예: {"ParamAngleX": 15})
  POST /live2d/emotion       — 감정 이름 → expression 자동 변환
  POST /live2d/expression    — expression 직접 지정 (이름 or 인덱스)
  POST /live2d/motion        — 모션 재생 (group, index)
  POST /live2d/motion/play_once — 모션 한 번 재생 후 idle 복귀
  POST /live2d/mouth         — 립싱크 값 설정 (0.0 ~ 1.0)
  POST /live2d/mouth/clear   — 립싱크 레이어 제거
  POST /live2d/reaction      — 반응 애니메이션 (nod/shake/surprised/superchat)
  POST /live2d/idle/start    — Idle 애니메이션 시작
  POST /live2d/idle/stop     — Idle 애니메이션 정지
  POST /live2d/chat          — 데스크톱 펫 채팅 (ai_chat 모듈 위임)
"""

import logging
import os
from pathlib import Path
from typing import Union

import aiohttp

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.ws_manager import ws_manager

logger = logging.getLogger(__name__)

# ── 라우터 초기화 ────────────────────────────────────────────────
live2d_router = APIRouter(prefix="/live2d", tags=["live2d"])

# 정적 파일 디렉토리 (ai_live2d/static/)
_STATIC_DIR = Path(__file__).parent.parent / "static"


def _chat_url() -> str:
    """ai_chat 모듈 베이스 URL (환경변수 AI_CHAT_URL, 기본 localhost:8002)."""
    return os.environ.get("AI_CHAT_URL", "http://localhost:8002").rstrip("/")


# ── 요청 모델 ────────────────────────────────────────────────────

class ParamRequest(BaseModel):
    """파라미터 직접 주입 요청 모델."""
    params: dict  # 예: {"ParamAngleX": 15.0, "ParamEyeLOpen": 0.8}

class EmotionRequest(BaseModel):
    """감정 이름 요청 모델."""
    emotion: str  # 예: "happy", "sad", "surprised", "thinking", "calm"

class ExpressionRequest(BaseModel):
    """표정 직접 지정 요청 모델."""
    expression: Union[str, int]  # 이름("F04") 또는 0-based 인덱스(3)

class MotionRequest(BaseModel):
    """모션 재생 요청 모델."""
    group: str = ""   # 모션 그룹명 (예: "Idle")
    index: int = 0    # 그룹 내 인덱스

class PlayOnceRequest(BaseModel):
    """모션 한 번 재생 후 idle 복귀 요청 모델."""
    group: str = "Idle"
    index: int = 0
    duration: int = 5330  # 모션 재생 후 idle 복귀까지 대기 시간 (ms)

class MouthRequest(BaseModel):
    """립싱크 값 설정 요청 모델."""
    value: float  # 0.0 (입 닫힘) ~ 1.0 (입 최대 개방)

class ReactionRequest(BaseModel):
    """반응 애니메이션 요청 모델."""
    name: str  # "nod" | "shake" | "surprised" | "superchat"

class ChatRequest(BaseModel):
    """데스크톱 펫 채팅 요청 모델."""
    message: str  # 사용자 입력 텍스트


# ── 라우트 정의 ──────────────────────────────────────────────────

@live2d_router.get("/")
async def viewer():
    """웹 뷰어 홈 — 정적 HTML로 리다이렉트."""
    return RedirectResponse("/live2d/static/")


@live2d_router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """실시간 Live2D 제어 WebSocket 엔드포인트.

    브라우저(웹 뷰어 / Electron 펫)에서 연결해
    서버에서 보내는 cmd 메시지를 수신한다.
    """
    await ws_manager.connect(ws)
    try:
        while True:
            # 클라이언트 → 서버 메시지 수신 (현재 로깅만)
            raw = await ws.receive_text()
            print(f"[Live2D WS 수신] {raw[:120]}")
    except WebSocketDisconnect:
        # 클라이언트 연결 해제 시 풀에서 제거
        ws_manager.disconnect(ws)


@live2d_router.get("/status")
async def status():
    """현재 연결된 WebSocket 클라이언트 수 반환."""
    return {"clients": ws_manager.count}


@live2d_router.post("/params")
async def set_params(req: ParamRequest):
    """Cubism 파라미터를 직접 주입한다.

    브라우저 측 AnimSystem.setManualParams()를 호출해
    모델의 특정 파라미터 값을 즉시 변경한다.
    """
    await ws_manager.broadcast({"cmd": "set_params", "params": req.params})
    return {"ok": True, "clients": ws_manager.count}


@live2d_router.post("/emotion")
async def set_emotion(req: EmotionRequest):
    """감정 이름으로 표정을 자동 변환한다.

    브라우저 측 EMOTION_MAP으로 expression 이름을 변환해 적용한다.
    지원 감정: calm, happy, sad, surprised, thinking, angry, excited 등
    """
    await ws_manager.broadcast({"cmd": "set_emotion", "emotion": req.emotion})
    return {"ok": True}


@live2d_router.post("/expression")
async def set_expression(req: ExpressionRequest):
    """표정(expression)을 이름 또는 인덱스로 직접 지정한다."""
    await ws_manager.broadcast({"cmd": "set_expression", "expression": req.expression})
    return {"ok": True}


@live2d_router.post("/motion")
async def play_motion(req: MotionRequest):
    """모션을 재생한다. 완료 후 브라우저가 자동으로 idle 복귀한다."""
    await ws_manager.broadcast({"cmd": "set_motion", "group": req.group, "index": req.index})
    return {"ok": True}


@live2d_router.post("/motion/play_once")
async def play_motion_once(req: PlayOnceRequest):
    """모션을 한 번 재생한 뒤 duration ms 후 idle로 복귀한다."""
    await ws_manager.broadcast({
        "cmd": "set_motion",
        "group": req.group,
        "index": req.index,
        "duration": req.duration,
    })
    return {"ok": True}


@live2d_router.post("/mouth")
async def set_mouth(req: MouthRequest):
    """립싱크 입 개방값을 설정한다 (0.0 = 닫힘, 1.0 = 최대 개방)."""
    clamped = max(0.0, min(1.0, req.value))
    await ws_manager.broadcast({"cmd": "set_mouth", "value": clamped})
    return {"ok": True}


@live2d_router.post("/mouth/clear")
async def clear_mouth():
    """립싱크 레이어를 제거하고 입을 닫는다."""
    await ws_manager.broadcast({"cmd": "clear_mouth"})
    return {"ok": True}


@live2d_router.post("/reaction")
async def trigger_reaction(req: ReactionRequest):
    """반응 애니메이션을 트리거한다.

    지원 반응:
      - nod: 고개 끄덕임
      - shake: 고개 가로젓기
      - surprised: 놀람 (눈 크게)
      - superchat: 슈퍼챗 감사 반응
    """
    await ws_manager.broadcast({"cmd": "reaction", "name": req.name})
    return {"ok": True}


@live2d_router.post("/idle/start")
async def idle_start():
    """Idle 애니메이션을 시작한다 (호흡 + 눈 깜빡임 + 고개 흔들림)."""
    await ws_manager.broadcast({"cmd": "idle_start"})
    return {"ok": True}


@live2d_router.post("/idle/stop")
async def idle_stop():
    """Idle 애니메이션을 정지한다."""
    await ws_manager.broadcast({"cmd": "idle_stop"})
    return {"ok": True}


# ── 데스크톱 펫 채팅 엔드포인트 ─────────────────────────────────

@live2d_router.post("/chat")
async def chat(req: ChatRequest):
    """데스크톱 펫 채팅 — ai_chat 모듈에 위임해 에메스 응답을 생성한다.

    처리 흐름:
      1. ai_chat POST /chat (mode=pet) 호출
      2. 응답의 emotion으로 WebSocket 클라이언트 표정 브로드캐스트
      3. reply + emotion 반환

    Args:
        req.message: 사용자 입력 텍스트

    Returns:
        reply: 에메스 응답 텍스트
        emotion: 감정 태그 (Live2D 표정 변경에 사용)
    """
    text = ""
    emotion = "calm"
    error_msg = None

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_chat_url()}/chat",
                json={"message": req.message, "mode": "pet"},
                timeout=aiohttp.ClientTimeout(total=30.0),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"ai_chat 응답 오류: HTTP {resp.status} — {body[:200]}")
                data = await resp.json()
                text = data.get("reply", "")
                emotion = data.get("emotion", "calm")
                if data.get("error"):
                    error_msg = data["error"]
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[Live2D] ai_chat 호출 실패: {e}")
        text = "죄송해요, 잠시 후 다시 말씀해주세요."
        emotion = "worried"

    await ws_manager.broadcast({"cmd": "set_emotion", "emotion": emotion})

    result: dict = {"reply": text, "emotion": emotion}
    if error_msg:
        result["error"] = error_msg
    return result


# ── 정적 파일 마운트 ─────────────────────────────────────────────

def mount_static(app) -> None:
    """FastAPI 앱에 Live2D 정적 파일 디렉토리를 마운트한다.

    /live2d/static/ 경로로 웹 뷰어 HTML, 모델 파일, JS 라이브러리를 제공한다.

    Args:
        app: FastAPI 애플리케이션 인스턴스
    """
    app.mount(
        "/live2d/static",
        StaticFiles(directory=str(_STATIC_DIR), html=True),
        name="live2d_static",
    )
