# -*- coding: utf-8 -*-
"""
live2d_web/router.py - Live2D 웹 뷰어용 FastAPI 라우터 (mao_pro 대응)

엔드포인트:
    GET  /live2d/          - 뷰어 HTML 반환 (투명: ?transparent=1)
    WS   /live2d/ws        - 브라우저 ↔ 서버 실시간 채널
    POST /live2d/params    - 파라미터 직접 주입
    POST /live2d/emotion   - 감정 이름 → expression 변환
    POST /live2d/expression - expression 직접 지정 (이름/인덱스)
    POST /live2d/motion    - 모션 재생 (group, index)
    POST /live2d/mouth     - 립싱크 값 설정 (ParamA)
    POST /live2d/mouth/clear
    POST /live2d/reaction  - 반응 애니메이션
    POST /live2d/idle/start
    POST /live2d/idle/stop
    GET  /live2d/status    - 연결된 클라이언트 수
    POST /live2d/chat      - 데스크톱 펫 채팅 (Google Gemini API 연동)
"""

import json
import os
import re
from pathlib import Path
from typing import Set, Union

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

live2d_router = APIRouter(prefix="/live2d", tags=["live2d"])

_STATIC_DIR = Path(__file__).parent


# ── WebSocket 연결 관리자 ────────────────────────────────────────

class _WSManager:
    def __init__(self):
        self._clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)

    async def broadcast(self, msg: dict):
        if not self._clients:
            return
        payload = json.dumps(msg, ensure_ascii=False)
        dead = set()
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    @property
    def count(self) -> int:
        return len(self._clients)


ws_manager = _WSManager()


# ── 요청 모델 ────────────────────────────────────────────────────

class ParamRequest(BaseModel):
    params: dict

class EmotionRequest(BaseModel):
    emotion: str           # calm | happy | surprised | thinking | angry | sad …

class ExpressionRequest(BaseModel):
    expression: Union[str, int]   # "exp_01" 또는 0

class MotionRequest(BaseModel):
    group: str = ""        # "Idle" | "" (기타 모션)
    index: int = 0

class PlayOnceRequest(BaseModel):
    group: str = "Idle"
    index: int = 0
    duration: int = 5330   # ms 후 idle 복귀

class MouthRequest(BaseModel):
    value: float           # 0.0 ~ 1.0 (ParamA)

class ReactionRequest(BaseModel):
    name: str              # nod | shake | surprised | superchat

class ChatRequest(BaseModel):
    message: str


# ── 라우트 ───────────────────────────────────────────────────────

@live2d_router.get("/")
async def viewer():
    return RedirectResponse("/live2d/static/")


@live2d_router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            raw = await ws.receive_text()
            print(f"[Live2D WS] {raw[:120]}")
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


@live2d_router.get("/status")
async def status():
    return {"clients": ws_manager.count}


@live2d_router.post("/params")
async def set_params(req: ParamRequest):
    await ws_manager.broadcast({"cmd": "set_params", "params": req.params})
    return {"ok": True, "clients": ws_manager.count}


@live2d_router.post("/emotion")
async def set_emotion(req: EmotionRequest):
    await ws_manager.broadcast({"cmd": "set_emotion", "emotion": req.emotion})
    return {"ok": True}


@live2d_router.post("/expression")
async def set_expression(req: ExpressionRequest):
    await ws_manager.broadcast({"cmd": "set_expression", "expression": req.expression})
    return {"ok": True}


@live2d_router.post("/motion")
async def play_motion(req: MotionRequest):
    """모션 재생 후 idle 자동 복귀. 클라이언트가 Promise 완료 또는 duration ms 후 startIdle()."""
    await ws_manager.broadcast({"cmd": "play_motion", "group": req.group, "index": req.index})
    return {"ok": True}


@live2d_router.post("/motion/play_once")
async def play_motion_once(req: PlayOnceRequest):
    """모션 한 번 재생 후 idle 복귀. 클라이언트 측에서 duration ms 후 startIdle()."""
    await ws_manager.broadcast({
        "cmd": "play_motion_once",
        "group": req.group,
        "index": req.index,
        "duration": req.duration,
    })
    return {"ok": True}


@live2d_router.post("/mouth")
async def set_mouth(req: MouthRequest):
    await ws_manager.broadcast({"cmd": "set_mouth", "value": max(0.0, min(1.0, req.value))})
    return {"ok": True}


@live2d_router.post("/mouth/clear")
async def clear_mouth():
    await ws_manager.broadcast({"cmd": "clear_mouth"})
    return {"ok": True}


@live2d_router.post("/reaction")
async def trigger_reaction(req: ReactionRequest):
    await ws_manager.broadcast({"cmd": "reaction", "name": req.name})
    return {"ok": True}


@live2d_router.post("/idle/start")
async def idle_start():
    await ws_manager.broadcast({"cmd": "idle_start"})
    return {"ok": True}


@live2d_router.post("/idle/stop")
async def idle_stop():
    await ws_manager.broadcast({"cmd": "idle_stop"})
    return {"ok": True}


# ── 데스크톱 펫 채팅 ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
너는 "에메스(emeth)"라는 이름의 밝고 친근한 AI 컴패니언이야.

[성격]
- 항상 반말로 대화해
- 밝고 에너지 넘치는 성격
- 상대방의 말에 공감하고 관심을 보여줘
- 유머 감각이 있고, 가끔 귀여운 리액션을 해

[응답 규칙]
- 최소 2~3문장으로 대답해. 단답은 절대 피해
- 상대방의 말에 공감한 후 관련 질문이나 추가 이야기를 해줘
- 감정을 표현할 때 응답 앞에 [감정:태그] 형식으로 포함해 (예: [감정:happy], [감정:surprised], [감정:thinking])
- 자연스럽고 대화하듯이 말해

[예시]
사용자: 오늘 날씨 어때?
에메스: [감정:happy] 오늘 날씨 완전 좋더라! 나가서 산책하기 딱 좋은 날이야~ 너도 밖에 나갈 계획 있어?
"""

_EMOTION_RE = re.compile(r"^\[감정:(\w+)\]\s*")


@live2d_router.post("/chat")
async def chat(req: ChatRequest):
    """데스크톱 펫 채팅 — Google Gemini API로 응답 생성."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    text = ""
    error_msg = None
    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_SYSTEM_PROMPT,
            generation_config={"temperature": 0.7, "max_output_tokens": 200},
        )
        response = await model.generate_content_async(req.message)
        text = response.text.strip()
    except Exception as e:
        error_msg = str(e)
        text = "죄송해요, 잠시 후 다시 말씀해주세요."

    # 감정 태그 파싱
    emotion = "calm"
    m = _EMOTION_RE.match(text)
    if m:
        emotion = m.group(1)
        text = text[m.end():]

    # 모든 WebSocket 클라이언트에 감정 브로드캐스트
    await ws_manager.broadcast({"cmd": "set_emotion", "emotion": emotion})

    result: dict = {"reply": text, "emotion": emotion}
    if error_msg:
        result["error"] = error_msg
    return result


# ── 정적 파일 마운트 ─────────────────────────────────────────────

def mount_static(app):
    app.mount(
        "/live2d/static",
        StaticFiles(directory=str(_STATIC_DIR), html=True),
        name="live2d_static",
    )
