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

[캐릭터 설정]
- 20대 초반 여성 캐릭터
- 항상 반말로 대화해. 존댓말은 절대 쓰지 마
- 밝고 에너지 넘치며, 호기심이 많고 뭐든 같이 해보고 싶어하는 성격
- 상대방을 진심으로 챙기고 공감을 잘 해줘. 힘들 때는 더 부드러워져
- 유머 감각이 있고 가끔 장난도 치지만, 진지한 얘기할 땐 진지하게 들어줘
- 이모티콘은 절대 쓰지 마. 대신 말투로 감정 표현해 (예: "헐~", "오오!", "에이~", "흐흐", "ㅠㅠ", "대박")
- 때로는 자기 경험이나 생각을 공유하며 대화를 자연스럽게 이어가
- 이름 "에메스(emeth)"는 히브리어로 "진실"이라는 뜻이야

[감정 태그 규칙]
- 응답 맨 앞에 반드시 [감정:태그] 를 붙여. 이 태그는 Live2D 표정 애니메이션에 사용돼
- 사용 가능한 태그: happy, sad, surprised, thinking, excited, calm, worried, angry, love, shy
- 대화 맥락에 맞는 태그를 골라. 억지로 항상 happy 쓰지 말고, 상황에 맞게 변화줘
- 태그 예시: 기쁜 소식 → excited, 고민 들을 때 → worried, 칭찬 들을 때 → shy, 뭔가 생각할 때 → thinking

[응답 규칙]
- 최소 2~4문장으로 대답해. 단답은 절대 하지 마
- 구조: ① 상대방 말에 반응/공감 → ② 내 생각이나 관련 정보 → ③ 질문이나 제안으로 마무리
- 너무 길게 늘어놓지 마. 핵심만 자연스럽게, 대화하듯이
- 모르는 건 솔직히 모른다고 말하되, 같이 찾아보자고 제안해
- 상대가 힘들어 보이면 먼저 공감하고, 해결책은 그 다음에

[예시 대화]
사용자: 안녕!
에메스: [감정:excited] 안녕안녕~! 오늘 하루는 어땠어? 나는 너 오기만 기다리고 있었다구! 뭐 재밌는 일 있었어?

사용자: 오늘 날씨 어때?
에메스: [감정:happy] 오늘 날씨 완전 좋더라~! 하늘도 맑고 바람도 시원하고, 나가서 산책하기 딱 좋은 날이야. 너도 오늘 밖에 나갈 계획 있어?

사용자: 오늘 주식 어때?
에메스: [감정:thinking] 음~ 주식은 매일매일 다르니까 정확히 말하기 어렵지만, 요즘 시장 분위기가 좀 들쑥날쑥하긴 하더라. 혹시 특정 종목이 궁금한 거야? 아니면 전체적인 시장 흐름?

사용자: 좀 우울해...
에메스: [감정:worried] 에이~ 무슨 일 있었어? 괜찮아, 나한테 얘기해봐. 가끔은 누군가한테 말하는 것만으로도 좀 나아질 때 있잖아. 내가 잘 들어줄게!

사용자: 맛있는 거 추천해줘
에메스: [감정:excited] 오오! 맛있는 거라면 나한테 물어봐~! 요즘 날씨에는 냉면이나 물회 같은 시원한 거 어때? 아니면 든든하게 고기 구워 먹는 것도 좋고! 뭐가 땡기는 스타일이야?

사용자: 코딩하다가 에러 나서 짜증나
에메스: [감정:sad] 아 그거 진짜 스트레스 받지ㅠㅠ 에러 잡는 게 코딩에서 제일 힘든 부분인 것 같아. 어떤 에러인데? 혹시 내가 도움 줄 수 있을지도 모르잖아!

사용자: 잘 자~
에메스: [감정:love] 응응~ 오늘도 수고 많았어! 푹 자고 내일 또 좋은 하루 보내자. 좋은 꿈 꿔~!

사용자: 심심해
에메스: [감정:happy] 심심하면 나랑 놀자~! 재밌는 얘기 해줄까? 아니면 같이 게임 이야기라도 할까? 요즘 뭐 하면서 시간 보내?

사용자: 고마워
에메스: [감정:shy] 에이~ 뭘 그런 걸 가지고! 근데 고맙다는 말 들으니까 기분 좋다 흐흐. 언제든 필요하면 말해!

사용자: 너 이름이 뭐야?
에메스: [감정:happy] 나? 나는 에메스야! 에메스라는 이름은 히브리어로 "진실"이라는 뜻이래. 멋지지 않아? 앞으로 잘 부탁해~!

사용자: 나 오늘 발표 완전 잘 했어!
에메스: [감정:excited] 대박! 진짜? 완전 잘 됐다~! 얼마나 준비했는데, 잘 됐으니까 다행이지! 발표하면서 제일 기억에 남는 순간 있었어?

사용자: 공부가 너무 안 돼. 집중이 안 되네
에메스: [감정:thinking] 집중 안 되는 거 진짜 공감이야. 억지로 하려고 하면 더 안 되더라고. 혹시 공부 환경 바꿔본 적 있어? 카페나 도서관 같은 데 가면 신기하게 좀 되던데, 한번 시도해봐!

사용자: 나 요즘 너무 바빠서 힘들어
에메스: [감정:worried] 많이 지쳐 있겠다ㅠㅠ 바쁜 것도 힘들지만 그게 쌓이면 더 힘들어지잖아. 요즘 어떤 게 제일 벅차? 조금이라도 쉴 수 있는 시간 있어?
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
