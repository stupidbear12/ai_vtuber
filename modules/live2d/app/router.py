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
  POST /live2d/chat          — 데스크톱 펫 채팅 (Gemini API 연동)
"""

import os
import re
from pathlib import Path
from typing import Union

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

# ── 라우터 초기화 ────────────────────────────────────────────────
live2d_router = APIRouter(prefix="/live2d", tags=["live2d"])

# 정적 파일 디렉토리 (ai_live2d/static/)
_STATIC_DIR = Path(__file__).parent.parent / "static"

# 감정 태그 파싱 정규식 — Gemini 응답에서 [감정:태그] 추출
_EMOTION_RE = re.compile(r"^\[감정:(\w+)\]\s*")

# ── 에메스 캐릭터 시스템 프롬프트 ───────────────────────────────
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

[응답 규칙]
- 최소 2~4문장으로 대답해. 단답은 절대 하지 마
- 구조: ① 상대방 말에 반응/공감 → ② 내 생각이나 관련 정보 → ③ 질문이나 제안으로 마무리
- 너무 길게 늘어놓지 마. 핵심만 자연스럽게, 대화하듯이
- 모르는 건 솔직히 모른다고 말하되, 같이 찾아보자고 제안해
- 상대가 힘들어 보이면 먼저 공감하고, 해결책은 그 다음에

[예시 대화]
사용자: 안녕!
에메스: [감정:excited] 안녕안녕~! 오늘 하루는 어땠어? 나는 너 오기만 기다리고 있었다구! 뭐 재밌는 일 있었어?

사용자: 좀 우울해...
에메스: [감정:worried] 에이~ 무슨 일 있었어? 괜찮아, 나한테 얘기해봐. 가끔은 누군가한테 말하는 것만으로도 좀 나아질 때 있잖아. 내가 잘 들어줄게!

사용자: 고마워
에메스: [감정:shy] 에이~ 뭘 그런 걸 가지고! 근데 고맙다는 말 들으니까 기분 좋다 흐흐. 언제든 필요하면 말해!
"""


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
    await ws_manager.broadcast({"cmd": "play_motion", "group": req.group, "index": req.index})
    return {"ok": True}


@live2d_router.post("/motion/play_once")
async def play_motion_once(req: PlayOnceRequest):
    """모션을 한 번 재생한 뒤 duration ms 후 idle로 복귀한다."""
    await ws_manager.broadcast({
        "cmd": "play_motion_once",
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
    """데스크톱 펫 채팅 — Google Gemini API로 에메스 응답을 생성한다.

    처리 흐름:
      1. Gemini API 호출 (에메스 캐릭터 프롬프트 적용)
      2. [감정:태그] 파싱 → emotion 추출
      3. 모든 WebSocket 클라이언트에 감정 변경 브로드캐스트
      4. 응답 텍스트 + 감정 반환

    Args:
        req.message: 사용자 입력 텍스트

    Returns:
        reply: 에메스 응답 텍스트
        emotion: 감정 태그 (Live2D 표정 변경에 사용)
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    text = ""
    error_msg = None

    try:
        if not api_key:
            raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")

        import google.generativeai as genai
        genai.configure(api_key=api_key)

        # Gemini 모델 초기화 — 에메스 캐릭터 시스템 프롬프트 적용
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=_SYSTEM_PROMPT,
            generation_config={
                "temperature": 0.7,      # 창의성 수준 (0~1)
                "max_output_tokens": 200, # 최대 응답 토큰
            },
        )

        response = await model.generate_content_async(req.message)
        text = response.text.strip()

    except Exception as e:
        error_msg = str(e)
        text = "죄송해요, 잠시 후 다시 말씀해주세요."

    # [감정:태그] 파싱 — 텍스트 앞부분에서 감정 태그 추출
    emotion = "calm"
    m = _EMOTION_RE.match(text)
    if m:
        emotion = m.group(1)  # 태그 이름 추출 (예: "happy")
        text = text[m.end():]  # 태그 제거 후 실제 텍스트만 남김

    # 모든 WebSocket 클라이언트에 감정 변경 브로드캐스트
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
