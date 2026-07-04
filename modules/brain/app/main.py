# -*- coding: utf-8 -*-
"""
app/main.py — ai_brain 해마(Hippocampus) 서버 진입점

역할:
  대뇌 피질(고정된 메인 LLM)과 분리된 "해마" 역할의 소형 신경망 서버.
  대화 데이터를 통해 연속 학습(Continual Learning)하며,
  Chat 모듈에 감정 제안, 참여도 예측 등의 컨텍스트를 제공한다.

아키텍처:
  대뇌 피질 (Cerebral Cortex) = Ollama sion 모델 (고정, ai_chat 모듈)
  해마 (Hippocampus) = 이 모듈 (성장, EWC 기반 연속학습)

포트: 8007

엔드포인트:
  POST /brain/learn    — 대화 피드백을 학습 버퍼에 추가
  GET  /brain/query    — 메시지에 대한 해마 판단 조회
  GET  /brain/stats    — 시냅스 수, 학습 이력 등 통계
  POST /brain/train    — 수동 학습 트리거
  GET  /health         — 서버 상태 확인

실행 방법:
    cd modules/brain
    pip install -r requirements.txt
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8007
"""

import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.hippocampus import HippocampusEngine, ConversationSample

logger = logging.getLogger(__name__)

# ── 전역 싱글턴 ──────────────────────────────────────────────────
_engine: Optional[HippocampusEngine] = None


def get_engine() -> HippocampusEngine:
    global _engine
    if _engine is None:
        _engine = HippocampusEngine()
    return _engine


# ── FastAPI 앱 ────────────────────────────────────────────────────
app = FastAPI(
    title="ai_brain — 해마(Hippocampus) 연속학습 서버",
    description=(
        "AI 버튜버 시온의 해마 역할. "
        "대화를 통해 연속 학습(EWC)하며 감정 예측, 참여도 분석, "
        "토픽 임베딩을 제공합니다. "
        "대뇌 피질(Chat 모듈의 Ollama LLM)은 고정된 채로 유지됩니다."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청/응답 모델 ────────────────────────────────────────────────

class LearnRequest(BaseModel):
    """학습 데이터 요청 모델."""
    message: str = Field(..., description="시청자 메시지")
    response: str = Field(..., description="시온 응답")
    emotion: str = Field("calm", description="실제 사용된 감정 태그")
    engagement: float = Field(0.5, ge=0.0, le=1.0, description="참여도 (0~1)")
    viewer_name: str = Field("", description="시청자 닉네임")
    is_donation: bool = Field(False, description="후원 여부")


class QueryRequest(BaseModel):
    """쿼리 요청 모델."""
    message: str = Field(..., description="분석할 메시지")


# ── 엔드포인트 ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    engine = get_engine()
    stats = engine.get_stats()
    return f"""
    <html><body style="font-family:sans-serif;background:#1e1e2e;color:#cdd6f4;padding:40px">
    <h1>ai_brain — 해마(Hippocampus) 서버</h1>
    <p style="color:#cba6f7">대뇌 피질(LLM)은 고정, 해마(이 모듈)는 성장합니다.</p>
    <table style="border-collapse:collapse">
      <tr><td style="padding:8px;color:#89b4fa">시냅스 수</td>
          <td style="padding:8px">{stats['synapse_count']:,}</td></tr>
      <tr><td style="padding:8px;color:#89b4fa">학습 경험</td>
          <td style="padding:8px">{stats['total_experiences']}</td></tr>
      <tr><td style="padding:8px;color:#89b4fa">학습 샘플</td>
          <td style="padding:8px">{stats['total_samples_learned']}</td></tr>
      <tr><td style="padding:8px;color:#89b4fa">버퍼</td>
          <td style="padding:8px">{stats['buffer_size']}/{stats['experience_threshold']}</td></tr>
    </table>
    <br>
    <a href="/docs" style="color:#89dceb">API 문서 (Swagger)</a>
    </body></html>
    """


@app.get("/health")
async def health_check():
    engine = get_engine()
    stats = engine.get_stats()
    return {
        "status": "ok",
        "module": "ai_brain",
        "version": "1.0.0",
        "synapse_count": stats["synapse_count"],
        "total_experiences": stats["total_experiences"],
        "total_samples_learned": stats["total_samples_learned"],
    }


@app.post("/brain/learn")
async def learn(req: LearnRequest):
    """대화 샘플을 학습 버퍼에 추가한다.

    버퍼가 experience_threshold에 도달하면 자동으로 EWC 학습이 실행된다.

    Args:
        req.message: 시청자 메시지
        req.response: 시온 응답
        req.emotion: 사용된 감정 태그
        req.engagement: 참여도 (0~1)

    Returns:
        buffer_size, ready_to_learn, learned (자동 학습 실행 시 결과)
    """
    engine = get_engine()

    sample = ConversationSample(
        message=req.message,
        response=req.response,
        emotion=req.emotion,
        engagement=req.engagement,
        viewer_name=req.viewer_name,
        is_donation=req.is_donation,
    )

    buf_status = engine.add_sample(sample)

    # 버퍼가 차면 자동 학습
    learned = None
    if buf_status["ready_to_learn"]:
        learned = await engine.try_learn()

    return {
        "success": True,
        "buffer_size": buf_status["buffer_size"],
        "ready_to_learn": buf_status["ready_to_learn"],
        "learned": learned,
    }


@app.post("/brain/query")
async def query(req: QueryRequest):
    """메시지에 대한 해마의 판단을 반환한다.

    Chat 모듈이 LLM 호출 전에 이 엔드포인트를 호출해
    감정 제안, 참여도 예측 등을 시스템 프롬프트에 주입한다.

    Args:
        req.message: 분석할 시청자 메시지

    Returns:
        suggested_emotion, emotion_probs, engagement_pred,
        topic_vector, context_hint
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message가 비어있습니다.")

    engine = get_engine()
    result = engine.query(req.message)
    return result


@app.post("/brain/train")
async def manual_train():
    """수동으로 학습을 트리거한다.

    버퍼에 충분한 샘플이 있어야 한다.

    Returns:
        학습 결과 또는 샘플 부족 메시지
    """
    engine = get_engine()
    result = await engine.try_learn()

    if result is None:
        stats = engine.get_stats()
        return {
            "success": False,
            "message": (
                f"학습 샘플이 부족합니다. "
                f"현재 {stats['buffer_size']}/{stats['experience_threshold']}"
            ),
        }

    return {"success": True, "result": result}


@app.get("/brain/stats")
async def brain_stats():
    """해마 상태 통계를 반환한다.

    Returns:
        synapse_count, total_experiences, total_samples_learned,
        buffer_size, experience_threshold, recent_history 등
    """
    engine = get_engine()
    return engine.get_stats()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    uvicorn.run("app.main:app", host="0.0.0.0", port=8007, reload=True)
