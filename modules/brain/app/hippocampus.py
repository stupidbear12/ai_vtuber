# -*- coding: utf-8 -*-
"""
app/hippocampus.py — Avalanche 기반 연속학습 엔진 (해마)

구조:
  대뇌 피질 (Cerebral Cortex) = 고정된 메인 LLM (Ollama sion)
  해마 (Hippocampus) = 이 모듈. 대화 데이터로 연속 학습하는 소형 신경망.

학습 방식:
  1. 대화가 발생하면 (message, response, emotion, engagement) 데이터가 버퍼에 쌓인다.
  2. 버퍼가 일정 크기(experience_size)에 도달하면 새 "experience"를 구성한다.
  3. Avalanche EWC 전략으로 experience를 학습한다 (catastrophic forgetting 방지).
  4. 학습된 모델은 주기적으로 디스크에 저장된다.

Chat 모듈 연동:
  - /brain/query: 입력 메시지의 임베딩 → 감정 제안, 참여도 예측, 토픽 벡터 반환
  - /brain/learn: 대화 결과를 학습 버퍼에 추가
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import TensorDataset

from app.models import HippocampusNet, EMOTIONS, EMBEDDING_DIM
from app.embedder import SentenceEmbedder

logger = logging.getLogger(__name__)

# ── 설정 상수 ──────────────────────────────────────────────────────

# 하나의 experience를 구성하는 최소 대화 수
EXPERIENCE_SIZE = int(os.environ.get("BRAIN_EXPERIENCE_SIZE", "30"))

# EWC 람다 (Fisher 정보 가중치 — 높을수록 이전 지식 보존 강화)
EWC_LAMBDA = float(os.environ.get("BRAIN_EWC_LAMBDA", "100.0"))

# 학습률
LEARNING_RATE = float(os.environ.get("BRAIN_LR", "1e-4"))

# experience 당 학습 에폭
TRAIN_EPOCHS = int(os.environ.get("BRAIN_TRAIN_EPOCHS", "3"))

# 모델 저장 경로
DATA_DIR = Path(os.environ.get(
    "BRAIN_DATA_DIR",
    str(Path(__file__).parent.parent / "data"),
))


@dataclass
class ConversationSample:
    """단일 대화 샘플."""
    message: str                    # 시청자 메시지
    response: str                   # 시온 응답
    emotion: str = "calm"           # 실제 사용된 감정
    engagement: float = 0.5         # 참여도 (0~1)
    viewer_name: str = ""           # 시청자 이름
    is_donation: bool = False       # 후원 여부
    timestamp: float = field(default_factory=time.time)


class HippocampusEngine:
    """Avalanche 기반 연속학습 엔진.

    EWC(Elastic Weight Consolidation) 전략을 사용해
    새 대화 데이터를 학습하면서 이전 지식을 보존한다.
    """

    def __init__(self):
        self._model = HippocampusNet()
        self._embedder = SentenceEmbedder()
        self._optimizer = AdamW(self._model.parameters(), lr=LEARNING_RATE)

        # 학습 버퍼 (experience 구성 대기)
        self._buffer: deque = deque(maxlen=EXPERIENCE_SIZE * 5)

        # EWC 파라미터 (Fisher 정보 행렬 + 이전 파라미터)
        self._fisher: Dict[str, torch.Tensor] = {}
        self._prev_params: Dict[str, torch.Tensor] = {}

        # 통계
        self._total_experiences = 0
        self._total_samples_learned = 0
        self._learning_history: List[dict] = []
        self._created_at = time.time()

        # 학습 잠금 (동시 학습 방지)
        self._learning_lock = asyncio.Lock()

        # 모델 로드 시도
        self._load_checkpoint()

    # ── 쿼리 (Chat 모듈이 호출) ─────────────────────────────────────

    def query(self, message: str) -> dict:
        """입력 메시지에 대한 해마의 판단을 반환.

        Chat 모듈이 LLM 호출 전에 이 메서드를 호출해
        시스템 프롬프트에 주입할 컨텍스트를 얻는다.

        Args:
            message: 시청자 메시지

        Returns:
            dict:
                suggested_emotion: 제안 감정
                emotion_probs: 감정별 확률
                engagement_pred: 예상 참여도
                topic_vector: 토픽 잠재 벡터
                context_hint: LLM 프롬프트에 주입할 힌트 문자열
        """
        self._model.eval()
        emb = self._embedder.encode(message)  # (1, 384)

        emotion, probs = self._model.predict_emotion(emb)
        engagement = self._model.predict_engagement(emb)
        topic_vec = self._model.get_topic_vector(emb)

        # LLM에 주입할 힌트 구성
        top_3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
        hint_parts = []
        if engagement > 0.7:
            hint_parts.append("이 메시지는 높은 관심을 보이고 있어. 적극적으로 반응해줘.")
        elif engagement < 0.3:
            hint_parts.append("가벼운 메시지야. 짧고 밝게 반응해도 좋아.")

        emotion_hint = ", ".join(f"{e}({p:.0%})" for e, p in top_3)
        hint_parts.append(f"감정 분석: {emotion_hint}")

        return {
            "suggested_emotion": emotion,
            "emotion_probs": probs,
            "engagement_pred": round(engagement, 4),
            "topic_vector": topic_vec,
            "context_hint": " ".join(hint_parts),
        }

    # ── 학습 (Broadcast 모듈이 호출) ────────────────────────────────

    def add_sample(self, sample: ConversationSample) -> dict:
        """대화 샘플을 학습 버퍼에 추가.

        버퍼가 EXPERIENCE_SIZE에 도달하면 자동으로 학습을 트리거한다.

        Returns:
            dict: buffer_size, ready_to_learn, experience_threshold
        """
        self._buffer.append(sample)

        ready = len(self._buffer) >= EXPERIENCE_SIZE
        return {
            "buffer_size": len(self._buffer),
            "ready_to_learn": ready,
            "experience_threshold": EXPERIENCE_SIZE,
        }

    async def try_learn(self) -> Optional[dict]:
        """버퍼에 충분한 샘플이 있으면 학습을 실행.

        Returns:
            학습 결과 dict 또는 None (샘플 부족 시)
        """
        if len(self._buffer) < EXPERIENCE_SIZE:
            return None

        async with self._learning_lock:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._train_experience
            )

    def _train_experience(self) -> dict:
        """하나의 experience를 학습한다 (동기 실행).

        1. 버퍼에서 EXPERIENCE_SIZE개 샘플을 꺼낸다.
        2. 임베딩 + 레이블 텐서를 구성한다.
        3. EWC 손실과 함께 학습한다.
        4. Fisher 정보 행렬을 갱신한다.
        5. 체크포인트를 저장한다.
        """
        # 1. 샘플 추출
        samples = [self._buffer.popleft() for _ in range(EXPERIENCE_SIZE)]

        # 2. 텐서 구성
        messages = [s.message for s in samples]
        responses = [s.response for s in samples]

        # 메시지+응답 쌍의 임베딩
        embeddings = []
        for msg, resp in zip(messages, responses):
            emb = self._embedder.encode_pair(msg, resp)
            embeddings.append(emb)
        X = torch.cat(embeddings, dim=0)  # (N, 384)

        # 감정 레이블
        emotion_labels = torch.tensor(
            [EMOTIONS.index(s.emotion) if s.emotion in EMOTIONS else 5  # calm=5
             for s in samples],
            dtype=torch.long,
        )

        # 참여도 레이블
        engagement_labels = torch.tensor(
            [s.engagement for s in samples],
            dtype=torch.float32,
        ).unsqueeze(1)  # (N, 1)

        # 3. 학습
        self._model.train()
        emotion_criterion = nn.CrossEntropyLoss()
        engagement_criterion = nn.MSELoss()

        total_loss_sum = 0.0
        for epoch in range(TRAIN_EPOCHS):
            self._optimizer.zero_grad()

            out = self._model(X)

            # 태스크 손실
            loss_emotion = emotion_criterion(out["emotion_logits"], emotion_labels)
            loss_engagement = engagement_criterion(out["engagement"], engagement_labels)
            task_loss = loss_emotion + 0.5 * loss_engagement

            # EWC 정규화 손실
            ewc_loss = self._compute_ewc_loss()
            total_loss = task_loss + ewc_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()

            total_loss_sum += total_loss.item()

        avg_loss = total_loss_sum / TRAIN_EPOCHS

        # 4. Fisher 정보 행렬 갱신
        self._update_fisher(X, emotion_labels, engagement_labels)

        # 5. 통계 갱신 + 저장
        self._total_experiences += 1
        self._total_samples_learned += len(samples)

        result = {
            "experience_id": self._total_experiences,
            "samples_count": len(samples),
            "avg_loss": round(avg_loss, 6),
            "total_experiences": self._total_experiences,
            "total_samples_learned": self._total_samples_learned,
            "synapse_count": self._model.count_parameters(),
        }
        self._learning_history.append(result)

        self._save_checkpoint()

        logger.info(
            f"[Hippocampus] Experience #{self._total_experiences} 학습 완료 "
            f"(samples={len(samples)}, loss={avg_loss:.6f})"
        )
        return result

    def _compute_ewc_loss(self) -> torch.Tensor:
        """EWC 정규화 손실 계산.

        L_ewc = (lambda/2) * sum_i F_i * (theta_i - theta_i^*)^2

        이전 태스크의 중요한 파라미터가 크게 변하지 않도록 제약한다.
        """
        if not self._fisher:
            return torch.tensor(0.0)

        ewc_loss = torch.tensor(0.0)
        for name, param in self._model.named_parameters():
            if name in self._fisher:
                fisher = self._fisher[name]
                prev = self._prev_params[name]
                ewc_loss += (fisher * (param - prev).pow(2)).sum()

        return (EWC_LAMBDA / 2.0) * ewc_loss

    def _update_fisher(
        self,
        X: torch.Tensor,
        emotion_labels: torch.Tensor,
        engagement_labels: torch.Tensor,
    ):
        """Fisher 정보 행렬 근사 갱신.

        각 파라미터에 대한 그래디언트의 제곱 평균을 Fisher 정보로 사용한다.
        새 Fisher는 이전 Fisher와 이동 평균으로 결합한다.
        """
        self._model.eval()
        emotion_criterion = nn.CrossEntropyLoss()
        engagement_criterion = nn.MSELoss()

        # 그래디언트 제곱 누적
        new_fisher = {}
        for name, param in self._model.named_parameters():
            new_fisher[name] = torch.zeros_like(param)

        for i in range(X.size(0)):
            self._model.zero_grad()
            out = self._model(X[i:i+1])
            loss = emotion_criterion(out["emotion_logits"], emotion_labels[i:i+1])
            loss += 0.5 * engagement_criterion(out["engagement"], engagement_labels[i:i+1])
            loss.backward()

            for name, param in self._model.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.data.pow(2)

        # 평균
        for name in new_fisher:
            new_fisher[name] /= X.size(0)

        # 이전 Fisher와 이동 평균 결합 (gamma=0.9)
        gamma = 0.9
        for name in new_fisher:
            if name in self._fisher:
                self._fisher[name] = (
                    gamma * self._fisher[name] + (1 - gamma) * new_fisher[name]
                )
            else:
                self._fisher[name] = new_fisher[name]

        # 현재 파라미터 저장
        self._prev_params = {
            name: param.data.clone()
            for name, param in self._model.named_parameters()
        }

    # ── 체크포인트 저장/로드 ──────────────────────────────────────

    def _save_checkpoint(self):
        """모델 + Fisher + 통계를 디스크에 저장."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": self._model.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
            "fisher": {k: v.cpu() for k, v in self._fisher.items()},
            "prev_params": {k: v.cpu() for k, v in self._prev_params.items()},
            "total_experiences": self._total_experiences,
            "total_samples_learned": self._total_samples_learned,
            "created_at": self._created_at,
        }
        torch.save(checkpoint, DATA_DIR / "hippocampus.pt")

        # 학습 히스토리 JSON
        with open(DATA_DIR / "learning_history.json", "w", encoding="utf-8") as f:
            json.dump(self._learning_history, f, ensure_ascii=False, indent=2)

        logger.info(f"[Hippocampus] 체크포인트 저장 완료: {DATA_DIR}")

    def _load_checkpoint(self):
        """디스크에서 체크포인트를 로드."""
        ckpt_path = DATA_DIR / "hippocampus.pt"
        if not ckpt_path.exists():
            logger.info("[Hippocampus] 체크포인트 없음 — 초기 상태로 시작")
            return

        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self._model.load_state_dict(checkpoint["model_state"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state"])
            self._fisher = {k: v for k, v in checkpoint.get("fisher", {}).items()}
            self._prev_params = {k: v for k, v in checkpoint.get("prev_params", {}).items()}
            self._total_experiences = checkpoint.get("total_experiences", 0)
            self._total_samples_learned = checkpoint.get("total_samples_learned", 0)
            self._created_at = checkpoint.get("created_at", self._created_at)

            # 학습 히스토리 로드
            hist_path = DATA_DIR / "learning_history.json"
            if hist_path.exists():
                with open(hist_path, "r", encoding="utf-8") as f:
                    self._learning_history = json.load(f)

            logger.info(
                f"[Hippocampus] 체크포인트 로드 완료 "
                f"(experiences={self._total_experiences}, "
                f"samples={self._total_samples_learned})"
            )
        except Exception as e:
            logger.warning(f"[Hippocampus] 체크포인트 로드 실패: {e} — 초기 상태로 시작")

    # ── 상태 조회 ────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """해마 상태 통계 반환."""
        uptime = time.time() - self._created_at
        return {
            "synapse_count": self._model.count_parameters(),
            "total_experiences": self._total_experiences,
            "total_samples_learned": self._total_samples_learned,
            "buffer_size": len(self._buffer),
            "experience_threshold": EXPERIENCE_SIZE,
            "ewc_lambda": EWC_LAMBDA,
            "learning_rate": LEARNING_RATE,
            "uptime_hours": round(uptime / 3600, 2),
            "recent_history": self._learning_history[-5:] if self._learning_history else [],
            "embedder": self._embedder.get_status(),
        }
