# -*- coding: utf-8 -*-
"""
app/models.py — 해마 소형 신경망 모델 정의

대뇌 피질(메인 LLM)은 고정된 채로 유지하고,
해마(이 모듈)의 소형 모델이 대화를 통해 연속 학습한다.

모델 구조:
  - 입력: sentence-transformers 임베딩 (384차원)
  - HippocampusNet: 다중 헤드 MLP
    - emotion_head: 감정 예측 (10 클래스)
    - engagement_head: 참여도 예측 (회귀)
    - topic_head: 토픽 임베딩 (64차원 잠재 공간)
"""

import torch
import torch.nn as nn


# 감정 클래스 (Chat 모듈과 동일)
EMOTIONS = [
    "happy", "sad", "surprised", "thinking", "excited",
    "calm", "worried", "angry", "love", "shy",
]
NUM_EMOTIONS = len(EMOTIONS)

# sentence-transformers 임베딩 차원 (all-MiniLM-L6-v2)
EMBEDDING_DIM = 384

# 내부 은닉층 차원
HIDDEN_DIM = 128

# 토픽 잠재 공간 차원
TOPIC_DIM = 64


class HippocampusNet(nn.Module):
    """해마 역할의 소형 다중 헤드 신경망.

    하나의 공유 인코더(shared encoder)에서 특성을 추출하고,
    세 개의 태스크 헤드가 각각 다른 역할을 수행한다:

    1. emotion_head: 대화 맥락 → 최적 감정 응답 예측
    2. engagement_head: 대화 → 시청자 참여도(반응 정도) 예측
    3. topic_head: 대화 → 토픽 잠재 벡터 (유사 대화 검색용)
    """

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = HIDDEN_DIM,
        num_emotions: int = NUM_EMOTIONS,
        topic_dim: int = TOPIC_DIM,
    ):
        super().__init__()

        # 공유 인코더: 임베딩 → 은닉 표현
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 감정 예측 헤드
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        # 참여도 예측 헤드 (0~1 스칼라)
        self.engagement_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 토픽 임베딩 헤드 (잠재 공간으로 매핑)
        self.topic_head = nn.Sequential(
            nn.Linear(hidden_dim, topic_dim),
            nn.LayerNorm(topic_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """가중치 초기화 — Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> dict:
        """순전파.

        Args:
            x: (batch, input_dim) 문장 임베딩 텐서

        Returns:
            dict with keys:
                emotion_logits: (batch, num_emotions)
                engagement: (batch, 1)
                topic_embedding: (batch, topic_dim)
        """
        h = self.shared_encoder(x)

        return {
            "emotion_logits": self.emotion_head(h),
            "engagement": self.engagement_head(h),
            "topic_embedding": self.topic_head(h),
        }

    def predict_emotion(self, x: torch.Tensor) -> tuple:
        """감정 예측 — softmax 확률과 최고 감정을 반환.

        Returns:
            (predicted_emotion: str, probabilities: dict)
        """
        with torch.no_grad():
            out = self.forward(x)
            probs = torch.softmax(out["emotion_logits"], dim=-1)
            idx = probs.argmax(dim=-1).item()
            prob_dict = {
                EMOTIONS[i]: round(probs[0, i].item(), 4)
                for i in range(len(EMOTIONS))
            }
            return EMOTIONS[idx], prob_dict

    def predict_engagement(self, x: torch.Tensor) -> float:
        """참여도 예측 — 0~1 스칼라."""
        with torch.no_grad():
            out = self.forward(x)
            return out["engagement"].item()

    def get_topic_vector(self, x: torch.Tensor) -> list:
        """토픽 잠재 벡터 반환."""
        with torch.no_grad():
            out = self.forward(x)
            return out["topic_embedding"].squeeze(0).tolist()

    def count_parameters(self) -> int:
        """학습 가능한 파라미터 수 (시냅스 수)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
