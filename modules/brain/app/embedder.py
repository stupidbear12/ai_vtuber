# -*- coding: utf-8 -*-
"""
app/embedder.py — 문장 임베딩 래퍼

sentence-transformers의 all-MiniLM-L6-v2 모델을 사용해
텍스트를 384차원 벡터로 변환한다.

이 임베딩 모델은 고정(frozen)이며 학습되지 않는다.
해마(HippocampusNet)의 입력으로만 사용된다.
"""

import logging
from typing import List, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)

# 모델 이름 (가볍고 한국어도 어느 정도 지원)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEmbedder:
    """문장 임베딩 생성기.

    sentence-transformers 라이브러리를 사용하며,
    CPU에서 동작한다 (소형 모델이라 GPU 불필요).
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model = None
        self._model_name = model_name
        self._dim = 384  # all-MiniLM-L6-v2 출력 차원

    def _load_model(self):
        """지연 로딩 — 첫 호출 시 모델을 로드한다."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                f"[Embedder] 모델 로드 완료: {self._model_name} (dim={self._dim})"
            )
        except ImportError:
            logger.warning(
                "[Embedder] sentence-transformers 미설치 — 랜덤 임베딩 폴백 사용"
            )
            self._model = "fallback"

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """텍스트를 임베딩 벡터로 변환.

        Args:
            texts: 단일 문자열 또는 문자열 리스트

        Returns:
            (N, dim) 형태의 float32 텐서
        """
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]

        if self._model == "fallback":
            # sentence-transformers 없을 때 랜덤 벡터 반환 (개발용)
            vecs = np.random.randn(len(texts), self._dim).astype(np.float32)
            # 정규화
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-8)
            return torch.from_numpy(vecs)

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return torch.from_numpy(embeddings.astype(np.float32))

    def encode_pair(self, text_a: str, text_b: str) -> torch.Tensor:
        """두 텍스트의 임베딩을 연결(concatenate)해 반환.

        (질문, 응답) 쌍의 결합 표현을 만든다.
        결과는 (1, dim) — 두 벡터의 평균.

        Args:
            text_a: 첫 번째 텍스트 (예: 시청자 메시지)
            text_b: 두 번째 텍스트 (예: 시온 응답)

        Returns:
            (1, dim) 텐서 (두 임베딩의 평균)
        """
        emb_a = self.encode(text_a)  # (1, dim)
        emb_b = self.encode(text_b)  # (1, dim)
        # 평균 풀링
        combined = (emb_a + emb_b) / 2.0
        # 재정규화
        combined = combined / (combined.norm(dim=-1, keepdim=True) + 1e-8)
        return combined

    def get_status(self) -> dict:
        """현재 상태 반환."""
        return {
            "model": self._model_name,
            "dim": self._dim,
            "loaded": self._model is not None and self._model != "fallback",
            "fallback": self._model == "fallback",
        }
