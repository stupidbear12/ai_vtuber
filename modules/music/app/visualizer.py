# -*- coding: utf-8 -*-
"""
Visualizer — 오디오 비주얼라이저 데이터 생성

역할:
  - FFT 스펙트럼 분석 → 프론트엔드 비주얼라이저용 데이터
  - 비트 감지 (킥/스네어/하이햇)
  - Live2D DJ 모션 트리거 이벤트 생성
  - WebSocket으로 실시간 비주얼 데이터 전송

의존성:
  pip install numpy scipy
"""

import asyncio
import logging
import numpy as np
from typing import Optional, List, Set, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpectrumData:
    """FFT 스펙트럼 데이터."""
    frequencies: np.ndarray     # Hz 값 배열
    magnitudes: np.ndarray      # 크기 배열 (dB)
    bands: dict                 # {"bass": float, "mid": float, "high": float}
    peak_frequency: float       # 최대 크기 주파수


@dataclass
class BeatEvent:
    """비트 감지 이벤트."""
    timestamp: float            # 초
    beat_type: str              # "kick" | "snare" | "hihat"
    intensity: float            # 0.0~1.0
    bpm_estimate: float         # 현재 추정 BPM


class Visualizer:
    """실시간 오디오 비주얼라이저.

    사용 패턴:
        viz = Visualizer(sample_rate=44100)
        viz.on_beat(callback)  # 비트 감지 콜백 등록
        spectrum = viz.analyze_chunk(audio_chunk)
        viz_data = viz.get_visualization_data()
    """

    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048):
        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._beat_callbacks: List[Callable] = []
        self._last_spectrum: Optional[SpectrumData] = None
        self._beat_history: List[BeatEvent] = []

    # ── FFT 분석 ──────────────────────────────────────────────────

    def analyze_chunk(self, audio_data: np.ndarray) -> SpectrumData:
        """오디오 청크의 FFT 스펙트럼 분석.

        Args:
            audio_data: shape (samples,) 또는 (samples, channels), float32

        Returns:
            SpectrumData

        TODO:
          1. 스테레오면 모노로 다운믹스
          2. 윈도우 함수 적용 (hann)
          3. np.fft.rfft() 실행
          4. 크기 스펙트럼 계산 (dB 변환)
          5. 주파수 밴드별 에너지 계산
             - bass: 20~250 Hz
             - mid: 250~4000 Hz
             - high: 4000~20000 Hz
          6. SpectrumData 반환
        """
        raise NotImplementedError

    # ── 비트 감지 ─────────────────────────────────────────────────

    def detect_beat(self, audio_data: np.ndarray, timestamp: float) -> Optional[BeatEvent]:
        """비트(킥/스네어) 감지.

        Args:
            audio_data: 오디오 청크
            timestamp: 현재 재생 시간 (초)

        Returns:
            BeatEvent 또는 None

        TODO:
          1. 저역(bass) 에너지 급증 감지 → kick
          2. 중역(mid) 에너지 급증 감지 → snare
          3. 고역(high) 에너지 급증 감지 → hihat
          4. 이전 에너지 대비 threshold 초과 시 이벤트 생성
          5. 등록된 콜백 호출
        """
        raise NotImplementedError

    def on_beat(self, callback: Callable[[BeatEvent], None]) -> None:
        """비트 감지 콜백 등록.

        콜백은 BeatEvent를 인자로 받음.
        Live2D DJ 모션 트리거에 사용.
        """
        self._beat_callbacks.append(callback)

    # ── 비주얼 데이터 ─────────────────────────────────────────────

    def get_visualization_data(self) -> dict:
        """프론트엔드 비주얼라이저용 데이터 패키지.

        Returns:
            {
                "spectrum": {"bass": float, "mid": float, "high": float},
                "peak_freq": float,
                "beat_detected": bool,
                "last_beat_type": str,
                "bpm": float,
            }

        TODO:
          1. _last_spectrum에서 밴드 에너지 추출
          2. _beat_history에서 최근 비트 정보
          3. dict로 패키징
        """
        raise NotImplementedError

    def get_live2d_motion_trigger(self) -> Optional[str]:
        """Live2D DJ 모션 트리거 이벤트 반환.

        Returns:
            모션 이름 (예: "dj_headbang", "dj_arm_wave", "dj_nod")
            또는 None (트리거 없음)

        TODO:
          1. 최근 비트 강도에 따라 모션 결정
             - kick (강) → "dj_headbang"
             - snare → "dj_arm_wave"
             - 일반 비트 → "dj_nod"
          2. 너무 빈번하지 않도록 쿨다운 적용
        """
        raise NotImplementedError
