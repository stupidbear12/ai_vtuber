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
        self._prev_energies: Optional[dict] = None   # EMA 에너지 (비트 감지용)
        self._last_motion_time: float = 0.0          # 모션 쿨다운 추적

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
        # 1. 스테레오 → 모노 다운믹스
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # fft_size에 맞게 자르거나 패딩
        chunk = audio_data[: self._fft_size].astype(np.float32)
        if len(chunk) < self._fft_size:
            chunk = np.pad(chunk, (0, self._fft_size - len(chunk)))

        # 2. Hann 윈도우 적용
        window = np.hanning(self._fft_size)
        windowed = chunk * window

        # 3. FFT
        fft_result = np.fft.rfft(windowed)

        # 4. 크기 스펙트럼 (선형) → dB
        magnitudes_linear = np.abs(fft_result)
        magnitudes_db = 20.0 * np.log10(magnitudes_linear + 1e-10)
        frequencies = np.fft.rfftfreq(self._fft_size, d=1.0 / self._sample_rate)

        # 5. 밴드별 평균 선형 에너지
        def _band_energy(f_min: float, f_max: float) -> float:
            mask = (frequencies >= f_min) & (frequencies < f_max)
            return float(magnitudes_linear[mask].mean()) if mask.any() else 0.0

        bands = {
            "bass": _band_energy(20, 250),
            "mid": _band_energy(250, 4000),
            "high": _band_energy(4000, 20000),
        }

        peak_frequency = float(frequencies[np.argmax(magnitudes_linear)])

        spectrum = SpectrumData(
            frequencies=frequencies,
            magnitudes=magnitudes_db,
            bands=bands,
            peak_frequency=peak_frequency,
        )
        self._last_spectrum = spectrum
        return spectrum

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
        spectrum = self.analyze_chunk(audio_data)
        bands = spectrum.bands

        # 첫 호출 — EMA 기준값 초기화 후 비트 없음으로 반환
        if self._prev_energies is None:
            self._prev_energies = dict(bands)
            return None

        # 에너지 급증 비율 계산
        THRESHOLD = 1.5   # EMA 대비 1.5배 초과 → 비트
        ALPHA = 0.3       # EMA 스무딩 계수

        ratios = {
            band: bands[band] / (self._prev_energies[band] + 1e-10)
            for band in ("bass", "mid", "high")
        }

        # 가장 급증한 밴드를 비트 유형으로 결정
        beat_type: Optional[str] = None
        intensity = 0.0
        if ratios["bass"] > THRESHOLD and ratios["bass"] >= max(ratios["mid"], ratios["high"]):
            beat_type = "kick"
            intensity = min(1.0, (ratios["bass"] - 1.0) / 2.0)
        elif ratios["mid"] > THRESHOLD and ratios["mid"] >= ratios["high"]:
            beat_type = "snare"
            intensity = min(1.0, (ratios["mid"] - 1.0) / 2.0)
        elif ratios["high"] > THRESHOLD:
            beat_type = "hihat"
            intensity = min(1.0, (ratios["high"] - 1.0) / 2.0)

        # EMA 업데이트
        for band in ("bass", "mid", "high"):
            self._prev_energies[band] = (
                ALPHA * bands[band] + (1 - ALPHA) * self._prev_energies[band]
            )

        if beat_type is None:
            return None

        # 최근 비트 간격으로 BPM 추정
        bpm_estimate = 0.0
        if len(self._beat_history) >= 2:
            recent = self._beat_history[-4:]
            intervals = [
                recent[i + 1].timestamp - recent[i].timestamp
                for i in range(len(recent) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            if avg_interval > 0:
                bpm_estimate = 60.0 / avg_interval

        event = BeatEvent(
            timestamp=timestamp,
            beat_type=beat_type,
            intensity=intensity,
            bpm_estimate=bpm_estimate,
        )
        self._beat_history.append(event)

        for cb in self._beat_callbacks:
            try:
                cb(event)
            except Exception:
                logger.exception("Beat callback error")

        return event

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
        if self._last_spectrum is None:
            return {
                "spectrum": {"bass": 0.0, "mid": 0.0, "high": 0.0},
                "peak_freq": 0.0,
                "beat_detected": False,
                "last_beat_type": "",
                "bpm": 0.0,
            }

        last_beat = self._beat_history[-1] if self._beat_history else None
        return {
            "spectrum": self._last_spectrum.bands,
            "peak_freq": self._last_spectrum.peak_frequency,
            "beat_detected": last_beat is not None,
            "last_beat_type": last_beat.beat_type if last_beat else "",
            "bpm": last_beat.bpm_estimate if last_beat else 0.0,
        }

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
        if not self._beat_history:
            return None

        import time
        now = time.monotonic()
        COOLDOWN = 0.5  # seconds

        if now - self._last_motion_time < COOLDOWN:
            return None

        last_beat = self._beat_history[-1]
        if last_beat.beat_type == "kick" and last_beat.intensity > 0.6:
            motion = "dj_headbang"
        elif last_beat.beat_type == "snare":
            motion = "dj_arm_wave"
        else:
            motion = "dj_nod"

        self._last_motion_time = now
        return motion
