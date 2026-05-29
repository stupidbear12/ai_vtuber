"""
lipsync.py - 오디오 파일 분석 -> 립싱크 값 생성 모듈

사용 라이브러리:
  1순위: librosa (정밀한 RMS 분석)
  2순위: pydub  (librosa 없을 때 fallback)

반환 형식:
  analyze_audio(audio_path) -> List[Tuple[int, float]]
  [(timestamp_ms, mouth_value), ...]
  - timestamp_ms : 밀리초 단위 타임스탬프
  - mouth_value  : 입 벌림 값 (0.0 ~ 1.0)
"""

import os
import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 내부 유틸 함수
# ---------------------------------------------------------------------------

def _normalize(values: list) -> list:
    """리스트를 0.0~1.0 으로 정규화"""
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx - mn < 1e-9:
        return [0.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def _smooth(values: list, alpha: float = 0.3) -> list:
    """
    지수 이동 평균(EMA)으로 값 스무딩
    alpha: 0 = 변화 없음, 1 = 스무딩 없음
    """
    if not values:
        return values
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


# ---------------------------------------------------------------------------
# librosa 기반 분석 (1순위)
# ---------------------------------------------------------------------------

def _analyze_with_librosa(audio_path: str, hop_ms: int = 30) -> List[Tuple[int, float]]:
    """
    librosa 로 RMS 에너지 분석 -> 립싱크 값 생성

    Parameters
    ----------
    audio_path : str   오디오 파일 경로 (mp3 / wav)
    hop_ms     : int   분석 간격 (밀리초), 기본 30ms

    Returns
    -------
    List[(timestamp_ms, mouth_value)]
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(audio_path, sr=None, mono=True)

    hop_length = int(sr * hop_ms / 1000)
    frame_length = hop_length * 2

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    # dB 범위: 보통 -80 ~ 0, 0에 가까울수록 큼
    # 입 벌림에 적합한 범위로 클리핑 (-40dB ~ 0dB)
    rms_clipped = np.clip(rms_db, -40.0, 0.0)

    normalized = (rms_clipped - (-40.0)) / 40.0   # 0.0~1.0
    smoothed   = _smooth(normalized.tolist(), alpha=0.4)

    result = []
    for i, val in enumerate(smoothed):
        ts_ms = int(i * hop_ms)
        result.append((ts_ms, round(float(val), 4)))

    return result


# ---------------------------------------------------------------------------
# pydub 기반 분석 (2순위 fallback)
# ---------------------------------------------------------------------------

def _analyze_with_pydub(audio_path: str, hop_ms: int = 30) -> List[Tuple[int, float]]:
    """
    pydub 로 청크별 dBFS 분석 -> 립싱크 값 생성

    Parameters
    ----------
    audio_path : str   오디오 파일 경로 (mp3 / wav)
    hop_ms     : int   분석 간격 (밀리초), 기본 30ms

    Returns
    -------
    List[(timestamp_ms, mouth_value)]
    """
    from pydub import AudioSegment

    ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
    if ext == "mp3":
        audio = AudioSegment.from_mp3(audio_path)
    elif ext == "wav":
        audio = AudioSegment.from_wav(audio_path)
    else:
        audio = AudioSegment.from_file(audio_path)

    audio = audio.set_channels(1)   # 모노 변환

    dbfs_values = []
    timestamps  = []

    for start_ms in range(0, len(audio), hop_ms):
        chunk = audio[start_ms : start_ms + hop_ms]
        if len(chunk) == 0:
            break
        # dBFS: -inf ~ 0, 무음 시 -inf
        dbfs = chunk.dBFS
        if math.isinf(dbfs):
            dbfs = -80.0
        dbfs = max(dbfs, -80.0)
        dbfs_values.append(dbfs)
        timestamps.append(start_ms)

    # -80 ~ 0 범위를 0.0~1.0 으로 변환
    normalized = [(v - (-80.0)) / 80.0 for v in dbfs_values]
    smoothed   = _smooth(normalized, alpha=0.4)

    return [(ts, round(v, 4)) for ts, v in zip(timestamps, smoothed)]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def analyze_audio(
    audio_path: str,
    hop_ms: int = 30,
) -> List[Tuple[int, float]]:
    """
    오디오 파일을 분석해 시간별 입 벌림 값을 반환합니다.

    Parameters
    ----------
    audio_path : str   mp3 / wav 파일 경로
    hop_ms     : int   분석 간격(밀리초), 기본 30ms (약 33fps)

    Returns
    -------
    List[Tuple[int, float]]
        [(timestamp_ms, mouth_value), ...]
        timestamp_ms : 밀리초
        mouth_value  : 0.0 (입 닫힘) ~ 1.0 (최대 벌림)

    Raises
    ------
    FileNotFoundError : 파일이 없을 때
    RuntimeError      : librosa / pydub 모두 없을 때
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

    # librosa 우선 시도
    try:
        import librosa  # noqa: F401
        return _analyze_with_librosa(audio_path, hop_ms)
    except ImportError:
        pass

    # pydub fallback
    try:
        import pydub  # noqa: F401
        print("[lipsync] librosa 없음, pydub fallback 사용")
        return _analyze_with_pydub(audio_path, hop_ms)
    except ImportError:
        pass

    raise RuntimeError(
        "립싱크 분석에 필요한 라이브러리가 없습니다.\n"
        "다음 중 하나를 설치하세요:\n"
        "  pip install librosa\n"
        "  pip install pydub"
    )


# ---------------------------------------------------------------------------
# 빠른 테스트 (직접 실행 시)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python lipsync.py <오디오파일>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"분석 중: {path}")
    frames = analyze_audio(path)
    print(f"총 프레임 수: {len(frames)}")
    print("처음 10 프레임:")
    for ts, val in frames[:10]:
        bar = "#" * int(val * 30)
        print(f"  {ts:6d}ms  {val:.4f}  {bar}")
