"""
data/preprocess.py
==================
오디오 파일 → 멜 스펙트로그램 변환 파이프라인

변환 스펙:
  - 샘플레이트: 22050 Hz (mono)
  - 멜 빈 수: 128
  - n_fft: 1024
  - hop_length: 256
  - 청크 길이: 256 프레임 (~2.97초)
  - 출력 shape: (128, 256) float32, 범위 [-1, 1]

사용법:
    python data/preprocess.py --input data/raw --output data/processed
    python data/preprocess.py --input data/raw --output data/processed --workers 4
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 오디오 / 스펙트로그램 하이퍼파라미터
# ──────────────────────────────────────────────
SAMPLE_RATE: int = 22050
N_MELS: int = 128
N_FFT: int = 1024
HOP_LENGTH: int = 256
CHUNK_FRAMES: int = 256          # 한 샘플의 시간 축 길이 (프레임)
TOP_DB: float = 80.0             # log-mel 클리핑 기준 (dB)
EPS: float = 1e-6                # 로그 연산 안정화 상수

# CHUNK_FRAMES × HOP_LENGTH / SAMPLE_RATE ≈ 2.97 초
CHUNK_SECONDS: float = CHUNK_FRAMES * HOP_LENGTH / SAMPLE_RATE

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}


# ──────────────────────────────────────────────
# 핵심 변환 함수
# ──────────────────────────────────────────────

def load_audio(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    오디오 파일 로드 → mono float32 배열 반환.

    librosa.load 는 내부적으로 ffmpeg/soundfile 을 사용하므로
    MP3, FLAC, OGG 등 다양한 포맷을 지원합니다.
    """
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def audio_to_melspectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    top_db: float = TOP_DB,
) -> np.ndarray:
    """
    오디오 배열 → log-mel 스펙트로그램 (dB 스케일).

    반환 shape: (n_mels, T)
    반환 dtype: float32
    """
    # 1) 멜 파워 스펙트로그램
    mel_power = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        window="hann",
        center=True,
        pad_mode="reflect",
    )

    # 2) dB 변환 (log-mel)
    mel_db = librosa.power_to_db(mel_power, ref=np.max, top_db=top_db)

    return mel_db.astype(np.float32)


def normalize(spec: np.ndarray, min_db: float = -TOP_DB) -> np.ndarray:
    """
    dB 범위 [min_db, 0] → [-1, 1] 선형 정규화.

    min_db: librosa top_db 클리핑 하한과 일치시킴.
    """
    spec = np.clip(spec, min_db, 0.0)
    spec = (spec - min_db) / (-min_db)   # [0, 1]
    spec = spec * 2.0 - 1.0             # [-1, 1]
    return spec.astype(np.float32)


def denormalize(spec: np.ndarray, min_db: float = -TOP_DB) -> np.ndarray:
    """normalize 의 역변환 — 모델 출력 → dB 스펙트로그램."""
    spec = (spec + 1.0) / 2.0           # [-1,1] → [0,1]
    spec = spec * (-min_db) + min_db    # [0,1] → [min_db, 0]
    return spec.astype(np.float32)


def slice_spectrogram(
    spec: np.ndarray,
    chunk_frames: int = CHUNK_FRAMES,
    overlap: float = 0.5,
) -> list[np.ndarray]:
    """
    (n_mels, T) 스펙트로그램을 고정 길이 청크로 분할.

    overlap: 0.0 (겹침 없음) ~ 0.9 (90% 겹침)
    반환: list of (n_mels, chunk_frames) 배열
    """
    if spec.shape[1] < chunk_frames:
        return []

    step = max(1, int(chunk_frames * (1.0 - overlap)))
    chunks = []
    start = 0
    while start + chunk_frames <= spec.shape[1]:
        chunk = spec[:, start : start + chunk_frames]
        chunks.append(chunk.copy())
        start += step

    return chunks


# ──────────────────────────────────────────────
# 단일 파일 처리
# ──────────────────────────────────────────────

def process_file(
    audio_path: Path,
    output_dir: Path,
    chunk_frames: int = CHUNK_FRAMES,
    overlap: float = 0.5,
) -> int:
    """
    오디오 파일 1개를 처리해 복수의 .npy 청크로 저장.

    반환: 저장된 청크 수
    """
    try:
        audio = load_audio(audio_path)
    except Exception as e:
        log.warning(f"로드 실패: {audio_path.name} — {e}")
        return 0

    if len(audio) < N_FFT:
        log.warning(f"너무 짧은 파일 스킵: {audio_path.name}")
        return 0

    mel = audio_to_melspectrogram(audio)
    mel = normalize(mel)
    chunks = slice_spectrogram(mel, chunk_frames, overlap)

    if not chunks:
        log.warning(f"청크 없음 스킵: {audio_path.name}")
        return 0

    stem = audio_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for i, chunk in enumerate(chunks):
        assert chunk.shape == (N_MELS, chunk_frames), (
            f"shape 불일치: {chunk.shape}"
        )
        out_path = output_dir / f"{stem}_chunk{i:04d}.npy"
        np.save(out_path, chunk)
        saved += 1

    log.debug(f"{audio_path.name} → {saved}개 청크")
    return saved


def _worker(args: tuple) -> int:
    """멀티프로세싱 워커 함수."""
    audio_path, output_dir, chunk_frames, overlap = args
    return process_file(
        Path(audio_path), Path(output_dir), chunk_frames, overlap
    )


# ──────────────────────────────────────────────
# 배치 처리
# ──────────────────────────────────────────────

def process_directory(
    input_dir: Path,
    output_dir: Path,
    chunk_frames: int = CHUNK_FRAMES,
    overlap: float = 0.5,
    workers: int = 1,
    recursive: bool = True,
) -> dict:
    """
    디렉토리 내 모든 오디오 파일을 일괄 처리.

    반환: {"files": int, "chunks": int, "failed": int}
    """
    pattern = "**/*" if recursive else "*"
    audio_files = [
        p for p in input_dir.glob(pattern)
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]

    if not audio_files:
        log.warning(f"처리할 오디오 파일 없음: {input_dir}")
        return {"files": 0, "chunks": 0, "failed": 0}

    log.info(f"총 {len(audio_files)}개 오디오 파일 발견")

    task_args = [
        (str(p), str(output_dir), chunk_frames, overlap)
        for p in audio_files
    ]

    total_chunks = 0
    failed = 0

    if workers > 1:
        log.info(f"멀티프로세싱 모드: {workers}개 워커")
        with mp.Pool(processes=workers) as pool:
            results = pool.map(_worker, task_args)
        for r in results:
            if r == 0:
                failed += 1
            else:
                total_chunks += r
    else:
        for i, args in enumerate(task_args, 1):
            n = _worker(args)
            if n == 0:
                failed += 1
            else:
                total_chunks += n
            if i % 10 == 0:
                log.info(f"진행: {i}/{len(audio_files)} 파일 처리됨")

    stats = {
        "files": len(audio_files),
        "chunks": total_chunks,
        "failed": failed,
    }
    log.info(
        f"완료 — 파일: {stats['files']}, "
        f"청크: {stats['chunks']}, "
        f"실패: {stats['failed']}"
    )
    return stats


# ──────────────────────────────────────────────
# CLI 진입점
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="오디오 → 멜 스펙트로그램 전처리 파이프라인"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="입력 오디오 디렉토리 (기본: data/raw)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="출력 .npy 디렉토리 (기본: data/processed)",
    )
    parser.add_argument(
        "--chunk_frames",
        type=int,
        default=CHUNK_FRAMES,
        help=f"청크 프레임 수 (기본: {CHUNK_FRAMES})",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="청크 간 겹침 비율 0.0~0.9 (기본: 0.5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="병렬 처리 워커 수 (기본: 1, CPU 코어 수 이하 권장)",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="서브디렉토리 탐색 비활성화",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 로그 활성화",
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    process_directory(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        chunk_frames=args.chunk_frames,
        overlap=args.overlap,
        workers=args.workers,
        recursive=not args.no_recursive,
    )


if __name__ == "__main__":
    main()
