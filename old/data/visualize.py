"""
data/visualize.py
=================
멜 스펙트로그램 시각화 도구 — 전처리 결과 검증용

기능:
  1. 단일 오디오 파일: 파형 + 멜 스펙트로그램 나란히 표시
  2. .npy 파일: 전처리된 스펙트로그램 표시
  3. 배치 그리드: 여러 청크를 한눈에 비교
  4. Augmentation 비교: 원본 vs 변환 결과

사용법:
    # 오디오 파일 시각화
    python data/visualize.py --audio data/raw/my_track.wav

    # 전처리된 .npy 파일 시각화
    python data/visualize.py --npy data/processed/my_track_chunk0000.npy

    # processed 폴더의 처음 N개 청크를 그리드로 표시
    python data/visualize.py --grid data/processed --n 16

    # Augmentation 비교
    python data/visualize.py --augment data/raw/my_track.wav
"""

import argparse
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# preprocess.py 와 동일한 상수
SAMPLE_RATE: int = 22050
N_MELS: int = 128
N_FFT: int = 1024
HOP_LENGTH: int = 256
CHUNK_FRAMES: int = 256
TOP_DB: float = 80.0


# ──────────────────────────────────────────────
# 헬퍼 함수
# ──────────────────────────────────────────────

def _load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32)


def _compute_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE,
        n_mels=N_MELS, n_fft=N_FFT,
        hop_length=HOP_LENGTH, power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=TOP_DB)


def _normalize(spec: np.ndarray) -> np.ndarray:
    spec = np.clip(spec, -TOP_DB, 0.0)
    return (spec / TOP_DB * 2.0 + 1.0).astype(np.float32)  # [-1, 1]


def _set_mel_yaxis(ax, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> None:
    """멜 스펙트로그램 y축 레이블 설정."""
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr // 2)
    tick_positions = np.linspace(0, n_mels - 1, 6).astype(int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{freqs[i]:.0f}" for i in tick_positions])
    ax.set_ylabel("Frequency (Hz)")


# ──────────────────────────────────────────────
# 1. 오디오 파일 → 파형 + 멜 스펙트로그램
# ──────────────────────────────────────────────

def plot_audio_and_mel(
    audio_path: Path,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    원본 오디오 파형과 멜 스펙트로그램을 나란히 시각화.

    파형(위) + 멜 스펙트로그램(아래) 레이아웃.
    """
    from typing import Optional  # noqa
    audio = _load_audio(audio_path)
    mel_db = _compute_mel(audio)

    duration = len(audio) / SAMPLE_RATE
    times_audio = np.linspace(0, duration, len(audio))
    times_mel = librosa.frames_to_time(
        np.arange(mel_db.shape[1]),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), constrained_layout=True)
    fig.suptitle(f"Audio Analysis: {audio_path.name}", fontsize=13, fontweight="bold")

    # 파형
    axes[0].plot(times_audio, audio, color="#4a90d9", linewidth=0.5, alpha=0.85)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].set_xlim(0, duration)
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")

    # 멜 스펙트로그램
    img = axes[1].imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[times_mel[0], times_mel[-1], 0, N_MELS],
    )
    plt.colorbar(img, ax=axes[1], label="dB")
    _set_mel_yaxis(axes[1])
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title(
        f"Mel Spectrogram  (n_mels={N_MELS}, hop={HOP_LENGTH}, n_fft={N_FFT})"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 2. 전처리된 .npy 파일 시각화
# ──────────────────────────────────────────────

def plot_npy(
    npy_path: Path,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """정규화된 [-1, 1] 멜 스펙트로그램 .npy 파일 시각화."""
    from typing import Optional  # noqa
    spec = np.load(npy_path).astype(np.float32)
    if spec.ndim == 3:
        spec = spec[0]  # (1, 128, 256) → (128, 256)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    fig.suptitle(f"Preprocessed Spectrogram: {npy_path.name}", fontsize=12)

    img = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=-1.0,
        vmax=1.0,
    )
    plt.colorbar(img, ax=ax, label="Normalized value [-1, 1]")
    _set_mel_yaxis(ax)
    ax.set_xlabel(f"Frame (hop={HOP_LENGTH}, ~{CHUNK_FRAMES * HOP_LENGTH / SAMPLE_RATE:.1f}s total)")
    ax.set_title(
        f"Shape: {spec.shape}  |  min={spec.min():.3f}  max={spec.max():.3f}"
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 3. 청크 그리드 시각화
# ──────────────────────────────────────────────

def plot_grid(
    data_dir: Path,
    n: int = 16,
    cols: int = 4,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """processed 폴더 내 .npy 청크를 그리드로 비교."""
    from typing import Optional  # noqa
    files = sorted(data_dir.glob("**/*.npy"))[:n]
    if not files:
        print(f"오류: {data_dir} 에 .npy 파일 없음")
        return

    rows = (len(files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5), constrained_layout=True)
    fig.suptitle(f"Spectrogram Grid ({len(files)} chunks from {data_dir.name})", fontsize=12)

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (ax, fpath) in enumerate(zip(axes_flat, files)):
        spec = np.load(fpath).astype(np.float32)
        if spec.ndim == 3:
            spec = spec[0]
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=1)
        ax.set_title(fpath.stem[-20:], fontsize=7)
        ax.axis("off")

    # 남은 서브플롯 숨기기
    for ax in axes_flat[len(files):]:
        ax.set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"저장: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 4. Augmentation 비교
# ──────────────────────────────────────────────

def plot_augmentation_comparison(
    audio_path: Path,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """원본 스펙트로그램과 각종 augmentation 결과를 4-panel 로 비교."""
    from typing import Optional  # noqa
    # 지연 임포트 (dataset 모듈 의존성)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.preprocess import (
        audio_to_melspectrogram,
        load_audio,
        normalize,
        slice_spectrogram,
    )
    from data.dataset import (
        augment_noise,
        augment_pitch_shift,
        augment_time_stretch,
    )

    audio = load_audio(audio_path)
    mel = normalize(audio_to_melspectrogram(audio))
    chunks = slice_spectrogram(mel, CHUNK_FRAMES, overlap=0.0)
    if not chunks:
        print("청크를 만들기에 오디오가 너무 짧습니다.")
        return
    original = chunks[0]

    rng = np.random.default_rng(0)
    variants = {
        "Original": original,
        "Pitch +2st": augment_pitch_shift(original, n_steps=2.0),
        "Time ×0.9": augment_time_stretch(original, rate=0.9),
        "Noise σ=0.05": augment_noise(original, std=0.05, rng=rng),
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    fig.suptitle(f"Augmentation Comparison: {audio_path.name}", fontsize=12)

    for ax, (title, spec) in zip(axes, variants.items()):
        ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"저장: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

from typing import Optional  # noqa: E402 (모듈 수준 import)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="멜 스펙트로그램 시각화 도구"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio", type=str, help="오디오 파일 경로 (.wav/.mp3 등)")
    group.add_argument("--npy", type=str, help=".npy 전처리 파일 경로")
    group.add_argument("--grid", type=str, help="processed 디렉토리 경로 (그리드 표시)")
    group.add_argument("--augment", type=str, help="오디오 파일 경로 (augmentation 비교)")

    parser.add_argument("--n", type=int, default=16, help="그리드 표시 청크 수 (기본: 16)")
    parser.add_argument("--cols", type=int, default=4, help="그리드 열 수 (기본: 4)")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="출력 이미지 저장 경로 (지정 안 하면 화면에만 표시)",
    )
    parser.add_argument("--no_show", action="store_true", help="화면 출력 비활성화")

    args = parser.parse_args()
    show = not args.no_show
    save_path = Path(args.save) if args.save else None

    if args.audio:
        plot_audio_and_mel(Path(args.audio), save_path=save_path, show=show)
    elif args.npy:
        plot_npy(Path(args.npy), save_path=save_path, show=show)
    elif args.grid:
        plot_grid(Path(args.grid), n=args.n, cols=args.cols, save_path=save_path, show=show)
    elif args.augment:
        plot_augmentation_comparison(Path(args.augment), save_path=save_path, show=show)


if __name__ == "__main__":
    main()
