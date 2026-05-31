"""
data/dataset.py
===============
JAX/NumPy 기반 멜 스펙트로그램 데이터로더

특징:
  - .npy 파일 직접 스트리밍 (메모리 효율)
  - JAX-compatible numpy 배열 반환
  - On-the-fly 데이터 Augmentation
      * Pitch shift (±2 semitone, 주파수 축 이동으로 근사)
      * Time stretch (0.9~1.1, 시간 축 리샘플링)
      * Gaussian noise 추가
  - 재현성을 위한 JAX PRNG 키 기반 셔플

사용법:
    from data.dataset import LofiDataset, create_dataloader

    dataset = LofiDataset("data/processed")
    loader  = create_dataloader(dataset, batch_size=16, shuffle=True)

    for batch in loader:
        # batch: jnp.ndarray, shape (B, 1, 128, 256)
        ...
"""

import os
import random
from pathlib import Path
from typing import Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np

# ──────────────────────────────────────────────
# 하이퍼파라미터 (preprocess.py 와 일치)
# ──────────────────────────────────────────────
N_MELS: int = 128
CHUNK_FRAMES: int = 256


# ──────────────────────────────────────────────
# Augmentation 함수 (numpy 연산)
# ──────────────────────────────────────────────

def augment_pitch_shift(
    spec: np.ndarray,
    n_steps: float,
) -> np.ndarray:
    """
    멜 스펙트로그램에서 pitch shift 를 근사합니다.

    실제 pitch shift 는 waveform 단계에서 수행해야 하지만,
    추론 속도를 위해 주파수 축(mel bin)을 정수 픽셀만큼
    roll 하는 방식으로 근사합니다.

    n_steps: semitone 단위 이동량 (양수 = 고음, 음수 = 저음)
             12 semitone ≈ mel bin 전체 범위의 1/8 이동 (128빈 기준 ~16빈)
    """
    shift_bins = int(round(n_steps * N_MELS / 48.0))  # 근사 계수
    return np.roll(spec, shift_bins, axis=0).astype(np.float32)


def augment_time_stretch(
    spec: np.ndarray,
    rate: float,
    target_frames: int = CHUNK_FRAMES,
) -> np.ndarray:
    """
    멜 스펙트로그램 시간 축을 rate 배 리샘플링 후 target_frames 로 크롭/패딩.

    rate < 1.0: 느리게 (더 많은 프레임 → 크롭)
    rate > 1.0: 빠르게 (더 적은 프레임 → 패딩)
    """
    n_mels, t = spec.shape
    new_t = max(1, int(round(t / rate)))

    # 선형 보간으로 리샘플링
    src_indices = np.linspace(0, t - 1, new_t)
    lo = np.floor(src_indices).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    alpha = src_indices - lo
    stretched = spec[:, lo] * (1 - alpha) + spec[:, hi] * alpha

    # target_frames 에 맞게 크롭 또는 패딩
    if stretched.shape[1] >= target_frames:
        result = stretched[:, :target_frames]
    else:
        pad_width = target_frames - stretched.shape[1]
        result = np.pad(stretched, ((0, 0), (0, pad_width)), mode="edge")

    return result.astype(np.float32)


def augment_noise(
    spec: np.ndarray,
    std: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Gaussian 노이즈 추가 — 정규화된 스펙트로그램 기준 std."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, std, spec.shape).astype(np.float32)
    return np.clip(spec + noise, -1.0, 1.0)


def apply_augmentation(
    spec: np.ndarray,
    rng: np.random.Generator,
    pitch_range: float = 2.0,
    time_range: tuple[float, float] = (0.9, 1.1),
    noise_std: float = 0.005,
    prob: float = 0.5,
) -> np.ndarray:
    """
    확률적으로 augmentation 적용.

    prob: 각 augmentation 적용 확률 (독립 베르누이)
    """
    # Pitch shift
    if rng.random() < prob:
        n_steps = rng.uniform(-pitch_range, pitch_range)
        spec = augment_pitch_shift(spec, n_steps)

    # Time stretch
    if rng.random() < prob:
        rate = rng.uniform(*time_range)
        spec = augment_time_stretch(spec, rate)

    # Gaussian noise
    if rng.random() < prob:
        spec = augment_noise(spec, std=noise_std, rng=rng)

    return spec


# ──────────────────────────────────────────────
# Dataset 클래스
# ──────────────────────────────────────────────

class LofiDataset:
    """
    전처리된 .npy 멜 스펙트로그램 데이터셋.

    각 .npy 파일은 shape (128, 256), dtype float32, 범위 [-1, 1].
    """

    def __init__(
        self,
        data_dir: str | Path,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self._files = sorted(self.data_dir.glob("**/*.npy"))
        if not self._files:
            raise FileNotFoundError(
                f"data_dir 에 .npy 파일이 없습니다: {self.data_dir}"
            )

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> np.ndarray:
        """
        단일 샘플 로드.

        반환: shape (1, 128, 256) float32  ← 채널 차원 포함
        """
        spec = np.load(self._files[idx]).astype(np.float32)

        # shape 검증
        if spec.shape != (N_MELS, CHUNK_FRAMES):
            raise ValueError(
                f"예상 shape ({N_MELS}, {CHUNK_FRAMES}), "
                f"실제 shape {spec.shape}: {self._files[idx]}"
            )

        if self.augment:
            spec = apply_augmentation(spec, self.rng)

        # 채널 차원 추가: (128, 256) → (1, 128, 256)
        return spec[np.newaxis, :, :]

    def shuffle(self, seed: Optional[int] = None) -> None:
        """파일 목록을 in-place 셔플."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self._files))
        self._files = [self._files[i] for i in idx]

    @property
    def file_paths(self) -> list[Path]:
        return list(self._files)


# ──────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────

def create_dataloader(
    dataset: LofiDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 0,
) -> Iterator[jnp.ndarray]:
    """
    제너레이터 기반 데이터로더.

    반환: jnp.ndarray, shape (batch_size, 1, 128, 256)

    사용법:
        loader = create_dataloader(dataset, batch_size=16)
        for batch in loader:
            # batch.shape == (16, 1, 128, 256)
            loss = train_step(params, batch)
    """
    n = len(dataset)
    indices = list(range(n))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    batch_buf: list[np.ndarray] = []
    for idx in indices:
        batch_buf.append(dataset[idx])
        if len(batch_buf) == batch_size:
            batch_np = np.stack(batch_buf, axis=0)  # (B, 1, 128, 256)
            yield jnp.array(batch_np)
            batch_buf = []

    # 남은 샘플 처리 (drop_last=False 인 경우)
    if batch_buf and not drop_last:
        batch_np = np.stack(batch_buf, axis=0)
        yield jnp.array(batch_np)


def infinite_dataloader(
    dataset: LofiDataset,
    batch_size: int = 16,
    shuffle: bool = True,
) -> Iterator[jnp.ndarray]:
    """
    훈련 루프용 무한 반복 데이터로더.

    에포크마다 셔플을 새로 수행합니다.
    """
    epoch = 0
    while True:
        yield from create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            seed=epoch,
        )
        epoch += 1


# ──────────────────────────────────────────────
# 테스트 / 간단한 사용 예시
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/processed"

    print(f"데이터 디렉토리: {data_dir}")
    try:
        ds = LofiDataset(data_dir, augment=True, seed=42)
    except FileNotFoundError as e:
        print(f"오류: {e}")
        print("먼저 data/preprocess.py 를 실행해 .npy 파일을 생성하세요.")
        sys.exit(1)

    print(f"총 샘플 수: {len(ds)}")

    # 첫 번째 샘플 확인
    sample = ds[0]
    print(f"샘플 shape: {sample.shape}")   # (1, 128, 256)
    print(f"값 범위: [{sample.min():.3f}, {sample.max():.3f}]")

    # 배치 생성 테스트
    loader = create_dataloader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"배치 shape: {batch.shape}")    # (4, 1, 128, 256)
    print(f"배치 dtype: {batch.dtype}")
    print("데이터로더 테스트 완료!")
