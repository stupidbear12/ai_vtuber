"""
preprocess_encodec.py

FMA small MP3 → EnCodec 24kHz continuous latents → .pt 파일 저장
저장 구조:
  encodec_latents/chunk_XXXXX.pt   — shape (128, 225) tensor
  encodec_latents/stats.json       — {"mean": [...], "std": [...]}  (128-dim)

사용법:
  python -m encodec_pipeline.preprocess_encodec \
      --data_dir data/raw/fma_small \
      --out_dir  encodec_latents \
      --chunk_sec 3.0 \
      --device cuda
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

# EnCodec import
from encodec import EncodecModel

# ──────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────
TARGET_SR = 24_000          # EnCodec 24kHz 요구 샘플레이트
LATENT_DIM = 128            # EnCodec encoder 출력 채널
CHUNK_SEC = 3.0             # 청크 길이 (초)
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SEC)   # 72,000 samples
# EnCodec stride = 320 → latent frames per chunk = 72000/320 = 225
LATENT_FRAMES = CHUNK_SAMPLES // 320        # 225


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_encodec(device: str) -> EncodecModel:
    """EnCodec 24kHz 모델을 로드하고 eval 모드로 반환."""
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)          # bandwidth는 quantize 시에만 영향
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


# ──────────────────────────────────────────────
# 오디오 로드 & 전처리
# ──────────────────────────────────────────────
def load_audio_mono_24k(path: str, device: str) -> torch.Tensor:
    """
    MP3/WAV 로드 → 모노 다운믹스 → 24kHz 리샘플링.
    반환: (1, N) float32 tensor on device
    """
    waveform, sr = torchaudio.load(path)          # (C, N)

    # 스테레오 → 모노
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 리샘플링
    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    # 클리핑 방지
    waveform = waveform.clamp(-1.0, 1.0)

    return waveform.to(device)                    # (1, N)


def split_into_chunks(waveform: torch.Tensor) -> list[torch.Tensor]:
    """
    (1, N) → 길이 CHUNK_SAMPLES 청크 리스트.
    마지막 불완전 청크는 버림.
    """
    total = waveform.shape[-1]
    n_chunks = total // CHUNK_SAMPLES
    chunks = []
    for i in range(n_chunks):
        start = i * CHUNK_SAMPLES
        chunk = waveform[:, start: start + CHUNK_SAMPLES]   # (1, 72000)
        chunks.append(chunk)
    return chunks


# ──────────────────────────────────────────────
# 잠재 벡터 추출
# ──────────────────────────────────────────────
@torch.no_grad()
def encode_chunk(model: EncodecModel, chunk: torch.Tensor) -> torch.Tensor:
    """
    chunk: (1, 72000)  → (1, 1, 128, 225)  → (128, 225)
    EnCodec encoder는 (B, C_audio, T) 입력을 받음.
    24kHz 모델은 모노(C_audio=1) 기대.
    """
    # (1, 72000) → (1, 1, 72000)  [B, C, T]
    x = chunk.unsqueeze(0)                          # (1, 1, 72000)
    latent = model.encoder(x)                       # (1, 128, 225)
    return latent.squeeze(0)                        # (128, 225)


# ──────────────────────────────────────────────
# 채널별 통계 계산
# ──────────────────────────────────────────────
def compute_stats(latent_dir: Path) -> dict:
    """
    저장된 .pt 파일들을 순회하며 채널별 mean/std 계산.
    Welford 온라인 알고리즘으로 메모리 효율적으로 계산.
    반환: {"mean": [128 floats], "std": [128 floats]}
    """
    pt_files = sorted(latent_dir.glob("chunk_*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No chunk_*.pt files found in {latent_dir}")

    # Welford online 알고리즘
    count = 0
    mean = torch.zeros(LATENT_DIM)
    M2 = torch.zeros(LATENT_DIM)

    for f in tqdm(pt_files, desc="Computing stats"):
        lat = torch.load(f, map_location="cpu")    # (128, T)
        # T 프레임에 걸쳐 flatten
        for t in range(lat.shape[1]):
            x = lat[:, t]                          # (128,)
            count += 1
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            M2 += delta * delta2

    variance = M2 / max(count - 1, 1)
    std = variance.sqrt()
    # std가 0인 채널 방지
    std = std.clamp(min=1e-6)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_samples": count,
    }


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EnCodec latent extraction")
    parser.add_argument("--data_dir", type=str,
                        default="data/raw/fma_small",
                        help="FMA small 루트 디렉토리")
    parser.add_argument("--out_dir", type=str,
                        default="encodec_latents",
                        help="잠재 벡터 저장 디렉토리")
    parser.add_argument("--chunk_sec", type=float, default=CHUNK_SEC)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_existing", action="store_true",
                        help="이미 처리된 파일 건너뜀")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {out_dir}")

    # 모델 로드
    print("Loading EnCodec 24kHz model...")
    model = load_encodec(args.device)
    print("Model loaded.")

    # MP3 파일 수집
    mp3_files = sorted(data_dir.rglob("*.mp3"))
    print(f"Found {len(mp3_files)} MP3 files")

    chunk_idx = 0
    error_count = 0

    for mp3_path in tqdm(mp3_files, desc="Processing files"):
        try:
            waveform = load_audio_mono_24k(str(mp3_path), args.device)
            chunks = split_into_chunks(waveform)

            for chunk in chunks:
                out_path = out_dir / f"chunk_{chunk_idx:06d}.pt"

                if args.skip_existing and out_path.exists():
                    chunk_idx += 1
                    continue

                latent = encode_chunk(model, chunk)    # (128, 225) on device
                torch.save(latent.cpu(), out_path)
                chunk_idx += 1

        except Exception as e:
            error_count += 1
            print(f"\n[WARN] Failed to process {mp3_path}: {e}")
            continue

    print(f"\nTotal chunks saved: {chunk_idx}")
    print(f"Errors: {error_count}")

    # 통계 계산 및 저장
    print("\nComputing channel-wise statistics...")
    stats = compute_stats(out_dir)
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats saved to {stats_path}")
    print(f"  n_samples: {stats['n_samples']}")
    print(f"  mean[:5]: {stats['mean'][:5]}")
    print(f"  std[:5]:  {stats['std'][:5]}")


if __name__ == "__main__":
    main()
