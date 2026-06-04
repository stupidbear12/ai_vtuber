"""
generate_encodec.py

학습된 체크포인트에서 lofi 음악 생성.

사용법:
  python -m encodec_pipeline.generate_encodec \
      --checkpoint checkpoints/encodec/latest.pt \
      --latent_dir encodec_latents \
      --num_samples 4 \
      --ddim_steps 50 \
      --output_dir generated \
      --device cuda

생성 파이프라인:
  가우시안 노이즈 → DDIM 50스텝 역방향 → 잠재 벡터 역정규화 → EnCodec 디코더 → WAV + MP3
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from encodec import EncodecModel

from .unet_1d import UNet1D
from .ddpm_encodec import DDPMScheduler


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────

def load_unet_from_checkpoint(ckpt_path: str, device: str) -> UNet1D:
    """체크포인트에서 UNet1D 가중치 로드."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = UNet1D(
        in_channels=128,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(2, 3),
        time_emb_dim=512,
        dropout=0.0,    # 추론 시 dropout 비활성화
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"UNet1D loaded from {ckpt_path}")
    print(f"  epoch={ckpt.get('epoch', '?')}, step={ckpt.get('global_step', '?')}")
    return model


def load_encodec(device: str) -> EncodecModel:
    """EnCodec 24kHz 디코더 로드."""
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_stats(latent_dir: str) -> tuple[torch.Tensor, torch.Tensor]:
    """stats.json에서 mean/std 로드."""
    stats_path = Path(latent_dir) / "stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"stats.json not found at {stats_path}. "
            "Run preprocess_encodec.py first."
        )
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std  = torch.tensor(stats["std"],  dtype=torch.float32)
    return mean, std


# ──────────────────────────────────────────────
# 오디오 저장
# ──────────────────────────────────────────────

def save_wav(audio: torch.Tensor, path: str, sr: int = 24000):
    """audio: (1, N) or (N,)"""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(path, audio.cpu(), sample_rate=sr)


def save_mp3(audio: torch.Tensor, path: str, sr: int = 24000, bitrate: str = "128k"):
    """
    torchaudio mp3 인코딩.
    torchaudio >= 0.12 에서 mp3 인코딩 가능.
    실패 시 wav만 저장하고 경고 출력.
    """
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    try:
        torchaudio.save(
            path, audio.cpu(), sample_rate=sr,
            format="mp3",
            encoding="MP3",
        )
    except Exception as e:
        print(f"  [WARN] MP3 인코딩 실패 ({e}). WAV만 저장됩니다.")


# ──────────────────────────────────────────────
# 생성 파이프라인
# ──────────────────────────────────────────────

@torch.no_grad()
def generate(
    unet: UNet1D,
    ddpm: DDPMScheduler,
    encodec_model: EncodecModel,
    mean: torch.Tensor,
    std: torch.Tensor,
    num_samples: int,
    device: str,
    ddim_steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 4,
) -> list[torch.Tensor]:
    """
    num_samples 개의 오디오 clip을 생성하여 반환.
    반환: list of (1, N) audio tensors on CPU
    """
    all_audio = []
    generated = 0

    pbar = tqdm(total=num_samples, desc="Generating")

    while generated < num_samples:
        cur_batch = min(batch_size, num_samples - generated)

        # 잠재 벡터 생성 (DDIM)
        latents = ddpm.ddim_sample(
            unet,
            shape=(cur_batch, 128, 225),
            device=device,
            steps=ddim_steps,
            eta=eta,
            show_progress=False,
        )  # (B, 128, 225)

        # 역정규화
        mean_d = mean[:, None].to(device)   # (128, 1)
        std_d  = std[:, None].to(device)    # (128, 1)
        latents_denorm = latents * std_d + mean_d    # (B, 128, 225)

        # EnCodec 디코딩 — 배치를 1개씩 처리 (메모리 절약)
        for i in range(cur_batch):
            lat = latents_denorm[i: i+1]          # (1, 128, 225)
            audio = encodec_model.decoder(lat)    # (1, 1, N)
            audio = audio.squeeze(0).clamp(-1.0, 1.0).cpu()  # (1, N)
            all_audio.append(audio)
            pbar.update(1)

        generated += cur_batch

    pbar.close()
    return all_audio


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate lofi music with EnCodec LDM")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="UNet1D 체크포인트 경로 (latest.pt 등)")
    parser.add_argument("--latent_dir", type=str, default="encodec_latents",
                        help="stats.json이 있는 디렉토리")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="생성할 샘플 수")
    parser.add_argument("--ddim_steps", type=int, default=50,
                        help="DDIM 샘플링 스텝 수")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0=결정론적, 1=DDPM과 동일)")
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--save_mp3", action="store_true",
                        help="WAV 외 MP3도 저장")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="생성 배치 크기")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")

    # ── 모델 로드 ──
    unet = load_unet_from_checkpoint(args.checkpoint, args.device)
    encodec_model = load_encodec(args.device)
    mean, std = load_stats(args.latent_dir)

    ddpm = DDPMScheduler(T=args.T, schedule=args.schedule).to(args.device)

    print(f"\nGenerating {args.num_samples} samples "
          f"({args.ddim_steps} DDIM steps, eta={args.eta}) ...")

    # ── 생성 ──
    audio_clips = generate(
        unet=unet,
        ddpm=ddpm,
        encodec_model=encodec_model,
        mean=mean,
        std=std,
        num_samples=args.num_samples,
        device=args.device,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        batch_size=args.batch_size,
    )

    # ── 저장 ──
    print(f"\nSaving {len(audio_clips)} audio clips to {output_dir} ...")
    for i, audio in enumerate(audio_clips):
        wav_path = str(output_dir / f"generated_{i:04d}.wav")
        save_wav(audio, wav_path, sr=24000)
        print(f"  WAV: {wav_path}")

        if args.save_mp3:
            mp3_path = str(output_dir / f"generated_{i:04d}.mp3")
            save_mp3(audio, mp3_path, sr=24000)
            print(f"  MP3: {mp3_path}")

    print(f"\nDone. {len(audio_clips)} clips saved to {output_dir}/")
    print(f"  Clip length: {audio_clips[0].shape[-1] / 24000:.2f}s (24kHz)")


if __name__ == "__main__":
    main()
