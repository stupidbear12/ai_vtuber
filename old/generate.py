"""
generate.py
저장된 체크포인트로부터 멜 스펙트로그램 생성 — PyTorch 구현
- DDIM 50스텝 빠른 생성
- .npy 저장 + matplotlib 시각화
- 배치 생성 지원
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from models.unet import UNet
from diffusion.ddpm import DDPMScheduler


# ──────────────────────────────────────────────────────────────
# 체크포인트 로드
# ──────────────────────────────────────────────────────────────
def load_checkpoint(
    checkpoint_dir: str,
    device: torch.device,
    channels=(64, 128, 256, 512),
    time_emb_dim: int = 256,
    num_heads: int = 8,
):
    """
    torch.load로 저장된 체크포인트 로드.
    Returns:
        model:  로드된 UNet (eval 모드)
        step:   마지막 저장 스텝
    """
    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    all_ckpts = sorted(ckpt_path.glob("ckpt_step*.pt"))
    if not all_ckpts:
        raise RuntimeError(f"No checkpoint files found in {checkpoint_dir}")

    latest = all_ckpts[-1]
    ckpt = torch.load(latest, map_location=device)

    model = UNet(
        channels=channels,
        time_emb_dim=time_emb_dim,
        num_heads=num_heads,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', 0)
    print(f"[Load] Restored checkpoint at step {step} ({latest.name})")
    return model, step


# ──────────────────────────────────────────────────────────────
# 생성 함수
# ──────────────────────────────────────────────────────────────
def generate(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    device: torch.device,
    num_samples: int = 4,
    ddim_steps: int = 50,
    eta: float = 0.0,
) -> np.ndarray:
    """
    DDIM으로 멜 스펙트로그램 배치 생성.
    Returns:
        samples: (num_samples, 1, 128, 256) numpy array
    """
    shape = (num_samples, 1, 128, 256)
    print(f"[Generate] DDIM {ddim_steps} steps, {num_samples} samples...")

    with torch.no_grad():
        samples = scheduler.ddim_sample(model, shape, device, steps=ddim_steps, eta=eta)

    return samples.cpu().numpy()  # (N, 1, 128, 256)


# ──────────────────────────────────────────────────────────────
# 저장 함수
# ──────────────────────────────────────────────────────────────
def save_results(
    samples: np.ndarray,   # (N, 1, 128, 256)
    output_dir: str,
    step: int,
    prefix: str = "gen",
):
    """생성된 스펙트로그램을 .npy + .png로 저장."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # npy 저장 (전체 배치)
    npy_file = out_path / f"{prefix}_step{step:07d}.npy"
    np.save(npy_file, samples)
    print(f"[Save] Saved numpy array → {npy_file}")

    # 그리드 이미지 저장
    N = samples.shape[0]
    fig, axes = plt.subplots(1, N, figsize=(4 * N, 4))
    if N == 1:
        axes = [axes]

    for idx, (ax, s) in enumerate(zip(axes, samples)):
        spec = s[0]   # (128, 256) — channel 차원 제거
        im = ax.imshow(spec, origin='lower', aspect='auto', cmap='magma')
        ax.set_title(f"Sample {idx + 1}", fontsize=10)
        ax.set_xlabel("Time Frames")
        ax.set_ylabel("Mel Bins")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Generated Mel Spectrograms (step {step})", fontsize=12)
    plt.tight_layout()
    grid_file = out_path / f"{prefix}_step{step:07d}_grid.png"
    plt.savefig(grid_file, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[Save] Saved grid visualization → {grid_file}")

    # 개별 파일 저장
    for idx, s in enumerate(samples):
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.imshow(s[0], origin='lower', aspect='auto', cmap='magma')
        ax2.set_title(f"Generated Mel Spectrogram #{idx + 1}")
        ax2.set_xlabel("Time Frames")
        ax2.set_ylabel("Mel Bins")
        single_file = out_path / f"{prefix}_step{step:07d}_{idx:02d}.png"
        plt.savefig(single_file, dpi=100, bbox_inches='tight')
        plt.close(fig2)

    print(f"[Save] Saved {N} individual PNGs to {out_path}/")


# ──────────────────────────────────────────────────────────────
# 통계 출력
# ──────────────────────────────────────────────────────────────
def print_sample_stats(samples: np.ndarray):
    print("\n[Stats] Generated sample statistics:")
    print(f"  Shape:  {samples.shape}")
    print(f"  Min:    {samples.min():.4f}")
    print(f"  Max:    {samples.max():.4f}")
    print(f"  Mean:   {samples.mean():.4f}")
    print(f"  Std:    {samples.std():.4f}")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Generate Lofi Mel Spectrograms (PyTorch)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir",     type=str, default="generated")
    p.add_argument("--num_samples",    type=int, default=4)
    p.add_argument("--ddim_steps",     type=int, default=50)
    p.add_argument("--eta",            type=float, default=0.0)
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--T",              type=int, default=1000)
    p.add_argument("--schedule",       type=str, default="cosine",
                   choices=["linear", "cosine"])
    p.add_argument("--channels",       nargs='+', type=int,
                   default=[64, 128, 256, 512])
    p.add_argument("--time_emb_dim",   type=int, default=256)
    p.add_argument("--num_heads",      type=int, default=8)
    p.add_argument("--prefix",         type=str, default="gen")
    p.add_argument("--device",         type=str, default="",
                   help="cuda / cpu (기본: 자동 감지)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 디바이스
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    torch.manual_seed(args.seed)

    # 1) 체크포인트 로드
    model, step = load_checkpoint(
        args.checkpoint_dir,
        device,
        channels=tuple(args.channels),
        time_emb_dim=args.time_emb_dim,
        num_heads=args.num_heads,
    )

    # 2) 스케줄러 초기화
    scheduler = DDPMScheduler(T=args.T, schedule=args.schedule)

    # 3) 생성
    samples = generate(
        model, scheduler, device,
        num_samples=args.num_samples,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
    )

    # 4) 통계 출력
    print_sample_stats(samples)

    # 5) 저장
    save_results(samples, args.output_dir, step, prefix=args.prefix)

    print("\n[Done] Generation complete!")
