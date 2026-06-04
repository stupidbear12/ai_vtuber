"""
train.py
Lofi Diffusion 학습 루프 — PyTorch 구현
- torch.cuda.amp 혼합 정밀도 (float16)
- AdamW + CosineAnnealingLR
- DataLoader (num_workers=4, pin_memory=True)
- torch.save / torch.load 체크포인트
- 매 500 스텝 샘플 생성 + 저장
- 학습 재개 지원
- tqdm 진행바
"""

import os
import sys
import time
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.unet import UNet
from diffusion.ddpm import DDPMScheduler


# ──────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────
DEFAULTS = dict(
    data_dir       = "data/processed",
    checkpoint_dir = "checkpoints",
    sample_dir     = "samples",
    # 모델
    channels       = (64, 128, 256, 512),
    time_emb_dim   = 256,
    num_heads      = 8,
    # DDPM
    T              = 1000,
    schedule       = "cosine",
    # 학습
    batch_size     = 8,
    lr             = 2e-4,
    weight_decay   = 1e-4,
    total_steps    = 200_000,
    # 로깅/저장
    log_every      = 50,
    save_every     = 2_000,
    sample_every   = 500,
    num_samples    = 4,
    # 기타
    mixed_prec     = True,
    seed           = 42,
    num_workers    = 4,
)


# ──────────────────────────────────────────────────────────────
# 데이터셋
# ──────────────────────────────────────────────────────────────
class MelDataset(Dataset):
    """
    data/processed/*.npy 파일을 로드하는 Dataset.
    각 파일 shape: (128, 256) 또는 (1, 128, 256)
    """
    def __init__(self, data_dir: str):
        data_path = Path(data_dir)
        self.files = sorted(data_path.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")
        print(f"[DataLoader] {len(self.files)} files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]        # (1, 128, 256)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.transpose(2, 0, 1)  # (H,W,1) → (1,H,W)
        return torch.from_numpy(arr)    # (1, 128, 256)


def make_loader(cfg: dict) -> DataLoader:
    dataset = MelDataset(cfg['data_dir'])
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg['num_workers'] > 0),
    )
    return loader


# ──────────────────────────────────────────────────────────────
# 체크포인트 유틸
# ──────────────────────────────────────────────────────────────
def save_checkpoint(ckpt_dir: str, model: nn.Module, optimizer, scaler: GradScaler,
                    scheduler, step: int, cfg: dict):
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    path = ckpt_path / f"ckpt_step{step:07d}.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'cfg': cfg,
    }, path)
    # 최신 5개만 유지
    all_ckpts = sorted(ckpt_path.glob("ckpt_step*.pt"))
    for old in all_ckpts[:-5]:
        old.unlink()
    print(f"  [Checkpoint] Saved → {path.name}")


def load_checkpoint(ckpt_dir: str, model: nn.Module, optimizer, scaler: GradScaler,
                    scheduler, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    all_ckpts = sorted(ckpt_path.glob("ckpt_step*.pt"))
    if not all_ckpts:
        print("[Checkpoint] No existing checkpoint found. Starting fresh.")
        return 0
    latest = all_ckpts[-1]
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    step = ckpt['step']
    print(f"[Checkpoint] Restored from step {step} ({latest.name})")
    return step


# ──────────────────────────────────────────────────────────────
# 샘플 생성 + 저장
# ──────────────────────────────────────────────────────────────
def generate_and_save_samples(
    model: nn.Module,
    ddpm: DDPMScheduler,
    sample_dir: str,
    step: int,
    num_samples: int,
    device: torch.device,
):
    model.eval()
    sample_path = Path(sample_dir)
    sample_path.mkdir(parents=True, exist_ok=True)

    shape = (num_samples, 1, 128, 256)
    with torch.no_grad():
        samples = ddpm.ddim_sample(model, shape, device, steps=50)
    samples_np = samples.cpu().numpy()  # (N, 1, 128, 256)

    # npy 저장
    npy_path = sample_path / f"sample_step{step:07d}.npy"
    np.save(npy_path, samples_np)

    # 시각화 저장
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    for ax, s in zip(axes, samples_np):
        ax.imshow(s[0], origin='lower', aspect='auto', cmap='magma')
        ax.axis('off')
    plt.suptitle(f"Step {step}", fontsize=12)
    plt.tight_layout()
    png_path = sample_path / f"sample_step{step:07d}.png"
    plt.savefig(png_path, dpi=100)
    plt.close(fig)
    print(f"  [Sample] saved → {npy_path.name}, {png_path.name}")
    model.train()


# ──────────────────────────────────────────────────────────────
# 메인 학습 루프
# ──────────────────────────────────────────────────────────────
def train(cfg: dict):
    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")
    if device.type == 'cuda':
        print(f"         GPU: {torch.cuda.get_device_name(0)}")
        print(f"         VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])

    # 모델
    model = UNet(
        channels=cfg['channels'],
        time_emb_dim=cfg['time_emb_dim'],
        num_heads=cfg['num_heads'],
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Total parameters: {total_params:,}")

    # DDPM 스케줄러
    ddpm = DDPMScheduler(T=cfg['T'], schedule=cfg['schedule'])

    # 옵티마이저
    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # LR 스케줄러
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg['total_steps'], eta_min=cfg['lr'] * 0.1)

    # AMP GradScaler
    scaler = GradScaler(enabled=cfg['mixed_prec'])

    # 체크포인트 복원
    Path(cfg['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    start_step = load_checkpoint(
        cfg['checkpoint_dir'], model, optimizer, scaler, lr_scheduler, device
    )

    # 데이터 로더 (무한 반복)
    loader = make_loader(cfg)
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ── 학습 루프 ──
    print(f"\n[Train] Starting from step {start_step} / {cfg['total_steps']}")
    model.train()

    pbar = tqdm(
        range(start_step, cfg['total_steps']),
        initial=start_step,
        total=cfg['total_steps'],
        desc="Training",
        dynamic_ncols=True,
    )

    loss_window = []
    t0 = time.time()

    for step in pbar:
        x0 = next_batch().to(device, non_blocking=True)  # (B, 1, 128, 256)
        B = x0.shape[0]
        t_steps = torch.randint(0, cfg['T'], (B,), device=device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=cfg['mixed_prec']):
            loss = ddpm.p_losses(model, x0, t_steps)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()

        loss_val = loss.item()
        loss_window.append(loss_val)
        if len(loss_window) > 100:
            loss_window.pop(0)

        # 로깅
        if (step + 1) % cfg['log_every'] == 0:
            avg_loss = sum(loss_window) / len(loss_window)
            lr_now = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr_now:.2e}")

        # 샘플 생성
        if (step + 1) % cfg['sample_every'] == 0:
            generate_and_save_samples(
                model, ddpm, cfg['sample_dir'], step + 1, cfg['num_samples'], device
            )

        # 체크포인트
        if (step + 1) % cfg['save_every'] == 0:
            save_checkpoint(
                cfg['checkpoint_dir'], model, optimizer, scaler, lr_scheduler, step + 1, cfg
            )

    # 최종 체크포인트
    save_checkpoint(
        cfg['checkpoint_dir'], model, optimizer, scaler, lr_scheduler, cfg['total_steps'], cfg
    )
    elapsed = time.time() - t0
    print(f"\n[Done] Training complete in {elapsed/3600:.2f}h. Final checkpoint saved.")


# ──────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train Lofi Diffusion Model (PyTorch)")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action=argparse.BooleanOptionalAction)
        elif isinstance(v, (list, tuple)):
            p.add_argument(f"--{k}", nargs='+', type=int, default=list(v))
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


if __name__ == "__main__":
    cfg = parse_args()
    cfg['channels'] = tuple(cfg['channels'])
    train(cfg)
