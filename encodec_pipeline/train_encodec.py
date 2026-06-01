"""
train_encodec.py

EnCodec ?좎옱 踰≫꽣 ?뷀벂??紐⑤뜽 ?숈뒿.

?ъ슜踰?
  python -m encodec_pipeline.train_encodec \
      --latent_dir encodec_latents \
      --output_dir checkpoints/encodec \
      --epochs 200 \
      --batch_size 16 \
      --lr 5e-5 \
      --device cuda

二쇱슂 湲곕뒫:
  - AdamW + CosineAnnealingLR
  - fp16 (GradScaler)
  - 泥댄겕?ъ씤?????/ ?ш컻
  - 留?500 ?ㅽ뀦留덈떎 ?섑뵆 ?좎옱 踰≫꽣 ?앹꽦 ??EnCodec ?붿퐫????WAV ???"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torchaudio

from encodec import EncodecModel

from .unet_1d import UNet1D
from .ddpm_encodec import DDPMScheduler


# ??????????????????????????????????????????????
# Dataset
# ??????????????????????????????????????????????

class EncodecLatentDataset(Dataset):
    """
    encodec_latents/ ?붾젆?좊━??.pt ?뚯씪??濡쒕뱶?섎뒗 ?곗씠?곗뀑.
    stats.json??mean/std濡?梨꾨꼸蹂??뺢퇋???섑뻾.
    """

    def __init__(self, latent_dir: str, normalize: bool = True):
        self.latent_dir = Path(latent_dir)
        self.files = sorted(self.latent_dir.glob("chunk_*.pt"))
        if not self.files:
            raise FileNotFoundError(
                f"No chunk_*.pt files in {latent_dir}. "
                "Run preprocess_encodec.py first."
            )

        self.normalize = normalize
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

        if normalize:
            stats_path = self.latent_dir / "stats.json"
            if not stats_path.exists():
                raise FileNotFoundError(f"stats.json not found at {stats_path}")
            with open(stats_path) as f:
                stats = json.load(f)
            self.mean = torch.tensor(stats["mean"], dtype=torch.float32)  # (128,)
            self.std = torch.tensor(stats["std"],  dtype=torch.float32)   # (128,)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        lat = torch.load(self.files[idx], map_location="cpu")  # (128, 225)
        if self.normalize and self.mean is not None:
            lat = (lat - self.mean[:, None]) / self.std[:, None]
        return lat


# ??????????????????????????????????????????????
# EnCodec ?붿퐫???섑띁
# ??????????????????????????????????????????????

def load_encodec_decoder(device: str) -> EncodecModel:
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def decode_latent(
    encodec_model: EncodecModel,
    latent: torch.Tensor,          # (1, 128, 225)  ?뺢퇋?붾맖
    mean: torch.Tensor,            # (128,)
    std: torch.Tensor,             # (128,)
    device: str,
) -> torch.Tensor:
    """
    ?뺢퇋?붾맂 ?좎옱 踰≫꽣 ????젙洹쒗솕 ??EnCodec ?붿퐫?????ㅻ뵒??(1, 1, N).
    """
    # ??젙洹쒗솕
    lat = latent * std[:, None].to(device) + mean[:, None].to(device)
    # (1, 128, 225) ??(1, 1, 72000)
    audio = encodec_model.decoder(lat)
    return audio   # (1, 1, N)


# ??????????????????????????????????????????????
# 泥댄겕?ъ씤??# ??????????????????????????????????????????????

def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    scheduler: DDPMScheduler,
    optimizer: AdamW,
    lr_scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    path = output_dir / f"ckpt_epoch{epoch:04d}_step{global_step:07d}.pt"
    torch.save(ckpt, path)
    # latest ?щ낵由?(?몄쓽)
    latest = output_dir / "latest.pt"
    torch.save(ckpt, latest)
    return path


def load_checkpoint(path: str, model, optimizer, lr_scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["global_step"]


# ??????????????????????????????????????????????
# ?섑뵆 ?앹꽦
# ??????????????????????????????????????????????

def generate_and_save_sample(
    model: nn.Module,
    ddpm: DDPMScheduler,
    encodec_model: EncodecModel,
    dataset: EncodecLatentDataset,
    output_dir: Path,
    global_step: int,
    device: str,
    ddim_steps: int = 50,
):
    """DDIM 50?ㅽ뀦?쇰줈 ?좎옱 踰≫꽣 ?앹꽦 ??WAV ???"""
    model.eval()
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)

    latent = ddpm.ddim_sample(
        model, shape=(1, 128, 225), device=device,
        steps=ddim_steps, show_progress=False
    )  # (1, 128, 225)

    if dataset.mean is not None:
        audio = decode_latent(
            encodec_model, latent,
            dataset.mean, dataset.std, device
        )  # (1, 1, N)
        audio = audio.squeeze(0).cpu()  # (1, N)
        wav_path = sample_dir / f"sample_step{global_step:07d}.wav"
        torchaudio.save(str(wav_path), audio, sample_rate=24000)
        print(f"  [Sample saved] {wav_path}")

    model.train()


# ??????????????????????????????????????????????
# ?숈뒿 猷⑦봽
# ??????????????????????????????????????????????

def train(args):
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ?? Dataset / DataLoader ??
    print(f"Loading dataset from {args.latent_dir} ...")
    dataset = EncodecLatentDataset(args.latent_dir, normalize=True)
    print(f"  Total chunks: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # ?? Models ??
    print("Building UNet1D ...")
    model = UNet1D(
        in_channels=128,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(2, 3),
        time_emb_dim=512,
        dropout=0.1,
    ).to(device)
    print(f"  Parameters: {model.num_parameters / 1e6:.1f}M")

    ddpm = DDPMScheduler(T=args.T, schedule=args.schedule).to(device)

    # EnCodec (?앹꽦 ?섑뵆?? ?숈뒿?먮뒗 誘몄궗??
    print("Loading EnCodec decoder ...")
    encodec_model = load_encodec_decoder(device)

    # ?? Optimizer / Scheduler ??
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = args.epochs * len(loader)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    # init_scale????쾶 ?ㅼ젙: 湲곕낯媛?65536? fp16 max(65504)??洹쇱젒??    # 泥?紐?step?먯꽌 gradient overflow ??NaN ??컻???쇱쑝??
    # 256?쇰줈 ??텛怨?growth_interval??湲멸쾶 ?≪븘 ?덉젙?곸쑝濡??ㅼ????곸듅.
    scaler = GradScaler("cuda", enabled=False,
                        init_scale=256.0, growth_interval=2000)

    # ?? Resume ??
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, scaler
        )
        start_epoch += 1
        print(f"  Resumed at epoch {start_epoch}, step {global_step}")

    # ?? ?숈뒿 ??
    print(f"\nStarting training: {args.epochs} epochs, batch={args.batch_size}")
    model.train()

    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        t_start = time.time()

        for batch_idx, x0 in enumerate(loader):
            x0 = x0.to(device)                           # (B, 128, 225)
            B = x0.shape[0]

            # 타임스텝 샘플링
            t = torch.randint(0, args.T, (B,), device=device)

            # Forward + Loss
            with autocast("cuda", enabled=False):
                loss = ddpm.p_losses(model, x0, t)

            # NaN/Inf loss 諛곗튂 嫄대꼫?.
            # GradScaler??inf gradient留?媛먯??섍퀬 NaN? ?듦낵?쒗궎誘濡?
            # NaN loss濡?backward瑜??ㅽ뻾?섎㈃ weight媛 ?ㅼ뿼??
            if not loss.isfinite():
                global_step += 1
                if global_step % args.log_every == 0:
                    print(f"[E{epoch:03d} S{global_step:07d}] NaN/Inf loss - batch skipped")
                continue

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            # ?? 濡쒓렇 ??
            if global_step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"[E{epoch:03d} S{global_step:07d}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}"
                )

            # ?? ?섑뵆 ?앹꽦 ??
            if global_step % args.sample_every == 0:
                print(f"  Generating sample at step {global_step} ...")
                generate_and_save_sample(
                    model, ddpm, encodec_model, dataset,
                    output_dir, global_step, device,
                    ddim_steps=50,
                )

        # ?? ?먰룷???붿빟 ??
        elapsed = time.time() - t_start
        avg_loss = epoch_loss / len(loader)
        print(
            f"\n=== Epoch {epoch:03d} done | "
            f"avg_loss={avg_loss:.4f} | "
            f"time={elapsed:.1f}s ===\n"
        )

        # ?? 泥댄겕?ъ씤????
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt_path = save_checkpoint(
                output_dir, model, ddpm, optimizer, lr_scheduler,
                scaler, epoch, global_step
            )
            print(f"  Checkpoint saved: {ckpt_path}")

    print("Training complete.")


# ??????????????????????????????????????????????
# CLI
# ??????????????????????????????????????????????

def main():
    parser = argparse.ArgumentParser(description="Train EnCodec Latent Diffusion")
    parser.add_argument("--latent_dir", type=str, default="encodec_latents")
    parser.add_argument("--output_dir", type=str, default="checkpoints/encodec")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--T", type=int, default=1000,
                        help="Diffusion timesteps")
    parser.add_argument("--schedule", type=str, default="cosine",
                        choices=["cosine", "linear"])
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate audio sample every N steps")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()


