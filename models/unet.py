"""
models/unet.py
U-Net 아키텍처 — PyTorch 구현
입력: (batch, 1, 128, 256) 멜 스펙트로그램 + (batch,) 타임스텝 t
출력: (batch, 1, 128, 256) 예측 노이즈
약 50M 파라미터 @ channels=[64,128,256,512]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 1. Time Embedding
# ──────────────────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    """사인/코사인 positional encoding → Linear → SiLU → Linear."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) int 또는 float 타임스텝
        Returns:
            emb: (batch, dim*4)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        t_f = t.float()
        args = t_f[:, None] * freqs[None, :]          # (batch, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (batch, dim)
        return self.mlp(emb)                           # (batch, dim*4)


# ──────────────────────────────────────────────────────────────
# 2. ResBlock
# ──────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """GroupNorm(8) + SiLU + Conv2d + TimeEmbedding 주입."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int,
                 num_groups: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (batch, in_ch, H, W)
            temb: (batch, time_emb_dim)
        Returns:
            out:  (batch, out_ch, H, W)
        """
        h = self.conv1(F.silu(self.norm1(x)))
        # time embedding 주입: (batch, out_ch) → (batch, out_ch, 1, 1)
        h = h + self.time_proj(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


# ──────────────────────────────────────────────────────────────
# 3. SelfAttention
# ──────────────────────────────────────────────────────────────
class SelfAttention(nn.Module):
    """2D feature map에 멀티헤드 self-attention 적용."""

    def __init__(self, ch: int, num_heads: int = 8, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, ch)
        self.attn = nn.MultiheadAttention(ch, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W)
        Returns:
            out: (batch, C, H, W)
        """
        B, C, H, W = x.shape
        h = self.norm(x)                              # (B, C, H, W)
        h = h.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        h, _ = self.attn(h, h, h)                    # (B, H*W, C)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        return x + h


# ──────────────────────────────────────────────────────────────
# 4. UNet
# ──────────────────────────────────────────────────────────────
class UNet(nn.Module):
    """
    전체 U-Net.
    입력: (batch, 1, 128, 256) 멜 스펙트로그램 + (batch,) 타임스텝
    출력: (batch, 1, 128, 256) 예측 노이즈
    약 50M 파라미터 @ channels=[64,128,256,512]
    """

    def __init__(
        self,
        in_ch: int = 1,
        channels: tuple = (64, 128, 256, 512),
        time_emb_dim: int = 256,
        num_heads: int = 8,
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_emb_dim)
        temb_dim = time_emb_dim * 4

        # 입력 projection
        self.in_proj = nn.Conv2d(in_ch, channels[0], 3, padding=1)

        # ── Encoder ──
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_chs = [channels[0]] + list(channels)  # [64, 64, 128, 256, 512]
        for i in range(len(channels)):
            ic = enc_chs[i]
            oc = enc_chs[i + 1]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock(ic, oc, temb_dim, num_groups, dropout),
                ResBlock(oc, oc, temb_dim, num_groups, dropout),
            ]))
            self.downs.append(
                nn.Conv2d(oc, oc, 3, stride=2, padding=1)  # stride=2 downsample
            )

        # ── Bottleneck ──
        bot_ch = channels[-1]
        self.bot1 = ResBlock(bot_ch, bot_ch, temb_dim, num_groups, dropout)
        self.bot_attn = SelfAttention(bot_ch, num_heads, num_groups)
        self.bot2 = ResBlock(bot_ch, bot_ch, temb_dim, num_groups, dropout)

        # ── Decoder ──
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_chs = list(reversed(channels))   # [512, 256, 128, 64]
        for i in range(len(dec_chs)):
            ic = dec_chs[i]
            oc = dec_chs[i + 1] if i + 1 < len(dec_chs) else dec_chs[-1]
            skip_ch = dec_chs[i]  # skip connection channel
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(ic, ic, 3, padding=1),
                )
            )
            self.dec_blocks.append(nn.ModuleList([
                ResBlock(ic + skip_ch, oc, temb_dim, num_groups, dropout),
                ResBlock(oc, oc, temb_dim, num_groups, dropout),
            ]))

        # ── 출력 projection ──
        self.out_norm = nn.GroupNorm(num_groups, dec_chs[-1])
        self.out_proj = nn.Conv2d(dec_chs[-1], in_ch, 3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 128, 256) 노이즈가 추가된 스펙트로그램
            t: (batch,) 타임스텝
        Returns:
            (batch, 1, 128, 256) 예측 노이즈
        """
        temb = self.time_emb(t)       # (batch, time_emb_dim*4)

        # 입력 projection
        x = self.in_proj(x)           # (batch, 64, 128, 256)

        # Encoder
        skips = []
        for (res1, res2), down in zip(self.enc_blocks, self.downs):
            x = res1(x, temb)
            x = res2(x, temb)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bot1(x, temb)
        x = self.bot_attn(x)
        x = self.bot2(x, temb)

        # Decoder
        for (up, (res1, res2)), skip in zip(
            zip(self.ups, self.dec_blocks), reversed(skips)
        ):
            x = up(x)
            x = torch.cat([x, skip], dim=1)   # skip connection concat
            x = res1(x, temb)
            x = res2(x, temb)

        # 출력 projection
        x = self.out_proj(F.silu(self.out_norm(x)))
        return x


# ──────────────────────────────────────────────────────────────
# 간단한 shape 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    x = torch.randn(2, 1, 128, 256, device=device)
    t = torch.tensor([100, 500], device=device)

    with torch.no_grad():
        out = model(x, t)

    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
