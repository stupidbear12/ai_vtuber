"""
unet_1d.py

1D U-Net for diffusion over EnCodec latent sequences.
입력 shape:  (B, 128, 225)   — (batch, latent_channels, latent_frames)
출력 shape:  (B, 128, 225)   — 예측된 노이즈 (또는 x0)

아키텍처:
  Encoder:  128 → 256 → 512 → 1024  (stride=2 downsampling)
  Bottleneck: ResBlock + SelfAttention + ResBlock
  Decoder:  skip-connection + upsample  (역순)
  총 파라미터: ~45-55M

참고: AudioLDM2, DDPM (Ho et al. 2020)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────
# 1. Time Embedding
# ──────────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding → 2-layer MLP.
    출력 dim: time_emb_dim
    """

    def __init__(self, model_channels: int, time_emb_dim: int):
        super().__init__()
        self.model_channels = model_channels
        half = model_channels // 2
        # 주파수 벡터 (고정, 학습 안 함)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("freqs", freqs)   # (half,)

        self.mlp = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) int64 timestep indices
        반환: (B, time_emb_dim)
        """
        t = t.float()
        args = t[:, None] * self.freqs[None, :]          # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1) # (B, model_channels)
        return self.mlp(emb)                              # (B, time_emb_dim)


# ──────────────────────────────────────────────────────
# 2. ResBlock1D
# ──────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """
    1D Residual block with time embedding injection.

    구조:
      x → GroupNorm → SiLU → Conv1d →
          + time_emb projection →
          GroupNorm → SiLU → Dropout → Conv1d →
          + shortcut (1×1 conv if needed)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection → out_channels (scale+shift 방식)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2),  # scale & shift
        )

        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        # shortcut
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        x:        (B, C_in, T)
        time_emb: (B, time_emb_dim)
        반환:     (B, C_out, T)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)                               # (B, C_out, T)

        # time embedding: scale & shift
        t = self.time_proj(time_emb)                   # (B, C_out*2)
        scale, shift = t.chunk(2, dim=-1)              # each (B, C_out)
        h = h * (1.0 + scale[:, :, None]) + shift[:, :, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


# ──────────────────────────────────────────────────────
# 3. SelfAttention1D
# ──────────────────────────────────────────────────────
class SelfAttention1D(nn.Module):
    """
    Multi-head self-attention over the sequence (T) dimension.
    입력: (B, C, T)  →  (B, C, T)

    내부적으로 (B, T, C) 로 transpose 후 MHA 적용.
    """

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 8):
        super().__init__()
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.norm = nn.GroupNorm(groups, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,    # (B, T, C) 인터페이스
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        반환: (B, C, T)

        fp16 autocast 환경에서 attention score가 fp16 최대값(65504)을 초과할 수 있어
        NaN이 발생하므로, attention 연산은 항상 fp32로 강제 실행.
        """
        B, C, T = x.shape
        h = self.norm(x)                              # (B, C, T)
        h = h.transpose(1, 2)                         # (B, T, C)
        # fp32 강제: autocast가 활성화돼 있어도 attention은 fp32
        with torch.amp.autocast("cuda", enabled=False):
            h = h.float()
            h, _ = self.attn(h, h, h)                 # (B, T, C) fp32
        h = h.to(x.dtype)                             # 원래 dtype으로 복귀
        h = h.transpose(1, 2)                         # (B, C, T)
        return x + h


# ──────────────────────────────────────────────────────
# 4. Downsample / Upsample
# ──────────────────────────────────────────────────────
class Downsample1D(nn.Module):
    """stride=2 Conv1d로 시퀀스 길이를 절반으로."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """2× 선형 보간 후 Conv1d."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="linear", align_corners=False)
        return self.conv(x)


# ──────────────────────────────────────────────────────
# 5. UNet1D
# ──────────────────────────────────────────────────────
class UNet1D(nn.Module):
    """
    1D U-Net for latent diffusion.

    채널 구성 (기본값):
      in_channels=128
      channel_mults=(1, 2, 4, 8)  → [128, 256, 512, 1024]

    파라미터 수: ~45-55M
    """

    def __init__(
        self,
        in_channels: int = 128,
        base_channels: int = 128,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: tuple = (2, 3),   # level 인덱스 (깊은 쪽에 attention)
        time_emb_dim: int = 512,
        groups: int = 8,
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.num_levels = len(channel_mults)

        channels_list = [base_channels * m for m in channel_mults]  # [128,256,512,1024]

        # ── Time Embedding ──
        self.time_emb = TimeEmbedding(base_channels, time_emb_dim)

        # ── Input projection ──
        self.input_proj = nn.Conv1d(in_channels, base_channels, kernel_size=1)

        # ── Encoder ──
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        c_in = base_channels
        self.skip_channels = [c_in]   # input_proj 출력도 skip으로

        for level, c_out in enumerate(channels_list):
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(
                    ResBlock1D(c_in, c_out, time_emb_dim, groups, dropout)
                )
                if level in attn_resolutions:
                    level_blocks.append(SelfAttention1D(c_out, num_heads, groups))
                self.skip_channels.append(c_out)
                c_in = c_out
            self.enc_blocks.append(level_blocks)
            # 마지막 레벨 제외 downsample
            if level < self.num_levels - 1:
                self.enc_downs.append(Downsample1D(c_in))
                self.skip_channels.append(c_in)  # downsample 전 skip 추가
            else:
                self.enc_downs.append(nn.Identity())

        # ── Bottleneck ──
        self.bottleneck = nn.ModuleList([
            ResBlock1D(c_in, c_in, time_emb_dim, groups, dropout),
            SelfAttention1D(c_in, num_heads, groups),
            ResBlock1D(c_in, c_in, time_emb_dim, groups, dropout),
        ])

        # ── Decoder ──
        self.dec_blocks = nn.ModuleList()
        self.dec_ups = nn.ModuleList()

        for level in reversed(range(self.num_levels)):
            c_out = channels_list[level]
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                # skip connection 채널 합산
                skip_c = self.skip_channels.pop()
                level_blocks.append(
                    ResBlock1D(c_in + skip_c, c_out, time_emb_dim, groups, dropout)
                )
                if level in attn_resolutions:
                    level_blocks.append(SelfAttention1D(c_out, num_heads, groups))
                c_in = c_out
            self.dec_blocks.append(level_blocks)
            if level > 0:
                self.dec_ups.append(Upsample1D(c_in))
            else:
                self.dec_ups.append(nn.Identity())

        # ── Output projection ──
        self.out_norm = nn.GroupNorm(groups, base_channels)
        self.out_proj = nn.Conv1d(base_channels, in_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 출력 projection: 작은 Normal로 초기화.
        # 0으로 초기화하면 초기 gradient norm이 폭발하여 fp16에서 NaN 유발.
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 128, 225)  — 노이즈가 추가된 잠재 벡터
        t: (B,)            — 타임스텝 인덱스
        반환: (B, 128, 225) — 예측 노이즈 (ε-prediction)
        """
        # Time embedding
        t_emb = self.time_emb(t)                   # (B, time_emb_dim)

        # Input projection
        h = self.input_proj(x)                     # (B, base_ch, T)
        skips = [h]

        # Encoder
        # 주의: skips.append는 ResBlock 이후에만 수행.
        # SelfAttention 이후에는 push하지 않음.
        # → __init__의 skip_channels 추적과 정확히 일치.
        for level, (level_blocks, down) in enumerate(
            zip(self.enc_blocks, self.enc_downs)
        ):
            for block in level_blocks:
                if isinstance(block, ResBlock1D):
                    h = block(h, t_emb)
                    skips.append(h)          # ResBlock 출력만 push
                else:                        # SelfAttention1D
                    h = block(h)             # attention은 push 안 함

            if level < self.num_levels - 1:
                h = down(h)
                skips.append(h)              # downsample 출력 push

        # Bottleneck
        for i, block in enumerate(self.bottleneck):
            if isinstance(block, ResBlock1D):
                h = block(h, t_emb)
            else:
                h = block(h)

        # Decoder
        for level_blocks, up in zip(self.dec_blocks, self.dec_ups):
            for block in level_blocks:
                if isinstance(block, ResBlock1D):
                    skip = skips.pop()
                    # 길이 불일치 보정 (downsampling stride 때문에 ±1 발생 가능)
                    if h.shape[-1] != skip.shape[-1]:
                        h = F.interpolate(h, size=skip.shape[-1], mode="linear",
                                          align_corners=False)
                    h = torch.cat([h, skip], dim=1)
                    h = block(h, t_emb)
                else:                              # SelfAttention1D
                    h = block(h)
            h = up(h)

        # 입력 해상도로 보정
        if h.shape[-1] != x.shape[-1]:
            h = F.interpolate(h, size=x.shape[-1], mode="linear", align_corners=False)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_proj(h)
        return h

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet1D().to(device)
    print(f"Parameters: {model.num_parameters / 1e6:.1f}M")

    B = 2
    x = torch.randn(B, 128, 225, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    with torch.no_grad():
        out = model(x, t)

    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print("✓ Shape check passed")
