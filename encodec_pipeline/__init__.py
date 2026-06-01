"""
EnCodec + Latent Diffusion pipeline for lofi music generation.

References:
  - "High Fidelity Neural Audio Compression" (Défossez et al., 2022)
  - "AudioLDM 2" (Liu et al., 2023)
"""

from .unet_1d import UNet1D, TimeEmbedding, ResBlock1D, SelfAttention1D
from .ddpm_encodec import DDPMScheduler, linear_beta_schedule, cosine_beta_schedule

__all__ = [
    "UNet1D",
    "TimeEmbedding",
    "ResBlock1D",
    "SelfAttention1D",
    "DDPMScheduler",
    "linear_beta_schedule",
    "cosine_beta_schedule",
]
