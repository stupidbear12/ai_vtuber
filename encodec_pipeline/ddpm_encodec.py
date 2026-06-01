"""
ddpm_encodec.py

DDPM 스케줄러 — EnCodec 잠재 공간(1D) 전용.

제공:
  linear_beta_schedule(T)           — 선형 beta 스케줄
  cosine_beta_schedule(T, s)        — cosine beta 스케줄 (AudioLDM2 권장)
  DDPMScheduler                     — forward / reverse / DDIM 샘플링

참고:
  - Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
  - Nichol & Dhariwal (2021) "Improved Denoising Diffusion Probabilistic Models"
  - Liu et al. (2023) "AudioLDM 2"
  - Song et al. (2021) "Denoising Diffusion Implicit Models"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ──────────────────────────────────────────────
# Beta 스케줄
# ──────────────────────────────────────────────

def linear_beta_schedule(T: int = 1000,
                          beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> torch.Tensor:
    """
    Ho et al. (2020) 선형 스케줄.
    반환: beta tensor shape (T,)
    """
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """
    Nichol & Dhariwal (2021) cosine 스케줄.
    AudioLDM2에서 권장하는 스케줄.

    alpha_bar(t) = cos²( (t/T + s) / (1 + s) * π/2 )

    반환: beta tensor shape (T,)
    """
    steps = T + 1
    t = torch.linspace(0, T, steps)
    alpha_bar = torch.cos(((t / T) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]          # 정규화 (t=0 에서 1)
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = betas.clamp(max=0.999)                # 수치 안정성
    return betas


# ──────────────────────────────────────────────
# DDPMScheduler
# ──────────────────────────────────────────────

class DDPMScheduler(nn.Module):
    """
    DDPM 노이즈 스케줄러.

    사용:
        scheduler = DDPMScheduler(T=1000, schedule="cosine").to(device)

        # 학습: forward process loss
        loss = scheduler.p_losses(model, x0, t)

        # 생성: 전체 역방향 샘플링
        sample = scheduler.p_sample_loop(model, shape=(1, 128, 225), device=device)

        # 빠른 생성: DDIM
        sample = scheduler.ddim_sample(model, shape=(1, 128, 225), device=device, steps=50)
    """

    def __init__(
        self,
        T: int = 1000,
        schedule: str = "cosine",    # "linear" | "cosine"
        loss_type: str = "mse",      # "mse" | "l1"
        predict: str = "noise",      # "noise" | "x0"  — 모델 예측 대상
    ):
        super().__init__()
        self.T = T
        self.loss_type = loss_type
        self.predict = predict

        # Beta 스케줄 계산
        if schedule == "cosine":
            betas = cosine_beta_schedule(T)
        elif schedule == "linear":
            betas = linear_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # ── 상수 등록 (버퍼) ──
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # forward q(x_t | x_0)
        self.register_buffer("sqrt_alphas_cumprod",
                             alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             (1.0 - alphas_cumprod).sqrt())

        # reverse p(x_{t-1} | x_t) 의 posterior variance
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             posterior_variance.clamp(min=1e-20).log())

        # reverse mean 계산 계수
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod),
        )

    # ── 유틸 ──────────────────────────────────

    def _extract(self, a: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """
        a: (T,) 1D 버퍼
        t: (B,) int 타임스텝
        shape: 타겟 shape (B, C, L)
        → (B, 1, 1) 로 broadcast 가능하게 reshape
        """
        B = t.shape[0]
        out = a.gather(-1, t)              # (B,)
        return out.reshape(B, *([1] * (len(shape) - 1)))

    # ── Forward Process q(x_t | x_0) ─────────

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x0: (B, C, L) — 원본 잠재 벡터
        t:  (B,)       — 타임스텝
        반환: x_t (B, C, L)

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise

    # ── Training Loss p_losses ────────────────

    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        model: UNet1D
        x0:   (B, C, L) 정규화된 잠재 벡터
        t:    (B,)
        반환: scalar loss

        ε-prediction: 모델이 노이즈를 예측하도록 학습
        """
        if noise is None:
            noise = torch.randn_like(x0)

        x_noisy = self.q_sample(x0, t, noise)    # (B, C, L)
        pred = model(x_noisy, t)                  # (B, C, L)

        if self.predict == "noise":
            target = noise
        else:  # "x0"
            target = x0

        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target)
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

    # ── Reverse Step p(x_{t-1} | x_t) ───────

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        DDPM 1스텝 역방향 샘플링.
        x_t: (B, C, L)
        t:   (B,)
        반환: x_{t-1}
        """
        pred = model(x_t, t)                       # 노이즈 예측

        # x0 복원
        if self.predict == "noise":
            sqrt_recip = self._extract(
                1.0 / self.sqrt_alphas_cumprod, t, x_t.shape
            )
            sqrt_recip_m1 = self._extract(
                (1.0 / self.alphas_cumprod - 1).sqrt(), t, x_t.shape
            )
            x0_pred = sqrt_recip * x_t - sqrt_recip_m1 * pred
        else:
            x0_pred = pred

        x0_pred = x0_pred.clamp(-3.0, 3.0)        # 수치 안정성

        # posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        # variance
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # t=0 일 때 노이즈 없음
        nonzero_mask = (t > 0).float().reshape(-1, *([1] * (x_t.ndim - 1)))
        noise = torch.randn_like(x_t)
        return mean + nonzero_mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device | str,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        DDPM 전체 역방향 샘플링 (T → 0).
        shape: (B, C, L)
        반환: (B, C, L)  — 생성된 잠재 벡터
        """
        model.eval()
        x = torch.randn(*shape, device=device)
        B = shape[0]

        timesteps = reversed(range(self.T))
        if show_progress:
            timesteps = tqdm(list(timesteps), desc="DDPM sampling")

        for i in timesteps:
            t = torch.full((B,), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t)

        return x

    # ── DDIM Sampling ─────────────────────────

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device | str,
        steps: int = 50,
        eta: float = 0.0,     # 0 = deterministic, 1 = DDPM
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        DDIM 빠른 샘플링 (Song et al., 2021).

        steps: 추론 스텝 수 (50 권장)
        eta:   확률성 파라미터 (0=결정론적, 1=DDPM과 동일)
        반환: (B, C, L)
        """
        model.eval()
        B = shape[0]

        # 균등 간격 타임스텝 서브시퀀스
        step_ratio = self.T // steps
        timesteps = list(reversed(range(0, self.T, step_ratio)))[:steps]

        x = torch.randn(*shape, device=device)

        if show_progress:
            timesteps_iter = tqdm(timesteps, desc="DDIM sampling")
        else:
            timesteps_iter = timesteps

        for i, t_val in enumerate(timesteps_iter):
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # 노이즈 예측
            eps_pred = model(x, t)

            # ᾱ_t, ᾱ_{t-1}
            alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
            if i + 1 < len(timesteps):
                t_prev_val = timesteps[i + 1]
                t_prev = torch.full((B,), t_prev_val, device=device, dtype=torch.long)
                alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            # x0 예측
            x0_pred = (
                x - (1.0 - alpha_bar_t).sqrt() * eps_pred
            ) / alpha_bar_t.sqrt()
            x0_pred = x0_pred.clamp(-3.0, 3.0)

            # DDIM 방향
            dir_xt = (1.0 - alpha_bar_prev - eta ** 2 * (
                (1.0 - alpha_bar_t) / alpha_bar_t *
                (1.0 - alpha_bar_prev / alpha_bar_t)
            ).clamp(min=0.0)).sqrt() * eps_pred

            noise = eta * (
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) *
                (1.0 - alpha_bar_t / alpha_bar_prev)
            ).clamp(min=0.0).sqrt() * torch.randn_like(x)

            x = alpha_bar_prev.sqrt() * x0_pred + dir_xt + noise

        return x


# ──────────────────────────────────────────────
# 빠른 테스트
# ──────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 더미 모델 (identity)
    class DummyModel(nn.Module):
        def forward(self, x, t):
            return torch.zeros_like(x)

    scheduler = DDPMScheduler(T=1000, schedule="cosine").to(device)
    model = DummyModel().to(device)

    B, C, L = 2, 128, 225
    x0 = torch.randn(B, C, L, device=device)
    t = torch.randint(0, 1000, (B,), device=device)

    # forward process
    xt = scheduler.q_sample(x0, t)
    print(f"q_sample output shape: {xt.shape}")

    # loss
    loss = scheduler.p_losses(model, x0, t)
    print(f"p_losses: {loss.item():.4f}")

    # DDIM (dummy → 노이즈 그대로)
    sample = scheduler.ddim_sample(
        model, shape=(1, C, L), device=device, steps=10, show_progress=False
    )
    print(f"ddim_sample output shape: {sample.shape}")
    print("✓ DDPMScheduler test passed")
