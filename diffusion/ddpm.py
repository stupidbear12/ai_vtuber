"""
diffusion/ddpm.py
DDPM 스케줄러 — PyTorch 구현
참고: Ho et al. 2020 (DDPM), Song et al. 2020 (DDIM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# 1. Beta 스케줄
# ──────────────────────────────────────────────────────────────

def linear_beta_schedule(T: int = 1000) -> torch.Tensor:
    """
    선형 beta 스케줄.
    β₁=1e-4, βT=0.02  (Ho et al. 2020)
    Returns: betas shape (T,)
    """
    beta_start = 1e-4
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """
    Cosine 스케줄 — 더 안정적인 학습.
    Nichol & Dhariwal 2021 (Improved DDPM)
    Returns: betas shape (T,)
    """
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_bar = torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clamp(betas, 0.0, 0.999)


# ──────────────────────────────────────────────────────────────
# 2. 유틸: 인덱싱 헬퍼
# ──────────────────────────────────────────────────────────────

def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
    """
    인덱스 t에 해당하는 스케줄 값을 추출 후 x_shape에 맞게 broadcast.
    Args:
        a: (T,) 스케줄 배열
        t: (batch,) 타임스텝
        x_shape: 타겟 텐서 shape (batch, C, H, W)
    Returns:
        (batch, 1, 1, 1) broadcast 가능한 형태
    """
    batch = t.shape[0]
    out = a.gather(0, t)                              # (batch,)
    return out.reshape((batch,) + (1,) * (len(x_shape) - 1))


# ──────────────────────────────────────────────────────────────
# 3. DDPMScheduler 클래스
# ──────────────────────────────────────────────────────────────

class DDPMScheduler:
    """
    DDPM 정방향/역방향 샘플링 + DDIM 빠른 생성.

    Usage:
        scheduler = DDPMScheduler(T=1000, schedule='cosine')
        # 학습
        loss = scheduler.p_losses(model, x0, t)
        # 생성
        samples = scheduler.p_sample_loop(model, shape, device)
    """

    def __init__(
        self,
        T: int = 1000,
        schedule: str = 'cosine',  # 'linear' | 'cosine'
        loss_type: str = 'l2',     # 'l2' | 'l1'
    ):
        self.T = T
        self.loss_type = loss_type

        if schedule == 'linear':
            betas = linear_beta_schedule(T)
        elif schedule == 'cosine':
            betas = cosine_beta_schedule(T)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        # 사전계산 상수들 (버퍼로 등록하지 않고 직접 저장; device는 동적으로 처리)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = torch.cat([torch.tensor([1.0]), alphas_bar[:-1]])

        self.register = {
            'betas': betas,
            'alphas': alphas,
            'alphas_bar': alphas_bar,
            'alphas_bar_prev': alphas_bar_prev,
            'sqrt_alphas_bar': torch.sqrt(alphas_bar),
            'sqrt_one_minus_alphas_bar': torch.sqrt(1.0 - alphas_bar),
            'sqrt_recip_alphas': torch.sqrt(1.0 / alphas),
            'posterior_variance': betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar),
            'posterior_log_variance': torch.log(
                torch.clamp(betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar), min=1e-20)
            ),
            'posterior_mean_coef1': betas * torch.sqrt(alphas_bar_prev) / (1.0 - alphas_bar),
            'posterior_mean_coef2': (1.0 - alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - alphas_bar),
        }

    def _get(self, key: str, device: torch.device) -> torch.Tensor:
        """스케줄 상수를 해당 device로 이동."""
        return self.register[key].to(device)

    # ────────────────────────────
    # 3-1. Forward (정방향) 노이즈 추가
    # ────────────────────────────
    def q_sample(
        self,
        x0: torch.Tensor,      # (batch, 1, H, W) 원본 스펙트로그램
        t: torch.Tensor,       # (batch,) 타임스텝
        noise: torch.Tensor,   # (batch, 1, H, W) 가우시안 노이즈
    ) -> torch.Tensor:
        """
        x_t = sqrt(ᾱ_t)*x0 + sqrt(1-ᾱ_t)*ε
        """
        device = x0.device
        sqrt_ab = _extract(self._get('sqrt_alphas_bar', device), t, x0.shape)
        sqrt_1m_ab = _extract(self._get('sqrt_one_minus_alphas_bar', device), t, x0.shape)
        return sqrt_ab * x0 + sqrt_1m_ab * noise

    # ────────────────────────────
    # 3-2. 학습 Loss 계산
    # ────────────────────────────
    def p_losses(
        self,
        model: nn.Module,
        x0: torch.Tensor,   # (batch, 1, H, W)
        t: torch.Tensor,    # (batch,)
    ) -> torch.Tensor:
        """
        단순 MSE: L = ||ε - ε_θ(x_t, t)||²
        Returns: scalar loss
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        noise_pred = model(x_t, t)

        if self.loss_type == 'l2':
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        return loss

    # ────────────────────────────
    # 3-3. 역방향 한 스텝 (p_sample)
    # ────────────────────────────
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,   # (batch, 1, H, W) 현재 노이즈 샘플
        t: torch.Tensor,     # (batch,) 타임스텝
    ) -> torch.Tensor:
        """
        DDPM 역방향 한 스텝: x_{t-1} ~ p_θ(x_{t-1}|x_t)
        """
        device = x_t.device
        batch = x_t.shape[0]

        noise_pred = model(x_t, t)

        # x0 예측
        sqrt_ab = _extract(self._get('sqrt_alphas_bar', device), t, x_t.shape)
        sqrt_1m_ab = _extract(self._get('sqrt_one_minus_alphas_bar', device), t, x_t.shape)
        x0_pred = (x_t - sqrt_1m_ab * noise_pred) / sqrt_ab
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        # 사후 평균
        coef1 = _extract(self._get('posterior_mean_coef1', device), t, x_t.shape)
        coef2 = _extract(self._get('posterior_mean_coef2', device), t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        # 사후 분산 (t=0이면 노이즈 없음)
        log_var = _extract(self._get('posterior_log_variance', device), t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(batch, *([1] * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    # ────────────────────────────
    # 3-4. 전체 생성 루프 (T → 0)
    # ────────────────────────────
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple,         # (batch, 1, H, W)
        device: torch.device,
    ) -> torch.Tensor:
        """
        순수 DDPM 생성: T 스텝 역방향 반복.
        Returns: x_0 (batch, 1, H, W)
        """
        x = torch.randn(shape, device=device)
        batch = shape[0]

        for i in reversed(range(self.T)):
            t_batch = torch.full((batch,), i, dtype=torch.long, device=device)
            x = self.p_sample(model, x, t_batch)
        return x

    # ────────────────────────────
    # 3-5. DDIM 빠른 생성
    # ────────────────────────────
    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple,
        device: torch.device,
        steps: int = 50,
        eta: float = 0.0,   # 0=결정론적, 1=DDPM과 동일
    ) -> torch.Tensor:
        """
        DDIM 빠른 생성 (Song et al. 2020).
        steps=50으로 T=1000 대비 20배 빠른 생성.
        Returns: x_0 (batch, 1, H, W)
        """
        step_ratio = self.T // steps
        timesteps = list(reversed(range(0, self.T, step_ratio)))  # T→0

        x = torch.randn(shape, device=device)
        batch = shape[0]
        alphas_bar = self._get('alphas_bar', device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((batch,), t_cur, dtype=torch.long, device=device)
            noise_pred = model(x, t_batch)

            ab_t = alphas_bar[t_cur]
            ab_tm1 = alphas_bar[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

            # x0 예측
            x0_pred = (x - torch.sqrt(1.0 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            # DDIM 업데이트
            sigma = (
                eta
                * torch.sqrt((1 - ab_tm1) / (1 - ab_t))
                * torch.sqrt(1 - ab_t / ab_tm1)
            )
            direction = torch.sqrt(1.0 - ab_tm1 - sigma ** 2) * noise_pred
            noise = torch.randn_like(x)

            x = torch.sqrt(ab_tm1) * x0_pred + direction + sigma * noise

        return x


# ──────────────────────────────────────────────────────────────
# 간단한 동작 테스트
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sched = DDPMScheduler(T=1000, schedule='cosine')

    x0 = torch.randn(2, 1, 128, 256)
    t = torch.tensor([0, 999])
    noise = torch.randn_like(x0)

    xt = sched.q_sample(x0, t, noise)
    print(f"q_sample output shape: {xt.shape}")
    print(f"betas[:5]:  {sched.register['betas'][:5]}")
    print(f"betas[-5:]: {sched.register['betas'][-5:]}")
    print("DDPMScheduler OK")
