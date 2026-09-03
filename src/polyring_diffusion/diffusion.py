"""Continuous coordinate and masked categorical diffusion utilities."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _extract(values: Tensor, timesteps: Tensor, target: Tensor) -> Tensor:
    shape = (timesteps.shape[0],) + (1,) * (target.ndim - 1)
    return values.gather(0, timesteps).reshape(shape)


class LinearNoiseSchedule(nn.Module):
    def __init__(self, timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        super().__init__()
        if timesteps < 2:
            raise ValueError("timesteps must be at least 2")
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        previous = torch.cat([torch.ones(1), alpha_bars[:-1]])
        posterior_variance = betas * (1.0 - previous) / (1.0 - alpha_bars)
        self.timesteps = timesteps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))

    def q_sample(self, clean: Tensor, timesteps: Tensor, noise: Tensor | None = None) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(clean) if noise is None else noise
        alpha_bar = _extract(self.alpha_bars, timesteps, clean)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        return noisy, noise

    def corrupt_tokens(
        self,
        clean: Tensor,
        timesteps: Tensor,
        mask_token: int,
        valid_mask: Tensor,
    ) -> Tensor:
        """Absorbing-mask corruption with the continuous schedule's alpha_bar."""

        alpha_bar = _extract(self.alpha_bars, timesteps, clean)
        keep = torch.rand(clean.shape, device=clean.device) < alpha_bar
        return torch.where(valid_mask & ~keep, torch.full_like(clean, mask_token), clean)

    @torch.no_grad()
    def p_step(self, noisy: Tensor, predicted_noise: Tensor, timesteps: Tensor) -> Tensor:
        beta = _extract(self.betas, timesteps, noisy)
        alpha = _extract(self.alphas, timesteps, noisy)
        alpha_bar = _extract(self.alpha_bars, timesteps, noisy)
        mean = (noisy - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
        variance = _extract(self.posterior_variance, timesteps, noisy)
        nonzero = (timesteps > 0).float().reshape((timesteps.shape[0],) + (1,) * (noisy.ndim - 1))
        return mean + nonzero * variance.sqrt() * torch.randn_like(noisy)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension % 2:
            raise ValueError("time embedding dimension must be even")
        self.dimension = dimension

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.dimension // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device))
        angles = timesteps.float()[:, None] * frequencies[None]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


def center_coordinates(coordinates: Tensor, node_mask: Tensor) -> Tensor:
    weights = node_mask.unsqueeze(-1).to(coordinates.dtype)
    center = (coordinates * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (coordinates - center) * weights


def normalize_vectors(vectors: Tensor, node_mask: Tensor) -> Tensor:
    normalized = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return normalized * node_mask.unsqueeze(-1).to(vectors.dtype)

