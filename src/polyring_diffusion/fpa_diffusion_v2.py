"""Consistent continuous, absorbing-discrete and SO(3) diffusion utilities."""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn


def extract(values: Tensor, t: Tensor, ndim: int) -> Tensor:
    return values[t].reshape((len(t),) + (1,) * (ndim - 1))


def masked_center(x: Tensor, mask: Tensor) -> Tensor:
    w = mask.unsqueeze(-1).to(x.dtype)
    return (x - (x * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp_min(1)) * w


def zero_center_noise(x: Tensor, mask: Tensor) -> Tensor:
    return masked_center(torch.randn_like(x), mask)


def skew(v: Tensor) -> Tensor:
    z = torch.zeros_like(v[..., 0]); x, y, q = v.unbind(-1)
    return torch.stack((z, -q, y, q, z, -x, -y, x, z), -1).reshape(v.shape[:-1] + (3, 3))


def so3_exp(v: Tensor) -> Tensor:
    theta = v.norm(dim=-1, keepdim=True).clamp_min(1e-8); k = skew(v / theta)
    eye = torch.eye(3, device=v.device, dtype=v.dtype).expand(v.shape[:-1] + (3, 3))
    return eye + torch.sin(theta)[..., None] * k + (1 - torch.cos(theta))[..., None] * (k @ k)


def so3_log(r: Tensor) -> Tensor:
    cos = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos); vee = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1)
    return vee * (theta / (2 * torch.sin(theta).clamp_min(1e-6))).unsqueeze(-1)


class FPADiffusion(nn.Module):
    """VP coordinate diffusion, Lie-algebra frame noise and absorbing tokens."""
    def __init__(self, timesteps=500, beta_start=1e-4, beta_end=.02, rotation_sigma_max=1.5):
        super().__init__(); beta = torch.linspace(beta_start, beta_end, timesteps); alpha = 1 - beta; abar = alpha.cumprod(0)
        self.timesteps = timesteps
        for name, value in {'beta': beta, 'alpha': alpha, 'abar': abar, 'abar_prev': torch.cat((torch.ones(1), abar[:-1])), 'rot_sigma': torch.linspace(1e-3, rotation_sigma_max, timesteps)}.items(): self.register_buffer(name, value)

    def q_coordinates(self, clean: Tensor, t: Tensor, mask: Tensor, reference_mask: Tensor | None = None):
        noise = zero_center_noise(clean, reference_mask if reference_mask is not None else mask)
        a = extract(self.abar, t, clean.ndim); noisy = (a.sqrt() * clean + (1 - a).sqrt() * noise) * mask.unsqueeze(-1)
        return noisy, noise

    def predict_x0(self, noisy: Tensor, eps: Tensor, t: Tensor) -> Tensor:
        a = extract(self.abar, t, noisy.ndim); return (noisy - (1 - a).sqrt() * eps) / a.sqrt().clamp_min(1e-6)

    def q_tokens(self, clean: Tensor, t: Tensor, mask_token: int, valid: Tensor):
        keep = torch.rand_like(clean.float()) < extract(self.abar, t, clean.ndim)
        return torch.where(valid & ~keep, torch.full_like(clean, mask_token), clean)

    @torch.no_grad()
    def p_tokens(self, noisy: Tensor, x0_logits: Tensor, t: Tensor, mask_token: int, valid: Tensor):
        sample = torch.distributions.Categorical(logits=x0_logits).sample()
        at = extract(self.abar, t, noisy.ndim); prev = extract(self.abar_prev, t, noisy.ndim)
        reveal = ((prev - at) / (1 - at).clamp_min(1e-8)).clamp(0, 1)
        choose = (torch.rand_like(noisy.float()) < reveal) & noisy.eq(mask_token) & valid
        return torch.where(choose, sample, noisy)

    def q_frames(self, clean: Tensor, t: Tensor, mask: Tensor):
        omega = torch.randn(clean.shape[:-2] + (3,), device=clean.device, dtype=clean.dtype) * mask.unsqueeze(-1)
        sigma = extract(self.rot_sigma, t, omega.ndim)
        return clean @ so3_exp(sigma * omega), omega

    @torch.no_grad()
    def p_coordinates(self, noisy: Tensor, eps: Tensor, t: Tensor, mask: Tensor,
                      score_clip: float | None = None, state_clip: float | None = None):
        # Keep reverse geometry in fp32: an overflowing score otherwise
        # contaminates every categorical head on the next denoiser call.
        original_dtype = noisy.dtype
        noisy, eps = noisy.float(), eps.float()
        if score_clip is not None:
            norm = eps.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            eps = eps * (float(score_clip) / norm).clamp_max(1.)
        beta=extract(self.beta,t,noisy.ndim);alpha=extract(self.alpha,t,noisy.ndim);abar=extract(self.abar,t,noisy.ndim)
        mean=(noisy-beta*eps/(1-abar).sqrt().clamp_min(1e-6))/alpha.sqrt();var=beta*(1-extract(self.abar_prev,t,noisy.ndim))/(1-abar).clamp_min(1e-8)
        nz=(t>0).reshape((len(t),)+(1,)*(noisy.ndim-1))
        out = mean+nz*var.sqrt()*torch.randn_like(noisy)
        if state_clip is not None:
            out = out.clamp(-float(state_clip), float(state_clip))
        return (out*mask.unsqueeze(-1)).to(original_dtype)

    @torch.no_grad()
    def p_frames(self, noisy: Tensor, omega: Tensor, t: Tensor, mask: Tensor):
        sigma = extract(self.rot_sigma, t, omega.ndim); prev = extract(torch.cat((torch.zeros(1,device=t.device),self.rot_sigma[:-1])), t, omega.ndim)
        step = (sigma - prev).clamp_min(0) * omega
        out = noisy @ so3_exp(-step); eye=torch.eye(3,device=out.device,dtype=out.dtype)
        return torch.where(mask[...,None,None],out,eye)
