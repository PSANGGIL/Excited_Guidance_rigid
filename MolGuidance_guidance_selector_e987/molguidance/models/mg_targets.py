"""Detached probability-space guidance distillation for absorbing-mask CTMC.

This is an explicit discrete extension, not the continuous MG theorem:
q_g = softmax((1-g) log p_u + g log p_c);
q_target = (1-min(|g-1|,1)) onehot(y) + min(|g-1|,1) q_g.
At g=1 it is exactly supervised CE. q_target is nonnegative and normalized,
so the absorbing reverse off-diagonal rates alpha'/(1-alpha) q_target are valid.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def soft_guided_target(clean, conditional_logits, unconditional_logits, scale,
                       support=None):
    if clean.shape != conditional_logits.shape or clean.shape != unconditional_logits.shape:
        raise ValueError("MG teacher and clean target shapes must match")
    if not all(torch.isfinite(x).all() for x in (clean, conditional_logits, unconditional_logits)):
        raise ValueError("nonfinite MG target/teacher")
    if (clean < 0).any() or not torch.allclose(clean.sum(-1), torch.ones_like(clean[:, 0])):
        raise ValueError("clean targets must be distributions")
    scale = torch.as_tensor(scale, device=clean.device, dtype=clean.dtype).reshape(-1, 1)
    if scale.numel() not in (1, clean.shape[0]) or not torch.isfinite(scale).all() or (scale < 0).any():
        raise ValueError("invalid MG scale")
    if support is None:
        support = torch.ones(clean.shape[-1], device=clean.device, dtype=torch.bool)
    support = support.to(device=clean.device, dtype=torch.bool)
    if not support.any():
        raise ValueError("empty MG support")
    # Mask after mixing finite log probabilities: 0 * -inf would create NaN.
    lc, lu = F.log_softmax(conditional_logits, -1), F.log_softmax(unconditional_logits, -1)
    guided = F.softmax((lu + scale * (lc - lu)).masked_fill(~support, -torch.inf), -1)
    mixing = (scale - 1.).abs().clamp(max=1.)
    target = (1. - mixing) * clean + mixing * guided
    # Unsupported clean labels are masked by the caller, not silently remapped.
    return target
