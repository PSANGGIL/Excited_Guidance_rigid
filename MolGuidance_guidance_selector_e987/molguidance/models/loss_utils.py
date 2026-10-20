"""Explicit categorical estimators and a released-objective compatibility mode."""

from __future__ import annotations

import torch


CATEGORICAL_REDUCTIONS = {
    "legacy_factorized", "aligned_all", "masked_weighted", "masked_unweighted"
}


def reduce_categorical_loss(loss, weight, active, mode="masked_weighted"):
    """Explicit categorical estimators; never infer masks from zero losses.

    masked_weighted = sum(active * weight * CE) / count(active), NOT / sum(weight).
    aligned_all includes ignored entries in the denominator. The legacy mode
    intentionally preserves the released outer-product objective for ablation.
    Empty masks return a differentiable zero, including at clean graph time.
    """
    if mode not in CATEGORICAL_REDUCTIONS:
        raise ValueError(f"unknown categorical reduction: {mode}")
    if loss.ndim != 1 or active.shape != loss.shape or active.dtype != torch.bool:
        raise ValueError("loss and active must be aligned vectors")
    weight = torch.as_tensor(weight, device=loss.device, dtype=loss.dtype)
    if weight.ndim == 0:
        weight = weight.expand_as(loss)
    elif weight.shape == (loss.numel(), 1):
        weight = weight[:, 0]
    if weight.shape != loss.shape:
        raise ValueError("categorical weights must align with loss entries")
    if not torch.isfinite(weight).all() or bool((weight < 0).any()):
        raise ValueError("categorical weights must be finite and non-negative")
    masked = torch.where(active, loss, torch.zeros_like(loss))
    if mode == "legacy_factorized":
        return masked.sum() / max(loss.numel(), 1) * (
            weight.sum() / max(loss.numel(), 1)
        )
    denominator = (
        loss.new_tensor(max(loss.numel(), 1))
        if mode == "aligned_all" else active.sum().clamp_min(1)
    )
    if mode == "masked_unweighted":
        return masked.sum() / denominator
    return (masked * weight).sum() / denominator


def reduce_time_scaled_loss_equivalent(
    unreduced_loss: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Reduce a time-scaled loss without materializing outer products.

    The released code broadcasts a categorical loss of shape [N] against a
    weight of shape [N, 1], then averages the resulting [N, N] tensor. Its
    scalar mean is exactly mean(loss) * mean(weight). Coordinate/vector losses
    shaped [N, D] retain their original row-wise weighted reduction.

    Ignored CTMC targets remain zero entries in unreduced_loss and therefore
    remain part of the released implementation's denominator.
    """

    if unreduced_loss.ndim < 1:
        raise ValueError("unreduced_loss must have a leading sample dimension")
    if sample_weight.ndim < 1:
        raise ValueError("sample_weight must have a leading sample dimension")
    if unreduced_loss.shape[0] != sample_weight.shape[0]:
        raise ValueError(
            "sample_weight and unreduced_loss must have the same leading "
            f"dimension, got {sample_weight.shape[0]} and "
            f"{unreduced_loss.shape[0]}"
        )

    if (
        unreduced_loss.ndim == 1
        and sample_weight.ndim == 2
        and sample_weight.shape[1] == 1
    ):
        return unreduced_loss.mean() * sample_weight.mean()

    return (unreduced_loss * sample_weight).mean()
