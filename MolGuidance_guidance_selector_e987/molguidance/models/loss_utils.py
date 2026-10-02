"""Loss reductions that preserve the released model's scalar objective."""

from __future__ import annotations

import torch


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
