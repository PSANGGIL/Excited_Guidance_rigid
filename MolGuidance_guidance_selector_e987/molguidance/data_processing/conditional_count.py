"""Condition-aware heavy-atom count sampling.

The generator historically sampled molecular size from the unconditional
training histogram even when a property target was supplied.  That changes the
joint distribution from ``p(n, c)`` to ``p(n) p(c)``.  This module implements a
small, non-parametric conditional sampler that preserves the empirical
``p(n | c)`` relationship without adding checkpoint parameters.

The implementation accepts vector-valued conditions so the same code remains
valid when the model moves from scalar EST conditioning to S1/EST/oscillator
strength conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


def _as_feature_matrix(values: torch.Tensor | Sequence[float]) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device="cpu")
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    elif tensor.ndim != 2:
        raise ValueError(
            f"conditioning values must be scalar, 1-D, or 2-D; got {tensor.shape}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("conditioning values contain NaN or infinity")
    return tensor


@dataclass(frozen=True)
class ConditionalCountDiagnostics:
    """Diagnostics for one conditional-count sampling call."""

    effective_sample_size: torch.Tensor
    nearest_distance: torch.Tensor


class ConditionalAtomCountSampler:
    """Kernel sampler for the empirical distribution ``p(n_atoms | property)``."""

    def __init__(
        self,
        atom_counts: torch.Tensor | Sequence[int],
        properties: torch.Tensor | Sequence[float],
    ) -> None:
        counts = torch.as_tensor(atom_counts, dtype=torch.long, device="cpu")
        property_matrix = _as_feature_matrix(properties)
        if counts.ndim != 1:
            raise ValueError(f"atom_counts must be 1-D; got {counts.shape}")
        if counts.numel() != property_matrix.shape[0]:
            raise ValueError(
                "atom count and property references have different lengths: "
                f"{counts.numel()} != {property_matrix.shape[0]}"
            )
        if counts.numel() == 0:
            raise ValueError("conditional atom-count reference is empty")
        if (counts <= 0).any():
            raise ValueError("atom counts must be positive")

        self.atom_counts = counts.contiguous()
        self.properties = property_matrix.contiguous()

    @classmethod
    def from_processed_file(cls, path: str | Path) -> "ConditionalAtomCountSampler":
        """Build the reference from a processed MolGuidance split.

        Counts are derived from ``node_idx_array`` rather than from a separate
        histogram so every property label remains paired with its own molecule.
        """

        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        required = {"node_idx_array", "properties"}
        missing = required.difference(data)
        if missing:
            raise ValueError(
                f"processed count reference {path} is missing keys {sorted(missing)}"
            )
        node_ranges = torch.as_tensor(data["node_idx_array"], dtype=torch.long)
        if node_ranges.ndim != 2 or node_ranges.shape[1] != 2:
            raise ValueError("node_idx_array must have shape [num_molecules, 2]")
        counts = node_ranges[:, 1] - node_ranges[:, 0]
        return cls(counts, data["properties"])

    @property
    def condition_dim(self) -> int:
        return int(self.properties.shape[1])

    def _bandwidth_vector(
        self, bandwidth: float | Sequence[float] | torch.Tensor
    ) -> torch.Tensor:
        bandwidth_tensor = torch.as_tensor(
            bandwidth, dtype=self.properties.dtype, device="cpu"
        ).reshape(-1)
        if bandwidth_tensor.numel() == 1:
            bandwidth_tensor = bandwidth_tensor.expand(self.condition_dim)
        if bandwidth_tensor.numel() != self.condition_dim:
            raise ValueError(
                "bandwidth dimension does not match conditioning dimension: "
                f"{bandwidth_tensor.numel()} != {self.condition_dim}"
            )
        if not torch.isfinite(bandwidth_tensor).all() or (
            bandwidth_tensor <= 0
        ).any():
            raise ValueError("all conditional-count bandwidths must be finite and > 0")
        return bandwidth_tensor

    def probabilities(
        self,
        targets: torch.Tensor | Sequence[float],
        bandwidth: float | Sequence[float] | torch.Tensor,
        *,
        nearest_k: int | None = None,
    ) -> tuple[torch.Tensor, ConditionalCountDiagnostics]:
        """Return row-normalized kernel weights over reference molecules."""

        target_matrix = _as_feature_matrix(targets)
        if target_matrix.shape[1] != self.condition_dim:
            raise ValueError(
                "target conditioning dimension does not match the reference: "
                f"{target_matrix.shape[1]} != {self.condition_dim}"
            )
        bandwidth_vector = self._bandwidth_vector(bandwidth)
        standardized = (
            target_matrix[:, None, :] - self.properties[None, :, :]
        ) / bandwidth_vector[None, None, :]
        squared_distance = standardized.square().sum(dim=-1)

        # Subtracting the best log weight keeps even out-of-support targets
        # numerically well-defined.  An optional nearest-k support prevents a
        # broad kernel from collapsing back to the global marginal.
        log_weights = -0.5 * squared_distance
        if nearest_k is not None:
            if nearest_k <= 0:
                raise ValueError("nearest_k must be positive when specified")
            k = min(int(nearest_k), self.atom_counts.numel())
            support = torch.zeros_like(log_weights, dtype=torch.bool)
            nearest = squared_distance.topk(k=k, dim=1, largest=False).indices
            support.scatter_(1, nearest, True)
            log_weights = log_weights.masked_fill(~support, -torch.inf)

        log_weights = log_weights - log_weights.max(dim=1, keepdim=True).values
        weights = torch.exp(log_weights)
        probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).tiny
        )
        diagnostics = ConditionalCountDiagnostics(
            effective_sample_size=probabilities.square().sum(dim=1).reciprocal(),
            nearest_distance=squared_distance.min(dim=1).values.sqrt(),
        )
        return probabilities, diagnostics

    def sample(
        self,
        targets: torch.Tensor | Sequence[float],
        bandwidth: float | Sequence[float] | torch.Tensor,
        *,
        nearest_k: int | None = None,
        generator: torch.Generator | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ConditionalCountDiagnostics]:
        probabilities, diagnostics = self.probabilities(
            targets, bandwidth, nearest_k=nearest_k
        )
        reference_indices = torch.multinomial(
            probabilities, num_samples=1, replacement=True, generator=generator
        ).squeeze(1)
        samples = self.atom_counts[reference_indices]
        if return_diagnostics:
            return samples, diagnostics
        return samples
