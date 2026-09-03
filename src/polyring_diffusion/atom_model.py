"""Radius-aware atom refinement conditioned on atom-ring membership."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RadiusAwareAtomLayer(nn.Module):
    def __init__(self, hidden_dim: int, bond_classes: int = 5) -> None:
        super().__init__()
        self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
        edge_dim = hidden_dim * 3 + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coord_gate = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        coordinates: Tensor,
        vdw_radii: Tensor,
        covalent_radii: Tensor,
        bond_types: Tensor,
        atom_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, atoms, _ = hidden.shape
        left = hidden[:, :, None, :].expand(batch, atoms, atoms, -1)
        right = hidden[:, None, :, :].expand(batch, atoms, atoms, -1)
        relative = coordinates[:, :, None, :] - coordinates[:, None, :, :]
        distance = relative.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vdw_sum = vdw_radii[:, :, None, None] + vdw_radii[:, None, :, None]
        covalent_sum = covalent_radii[:, :, None, None] + covalent_radii[:, None, :, None]
        vdw_ratio = distance / vdw_sum.clamp_min(1e-6)
        covalent_ratio = distance / covalent_sum.clamp_min(1e-6)
        overlap = torch.relu(1.0 - vdw_ratio)
        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        pair_mask &= ~torch.eye(atoms, dtype=torch.bool, device=hidden.device)[None]
        edge = self.edge_mlp(
            torch.cat(
                [
                    left,
                    right,
                    self.bond_embedding(bond_types),
                    distance,
                    vdw_ratio,
                    covalent_ratio,
                    overlap,
                ],
                dim=-1,
            )
        )
        edge = edge * pair_mask.unsqueeze(-1)
        degree = pair_mask.sum(dim=2, keepdim=True).clamp_min(1).to(hidden.dtype)
        aggregated = edge.sum(dim=2) / degree
        hidden = self.norm(hidden + self.node_mlp(torch.cat([hidden, aggregated], dim=-1)))
        hidden = hidden * atom_mask.unsqueeze(-1)
        delta = (relative * self.coord_gate(edge)).sum(dim=2) / degree
        coordinates = coordinates + delta * atom_mask.unsqueeze(-1)
        return hidden, coordinates


class RadiusAwareAtomRefiner(nn.Module):
    """Refine decoded atom coordinates without changing the ring topology."""

    def __init__(
        self,
        atom_feature_dim: int,
        ring_hidden_dim: int,
        hidden_dim: int = 192,
        num_layers: int = 4,
        bond_classes: int = 5,
    ) -> None:
        super().__init__()
        self.atom_projection = nn.Linear(atom_feature_dim + ring_hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            RadiusAwareAtomLayer(hidden_dim, bond_classes=bond_classes) for _ in range(num_layers)
        )
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        atom_features: Tensor,
        coordinates: Tensor,
        vdw_radii: Tensor,
        covalent_radii: Tensor,
        bond_types: Tensor,
        membership: Tensor,
        ring_hidden: Tensor,
        atom_mask: Tensor,
    ) -> dict[str, Tensor]:
        """membership has shape [batch, atoms, rings] and may be multi-hot."""

        weights = membership / membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ring_context = torch.bmm(weights, ring_hidden)
        hidden = self.atom_projection(torch.cat([atom_features, ring_context], dim=-1))
        initial = coordinates
        for layer in self.layers:
            hidden, coordinates = layer(
                hidden,
                coordinates,
                vdw_radii,
                covalent_radii,
                bond_types,
                atom_mask,
            )
        # Scalar gate times equivariant displacement keeps the output equivariant.
        coordinate_noise = (coordinates - initial) * self.noise_head(hidden)
        return {"coordinate_noise": coordinate_noise, "hidden": hidden}


def vdw_overlap_loss(
    clean_estimate: Tensor,
    vdw_radii: Tensor,
    bond_types: Tensor,
    atom_mask: Tensor,
    margin: float = 0.85,
) -> Tensor:
    """Late-stage, non-bonded excluded-volume penalty evaluated on x0 estimates."""

    atoms = clean_estimate.shape[1]
    distance = torch.cdist(clean_estimate, clean_estimate).clamp_min(1e-6)
    minimum = margin * (vdw_radii[:, :, None] + vdw_radii[:, None, :])
    pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
    pair_mask &= torch.triu(torch.ones(atoms, atoms, dtype=torch.bool, device=atom_mask.device), 1)
    nonbonded = bond_types.eq(0)
    penalty = torch.relu(minimum - distance).square()
    selected = penalty[pair_mask & nonbonded]
    return selected.mean() if selected.numel() else penalty.new_zeros(())
