"""Diffusion model whose state is heavy atoms plus fused-aromatic ring cells.

Unlike the legacy rigid-ring model, this model never replaces atoms by a ring
template.  It jointly denoises unique heavy atoms and auxiliary, variably-sized
FPA cells.  Cell fusion is a general graph with explicit directional ports.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .diffusion import SinusoidalTimeEmbedding


class SizeAwareEntityLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_classes: int = 10):
        super().__init__()
        self.relation = nn.Embedding(relation_classes, hidden_dim)
        # source, target, relation, normalized center distance, surface gap,
        # radius ratio, normal dot, tangent dot
        self.message = nn.Sequential(nn.Linear(3 * hidden_dim + 5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.update = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.coord_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.normal_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.tangent_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden, positions, normals, tangents, radii, relations, mask):
        count = hidden.shape[1]
        left = hidden[:, :, None, :].expand(-1, count, count, -1)
        right = hidden[:, None, :, :].expand(-1, count, count, -1)
        pair_mask = mask[:, :, None] & mask[:, None, :] & ~torch.eye(count, dtype=torch.bool, device=mask.device)[None]
        rel = positions[:, :, None] - positions[:, None, :]
        distance = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pair_radius = (radii[:, :, None] + radii[:, None, :]).clamp_min(1e-4)
        normalized_distance = distance / pair_radius
        surface_gap = (distance - pair_radius) / pair_radius
        radius_ratio = torch.log((radii[:, :, None] + 1e-4) / (radii[:, None, :] + 1e-4))
        normal_dot = (normals[:, :, None] * normals[:, None, :]).sum(-1, keepdim=True)
        tangent_dot = (tangents[:, :, None] * tangents[:, None, :]).sum(-1, keepdim=True)
        msg = self.message(torch.cat((left, right, self.relation(relations), normalized_distance, surface_gap, radius_ratio, normal_dot, tangent_dot), -1))
        msg = msg * pair_mask.unsqueeze(-1)
        degree = pair_mask.sum(2, keepdim=True).clamp_min(1).to(hidden.dtype)
        hidden = self.norm(hidden + self.update(torch.cat((hidden, msg.sum(2) / degree), -1))) * mask.unsqueeze(-1)
        unit = rel / (1.0 + distance)
        positions = positions + (unit * torch.tanh(self.coord_weight(msg))).sum(2) / degree * mask.unsqueeze(-1)
        normals = normals + ((normals[:, None] - normals[:, :, None]) * torch.tanh(self.normal_weight(msg))).sum(2) / degree
        tangents = tangents + ((tangents[:, None] - tangents[:, :, None]) * torch.tanh(self.tangent_weight(msg))).sum(2) / degree
        return hidden, positions, normals, tangents


class FPAEntityDenoiser(nn.Module):
    """From-scratch topology and SE(3) denoiser for heavy atoms and FPA cells."""

    def __init__(self, num_atom_types: int, num_cell_types: int, atom_radii,
                 hidden_dim: int = 192, num_layers: int = 6, time_dim: int = 64,
                 condition_dim: int = 0, condition_dropout: float = .2,
                 max_cell_ports: int = 8, charge_min: int = -2, charge_max: int = 2,
                 max_atoms: int = 128, max_cells: int = 32):
        super().__init__()
        self.num_atom_types = int(num_atom_types); self.num_cell_types = int(num_cell_types)
        self.atom_mask_token = self.num_atom_types; self.atom_pad_token = self.num_atom_types + 1
        self.cell_mask_token = self.num_cell_types; self.cell_pad_token = self.num_cell_types + 1
        self.charge_min = charge_min; self.charge_max = charge_max; self.num_charges = charge_max - charge_min + 1
        self.charge_mask_token = self.num_charges; self.max_cell_ports = max_cell_ports
        self.condition_dim = condition_dim; self.condition_dropout = condition_dropout
        self.atom_embedding = nn.Embedding(self.num_atom_types + 2, hidden_dim)
        self.cell_embedding = nn.Embedding(self.num_cell_types + 2, hidden_dim)
        self.charge_embedding = nn.Embedding(self.num_charges + 1, hidden_dim)
        self.kind_embedding = nn.Embedding(2, hidden_dim)
        self.cell_size_embedding = nn.Embedding(max_cell_ports + 1, hidden_dim)
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_projection = nn.Sequential(nn.Linear(3 * hidden_dim + time_dim + 1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.condition_projection = nn.Sequential(nn.Linear(condition_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) if condition_dim else None
        self.layers = nn.ModuleList(SizeAwareEntityLayer(hidden_dim) for _ in range(num_layers))
        self.atom_type_head = nn.Linear(hidden_dim, self.num_atom_types)
        self.charge_head = nn.Linear(hidden_dim, self.num_charges)
        self.cell_type_head = nn.Linear(hidden_dim, self.num_cell_types)
        self.cell_size_head = nn.Linear(hidden_dim, max_cell_ports + 1)
        pair_dim = 2 * hidden_dim + 2
        self.atom_bond_head = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 5))
        self.fusion_head = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.membership_head = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.left_port_head = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, max_cell_ports))
        self.right_port_head = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, max_cell_ports))
        count_input = max(1, condition_dim)
        self.atom_count_head = nn.Sequential(nn.Linear(count_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, max_atoms + 1))
        self.cell_count_head = nn.Sequential(nn.Linear(count_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, max_cells + 1))
        radii = torch.as_tensor(atom_radii, dtype=torch.float32)
        self.register_buffer('atom_radii', torch.cat((radii, radii.mean()[None], radii.mean()[None])), persistent=False)

    def charge_index(self, charges: Tensor) -> Tensor:
        return (charges - self.charge_min).clamp(0, self.num_charges - 1)

    @staticmethod
    def _pair_features(left: Tensor, right: Tensor, left_pos: Tensor, right_pos: Tensor) -> Tensor:
        distance = (left_pos[:, :, None] - right_pos[:, None, :]).norm(dim=-1, keepdim=True)
        return torch.cat((left[:, :, None].expand(-1, -1, right.shape[1], -1), right[:, None].expand(-1, left.shape[1], -1, -1), distance, distance.square()), -1)

    def forward(self, atom_types, atom_charges, atom_positions, atom_bonds, atom_mask,
                cell_types, cell_sizes, cell_centers, cell_normals, cell_tangents,
                cell_radii, fusion_types, membership, cell_mask, timesteps,
                condition: Tensor | None = None, condition_drop_mask: Tensor | None = None):
        na = atom_types.shape[1]
        initial_atom_positions, initial_cell_centers = atom_positions, cell_centers
        initial_normals, initial_tangents = cell_normals, cell_tangents
        time_a = self.time_embedding(timesteps)[:, None].expand(-1, na, -1)
        time_c = self.time_embedding(timesteps)[:, None].expand(-1, cell_types.shape[1], -1)
        atom_radius = self.atom_radii[atom_types].unsqueeze(-1)
        ah = self.input_projection(torch.cat((self.atom_embedding(atom_types), self.charge_embedding(atom_charges), self.kind_embedding.weight[0][None, None].expand(atom_types.shape[0], na, -1), time_a, atom_radius), -1))
        cell_size = cell_sizes.clamp(0, self.max_cell_ports)
        ch = self.input_projection(torch.cat((self.cell_embedding(cell_types), self.cell_size_embedding(cell_size), self.kind_embedding.weight[1][None, None].expand(cell_types.shape[0], cell_types.shape[1], -1), time_c, cell_radii.unsqueeze(-1)), -1))
        if self.condition_projection is not None:
            if condition is None: condition = ah.new_zeros((ah.shape[0], self.condition_dim))
            cond = self.condition_projection(condition.to(ah.dtype))
            if condition_drop_mask is not None: cond = cond * (~condition_drop_mask).to(cond.dtype).unsqueeze(-1)
            ah, ch = ah + cond[:, None], ch + cond[:, None]
        hidden = torch.cat((ah * atom_mask.unsqueeze(-1), ch * cell_mask.unsqueeze(-1)), 1)
        positions = torch.cat((atom_positions, cell_centers), 1)
        zeros = torch.zeros_like(atom_positions)
        normals = torch.cat((zeros, cell_normals), 1); tangents = torch.cat((zeros, cell_tangents), 1)
        radii = torch.cat((atom_radius, cell_radii.unsqueeze(-1)), 1)
        mask = torch.cat((atom_mask, cell_mask), 1)
        relations = atom_bonds.new_zeros((atom_types.shape[0], mask.shape[1], mask.shape[1]))
        relations[:, :na, :na] = atom_bonds.clamp(0, 4)
        relations[:, na:, na:] = torch.where(fusion_types.eq(1), 8, torch.where(fusion_types.eq(2), 9, 0))
        relations[:, na:, :na] = torch.where(membership.eq(1), 6, torch.where(membership.eq(2), 7, 0))
        relations[:, :na, na:] = relations[:, na:, :na].transpose(1, 2)
        for layer in self.layers:
            hidden, positions, normals, tangents = layer(hidden, positions, normals, tangents, radii, relations, mask)
            cell_normals = F.normalize(normals[:, na:], dim=-1) * cell_mask.unsqueeze(-1)
            cell_tangents = tangents[:, na:] - (tangents[:, na:] * cell_normals).sum(-1, keepdim=True) * cell_normals
            cell_tangents = F.normalize(cell_tangents, dim=-1) * cell_mask.unsqueeze(-1)
            normals = torch.cat((normals[:, :na], cell_normals), 1)
            tangents = torch.cat((tangents[:, :na], cell_tangents), 1)
        ah, ch = hidden[:, :na], hidden[:, na:]
        atom_positions, cell_centers = positions[:, :na], positions[:, na:]
        atom_pair = self._pair_features(ah, ah, atom_positions, atom_positions)
        cell_pair = self._pair_features(ch, ch, cell_centers, cell_centers)
        cross_pair = self._pair_features(ch, ah, cell_centers, atom_positions)
        count_condition = condition if condition is not None else ah.new_zeros((ah.shape[0], 1))
        if condition_drop_mask is not None: count_condition = count_condition * (~condition_drop_mask).to(count_condition.dtype).unsqueeze(-1)
        return {
            'atom_coordinate_noise': atom_positions - initial_atom_positions,
            'cell_center_noise': cell_centers - initial_cell_centers,
            'cell_normal_noise': normals[:, na:] - initial_normals,
            'cell_tangent_noise': tangents[:, na:] - initial_tangents,
            'atom_type_logits': self.atom_type_head(ah), 'atom_charge_logits': self.charge_head(ah),
            'atom_bond_logits': self.atom_bond_head(atom_pair),
            'cell_type_logits': self.cell_type_head(ch), 'cell_size_logits': self.cell_size_head(ch),
            'fusion_logits': self.fusion_head(cell_pair),
            'left_port_logits': self.left_port_head(cell_pair), 'right_port_logits': self.right_port_head(cell_pair),
            'membership_logits': self.membership_head(cross_pair),
            'atom_count_logits': self.atom_count_head(count_condition), 'cell_count_logits': self.cell_count_head(count_condition),
            'atom_hidden': ah, 'cell_hidden': ch,
        }
