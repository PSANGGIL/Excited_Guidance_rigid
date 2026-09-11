"""SE(3)-equivariant denoiser with template-free geometric ring cells."""
from __future__ import annotations

import torch
from torch import nn

from .diffusion import SinusoidalTimeEmbedding
from .fpa_diffusion_v2 import masked_center


class EntityBlock(nn.Module):
    def __init__(self, hidden: int, relations: int = 21):
        super().__init__()
        self.rel = nn.Embedding(relations, hidden)
        self.msg = nn.Sequential(nn.Linear(3*hidden+8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.up = nn.Sequential(nn.Linear(2*hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, pos, frame, radius, kind, relation, mask):
        n = h.shape[1]
        pair = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=h.device)[None]
        delta = pos[:, :, None]-pos[:, None]
        dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        radius_sum = (radius[:, :, None]+radius[:, None]).clamp_min(1e-4)
        normal, tangent = frame[..., 2], frame[..., 0]
        src_cell = kind[:, :, None, None].to(pos.dtype)
        dst_cell = kind[:, None, :, None].to(pos.dtype)
        height = (delta*normal[:, :, None]).sum(-1, keepdim=True)
        axial = height.abs()/(.18*radius[:, :, None]+.1)
        planar = torch.sqrt((dist.square()-height.square()).clamp_min(0))/radius[:, :, None].clamp_min(.1)
        geo = (
            dist/radius_sum, (dist-radius_sum)/radius_sum,
            torch.log((radius[:, :, None]+1e-4)/(radius[:, None]+1e-4)),
            (normal[:, :, None]*normal[:, None]).sum(-1, keepdim=True),
            (tangent[:, :, None]*tangent[:, None]).sum(-1, keepdim=True),
            axial*src_cell, planar*src_cell, src_cell-dst_cell,
        )
        left = h[:, :, None].expand(-1, n, n, -1)
        right = h[:, None].expand(-1, n, n, -1)
        message = self.msg(torch.cat((left, right, self.rel(relation), *geo), -1))*pair.unsqueeze(-1)
        degree = pair.sum(2, keepdim=True).clamp_min(1).to(h.dtype)
        return self.norm(h+self.up(torch.cat((h, message.sum(2)/degree), -1)))*mask.unsqueeze(-1)


class TemplateFreeFPAEntityDenoiser(nn.Module):
    def __init__(self, num_atom_types, atom_radii, hidden_dim=192, num_layers=8,
                 time_dim=64, condition_dim=120, condition_dropout=.2,
                 min_ring_size=3, max_ring_size=10, max_atoms=128, max_cells=32,
                 charge_min=-2, charge_max=2):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.max_atoms, self.max_cells = max_atoms, max_cells
        self.condition_dim, self.condition_dropout = condition_dim, condition_dropout
        self.charge_min = charge_min
        self.num_charges = charge_max-charge_min+1
        self.min_ring_size, self.max_ring_size = min_ring_size, max_ring_size
        self.num_ring_sizes = max_ring_size-min_ring_size+1
        self.atom_mask_token, self.atom_pad_token = num_atom_types, num_atom_types+1
        self.charge_mask_token = self.num_charges
        self.hydrogen_mask_token = 5
        self.radical_mask_token = 3
        self.size_mask_token, self.size_pad_token = self.num_ring_sizes, self.num_ring_sizes+1
        self.aromatic_mask_token = 2
        self.relation_mask_token = 12
        h = hidden_dim
        self.atom_emb = nn.Embedding(num_atom_types+2, h)
        self.charge_emb = nn.Embedding(self.num_charges+1, h)
        self.hydrogen_emb = nn.Embedding(6, h)
        self.radical_emb = nn.Embedding(4, h)
        self.size_emb = nn.Embedding(self.num_ring_sizes+2, h)
        self.aromatic_emb = nn.Embedding(3, h)
        self.kind_emb = nn.Embedding(2, h)
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.atom_in = nn.Sequential(nn.Linear(5*h+time_dim+1, h), nn.SiLU(), nn.Linear(h, h))
        self.cell_in = nn.Sequential(nn.Linear(3*h+time_dim+1, h), nn.SiLU(), nn.Linear(h, h))
        self.cond = nn.Sequential(nn.Linear(condition_dim, h), nn.SiLU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList(EntityBlock(h) for _ in range(num_layers))
        self.atom_type = nn.Linear(h, num_atom_types)
        self.charge = nn.Linear(h, self.num_charges)
        self.hydrogen = nn.Linear(h, 5)
        self.radical = nn.Linear(h, 3)
        self.cell_size = nn.Linear(h, self.num_ring_sizes)
        self.cell_aromatic = nn.Linear(h, 2)
        self.radius_eps = nn.Linear(h, 1)
        self.omega_coeff = nn.Linear(h, 3)
        self.coord_pair = nn.Sequential(nn.Linear(2*h+2, h), nn.SiLU(), nn.Linear(h, 1, bias=False))
        self.cell_coord_pair = nn.Sequential(nn.Linear(2*h+2, h), nn.SiLU(), nn.Linear(h, 1, bias=False))
        self.bond = nn.Sequential(nn.Linear(2*h+2, h), nn.SiLU(), nn.Linear(h, 5))
        self.relation = nn.Sequential(nn.Linear(2*h+2, h), nn.SiLU(), nn.Linear(h, 12))
        self.member = nn.Sequential(nn.Linear(2*h+2, h), nn.SiLU(), nn.Linear(h, 2))
        self.atom_count = nn.Sequential(nn.Linear(condition_dim, h), nn.SiLU(), nn.Linear(h, max_atoms+1))
        self.cell_count = nn.Sequential(nn.Linear(condition_dim, h), nn.SiLU(), nn.Linear(h, max_cells+1))
        radii = torch.as_tensor(atom_radii, dtype=torch.float32)
        self.register_buffer("atom_radii", torch.cat((radii, radii.mean()[None], radii.mean()[None])), persistent=False)

    def encode_size(self, size):
        return (size-self.min_ring_size).clamp(0, self.num_ring_sizes-1)

    def decode_size(self, index):
        return index+self.min_ring_size

    def charge_index(self, charge):
        return (charge-self.charge_min).clamp(0, self.num_charges-1)

    @staticmethod
    def pair(h1, h2, p1, p2):
        d = (p1[:, :, None]-p2[:, None]).norm(dim=-1, keepdim=True)
        return torch.cat((h1[:, :, None].expand(-1, -1, h2.shape[1], -1),
                          h2[:, None].expand(-1, h1.shape[1], -1, -1), d, d.square()), -1)

    def forward(self, atom_types, charges, hydrogens, radicals, atom_pos, atom_bonds, atom_mask,
                cell_sizes, cell_aromatic, cell_pos, cell_frames, cell_radius, cell_relations,
                membership, cell_mask, t, condition, drop=None):
        b, na = atom_types.shape
        nc = cell_sizes.shape[1]
        te = self.time(t)
        atom_radius = self.atom_radii[atom_types].unsqueeze(-1)
        atom_h = self.atom_in(torch.cat((self.atom_emb(atom_types), self.charge_emb(charges), self.hydrogen_emb(hydrogens), self.radical_emb(radicals),
            self.kind_emb.weight[0][None, None].expand(b, na, -1),
            te[:, None].expand(-1, na, -1), atom_radius), -1))
        cell_h = self.cell_in(torch.cat((self.size_emb(cell_sizes), self.aromatic_emb(cell_aromatic),
            self.kind_emb.weight[1][None, None].expand(b, nc, -1),
            te[:, None].expand(-1, nc, -1), cell_radius.unsqueeze(-1)), -1))
        condition_input = condition.to(atom_h.dtype)
        if drop is not None:
            condition_input = condition_input*(~drop).to(condition_input.dtype).unsqueeze(-1)
        condition_h = self.cond(condition_input)
        atom_h, cell_h = atom_h+condition_h[:, None], cell_h+condition_h[:, None]
        h = torch.cat((atom_h*atom_mask.unsqueeze(-1), cell_h*cell_mask.unsqueeze(-1)), 1)
        pos = torch.cat((atom_pos, cell_pos), 1)
        atom_frame = torch.zeros((b, na, 3, 3), device=h.device, dtype=atom_pos.dtype)
        frame = torch.cat((atom_frame, cell_frames), 1)
        radius = torch.cat((atom_radius, cell_radius.exp().clamp(.1, 10.).unsqueeze(-1)), 1)
        kind = torch.cat((torch.zeros_like(atom_mask), torch.ones_like(cell_mask)), 1)
        mask = torch.cat((atom_mask, cell_mask), 1)
        rel = atom_bonds.new_zeros((b, na+nc, na+nc))
        rel[:, :na, :na] = atom_bonds
        rel[:, na:, na:] = cell_relations+8
        rel[:, na:, :na] = torch.where(membership.eq(1), 6, torch.where(membership.eq(2), 7, 0))
        rel[:, :na, na:] = rel[:, na:, :na].transpose(1, 2)
        for block in self.blocks:
            h = block(h, pos, frame, radius, kind, rel, mask)
        atom_h, cell_h = h[:, :na], h[:, na:]
        aa = self.pair(atom_h, atom_h, atom_pos, atom_pos)
        cc = self.pair(cell_h, cell_h, cell_pos, cell_pos)
        ca = self.pair(cell_h, atom_h, cell_pos, atom_pos)
        bond_logits = self.bond(aa); bond_logits = (bond_logits+bond_logits.transpose(1, 2))/2
        relation_logits = self.relation(cc); relation_logits = (relation_logits+relation_logits.transpose(1, 2))/2
        atom_delta = atom_pos[:, :, None]-atom_pos[:, None]
        atom_weight = self.coord_pair(aa); atom_weight = (atom_weight+atom_weight.transpose(1, 2))/2
        atom_eps = masked_center((atom_delta*atom_weight).sum(2), atom_mask)
        all_delta = cell_pos[:, :, None]-pos[:, None]
        cell_eps = (all_delta*self.cell_coord_pair(self.pair(cell_h, h, cell_pos, pos))).sum(2)/mask.sum(1)[:, None, None].clamp_min(1)
        cell_eps = cell_eps*cell_mask.unsqueeze(-1)
        coeff = self.omega_coeff(cell_h)
        rotation_score = (cell_frames*coeff[..., :, None]).sum(-2)*cell_mask.unsqueeze(-1)
        return {
            "atom_eps": atom_eps, "cell_eps": cell_eps,
            "radius_eps": self.radius_eps(cell_h).squeeze(-1)*cell_mask,
            "rotation_score": rotation_score,
            "atom_type_logits": self.atom_type(atom_h), "charge_logits": self.charge(atom_h),
            "hydrogen_logits": self.hydrogen(atom_h), "radical_logits": self.radical(atom_h),
            "bond_logits": bond_logits, "cell_size_logits": self.cell_size(cell_h),
            "cell_aromatic_logits": self.cell_aromatic(cell_h),
            "relation_logits": relation_logits, "membership_logits": self.member(ca),
            "atom_count_logits": self.atom_count(condition_input), "cell_count_logits": self.cell_count(condition_input),
        }
