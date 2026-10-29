"""V6 heavy-atom diffusion with joint aromatic/Kekule reverse state."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .chemistry_v6 import AtomStateCodec
from .model import (
    ConditionEmbedding,
    JointDiffusion,
    JointLayer,
    TimeEmbedding,
    _ce,
    aromatic_component_loss,
    aromatic_cycle_loss,
    geometry_consistency_losses,
)
from .model_v5 import global_chemistry_losses_v5


class HeavyAtomJointDDPMV6(nn.Module):
    """Predict x0 atom, edge, bond order and aromatic edge jointly.

    Aromatic membership is no longer an auxiliary scalar.  It is embedded in
    every noisy edge and has its own two-state D3PM posterior during sampling.
    Kekule order remains a separate categorical state, allowing the network to
    represent resonance-equivalent single/double assignments.
    """

    def __init__(
        self,
        atom_states,
        hidden=256,
        max_atoms=69,
        layers=8,
        condition_dropout=.1,
        s1_rbf_bins=24,
        t1_rbf_bins=24,
        condition_rbf_min=-5.,
        condition_rbf_max=5.,
        condition_rbf_width_scale=1.,
    ):
        super().__init__()
        self.codec = atom_states if isinstance(atom_states, AtomStateCodec) else AtomStateCodec.from_metadata(atom_states)
        self.max_atoms = int(max_atoms)
        self.state_types = len(self.codec)
        self.condition_dropout = float(condition_dropout)
        self.cond = ConditionEmbedding(hidden, s1_rbf_bins, t1_rbf_bins, condition_rbf_min, condition_rbf_max, condition_rbf_width_scale)
        self.null = nn.Parameter(torch.zeros(hidden))
        self.time = TimeEmbedding(hidden)

        elements = sorted({state.symbol for state in self.codec.states})
        charges = sorted({state.formal_charge for state in self.codec.states})
        valences = sorted({state.valence for state in self.codec.states})
        element_index = {value: index for index, value in enumerate(elements)}
        charge_index = {value: index for index, value in enumerate(charges)}
        valence_index = {value: index for index, value in enumerate(valences)}
        self.register_buffer("state_element_id", torch.tensor([element_index[x.symbol] for x in self.codec.states]))
        self.register_buffer("state_charge_id", torch.tensor([charge_index[x.formal_charge] for x in self.codec.states]))
        self.register_buffer("state_valence_id", torch.tensor([valence_index[x.valence] for x in self.codec.states]))
        self.register_buffer("state_role_id", torch.tensor([x.aromatic_role for x in self.codec.states]))
        self.register_buffer("state_charge_value", torch.tensor([x.formal_charge for x in self.codec.states], dtype=torch.float32))
        self.register_buffer("state_valence_value", torch.tensor([x.valence for x in self.codec.states], dtype=torch.float32))
        self.register_buffer("state_h_min", torch.tensor([x.hydrogen_min for x in self.codec.states], dtype=torch.float32))
        self.register_buffer("state_h_max", torch.tensor([x.hydrogen_max for x in self.codec.states], dtype=torch.float32))
        self.register_buffer("state_aromatic_value", torch.tensor([x.aromatic_role > 0 for x in self.codec.states], dtype=torch.float32))

        self.state_token = nn.Embedding(self.state_types, hidden)
        self.element_embedding = nn.Embedding(len(elements), hidden)
        self.charge_embedding = nn.Embedding(len(charges), hidden)
        self.valence_embedding = nn.Embedding(len(valences), hidden)
        self.role_embedding = nn.Embedding(3, hidden)
        self.edge_embedding = nn.Embedding(2, hidden)
        self.bond_embedding = nn.Embedding(3, hidden)
        self.aromatic_embedding = nn.Embedding(2, hidden)
        self.layers = nn.ModuleList(JointLayer(hidden) for _ in range(layers))
        self.count_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.max_atoms))
        self.state_head = nn.Linear(hidden, self.state_types)
        self.edge_head = nn.Linear(hidden, 2)
        self.bond_head = nn.Linear(hidden, 3)
        self.aromatic_head = nn.Linear(hidden, 2)
        self.coordinate_head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Linear(hidden, 1, bias=False))

    def condition_context(self, condition, unconditional=False):
        context = self.cond(condition)
        if unconditional:
            return self.null[None].expand_as(context)
        if self.training and self.condition_dropout:
            dropped = torch.rand(len(condition), device=condition.device) < self.condition_dropout
            context = torch.where(dropped[:, None], self.null[None], context)
        return context

    def state_embedding(self, state_ids):
        return (
            self.state_token(state_ids)
            + self.element_embedding(self.state_element_id[state_ids])
            + self.charge_embedding(self.state_charge_id[state_ids])
            + self.valence_embedding(self.state_valence_id[state_ids])
            + self.role_embedding(self.state_role_id[state_ids])
        )

    def forward(self, condition, timestep, noisy, unconditional=False):
        mask = noisy["atom_mask"]
        positions = noisy["positions"]
        _, count = mask.shape
        if count > self.max_atoms:
            raise ValueError("atom dimension exceeds max_atoms")
        condition_context = self.condition_context(condition, unconditional)
        context = condition_context + self.time(timestep)
        nodes = (context[:, None] + self.state_embedding(noisy["atom_state"])) * mask[..., None]
        pair = mask[:, :, None] & mask[:, None, :]
        gate = noisy["edge_exists"].to(positions.dtype)[..., None]
        edge_chemistry = self.bond_embedding(noisy["bond_type"]) + self.aromatic_embedding(noisy["aromatic_edge"])
        edges = self.edge_embedding(noisy["edge_exists"]) + gate * edge_chemistry
        edges = (edges + edges.transpose(1, 2)) * .5 * pair[..., None]
        for layer in self.layers:
            nodes, edges = layer(nodes, edges, positions, mask)
        pooled = (nodes * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

        delta = positions[:, :, None] - positions[:, None]
        distance = delta.norm(dim=-1, keepdim=True).clamp_max(50.)
        off_diagonal = pair & ~torch.eye(count, dtype=torch.bool, device=positions.device)[None]
        weight = self.coordinate_head(torch.cat((edges, distance), -1)) * off_diagonal[..., None]
        epsilon = (weight * delta).sum(2) / off_diagonal.sum(2).clamp_min(1)[..., None]
        epsilon -= (epsilon * mask[..., None]).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)[..., None]

        edge_logits = self.edge_head(edges)
        bond_type_logits = self.bond_head(edges)
        edge_log_probability = F.log_softmax(edge_logits, -1)
        type_log_probability = F.log_softmax(bond_type_logits, -1)
        bond_logits = torch.cat((edge_log_probability[..., :1], edge_log_probability[..., 1:2] + type_log_probability), -1)
        return {
            "count_logits": self.count_head(condition_context),
            "state_logits": self.state_head(nodes),
            "edge_logits": edge_logits,
            "bond_type_logits": bond_type_logits,
            "bond_logits": bond_logits,
            "aromatic_edge_logits": self.aromatic_head(edges),
            "coordinate_eps": epsilon * mask[..., None],
        }


def _soft_topology_statistics(presence: torch.Tensor, double: torch.Tensor, mask: torch.Tensor):
    """Differentiable small-cycle/bridgehead/cumulene surrogates per graph."""
    pair = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    adjacency = presence.masked_fill(~pair | eye, 0)
    adjacency = (adjacency + adjacency.transpose(1, 2)) * .5
    a2 = adjacency @ adjacency
    a3 = a2 @ adjacency
    a4 = a2 @ a2
    triangles = a3.diagonal(dim1=1, dim2=2).sum(-1) / 6
    degree = adjacency.sum(-1)
    edges = degree.sum(-1) / 2
    wedges = (degree * (degree - 1)).sum(-1) / 2
    squares = ((a4.diagonal(dim1=1, dim2=2).sum(-1) - 2 * edges - 4 * wedges) / 8).clamp_min(0)
    cycle_incidence = a3.diagonal(dim1=1, dim2=2) + .25 * a4.diagonal(dim1=1, dim2=2)
    bridgeheads = (F.relu(degree - 2) * torch.sigmoid(cycle_incidence - 1)).sum(-1)
    double_degree = (double * pair).sum(-1)
    cumulenes = F.relu(double_degree - 1).sum(-1)
    return triangles + squares, bridgeheads, cumulenes


def topology_losses_v6(outputs, batch):
    mask = batch["atom_mask"]
    atom_count = mask.shape[1]
    pair = mask[:, :, None] & mask[:, None, :] & ~torch.eye(atom_count, dtype=torch.bool, device=mask.device)[None]
    bond_probability = F.softmax(outputs["bond_logits"].float(), -1)
    predicted_presence = (1 - bond_probability[..., 0]) * pair
    predicted_double = bond_probability[..., 2] * pair
    target_presence = ((batch["atom_bonds"] > 0) & pair).float()
    target_double = ((batch["atom_bonds"] == 2) & pair).float()
    predicted = _soft_topology_statistics(predicted_presence, predicted_double, mask)
    target = _soft_topology_statistics(target_presence, target_double, mask)
    nodes = mask.sum(-1).float().clamp_min(1)
    # A dense random graph has O(N^4) four-cycles.  Dividing by N (as for
    # degree-like counts) lets this auxiliary term swamp the categorical
    # likelihood at initialization.  Use the number of possible undirected
    # triangles plus 4-cycles, i.e. C(N,3) + 3*C(N,4), as its natural scale.
    possible_triangles = nodes * (nodes - 1).clamp_min(0) * (nodes - 2).clamp_min(0) / 6
    possible_squares = nodes * (nodes - 1).clamp_min(0) * (nodes - 2).clamp_min(0) * (nodes - 3).clamp_min(0) / 8
    possible_small_cycles = (possible_triangles + possible_squares).clamp_min(1)
    small_scale = torch.log1p(possible_small_cycles)
    predicted_small_density = torch.log1p(predicted[0].clamp_min(0)) / small_scale
    target_small_density = torch.log1p(target[0].clamp_min(0)) / small_scale
    return {
        "small_ring_prior": .08 * F.smooth_l1_loss(predicted_small_density, target_small_density),
        "bridgehead_prior": .05 * F.smooth_l1_loss(predicted[1] / nodes, target[1] / nodes),
        "cumulene_prior": .08 * F.smooth_l1_loss(predicted[2] / nodes, target[2] / nodes),
    }


def joint_training_loss_v6(model, diffusion: JointDiffusion, batch):
    mask = batch["atom_mask"]
    batch_size, atom_count = mask.shape
    timestep = torch.randint(diffusion.timesteps, (batch_size,), device=mask.device)
    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(atom_count, dtype=torch.bool, device=mask.device)[None]
    upper = pair & torch.triu(torch.ones(atom_count, atom_count, dtype=torch.bool, device=mask.device), 1)[None]
    noisy_positions, coordinate_noise = diffusion.q_coords(batch["atom_positions"], timestep, mask)
    edge = (batch["atom_bonds"] > 0).long()
    bond_type = (batch["atom_bonds"] - 1).clamp_min(0)
    aromatic = batch["aromatic_edge_mask"].long()
    aromatic_bool = aromatic.bool()
    kekule_aromatic = aromatic_bool & (batch["atom_bonds"] >= 1) & (batch["atom_bonds"] <= 2)
    regular_bond = upper & edge.bool() & ~kekule_aromatic
    noisy_state = diffusion.q_token(batch["atom_state"], timestep, model.state_types, mask, name="atom")
    noisy_edge = diffusion.q_token(edge, timestep, 2, pair & ~diagonal, True, "edge")
    noisy_bond_type = diffusion.q_token(bond_type, timestep, 3, pair & ~diagonal, True, "bond_type")
    noisy_aromatic = diffusion.q_token(aromatic, timestep, 2, pair & ~diagonal, True, "aromatic_edge")
    noisy = {
        "atom_mask": mask,
        "atom_state": noisy_state,
        "edge_exists": noisy_edge,
        "bond_type": noisy_bond_type,
        "aromatic_edge": noisy_aromatic,
        "positions": noisy_positions,
    }
    outputs = model(batch["condition"], timestep, noisy)
    aromatic_logits = outputs["aromatic_edge_logits"]
    aromatic_scalar = aromatic_logits[..., 1] - aromatic_logits[..., 0]
    positives = aromatic[upper].sum().float()
    negatives = upper.sum().float() - positives
    aromatic_weight = aromatic_logits.new_tensor([1., float((negatives / positives.clamp_min(1)).clamp(1, 50))])
    aromatic_probability = F.softmax(aromatic_logits.float(), -1)[..., 1]
    edge_probability = F.softmax(outputs["edge_logits"].float(), -1)[..., 1]
    triple_probability = F.softmax(outputs["bond_type_logits"].float(), -1)[..., 2]
    aromatic_joint = (F.relu(aromatic_probability - edge_probability) + aromatic_probability * triple_probability)[upper].mean()
    terms = {
        "count": F.cross_entropy(outputs["count_logits"], mask.sum(1) - 1),
        "state": _ce(outputs["state_logits"], batch["atom_state"], mask),
        "edge_exists": _ce(outputs["edge_logits"], edge, upper),
        "bond_type": _ce(outputs["bond_type_logits"], bond_type, regular_bond),
        "aromatic_edge": F.cross_entropy(aromatic_logits[upper], aromatic[upper], weight=aromatic_weight),
        "aromatic_joint": .2 * aromatic_joint,
        "aromatic_cycle": aromatic_cycle_loss(
            outputs["bond_logits"], aromatic_scalar, batch["aromatic_cycle_edges"],
            batch["aromatic_cycle_double_count"], batch["ring_mask"], batch["aromatic_cycle_order"],
            batch["aromatic_cycle_size"], kekule_aromatic, aromatic_bool, upper,
        ),
        "aromatic_component": aromatic_component_loss(outputs, batch),
        "coordinate": ((outputs["coordinate_eps"] - coordinate_noise).square() * mask[..., None]).sum() / mask.sum().clamp_min(1),
        "vlb_state": .1 * diffusion.categorical_vlb(batch["atom_state"], noisy_state, outputs["state_logits"], timestep, model.state_types, mask, "atom"),
        "vlb_edge": .1 * diffusion.categorical_vlb(edge, noisy_edge, outputs["edge_logits"], timestep, 2, upper, "edge"),
        "vlb_bond_type": .1 * diffusion.categorical_vlb(bond_type, noisy_bond_type, outputs["bond_type_logits"], timestep, 3, regular_bond, "bond_type"),
        "vlb_aromatic_edge": .1 * diffusion.categorical_vlb(aromatic, noisy_aromatic, aromatic_logits, timestep, 2, upper, "aromatic_edge"),
    }
    terms.update(global_chemistry_losses_v5(outputs, batch, model))
    terms.update(topology_losses_v6(outputs, batch))
    terms.update(geometry_consistency_losses(outputs, batch, noisy_positions, timestep, diffusion))
    return sum(terms.values()), terms


__all__ = ["HeavyAtomJointDDPMV6", "joint_training_loss_v6", "topology_losses_v6"]
