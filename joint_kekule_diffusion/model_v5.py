"""V5 heavy-atom joint-state graph/coordinate diffusion.

Unlike v4, hydrogen and radical are not independently diffused.  The discrete
node variable is an observed, chemically joint heavy-atom state containing
element, formal charge, total valence and aromatic role.  Hydrogen is a
residual constraint of that state and the final heavy-atom bond graph.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .chemistry_v5 import AtomStateCodec, HeavyAtomState
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


class HeavyAtomJointDDPM(nn.Module):
    """Joint atom-state/edge x0 predictor with an equivariant epsilon head."""

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
        self.layers = nn.ModuleList(JointLayer(hidden) for _ in range(layers))
        self.count_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, self.max_atoms))
        self.state_head = nn.Linear(hidden, self.state_types)
        self.edge_head = nn.Linear(hidden, 2)
        self.bond_head = nn.Linear(hidden, 3)
        self.aromatic_head = nn.Linear(hidden, 1)
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
        edges = self.edge_embedding(noisy["edge_exists"]) + gate * self.bond_embedding(noisy["bond_type"])
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
            "aromatic_edge_logits": self.aromatic_head(edges).squeeze(-1),
            "coordinate_eps": epsilon * mask[..., None],
        }


def global_chemistry_losses_v5(outputs, batch, model):
    """Joint-state valence, neutrality, topology and connectivity losses."""
    mask = batch["atom_mask"]
    atom_count = mask.shape[1]
    pair = mask[:, :, None] & mask[:, None, :] & ~torch.eye(atom_count, dtype=torch.bool, device=mask.device)[None]
    state_probability = F.softmax(outputs["state_logits"].float(), -1)
    bond_probability = F.softmax(outputs["bond_logits"].float(), -1)
    orders = bond_probability.new_tensor([0., 1., 2., 3.])
    expected_bond = (bond_probability * orders).sum(-1) * pair
    presence = (1 - bond_probability[..., 0]) * pair
    target_presence = (batch["atom_bonds"] > 0) * pair

    expected_valence = (state_probability * model.state_valence_value).sum(-1)
    expected_h_min = (state_probability * model.state_h_min).sum(-1)
    expected_h_max = (state_probability * model.state_h_max).sum(-1)
    # Keep chemical-state/bond-order consistency on true neighbours. Edge CE,
    # degree and cycle-rank losses separately suppress false-positive topology.
    chemistry_bond = expected_bond * target_presence
    residual_hydrogen = expected_valence - chemistry_bond.sum(-1)
    hydrogen_target = batch["atom_hydrogen"].float()
    implicit_hydrogen = F.smooth_l1_loss(residual_hydrogen[mask], hydrogen_target[mask])
    support_violation = (
        F.relu(expected_h_min - residual_hydrogen).square()
        + F.relu(residual_hydrogen - expected_h_max).square()
    )[mask].mean()

    expected_charge = (state_probability * model.state_charge_value).sum(-1) * mask
    target_charge = (batch["atom_charge"].float() - 2).mul(mask).sum(-1)
    charge_balance = F.smooth_l1_loss(expected_charge.sum(-1), target_charge)
    expected_aromatic = (state_probability * model.state_aromatic_value).sum(-1)
    target_aromatic = batch["aromatic_edge_mask"].any(-1).float()
    aromatic_probability = expected_aromatic[mask].clamp(1e-6, 1 - 1e-6)
    aromatic_target = target_aromatic[mask]
    aromatic_role = -(aromatic_target * aromatic_probability.log() + (1 - aromatic_target) * torch.log1p(-aromatic_probability)).mean()

    degree = F.smooth_l1_loss(presence.sum(-1)[mask], target_presence.sum(-1).float()[mask])
    nodes = mask.sum(-1).float().clamp_min(1)
    predicted_cycle_rank = presence.sum((1, 2)) * .5 - nodes + 1
    target_cycle_rank = target_presence.sum((1, 2)).float() * .5 - nodes + 1
    cycle_rank = F.smooth_l1_loss(predicted_cycle_rank / nodes, target_cycle_rank / nodes)

    connectivity_terms = []
    for count in torch.unique(mask.sum(-1)).tolist():
        rows = mask.sum(-1) == count
        size = int(count)
        if size <= 1:
            continue
        probabilities = presence[rows, :size, :size]
        laplacian = torch.diag_embed(probabilities.sum(-1)) - probabilities
        eye = torch.eye(size - 1, device=mask.device, dtype=probabilities.dtype)
        sign, logdet = torch.linalg.slogdet(laplacian[:, :size - 1, :size - 1] + eye * 1e-6)
        connectivity_terms.append((F.softplus(-logdet / size) + (sign <= 0).to(logdet.dtype) * 10).mean())
    connectivity = torch.stack(connectivity_terms).mean() if connectivity_terms else outputs["bond_logits"].sum() * 0
    return {
        "implicit_hydrogen": .35 * implicit_hydrogen,
        "valence_support": .5 * support_violation,
        "charge_balance": .2 * charge_balance,
        "aromatic_role": .1 * aromatic_role,
        "degree": .1 * degree,
        "cycle_rank": .25 * cycle_rank,
        "connectivity": .2 * connectivity,
    }


def joint_training_loss_v5(model, diffusion: JointDiffusion, batch):
    mask = batch["atom_mask"]
    batch_size, atom_count = mask.shape
    timestep = torch.randint(diffusion.timesteps, (batch_size,), device=mask.device)
    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(atom_count, dtype=torch.bool, device=mask.device)[None]
    upper = pair & torch.triu(torch.ones(atom_count, atom_count, dtype=torch.bool, device=mask.device), 1)[None]
    noisy_positions, coordinate_noise = diffusion.q_coords(batch["atom_positions"], timestep, mask)
    edge = (batch["atom_bonds"] > 0).long()
    bond_type = (batch["atom_bonds"] - 1).clamp_min(0)
    aromatic = batch["aromatic_edge_mask"].bool()
    kekule_aromatic = aromatic & (batch["atom_bonds"] >= 1) & (batch["atom_bonds"] <= 2)
    regular_bond = upper & edge.bool() & ~kekule_aromatic
    noisy_state = diffusion.q_token(batch["atom_state"], timestep, model.state_types, mask, name="atom")
    noisy_edge = diffusion.q_token(edge, timestep, 2, pair & ~diagonal, True, "edge")
    noisy_bond_type = diffusion.q_token(bond_type, timestep, 3, pair & ~diagonal, True, "bond_type")
    noisy = {
        "atom_mask": mask,
        "atom_state": noisy_state,
        "edge_exists": noisy_edge,
        "bond_type": noisy_bond_type,
        "positions": noisy_positions,
    }
    outputs = model(batch["condition"], timestep, noisy)
    terms = {
        "count": F.cross_entropy(outputs["count_logits"], mask.sum(1) - 1),
        "state": _ce(outputs["state_logits"], batch["atom_state"], mask),
        "edge_exists": _ce(outputs["edge_logits"], edge, upper),
        "bond_type": _ce(outputs["bond_type_logits"], bond_type, regular_bond),
        "aromatic_cycle": aromatic_cycle_loss(
            outputs["bond_logits"], outputs["aromatic_edge_logits"], batch["aromatic_cycle_edges"],
            batch["aromatic_cycle_double_count"], batch["ring_mask"], batch["aromatic_cycle_order"],
            batch["aromatic_cycle_size"], kekule_aromatic, aromatic, upper,
        ),
        "aromatic_component": aromatic_component_loss(outputs, batch),
        "coordinate": ((outputs["coordinate_eps"] - coordinate_noise).square() * mask[..., None]).sum() / mask.sum().clamp_min(1),
        "vlb_state": .1 * diffusion.categorical_vlb(batch["atom_state"], noisy_state, outputs["state_logits"], timestep, model.state_types, mask, "atom"),
        "vlb_edge": .1 * diffusion.categorical_vlb(edge, noisy_edge, outputs["edge_logits"], timestep, 2, upper, "edge"),
        "vlb_bond_type": .1 * diffusion.categorical_vlb(bond_type, noisy_bond_type, outputs["bond_type_logits"], timestep, 3, regular_bond, "bond_type"),
    }
    terms.update(global_chemistry_losses_v5(outputs, batch, model))
    terms.update(geometry_consistency_losses(outputs, batch, noisy_positions, timestep, diffusion))
    return sum(terms.values()), terms


__all__ = ["HeavyAtomJointDDPM", "joint_training_loss_v5", "global_chemistry_losses_v5"]
