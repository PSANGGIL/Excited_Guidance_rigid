"""V7 heavy-atom diffusion with one chemically joint categorical edge state."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .chemistry_v5 import AtomStateCodec
from .model import (
    ConditionEmbedding, JointDiffusion, JointLayer, TimeEmbedding, _ce,
    aromatic_component_loss, aromatic_cycle_loss, geometry_consistency_losses,
)
from .model_v5 import global_chemistry_losses_v5

EDGE_NONE = 0
EDGE_SINGLE = 1
EDGE_DOUBLE = 2
EDGE_TRIPLE = 3
EDGE_AROMATIC_SINGLE = 4
EDGE_AROMATIC_DOUBLE = 5
EDGE_AROMATIC_TRIPLE = 6
JOINT_EDGE_CLASSES = 7


def encode_joint_edge_state(bonds: torch.Tensor, aromatic_edges: torch.Tensor) -> torch.Tensor:
    """Encode Kekule bond order and aromatic membership without invalid states."""
    bonds = torch.as_tensor(bonds).long()
    aromatic = torch.as_tensor(aromatic_edges).bool()
    if bonds.shape != aromatic.shape:
        raise ValueError("bond and aromatic tensors must have the same shape")
    if ((bonds < 0) | (bonds > 3)).any():
        raise ValueError("bond order must be in [0,3]")
    if (aromatic & ~((bonds == 1) | (bonds == 2) | (bonds == 3))).any():
        raise ValueError("aromatic edges must have Kekule order one, two or three")
    state = bonds.clone()
    state = torch.where(aromatic & (bonds == 1), state.new_tensor(EDGE_AROMATIC_SINGLE), state)
    state = torch.where(aromatic & (bonds == 2), state.new_tensor(EDGE_AROMATIC_DOUBLE), state)
    return torch.where(aromatic & (bonds == 3), state.new_tensor(EDGE_AROMATIC_TRIPLE), state)


def decode_joint_edge_state(edge_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Kekule bond orders and aromatic-edge flags."""
    state = torch.as_tensor(edge_state).long()
    if ((state < 0) | (state >= JOINT_EDGE_CLASSES)).any():
        raise ValueError("joint edge state outside vocabulary")
    bonds = torch.zeros_like(state)
    for edge_class, order in (
        (EDGE_SINGLE, 1), (EDGE_DOUBLE, 2), (EDGE_TRIPLE, 3),
        (EDGE_AROMATIC_SINGLE, 1), (EDGE_AROMATIC_DOUBLE, 2), (EDGE_AROMATIC_TRIPLE, 3),
    ):
        bonds = torch.where(state == edge_class, bonds.new_tensor(order), bonds)
    aromatic = ((state == EDGE_AROMATIC_SINGLE) | (state == EDGE_AROMATIC_DOUBLE)
                | (state == EDGE_AROMATIC_TRIPLE))
    return bonds, aromatic


def joint_edge_log_views(joint_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive normalized bond-order, existence and aromatic log probabilities."""
    if joint_logits.shape[-1] != JOINT_EDGE_CLASSES:
        raise ValueError("joint edge logits must have seven classes")
    logp = F.log_softmax(joint_logits.float(), -1)
    bond = torch.stack((
        logp[..., EDGE_NONE],
        torch.logsumexp(logp[..., [EDGE_SINGLE, EDGE_AROMATIC_SINGLE]], -1),
        torch.logsumexp(logp[..., [EDGE_DOUBLE, EDGE_AROMATIC_DOUBLE]], -1),
        torch.logsumexp(logp[..., [EDGE_TRIPLE, EDGE_AROMATIC_TRIPLE]], -1),
    ), -1)
    existence = torch.stack((logp[..., EDGE_NONE], torch.logsumexp(logp[..., 1:], -1)), -1)
    aromatic = torch.stack((torch.logsumexp(logp[..., :4], -1), torch.logsumexp(logp[..., 4:], -1)), -1)
    return bond, existence, aromatic


class HeavyAtomJointDDPMV7(nn.Module):
    """Joint atom-state/edge-state denoiser with an equivariant epsilon head."""

    def __init__(
        self, atom_states, hidden=256, max_atoms=69, layers=8,
        condition_dropout=.1, s1_rbf_bins=24, t1_rbf_bins=24,
        condition_rbf_min=-5., condition_rbf_max=5.,
        condition_rbf_width_scale=1., model_dropout=.1, count_dropout=.2,
    ):
        super().__init__()
        self.codec = atom_states if isinstance(atom_states, AtomStateCodec) else AtomStateCodec.from_metadata(atom_states)
        self.max_atoms = int(max_atoms)
        self.state_types = len(self.codec)
        self.condition_dropout = float(condition_dropout)
        self.model_dropout = float(model_dropout)
        self.count_dropout = float(count_dropout)
        if not 0 <= self.model_dropout < 1 or not 0 <= self.count_dropout < 1:
            raise ValueError("dropout probabilities must be in [0,1)")
        self.cond = ConditionEmbedding(
            hidden, s1_rbf_bins, t1_rbf_bins, condition_rbf_min,
            condition_rbf_max, condition_rbf_width_scale,
        )
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
        self.edge_embedding = nn.Embedding(JOINT_EDGE_CLASSES, hidden)
        self.layers = nn.ModuleList(JointLayer(hidden) for _ in range(layers))
        # CFG condition dropout removes the complete condition for a molecule.
        # These ordinary dropouts regularize representations without deleting
        # atom or edge tokens from the molecular graph.
        self.context_dropout = nn.Dropout(self.model_dropout)
        self.node_dropout = nn.Dropout(self.model_dropout)
        self.edge_dropout = nn.Dropout(self.model_dropout)
        self.head_dropout = nn.Dropout(self.model_dropout)
        self.count_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(self.count_dropout),
            nn.Linear(hidden, self.max_atoms),
        )
        self.state_head = nn.Linear(hidden, self.state_types)
        self.joint_edge_head = nn.Linear(hidden, JOINT_EDGE_CLASSES)
        self.coordinate_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.SiLU(), nn.Dropout(self.model_dropout),
            nn.Linear(hidden, 1, bias=False),
        )

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
        mask, positions = noisy["atom_mask"], noisy["positions"]
        _, count = mask.shape
        if count > self.max_atoms:
            raise ValueError("atom dimension exceeds max_atoms")
        condition_context = self.condition_context(condition, unconditional)
        context = self.context_dropout(condition_context + self.time(timestep))
        nodes = (context[:, None] + self.state_embedding(noisy["atom_state"])) * mask[..., None]
        pair = mask[:, :, None] & mask[:, None, :]
        off_diagonal = pair & ~torch.eye(count, dtype=torch.bool, device=positions.device)[None]
        edges = self.edge_embedding(noisy["edge_state"])
        edges = (edges + edges.transpose(1, 2)) * .5 * off_diagonal[..., None]
        for layer in self.layers:
            previous_nodes, previous_edges = nodes, edges
            proposed_nodes, proposed_edges = layer(nodes, edges, positions, mask)
            nodes = (
                previous_nodes + self.node_dropout(proposed_nodes - previous_nodes)
            ) * mask[..., None]
            edge_update = self.edge_dropout(proposed_edges - previous_edges)
            edges = previous_edges + (edge_update + edge_update.transpose(1, 2)) * .5
            edges = edges * off_diagonal[..., None]

        delta = positions[:, :, None] - positions[:, None]
        distance = delta.norm(dim=-1, keepdim=True).clamp_max(50.)
        weight = self.coordinate_head(torch.cat((edges, distance), -1)) * off_diagonal[..., None]
        epsilon = (weight * delta).sum(2) / off_diagonal.sum(2).clamp_min(1)[..., None]
        epsilon -= (epsilon * mask[..., None]).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)[..., None]

        joint_logits = self.joint_edge_head(self.head_dropout(edges))
        bond_logp, edge_logp, aromatic_logp = joint_edge_log_views(joint_logits)
        return {
            "count_logits": self.count_head(condition_context),
            "state_logits": self.state_head(self.head_dropout(nodes)),
            "joint_edge_logits": joint_logits,
            "bond_logits": bond_logp,
            "edge_logits": edge_logp,
            "aromatic_edge_logits": aromatic_logp,
            "coordinate_eps": epsilon * mask[..., None],
        }


def _expected_small_cycles(presence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Expected number of simple 3/4 cycles for independent Bernoulli edges."""
    pair = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    adjacency = presence.masked_fill(~pair | eye, 0)
    adjacency = (adjacency + adjacency.transpose(1, 2)) * .5
    a2 = adjacency @ adjacency
    triangles = (a2 @ adjacency).diagonal(dim1=1, dim2=2).sum(-1) / 6
    trace_a4 = (a2 @ a2).diagonal(dim1=1, dim2=2).sum(-1)
    squared_row_sum = adjacency.square().sum(-1)
    repeated_walks = 2 * squared_row_sum.square().sum(-1) - adjacency.pow(4).sum((1, 2))
    squares = ((trace_a4 - repeated_walks) / 8).clamp_min(0)
    return triangles + squares


def _expected_cumulene_atoms(double_probability: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Exact P(atom has at least two double bonds) under independent edges."""
    pair = mask[:, :, None] & mask[:, None, :]
    probability = double_probability.masked_fill(~pair, 0).clamp(0, 1)
    p0 = torch.ones_like(probability[..., 0])
    p1 = torch.zeros_like(p0)
    for neighbour in range(probability.shape[-1]):
        p = probability[..., neighbour]
        p1, p0 = p1 * (1 - p) + p0 * p, p0 * (1 - p)
    return ((1 - p0 - p1).clamp(0, 1) * mask).sum(-1)


def _soft_cyclic_core(
    presence: torch.Tensor, mask: torch.Tensor, iterations: int = 8,
    temperature: float = 6., core_degree_threshold: float = 1.25,
) -> torch.Tensor:
    """Differentiable 2-core membership derived only from predicted adjacency.

    Repeated soft leaf peeling removes paths while retaining cycle atoms. It is
    used for ring-system and bridgehead surrogates, replacing the disconnected
    topology prediction head used by the first v7 implementation.
    """
    pair = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    adjacency = presence.masked_fill(~pair | eye, 0)
    adjacency = (adjacency + adjacency.transpose(1, 2)) * .5
    core = mask.to(adjacency.dtype)
    for _ in range(iterations):
        degree_in_core = (adjacency * core[:, None, :]).sum(-1)
        # A threshold of 1.5 makes repeated sigmoid updates collapse even a
        # perfect degree-2 cycle to zero. 1.25 retains the high-membership
        # cycle fixed point while still peeling degree-1 paths.
        core = torch.sigmoid(
            temperature * (degree_in_core - core_degree_threshold)
        ) * mask
    return core


def _adjacency_topology_surrogates(presence: torch.Tensor, mask: torch.Tensor):
    """Cycle-rank, ring-system and bridgehead surrogates from adjacency."""
    nodes = mask.sum(-1).to(presence.dtype).clamp_min(1)
    pair = mask[:, :, None] & mask[:, None, :]
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
    adjacency = presence.masked_fill(~pair | eye, 0)
    adjacency = (adjacency + adjacency.transpose(1, 2)) * .5
    degree = adjacency.sum(-1)
    cycle_rank = (adjacency.sum((1, 2)) * .5 - nodes + 1).clamp_min(0)
    cyclic_core = _soft_cyclic_core(adjacency, mask)
    # Soft 2-core mass is an adjacency-coupled approximation of maximum
    # biconnected ring-system size. It cannot be satisfied by a detached head.
    ring_system = cyclic_core.sum(-1)
    bridgehead_probability = cyclic_core * torch.sigmoid(8 * (degree - 2.75))
    bridgeheads = bridgehead_probability.sum(-1)
    return cycle_rank, ring_system, bridgeheads


def _count_conditional_moment_loss(predicted, target, atom_count, maximum_count):
    """Match per-count first/second moments without CPU synchronization."""
    index = atom_count.long()
    scale = atom_count.to(predicted.dtype).clamp_min(1)
    p = predicted / scale
    y = target / scale
    population = predicted.new_zeros(maximum_count).scatter_add(
        0, index, torch.ones_like(predicted),
    )
    p_sum = predicted.new_zeros(maximum_count).scatter_add(0, index, p)
    y_sum = predicted.new_zeros(maximum_count).scatter_add(0, index, y)
    p2_sum = predicted.new_zeros(maximum_count).scatter_add(0, index, p.square())
    y2_sum = predicted.new_zeros(maximum_count).scatter_add(0, index, y.square())
    present = population > 0
    multiple = population > 1
    first = F.smooth_l1_loss(
        p_sum[present] / population[present], y_sum[present] / population[present],
    )
    second = (
        F.smooth_l1_loss(
            p2_sum[multiple] / population[multiple],
            y2_sum[multiple] / population[multiple],
        ) if multiple.any() else predicted.sum() * 0
    )
    return (first + second) * .5


def _count_conditional_distribution_loss(
    predicted, target, atom_count, maximum_count,
):
    """Match the full empirical topology histogram conditional on atom count.

    Linear interpolation between adjacent discrete bins keeps gradients with
    respect to predicted graph statistics while target counts remain exact.
    Empty atom-count rows contribute zero and no host synchronization is used.
    """
    bins = torch.arange(maximum_count, device=predicted.device, dtype=predicted.dtype)
    histogram_value = predicted.clamp(0, maximum_count - 1)
    predicted_weight = (1 - (histogram_value[:, None] - bins[None]).abs()).clamp_min(0)
    target_index = target.round().long().clamp(0, maximum_count - 1)
    target_weight = F.one_hot(target_index, maximum_count).to(predicted.dtype)
    count_index = atom_count.long()[:, None].expand(-1, maximum_count)
    predicted_histogram = predicted.new_zeros(maximum_count, maximum_count)
    target_histogram = predicted.new_zeros(maximum_count, maximum_count)
    predicted_histogram.scatter_add_(0, count_index, predicted_weight)
    target_histogram.scatter_add_(0, count_index, target_weight)
    population = target_histogram.sum(-1, keepdim=True)
    present = (population > 0).to(predicted.dtype)
    predicted_histogram = predicted_histogram / population.clamp_min(1)
    target_histogram = target_histogram / population.clamp_min(1)
    # Normalized one-dimensional Wasserstein-1 distance compares all
    # conditional quantiles and retains a gradient when prediction and target
    # bins do not overlap (unlike total variation on disjoint histograms).
    distance = (
        predicted_histogram.cumsum(-1) - target_histogram.cumsum(-1)
    ).abs().mean(-1)
    return (distance * present.squeeze(-1)).sum() / present.sum().clamp_min(1)


def topology_losses_v7(outputs, batch):
    mask = batch["atom_mask"]
    nodes = mask.sum(-1).float().clamp_min(1)
    bond_probability = F.softmax(outputs["bond_logits"].float(), -1)
    presence = 1 - bond_probability[..., 0]
    predicted_small = _expected_small_cycles(presence, mask)
    predicted_cumulene = _expected_cumulene_atoms(bond_probability[..., 2], mask)
    predicted_cycle_rank, predicted_ring_system, predicted_bridgeheads = (
        _adjacency_topology_surrogates(presence, mask)
    )
    target = batch["topology_target"].float()
    # Columns: cycle rank, maximum ring-system atoms, 3/4-rings,
    # bridgeheads, cumulenes. Pointwise supervision retains each molecule's
    # topology while moment terms preserve the atom-count conditional DB
    # distribution rather than collapsing to one topology per count.
    cycle_target, system_target, small_target, bridge_target, cumulene_target = target.unbind(-1)
    atom_count = mask.sum(-1)
    cycle = (
        F.smooth_l1_loss(predicted_cycle_rank / nodes, cycle_target / nodes)
        + _count_conditional_moment_loss(
            predicted_cycle_rank, cycle_target, atom_count, mask.shape[1] + 1,
        )
        + _count_conditional_distribution_loss(
            predicted_cycle_rank, cycle_target, atom_count, mask.shape[1] + 1,
        )
    )
    ring_system = (
        F.smooth_l1_loss(predicted_ring_system / nodes, system_target / nodes)
        + _count_conditional_moment_loss(
            predicted_ring_system, system_target, atom_count, mask.shape[1] + 1,
        )
        + _count_conditional_distribution_loss(
            predicted_ring_system, system_target, atom_count, mask.shape[1] + 1,
        )
    )
    small_scale = torch.log1p(nodes).clamp_min(1)
    small_match = F.smooth_l1_loss(
        torch.log1p(predicted_small) / small_scale,
        torch.log1p(small_target) / small_scale,
    )
    # Log compression keeps the loss finite for an initially dense random
    # graph while the larger coefficient remains much stronger than old v7.
    small_excess = F.smooth_l1_loss(
        torch.log1p(F.relu(predicted_small - small_target)) / small_scale,
        torch.zeros_like(predicted_small),
    )
    bridgehead = F.smooth_l1_loss(predicted_bridgeheads / nodes, bridge_target / nodes)
    cumulene = F.smooth_l1_loss(predicted_cumulene / nodes, cumulene_target / nodes)
    return {
        "cycle_rank_prior": .35 * cycle,
        "ring_system_prior": .35 * ring_system,
        "small_ring_prior": .5 * (small_match + small_excess),
        "bridgehead_prior": .25 * bridgehead,
        "cumulene_prior": .1 * cumulene,
    }


def joint_training_loss_v7(model, diffusion: JointDiffusion, batch):
    mask = batch["atom_mask"]
    batch_size, atom_count = mask.shape
    timestep = torch.randint(diffusion.timesteps, (batch_size,), device=mask.device)
    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(atom_count, dtype=torch.bool, device=mask.device)[None]
    upper = pair & torch.triu(torch.ones(atom_count, atom_count, dtype=torch.bool, device=mask.device), 1)[None]
    noisy_positions, coordinate_noise = diffusion.q_coords(batch["atom_positions"], timestep, mask)
    clean_edge = encode_joint_edge_state(batch["atom_bonds"], batch["aromatic_edge_mask"])
    noisy_state = diffusion.q_token(batch["atom_state"], timestep, model.state_types, mask, name="atom")
    noisy_edge = diffusion.q_token(clean_edge, timestep, JOINT_EDGE_CLASSES, pair & ~diagonal, True, "joint_edge")
    noisy = {"atom_mask": mask, "atom_state": noisy_state, "edge_state": noisy_edge, "positions": noisy_positions}
    outputs = model(batch["condition"], timestep, noisy)
    aromatic = batch["aromatic_edge_mask"].bool()
    kekule_aromatic = aromatic & ((batch["atom_bonds"] == 1) | (batch["atom_bonds"] == 2))
    aromatic_scalar = outputs["aromatic_edge_logits"][..., 1] - outputs["aromatic_edge_logits"][..., 0]
    positives = aromatic[upper].sum().float()
    negatives = upper.sum().float() - positives
    aromatic_weight = outputs["joint_edge_logits"].new_tensor([1., float((negatives / positives.clamp_min(1)).clamp(1, 50))])
    # A particular RDKit Kekule placement is not a physical target. Aromatic
    # single/double order is supervised by the exact cycle partition below;
    # the local D3PM CE/VLB sees only non-aromatic classes plus the genuinely
    # distinct CECPAD aromatic-triple state.
    resonance_invariant_edge_mask = upper & (~aromatic | (clean_edge == EDGE_AROMATIC_TRIPLE))
    terms = {
        "count": F.cross_entropy(
            outputs["count_logits"], mask.sum(1) - 1, label_smoothing=.05,
        ),
        "state": _ce(outputs["state_logits"], batch["atom_state"], mask),
        "joint_edge": _ce(outputs["joint_edge_logits"], clean_edge, resonance_invariant_edge_mask),
        "aromatic_edge": F.cross_entropy(outputs["aromatic_edge_logits"][upper], aromatic[upper].long(), weight=aromatic_weight),
        "aromatic_cycle": aromatic_cycle_loss(
            outputs["bond_logits"], aromatic_scalar, batch["aromatic_cycle_edges"],
            batch["aromatic_cycle_double_count"], batch["ring_mask"], batch["aromatic_cycle_order"],
            batch["aromatic_cycle_size"], kekule_aromatic, aromatic, upper,
        ),
        "aromatic_component": aromatic_component_loss(outputs, batch),
        "coordinate": ((outputs["coordinate_eps"] - coordinate_noise).square() * mask[..., None]).sum() / mask.sum().clamp_min(1),
        "vlb_state": .1 * diffusion.categorical_vlb(
            batch["atom_state"], noisy_state, outputs["state_logits"], timestep,
            model.state_types, mask, "atom",
        ),
        "vlb_joint_edge": .1 * diffusion.categorical_vlb(
            clean_edge, noisy_edge, outputs["joint_edge_logits"], timestep,
            JOINT_EDGE_CLASSES, resonance_invariant_edge_mask, "joint_edge",
        ),
    }
    chemistry_terms = global_chemistry_losses_v5(outputs, batch, model)
    # The adjacency topology objective below replaces the older duplicated
    # edge-count/cycle-rank surrogate.
    chemistry_terms.pop("cycle_rank", None)
    terms.update(chemistry_terms)
    terms.update(topology_losses_v7(outputs, batch))
    terms.update(geometry_consistency_losses(outputs, batch, noisy_positions, timestep, diffusion))
    return sum(terms.values()), terms


__all__ = [
    "EDGE_NONE", "EDGE_SINGLE", "EDGE_DOUBLE", "EDGE_TRIPLE",
    "EDGE_AROMATIC_SINGLE", "EDGE_AROMATIC_DOUBLE", "EDGE_AROMATIC_TRIPLE", "JOINT_EDGE_CLASSES",
    "encode_joint_edge_state", "decode_joint_edge_state", "joint_edge_log_views",
    "HeavyAtomJointDDPMV7", "joint_training_loss_v7", "topology_losses_v7",
]

