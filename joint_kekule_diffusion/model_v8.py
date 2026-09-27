"""V8 heavy-atom diffusion with one chemically joint categorical edge state."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .chemistry_v5 import AtomStateCodec
from .model import (
    ConditionEmbedding, JointDiffusion, JointLayer, TimeEmbedding, _ce,
    geometry_consistency_losses,
)

EDGE_NONE = 0
EDGE_SINGLE = 1
EDGE_DOUBLE = 2
EDGE_TRIPLE = 3
EDGE_AROMATIC_SINGLE = 4
EDGE_AROMATIC_DOUBLE = 5
EDGE_AROMATIC_TRIPLE = 6
JOINT_EDGE_CLASSES = 7
RING_SIZE_MIN = 3
RING_SIZE_MAX = 12
RING_SIZE_CLASSES = 12


def ring_size_to_class(size):
    """Map zero/non-ring and cycle sizes to the finite v8 vocabulary."""
    value = torch.as_tensor(size).long()
    encoded = (value - (RING_SIZE_MIN - 1)).clamp(1, RING_SIZE_CLASSES - 1)
    return torch.where(value > 0, encoded, torch.zeros_like(encoded))


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


class HeavyAtomJointDDPMV8(nn.Module):
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
        # This head estimates whether a realized edge is cyclic. Final joint
        # projection uses flow connectivity and imposes no spanning-tree gauge.
        self.ring_closure_head = nn.Linear(hidden, 2)
        self.ring_size_head = nn.Linear(hidden, RING_SIZE_CLASSES)
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
        ring_closure_logits = self.ring_closure_head(self.head_dropout(edges))
        ring_size_logits = self.ring_size_head(self.head_dropout(edges))
        bond_logp, edge_logp, aromatic_logp = joint_edge_log_views(joint_logits)
        return {
            "count_logits": self.count_head(condition_context),
            "state_logits": self.state_head(self.head_dropout(nodes)),
            "joint_edge_logits": joint_logits,
            "bond_logits": bond_logp,
            "edge_logits": edge_logp,
            "aromatic_edge_logits": aromatic_logp,
            "ring_closure_logits": ring_closure_logits,
            "ring_size_logits": ring_size_logits,
            "coordinate_eps": epsilon * mask[..., None],
        }



def ring_closure_losses_v8(outputs, batch):
    """Train cyclic-edge candidates and their minimum cycle lengths.

    A unique spanning tree is an arbitrary gauge for symmetric rings. The
    network therefore labels every non-bridge cyclic edge as a possible ring
    closure. Final joint projection scores the realized graph cycles against
    learned minimum-cycle targets and the train-only DB prior.
    """
    mask = batch["atom_mask"]
    count = mask.shape[1]
    pair = mask[:, :, None] & mask[:, None, :]
    upper = pair & torch.triu(
        torch.ones(count, count, dtype=torch.bool, device=mask.device), 1,
    )[None]
    closure_target = batch["ring_closure_target"].bool()
    size_target = batch["ring_size_target"].long()
    positives = closure_target[upper].sum().float()
    negatives = upper.sum().float() - positives
    weight = outputs["ring_closure_logits"].new_tensor([
        1., float((negatives / positives.clamp_min(1)).clamp(1, 50)),
    ])
    closure = F.cross_entropy(
        outputs["ring_closure_logits"][upper], closure_target[upper].long(),
        weight=weight,
    )
    size_counts = torch.bincount(
        size_target[upper], minlength=RING_SIZE_CLASSES,
    ).to(outputs["ring_size_logits"].dtype)
    size_weight = torch.sqrt(size_counts[0] / size_counts.clamp_min(1)).clamp(1, 20)
    size_weight[0] = 1
    ring_size = F.cross_entropy(
        outputs["ring_size_logits"][upper], size_target[upper], weight=size_weight,
    )

    edge_probability = outputs["edge_logits"].float().softmax(-1)[..., 1]
    closure_probability = outputs["ring_closure_logits"].float().softmax(-1)[..., 1]
    aromatic_probability = outputs["aromatic_edge_logits"].float().softmax(-1)[..., 1]
    consistency = (
        F.relu(closure_probability - edge_probability).square()[upper].mean()
        + F.relu(aromatic_probability - closure_probability).square()[upper].mean()
    )

    return {
        "ring_closure": .35 * closure,
        "ring_size": .35 * ring_size,
        "ring_head_consistency": .2 * consistency,
    }
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


def topology_losses_v8(outputs, batch):
    """Retain non-ring topology regularizers without duplicating closure loss."""
    mask = batch["atom_mask"]
    nodes = mask.sum(-1).float().clamp_min(1)
    bond_probability = F.softmax(outputs["bond_logits"].float(), -1)
    presence = 1 - bond_probability[..., 0]
    predicted_cumulene = _expected_cumulene_atoms(bond_probability[..., 2], mask)
    target = batch["topology_target"].float()
    _, _, _, _, cumulene_target = target.unbind(-1)
    return {
        "cumulene_prior": .1 * F.smooth_l1_loss(
            predicted_cumulene / nodes, cumulene_target / nodes,
        ),
    }



def aromatic_system_loss_v8(outputs, batch):
    """Resonance-invariant Kekule supervision over whole aromatic systems.

    Ring-local double counts are not invariant for fused systems. Per-atom
    aromatic double-bond incidence and the component-wide total are invariant
    under equivalent Kekule placements in this dataset, while RDKit performs
    the final global existence check during projection.
    """
    aromatic = batch["aromatic_edge_mask"].bool()
    eligible = aromatic & ((batch["atom_bonds"] == 1) | (batch["atom_bonds"] == 2))
    zero = outputs["bond_logits"].sum() * 0
    if not eligible.any():
        return zero
    bond_probability = F.softmax(outputs["bond_logits"].float(), -1)
    allowed_probability = bond_probability[..., 1:3].sum(-1).clamp_min(1e-8)
    allowed = -allowed_probability[eligible].log().mean()
    predicted_double = (bond_probability[..., 2] * eligible).sum(-1)
    target_double = ((batch["atom_bonds"] == 2) & eligible).sum(-1).float()
    aromatic_atoms = eligible.any(-1)
    incidence = F.smooth_l1_loss(
        predicted_double[aromatic_atoms], target_double[aromatic_atoms],
    )
    predicted_total = predicted_double.sum(-1) * .5
    target_total = target_double.sum(-1) * .5
    total = F.smooth_l1_loss(predicted_total, target_total)
    return allowed + .5 * incidence + .2 * total


def _normalized_algebraic_connectivity(adjacency: torch.Tensor) -> torch.Tensor:
    """Second eigenvalue of the symmetric normalized graph Laplacian."""
    degree = adjacency.sum(-1).clamp_min(1e-8)
    inverse_root = degree.rsqrt()
    normalized = adjacency * inverse_root[:, :, None] * inverse_root[:, None, :]
    identity = torch.eye(
        adjacency.shape[-1], dtype=adjacency.dtype, device=adjacency.device,
    )[None]
    eigenvalues = torch.linalg.eigvalsh(identity - normalized)
    return eigenvalues[:, 1] if adjacency.shape[-1] > 1 else eigenvalues[:, 0]


def global_chemistry_losses_v8(outputs, batch, model):
    """Valence, charge, edge-count and connectivity from the predicted graph.

    Unlike v5, all predicted edges contribute to valence. This makes a
    false-positive bond chemically expensive instead of hiding it behind the
    clean target neighbour mask.
    """
    mask = batch["atom_mask"]
    atom_count = mask.shape[1]
    pair = (
        mask[:, :, None] & mask[:, None, :]
        & ~torch.eye(atom_count, dtype=torch.bool, device=mask.device)[None]
    )
    state_probability = F.softmax(outputs["state_logits"].float(), -1)
    bond_probability = F.softmax(outputs["bond_logits"].float(), -1)
    orders = bond_probability.new_tensor([0., 1., 2., 3.])
    expected_bond = (bond_probability * orders).sum(-1) * pair
    presence = (1 - bond_probability[..., 0]) * pair
    target_presence = (batch["atom_bonds"] > 0) * pair

    expected_valence = (state_probability * model.state_valence_value).sum(-1)
    expected_h_min = (state_probability * model.state_h_min).sum(-1)
    expected_h_max = (state_probability * model.state_h_max).sum(-1)
    residual_hydrogen = expected_valence - expected_bond.sum(-1)
    hydrogen_target = batch["atom_hydrogen"].float()
    implicit_hydrogen = F.smooth_l1_loss(
        residual_hydrogen[mask], hydrogen_target[mask],
    )
    # Zero inside the supported hydrogen interval. Smooth-L1 is quadratic
    # near either boundary but only linear for a grossly dense random graph,
    # so this chemical term cannot drown the categorical denoising signal.
    lower_violation = F.relu(expected_h_min - residual_hydrogen)
    upper_violation = F.relu(residual_hydrogen - expected_h_max)
    support_violation = (
        F.smooth_l1_loss(lower_violation[mask], torch.zeros_like(lower_violation[mask]))
        + F.smooth_l1_loss(upper_violation[mask], torch.zeros_like(upper_violation[mask]))
    )

    expected_charge = (state_probability * model.state_charge_value).sum(-1) * mask
    target_charge = (batch["atom_charge"].float() - 2).mul(mask).sum(-1)
    charge_balance = F.smooth_l1_loss(expected_charge.sum(-1), target_charge)
    expected_aromatic = (state_probability * model.state_aromatic_value).sum(-1)
    target_aromatic = batch["aromatic_edge_mask"].any(-1).float()
    aromatic_probability = expected_aromatic[mask].clamp(1e-6, 1 - 1e-6)
    aromatic_target = target_aromatic[mask]
    aromatic_role = -(
        aromatic_target * aromatic_probability.log()
        + (1 - aromatic_target) * torch.log1p(-aromatic_probability)
    ).mean()

    degree = F.smooth_l1_loss(
        presence.sum(-1)[mask], target_presence.sum(-1).float()[mask],
    )
    nodes = mask.sum(-1).float().clamp_min(1)
    edge_count = F.smooth_l1_loss(
        presence.sum((1, 2)) * .5 / nodes,
        target_presence.sum((1, 2)).float() * .5 / nodes,
    )
    connectivity_terms = []
    for count in torch.unique(mask.sum(-1)).tolist():
        rows = mask.sum(-1) == count
        size = int(count)
        if size <= 1:
            continue
        predicted = _normalized_algebraic_connectivity(
            presence[rows, :size, :size],
        )
        target = _normalized_algebraic_connectivity(
            target_presence[rows, :size, :size].float(),
        )
        connectivity_terms.append(F.smooth_l1_loss(
            predicted, target, reduction="none",
        ))
    connectivity = (
        torch.cat(connectivity_terms).mean()
        if connectivity_terms else outputs["bond_logits"].sum() * 0
    )
    return {
        "implicit_hydrogen": .35 * implicit_hydrogen,
        "valence_support": .5 * support_violation,
        "charge_balance": .2 * charge_balance,
        "aromatic_role": .1 * aromatic_role,
        "degree": .1 * degree,
        "edge_count": .1 * edge_count,
        "connectivity": .2 * connectivity,
    }
def joint_training_loss_v8(model, diffusion: JointDiffusion, batch):
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
    positives = aromatic[upper].sum().float()
    negatives = upper.sum().float() - positives
    aromatic_weight = outputs["joint_edge_logits"].new_tensor([
        1., float((negatives / positives.clamp_min(1)).clamp(1, 50)),
    ])
    # A particular RDKit Kekule placement is not a physical target. Local
    # CE/VLB excludes aromatic single/double placement; the system loss below
    # supervises resonance-invariant incidence. Aromatic triples remain local.
    resonance_invariant_edge_mask = upper & (
        ~aromatic | (clean_edge == EDGE_AROMATIC_TRIPLE)
    )
    class_counts = torch.bincount(
        clean_edge[resonance_invariant_edge_mask], minlength=JOINT_EDGE_CLASSES,
    ).to(outputs["joint_edge_logits"].dtype)
    joint_edge_weight = torch.sqrt(
        class_counts.max().clamp_min(1) / class_counts.clamp_min(1)
    ).clamp(1, 20)
    terms = {
        "count": F.cross_entropy(
            outputs["count_logits"], mask.sum(1) - 1, label_smoothing=.05,
        ),
        "state": _ce(outputs["state_logits"], batch["atom_state"], mask),
        "joint_edge": _ce(
            outputs["joint_edge_logits"], clean_edge,
            resonance_invariant_edge_mask, joint_edge_weight,
        ),
        "aromatic_edge": F.cross_entropy(
            outputs["aromatic_edge_logits"][upper], aromatic[upper].long(),
            weight=aromatic_weight,
        ),
        "aromatic_system": aromatic_system_loss_v8(outputs, batch),
        "coordinate": (
            (outputs["coordinate_eps"] - coordinate_noise).square()
            * mask[..., None]
        ).sum() / mask.sum().clamp_min(1),
        "vlb_state": .1 * diffusion.categorical_vlb(
            batch["atom_state"], noisy_state, outputs["state_logits"], timestep,
            model.state_types, mask, "atom",
        ),
        "vlb_joint_edge": .1 * diffusion.categorical_vlb(
            clean_edge, noisy_edge, outputs["joint_edge_logits"], timestep,
            JOINT_EDGE_CLASSES, resonance_invariant_edge_mask, "joint_edge",
        ),
    }
    terms.update(global_chemistry_losses_v8(outputs, batch, model))
    terms.update(ring_closure_losses_v8(outputs, batch))
    terms.update(topology_losses_v8(outputs, batch))
    terms.update(geometry_consistency_losses(
        outputs, batch, noisy_positions, timestep, diffusion,
    ))
    return sum(terms.values()), terms


__all__ = [
    "EDGE_NONE", "EDGE_SINGLE", "EDGE_DOUBLE", "EDGE_TRIPLE",
    "EDGE_AROMATIC_SINGLE", "EDGE_AROMATIC_DOUBLE", "EDGE_AROMATIC_TRIPLE", "JOINT_EDGE_CLASSES",
    "encode_joint_edge_state", "decode_joint_edge_state", "joint_edge_log_views",
    "RING_SIZE_MIN", "RING_SIZE_MAX", "RING_SIZE_CLASSES", "ring_size_to_class",
    "HeavyAtomJointDDPMV8", "joint_training_loss_v8", "ring_closure_losses_v8",
    "topology_losses_v8",
]
