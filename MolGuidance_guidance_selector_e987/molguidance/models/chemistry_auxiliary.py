"""Differentiable chemistry consistency losses for joint atom/edge endpoints."""

from __future__ import annotations

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from molguidance.data_processing.utils import get_batch_idxs


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a finite weighted mean, including for an empty active set."""
    if active is None:
        active = torch.ones_like(values, dtype=torch.bool)
    if values.ndim != 1 or weights.shape != values.shape or active.shape != values.shape:
        raise ValueError("values, weights, and active must be aligned vectors")
    active_float = active.to(weights.dtype)
    denominator = active_float.sum()
    if not bool(denominator > 0):
        return values.sum() * 0.0
    # Keep time scaling mathematically identical to mean(w_i * L_i).
    # Dividing by sum(w_i) would silently cancel a common timestep weight.
    return (values * weights * active_float).sum() / denominator


class JointChemistryAuxiliaryLoss(nn.Module):
    """Score coupled atom-state and Kekule-edge endpoint probabilities.

    The current CTMC state can already reveal some endpoint categories. For
    those entries the exact observed one-hot value is used; model probabilities
    are used only for masked entries. This keeps the global chemistry losses
    aligned with the conditional denoising problem and prevents a trivial copy
    objective from dominating the auxiliary terms.
    """

    loss_keys = (
        "chem_edge_existence",
        "chem_valence",
        "chem_charge",
        "chem_connectivity",
        "chem_aromatic",
    )

    def __init__(
        self,
        state_charges: torch.Tensor,
        state_valence_probabilities: torch.Tensor,
        state_support: torch.Tensor,
        state_is_fake: torch.Tensor | None = None,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        state_charges = torch.as_tensor(state_charges, dtype=torch.float32)
        valence_probabilities = torch.as_tensor(
            state_valence_probabilities, dtype=torch.float32
        )
        state_support = torch.as_tensor(state_support, dtype=torch.bool)
        if state_is_fake is None:
            state_is_fake = torch.zeros_like(state_support)
        else:
            state_is_fake = torch.as_tensor(state_is_fake, dtype=torch.bool)
        if state_charges.ndim != 1:
            raise ValueError("state_charges must be one-dimensional")
        if valence_probabilities.ndim != 2:
            raise ValueError("state_valence_probabilities must be a matrix")
        if valence_probabilities.shape[0] != state_charges.numel():
            raise ValueError("state charge and valence vocabularies disagree")
        if state_support.shape != state_charges.shape:
            raise ValueError("state_support must match the atom-state vocabulary")
        if state_is_fake.shape != state_charges.shape:
            raise ValueError("state_is_fake must match the atom-state vocabulary")
        if int(state_is_fake.sum()) > 1:
            raise ValueError("at most one synthetic fake atom state is supported")
        if not bool(state_support.any()):
            raise ValueError("at least one atom state must be train-supported")
        supported_rows = valence_probabilities[state_support]
        if not torch.allclose(
            supported_rows.sum(dim=-1),
            torch.ones(supported_rows.shape[0]),
            atol=1.0e-6,
        ):
            raise ValueError(
                "every train-supported state needs a normalized valence distribution"
            )
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.eps = float(eps)
        self.register_buffer("state_charges", state_charges, persistent=False)
        self.register_buffer(
            "state_valence_probabilities",
            valence_probabilities,
            persistent=False,
        )
        self.register_buffer("state_support", state_support, persistent=False)
        self.register_buffer("state_is_fake", state_is_fake, persistent=False)

    def _effective_endpoint_probabilities(
        self,
        g: dgl.DGLGraph,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        upper_edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_states = self.state_charges.numel()
        if node_logits.shape != (g.num_nodes(), n_states):
            raise ValueError(
                "node logits do not match graph/vocabulary dimensions: "
                f"{tuple(node_logits.shape)}"
            )
        n_upper_edges = int(upper_edge_mask.sum())
        if edge_logits.shape != (n_upper_edges, 4):
            raise ValueError(
                "joint Kekule edge logits must have shape "
                f"({n_upper_edges}, 4), got {tuple(edge_logits.shape)}"
            )
        if g.ndata["s_t"].shape[-1] != n_states + 1:
            raise ValueError("s_t must contain one residual CTMC mask class")
        if g.edata["e_t"].shape[-1] != 5:
            raise ValueError("e_t must contain four edge classes plus CTMC mask")

        if not torch.isfinite(node_logits).all() or not torch.isfinite(
            edge_logits
        ).all():
            raise FloatingPointError("non-finite joint chemistry logits")
        supported_logits = node_logits.masked_fill(
            ~self.state_support.unsqueeze(0), -torch.inf
        )
        node_probabilities = F.softmax(supported_logits, dim=-1)
        node_state_t = g.ndata["s_t"].argmax(dim=-1)
        node_is_masked = node_state_t == n_states
        known_node = g.ndata["s_t"][:, :n_states].to(node_probabilities.dtype)
        effective_node = torch.where(
            node_is_masked.unsqueeze(-1), node_probabilities, known_node
        )

        edge_probabilities = F.softmax(edge_logits, dim=-1)
        edge_t_upper = g.edata["e_t"][upper_edge_mask]
        edge_state_t = edge_t_upper.argmax(dim=-1)
        edge_is_masked = edge_state_t == 4
        known_edge = edge_t_upper[:, :4].to(edge_probabilities.dtype)
        effective_edge = torch.where(
            edge_is_masked.unsqueeze(-1), edge_probabilities, known_edge
        )
        return effective_node, effective_edge, edge_is_masked

    def forward(
        self,
        g: dgl.DGLGraph,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        graph_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if upper_edge_mask.dtype != torch.bool or upper_edge_mask.shape != (
            g.num_edges(),
        ):
            raise ValueError("upper_edge_mask must be a boolean mask over graph edges")
        expected_upper = sum(
            int(n) * (int(n) - 1) // 2 for n in g.batch_num_nodes().tolist()
        )
        if int(upper_edge_mask.sum()) != expected_upper:
            raise ValueError("graph is not packed as complete upper/lower edge pairs")

        batch_size = g.batch_size
        if graph_weights is None:
            graph_weights = node_logits.new_ones(batch_size)
        else:
            graph_weights = torch.as_tensor(
                graph_weights, device=node_logits.device, dtype=node_logits.dtype
            )
        if graph_weights.shape != (batch_size,):
            raise ValueError(f"graph_weights must have shape ({batch_size},)")
        if not torch.isfinite(graph_weights).all() or bool((graph_weights < 0).any()):
            raise ValueError("graph_weights must be finite and non-negative")

        node_batch_idx, edge_batch_idx = get_batch_idxs(g)
        upper_edge_batch_idx = edge_batch_idx[upper_edge_mask]
        effective_node, effective_edge, edge_is_masked = (
            self._effective_endpoint_probabilities(
                g, node_logits, edge_logits, upper_edge_mask
            )
        )
        predicted_edge = F.softmax(edge_logits, dim=-1)
        target_edge_class = g.edata["e_1_true"][upper_edge_mask].argmax(dim=-1)
        target_edge_exists = target_edge_class != 0

        # Balanced binary edge-existence loss on unknown pairs only. This is an
        # auxiliary view of the same joint edge state, not an independent head.
        predicted_edge_exists = (1.0 - predicted_edge[:, 0]).clamp(
            self.eps, 1.0 - self.eps
        )
        existence_nll = torch.where(
            target_edge_exists,
            -torch.log(predicted_edge_exists),
            -torch.log1p(-predicted_edge_exists),
        )
        edge_item_weights = graph_weights[upper_edge_batch_idx]
        positive_loss = _weighted_mean(
            existence_nll,
            edge_item_weights,
            edge_is_masked & target_edge_exists,
        )
        negative_loss = _weighted_mean(
            existence_nll,
            edge_item_weights,
            edge_is_masked & ~target_edge_exists,
        )
        has_positive = bool((edge_is_masked & target_edge_exists).any())
        has_negative = bool((edge_is_masked & ~target_edge_exists).any())
        if has_positive and has_negative:
            edge_existence_loss = 0.5 * (positive_loss + negative_loss)
        elif has_positive:
            edge_existence_loss = positive_loss
        else:
            edge_existence_loss = negative_loss

        # Expected heavy-atom bond-order valence at every node.
        upper_src, upper_dst = g.edges()
        upper_src = upper_src[upper_edge_mask]
        upper_dst = upper_dst[upper_edge_mask]
        bond_orders = edge_logits.new_tensor([0.0, 1.0, 2.0, 3.0])
        expected_order = effective_edge @ bond_orders
        expected_valence = edge_logits.new_zeros(g.num_nodes())
        expected_valence.index_add_(0, upper_src, expected_order)
        expected_valence.index_add_(0, upper_dst, expected_order)

        supported_probabilities = effective_node[:, self.state_support]
        supported_valence = self.state_valence_probabilities[self.state_support]
        valence_values = torch.arange(
            supported_valence.shape[-1],
            device=edge_logits.device,
            dtype=edge_logits.dtype,
        )
        squared_error = (
            expected_valence[:, None, None] - valence_values[None, None, :]
        ).square()
        # A state may have several chemically supported heavy-valence modes
        # (notably P and S). Penalize distance to the closest observed mode;
        # weighting modes by their training frequency would incorrectly make a
        # rare but valid valence incur non-zero oracle loss.
        allowed_valence = supported_valence > 0
        state_valence_energy = torch.where(
            allowed_valence.unsqueeze(0),
            squared_error,
            torch.full_like(squared_error, torch.inf),
        ).amin(dim=-1)
        node_valence_energy = (
            supported_probabilities * state_valence_energy
        ).sum(dim=-1)
        valence_active = supported_probabilities.sum(dim=-1) > self.eps
        valence_loss = _weighted_mean(
            node_valence_energy,
            graph_weights[node_batch_idx],
            valence_active,
        )

        # Preserve the true net charge rather than forcing every atom neutral;
        # this remains compatible with zwitterions and charged molecules.
        predicted_charge = effective_node @ self.state_charges
        true_charge = g.ndata["s_1_true"].to(node_logits.dtype) @ self.state_charges
        predicted_charge_by_graph = node_logits.new_zeros(batch_size)
        true_charge_by_graph = node_logits.new_zeros(batch_size)
        predicted_charge_by_graph.index_add_(0, node_batch_idx, predicted_charge)
        true_charge_by_graph.index_add_(0, node_batch_idx, true_charge)
        target_state = g.ndata["s_1_true"].argmax(dim=-1)
        target_is_real = ~self.state_is_fake[target_state]
        if "is_real_1_true" in g.ndata:
            labelled_is_real = g.ndata["is_real_1_true"].bool()
            if labelled_is_real.shape != target_is_real.shape or not torch.equal(
                labelled_is_real, target_is_real
            ):
                raise ValueError(
                    "is_real_1_true disagrees with the joint fake-state target"
                )
        atom_counts = node_logits.new_zeros(batch_size)
        atom_counts.index_add_(
            0, node_batch_idx, target_is_real.to(node_logits.dtype)
        )
        atom_counts = atom_counts.clamp_min(1.0)
        charge_error = (
            predicted_charge_by_graph - true_charge_by_graph
        ).square() / atom_counts
        charge_loss = _weighted_mean(charge_error, graph_weights)

        if "e_connectivity_weight" not in g.edata:
            raise ValueError(
                "chemistry auxiliary loss requires e_connectivity_weight"
            )
        connectivity_weight = g.edata["e_connectivity_weight"][
            upper_edge_mask
        ].to(edge_logits.dtype)
        if connectivity_weight.shape != predicted_edge_exists.shape:
            raise ValueError("connectivity weights do not align with upper edges")
        if not torch.isfinite(connectivity_weight).all() or bool(
            (connectivity_weight < 0).any()
        ):
            raise ValueError("connectivity weights must be finite and non-negative")
        connectivity_edge_loss = connectivity_weight * -torch.log(
            (1.0 - effective_edge[:, 0]).clamp_min(self.eps)
        )
        connectivity_sum = edge_logits.new_zeros(batch_size)
        connectivity_norm = edge_logits.new_zeros(batch_size)
        connectivity_sum.index_add_(
            0, upper_edge_batch_idx, connectivity_edge_loss
        )
        connectivity_norm.index_add_(
            0, upper_edge_batch_idx, connectivity_weight
        )
        connectivity_per_graph = connectivity_sum / connectivity_norm.clamp_min(
            self.eps
        )
        connectivity_loss = _weighted_mean(
            connectivity_per_graph,
            graph_weights,
            connectivity_norm > self.eps,
        )

        if "e_aromatic_true" not in g.edata:
            raise ValueError("chemistry auxiliary loss requires e_aromatic_true")
        aromatic_mask = g.edata["e_aromatic_true"][upper_edge_mask].bool()
        aromatic_probability = (
            effective_edge[:, 1] + effective_edge[:, 2]
        ).clamp_min(self.eps)
        aromatic_perimeter_nll = -torch.log(aromatic_probability)
        aromatic_perimeter_loss = _weighted_mean(
            aromatic_perimeter_nll,
            edge_item_weights,
            aromatic_mask,
        )

        aromatic_float = aromatic_mask.to(edge_logits.dtype)
        aromatic_incidence = edge_logits.new_zeros(g.num_nodes())
        predicted_double_degree = edge_logits.new_zeros(g.num_nodes())
        true_double_degree = edge_logits.new_zeros(g.num_nodes())
        predicted_aromatic_double = effective_edge[:, 2] * aromatic_float
        true_aromatic_double = (
            (target_edge_class == 2) & aromatic_mask
        ).to(edge_logits.dtype)
        for node_index in (upper_src, upper_dst):
            aromatic_incidence.index_add_(0, node_index, aromatic_float)
            predicted_double_degree.index_add_(
                0, node_index, predicted_aromatic_double
            )
            true_double_degree.index_add_(
                0, node_index, true_aromatic_double
            )
        # Alternative Kekule matchings generally preserve the required number
        # of incident double bonds at each aromatic atom. This enforces local
        # alternating feasibility without selecting one perimeter assignment.
        aromatic_degree_error = (
            predicted_double_degree - true_double_degree
        ).square()
        aromatic_degree_loss = _weighted_mean(
            aromatic_degree_error,
            graph_weights[node_batch_idx],
            aromatic_incidence > 0,
        )
        aromatic_loss = aromatic_perimeter_loss + aromatic_degree_loss

        losses = {
            "chem_edge_existence": edge_existence_loss,
            "chem_valence": valence_loss,
            "chem_charge": charge_loss,
            "chem_connectivity": connectivity_loss,
            "chem_aromatic": aromatic_loss,
        }
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise FloatingPointError("non-finite chemistry auxiliary loss")
        return losses
