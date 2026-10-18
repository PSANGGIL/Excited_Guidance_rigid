"""E(3)-invariant residual self-conditioning for molecular endpoint models.

The previous endpoint estimate is deliberately kept outside the graph state.
Categorical estimates are probabilities and coordinate feedback enters only
through distances, so the residual cannot leak an absolute frame into the GVP.
"""

from __future__ import annotations

from collections.abc import Sequence

import dgl
import dgl.function as fn
import torch
from torch import nn

from molguidance.models.gvp import _norm_no_nan, _rbf


class SelfConditioningResidualLayer(nn.Module):
    """Add residual node/edge embeddings from a previous endpoint estimate.

    Parameters are learned only when self-conditioning is enabled.  Passing
    ``previous_endpoint=None`` is an exact identity operation, which is needed
    for the first reverse step and the non-self-conditioned half of training.
    """

    def __init__(
        self,
        node_feature_names: Sequence[str],
        node_feature_dims: Sequence[int],
        edge_probability_dim: int,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        rbf_dim: int,
        rbf_dmax: float,
    ) -> None:
        super().__init__()
        if len(node_feature_names) != len(node_feature_dims):
            raise ValueError("node feature names and dimensions must have equal length")
        if not node_feature_names:
            raise ValueError("self-conditioning requires a categorical node feature")
        if any(int(width) <= 0 for width in node_feature_dims):
            raise ValueError("self-conditioning category dimensions must be positive")
        if edge_probability_dim <= 0 or rbf_dim <= 0 or rbf_dmax <= 0:
            raise ValueError("edge dimension and RBF settings must be positive")

        self.node_feature_names = tuple(node_feature_names)
        self.node_feature_dims = tuple(int(width) for width in node_feature_dims)
        self.edge_probability_dim = int(edge_probability_dim)
        self.rbf_dim = int(rbf_dim)
        self.rbf_dmax = float(rbf_dmax)

        node_input_dim = (
            int(node_hidden_dim) + sum(self.node_feature_dims) + self.rbf_dim
        )
        edge_input_dim = (
            int(edge_hidden_dim) + self.edge_probability_dim + self.rbf_dim
        )
        self.node_residual = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            nn.SiLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
        )
        self.edge_residual = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )

    @staticmethod
    def _validate_probability_tensor(
        value: torch.Tensor,
        expected_shape: tuple[int, int],
        name: str,
    ) -> torch.Tensor:
        if value.shape != expected_shape:
            raise ValueError(
                f"self-conditioning {name!r} must have shape {expected_shape}, "
                f"got {tuple(value.shape)}"
            )
        # Numerical canonicalization happens before this layer. Avoid Python
        # bool reductions here: they would synchronize GPU and CPU at every
        # reverse step.
        return value.detach()

    def forward(
        self,
        g: dgl.DGLGraph,
        node_hidden: torch.Tensor,
        edge_hidden: torch.Tensor,
        current_positions: torch.Tensor,
        upper_edge_mask: torch.Tensor,
        previous_endpoint: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if previous_endpoint is None:
            return node_hidden, edge_hidden

        expected_keys = {"x", "e", *self.node_feature_names}
        missing = expected_keys - set(previous_endpoint)
        if missing:
            raise KeyError(
                "self-conditioning endpoint is missing: " + ", ".join(sorted(missing))
            )
        previous_positions = previous_endpoint["x"]
        if previous_positions.shape != current_positions.shape:
            raise ValueError(
                "self-conditioning coordinates must match current coordinates: "
                f"{tuple(previous_positions.shape)} != {tuple(current_positions.shape)}"
            )
        previous_positions = previous_positions.detach()

        node_probabilities = []
        for name, width in zip(self.node_feature_names, self.node_feature_dims):
            node_probabilities.append(
                self._validate_probability_tensor(
                    previous_endpoint[name], (g.num_nodes(), width), name
                )
            )
        node_probability_features = torch.cat(node_probabilities, dim=-1)

        if g.num_edges() % 2:
            raise ValueError(
                "self-conditioning expects paired directed edges for each upper edge"
            )
        if upper_edge_mask.shape != (g.num_edges(),):
            raise ValueError(
                "upper_edge_mask must contain one entry per directed edge"
            )
        n_upper_edges = g.num_edges() // 2
        upper_edge_probabilities = self._validate_probability_tensor(
            previous_endpoint["e"],
            (n_upper_edges, self.edge_probability_dim),
            "e",
        )
        edge_probability_features = edge_hidden.new_zeros(
            (g.num_edges(), self.edge_probability_dim)
        )
        edge_probability_features[upper_edge_mask] = upper_edge_probabilities
        edge_probability_features[~upper_edge_mask] = upper_edge_probabilities

        displacement = _norm_no_nan(
            previous_positions - current_positions, keepdims=False
        )
        node_distance_features = _rbf(
            displacement, D_max=self.rbf_dmax, D_count=self.rbf_dim
        )

        with g.local_scope():
            g.ndata["sc_current_x"] = current_positions
            g.ndata["sc_previous_x"] = previous_positions
            g.apply_edges(fn.u_sub_v("sc_current_x", "sc_current_x", "sc_current_dx"))
            g.apply_edges(fn.u_sub_v("sc_previous_x", "sc_previous_x", "sc_previous_dx"))
            current_distances = _norm_no_nan(g.edata["sc_current_dx"])
            previous_distances = _norm_no_nan(g.edata["sc_previous_dx"])
        edge_distance_features = _rbf(
            previous_distances, D_max=self.rbf_dmax, D_count=self.rbf_dim
        ) - _rbf(
            current_distances, D_max=self.rbf_dmax, D_count=self.rbf_dim
        )

        node_update = self.node_residual(
            torch.cat(
                (node_hidden, node_probability_features, node_distance_features),
                dim=-1,
            )
        )
        edge_update = self.edge_residual(
            torch.cat(
                (edge_hidden, edge_probability_features, edge_distance_features),
                dim=-1,
            )
        )
        return node_hidden + node_update, edge_hidden + edge_update
