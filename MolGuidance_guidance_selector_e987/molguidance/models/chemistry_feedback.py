"""Detached, permutation-invariant atom/edge chemistry feedback.

No target, property label, RDKit repair, or hard projection is used here.
The residual supplies contradictions in the PREVIOUS prediction to the next
denoising pass. It is not a guarantee of a globally valid Kekule assignment.
"""
import torch
from torch import nn

from molguidance.data_processing.utils import get_batch_idxs


class JointChemicalFeedback(nn.Module):
    def __init__(self, state_metadata, node_hidden_dim, edge_hidden_dim):
        super().__init__()
        if not state_metadata:
            raise ValueError("chemistry feedback requires joint atom-state metadata")
        features = []
        for state in state_metadata:
            valences = [int(v) for v, count in state["train_heavy_valence_counts"].items() if count > 0]
            features.append([
                min(valences, default=0), max(valences, default=0),
                state["hydrogens"], state["formal_charge"],
                float(state.get("is_fake", False)),
            ])
        self.register_buffer("state_features", torch.tensor(features, dtype=torch.float32), persistent=False)
        self.node_residual = nn.Sequential(
            nn.Linear(10, node_hidden_dim), nn.SiLU(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
        )
        self.edge_residual = nn.Sequential(
            nn.Linear(24, edge_hidden_dim), nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )
        # New feedback begins as an exact identity; old parameters are not
        # perturbed by arbitrary untrained corrections.
        for net in (self.node_residual, self.edge_residual):
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def features(self, g, upper_edge_mask, previous):
        node = previous["s"].detach()
        edge = previous["e"].detach()
        if node.shape != (g.num_nodes(), self.state_features.shape[0]):
            raise ValueError("feedback state probabilities do not match metadata")
        src, dst = g.edges()
        src, dst = src[upper_edge_mask], dst[upper_edge_mask]
        if edge.shape != (src.numel(), 4):
            raise ValueError("feedback requires four Kekule edge probabilities")
        orders = edge.new_tensor([0., 1., 2., 3.])
        mean_order = edge @ orders
        variance = (edge @ orders.square() - mean_order.square()).clamp_min(0.)
        edge_summary = torch.stack((mean_order, variance, 1. - edge[:, 0]), -1)
        totals = node.new_zeros(g.num_nodes(), 3)
        totals.index_add_(0, src, edge_summary)
        totals.index_add_(0, dst, edge_summary)
        state = node @ self.state_features.to(node)
        node_batch, _ = get_batch_idxs(g)
        net_charge = node.new_zeros(g.batch_size)
        net_charge.index_add_(0, node_batch, state[:, 3])
        feat = torch.stack((
            totals[:, 0] / 8., totals[:, 1].sqrt() / 8., totals[:, 2] / 8.,
            state[:, 0] / 8., state[:, 1] / 8.,
            (totals[:, 0] - .5 * (state[:, 0] + state[:, 1])) / 8.,
            state[:, 2] / 4., state[:, 3] / 4.,
            net_charge[node_batch] / 8., state[:, 4],
        ), -1)
        # Symmetric endpoint features prevent directional edge discrepancies.
        pair_feat = torch.cat((feat[src] + feat[dst], (feat[src] - feat[dst]).abs(), edge), -1)
        return feat, pair_feat

    def forward(self, g, node_hidden, edge_hidden, upper_edge_mask, previous):
        if previous is None:
            return node_hidden, edge_hidden
        node_feat, upper_feat = self.features(g, upper_edge_mask, previous)
        upper_update = self.edge_residual(upper_feat)
        edge_update = torch.zeros_like(edge_hidden)
        edge_update[upper_edge_mask] = upper_update
        edge_update[~upper_edge_mask] = upper_update
        return node_hidden + self.node_residual(node_feat), edge_hidden + edge_update
