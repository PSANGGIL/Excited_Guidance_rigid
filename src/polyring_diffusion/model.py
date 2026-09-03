"""Dense equivariant denoiser over a padded ring graph."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .chem import element_record
from .diffusion import SinusoidalTimeEmbedding
from .graph import ConnectionType
from .templates import DEFAULT_LIBRARY, TemplateLibrary


def template_feature_table(library: TemplateLibrary = DEFAULT_LIBRARY) -> Tensor:
    rows: list[list[float]] = []
    for template in library:
        vdw = [element_record(symbol).vdw_radius for symbol in template.atoms]
        covalent = [element_record(symbol).covalent_radius for symbol in template.atoms]
        hetero = sum(symbol != "C" for symbol in template.atoms)
        rows.append(
            [
                template.size / 8.0,
                sum(vdw) / len(vdw) / 2.0,
                sum(covalent) / len(covalent) / 1.2,
                hetero / template.size,
            ]
        )
    table = torch.tensor(rows, dtype=torch.float32)
    mean = table.mean(dim=0, keepdim=True)
    pad = torch.zeros_like(mean)
    return torch.cat([table, mean, pad], dim=0)


class DenseEquivariantLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        edge_dim = hidden_dim * 3 + 2
        self.message = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.center_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.normal_relative_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.normal_self_weight = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden: Tensor,
        centers: Tensor,
        normals: Tensor,
        connection_embedding: Tensor,
        physical_radius: Tensor,
        node_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, nodes, _ = hidden.shape
        left = hidden[:, :, None, :].expand(batch, nodes, nodes, -1)
        right = hidden[:, None, :, :].expand(batch, nodes, nodes, -1)
        relative = centers[:, :, None, :] - centers[:, None, :, :]
        distance2 = relative.square().sum(dim=-1, keepdim=True)
        pair_scale = physical_radius[:, :, None, :] + physical_radius[:, None, :, :]
        normalized_distance2 = distance2 / pair_scale.square().clamp_min(1e-6)
        normal_dot = (normals[:, :, None, :] * normals[:, None, :, :]).sum(dim=-1, keepdim=True)
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        pair_mask &= ~torch.eye(nodes, dtype=torch.bool, device=hidden.device)[None]
        message = self.message(
            torch.cat([left, right, connection_embedding, normalized_distance2, normal_dot], dim=-1)
        )
        message = message * pair_mask.unsqueeze(-1)
        degree = pair_mask.sum(dim=2, keepdim=True).clamp_min(1).to(hidden.dtype)
        aggregated = message.sum(dim=2) / degree
        hidden = self.norm(hidden + self.node_update(torch.cat([hidden, aggregated], dim=-1)))
        hidden = hidden * node_mask.unsqueeze(-1)

        center_delta = (relative * self.center_weight(message)).sum(dim=2) / degree
        normal_relative = normals[:, :, None, :] - normals[:, None, :, :]
        normal_delta = (
            normal_relative * self.normal_relative_weight(message)
            + normals[:, :, None, :] * self.normal_self_weight(message)
        ).sum(dim=2) / degree
        center_delta = center_delta * node_mask.unsqueeze(-1)
        normal_delta = normal_delta * node_mask.unsqueeze(-1)
        return hidden, centers + center_delta, normals + normal_delta, message


class RingDenoiser(nn.Module):
    """Predict clean categorical states and continuous noise for ring poses."""

    def __init__(
        self,
        hidden_dim: int = 192,
        num_layers: int = 6,
        time_dim: int = 64,
        max_ports: int = 8,
        library: TemplateLibrary = DEFAULT_LIBRARY,
    ) -> None:
        super().__init__()
        self.template_names = library.names
        self.num_templates = len(library)
        self.ring_mask_token = self.num_templates
        self.ring_pad_token = self.num_templates + 1
        self.connection_mask_token = len(ConnectionType)
        self.max_ports = max_ports

        self.ring_embedding = nn.Embedding(self.num_templates + 2, hidden_dim)
        self.connection_embedding = nn.Embedding(len(ConnectionType) + 1, hidden_dim)
        self.register_buffer("template_features", template_feature_table(library))
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_dim + time_dim + self.template_features.shape[1], hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(DenseEquivariantLayer(hidden_dim) for _ in range(num_layers))
        self.ring_head = nn.Linear(hidden_dim, self.num_templates)
        pair_dim = hidden_dim * 3 + 2
        self.connection_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, len(ConnectionType))
        )
        self.port_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, max_ports)
        )

    def forward(
        self,
        ring_types: Tensor,
        centers: Tensor,
        normals: Tensor,
        connection_types: Tensor,
        node_mask: Tensor,
        timesteps: Tensor,
    ) -> dict[str, Tensor]:
        initial_centers = centers
        initial_normals = normals
        time = self.time_embedding(timesteps)[:, None, :].expand(-1, ring_types.shape[1], -1)
        features = self.template_features[ring_types]
        # Column one is mean vdW radius / 2.0. It is a scale, not a chemical ID.
        physical_radius = features[..., 1:2]
        hidden = self.input_projection(
            torch.cat([self.ring_embedding(ring_types), features, time], dim=-1)
        )
        hidden = hidden * node_mask.unsqueeze(-1)
        edge_embedding = self.connection_embedding(connection_types)
        messages: Tensor | None = None
        for layer in self.layers:
            hidden, centers, normals, messages = layer(
                hidden, centers, normals, edge_embedding, physical_radius, node_mask
            )
        assert messages is not None
        left = hidden[:, :, None, :].expand(-1, -1, hidden.shape[1], -1)
        right = hidden[:, None, :, :].expand(-1, hidden.shape[1], -1, -1)
        relative = centers[:, :, None, :] - centers[:, None, :, :]
        distance2 = relative.square().sum(dim=-1, keepdim=True)
        pair_scale = physical_radius[:, :, None, :] + physical_radius[:, None, :, :]
        normalized_distance2 = distance2 / pair_scale.square().clamp_min(1e-6)
        normal_dot = (normals[:, :, None, :] * normals[:, None, :, :]).sum(dim=-1, keepdim=True)
        pair_features = torch.cat(
            [left, right, messages, normalized_distance2, normal_dot], dim=-1
        )
        connection_logits = self.connection_head(pair_features)
        connection_logits = 0.5 * (connection_logits + connection_logits.transpose(1, 2))
        return {
            "center_noise": centers - initial_centers,
            "normal_noise": normals - initial_normals,
            "ring_logits": self.ring_head(hidden),
            "connection_logits": connection_logits,
            "port_logits": self.port_head(pair_features),
            "hidden": hidden,
        }
