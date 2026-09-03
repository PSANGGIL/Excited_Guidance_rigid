"""Synthetic ring-graph data used to verify the end-to-end prototype."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .decoder import DecodeError, decode_ring_graph
from .graph import ConnectionType, RingConnection, RingGraph, RingNode
from .templates import DEFAULT_LIBRARY, TemplateLibrary


@dataclass(frozen=True)
class SyntheticDataConfig:
    num_samples: int = 20_000
    min_rings: int = 2
    max_rings: int = 8
    peri_probability: float = 0.2
    templates: tuple[str, ...] = DEFAULT_LIBRARY.names


def _unused_carbon_vertices(template, ring_index: int, used: set[tuple[int, int]]) -> list[int]:
    return [
        port
        for port, symbol in enumerate(template.atoms)
        if symbol == "C" and (ring_index, port) not in used
    ]


class SyntheticRingDataset(Dataset[dict[str, Tensor]]):
    """Deterministic-by-index synthetic graphs for smoke training.

    This is deliberately not presented as chemical training data. It exercises
    variable cardinality, cata/peri topology, fusion ports, and padding before a
    real RDKit preprocessing pipeline is connected.
    """

    def __init__(
        self,
        config: SyntheticDataConfig,
        library: TemplateLibrary = DEFAULT_LIBRARY,
        seed: int = 7,
    ) -> None:
        self.config = config
        self.library = library
        self.seed = seed
        self.template_names = tuple(config.templates)
        for name in self.template_names:
            library.get(name)
        self.template_to_id = {name: index for index, name in enumerate(library.names)}

    def __len__(self) -> int:
        return self.config.num_samples

    def _connection(
        self,
        rng: random.Random,
        nodes: list[RingNode],
        left_index: int,
        right_index: int,
        used_edges: set[tuple[int, int]],
        used_vertices: set[tuple[int, int]],
        prefer_fused: bool = True,
    ) -> RingConnection:
        left = self.library.get(nodes[left_index].template)
        right = self.library.get(nodes[right_index].template)
        compatible = [
            pair
            for pair in left.compatible_edge_ports(right)
            if (left_index, pair[0]) not in used_edges and (right_index, pair[1]) not in used_edges
        ]
        if prefer_fused and compatible:
            lp, rp = rng.choice(compatible)
            used_edges.update(((left_index, lp), (right_index, rp)))
            return RingConnection(left_index, right_index, ConnectionType.FUSED, lp, rp)

        left_ports = _unused_carbon_vertices(left, left_index, used_vertices)
        right_ports = _unused_carbon_vertices(right, right_index, used_vertices)
        if not left_ports or not right_ports:
            raise DecodeError("No free carbon vertex ports")
        lp, rp = rng.choice(left_ports), rng.choice(right_ports)
        used_vertices.update(((left_index, lp), (right_index, rp)))
        return RingConnection(left_index, right_index, ConnectionType.SINGLE, lp, rp)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = random.Random(self.seed + index)
        count = rng.randint(self.config.min_rings, self.config.max_rings)
        nodes = []
        for ring_index in range(count):
            angle = ring_index * 0.55
            center = (2.35 * ring_index, 0.7 * math.sin(angle), 0.15 * math.cos(angle))
            nodes.append(RingNode(rng.choice(self.template_names), center=center))

        used_edges: set[tuple[int, int]] = set()
        used_vertices: set[tuple[int, int]] = set()
        edges: list[RingConnection] = []
        for right in range(1, count):
            left = right - 1
            try:
                edges.append(
                    self._connection(
                        rng,
                        nodes,
                        left,
                        right,
                        used_edges,
                        used_vertices,
                        prefer_fused=rng.random() < 0.75,
                    )
                )
            except DecodeError:
                edges.append(
                    self._connection(
                        rng, nodes, left, right, used_edges, used_vertices, prefer_fused=False
                    )
                )

        if count >= 4 and rng.random() < self.config.peri_probability:
            for left, right in ((0, count - 1), (0, count - 2)):
                try:
                    candidate = self._connection(
                        rng, nodes, left, right, used_edges, used_vertices, prefer_fused=True
                    )
                    trial = RingGraph(nodes=list(nodes), connections=edges + [candidate])
                    trial.validate(self.library)
                    decode_ring_graph(trial, self.library)
                    edges.append(candidate)
                    break
                except (ValueError, DecodeError):
                    continue

        graph = RingGraph(nodes=nodes, connections=edges)
        graph.validate(self.library)
        max_rings = self.config.max_rings
        ring_types = torch.full((max_rings,), len(self.library) + 1, dtype=torch.long)
        node_mask = torch.zeros(max_rings, dtype=torch.bool)
        centers = torch.zeros(max_rings, 3)
        normals = torch.zeros(max_rings, 3)
        connection_types = torch.zeros(max_rings, max_rings, dtype=torch.long)
        ports = torch.zeros(max_rings, max_rings, dtype=torch.long)
        for node_index, node in enumerate(nodes):
            ring_types[node_index] = self.template_to_id[node.template]
            node_mask[node_index] = True
            centers[node_index] = torch.tensor(node.center)
            normals[node_index] = torch.tensor(node.normalized_normal())
        valid_centers = centers[node_mask]
        centers[node_mask] = valid_centers - valid_centers.mean(dim=0, keepdim=True)
        for edge in edges:
            connection_types[edge.left, edge.right] = int(edge.kind)
            connection_types[edge.right, edge.left] = int(edge.kind)
            ports[edge.left, edge.right] = edge.left_port
            ports[edge.right, edge.left] = edge.right_port
        return {
            "ring_types": ring_types,
            "node_mask": node_mask,
            "centers": centers,
            "normals": normals,
            "connection_types": connection_types,
            "ports": ports,
        }

