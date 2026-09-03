"""Hierarchical graph records: atoms, rings, membership, and ring connections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import sqrt

from .templates import DEFAULT_LIBRARY, TemplateLibrary


class ConnectionType(IntEnum):
    NONE = 0
    SINGLE = 1
    FUSED = 2
    SPIRO = 3


@dataclass(frozen=True)
class RingNode:
    template: str
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def normalized_normal(self) -> tuple[float, float, float]:
        norm = sqrt(sum(value * value for value in self.normal))
        if norm < 1e-8:
            return (0.0, 0.0, 1.0)
        return tuple(value / norm for value in self.normal)  # type: ignore[return-value]


@dataclass(frozen=True)
class RingConnection:
    left: int
    right: int
    kind: ConnectionType
    left_port: int
    right_port: int

    def canonical_pair(self) -> tuple[int, int]:
        return min(self.left, self.right), max(self.left, self.right)


@dataclass
class RingGraph:
    nodes: list[RingNode]
    connections: list[RingConnection] = field(default_factory=list)

    def validate(self, library: TemplateLibrary = DEFAULT_LIBRARY) -> None:
        if not self.nodes:
            raise ValueError("A ring graph must contain at least one ring")
        for node in self.nodes:
            library.get(node.template)

        seen_pairs: set[tuple[int, int]] = set()
        used_edge_ports: set[tuple[int, int]] = set()
        used_vertex_ports: set[tuple[int, int]] = set()
        for connection in self.connections:
            if connection.kind is ConnectionType.NONE:
                raise ValueError("NONE is not a materialized connection")
            if connection.left == connection.right:
                raise ValueError("Self-connections are not allowed")
            if min(connection.left, connection.right) < 0 or max(connection.left, connection.right) >= len(self.nodes):
                raise ValueError("Connection references a missing ring node")
            pair = connection.canonical_pair()
            if pair in seen_pairs:
                raise ValueError(f"Multiple ring connections for pair {pair} are not supported")
            seen_pairs.add(pair)

            left = library.get(self.nodes[connection.left].template)
            right = library.get(self.nodes[connection.right].template)
            if connection.kind is ConnectionType.FUSED:
                left.check_edge_port(connection.left_port)
                right.check_edge_port(connection.right_port)
                keys = ((connection.left, connection.left_port), (connection.right, connection.right_port))
                if any(key in used_edge_ports for key in keys):
                    raise ValueError("A fused edge port cannot be used twice")
                used_edge_ports.update(keys)
            else:
                left.check_vertex_port(connection.left_port)
                right.check_vertex_port(connection.right_port)
                keys = ((connection.left, connection.left_port), (connection.right, connection.right_port))
                if any(key in used_vertex_ports for key in keys):
                    raise ValueError("A vertex attachment port cannot be used twice")
                used_vertex_ports.update(keys)

    def is_connected(self) -> bool:
        if len(self.nodes) == 1:
            return True
        adjacency = [[] for _ in self.nodes]
        for edge in self.connections:
            adjacency[edge.left].append(edge.right)
            adjacency[edge.right].append(edge.left)
        visited = {0}
        stack = [0]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return len(visited) == len(self.nodes)

    @property
    def cyclomatic_number(self) -> int:
        """Cycle rank of the ring-adjacency graph, useful for peri diagnostics."""

        return len(self.connections) - len(self.nodes) + (1 if self.is_connected() else 0)

