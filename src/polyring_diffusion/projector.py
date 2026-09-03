"""Hard projection of noisy ring-connection proposals onto valid assemblies."""

from __future__ import annotations

from dataclasses import dataclass

from .decoder import DecodeError, decode_ring_graph
from .graph import ConnectionType, RingConnection, RingGraph, RingNode
from .templates import DEFAULT_LIBRARY, TemplateLibrary


@dataclass(frozen=True)
class ConnectionCandidate:
    score: float
    connection: RingConnection


class RingConstraintProjector:
    def __init__(
        self,
        library: TemplateLibrary = DEFAULT_LIBRARY,
        max_extra_fusions: int = 2,
    ) -> None:
        self.library = library
        self.max_extra_fusions = max_extra_fusions

    def _valid_trial(self, nodes: list[RingNode], edges: list[RingConnection]) -> bool:
        graph = RingGraph(nodes=list(nodes), connections=list(edges))
        try:
            graph.validate(self.library)
            decode_ring_graph(graph, self.library)
        except (ValueError, DecodeError):
            return False
        return True

    def project(
        self,
        nodes: list[RingNode],
        candidates: list[ConnectionCandidate],
        require_connected: bool = True,
    ) -> RingGraph:
        """Greedily keep high-score proposals without violating ring ports.

        A first pass builds a spanning forest. A second pass adds a bounded
        number of closures, allowing peri-like ring-adjacency cycles.
        """

        if not nodes:
            raise ValueError("Cannot project an empty node set")
        ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
        parent = list(range(len(nodes)))

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[b] = a

        selected: list[RingConnection] = []
        for candidate in ordered:
            edge = candidate.connection
            if edge.kind is ConnectionType.NONE or find(edge.left) == find(edge.right):
                continue
            trial = selected + [edge]
            if self._valid_trial(nodes, trial):
                selected.append(edge)
                union(edge.left, edge.right)

        graph = RingGraph(nodes=list(nodes), connections=selected)
        if require_connected and not graph.is_connected():
            raise DecodeError("Candidate set cannot produce a connected valid ring graph")

        extra = 0
        existing_pairs = {edge.canonical_pair() for edge in selected}
        for candidate in ordered:
            if extra >= self.max_extra_fusions:
                break
            edge = candidate.connection
            if edge.kind is ConnectionType.NONE or edge.canonical_pair() in existing_pairs:
                continue
            trial = selected + [edge]
            if self._valid_trial(nodes, trial):
                selected.append(edge)
                existing_pairs.add(edge.canonical_pair())
                extra += 1

        return RingGraph(nodes=list(nodes), connections=selected)

