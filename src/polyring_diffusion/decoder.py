"""Deterministic conversion from a ring graph to an atom/bond graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from .graph import ConnectionType, RingGraph
from .templates import DEFAULT_LIBRARY, TemplateLibrary


class DecodeError(ValueError):
    pass


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


@dataclass(frozen=True)
class AtomRecord:
    symbol: str
    formal_charge: int = 0
    ring_membership: tuple[int, ...] = ()


@dataclass(frozen=True)
class BondRecord:
    left: int
    right: int
    kind: str


@dataclass
class AtomGraph:
    atoms: list[AtomRecord]
    bonds: list[BondRecord]
    ring_to_atoms: dict[int, tuple[int, ...]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "atoms": [
                {
                    "symbol": atom.symbol,
                    "formal_charge": atom.formal_charge,
                    "rings": list(atom.ring_membership),
                }
                for atom in self.atoms
            ],
            "bonds": [
                {"left": bond.left, "right": bond.right, "kind": bond.kind}
                for bond in self.bonds
            ],
            "ring_to_atoms": {str(key): list(value) for key, value in self.ring_to_atoms.items()},
        }


def decode_ring_graph(
    graph: RingGraph,
    library: TemplateLibrary = DEFAULT_LIBRARY,
) -> AtomGraph:
    """Decode templates and merge atoms selected by fused/spiro ports.

    Ring templates preserve the cycle skeleton. Aromatic/Kekule assignment is
    intentionally deferred to an optional chemistry backend after all fusions.
    """

    graph.validate(library)
    offsets: list[int] = []
    symbols: list[str] = []
    charges: list[int] = []
    initial_membership: list[set[int]] = []
    for ring_index, node in enumerate(graph.nodes):
        template = library.get(node.template)
        offsets.append(len(symbols))
        symbols.extend(template.atoms)
        charges.extend(template.charges)
        initial_membership.extend({ring_index} for _ in template.atoms)

    dsu = _DisjointSet(len(symbols))
    extra_bonds: list[tuple[int, int, str]] = []
    for connection in graph.connections:
        left_template = library.get(graph.nodes[connection.left].template)
        right_template = library.get(graph.nodes[connection.right].template)
        left_offset, right_offset = offsets[connection.left], offsets[connection.right]
        if connection.kind is ConnectionType.FUSED:
            la, lb = left_template.edge_atoms(connection.left_port)
            ra, rb = right_template.edge_atoms(connection.right_port)
            merges = ((left_offset + la, right_offset + rb), (left_offset + lb, right_offset + ra))
            for first, second in merges:
                if symbols[first] != symbols[second] or charges[first] != charges[second]:
                    raise DecodeError("Fused ports require matching atom types and charges")
                dsu.union(first, second)
        elif connection.kind is ConnectionType.SPIRO:
            left_template.check_vertex_port(connection.left_port)
            right_template.check_vertex_port(connection.right_port)
            first = left_offset + connection.left_port
            second = right_offset + connection.right_port
            if symbols[first] != symbols[second] or charges[first] != charges[second]:
                raise DecodeError("Spiro ports require matching atom types and charges")
            dsu.union(first, second)
        elif connection.kind is ConnectionType.SINGLE:
            left_template.check_vertex_port(connection.left_port)
            right_template.check_vertex_port(connection.right_port)
            extra_bonds.append(
                (
                    left_offset + connection.left_port,
                    right_offset + connection.right_port,
                    "SINGLE",
                )
            )
        else:
            raise DecodeError(f"Unsupported connection: {connection.kind}")

    roots: dict[int, int] = {}
    merged_symbols: list[str] = []
    merged_charges: list[int] = []
    merged_membership: list[set[int]] = []
    for atom_index, symbol in enumerate(symbols):
        root = dsu.find(atom_index)
        if root not in roots:
            roots[root] = len(merged_symbols)
            merged_symbols.append(symbol)
            merged_charges.append(charges[atom_index])
            merged_membership.append(set())
        target = roots[root]
        if merged_symbols[target] != symbol or merged_charges[target] != charges[atom_index]:
            raise DecodeError("An atom merge produced incompatible chemistry")
        merged_membership[target].update(initial_membership[atom_index])

    bond_kinds: dict[tuple[int, int], str] = {}

    def add_bond(raw_left: int, raw_right: int, kind: str) -> None:
        left = roots[dsu.find(raw_left)]
        right = roots[dsu.find(raw_right)]
        if left == right:
            raise DecodeError("Atom merging collapsed a bond into a self-loop")
        pair = (min(left, right), max(left, right))
        previous = bond_kinds.get(pair)
        if previous is not None and previous != kind:
            raise DecodeError(f"Conflicting bond assignments for {pair}: {previous} vs {kind}")
        bond_kinds[pair] = kind

    for ring_index, node in enumerate(graph.nodes):
        template = library.get(node.template)
        for atom_index in range(template.size):
            add_bond(
                offsets[ring_index] + atom_index,
                offsets[ring_index] + (atom_index + 1) % template.size,
                "AROMATIC" if template.aromatic else "SINGLE",
            )
    for left, right, kind in extra_bonds:
        add_bond(left, right, kind)

    atoms = [
        AtomRecord(symbol, charge, tuple(sorted(membership)))
        for symbol, charge, membership in zip(merged_symbols, merged_charges, merged_membership)
    ]
    bonds = [BondRecord(left, right, kind) for (left, right), kind in sorted(bond_kinds.items())]
    ring_to_atoms = {
        ring_index: tuple(index for index, atom in enumerate(atoms) if ring_index in atom.ring_membership)
        for ring_index in range(len(graph.nodes))
    }
    return AtomGraph(atoms=atoms, bonds=bonds, ring_to_atoms=ring_to_atoms)

