"""V8 chemistry solver, train-only references, geometry and condition evaluation."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
import rdkit
import scipy
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp

from .chemistry_v5 import (
    AROMATIC_HETERO_H,
    AtomStateCodec,
    HeavyAtomState,
    collate_hp1kg_v5,
    decode_heavy_graph,
    hp1kg_records,
)
from .chemistry_v6 import StructureAssessment, TopologySample, _topology_counts, assess_structure
from .hp1kg_data import HP1KGDataset
from .model_v8 import (
    EDGE_AROMATIC_DOUBLE,
    EDGE_AROMATIC_SINGLE,
    EDGE_AROMATIC_TRIPLE,
    JOINT_EDGE_CLASSES,
    RING_SIZE_CLASSES,
    ring_size_to_class,
    decode_joint_edge_state,
)


def build_train_codec_v8(root: str | Path) -> AtomStateCodec:
    """Build state keys from train only and use chemistry-derived H bounds."""
    rows, atom_map = hp1kg_records(root, ("train",))
    observed = AtomStateCodec.from_records(rows, atom_map)
    states = []
    for state in observed.states:
        lower = 1 if state.aromatic_role == AROMATIC_HETERO_H else 0
        upper = max(lower, min(4, state.valence))
        states.append(HeavyAtomState(
            state.symbol, state.formal_charge, state.valence,
            state.aromatic_role, lower, upper,
        ))
    return AtomStateCodec(states)


def _ring_edge_targets_from_graph(bonds, ring_sizes=()):
    """Return index-safe cyclic-edge, minimum-cycle-size and ring-count targets.

    An edge is cyclic iff its endpoints remain connected after deleting that
    edge. The shortest replacement path plus the deleted edge is the minimum
    cycle containing it. This definition is independent of an arbitrary cycle
    basis and uses exactly the atom indices of the stored tensors.

    The histogram counts rings rather than cyclic edges. Keeping those units
    separate prevents a benzene ring from being counted six times in training
    but once during evaluation.
    """
    bonds = torch.as_tensor(bonds).long()
    if bonds.ndim != 2 or bonds.shape[0] != bonds.shape[1]:
        raise ValueError("bond matrix must be square")
    if not torch.equal(bonds, bonds.T):
        raise ValueError("bond matrix must be symmetric")
    count = bonds.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    graph.add_edges_from(
        (left, right) for left in range(count) for right in range(left + 1, count)
        if int(bonds[left, right]) > 0
    )
    size_target = torch.zeros(count, count, dtype=torch.long)
    for left, right in tuple(graph.edges()):
        graph.remove_edge(left, right)
        try:
            cycle_size = nx.shortest_path_length(graph, left, right) + 1
        except nx.NetworkXNoPath:
            cycle_size = 0
        graph.add_edge(left, right)
        if cycle_size:
            size_class = int(ring_size_to_class(cycle_size))
            size_target[left, right] = size_target[right, left] = size_class
    closure = size_target > 0
    histogram = torch.zeros(RING_SIZE_CLASSES, dtype=torch.float32)
    for ring_size in ring_sizes:
        histogram[int(ring_size_to_class(int(ring_size)))] += 1
    return closure, size_target, histogram


def _minimum_cycle_sizes(bonds):
    """Return a relabeling-invariant minimum-cycle-basis size multiset."""
    bonds = torch.as_tensor(bonds).long()
    graph = nx.Graph()
    graph.add_nodes_from(range(bonds.shape[0]))
    graph.add_edges_from(
        (left, right)
        for left in range(bonds.shape[0])
        for right in range(left + 1, bonds.shape[0])
        if int(bonds[left, right]) > 0
    )
    return tuple(sorted(len(cycle) for cycle in nx.minimum_cycle_basis(graph)))


def _ring_topology_from_graph(bonds, aromatic_edges):
    """Cycle topology from graph invariants rather than RDKit SSSR caches."""
    bonds = torch.as_tensor(bonds).long()
    aromatic = torch.as_tensor(aromatic_edges).bool()
    graph = nx.Graph()
    graph.add_nodes_from(range(bonds.shape[0]))
    graph.add_edges_from(
        (left, right)
        for left in range(bonds.shape[0])
        for right in range(left + 1, bonds.shape[0])
        if int(bonds[left, right]) > 0
    )
    components = nx.number_connected_components(graph) if graph else 0
    cycle_rank = max(0, graph.number_of_edges() - graph.number_of_nodes() + components)
    max_ring_system = max(
        (
            len(component)
            for component in nx.biconnected_components(graph)
            if graph.subgraph(component).number_of_edges() >= len(component)
        ),
        default=0,
    )
    ring_sizes = tuple(sorted(len(cycle) for cycle in nx.minimum_cycle_basis(graph)))
    aromatic_graph = nx.Graph()
    aromatic_graph.add_nodes_from(range(bonds.shape[0]))
    aromatic_graph.add_edges_from(
        (left, right)
        for left in range(bonds.shape[0])
        for right in range(left + 1, bonds.shape[0])
        if bool(aromatic[left, right])
    )
    aromatic_sizes = tuple(sorted(
        len(cycle) for cycle in nx.minimum_cycle_basis(aromatic_graph)
    ))
    # RDKit's bridgehead descriptor is graph-topological.  Initializing ring
    # information on an unsanitized all-carbon realization avoids valence
    # assumptions while reproducing the descriptor on all stored molecules.
    editable = Chem.RWMol()
    for _ in range(bonds.shape[0]):
        editable.AddAtom(Chem.Atom("C"))
    for left, right in graph.edges():
        editable.AddBond(int(left), int(right), Chem.BondType.SINGLE)
    topology_molecule = editable.GetMol()
    Chem.GetSymmSSSR(topology_molecule)
    bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(topology_molecule)
    return RingTopologySampleV8(
        cycle_rank, max_ring_system, ring_sizes, aromatic_sizes, bridgeheads,
    )


def _cyclic_bond_signatures(states, bonds, aromatic_edges):
    """Invariant ring environments based on cyclic bonds and minimum size."""
    states = list(map(int, states))
    bonds = torch.as_tensor(bonds).long()
    aromatic = torch.as_tensor(aromatic_edges).bool()
    cyclic, size_class, _ = _ring_edge_targets_from_graph(bonds)
    cyclic_degree = cyclic.sum(-1)
    signatures = []
    for left in range(len(states)):
        for right in range(left + 1, len(states)):
            if not bool(cyclic[left, right]):
                continue
            state_pair = sorted((states[left], states[right]))
            signatures.append((
                state_pair[0], state_pair[1],
                0 if bool(aromatic[left, right]) else int(bonds[left, right]),
                int(bool(aromatic[left, right])),
                int(size_class[left, right]),
                min(int(cyclic_degree[left]), int(cyclic_degree[right])),
                max(int(cyclic_degree[left]), int(cyclic_degree[right])),
            ))
    return tuple(sorted(signatures))


def _ring_edge_targets(mol):
    """RDKit convenience wrapper retaining the molecule's atom order."""
    count = mol.GetNumAtoms()
    bonds = torch.zeros(count, count, dtype=torch.long)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[left, right] = bonds[right, left] = max(
            1, int(round(bond.GetBondTypeAsDouble())),
        )
    return _ring_edge_targets_from_graph(
        bonds, _minimum_cycle_sizes(bonds),
    )


def _ring_topology(mol):
    count = mol.GetNumAtoms()
    bonds = torch.zeros(count, count, dtype=torch.long)
    aromatic = torch.zeros(count, count, dtype=torch.bool)
    for bond in mol.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[left, right] = bonds[right, left] = max(
            1, int(round(bond.GetBondTypeAsDouble())),
        )
        if bond.GetIsAromatic():
            aromatic[left, right] = aromatic[right, left] = True
    return _ring_topology_from_graph(bonds, aromatic)


@dataclass(frozen=True)
class RingTopologySampleV8:
    closure_count: int
    max_ring_system: int
    ring_sizes: tuple[int, ...]
    aromatic_ring_sizes: tuple[int, ...]
    bridgeheads: int


class HP1KGV8Dataset(HP1KGDataset):
    """HP1KG split with ring-closure and topology targets cached once."""

    def __init__(self, root, split, condition_stats=None, atom_map=None):
        super().__init__(root, split, condition_stats, atom_map)
        enriched = []
        for row in self.rows:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                raise ValueError(f"invalid cached SMILES: {row['smiles']}")
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            if mol.GetNumAtoms() != bonds.shape[0]:
                raise ValueError(f"cached graph/SMILES atom-count mismatch: {row['id']}")
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            topology = _ring_topology_from_graph(bonds, aromatic)
            ring_sizes = _minimum_cycle_sizes(bonds)
            closure, size_target, histogram = _ring_edge_targets_from_graph(
                bonds, ring_sizes,
            )
            if (closure & ~(bonds > 0)).any():
                raise ValueError(f"cyclic-edge target on a nonbond: {row['id']}")
            if (aromatic & ~closure).any():
                raise ValueError(f"aromatic edge is not cyclic: {row['id']}")
            bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            cumulenes = int(((bonds == 2).sum(-1) >= 2).sum())
            enriched.append({
                **row,
                "topology_target": torch.tensor(
                    [
                        topology.closure_count,
                        topology.max_ring_system,
                        sum(size in (3, 4) for size in ring_sizes),
                        bridgeheads,
                        cumulenes,
                    ],
                    dtype=torch.float32,
                ),
                "ring_closure_target": closure,
                "ring_size_target": size_target,
                "ring_closure_histogram": histogram,
            })
        self.rows = enriched


def collate_hp1kg_v8(rows, max_atoms, codec, atom_map):
    output = collate_hp1kg_v5(rows, max_atoms, codec, atom_map)
    output["topology_target"] = torch.stack([row["topology_target"] for row in rows])
    closure = torch.zeros(len(rows), max_atoms, max_atoms, dtype=torch.bool)
    size = torch.zeros(len(rows), max_atoms, max_atoms, dtype=torch.long)
    for batch_index, row in enumerate(rows):
        count = row["ring_closure_target"].shape[0]
        closure[batch_index, :count, :count] = row["ring_closure_target"]
        size[batch_index, :count, :count] = row["ring_size_target"]
    output["ring_closure_target"] = closure
    output["ring_size_target"] = size
    output["ring_closure_histogram"] = torch.stack([
        row["ring_closure_histogram"] for row in rows
    ])
    return output

def _bond_signature(left: int, right: int, order: int, aromatic: bool) -> tuple[int, int, int, int]:
    left, right = sorted((int(left), int(right)))
    return left, right, 0 if aromatic else int(order), int(bool(aromatic))


def _local_signature(index: int, states: Sequence[int], bonds: torch.Tensor, aromatic: torch.Tensor) -> tuple:
    neighbours = []
    for other in range(len(states)):
        order = int(bonds[index, other])
        if order:
            is_aromatic = bool(aromatic[index, other])
            neighbours.append((int(states[other]), 0 if is_aromatic else order, int(is_aromatic)))
    return int(states[index]), tuple(sorted(neighbours))

def ring_support_warnings_v8(state_ids, bonds, aromatic_edges, molecule, reference):
    states = list(map(int, state_ids))
    bonds = torch.as_tensor(bonds).long()
    aromatic = torch.as_tensor(aromatic_edges).bool()
    warnings = set()
    for signature in _cyclic_bond_signatures(states, bonds, aromatic):
        if signature not in reference.ring_signatures:
            warnings.add("unseen_train_ring_environment")
    return tuple(sorted(warnings))
@dataclass
class ChemistryReferenceV8:
    """Train-only ring topology, ring environment and local support."""

    topology_by_count: dict[int, tuple[RingTopologySampleV8, ...]]
    condition_by_count: dict[int, tuple[tuple[float, float], ...]]
    bond_signatures: frozenset[tuple[int, int, int, int]]
    local_signatures: frozenset[tuple]
    ring_signatures: frozenset[tuple]

    @classmethod
    def from_records(cls, records, atom_map, codec):
        records = list(records)
        topology = collections.defaultdict(list)
        conditions = collections.defaultdict(list)
        s1 = torch.tensor([float(row["s1"]) for row in records])
        t1 = torch.tensor([float(row["t1"]) for row in records])
        condition_mean = (float(s1.mean()), float(t1.mean()))
        condition_std = (float(s1.std()), float(t1.std()))
        bond_support = set()
        local_support = set()
        ring_support = set()
        for row in records:
            states = codec.encode_record(row, atom_map).tolist()
            decoded = decode_heavy_graph(states, row["kekule_bonds"], codec)
            mol = decoded.molecule
            if not decoded.valid or mol is None or len(Chem.GetMolFrags(mol)) != 1:
                raise ValueError(f"stored graph cannot be decoded in its native atom order: {row['id']}")
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            topology[mol.GetNumHeavyAtoms()].append(
                _ring_topology_from_graph(bonds, aromatic),
            )
            conditions[mol.GetNumHeavyAtoms()].append((
                (float(row["s1"]) - condition_mean[0]) / condition_std[0],
                (float(row["t1"]) - condition_mean[1]) / condition_std[1],
            ))
            ring_support.update(
                _cyclic_bond_signatures(states, bonds, aromatic),
            )
            for i in range(len(states)):
                local_support.add(_local_signature(i, states, bonds, aromatic))
                for j in range(i + 1, len(states)):
                    if int(bonds[i, j]):
                        bond_support.add(_bond_signature(
                            states[i], states[j], int(bonds[i, j]), bool(aromatic[i, j]),
                        ))
        return cls(
            {int(key): tuple(values) for key, values in topology.items()},
            {int(key): tuple(values) for key, values in conditions.items()},
            frozenset(bond_support), frozenset(local_support), frozenset(ring_support),
        )

    @classmethod
    def from_metadata(cls, payload):
        return cls(
            {
                int(key): tuple(RingTopologySampleV8(
                    int(value["closure_count"]), int(value["max_ring_system"]),
                    tuple(map(int, value["ring_sizes"])),
                    tuple(map(int, value["aromatic_ring_sizes"])),
                    int(value["bridgeheads"]),
                ) for value in values)
                for key, values in payload["topology_by_count"].items()
            },
            {
                int(key): tuple(
                    tuple(map(float, value)) for value in values
                )
                for key, values in payload["condition_by_count"].items()
            },
            frozenset(tuple(map(int, value)) for value in payload["bond_signatures"]),
            frozenset(
                (int(value[0]), tuple(tuple(map(int, neighbour)) for neighbour in value[1]))
                for value in payload["local_signatures"]
            ),
            frozenset(
                tuple(map(int, value)) for value in payload["ring_signatures"]
            ),
        )

    def to_metadata(self):
        return {
            "condition_by_count": {
                str(key): [list(value) for value in values]
                for key, values in sorted(self.condition_by_count.items())
            },
            "topology_by_count": {
                str(key): [asdict(value) for value in values]
                for key, values in sorted(self.topology_by_count.items())
            },
            "bond_signatures": [list(value) for value in sorted(self.bond_signatures)],
            "local_signatures": [
                [value[0], [list(neighbour) for neighbour in value[1]]]
                for value in sorted(self.local_signatures)
            ],
            "ring_signatures": [
                list(value) for value in sorted(self.ring_signatures)
            ],
        }

    def nearest(self, count: int, minimum: int = 24):
        pooled = []
        for key in sorted(self.topology_by_count, key=lambda item: (abs(item - count), item)):
            pooled.extend(self.topology_by_count[key])
            if len(pooled) >= minimum:
                break
        return tuple(pooled)

    def nearest_conditioned(self, count: int, minimum: int = 24):
        pooled = []
        for key in sorted(
            self.topology_by_count,
            key=lambda item: (abs(item - count), item),
        ):
            topologies = self.topology_by_count[key]
            conditions = self.condition_by_count[key]
            if len(topologies) != len(conditions):
                raise ValueError("topology/condition reference length mismatch")
            pooled.extend(zip(topologies, conditions))
            if len(pooled) >= minimum:
                break
        return tuple(pooled)

    @staticmethod
    def topology_distance(count, left, right):
        left_sizes, right_sizes = collections.Counter(left.ring_sizes), collections.Counter(right.ring_sizes)
        size_keys = set(left_sizes) | set(right_sizes)
        size_distance = sum(abs(left_sizes[key] - right_sizes[key]) for key in size_keys)
        left_aromatic = collections.Counter(left.aromatic_ring_sizes)
        right_aromatic = collections.Counter(right.aromatic_ring_sizes)
        aromatic_keys = set(left_aromatic) | set(right_aromatic)
        aromatic_distance = sum(abs(left_aromatic[key] - right_aromatic[key]) for key in aromatic_keys)
        return float(
            abs(left.closure_count - right.closure_count) / max(1., math.sqrt(max(1, count)))
            + abs(left.max_ring_system - right.max_ring_system) / max(2., .25 * max(1, count))
            + 1.5 * size_distance / max(1, len(left.ring_sizes), len(right.ring_sizes))
            + .5 * aromatic_distance / max(1, len(left.aromatic_ring_sizes), len(right.aromatic_ring_sizes))
            + abs(left.bridgeheads - right.bridgeheads) / max(1., math.sqrt(max(1, count)))
        )

    def topology_score(self, count, observed, condition=None, bandwidth=1.25):
        """Kernel P(topology | atom count, normalized S1, normalized T1)."""
        rows = self.nearest_conditioned(count)
        if not rows:
            return 1.
        target = None
        if condition is not None:
            target = tuple(map(float, torch.as_tensor(condition).flatten()))
            if len(target) != 2 or not all(math.isfinite(value) for value in target):
                raise ValueError("topology condition must be two finite values")
        weighted = collections.defaultdict(float)
        for candidate, candidate_condition in rows:
            if target is None:
                weight = 1.
            else:
                distance = sum(
                    (left - right) ** 2
                    for left, right in zip(target, candidate_condition)
                )
                weight = math.exp(-.5 * distance / (bandwidth * bandwidth))
            weighted[candidate] += weight
        if max(weighted.values(), default=0.) < 1e-8:
            weighted = collections.Counter(candidate for candidate, _ in rows)
        peak = max(weighted.values())
        return float(max(
            math.exp(-self.topology_distance(count, observed, candidate))
            * (frequency + .25) / (peak + .25)
            for candidate, frequency in weighted.items()
        ))

    def ring_size_probability(self, count, size, alpha=.25):
        rows = self.nearest(count)
        counts = collections.Counter(value for row in rows for value in row.ring_sizes)
        support = range(3, max(14, count + 1))
        denominator = sum(counts.values()) + alpha * len(support)
        return float((counts.get(int(size), 0) + alpha) / max(denominator, alpha))

    def sample_topology(self, count, rng):
        rows = self.nearest(count)
        return rows[rng.randrange(len(rows))] if rows else None

    def expected_ring_histogram(self, counts):
        histogram = collections.Counter()
        for count in counts:
            rows = self.nearest(int(count))
            if not rows:
                continue
            scale = 1 / len(rows)
            for row in rows:
                for size in row.ring_sizes:
                    histogram[int(size)] += scale
        return histogram

def build_train_reference_v8(root, codec=None):
    codec = codec or build_train_codec_v8(root)
    rows, atom_map = hp1kg_records(root, ("train",))
    return ChemistryReferenceV8.from_records(rows, atom_map, codec)


def support_warnings_v8(state_ids, bonds, aromatic_edges, reference):
    states = list(map(int, state_ids))
    bonds = torch.as_tensor(bonds).long()
    aromatic = torch.as_tensor(aromatic_edges).bool()
    warnings = set()
    for i in range(len(states)):
        if _local_signature(i, states, bonds, aromatic) not in reference.local_signatures:
            warnings.add("unseen_train_local_atom_environment")
        for j in range(i + 1, len(states)):
            if int(bonds[i, j]):
                signature = _bond_signature(states[i], states[j], int(bonds[i, j]), bool(aromatic[i, j]))
                if signature not in reference.bond_signatures:
                    warnings.add("unseen_train_bond_environment")
    return tuple(sorted(warnings))


@dataclass
class JointProjectionResultV8:
    state_ids: torch.Tensor
    edge_state: torch.Tensor
    bonds: torch.Tensor
    aromatic_edges: torch.Tensor
    hydrogens: torch.Tensor
    valid: bool
    reason: str
    solver_used: bool
    solver_status: str
    topology_score: float = 1.
    topology_reprojections: int = 0


def _decoded_assignment(state_ids, edge_state, codec):
    bonds, proposed_aromatic = decode_joint_edge_state(edge_state)
    decoded = decode_heavy_graph(state_ids, bonds, codec)
    if not decoded.valid:
        return None, bonds, proposed_aromatic, decoded.error or "strict decode failed"
    perceived = torch.zeros_like(bonds, dtype=torch.bool)
    for bond in decoded.molecule.GetBonds():
        if bond.GetIsAromatic():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            perceived[i, j] = perceived[j, i] = True
    if not torch.equal(proposed_aromatic.bool(), perceived):
        mismatch = int((proposed_aromatic.bool() ^ perceived).triu(1).sum())
        return None, bonds, perceived, f"aromatic edge/perception mismatch on {mismatch} bonds"
    for index, state_id in enumerate(state_ids):
        state = codec.states[int(state_id)]
        atom = decoded.molecule.GetAtomWithIdx(index)
        role = AtomStateCodec.aromatic_role(
            state.symbol, atom.GetIsAromatic(), int(atom.GetTotalNumHs(includeNeighbors=True)),
        )
        if role != state.aromatic_role:
            return None, bonds, perceived, f"aromatic role mismatch at atom {index}"
    return decoded, bonds, perceived, ""


def decode_joint_assignment_v8(state_ids, edge_state, codec):
    """Strictly decode a joint assignment without repair."""
    return _decoded_assignment(state_ids, edge_state, codec)


def _symmetric_argmax(logits):
    logits = (logits.float() + logits.float().transpose(0, 1)) * .5
    chosen = logits.argmax(-1)
    upper = torch.triu(chosen, diagonal=1)
    return upper + upper.T


def _candidate_states(state_logp, codec, top_k=6):
    output = []
    charges = sorted({state.formal_charge for state in codec.states})
    roles = sorted({state.aromatic_role for state in codec.states})
    for row in state_logp:
        chosen = set(torch.topk(row, min(top_k, len(row))).indices.tolist())
        for charge in charges:
            choices = [i for i, state in enumerate(codec.states) if state.formal_charge == charge]
            chosen.add(max(choices, key=lambda i: float(row[i])))
        for role in roles:
            choices = [i for i, state in enumerate(codec.states) if state.aromatic_role == role]
            if choices:
                chosen.add(max(choices, key=lambda i: float(row[i])))
        output.append(sorted(chosen))
    return output


def _candidate_edges(edge_logp, top_k=8, probability_threshold=.03, complete=False):
    count = edge_logp.shape[0]
    if complete:
        return [(i, j) for i in range(count) for j in range(i + 1, count)]
    nonedge = edge_logp[..., 0]
    best_nonzero = edge_logp[..., 1:].max(-1).values
    gain = best_nonzero - nonedge
    probability = edge_logp[..., 1:].exp().sum(-1)
    pairs = set()
    for i in range(count):
        ranking = torch.topk(gain[i].masked_fill(torch.arange(count) == i, -torch.inf), min(top_k, count - 1)).indices
        pairs.update(tuple(sorted((i, int(j)))) for j in ranking.tolist() if i != int(j))
    pairs.update(
        (i, j) for i in range(count) for j in range(i + 1, count)
        if float(probability[i, j]) >= probability_threshold
    )
    complete = nx.Graph()
    complete.add_nodes_from(range(count))
    complete.add_weighted_edges_from(
        (i, j, float(gain[i, j]))
        for i in range(count) for j in range(i + 1, count)
    )
    pairs.update(tuple(sorted(edge)) for edge in nx.maximum_spanning_tree(complete).edges())
    return sorted(pairs)


def _solve_joint_milp(
    state_logits, edge_logits, codec, target_charge, time_limit, max_attempts,
    state_top_k=6, edge_top_k=8, probability_threshold=.03, complete_edges=False,
    topology_reference=None, topology_condition=None, topology_prior_weight=1.5,
    topology_reproject_threshold=.2, rng=None, excluded_assignments=(),
    ring_closure_logits=None, ring_size_logits=None,
    geometry_reference=None, positions=None, geometry_prior_weight=.05,
):
    """Joint chemistry MAP solve without a spanning-tree topology gauge.

    Connectivity is imposed by a single-commodity flow over the selected edge
    variables. Ring size and cyclic-edge heads are evaluated on the realized
    graph and rerank multiple feasible solutions; no arbitrary fundamental
    cycle is treated as a chemical ring. RDKit sanitization is the final global
    aromatic/Kekule existence test.
    """
    count, _ = state_logits.shape
    state_logp = torch.log_softmax(state_logits.float(), -1).cpu()
    edge_logp = torch.log_softmax(
        (edge_logits.float() + edge_logits.float().transpose(0, 1)) * .5, -1,
    ).cpu()
    if ring_closure_logits is None:
        closure_logp = torch.zeros(count, count, 2)
    else:
        closure_logp = torch.log_softmax(
            (
                ring_closure_logits.float()
                + ring_closure_logits.float().transpose(0, 1)
            ) * .5,
            -1,
        ).cpu()
    if ring_size_logits is None:
        size_logp = torch.zeros(count, count, RING_SIZE_CLASSES)
    else:
        size_logp = torch.log_softmax(
            (
                ring_size_logits.float()
                + ring_size_logits.float().transpose(0, 1)
            ) * .5,
            -1,
        ).cpu()
    state_candidates = _candidate_states(state_logp, codec, state_top_k)
    edge_pairs = _candidate_edges(
        edge_logp, edge_top_k, probability_threshold, complete_edges,
    )
    candidate_graph = nx.Graph()
    candidate_graph.add_nodes_from(range(count))
    candidate_graph.add_edges_from(edge_pairs)
    if count > 1 and not nx.is_connected(candidate_graph):
        return (
            None, None, None, None, None,
            "candidate edge graph is disconnected", 0., 0,
        )
    rng = rng or random.Random(0)
    position_tensor = (
        torch.as_tensor(positions).float().cpu()
        if positions is not None else None
    )
    if position_tensor is not None and position_tensor.shape != (count, 3):
        raise ValueError("projection positions must have shape [N,3]")

    objective, lower_bounds, upper_bounds, integrality = [], [], [], []
    atom_variables, edge_variables, flow_variables = {}, {}, {}

    def variable(cost, lower, upper, integer):
        index = len(objective)
        objective.append(float(cost))
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
        integrality.append(int(integer))
        return index

    def geometry_cost(edge_class, left, right):
        if (
            geometry_reference is None or position_tensor is None
            or geometry_prior_weight <= 0
        ):
            return 0.
        order_for_class = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
        aromatic = int(edge_class >= EDGE_AROMATIC_SINGLE)
        order = 0 if aromatic else order_for_class[edge_class]
        rows = [
            value for key, value in geometry_reference.bond_length_stats.items()
            if int(key[2]) == order and int(key[3]) == aromatic
        ]
        if not rows:
            return 0.
        total = sum(max(1, int(value[2])) for value in rows)
        mean = sum(value[0] * max(1, int(value[2])) for value in rows) / total
        scale = max(
            .1,
            sum(value[1] * max(1, int(value[2])) for value in rows) / total,
        )
        distance = float((position_tensor[left] - position_tensor[right]).norm())
        return float(geometry_prior_weight) * min(((distance - mean) / scale) ** 2, 25.)

    for atom, candidates in enumerate(state_candidates):
        for state in candidates:
            atom_variables[atom, state] = variable(
                -float(state_logp[atom, state]), 0, 1, 1,
            )
    for edge_index, (left, right) in enumerate(edge_pairs):
        none_score = float(edge_logp[left, right, 0])
        for edge_class in range(1, JOINT_EDGE_CLASSES):
            edge_variables[edge_index, edge_class] = variable(
                -(float(edge_logp[left, right, edge_class]) - none_score)
                + geometry_cost(edge_class, left, right),
                0, 1, 1,
            )
        if count > 1:
            flow_variables[edge_index, left, right] = variable(
                0, 0, count - 1, 0,
            )
            flow_variables[edge_index, right, left] = variable(
                0, 0, count - 1, 0,
            )

    rows, row_lower, row_upper = [], [], []

    def constraint(coefficients, lower=-np.inf, upper=np.inf):
        rows.append(dict(coefficients))
        row_lower.append(float(lower))
        row_upper.append(float(upper))

    for atom, candidates in enumerate(state_candidates):
        constraint(
            {atom_variables[atom, state]: 1 for state in candidates}, 1, 1,
        )
    constraint({
        atom_variables[atom, state]: codec.states[state].formal_charge
        for atom, candidates in enumerate(state_candidates)
        for state in candidates
    }, target_charge, target_charge)

    incident = collections.defaultdict(list)
    aromatic_classes = (
        EDGE_AROMATIC_SINGLE, EDGE_AROMATIC_DOUBLE, EDGE_AROMATIC_TRIPLE,
    )

    def edge_expression(edge_index, classes):
        return {
            edge_variables[edge_index, edge_class]: 1
            for edge_class in classes
        }

    for edge_index, pair in enumerate(edge_pairs):
        left, right = pair
        selected = edge_expression(
            edge_index, range(1, JOINT_EDGE_CLASSES),
        )
        constraint(selected, upper=1)
        incident[left].append(edge_index)
        incident[right].append(edge_index)
        aromatic_vars = edge_expression(edge_index, aromatic_classes)
        for atom in pair:
            endpoint = dict(aromatic_vars)
            for state in state_candidates[atom]:
                if codec.states[state].aromatic_role > 0:
                    variable_index = atom_variables[atom, state]
                    endpoint[variable_index] = endpoint.get(variable_index, 0) - 1
            constraint(endpoint, upper=0)
        if count > 1:
            for source, destination in ((left, right), (right, left)):
                capacity = {
                    flow_variables[edge_index, source, destination]: 1,
                }
                for variable_index, value in selected.items():
                    capacity[variable_index] = -(count - 1) * value
                constraint(capacity, upper=0)

    # Single-commodity flow makes the selected graph connected without fixing
    # which selected edges form a spanning tree.
    if count > 1:
        for atom in range(count):
            balance = {}
            for edge_index, (left, right) in enumerate(edge_pairs):
                if atom == left:
                    outgoing = flow_variables[edge_index, left, right]
                    incoming = flow_variables[edge_index, right, left]
                elif atom == right:
                    outgoing = flow_variables[edge_index, right, left]
                    incoming = flow_variables[edge_index, left, right]
                else:
                    continue
                balance[outgoing] = balance.get(outgoing, 0) + 1
                balance[incoming] = balance.get(incoming, 0) - 1
            target = count - 1 if atom == 0 else -1
            constraint(balance, target, target)

    order_for_class = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
    for atom, candidates in enumerate(state_candidates):
        heavy, aromatic_degree = {}, {}
        for edge_index in incident[atom]:
            for edge_class, order in order_for_class.items():
                heavy[edge_variables[edge_index, edge_class]] = order
            aromatic_degree.update(edge_expression(edge_index, aromatic_classes))
        minimum, maximum = dict(heavy), dict(heavy)
        for state in candidates:
            item = codec.states[state]
            minimum[atom_variables[atom, state]] = -(
                item.valence - item.hydrogen_max
            )
            maximum[atom_variables[atom, state]] = -(
                item.valence - item.hydrogen_min
            )
        constraint(minimum, lower=0)
        constraint(maximum, upper=0)

        lower_aromatic, upper_aromatic = dict(aromatic_degree), dict(aromatic_degree)
        for state in candidates:
            role = codec.states[state].aromatic_role
            if role > 0:
                lower_aromatic[atom_variables[atom, state]] = -2
                upper_aromatic[atom_variables[atom, state]] = -3
        constraint(lower_aromatic, lower=0)
        constraint(upper_aromatic, upper=0)

    binary_indices = list(atom_variables.values()) + list(edge_variables.values())

    def exclude_assignment(states, edges):
        selected = set()
        for atom, state in enumerate(map(int, states)):
            variable_index = atom_variables.get((atom, state))
            if variable_index is not None:
                selected.add(variable_index)
        edges = torch.as_tensor(edges).long()
        for edge_index, (left, right) in enumerate(edge_pairs):
            edge_class = int(edges[left, right])
            variable_index = edge_variables.get((edge_index, edge_class))
            if variable_index is not None:
                selected.add(variable_index)
        if selected:
            # Exclude exactly this binary assignment. Using only positive
            # coefficients would also exclude every graph superset, preventing
            # a low-prior tree from being repaired by adding a ring closure.
            constraint(
                {
                    index: (1 if index in selected else -1)
                    for index in binary_indices
                },
                upper=len(selected) - 1,
            )

    for states, edges in excluded_assignments:
        exclude_assignment(states, edges)

    last_reason = "MILP returned no solution"
    best = best_accepted = None
    reprojections = 0
    rerank_candidates = (
        topology_prior_weight > 0
        and (
            topology_reference is not None
            or ring_closure_logits is not None
            or ring_size_logits is not None
        )
    )
    for attempt in range(max_attempts):
        matrix = sparse.lil_matrix(
            (len(rows), len(objective)), dtype=np.float64,
        )
        for row_index, coefficients in enumerate(rows):
            for column, value in coefficients.items():
                matrix[row_index, column] = value
        result = milp(
            np.asarray(objective),
            integrality=np.asarray(integrality),
            bounds=Bounds(
                np.asarray(lower_bounds), np.asarray(upper_bounds),
            ),
            constraints=LinearConstraint(
                matrix.tocsr(), np.asarray(row_lower), np.asarray(row_upper),
            ),
            options={
                "time_limit": float(time_limit) / max(1, max_attempts),
                "mip_rel_gap": .01,
                "presolve": True,
            },
        )
        if result.x is None:
            last_reason = f"MILP status={result.status}: {result.message}"
            break
        values = result.x
        state_ids = [
            max(
                candidates,
                key=lambda state: values[atom_variables[atom, state]],
            )
            for atom, candidates in enumerate(state_candidates)
        ]
        edge_state = torch.zeros(count, count, dtype=torch.long)
        for edge_index, (left, right) in enumerate(edge_pairs):
            choices = [
                (values[edge_variables[edge_index, state]], state)
                for state in range(1, JOINT_EDGE_CLASSES)
            ]
            value, edge_class = max(choices)
            if value > .5:
                edge_state[left, right] = edge_state[right, left] = edge_class
        decoded, bonds, aromatic, reason = _decoded_assignment(
            state_ids, edge_state, codec,
        )
        selected = {
            index for index in binary_indices if values[index] > .5
        }
        no_good = {
            index: (1 if index in selected else -1)
            for index in binary_indices
        }
        if decoded is None:
            last_reason = reason
            constraint(no_good, upper=len(selected) - 1)
            continue

        topology_score = 1.
        if topology_reference is not None:
            topology_score = topology_reference.topology_score(
                count, _ring_topology(decoded.molecule), topology_condition,
            )
        cyclic, minimum_size, _ = _ring_edge_targets_from_graph(bonds)
        learned_terms = []
        for left in range(count):
            for right in range(left + 1, count):
                if int(bonds[left, right]) == 0:
                    continue
                cyclic_class = int(bool(cyclic[left, right]))
                learned_terms.append(float(
                    closure_logp[left, right, cyclic_class]
                ))
                if cyclic_class:
                    learned_terms.append(float(
                        size_logp[left, right, int(minimum_size[left, right])]
                    ))
        learned_score = (
            sum(learned_terms) / len(learned_terms) if learned_terms else 0.
        )
        edge_count = int((edge_state > 0).triu(1).sum())
        cycle_rank = max(0, edge_count - count + 1)
        status = (
            f"status={result.status};attempt={attempt + 1};"
            f"ring_topology_score={topology_score:.6g};"
            f"learned_topology_logp={learned_score:.6g};"
            f"graph_edges={edge_count};cycle_rank={cycle_rank}"
        )
        combined = (
            -float(result.fun)
            + topology_prior_weight * (
                math.log(max(topology_score, 1e-12)) + learned_score
            )
        )
        candidate = (
            state_ids, edge_state, bonds, aromatic, decoded, status,
            topology_score, reprojections, combined,
        )
        if best is None or combined > best[8]:
            best = candidate
        accepted = (
            topology_reference is None
            or topology_score >= topology_reproject_threshold
        )
        if accepted and (
            best_accepted is None or combined > best_accepted[8]
        ):
            best_accepted = candidate
        if accepted and not rerank_candidates:
            return candidate[:8]
        if not accepted:
            reprojections += 1
        constraint(no_good, upper=len(selected) - 1)
    chosen = best_accepted if best_accepted is not None else best
    if chosen is not None:
        fallback = best_accepted is None and topology_reference is not None
        status = (
            chosen[5]
            + f";candidate_rerank={int(rerank_candidates)}"
            + f";low_ring_prior_fallback={int(fallback)}"
            + f";reprojections={reprojections}"
        )
        return (*chosen[:5], status, chosen[6], reprojections)
    return None, None, None, None, None, last_reason, 0., reprojections


def project_joint_graph_v8(
    state_logits, joint_edge_logits, codec, target_charge=0,
    solver_time_limit=8., solver_attempts=4, topology_reference=None,
    topology_prior_weight=1.5, topology_reproject_threshold=.2,
    topology_seed=0, ring_closure_logits=None, ring_size_logits=None,
    geometry_reference=None, positions=None, geometry_prior_weight=.05,
    topology_condition=None,
):
    """Joint MAP projection with soft topology and geometry regularization."""
    count = int(state_logits.shape[0])
    if joint_edge_logits.shape != (count, count, JOINT_EDGE_CLASSES):
        raise ValueError(
            f"joint edge logits must have shape [N,N,{JOINT_EDGE_CLASSES}]"
        )
    if (
        topology_prior_weight < 0
        or not 0 <= topology_reproject_threshold <= 1
        or geometry_prior_weight < 0
    ):
        raise ValueError("invalid topology/geometry projection configuration")
    rng = random.Random(int(topology_seed))
    state_ids = state_logits.float().argmax(-1).cpu().tolist()
    edge_state = _symmetric_argmax(joint_edge_logits.cpu())
    decoded, bonds, aromatic, reason = _decoded_assignment(
        state_ids, edge_state, codec,
    )
    fast_fallback = None
    excluded = []
    initial_reprojections = 0
    if decoded is not None:
        score = 1.
        if topology_reference is not None:
            score = topology_reference.topology_score(
                count, _ring_topology(decoded.molecule), topology_condition,
            )
        geometry_ok = True
        geometry_z = None
        if geometry_reference is not None and positions is not None:
            geometry_check = assess_generated_coordinates(
                decoded.molecule, state_ids, bonds, aromatic, positions,
                codec, geometry_reference, optimize=False,
            )
            geometry_z = geometry_check.bond_z_rms
            geometry_ok = (
                geometry_check.finite
                and (geometry_z is None or geometry_z <= 4.)
                and geometry_check.steric_clashes == 0
            )
        topology_ok = (
            topology_reference is None
            or score >= topology_reproject_threshold
        )
        if topology_ok and geometry_ok:
            return JointProjectionResultV8(
                torch.tensor(state_ids), edge_state, bonds, aromatic,
                torch.tensor(decoded.hydrogens), True, "", False,
                (
                    f"argmax_feasible;ring_topology_score={score:.6g};"
                    f"bond_z_rms={geometry_z}"
                ),
                score, 0,
            )
        fast_fallback = (
            state_ids, edge_state, bonds, aromatic, decoded, score,
        )
        excluded.append((state_ids, edge_state))
        initial_reprojections = 1

    common = dict(
        topology_reference=topology_reference,
        topology_condition=topology_condition,
        topology_prior_weight=topology_prior_weight,
        topology_reproject_threshold=topology_reproject_threshold,
        rng=rng,
        excluded_assignments=excluded,
        ring_closure_logits=ring_closure_logits,
        ring_size_logits=ring_size_logits,
        geometry_reference=geometry_reference,
        positions=positions,
        geometry_prior_weight=geometry_prior_weight,
    )
    first_budget = max(.1, float(solver_time_limit) * .5)
    solved = _solve_joint_milp(
        state_logits.cpu(), joint_edge_logits.cpu(), codec, target_charge,
        first_budget, max(1, solver_attempts), **common,
    )
    (
        state_ids, edge_state, bonds, aromatic, decoded, status, score,
        reprojections,
    ) = solved
    if decoded is None:
        complete = count <= 24
        solved = _solve_joint_milp(
            state_logits.cpu(), joint_edge_logits.cpu(), codec, target_charge,
            max(.1, float(solver_time_limit) - first_budget),
            max(1, solver_attempts),
            state_top_k=(
                len(codec.states) if complete else min(16, len(codec.states))
            ),
            edge_top_k=(
                max(1, count - 1)
                if complete else min(16, max(1, count - 1))
            ),
            probability_threshold=(0. if complete else .005),
            complete_edges=complete,
            **common,
        )
        (
            state_ids, edge_state, bonds, aromatic, decoded, expanded_status,
            score, expanded_reprojections,
        ) = solved
        reprojections += expanded_reprojections
        status = f"initial={status};expanded={expanded_status}"
    reprojections += initial_reprojections
    if decoded is None and fast_fallback is not None:
        state_ids, edge_state, bonds, aromatic, decoded, score = fast_fallback
        status = f"{status};ring_reprojection_failed_fallback=1"
    if decoded is None:
        size = count
        return JointProjectionResultV8(
            torch.zeros(size, dtype=torch.long),
            torch.zeros(size, size, dtype=torch.long),
            torch.zeros(size, size, dtype=torch.long),
            torch.zeros(size, size, dtype=torch.bool),
            torch.zeros(size, dtype=torch.long),
            False, status or reason, True, status, 0., reprojections,
        )
    return JointProjectionResultV8(
        torch.tensor(state_ids), edge_state, bonds, aromatic,
        torch.tensor(decoded.hydrogens), True, "", True, status,
        score, reprojections,
    )


@dataclass
class GeometryReferenceV8:
    bond_length_stats: dict[tuple[int, int, int, int], tuple[float, float, int]]

    @classmethod
    def from_records(cls, records, atom_map, codec):
        values = collections.defaultdict(list)
        for row in records:
            states = codec.encode_record(row, atom_map).tolist()
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            positions = torch.as_tensor(row["atom_positions"]).float()
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    if int(bonds[i, j]):
                        key = _bond_signature(states[i], states[j], int(bonds[i, j]), bool(aromatic[i, j]))
                        values[key].append(float((positions[i] - positions[j]).norm()))
        stats = {}
        for key, lengths in values.items():
            array = np.asarray(lengths, dtype=np.float64)
            stats[key] = (float(array.mean()), float(max(array.std(ddof=1) if len(array) > 1 else 0., .05)), len(array))
        return cls(stats)

    def to_metadata(self):
        return {"bond_length_stats": [[list(key), list(value)] for key, value in sorted(self.bond_length_stats.items())]}

    @classmethod
    def from_metadata(cls, payload):
        return cls({tuple(map(int, key)): (float(value[0]), float(value[1]), int(value[2])) for key, value in payload["bond_length_stats"]})


@dataclass(frozen=True)
class GeneratedGeometryAssessmentV8:
    finite: bool
    bond_z_rms: float | None
    bond_outliers: int | None
    steric_clashes: int | None
    forcefield: str
    forcefield_initial_energy: float | None
    forcefield_final_energy: float | None
    relaxation_energy_per_heavy_atom: float | None
    forcefield_converged: bool
    warnings: tuple[str, ...]


def assess_generated_coordinates(mol, state_ids, bonds, aromatic, positions, codec, reference, optimize=True):
    positions = torch.as_tensor(positions).float()
    warnings = set()
    if positions.shape != (mol.GetNumAtoms(), 3) or not torch.isfinite(positions).all():
        return GeneratedGeometryAssessmentV8(False, None, None, None, "none", None, None, None, False, ("nonfinite_or_bad_coordinate_shape",))
    zscores = []
    bonds = torch.as_tensor(bonds).long()
    aromatic = torch.as_tensor(aromatic).bool()
    for i in range(len(state_ids)):
        for j in range(i + 1, len(state_ids)):
            if int(bonds[i, j]):
                key = _bond_signature(state_ids[i], state_ids[j], int(bonds[i, j]), bool(aromatic[i, j]))
                if key in reference.bond_length_stats:
                    mean, std, _ = reference.bond_length_stats[key]
                    zscores.append(abs(float((positions[i] - positions[j]).norm()) - mean) / std)
    bond_z_rms = math.sqrt(sum(value * value for value in zscores) / len(zscores)) if zscores else None
    bond_outliers = sum(value > 4 for value in zscores) if zscores else 0
    if bond_outliers:
        warnings.add("generated_bond_length_outlier")

    topology_distance = Chem.GetDistanceMatrix(mol)
    table = Chem.GetPeriodicTable()
    clashes = 0
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            if topology_distance[i, j] < 3:
                continue
            cutoff = .65 * (
                table.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                + table.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
            )
            clashes += int(float((positions[i] - positions[j]).norm()) < cutoff)
    if clashes:
        warnings.add("generated_steric_clash")

    field = "none"
    initial = final = relaxation = None
    converged = False
    if optimize:
        with_coordinates = Chem.Mol(mol)
        with_coordinates.RemoveAllConformers()
        conformer = Chem.Conformer(with_coordinates.GetNumAtoms())
        conformer.Set3D(True)
        for index, xyz in enumerate(positions.tolist()):
            conformer.SetAtomPosition(index, tuple(map(float, xyz)))
        with_coordinates.AddConformer(conformer, assignId=True)
        embedded = Chem.AddHs(with_coordinates, addCoords=True)
        try:
            if AllChem.MMFFHasAllMoleculeParams(embedded):
                field = "MMFF94"
                properties = AllChem.MMFFGetMoleculeProperties(embedded)
                forcefield = AllChem.MMFFGetMoleculeForceField(embedded, properties)
                initial = float(forcefield.CalcEnergy())
                status = AllChem.MMFFOptimizeMolecule(embedded, maxIters=500)
                forcefield = AllChem.MMFFGetMoleculeForceField(embedded, properties)
            else:
                field = "UFF"
                forcefield = AllChem.UFFGetMoleculeForceField(embedded)
                initial = float(forcefield.CalcEnergy())
                status = AllChem.UFFOptimizeMolecule(embedded, maxIters=500)
                forcefield = AllChem.UFFGetMoleculeForceField(embedded)
            final = float(forcefield.CalcEnergy())
            converged = status == 0
            relaxation = max(0., initial - final) / max(1, mol.GetNumHeavyAtoms())
            if not converged:
                warnings.add("generated_forcefield_not_converged")
        except Exception as exc:
            warnings.add("generated_forcefield_exception:" + type(exc).__name__)
    return GeneratedGeometryAssessmentV8(
        True, bond_z_rms, bond_outliers, clashes, field, initial, final,
        relaxation, converged, tuple(sorted(warnings)),
    )


def topology_assessment_v8(mol, inherited_warnings=()):
    values = TopologySample(*_topology_counts(mol))
    ring_topology = _ring_topology(mol)
    warnings = set(map(str, inherited_warnings))
    if any(size in (3, 4) for size in ring_topology.ring_sizes):
        warnings.add("three_or_four_member_ring")
    if values.bridgeheads:
        warnings.add("bridgehead_topology")
    if values.cumulenes:
        warnings.add("cumulene_topology")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        warnings.add("radical")
    if len(Chem.GetMolFrags(mol)) != 1:
        warnings.add("disconnected")
    return values, tuple(sorted(warnings))


def _condition_features(mol, fingerprint_size=512):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_size)
    fingerprint = generator.GetFingerprint(mol)
    bits = np.zeros((fingerprint_size,), dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fingerprint, bits)
    descriptors = np.asarray([
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Lipinski.NumRotatableBonds(mol),
        sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()),
    ], dtype=np.float64)
    return np.concatenate((bits, descriptors))


@dataclass
class ConditionSurrogateV8:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    target_mean: np.ndarray
    fingerprint_size: int
    validation_metrics: dict

    @classmethod
    def fit(cls, train_records, validation_records, ridge=10., fingerprint_size=512):
        def arrays(records):
            features, targets = [], []
            for row in records:
                mol = Chem.MolFromSmiles(str(row["smiles"]))
                if mol is None:
                    continue
                features.append(_condition_features(mol, fingerprint_size))
                targets.append((float(row["s1"]), float(row["t1"])))
            return np.asarray(features), np.asarray(targets)

        features, targets = arrays(train_records)
        mean = features.mean(0)
        scale = features.std(0)
        scale[scale < 1e-6] = 1.
        standardized = (features - mean) / scale
        target_mean = targets.mean(0)
        gram = standardized.T @ standardized + float(ridge) * np.eye(standardized.shape[1])
        coefficients = np.linalg.solve(gram, standardized.T @ (targets - target_mean))
        surrogate = cls(mean, scale, coefficients, target_mean, fingerprint_size, {})
        val_features, val_targets = arrays(validation_records)
        prediction = surrogate.predict_features(val_features)
        errors = prediction - val_targets
        baseline = ((val_targets - targets.mean(0)) ** 2).sum(0)
        residual = (errors ** 2).sum(0)
        surrogate.validation_metrics = {
            "samples": int(len(val_targets)),
            "s1_mae": float(np.abs(errors[:, 0]).mean()),
            "t1_mae": float(np.abs(errors[:, 1]).mean()),
            "s1_r2": float(1 - residual[0] / max(baseline[0], 1e-12)),
            "t1_r2": float(1 - residual[1] / max(baseline[1], 1e-12)),
        }
        return surrogate

    def predict_features(self, features):
        features = np.asarray(features, dtype=np.float64)
        return (features - self.feature_mean) / self.feature_scale @ self.coefficients + self.target_mean

    def predict(self, mol):
        return self.predict_features(_condition_features(mol, self.fingerprint_size)[None])[0]

    def to_metadata(self):
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "target_mean": self.target_mean.tolist(),
            "fingerprint_size": self.fingerprint_size,
            "validation_metrics": self.validation_metrics,
        }

    @classmethod
    def from_metadata(cls, payload):
        return cls(
            np.asarray(payload["feature_mean"], dtype=np.float64),
            np.asarray(payload["feature_scale"], dtype=np.float64),
            np.asarray(payload["coefficients"], dtype=np.float64),
            np.asarray(payload["target_mean"], dtype=np.float64),
            int(payload["fingerprint_size"]),
            dict(payload["validation_metrics"]),
        )


def dataset_fingerprint(root):
    root = Path(root)
    digest = hashlib.sha256()
    for name in ("manifest.json", "train.pt", "val.pt", "test.pt"):
        path = root / name
        digest.update(name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def source_fingerprints_v8():
    """Hash every transitive local source used by V8 train or sampling."""
    base = Path(__file__).resolve().parent
    names = (
        "model.py", "model_v5.py", "chemistry_v5.py", "chemistry_v6.py",
        "hp1kg_data.py", "model_v8.py", "chemistry_v8.py", "sample.py",
        "sample_v8.py", "train_v8.py",
    )
    output = {}
    for name in names:
        digest = hashlib.sha256()
        with (base / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        output[name] = digest.hexdigest()
    return output


def runtime_versions_v8():
    """Version lock for libraries that define kernels or chemistry rules."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "rdkit": rdkit.__version__,
        "scipy": scipy.__version__,
        "numpy": np.__version__,
        "networkx": nx.__version__,
    }


def metadata_fingerprint(metadata):
    payload = {key: value for key, value in metadata.items() if key != "metadata_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_v8_references(root, codec=None):
    codec = codec or build_train_codec_v8(root)
    train, atom_map = hp1kg_records(root, ("train",))
    validation, validation_atom_map = hp1kg_records(root, ("val",))
    if tuple(atom_map) != tuple(validation_atom_map):
        raise ValueError("atom maps differ across splits")
    chemistry = ChemistryReferenceV8.from_records(train, atom_map, codec)
    geometry = GeometryReferenceV8.from_records(train, atom_map, codec)
    surrogate = ConditionSurrogateV8.fit(train, validation)
    return chemistry, geometry, surrogate


__all__ = [
    "HP1KGV8Dataset", "collate_hp1kg_v8", "build_train_codec_v8",
    "ChemistryReferenceV8", "build_train_reference_v8", "support_warnings_v8", "ring_support_warnings_v8",
    "JointProjectionResultV8", "project_joint_graph_v8", "decode_joint_assignment_v8",
    "GeometryReferenceV8", "GeneratedGeometryAssessmentV8",
    "assess_generated_coordinates", "topology_assessment_v8",
    "ConditionSurrogateV8", "build_v8_references",
    "dataset_fingerprint", "metadata_fingerprint", "source_fingerprints_v8",
    "runtime_versions_v8",
    "StructureAssessment", "assess_structure", "TopologySample", "RingTopologySampleV8", "_ring_topology", "_ring_edge_targets",
]
