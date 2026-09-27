"""Heavy-atom chemistry codec and database-prior graph projection for v5.

Hydrogen is not a diffusion node or an independently sampled categorical
variable.  A heavy-atom state specifies element, formal charge, total valence
and aromatic role.  Once the heavy-atom bond graph is fixed, hydrogen count is
the residual valence and RDKit must agree with that residual during decoding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import collections
import json
import math
import random

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


AROMATIC_NONE = 0
AROMATIC_PI = 1
AROMATIC_HETERO_H = 2


@dataclass(frozen=True, order=True)
class HeavyAtomState:
    symbol: str
    formal_charge: int
    valence: int
    aromatic_role: int
    hydrogen_min: int
    hydrogen_max: int


class AtomStateCodec:
    """Observed joint heavy-atom chemistry vocabulary.

    ``hydrogen_min``/``hydrogen_max`` are support bounds observed for a joint
    (element, charge, valence, aromatic-role) state.  They constrain the graph
    projection but hydrogen itself is never diffused.
    """

    def __init__(self, states: Sequence[HeavyAtomState]):
        if not states:
            raise ValueError("atom-state vocabulary is empty")
        self.states = tuple(states)
        keys = [(s.symbol, s.formal_charge, s.valence, s.aromatic_role) for s in self.states]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate joint atom state")
        self._index = {key: i for i, key in enumerate(keys)}

    def __len__(self) -> int:
        return len(self.states)

    @staticmethod
    def aromatic_role(symbol: str, aromatic: bool, hydrogens: int) -> int:
        if not aromatic:
            return AROMATIC_NONE
        if symbol not in ("C", "B") and hydrogens > 0:
            return AROMATIC_HETERO_H
        return AROMATIC_PI

    @classmethod
    def from_records(cls, records: Iterable[Mapping], atom_map: Sequence[str]) -> "AtomStateCodec":
        support: dict[tuple[str, int, int, int], list[int]] = collections.defaultdict(list)
        for row in records:
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic_edges = torch.as_tensor(row["aromatic_edges"]).bool()
            for i, atom_index in enumerate(torch.as_tensor(row["atom_types"]).tolist()):
                symbol = atom_map[int(atom_index)]
                charge = int(torch.as_tensor(row["atom_charges"])[i])
                hydrogens = int(torch.as_tensor(row["atom_hydrogens"])[i])
                bond_valence = int(bonds[i].sum())
                valence = bond_valence + hydrogens
                role = cls.aromatic_role(symbol, bool(aromatic_edges[i].any()), hydrogens)
                support[(symbol, charge, valence, role)].append(hydrogens)
        states = [
            HeavyAtomState(*key, min(values), max(values))
            for key, values in sorted(support.items())
        ]
        return cls(states)

    @classmethod
    def from_metadata(cls, payload: Sequence[Mapping]) -> "AtomStateCodec":
        return cls([HeavyAtomState(**dict(item)) for item in payload])

    def to_metadata(self) -> list[dict]:
        return [asdict(state) for state in self.states]

    def encode_record(self, row: Mapping, atom_map: Sequence[str]) -> torch.Tensor:
        bonds = torch.as_tensor(row["kekule_bonds"]).long()
        aromatic_edges = torch.as_tensor(row["aromatic_edges"]).bool()
        out = []
        for i, atom_index in enumerate(torch.as_tensor(row["atom_types"]).tolist()):
            symbol = atom_map[int(atom_index)]
            charge = int(torch.as_tensor(row["atom_charges"])[i])
            hydrogens = int(torch.as_tensor(row["atom_hydrogens"])[i])
            valence = int(bonds[i].sum()) + hydrogens
            role = self.aromatic_role(symbol, bool(aromatic_edges[i].any()), hydrogens)
            key = (symbol, charge, valence, role)
            if key not in self._index:
                raise ValueError(f"atom state not in vocabulary: {key}")
            out.append(self._index[key])
        return torch.tensor(out, dtype=torch.long)

    def feature_tensors(self, device=None) -> dict[str, torch.Tensor]:
        symbols = sorted({state.symbol for state in self.states})
        symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
        return {
            "element": torch.tensor([symbol_index[x.symbol] for x in self.states], device=device),
            "charge": torch.tensor([x.formal_charge for x in self.states], device=device),
            "valence": torch.tensor([x.valence for x in self.states], device=device),
            "role": torch.tensor([x.aromatic_role for x in self.states], device=device),
            "h_min": torch.tensor([x.hydrogen_min for x in self.states], device=device),
            "h_max": torch.tensor([x.hydrogen_max for x in self.states], device=device),
        }


def hp1kg_records(root: str | Path, splits: Sequence[str] = ("train", "val", "test")) -> tuple[list[Mapping], tuple[str, ...]]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    rows: list[Mapping] = []
    for split in splits:
        path = root / f"{split}.pt"
        if path.exists():
            rows.extend(torch.load(path, map_location="cpu", weights_only=False))
    return rows, tuple(manifest["atom_map"])


def build_hp1kg_codec(root: str | Path) -> AtomStateCodec:
    rows, atom_map = hp1kg_records(root)
    return AtomStateCodec.from_records(rows, atom_map)


def collate_hp1kg_v5(rows: Sequence[Mapping], max_atoms: int, codec: AtomStateCodec, atom_map: Sequence[str]):
    from .hp1kg_data import collate_hp1kg

    out = collate_hp1kg(rows, max_atoms=max_atoms)
    state = torch.zeros(len(rows), max_atoms, dtype=torch.long)
    for batch_index, row in enumerate(rows):
        encoded = codec.encode_record(row, atom_map)
        state[batch_index, : len(encoded)] = encoded
    out["atom_state"] = state
    return out


def _ring_system_atom_max(mol: Chem.Mol) -> int:
    graph = nx.Graph()
    graph.add_nodes_from(range(mol.GetNumAtoms()))
    graph.add_edges_from((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds())
    cyclic = []
    for component in nx.biconnected_components(graph):
        if graph.subgraph(component).number_of_edges() >= len(component):
            cyclic.append(len(component))
    return max(cyclic, default=0)


@dataclass
class TopologyPrior:
    cycle_rank_by_count: dict[int, tuple[int, ...]]
    max_ring_system_by_count: dict[int, tuple[int, ...]]
    ring_size_counts: dict[int, int]
    max_observed_ring_size: int

    @classmethod
    def from_records(cls, records: Iterable[Mapping]) -> "TopologyPrior":
        ranks: dict[int, list[int]] = collections.defaultdict(list)
        systems: dict[int, list[int]] = collections.defaultdict(list)
        ring_sizes: collections.Counter[int] = collections.Counter()
        for row in records:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                continue
            count = mol.GetNumHeavyAtoms()
            cycle_rank = mol.GetNumBonds() - mol.GetNumAtoms() + 1
            ranks[count].append(max(0, int(cycle_rank)))
            systems[count].append(_ring_system_atom_max(mol))
            ring_sizes.update(map(len, mol.GetRingInfo().AtomRings()))
        return cls(
            {key: tuple(values) for key, values in ranks.items()},
            {key: tuple(values) for key, values in systems.items()},
            dict(ring_sizes),
            max(ring_sizes, default=6),
        )

    @classmethod
    def from_metadata(cls, payload: Mapping) -> "TopologyPrior":
        return cls(
            {int(k): tuple(map(int, v)) for k, v in payload["cycle_rank_by_count"].items()},
            {int(k): tuple(map(int, v)) for k, v in payload["max_ring_system_by_count"].items()},
            {int(k): int(v) for k, v in payload["ring_size_counts"].items()},
            int(payload["max_observed_ring_size"]),
        )

    def to_metadata(self) -> dict:
        return {
            "cycle_rank_by_count": {str(k): list(v) for k, v in self.cycle_rank_by_count.items()},
            "max_ring_system_by_count": {str(k): list(v) for k, v in self.max_ring_system_by_count.items()},
            "ring_size_counts": {str(k): int(v) for k, v in self.ring_size_counts.items()},
            "max_observed_ring_size": int(self.max_observed_ring_size),
        }

    @staticmethod
    def _nearest(values: Mapping[int, Sequence[int]], count: int, min_samples: int = 12) -> Sequence[int]:
        """Pool nearest atom-count bins until the conditional prior is stable."""
        if not values:
            return (0,)
        pooled: list[int] = []
        for key in sorted(values, key=lambda item: (abs(item - count), item)):
            pooled.extend(map(int, values[key]))
            if len(pooled) >= min_samples:
                break
        return tuple(pooled)

    def sample_targets(self, count: int, rng: random.Random) -> tuple[int, int]:
        ranks = self._nearest(self.cycle_rank_by_count, count)
        systems = self._nearest(self.max_ring_system_by_count, count)
        return int(rng.choice(ranks)), int(rng.choice(systems))

    def ring_log_bias(self, size: int, smoothing: float = 0.5) -> float:
        # All sizes receive finite support; unseen and extreme rings are softly
        # disfavoured instead of being forbidden.
        upper = max(self.max_observed_ring_size, size, 3)
        total = sum(self.ring_size_counts.values()) + smoothing * (upper - 2)
        probability = (self.ring_size_counts.get(size, 0) + smoothing) / max(total, smoothing)
        peak = (max(self.ring_size_counts.values(), default=1) + smoothing) / max(total, smoothing)
        return math.log(probability) - math.log(peak)


def build_hp1kg_topology_prior(root: str | Path) -> TopologyPrior:
    rows, _ = hp1kg_records(root, ("train",))
    return TopologyPrior.from_records(rows)


@dataclass
class ProjectionResult:
    state_ids: torch.Tensor
    bonds: torch.Tensor
    hydrogens: torch.Tensor
    valid: bool
    reason: str
    target_cycle_rank: int
    realized_cycle_rank: int


def _neutral_state_assignment(logits: torch.Tensor, codec: AtomStateCodec, target_charge: int = 0) -> list[int] | None:
    """Exact dynamic program for maximum-likelihood whole-molecule charge."""
    score = torch.log_softmax(logits.float(), -1).cpu()
    charges = [state.formal_charge for state in codec.states]
    dp: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for row in score:
        nxt: dict[int, tuple[float, list[int]]] = {}
        for total, (base_score, path) in dp.items():
            for state_index, charge in enumerate(charges):
                new_total = total + charge
                value = base_score + float(row[state_index])
                old = nxt.get(new_total)
                if old is None or value > old[0]:
                    nxt[new_total] = (value, path + [state_index])
        dp = nxt
    return dp.get(target_charge, (0.0, None))[1]


def _candidate_cycle_component(path: Sequence[int], systems: Sequence[set[int]]) -> tuple[set[int], list[int]]:
    merged = set(path)
    touched = []
    for index, system in enumerate(systems):
        if merged & system:
            merged.update(system)
            touched.append(index)
    return merged, touched


def project_heavy_graph(
    state_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    bond_type_logits: torch.Tensor,
    aromatic_edge_logits: torch.Tensor,
    codec: AtomStateCodec,
    prior: TopologyPrior,
    rng: random.Random,
    target_charge: int = 0,
    ring_prior_weight: float = 1.5,
    ring_system_weight: float = 0.75,
) -> ProjectionResult:
    """Jointly project states and edges to a neutral connected heavy graph.

    The number of closures and maximum ring-system scale are sampled from the
    database conditioned on heavy-atom count.  Every possible ring size keeps
    finite probability; the database distribution changes ranking rather than
    imposing a hand-written ring-size ban.
    """
    count = int(state_logits.shape[0])
    empty_bonds = torch.zeros(count, count, dtype=torch.long)
    empty_h = torch.zeros(count, dtype=torch.long)
    assignment = _neutral_state_assignment(state_logits, codec, target_charge)
    if assignment is None:
        return ProjectionResult(torch.zeros(count, dtype=torch.long), empty_bonds, empty_h, False, "no charge-neutral atom-state assignment", 0, 0)
    states = [codec.states[index] for index in assignment]
    state_ids = torch.tensor(assignment, dtype=torch.long)
    capacity = [state.valence - state.hydrogen_min for state in states]
    minimum = [max(0, state.valence - state.hydrogen_max) for state in states]
    if count > 1 and any(value < 1 for value in capacity):
        return ProjectionResult(state_ids, empty_bonds, empty_h, False, "zero-connectivity-capacity atom state", 0, 0)

    edge_probability = edge_logits.float().softmax(-1)[..., 1].cpu()
    type_probability = bond_type_logits.float().softmax(-1).cpu()
    aromatic_probability = aromatic_edge_logits.float().sigmoid().cpu()
    candidates = []
    for i in range(count):
        for j in range(i + 1, count):
            score = math.log(max(float(edge_probability[i, j]), 1e-12))
            candidates.append((score, i, j))
    candidates.sort(reverse=True)

    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    used = [0] * count
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for score, i, j in candidates:
        if find(i) == find(j) or used[i] >= capacity[i] or used[j] >= capacity[j]:
            continue
        graph.add_edge(i, j, score=score)
        used[i] += 1
        used[j] += 1
        left, right = find(i), find(j)
        parent[left] = right
        if graph.number_of_edges() == count - 1:
            break
    if count > 1 and not nx.is_connected(graph):
        return ProjectionResult(state_ids, empty_bonds, empty_h, False, "valence-constrained spanning tree failed", 0, 0)

    # Project the model closure count onto empirical support for this size.
    upper_triangle = torch.triu(torch.ones(count, count, dtype=torch.bool), 1)
    predicted_edges = int((edge_probability[upper_triangle] >= 0.5).sum())
    predicted_cycle_rank = max(0, predicted_edges - max(0, count - 1))
    supported_values = prior._nearest(prior.cycle_rank_by_count, count)
    supported = collections.Counter(supported_values)
    target_cycle_rank = min(supported, key=lambda value: (abs(value - predicted_cycle_rank), -supported[value], value)) if min(supported) <= predicted_cycle_rank <= max(supported) else int(rng.choice(supported_values))
    predicted_graph = nx.Graph()
    predicted_graph.add_nodes_from(range(count))
    predicted_graph.add_edges_from((i, j) for i in range(count) for j in range(i + 1, count) if float(edge_probability[i, j]) >= 0.5)
    predicted_system_atoms = max((len(component) for component in nx.biconnected_components(predicted_graph) if predicted_graph.subgraph(component).number_of_edges() >= len(component)), default=0)
    supported_system_values = prior._nearest(prior.max_ring_system_by_count, count)
    supported_systems = collections.Counter(supported_system_values)
    target_system_atoms = min(supported_systems, key=lambda value: (abs(value - predicted_system_atoms), -supported_systems[value], value)) if min(supported_systems) <= predicted_system_atoms <= max(supported_systems) else int(rng.choice(supported_system_values))
    target_cycle_rank = max(0, min(target_cycle_rank, count * (count - 1) // 2 - max(0, count - 1)))
    predicted_usable = (count <= 1 or nx.is_connected(predicted_graph)) and predicted_cycle_rank == target_cycle_rank and predicted_system_atoms == target_system_atoms and all(predicted_graph.degree(i) <= capacity[i] for i in range(count))
    if predicted_usable:
        graph = predicted_graph.copy()
        used = [graph.degree(i) for i in range(count)]
        closure_steps = 0
    else:
        closure_steps = target_cycle_rank
    selected = {tuple(sorted(edge)) for edge in graph.edges()}
    cycle_systems: list[set[int]] = []
    for _ in range(closure_steps):
        ranked = []
        for base_score, i, j in candidates:
            if (i, j) in selected or used[i] >= capacity[i] or used[j] >= capacity[j]:
                continue
            try:
                path = nx.shortest_path(graph, i, j)
            except nx.NetworkXNoPath:
                continue
            ring_size = len(path)
            system, _ = _candidate_cycle_component(path, cycle_systems)
            system_penalty = max(0, len(system) - target_system_atoms)
            fusion_overlap = sum(1 for node in path if any(node in old for old in cycle_systems))
            score = base_score + ring_prior_weight * prior.ring_log_bias(ring_size)
            score -= ring_system_weight * system_penalty + 0.2 * fusion_overlap
            ranked.append((score, base_score, i, j, path, system))
        if not ranked:
            break
        _, base_score, i, j, path, system = max(ranked)
        graph.add_edge(i, j, score=base_score)
        selected.add((i, j))
        used[i] += 1
        used[j] += 1
        merged, touched = _candidate_cycle_component(path, cycle_systems)
        for index in reversed(touched):
            cycle_systems.pop(index)
        cycle_systems.append(merged)

    realized_cycle_rank = graph.number_of_edges() - count + (1 if count else 0)
    if realized_cycle_rank != target_cycle_rank:
        return ProjectionResult(state_ids, empty_bonds, empty_h, False, "target cycle rank infeasible under valence constraints", target_cycle_rank, realized_cycle_rank)

    bonds = torch.zeros(count, count, dtype=torch.long)
    for i, j in graph.edges():
        bonds[i, j] = bonds[j, i] = 1

    # Kekulize predicted aromatic cyclic components globally by matching.  The
    # hetero-H role cannot receive a double bond in the matching.
    aromatic_graph = nx.Graph()
    aromatic_graph.add_nodes_from(range(count))
    for i, j in graph.edges():
        if states[i].aromatic_role and states[j].aromatic_role and float(aromatic_probability[i, j]) >= 0.5:
            aromatic_graph.add_edge(i, j, weight=float(type_probability[i, j, 1]))
    for component in nx.biconnected_components(aromatic_graph):
        sub = aromatic_graph.subgraph(component)
        if sub.number_of_edges() < sub.number_of_nodes():
            continue
        matching_graph = nx.Graph()
        for i, j, data in sub.edges(data=True):
            if states[i].aromatic_role == AROMATIC_HETERO_H or states[j].aromatic_role == AROMATIC_HETERO_H:
                continue
            if used[i] < capacity[i] and used[j] < capacity[j]:
                single = max(float(type_probability[i, j, 0]), 1e-12)
                double = max(float(type_probability[i, j, 1]), 1e-12)
                log_odds = math.log(double) - math.log(single)
                if log_odds > 0: matching_graph.add_edge(i, j, weight=log_odds)
        for i, j in nx.max_weight_matching(matching_graph, maxcardinality=False):
            bonds[i, j] = bonds[j, i] = 2
            used[i] += 1
            used[j] += 1

    # Upgrade remaining bonds according to predicted type while respecting the
    # joint-state valence capacity.
    for order in (2, 3):
        upgrades = []
        for i, j in graph.edges():
            current = int(bonds[i, j])
            desired = int(type_probability[i, j].argmax()) + 1
            if current != order - 1 or desired < order:
                continue
            gain = float(type_probability[i, j, order - 1] - type_probability[i, j, order - 2])
            upgrades.append((gain, i, j, order))
        for _, i, j, _ in sorted(upgrades, reverse=True):
            if used[i] >= capacity[i] or used[j] >= capacity[j]:
                continue
            bonds[i, j] = bonds[j, i] = order
            used[i] += 1
            used[j] += 1

    # If an observed state requires a minimum heavy-atom bond valence, use the
    # best existing edges for further upgrades.  No new closure is introduced.
    while any(used[i] < minimum[i] for i in range(count)):
        options = []
        for i, j in graph.edges():
            current = int(bonds[i, j])
            if current >= 3 or used[i] >= capacity[i] or used[j] >= capacity[j]:
                continue
            need = int(used[i] < minimum[i]) + int(used[j] < minimum[j])
            if not need:
                continue
            options.append((need + float(type_probability[i, j, current]), i, j))
        if not options:
            break
        _, i, j = max(options)
        bonds[i, j] += 1
        bonds[j, i] += 1
        used[i] += 1
        used[j] += 1

    hydrogens = torch.tensor([state.valence - used[i] for i, state in enumerate(states)], dtype=torch.long)
    for i, (state, hydrogen) in enumerate(zip(states, hydrogens.tolist())):
        if not state.hydrogen_min <= hydrogen <= state.hydrogen_max:
            return ProjectionResult(state_ids, bonds, hydrogens, False, f"atom {i} hydrogen residual outside observed state support", target_cycle_rank, realized_cycle_rank)
    return ProjectionResult(state_ids, bonds, hydrogens, True, "", target_cycle_rank, realized_cycle_rank)


@dataclass(frozen=True)
class DecodeResultV5:
    molecule: object | None
    smiles: str | None
    valid: bool
    connected: bool
    aromatic_rings: int
    hydrogens: tuple[int, ...]
    error: str | None = None


def decode_heavy_graph(
    state_ids: Sequence[int],
    bonds: torch.Tensor | Sequence[Sequence[int]],
    codec: AtomStateCodec,
    require_neutral: bool = True,
    require_closed_shell: bool = True,
) -> DecodeResultV5:
    """Decode a heavy graph and require RDKit to agree with residual H counts."""
    try:
        bond_tensor = torch.as_tensor(bonds).long()
        if bond_tensor.shape != (len(state_ids), len(state_ids)):
            raise ValueError("bond matrix shape mismatch")
        if not torch.equal(bond_tensor, bond_tensor.T):
            raise ValueError("asymmetric bond matrix")
        rw = Chem.RWMol()
        expected_h = []
        for index, state_id in enumerate(state_ids):
            state = codec.states[int(state_id)]
            bond_valence = int(bond_tensor[index].sum())
            hydrogen = state.valence - bond_valence
            if not state.hydrogen_min <= hydrogen <= state.hydrogen_max:
                raise ValueError(f"hydrogen residual {hydrogen} outside state support at atom {index}")
            atom = Chem.Atom(state.symbol)
            atom.SetFormalCharge(state.formal_charge)
            if state.aromatic_role == AROMATIC_HETERO_H:
                atom.SetNumExplicitHs(hydrogen)
                atom.SetNoImplicit(True)
            else:
                atom.SetNoImplicit(False)
            rw.AddAtom(atom)
            expected_h.append(hydrogen)
        bond_types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        for i in range(len(state_ids)):
            for j in range(i + 1, len(state_ids)):
                order = int(bond_tensor[i, j])
                if order:
                    if order not in bond_types:
                        raise ValueError(f"unsupported bond order {order}")
                    rw.AddBond(i, j, bond_types[order])
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        connected = len(Chem.GetMolFrags(mol)) == 1
        actual_h = tuple(int(atom.GetTotalNumHs(includeNeighbors=True)) for atom in mol.GetAtoms())
        if actual_h != tuple(expected_h):
            raise ValueError(f"RDKit hydrogen assignment disagrees: expected={tuple(expected_h)} actual={actual_h}")
        radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        net_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        if not connected:
            raise ValueError("decoded graph is disconnected")
        if require_closed_shell and radicals:
            raise ValueError(f"decoded graph has {radicals} radical electrons")
        if require_neutral and net_charge:
            raise ValueError(f"decoded graph has net charge {net_charge}")
        aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(mol))
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        return DecodeResultV5(mol, smiles, True, connected, aromatic_rings, actual_h)
    except Exception as exc:
        return DecodeResultV5(None, None, False, False, 0, (), f"{type(exc).__name__}: {exc}")


def materialize_hydrogens(mol: Chem.Mol, add_coordinates: bool = True) -> Chem.Mol:
    """Materialize implicit H exactly once after strict heavy-graph decoding."""
    base = Chem.Mol(mol)
    Chem.SanitizeMol(base)
    if any(atom.GetAtomicNum() == 1 for atom in base.GetAtoms()):
        raise ValueError("hydrogen atoms are already materialized")
    if any(atom.GetNumRadicalElectrons() for atom in base.GetAtoms()):
        raise ValueError("refusing to materialize hydrogen on an open-shell graph")
    return Chem.AddHs(base, explicitOnly=False, addCoords=add_coordinates)


__all__ = [
    "AROMATIC_NONE", "AROMATIC_PI", "AROMATIC_HETERO_H", "HeavyAtomState",
    "AtomStateCodec", "TopologyPrior", "ProjectionResult", "DecodeResultV5",
    "build_hp1kg_codec", "build_hp1kg_topology_prior", "collate_hp1kg_v5",
    "project_heavy_graph", "decode_heavy_graph", "materialize_hydrogens",
]
