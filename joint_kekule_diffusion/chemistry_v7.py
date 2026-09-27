"""V7 chemistry solver, train-only references, geometry and condition evaluation."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import networkx as nx
import numpy as np
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
from .model_v7 import (
    EDGE_AROMATIC_DOUBLE,
    EDGE_AROMATIC_SINGLE,
    EDGE_AROMATIC_TRIPLE,
    JOINT_EDGE_CLASSES,
    decode_joint_edge_state,
)


def build_train_codec_v7(root: str | Path) -> AtomStateCodec:
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


class HP1KGV7Dataset(HP1KGDataset):
    """HP1KG split with topology targets cached once in process memory."""

    def __init__(self, root, split, condition_stats=None, atom_map=None):
        super().__init__(root, split, condition_stats, atom_map)
        enriched = []
        for row in self.rows:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None:
                raise ValueError(f"invalid cached SMILES: {row['smiles']}")
            values = _topology_counts(mol)
            enriched.append({
                **row,
                "topology_target": torch.tensor(
                    [values[0], values[1], values[2], values[3], values[4]],
                    dtype=torch.float32,
                ),
            })
        self.rows = enriched


def collate_hp1kg_v7(rows, max_atoms, codec, atom_map):
    output = collate_hp1kg_v5(rows, max_atoms, codec, atom_map)
    output["topology_target"] = torch.stack([row["topology_target"] for row in rows])
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


@dataclass
class ChemistryReferenceV7:
    """Train-only local support and joint topology reference."""

    topology_by_count: dict[int, tuple[TopologySample, ...]]
    bond_signatures: frozenset[tuple[int, int, int, int]]
    local_signatures: frozenset[tuple]

    @classmethod
    def from_records(cls, records, atom_map, codec):
        topology = collections.defaultdict(list)
        bond_support = set()
        local_support = set()
        for row in records:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                continue
            topology[mol.GetNumHeavyAtoms()].append(TopologySample(*_topology_counts(mol)))
            states = codec.encode_record(row, atom_map).tolist()
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            for i in range(len(states)):
                local_support.add(_local_signature(i, states, bonds, aromatic))
                for j in range(i + 1, len(states)):
                    if int(bonds[i, j]):
                        bond_support.add(_bond_signature(
                            states[i], states[j], int(bonds[i, j]), bool(aromatic[i, j]),
                        ))
        return cls(
            {int(key): tuple(values) for key, values in topology.items()},
            frozenset(bond_support),
            frozenset(local_support),
        )

    @classmethod
    def from_metadata(cls, payload):
        return cls(
            {
                int(key): tuple(TopologySample(**value) for value in values)
                for key, values in payload["topology_by_count"].items()
            },
            frozenset(tuple(map(int, value)) for value in payload["bond_signatures"]),
            frozenset(
                (int(value[0]), tuple(tuple(map(int, neighbour)) for neighbour in value[1]))
                for value in payload["local_signatures"]
            ),
        )

    def to_metadata(self):
        return {
            "topology_by_count": {
                str(key): [asdict(value) for value in values]
                for key, values in sorted(self.topology_by_count.items())
            },
            "bond_signatures": [list(value) for value in sorted(self.bond_signatures)],
            "local_signatures": [
                [value[0], [list(neighbour) for neighbour in value[1]]]
                for value in sorted(self.local_signatures)
            ],
        }

    def nearest(self, count: int, minimum: int = 24):
        pooled = []
        for key in sorted(self.topology_by_count, key=lambda item: (abs(item - count), item)):
            pooled.extend(self.topology_by_count[key])
            if len(pooled) >= minimum:
                break
        return tuple(pooled)

    @staticmethod
    def topology_distance(count: int, left: TopologySample, right: TopologySample) -> float:
        """Chemistry-weighted distance between joint topology states."""
        cycle_scale = max(1., math.sqrt(max(1, count)))
        system_scale = max(2., .25 * max(1, count))
        return float(
            abs(left.cycle_rank - right.cycle_rank) / cycle_scale
            + abs(left.max_ring_system - right.max_ring_system) / system_scale
            + 2.0 * abs(left.small_rings - right.small_rings)
            + 1.5 * abs(left.bridgeheads - right.bridgeheads)
            + abs(left.cumulenes - right.cumulenes)
            + .25 * abs(left.aromatic_rings - right.aromatic_rings)
        )

    def topology_score(self, count: int, observed: TopologySample) -> float:
        """Kernel-smoothed count-conditional DB topology support in [0,1]."""
        rows = self.nearest(count)
        if not rows:
            return 1.
        histogram = collections.Counter(rows)
        peak = max(histogram.values())
        return float(max(
            math.exp(-self.topology_distance(count, observed, candidate))
            * (frequency + .25) / (peak + .25)
            for candidate, frequency in histogram.items()
        ))

    def sample_topology(self, count: int, rng: random.Random) -> TopologySample | None:
        """Sample the empirical joint topology distribution near atom count."""
        rows = self.nearest(count)
        return rows[rng.randrange(len(rows))] if rows else None


def build_train_reference_v7(root, codec=None):
    codec = codec or build_train_codec_v7(root)
    rows, atom_map = hp1kg_records(root, ("train",))
    return ChemistryReferenceV7.from_records(rows, atom_map, codec)


def support_warnings_v7(state_ids, bonds, aromatic_edges, reference):
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
class JointProjectionResultV7:
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
    for index, state_id in enumerate(state_ids):
        state = codec.states[int(state_id)]
        atom = decoded.molecule.GetAtomWithIdx(index)
        role = AtomStateCodec.aromatic_role(
            state.symbol, atom.GetIsAromatic(), int(atom.GetTotalNumHs(includeNeighbors=True)),
        )
        if role != state.aromatic_role:
            return None, bonds, perceived, f"aromatic role mismatch at atom {index}"
    return decoded, bonds, perceived, ""


def decode_joint_assignment_v7(state_ids, edge_state, codec):
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
    topology_reference=None, topology_target=None, topology_prior_weight=1.5,
    topology_reproject_threshold=.2, rng=None, excluded_assignments=(),
):
    """Solve joint chemistry with a soft count-conditional topology prior.

    Connected-graph cycle rank is linear in selected edge count, so its
    deviation from a sampled DB topology is included directly in the MILP
    objective. Non-linear ring-system/bridgehead/small-ring statistics are
    scored on feasible solutions and low-support solutions are probabilistically
    excluded before resolving. Nothing is imposed as a hard topology cap.
    """
    count, state_types = state_logits.shape
    state_logp = torch.log_softmax(state_logits.float(), -1).cpu()
    edge_logp = torch.log_softmax(
        (edge_logits.float() + edge_logits.float().transpose(0, 1)) * .5, -1,
    ).cpu()
    state_candidates = _candidate_states(state_logp, codec, state_top_k)
    edge_pairs = _candidate_edges(edge_logp, edge_top_k, probability_threshold, complete_edges)
    rng = rng or random.Random(0)

    objective = []
    lower_bounds = []
    upper_bounds = []
    integrality = []
    atom_variables = {}
    edge_variables = {}
    flow_variables = {}

    def variable(cost, lower, upper, integer):
        index = len(objective)
        objective.append(float(cost))
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
        integrality.append(int(integer))
        return index

    for atom, candidates in enumerate(state_candidates):
        for state in candidates:
            atom_variables[atom, state] = variable(-float(state_logp[atom, state]), 0, 1, 1)
    for edge_index, (left, right) in enumerate(edge_pairs):
        none_score = float(edge_logp[left, right, 0])
        for edge_class in range(1, JOINT_EDGE_CLASSES):
            edge_variables[edge_index, edge_class] = variable(
                -(float(edge_logp[left, right, edge_class]) - none_score), 0, 1, 1,
            )
        flow_variables[edge_index, 0] = variable(0, 0, max(0, count - 1), 0)
        flow_variables[edge_index, 1] = variable(0, 0, max(0, count - 1), 0)

    topology_positive = topology_negative = None
    if topology_target is not None and topology_prior_weight > 0:
        topology_positive = variable(topology_prior_weight, 0, max(1, count), 0)
        topology_negative = variable(topology_prior_weight, 0, max(1, count), 0)

    rows = []
    row_lower = []
    row_upper = []

    def constraint(coefficients, lower=-np.inf, upper=np.inf):
        rows.append(dict(coefficients))
        row_lower.append(float(lower))
        row_upper.append(float(upper))

    for atom, candidates in enumerate(state_candidates):
        constraint({atom_variables[atom, state]: 1 for state in candidates}, 1, 1)
    constraint({
        atom_variables[atom, state]: codec.states[state].formal_charge
        for atom, candidates in enumerate(state_candidates) for state in candidates
    }, target_charge, target_charge)

    incident = collections.defaultdict(list)
    for edge_index, (left, right) in enumerate(edge_pairs):
        constraint({edge_variables[edge_index, state]: 1 for state in range(1, JOINT_EDGE_CLASSES)}, upper=1)
        incident[left].append(edge_index)
        incident[right].append(edge_index)
        aromatic_vars = {
            edge_variables[edge_index, EDGE_AROMATIC_SINGLE]: 1,
            edge_variables[edge_index, EDGE_AROMATIC_DOUBLE]: 1,
            edge_variables[edge_index, EDGE_AROMATIC_TRIPLE]: 1,
        }
        for atom in (left, right):
            endpoint = dict(aromatic_vars)
            for state in state_candidates[atom]:
                if codec.states[state].aromatic_role > 0:
                    endpoint[atom_variables[atom, state]] = endpoint.get(atom_variables[atom, state], 0) - 1
            constraint(endpoint, upper=0)

    if topology_positive is not None:
        # For a connected graph: cycle_rank = E - V + 1.
        edge_count = {
            edge_variables[edge_index, edge_class]: 1
            for edge_index in range(len(edge_pairs))
            for edge_class in range(1, JOINT_EDGE_CLASSES)
        }
        edge_count[topology_positive] = -1
        edge_count[topology_negative] = 1
        target_edges = max(0, count - 1 + int(topology_target.cycle_rank))
        constraint(edge_count, target_edges, target_edges)

    order_for_class = {1: 1, 2: 2, 3: 3, 4: 1, 5: 2, 6: 3}
    for atom, candidates in enumerate(state_candidates):
        heavy = {}
        aromatic_degree = {}
        aromatic_double = {}
        for edge_index in incident[atom]:
            for edge_class, order in order_for_class.items():
                heavy[edge_variables[edge_index, edge_class]] = order
            aromatic_degree[edge_variables[edge_index, EDGE_AROMATIC_SINGLE]] = 1
            aromatic_degree[edge_variables[edge_index, EDGE_AROMATIC_DOUBLE]] = 1
            aromatic_degree[edge_variables[edge_index, EDGE_AROMATIC_TRIPLE]] = 1
            aromatic_double[edge_variables[edge_index, EDGE_AROMATIC_DOUBLE]] = 1
            aromatic_double[edge_variables[edge_index, EDGE_AROMATIC_TRIPLE]] = 1
        minimum = dict(heavy)
        maximum = dict(heavy)
        for state in candidates:
            item = codec.states[state]
            minimum[atom_variables[atom, state]] = -(item.valence - item.hydrogen_max)
            maximum[atom_variables[atom, state]] = -(item.valence - item.hydrogen_min)
        constraint(minimum, lower=0)
        constraint(maximum, upper=0)

        lower_aromatic = dict(aromatic_degree)
        upper_aromatic = dict(aromatic_degree)
        double_limit = dict(aromatic_double)
        for state in candidates:
            role = codec.states[state].aromatic_role
            if role > 0:
                lower_aromatic[atom_variables[atom, state]] = -2
                upper_aromatic[atom_variables[atom, state]] = -3
            if role == AROMATIC_HETERO_H:
                double_limit[atom_variables[atom, state]] = 1
        constraint(lower_aromatic, lower=0)
        constraint(upper_aromatic, upper=0)
        constraint(double_limit, upper=1)

    if count > 1:
        for edge_index in range(len(edge_pairs)):
            presence = {edge_variables[edge_index, state]: -(count - 1) for state in range(1, JOINT_EDGE_CLASSES)}
            for direction in (0, 1):
                flow = dict(presence)
                flow[flow_variables[edge_index, direction]] = 1
                constraint(flow, upper=0)
        for atom in range(count):
            balance = {}
            for edge_index, (left, right) in enumerate(edge_pairs):
                if atom == left:
                    balance[flow_variables[edge_index, 0]] = 1
                    balance[flow_variables[edge_index, 1]] = -1
                elif atom == right:
                    balance[flow_variables[edge_index, 1]] = 1
                    balance[flow_variables[edge_index, 0]] = -1
            target = count - 1 if atom == 0 else -1
            constraint(balance, target, target)

    binary_indices = list(atom_variables.values()) + list(edge_variables.values())

    def exclude_assignment(states, edges):
        selected = []
        for atom, state in enumerate(map(int, states)):
            variable_index = atom_variables.get((atom, state))
            if variable_index is not None:
                selected.append(variable_index)
        edges = torch.as_tensor(edges).long()
        for edge_index, (left, right) in enumerate(edge_pairs):
            edge_class = int(edges[left, right])
            variable_index = edge_variables.get((edge_index, edge_class))
            if variable_index is not None:
                selected.append(variable_index)
        if selected:
            constraint({index: 1 for index in selected}, upper=len(selected) - 1)

    for states, edges in excluded_assignments:
        exclude_assignment(states, edges)

    last_reason = "MILP returned no solution"
    best = None
    reprojections = 0
    for attempt in range(max_attempts):
        matrix = sparse.lil_matrix((len(rows), len(objective)), dtype=np.float64)
        for row_index, coefficients in enumerate(rows):
            for column, value in coefficients.items():
                matrix[row_index, column] = value
        result = milp(
            np.asarray(objective),
            integrality=np.asarray(integrality),
            bounds=Bounds(np.asarray(lower_bounds), np.asarray(upper_bounds)),
            constraints=LinearConstraint(matrix.tocsr(), np.asarray(row_lower), np.asarray(row_upper)),
            options={"time_limit": float(time_limit) / max(1, max_attempts), "mip_rel_gap": 0.01, "presolve": True},
        )
        if result.x is None:
            last_reason = f"MILP status={result.status}: {result.message}"
            break
        values = result.x
        state_ids = [
            max(candidates, key=lambda state: values[atom_variables[atom, state]])
            for atom, candidates in enumerate(state_candidates)
        ]
        edge_state = torch.zeros(count, count, dtype=torch.long)
        for edge_index, (left, right) in enumerate(edge_pairs):
            choices = [(values[edge_variables[edge_index, state]], state) for state in range(1, JOINT_EDGE_CLASSES)]
            value, edge_class = max(choices)
            if value > .5:
                edge_state[left, right] = edge_state[right, left] = edge_class
        decoded, bonds, aromatic, reason = _decoded_assignment(state_ids, edge_state, codec)
        selected = {index for index in binary_indices if values[index] > .5}
        no_good = {index: (1 if index in selected else -1) for index in binary_indices}
        if decoded is None:
            last_reason = reason
            constraint(no_good, upper=len(selected) - 1)
            continue

        score = 1.
        if topology_reference is not None:
            observed = TopologySample(*_topology_counts(decoded.molecule))
            score = topology_reference.topology_score(count, observed)
        status = f"status={result.status};attempt={attempt + 1};topology_score={score:.6g}"
        candidate = (state_ids, edge_state, bonds, aromatic, decoded, status, score, reprojections)
        if best is None or score > best[6]:
            best = candidate
        acceptance = 1. if topology_reproject_threshold <= 0 else min(
            1., score / topology_reproject_threshold,
        )
        if topology_reference is None or score >= topology_reproject_threshold or rng.random() < acceptance:
            return candidate
        reprojections += 1
        constraint(no_good, upper=len(selected) - 1)

    if best is not None:
        status = best[5] + f";low_prior_fallback=1;reprojections={reprojections}"
        return (*best[:5], status, best[6], reprojections)
    return None, None, None, None, None, last_reason, 0., reprojections


def project_joint_graph_v7(
    state_logits, joint_edge_logits, codec, target_charge=0,
    solver_time_limit=8., solver_attempts=4, topology_reference=None,
    topology_prior_weight=1.5, topology_reproject_threshold=.2,
    topology_seed=0,
):
    """Joint MAP projection with soft empirical topology regularization."""
    count = int(state_logits.shape[0])
    if joint_edge_logits.shape != (count, count, JOINT_EDGE_CLASSES):
        raise ValueError(f"joint edge logits must have shape [N,N,{JOINT_EDGE_CLASSES}]")
    if topology_prior_weight < 0 or not 0 <= topology_reproject_threshold <= 1:
        raise ValueError("invalid topology prior/reprojection configuration")
    rng = random.Random(int(topology_seed))
    topology_target = (
        topology_reference.sample_topology(count, rng)
        if topology_reference is not None else None
    )
    state_ids = state_logits.float().argmax(-1).cpu().tolist()
    edge_state = _symmetric_argmax(joint_edge_logits.cpu())
    decoded, bonds, aromatic, reason = _decoded_assignment(state_ids, edge_state, codec)
    fast_fallback = None
    excluded = []
    initial_reprojections = 0
    if decoded is not None:
        score = 1.
        if topology_reference is not None:
            score = topology_reference.topology_score(
                count, TopologySample(*_topology_counts(decoded.molecule)),
            )
        acceptance = 1. if topology_reproject_threshold <= 0 else min(
            1., score / topology_reproject_threshold,
        )
        if topology_reference is None or score >= topology_reproject_threshold or rng.random() < acceptance:
            return JointProjectionResultV7(
                torch.tensor(state_ids), edge_state, bonds, aromatic,
                torch.tensor(decoded.hydrogens), True, "", False,
                f"argmax_feasible;topology_score={score:.6g}", score, 0,
            )
        fast_fallback = (state_ids, edge_state, bonds, aromatic, decoded, score)
        excluded.append((state_ids, edge_state))
        initial_reprojections = 1

    common = dict(
        topology_reference=topology_reference,
        topology_target=topology_target,
        topology_prior_weight=topology_prior_weight,
        topology_reproject_threshold=topology_reproject_threshold,
        rng=rng,
        excluded_assignments=excluded,
    )
    first_budget = max(.1, float(solver_time_limit) * .5)
    solved = _solve_joint_milp(
        state_logits.cpu(), joint_edge_logits.cpu(), codec, target_charge,
        first_budget, max(1, solver_attempts), **common,
    )
    state_ids, edge_state, bonds, aromatic, decoded, status, score, reprojections = solved
    if decoded is None:
        complete = count <= 24
        solved = _solve_joint_milp(
            state_logits.cpu(), joint_edge_logits.cpu(), codec, target_charge,
            max(.1, float(solver_time_limit) - first_budget), max(1, solver_attempts),
            state_top_k=(len(codec.states) if complete else min(16, len(codec.states))),
            edge_top_k=(max(1, count - 1) if complete else min(16, max(1, count - 1))),
            probability_threshold=(0. if complete else .005), complete_edges=complete,
            **common,
        )
        state_ids, edge_state, bonds, aromatic, decoded, expanded_status, score, expanded_reprojections = solved
        reprojections += expanded_reprojections
        status = f"initial={status};expanded={expanded_status}"
    reprojections += initial_reprojections
    if decoded is None and fast_fallback is not None:
        state_ids, edge_state, bonds, aromatic, decoded, score = fast_fallback
        status = f"{status};topology_reprojection_failed_fallback=1"
    if decoded is None:
        size = count
        return JointProjectionResultV7(
            torch.zeros(size, dtype=torch.long), torch.zeros(size, size, dtype=torch.long),
            torch.zeros(size, size, dtype=torch.long), torch.zeros(size, size, dtype=torch.bool),
            torch.zeros(size, dtype=torch.long), False, status or reason, True, status,
            0., reprojections,
        )
    return JointProjectionResultV7(
        torch.tensor(state_ids), edge_state, bonds, aromatic,
        torch.tensor(decoded.hydrogens), True, "", True, status,
        score, reprojections,
    )


@dataclass
class GeometryReferenceV7:
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
class GeneratedGeometryAssessmentV7:
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
        return GeneratedGeometryAssessmentV7(False, None, None, None, "none", None, None, None, False, ("nonfinite_or_bad_coordinate_shape",))
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
    return GeneratedGeometryAssessmentV7(
        True, bond_z_rms, bond_outliers, clashes, field, initial, final,
        relaxation, converged, tuple(sorted(warnings)),
    )


def topology_assessment_v7(mol, inherited_warnings=()):
    values = TopologySample(*_topology_counts(mol))
    warnings = set(map(str, inherited_warnings))
    if values.small_rings:
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
class ConditionSurrogateV7:
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


def source_fingerprints_v7():
    base = Path(__file__).resolve().parent
    names = ("model.py", "model_v7.py", "chemistry_v7.py", "sample_v7.py", "train_v7.py")
    output = {}
    for name in names:
        digest = hashlib.sha256()
        with (base / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        output[name] = digest.hexdigest()
    return output


def metadata_fingerprint(metadata):
    payload = {key: value for key, value in metadata.items() if key != "metadata_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_v7_references(root, codec=None):
    codec = codec or build_train_codec_v7(root)
    train, atom_map = hp1kg_records(root, ("train",))
    validation, validation_atom_map = hp1kg_records(root, ("val",))
    if tuple(atom_map) != tuple(validation_atom_map):
        raise ValueError("atom maps differ across splits")
    chemistry = ChemistryReferenceV7.from_records(train, atom_map, codec)
    geometry = GeometryReferenceV7.from_records(train, atom_map, codec)
    surrogate = ConditionSurrogateV7.fit(train, validation)
    return chemistry, geometry, surrogate


__all__ = [
    "HP1KGV7Dataset", "collate_hp1kg_v7", "build_train_codec_v7",
    "ChemistryReferenceV7", "build_train_reference_v7", "support_warnings_v7",
    "JointProjectionResultV7", "project_joint_graph_v7", "decode_joint_assignment_v7",
    "GeometryReferenceV7", "GeneratedGeometryAssessmentV7",
    "assess_generated_coordinates", "topology_assessment_v7",
    "ConditionSurrogateV7", "build_v7_references",
    "dataset_fingerprint", "metadata_fingerprint", "source_fingerprints_v7",
    "StructureAssessment", "assess_structure", "TopologySample",
]

