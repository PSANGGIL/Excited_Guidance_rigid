"""Database-supported chemistry projection and structural evaluation for v6.

V6 keeps the lossless v5 heavy-atom codec, but treats aromatic membership as
an explicit reverse-diffusion edge state.  The final graph is still written as
a Kekule single/double graph so RDKit, the training target and the sampler use
one bond convention.
"""
from __future__ import annotations

import collections
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import networkx as nx
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .chemistry_v5 import (
    AROMATIC_HETERO_H,
    AROMATIC_NONE,
    AROMATIC_PI,
    AtomStateCodec,
    DecodeResultV5,
    HeavyAtomState,
    ProjectionResult,
    TopologyPrior,
    build_hp1kg_codec,
    collate_hp1kg_v5,
    decode_heavy_graph,
    hp1kg_records,
    materialize_hydrogens,
    project_heavy_graph as project_heavy_graph_v5,
)


def _canonical_bond_signature(left: int, right: int, order: int, aromatic: bool) -> tuple[int, int, int, int]:
    left, right = sorted((int(left), int(right)))
    return left, right, int(order), int(bool(aromatic))


def _local_signature(index: int, states: Sequence[int], bonds: torch.Tensor, aromatic: torch.Tensor) -> tuple:
    neighbours = []
    for other in range(len(states)):
        order = int(bonds[index, other])
        if order:
            neighbours.append((int(states[other]), order, int(bool(aromatic[index, other]))))
    return int(states[index]), tuple(sorted(neighbours))


def _topology_counts(mol: Chem.Mol) -> tuple[int, int, int, int, int, int]:
    """Return cycle rank, max ring system, small rings, bridgeheads, cumulenes, aromatic rings."""
    count = mol.GetNumAtoms()
    components = max(1, len(Chem.GetMolFrags(mol)))
    cycle_rank = max(0, mol.GetNumBonds() - count + components)
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    graph.add_edges_from((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds())
    max_system = max(
        (len(component) for component in nx.biconnected_components(graph)
         if graph.subgraph(component).number_of_edges() >= len(component)),
        default=0,
    )
    rings = tuple(mol.GetRingInfo().AtomRings())
    small = sum(len(ring) in (3, 4) for ring in rings)
    bridgeheads = int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    cumulenes = sum(
        sum(bond.GetBondTypeAsDouble() == 2.0 for bond in atom.GetBonds()) >= 2
        for atom in mol.GetAtoms()
    )
    aromatic = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    return cycle_rank, max_system, int(small), bridgeheads, int(cumulenes), aromatic


@dataclass(frozen=True)
class TopologySample:
    cycle_rank: int
    max_ring_system: int
    small_rings: int
    bridgeheads: int
    cumulenes: int
    aromatic_rings: int


@dataclass
class ChemistryPriorV6:
    """Empirical topology and exact local chemistry support.

    Topology rows are kept jointly (rather than as independent histograms), so
    fused-ring, bridgehead and cumulene correlations from the DB are retained.
    Exact bond/local supports are built from all DB splits; topology frequencies
    are estimated from train only.
    """

    base: TopologyPrior
    topology_by_count: dict[int, tuple[TopologySample, ...]]
    bond_signatures: frozenset[tuple[int, int, int, int]]
    local_signatures: frozenset[tuple]

    @classmethod
    def from_records(
        cls,
        train_records: Iterable[Mapping],
        support_records: Iterable[Mapping],
        atom_map: Sequence[str],
        codec: AtomStateCodec,
    ) -> "ChemistryPriorV6":
        train_rows = list(train_records)
        topology: dict[int, list[TopologySample]] = collections.defaultdict(list)
        for row in train_rows:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                continue
            values = _topology_counts(mol)
            topology[mol.GetNumHeavyAtoms()].append(TopologySample(*values))

        bond_support: set[tuple[int, int, int, int]] = set()
        local_support: set[tuple] = set()
        for row in support_records:
            states = codec.encode_record(row, atom_map).tolist()
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            for i in range(len(states)):
                local_support.add(_local_signature(i, states, bonds, aromatic))
                for j in range(i + 1, len(states)):
                    if int(bonds[i, j]):
                        bond_support.add(_canonical_bond_signature(states[i], states[j], int(bonds[i, j]), bool(aromatic[i, j])))
        return cls(
            TopologyPrior.from_records(train_rows),
            {key: tuple(values) for key, values in topology.items()},
            frozenset(bond_support),
            frozenset(local_support),
        )

    @classmethod
    def from_metadata(cls, payload: Mapping) -> "ChemistryPriorV6":
        return cls(
            TopologyPrior.from_metadata(payload["base"]),
            {
                int(key): tuple(TopologySample(**item) for item in values)
                for key, values in payload["topology_by_count"].items()
            },
            frozenset(tuple(map(int, value)) for value in payload["bond_signatures"]),
            frozenset(
                (int(value[0]), tuple(tuple(map(int, neighbour)) for neighbour in value[1]))
                for value in payload["local_signatures"]
            ),
        )

    def to_metadata(self) -> dict:
        return {
            "base": self.base.to_metadata(),
            "topology_by_count": {
                str(key): [asdict(value) for value in values]
                for key, values in self.topology_by_count.items()
            },
            "bond_signatures": [list(value) for value in sorted(self.bond_signatures)],
            "local_signatures": [
                [value[0], [list(neighbour) for neighbour in value[1]]]
                for value in sorted(self.local_signatures)
            ],
        }

    def nearest(self, count: int, minimum: int = 24) -> tuple[TopologySample, ...]:
        pooled: list[TopologySample] = []
        for key in sorted(self.topology_by_count, key=lambda item: (abs(item - count), item)):
            pooled.extend(self.topology_by_count[key])
            if len(pooled) >= minimum:
                break
        return tuple(pooled)

    def sample(self, count: int, rng: random.Random) -> TopologySample:
        values = self.nearest(count)
        return rng.choice(values) if values else TopologySample(0, 0, 0, 0, 0, 0)

    def topology_acceptance(self, count: int, observed: TopologySample) -> float:
        """Frequency-ratio prior for the three historically overgenerated motifs."""
        rows = self.nearest(count)
        if not rows:
            return 1.0
        ratios = []
        for field in ("small_rings", "bridgeheads", "cumulenes"):
            histogram = collections.Counter(getattr(row, field) for row in rows)
            peak = max(histogram.values())
            ratios.append((histogram.get(getattr(observed, field), 0) + 0.25) / (peak + 0.25))
        return float(math.prod(ratios) ** (1 / len(ratios)))


def build_hp1kg_chemistry_prior(root: str | Path, codec: AtomStateCodec | None = None) -> ChemistryPriorV6:
    codec = codec or build_hp1kg_codec(root)
    train, atom_map = hp1kg_records(root, ("train",))
    support, support_atom_map = hp1kg_records(root, ("train", "val", "test"))
    if tuple(atom_map) != tuple(support_atom_map):
        raise ValueError("atom maps differ between topology and support records")
    return ChemistryPriorV6.from_records(train, support, atom_map, codec)


def support_warnings(
    state_ids: Sequence[int],
    bonds: torch.Tensor,
    aromatic_edges: torch.Tensor,
    prior: ChemistryPriorV6,
) -> tuple[str, ...]:
    states = list(map(int, state_ids))
    bonds = torch.as_tensor(bonds).long()
    aromatic_edges = torch.as_tensor(aromatic_edges).bool()
    warnings: set[str] = set()
    for i in range(len(states)):
        if _local_signature(i, states, bonds, aromatic_edges) not in prior.local_signatures:
            warnings.add("unseen_local_atom_environment")
        for j in range(i + 1, len(states)):
            if int(bonds[i, j]):
                signature = _canonical_bond_signature(states[i], states[j], int(bonds[i, j]), bool(aromatic_edges[i, j]))
                if signature not in prior.bond_signatures:
                    warnings.add("unseen_bond_environment")
    return tuple(sorted(warnings))


@dataclass
class ProjectionResultV6:
    state_ids: torch.Tensor
    bonds: torch.Tensor
    aromatic_edges: torch.Tensor
    hydrogens: torch.Tensor
    valid: bool
    reason: str
    target_cycle_rank: int
    realized_cycle_rank: int
    topology_acceptance: float
    warnings: tuple[str, ...]


def project_heavy_graph_v6(
    state_logits: torch.Tensor,
    edge_logits: torch.Tensor,
    bond_type_logits: torch.Tensor,
    aromatic_edge_logits: torch.Tensor,
    codec: AtomStateCodec,
    prior: ChemistryPriorV6,
    rng: random.Random,
    target_charge: int = 0,
    enforce_support: bool = True,
    apply_topology_prior: bool = True,
) -> ProjectionResultV6:
    """Project one jointly denoised state and apply DB-derived support priors."""
    if aromatic_edge_logits.ndim != 3 or aromatic_edge_logits.shape[-1] != 2:
        raise ValueError("v6 aromatic logits must have shape [N,N,2]")
    aromatic_log_odds = aromatic_edge_logits.float()[..., 1] - aromatic_edge_logits.float()[..., 0]
    base = project_heavy_graph_v5(
        state_logits, edge_logits, bond_type_logits, aromatic_log_odds,
        codec, prior.base, rng, target_charge,
        ring_prior_weight=2.5, ring_system_weight=1.25,
    )
    empty_aromatic = torch.zeros_like(base.bonds, dtype=torch.bool)
    if not base.valid:
        return ProjectionResultV6(
            base.state_ids, base.bonds, empty_aromatic, base.hydrogens, False,
            base.reason, base.target_cycle_rank, base.realized_cycle_rank, 0.0, (),
        )
    decoded = decode_heavy_graph(base.state_ids.tolist(), base.bonds, codec)
    if not decoded.valid:
        return ProjectionResultV6(
            base.state_ids, base.bonds, empty_aromatic, base.hydrogens, False,
            decoded.error or "strict decode failed", base.target_cycle_rank,
            base.realized_cycle_rank, 0.0, (),
        )
    aromatic = torch.zeros_like(base.bonds, dtype=torch.bool)
    for bond in decoded.molecule.GetBonds():
        if bond.GetIsAromatic():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            aromatic[i, j] = aromatic[j, i] = True
    warnings = support_warnings(base.state_ids.tolist(), base.bonds, aromatic, prior)
    if enforce_support and warnings:
        return ProjectionResultV6(
            base.state_ids, base.bonds, aromatic, base.hydrogens, False,
            ";".join(warnings), base.target_cycle_rank, base.realized_cycle_rank, 0.0, warnings,
        )
    values = TopologySample(*_topology_counts(decoded.molecule))
    acceptance = prior.topology_acceptance(len(base.state_ids), values)
    if apply_topology_prior and rng.random() > acceptance:
        return ProjectionResultV6(
            base.state_ids, base.bonds, aromatic, base.hydrogens, False,
            "rejected_by_db_topology_prior", base.target_cycle_rank,
            base.realized_cycle_rank, acceptance, warnings,
        )
    return ProjectionResultV6(
        base.state_ids, base.bonds, aromatic, base.hydrogens, True, "",
        base.target_cycle_rank, base.realized_cycle_rank, acceptance, warnings,
    )


@dataclass(frozen=True)
class StructureAssessment:
    etkdg_success: bool
    forcefield: str
    forcefield_converged: bool
    forcefield_final_energy: float | None
    strain_energy: float | None
    strain_energy_per_heavy_atom: float | None
    steric_clashes: int | None
    small_rings: int
    bridgeheads: int
    cumulenes: int
    chemically_clean: bool
    warnings: tuple[str, ...]


def assess_structure(
    mol: Chem.Mol,
    inherited_warnings: Sequence[str] = (),
    evaluate_3d: bool = True,
    seed: int = 1701,
) -> StructureAssessment:
    """Run explicit structural warnings plus ETKDG/FF strain diagnostics."""
    _, _, small, bridgeheads, cumulenes, _ = _topology_counts(mol)
    warnings = set(map(str, inherited_warnings))
    if small:
        warnings.add("three_or_four_member_ring")
    if bridgeheads:
        warnings.add("bridgehead_topology")
    if cumulenes:
        warnings.add("cumulene_topology")
    if any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()):
        warnings.add("radical")
    if len(Chem.GetMolFrags(mol)) != 1:
        warnings.add("disconnected")

    etkdg_success = False
    field = "none"
    converged = False
    energy = energy_per_atom = None
    final_energy = None
    clashes = None
    if evaluate_3d:
        embedded = Chem.AddHs(Chem.Mol(mol), addCoords=False)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) & 0x7FFFFFFF
        params.useRandomCoords = False
        params.clearConfs = True
        try:
            etkdg_success = AllChem.EmbedMolecule(embedded, params) == 0
            if etkdg_success:
                if AllChem.MMFFHasAllMoleculeParams(embedded):
                    field = "MMFF94"
                    properties = AllChem.MMFFGetMoleculeProperties(embedded)
                    initial_forcefield = AllChem.MMFFGetMoleculeForceField(embedded, properties)
                    initial_energy = float(initial_forcefield.CalcEnergy())
                    status = AllChem.MMFFOptimizeMolecule(embedded, maxIters=500)
                    final_forcefield = AllChem.MMFFGetMoleculeForceField(embedded, properties)
                else:
                    field = "UFF"
                    initial_forcefield = AllChem.UFFGetMoleculeForceField(embedded)
                    initial_energy = float(initial_forcefield.CalcEnergy())
                    status = AllChem.UFFOptimizeMolecule(embedded, maxIters=500)
                    final_forcefield = AllChem.UFFGetMoleculeForceField(embedded)
                converged = status == 0
                final_energy = float(final_forcefield.CalcEnergy())
                # Relaxation energy is a more interpretable ETKDG strain
                # diagnostic than an absolute MMFF/UFF energy across elements.
                energy = max(0., initial_energy - final_energy)
                energy_per_atom = energy / max(1, mol.GetNumHeavyAtoms())
                conformer = embedded.GetConformer()
                topology_distance = Chem.GetDistanceMatrix(embedded)
                table = Chem.GetPeriodicTable()
                clashes = 0
                for i in range(embedded.GetNumAtoms()):
                    for j in range(i + 1, embedded.GetNumAtoms()):
                        if topology_distance[i, j] < 3:
                            continue
                        distance = conformer.GetAtomPosition(i).Distance(conformer.GetAtomPosition(j))
                        cutoff = 0.65 * (
                            table.GetRvdw(embedded.GetAtomWithIdx(i).GetAtomicNum())
                            + table.GetRvdw(embedded.GetAtomWithIdx(j).GetAtomicNum())
                        )
                        clashes += int(distance < cutoff)
            else:
                warnings.add("etkdg_failed")
        except Exception as exc:
            warnings.add("etkdg_exception:" + type(exc).__name__)
        if etkdg_success and not converged:
            warnings.add("forcefield_not_converged")
        if clashes:
            warnings.add("steric_clash")
    clean = not warnings
    return StructureAssessment(
        etkdg_success, field, converged, final_energy, energy, energy_per_atom, clashes,
        small, bridgeheads, cumulenes, clean, tuple(sorted(warnings)),
    )


__all__ = [
    "AROMATIC_NONE", "AROMATIC_PI", "AROMATIC_HETERO_H", "HeavyAtomState",
    "AtomStateCodec", "DecodeResultV5", "ChemistryPriorV6", "TopologySample",
    "ProjectionResultV6", "StructureAssessment", "build_hp1kg_codec",
    "build_hp1kg_chemistry_prior", "collate_hp1kg_v5", "decode_heavy_graph",
    "materialize_hydrogens", "project_heavy_graph_v6", "support_warnings",
    "assess_structure",
]
