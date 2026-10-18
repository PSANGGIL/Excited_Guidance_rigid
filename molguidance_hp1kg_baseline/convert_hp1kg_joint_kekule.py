#!/usr/bin/env python3
"""Export HP1KG as joint atom states and Kekule bond classes.

The codec state is ``(element, formal charge, attached hydrogen count)``.
Its vocabulary is built from all declared splits so encoding/decoding remains
lossless, while ``train_supported`` separately records which states a
generative model is allowed to sample. Edges use only
``NONE/SINGLE/DOUBLE/TRIPLE``; aromaticity is recovered by RDKit sanitization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from rdkit import Chem
from torch.nn.functional import one_hot


SPLITS = ("train", "val", "test")
N_BOND_TYPES = 4
MIN_CHARGE = -2
MAX_CHARGE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz"
        ),
    )
    parser.add_argument("--property", choices=("s1", "t1", "est"), default="est")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_ranges(lengths: list[int]) -> torch.Tensor:
    lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    ranges = torch.zeros((len(lengths), 2), dtype=torch.int32)
    ranges[:, 1] = torch.cumsum(lengths_tensor, dim=0)
    ranges[1:, 0] = ranges[:-1, 1]
    return ranges


def atom_state_keys(record: dict[str, Any]) -> list[tuple[int, int, int]]:
    atom_types = torch.as_tensor(record["atom_types"], dtype=torch.int64)
    charges = torch.as_tensor(record["atom_charges"], dtype=torch.int64)
    hydrogens = torch.as_tensor(record["atom_hydrogens"], dtype=torch.int64)
    if not (atom_types.shape == charges.shape == hydrogens.shape):
        raise ValueError("atom type, charge, and hydrogen arrays must align")
    return [
        (int(atom_type), int(charge), int(hydrogen))
        for atom_type, charge, hydrogen in zip(atom_types, charges, hydrogens)
    ]


def build_atom_state_metadata(
    split_records: dict[str, list[dict[str, Any]]], atom_map: list[str]
) -> tuple[dict[str, Any], dict[tuple[int, int, int], int]]:
    split_states = {
        split: {
            state
            for record in records
            for state in atom_state_keys(record)
        }
        for split, records in split_records.items()
    }
    union_states = sorted(set().union(*split_states.values()))
    state_to_idx = {state: idx for idx, state in enumerate(union_states)}
    train_states = split_states["train"]

    train_valence_counts: dict[tuple[int, int, int], Counter[int]] = {
        state: Counter() for state in union_states
    }
    for record_idx, record in enumerate(split_records["train"]):
        states = atom_state_keys(record)
        kekule = torch.as_tensor(record["kekule_bonds"], dtype=torch.int64)
        if kekule.shape != (len(states), len(states)):
            raise ValueError(
                f"train record {record_idx}: Kekule matrix and atom states disagree"
            )
        heavy_valences = kekule.sum(dim=1)
        for state, valence in zip(states, heavy_valences.tolist()):
            if valence < 0:
                raise ValueError(
                    f"train record {record_idx}: negative heavy-atom valence"
                )
            train_valence_counts[state][int(valence)] += 1

    metadata = {
        "version": 2,
        "definition": "(element, formal_charge, attached_hydrogen_count)",
        "states": [
            {
                "atom_type_index": atom_type,
                "symbol": atom_map[atom_type],
                "formal_charge": charge,
                "hydrogens": hydrogens,
                "train_heavy_valence_counts": {
                    str(valence): count
                    for valence, count in sorted(
                        train_valence_counts[(atom_type, charge, hydrogens)].items()
                    )
                },
            }
            for atom_type, charge, hydrogens in union_states
        ],
        "train_supported": [state in train_states for state in union_states],
        "split_state_counts": {
            split: len(states) for split, states in split_states.items()
        },
    }
    return metadata, state_to_idx


def uniform_spanning_tree_edge_weights(kekule: torch.Tensor) -> torch.Tensor:
    """Return UST edge-inclusion probabilities via effective resistance.

    For an unweighted connected graph, P(edge in a uniform spanning tree) is
    the edge effective resistance. Foster's theorem then gives a total weight
    of n-1, providing a strong per-molecule integrity check.
    """
    kekule = torch.as_tensor(kekule, dtype=torch.int64)
    if kekule.ndim != 2 or kekule.shape[0] != kekule.shape[1]:
        raise ValueError("Kekule adjacency must be square")
    if not torch.equal(kekule, kekule.T):
        raise ValueError("Kekule adjacency must be symmetric")
    n_atoms = kekule.shape[0]
    adjacency = (kekule > 0).to(torch.float64)
    adjacency.fill_diagonal_(0.0)
    if n_atoms <= 1:
        return torch.zeros_like(adjacency, dtype=torch.float32)

    laplacian = torch.diag(adjacency.sum(dim=1)) - adjacency
    rank = int(torch.linalg.matrix_rank(laplacian, hermitian=True))
    if rank != n_atoms - 1:
        raise ValueError(
            f"molecular graph must be connected, Laplacian rank {rank} != {n_atoms - 1}"
        )
    laplacian_pinv = torch.linalg.pinv(laplacian, hermitian=True)
    diagonal = torch.diagonal(laplacian_pinv)
    resistance = (
        diagonal[:, None]
        + diagonal[None, :]
        - 2.0 * laplacian_pinv
    )
    resistance = 0.5 * (resistance + resistance.T)
    weights = torch.where(adjacency.bool(), resistance.clamp_min(0.0), 0.0)
    foster_sum = float(torch.triu(weights, diagonal=1).sum())
    if abs(foster_sum - (n_atoms - 1)) > 1.0e-6 * max(1, n_atoms):
        raise ValueError(
            f"effective-resistance Foster sum {foster_sum} != {n_atoms - 1}"
        )
    if bool((weights > 1.0 + 1.0e-6).any()):
        raise ValueError("UST edge-inclusion probability exceeds one")
    return weights.to(torch.float32)

def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def rebuild_molecule(
    states: torch.Tensor,
    bonds: torch.Tensor,
    metadata: dict[str, Any],
) -> Chem.Mol:
    rw_mol = Chem.RWMol()
    vocab = metadata["states"]
    for state_idx in states.tolist():
        state = vocab[int(state_idx)]
        atom = Chem.Atom(state["symbol"])
        atom.SetFormalCharge(int(state["formal_charge"]))
        atom.SetNumExplicitHs(int(state["hydrogens"]))
        atom.SetNoImplicit(True)
        rw_mol.AddAtom(atom)

    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    upper = torch.triu(bonds, diagonal=1)
    for src, dst in upper.nonzero(as_tuple=False).tolist():
        rw_mol.AddBond(src, dst, bond_map[int(upper[src, dst])])
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def property_value(record: dict[str, Any], property_name: str) -> float:
    if property_name == "est":
        value = float(
            torch.tensor(record["s1"], dtype=torch.float32)
            - torch.tensor(record["t1"], dtype=torch.float32)
        )
    else:
        value = float(record[property_name])
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"non-finite {property_name}")
    return value


def convert_records(
    records: list[dict[str, Any]],
    atom_map: list[str],
    property_name: str,
    metadata: dict[str, Any],
    state_to_idx: dict[tuple[int, int, int], int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    positions: list[torch.Tensor] = []
    atom_types: list[torch.Tensor] = []
    atom_charges: list[torch.Tensor] = []
    atom_hydrogens: list[torch.Tensor] = []
    atom_states: list[torch.Tensor] = []
    bond_types: list[torch.Tensor] = []
    bond_idxs: list[torch.Tensor] = []
    bond_connectivity_weights: list[torch.Tensor] = []
    bond_aromatic_flags: list[torch.Tensor] = []
    properties: list[float] = []
    smiles: list[str] = []
    node_lengths: list[int] = []
    edge_lengths: list[int] = []
    aromatic_molecules = 0
    identity_matches = 0
    foster_sum_error_max = 0.0
    connectivity_weight_min = float("inf")
    connectivity_weight_max = 0.0

    for record_idx, record in enumerate(records):
        pos = torch.as_tensor(record["atom_positions"], dtype=torch.float32)
        type_idx = torch.as_tensor(record["atom_types"], dtype=torch.int64)
        charges = torch.as_tensor(record["atom_charges"], dtype=torch.int32)
        hydrogens = torch.as_tensor(record["atom_hydrogens"], dtype=torch.int64)
        radicals = torch.as_tensor(record["atom_radicals"], dtype=torch.int64)
        kekule = torch.as_tensor(record["kekule_bonds"], dtype=torch.int64)
        aromatic = torch.as_tensor(record["aromatic_edges"], dtype=torch.bool)
        n_atoms = int(type_idx.numel())

        if pos.shape != (n_atoms, 3) or not torch.isfinite(pos).all():
            raise ValueError(f"record {record_idx}: invalid coordinates")
        if not (charges.shape == hydrogens.shape == radicals.shape == (n_atoms,)):
            raise ValueError(f"record {record_idx}: node arrays do not align")
        if kekule.shape != (n_atoms, n_atoms) or not torch.equal(kekule, kekule.T):
            raise ValueError(f"record {record_idx}: invalid Kekule adjacency")
        if aromatic.shape != (n_atoms, n_atoms) or not torch.equal(aromatic, aromatic.T):
            raise ValueError(f"record {record_idx}: invalid aromatic adjacency")
        if bool((aromatic & (kekule == 0)).any()):
            raise ValueError(f"record {record_idx}: aromatic flag on a non-bond")
        if kekule.min() < 0 or kekule.max() >= N_BOND_TYPES:
            raise ValueError(f"record {record_idx}: unsupported Kekule bond class")
        if type_idx.min() < 0 or type_idx.max() >= len(atom_map):
            raise ValueError(f"record {record_idx}: atom type outside atom_map")
        if charges.min() < MIN_CHARGE or charges.max() > MAX_CHARGE:
            raise ValueError(f"record {record_idx}: formal charge outside supported range")
        if hydrogens.min() < 0:
            raise ValueError(f"record {record_idx}: negative attached-H count")
        if bool((radicals != 0).any()):
            raise ValueError(
                f"record {record_idx}: radical state is not represented by this codec"
            )

        state_indices = torch.tensor(
            [state_to_idx[state] for state in atom_state_keys(record)],
            dtype=torch.int64,
        )
        rebuilt = rebuild_molecule(state_indices, kekule, metadata)
        source_mol = Chem.MolFromSmiles(str(record["smiles"]))
        if source_mol is None:
            raise ValueError(f"record {record_idx}: source SMILES cannot be parsed")
        source_identity = canonical_smiles(source_mol)
        rebuilt_identity = canonical_smiles(rebuilt)
        if source_identity != rebuilt_identity:
            raise ValueError(
                f"record {record_idx}: oracle identity mismatch: "
                f"{source_identity} != {rebuilt_identity}"
            )
        identity_matches += 1

        upper = torch.triu(kekule, diagonal=1)
        edge_index = upper.nonzero(as_tuple=False).to(torch.int32)
        edge_type = upper[edge_index[:, 0], edge_index[:, 1]].to(torch.int32)
        connectivity_matrix = uniform_spanning_tree_edge_weights(kekule)
        edge_connectivity = connectivity_matrix[
            edge_index[:, 0], edge_index[:, 1]
        ]
        edge_aromatic = aromatic[edge_index[:, 0], edge_index[:, 1]]
        foster_error = abs(float(edge_connectivity.sum()) - max(0, n_atoms - 1))
        foster_sum_error_max = max(foster_sum_error_max, foster_error)
        if edge_connectivity.numel():
            connectivity_weight_min = min(
                connectivity_weight_min, float(edge_connectivity.min())
            )
            connectivity_weight_max = max(
                connectivity_weight_max, float(edge_connectivity.max())
            )

        positions.append(pos)
        atom_types.append(one_hot(type_idx, num_classes=len(atom_map)).to(torch.bool))
        atom_charges.append(charges)
        atom_hydrogens.append(hydrogens)
        atom_states.append(state_indices)
        bond_types.append(edge_type)
        bond_idxs.append(edge_index)
        bond_connectivity_weights.append(edge_connectivity)
        bond_aromatic_flags.append(edge_aromatic)
        properties.append(property_value(record, property_name))
        smiles.append(str(record["smiles"]))
        node_lengths.append(n_atoms)
        edge_lengths.append(int(edge_type.numel()))
        aromatic_molecules += int(bool(aromatic.any()))

    data = {
        "smiles": smiles,
        "positions": torch.cat(positions, dim=0),
        "atom_types": torch.cat(atom_types, dim=0),
        "atom_charges": torch.cat(atom_charges, dim=0),
        "atom_hydrogens": torch.cat(atom_hydrogens, dim=0),
        "atom_states": torch.cat(atom_states, dim=0),
        "bond_types": torch.cat(bond_types, dim=0),
        "bond_idxs": torch.cat(bond_idxs, dim=0),
        "bond_connectivity_weights": torch.cat(
            bond_connectivity_weights, dim=0
        ),
        "bond_aromatic_flags": torch.cat(bond_aromatic_flags, dim=0),
        "node_idx_array": index_ranges(node_lengths),
        "edge_idx_array": index_ranges(edge_lengths),
        "properties": torch.tensor(properties, dtype=torch.float32),
        "property_names": property_name,
    }
    stats = {
        "molecules": len(records),
        "atoms": sum(node_lengths),
        "bonds": sum(edge_lengths),
        "atom_count_min": min(node_lengths),
        "atom_count_max": max(node_lengths),
        "property_mean": float(data["properties"].mean()),
        "property_std": float(data["properties"].std()),
        "aromatic_molecules": aromatic_molecules,
        "oracle_identity_matches": identity_matches,
        "oracle_identity_fraction": identity_matches / len(records),
        "foster_sum_error_max": foster_sum_error_max,
        "connectivity_weight_min": (
            0.0 if connectivity_weight_min == float("inf")
            else connectivity_weight_min
        ),
        "connectivity_weight_max": connectivity_weight_max,
    }
    return data, stats


def marginal_distributions(
    data: dict[str, Any], n_atom_types: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    atom_types = data["atom_types"]
    charges = data["atom_charges"]
    atom_idx = atom_types.to(torch.float32).argmax(dim=1)
    p_a = atom_types.to(torch.float32).sum(dim=0)
    p_a /= p_a.sum()

    p_c = torch.zeros(6, dtype=torch.float32)
    for charge in range(MIN_CHARGE, MAX_CHARGE + 1):
        p_c[charge - MIN_CHARGE] = (charges == charge).sum()
    p_c /= p_c.sum()

    bond_counts = torch.zeros(N_BOND_TYPES, dtype=torch.float32)
    bond_counts.scatter_add_(
        0,
        data["bond_types"].to(torch.int64),
        torch.ones_like(data["bond_types"], dtype=torch.float32),
    )
    node_lengths = data["node_idx_array"][:, 1] - data["node_idx_array"][:, 0]
    total_pairs = int(sum(int(n) * (int(n) - 1) // 2 for n in node_lengths))
    bond_counts[0] = total_pairs - int(data["bond_types"].numel())
    p_e = bond_counts / bond_counts.sum()

    p_c_given_a = torch.zeros((n_atom_types, 6), dtype=torch.float32)
    for atom_type in range(n_atom_types):
        mask = atom_idx == atom_type
        for charge in range(MIN_CHARGE, MAX_CHARGE + 1):
            p_c_given_a[atom_type, charge - MIN_CHARGE] = (
                charges[mask] == charge
            ).sum()
    row_sums = p_c_given_a.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    p_c_given_a /= row_sums
    return p_a, p_c, p_e, p_c_given_a


def save_split(output: Path, name: str, data: dict[str, Any]) -> None:
    prefix = f"{name}_data"
    torch.save(data, output / f"{prefix}_processed.pt")
    with (output / f"{prefix}_smiles.pkl").open("wb") as handle:
        pickle.dump(data["smiles"], handle)
    lengths = data["node_idx_array"][:, 1] - data["node_idx_array"][:, 0]
    torch.save(torch.unique(lengths, return_counts=True), output / f"{prefix}_n_atoms_histogram.pt")
    props = data["properties"]
    torch.save(
        {
            "mean": props.mean(),
            "std": props.std(),
            "property_names": data["property_names"],
        },
        output / f"{prefix}_property_normalization.pt",
    )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    atom_map = list(manifest["atom_map"])
    split_records = {
        split: torch.load(
            source / f"{split}.pt", map_location="cpu", weights_only=False
        )
        for split in SPLITS
    }
    metadata, state_to_idx = build_atom_state_metadata(split_records, atom_map)
    (output / "joint_atom_state_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )

    converted = {}
    report: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "property": args.property,
        "chemistry_representation": "joint_atom_state+kekule_bond",
        "atom_map": atom_map,
        "atom_state_count": len(metadata["states"]),
        "train_supported_atom_state_count": sum(metadata["train_supported"]),
        "edge_classes": ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE"],
        "source_manifest_sha256": sha256(manifest_path),
        "source_split_sha256": {},
        "splits": {},
    }
    for split in SPLITS:
        split_path = source / f"{split}.pt"
        data, stats = convert_records(
            split_records[split],
            atom_map,
            args.property,
            metadata,
            state_to_idx,
        )
        converted[split] = data
        save_split(output, split, data)
        report["source_split_sha256"][split] = sha256(split_path)
        report["splits"][split] = stats

    save_split(output, "train_a", converted["train"])
    save_split(output, "train_b", converted["train"])
    torch.save(
        marginal_distributions(converted["train"], len(atom_map)),
        output / "train_data_marginal_dists.pt",
    )
    state_counts = torch.bincount(
        converted["train"]["atom_states"], minlength=len(metadata["states"])
    ).float()
    torch.save(
        state_counts / state_counts.sum(),
        output / "train_data_joint_state_marginal.pt",
    )
    report["train_alias_policy"] = "train_a=train_b=train"
    report["oracle_identity_total"] = sum(
        split_stats["oracle_identity_matches"]
        for split_stats in report["splits"].values()
    )
    report["oracle_identity_expected"] = sum(
        split_stats["molecules"] for split_stats in report["splits"].values()
    )
    if report["oracle_identity_total"] != report["oracle_identity_expected"]:
        raise RuntimeError("full-dataset oracle identity validation failed")
    (output / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
