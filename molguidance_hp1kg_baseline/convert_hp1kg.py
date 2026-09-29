#!/usr/bin/env python3
"""Convert the existing HP1KG tensors to MolGuidance's native dataset schema.

This adapter deliberately lives outside the pristine MolGuidance clone.  It
does not change the official model, CTMC, loss, or sampler implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.nn.functional import one_hot


SPLITS = ("train", "val", "test")
N_BOND_TYPES = 5  # none, single, double, triple, aromatic
MIN_CHARGE = -2
MAX_CHARGE = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("joint_kekule_diffusion/data_hp1kg_nevpt2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("molguidance_hp1kg_baseline/data_s1"),
    )
    parser.add_argument("--property", choices=("s1", "t1"), default="s1")
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


def convert_records(
    records: list[dict[str, Any]], atom_map: list[str], property_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    positions: list[torch.Tensor] = []
    atom_types: list[torch.Tensor] = []
    atom_charges: list[torch.Tensor] = []
    bond_types: list[torch.Tensor] = []
    bond_idxs: list[torch.Tensor] = []
    properties: list[float] = []
    smiles: list[str] = []
    node_lengths: list[int] = []
    edge_lengths: list[int] = []

    radical_molecules = 0
    atoms_with_hydrogen = 0
    total_implicit_hydrogens = 0
    aromatic_molecules = 0

    for record_idx, record in enumerate(records):
        pos = torch.as_tensor(record["atom_positions"], dtype=torch.float32)
        type_idx = torch.as_tensor(record["atom_types"], dtype=torch.int64)
        charges = torch.as_tensor(record["atom_charges"], dtype=torch.int32)
        kekule = torch.as_tensor(record["kekule_bonds"], dtype=torch.int64)
        aromatic = torch.as_tensor(record["aromatic_edges"], dtype=torch.bool)

        n_atoms = int(type_idx.numel())
        if pos.shape != (n_atoms, 3):
            raise ValueError(f"record {record_idx}: invalid position shape {pos.shape}")
        if kekule.shape != (n_atoms, n_atoms) or aromatic.shape != (n_atoms, n_atoms):
            raise ValueError(f"record {record_idx}: invalid adjacency shape")
        if not torch.isfinite(pos).all():
            raise ValueError(f"record {record_idx}: non-finite coordinates")
        if type_idx.min() < 0 or type_idx.max() >= len(atom_map):
            raise ValueError(f"record {record_idx}: atom type outside atom_map")
        if charges.min() < MIN_CHARGE or charges.max() > MAX_CHARGE:
            raise ValueError(
                f"record {record_idx}: formal charge outside [{MIN_CHARGE}, {MAX_CHARGE}]"
            )
        if not torch.equal(kekule, kekule.T):
            raise ValueError(f"record {record_idx}: asymmetric Kekule adjacency")
        if not torch.equal(aromatic, aromatic.T):
            raise ValueError(f"record {record_idx}: asymmetric aromatic adjacency")

        joint_bonds = kekule.clone()
        joint_bonds[aromatic] = 4
        joint_bonds.fill_diagonal_(0)
        if joint_bonds.min() < 0 or joint_bonds.max() >= N_BOND_TYPES:
            raise ValueError(f"record {record_idx}: unsupported bond class")

        upper = torch.triu(joint_bonds, diagonal=1)
        edge_index = upper.nonzero(as_tuple=False).to(torch.int32)
        edge_type = upper[edge_index[:, 0], edge_index[:, 1]].to(torch.int32)

        prop = float(record[property_name])
        if not torch.isfinite(torch.tensor(prop)):
            raise ValueError(f"record {record_idx}: non-finite {property_name}")

        positions.append(pos)
        atom_types.append(one_hot(type_idx, num_classes=len(atom_map)).to(torch.bool))
        atom_charges.append(charges)
        bond_types.append(edge_type)
        bond_idxs.append(edge_index)
        properties.append(prop)
        smiles.append(str(record["smiles"]))
        node_lengths.append(n_atoms)
        edge_lengths.append(int(edge_type.numel()))

        radicals = torch.as_tensor(record.get("atom_radicals", []), dtype=torch.int64)
        hydrogens = torch.as_tensor(record.get("atom_hydrogens", []), dtype=torch.int64)
        radical_molecules += int(radicals.numel() > 0 and bool((radicals != 0).any()))
        atoms_with_hydrogen += int((hydrogens > 0).sum())
        total_implicit_hydrogens += int(hydrogens.sum())
        aromatic_molecules += int(bool(aromatic.any()))

    data = {
        "smiles": smiles,
        "positions": torch.cat(positions, dim=0),
        "atom_types": torch.cat(atom_types, dim=0),
        "atom_charges": torch.cat(atom_charges, dim=0),
        "bond_types": torch.cat(bond_types, dim=0),
        "bond_idxs": torch.cat(bond_idxs, dim=0),
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
        "radical_molecules_unsupported_by_official_schema": radical_molecules,
        "atoms_with_implicit_hydrogen_not_explicit_nodes": atoms_with_hydrogen,
        "implicit_hydrogens_not_explicit_nodes": total_implicit_hydrogens,
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
    row_sums = p_c_given_a.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0e-8
    p_c_given_a /= row_sums
    return p_a, p_c, p_e, p_c_given_a


def save_split(output: Path, name: str, data: dict[str, Any]) -> None:
    prefix = f"{name}_data"
    torch.save(data, output / f"{prefix}_processed.pt")
    with (output / f"{prefix}_smiles.pkl").open("wb") as handle:
        pickle.dump(data["smiles"], handle)

    lengths = data["node_idx_array"][:, 1] - data["node_idx_array"][:, 0]
    n_atoms, counts = torch.unique(lengths, return_counts=True)
    torch.save((n_atoms, counts), output / f"{prefix}_n_atoms_histogram.pt")

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
    converted: dict[str, dict[str, Any]] = {}
    report: dict[str, Any] = {
        "source": str(source),
        "output": str(output),
        "property": args.property,
        "atom_map": atom_map,
        "source_manifest_sha256": sha256(manifest_path),
        "source_split_sha256": {},
        "splits": {},
        "edge_classes": ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
        "official_code_changes": 0,
    }

    for split in SPLITS:
        split_path = source / f"{split}.pt"
        records = torch.load(split_path, map_location="cpu", weights_only=False)
        data, stats = convert_records(records, atom_map, args.property)
        converted[split] = data
        save_split(output, split, data)
        report["source_split_sha256"][split] = sha256(split_path)
        report["splits"][split] = stats

    # The unmodified official data module only selects train_a or train_b.
    # Store the full training split under both aliases so the baseline uses the
    # exact pre-existing train/val/test split without changing official code.
    save_split(output, "train_a", converted["train"])
    save_split(output, "train_b", converted["train"])

    marginals = marginal_distributions(converted["train"], len(atom_map))
    torch.save(marginals, output / "train_data_marginal_dists.pt")

    report["train_alias_policy"] = "train_a=train_b=train (full existing train split)"
    report["limitations"] = [
        "The unmodified MolGuidance condition embedder accepts one scalar; this export uses only the selected S1 or T1 property.",
        "The source tensors contain heavy atoms only; stored hydrogen counts are not node targets in the official schema.",
        "The official schema has no radical-electron target; affected source molecules are counted in this report.",
    ]
    (output / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
