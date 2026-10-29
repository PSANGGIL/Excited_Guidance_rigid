#!/usr/bin/env python3
"""Compare generated molecules with the HP1KG training distribution.

The script keeps topology, training-support, generated-coordinate quality, and
XTB/STDA condition fidelity as separate concepts.  This is intentional: the
project's ``chemical_clean`` flag only covers a small set of topology warnings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


NUMERIC_METRICS = (
    "heavy_atoms",
    "hetero_atoms",
    "hetero_fraction",
    "molecular_weight",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "fraction_csp3",
    "formal_charge",
    "absolute_formal_charge",
    "explicit_radical_electrons",
    "total_hydrogens",
    "bonds",
    "single_bonds",
    "double_bonds",
    "triple_bonds",
    "aromatic_bonds",
    "cycle_rank",
    "rings",
    "aromatic_rings",
    "ring_atom_fraction",
    "aromatic_atom_fraction",
    "small_rings_3_4",
    "large_rings_ge_9",
    "largest_ring",
    "ring_systems",
    "largest_ring_system",
    "fused_ring_pairs",
    "spiro_ring_pairs",
    "bridgehead_atoms",
    "spiro_atoms",
    "cumulene_centers",
)


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def ring_components(rings: list[set[int]]) -> list[list[int]]:
    adjacency = [set() for _ in rings]
    for i, left in enumerate(rings):
        for j in range(i + 1, len(rings)):
            if left & rings[j]:
                adjacency[i].add(j)
                adjacency[j].add(i)
    components: list[list[int]] = []
    seen: set[int] = set()
    for start in range(len(rings)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for other in adjacency[current]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        components.append(component)
    return components


def molecule_metrics(mol: Chem.Mol) -> dict[str, object]:
    atoms = list(mol.GetAtoms())
    bonds = list(mol.GetBonds())
    heavy = max(mol.GetNumHeavyAtoms(), 1)
    fragments = len(Chem.GetMolFrags(mol))
    rings_tuple = list(mol.GetRingInfo().AtomRings())
    rings = [set(ring) for ring in rings_tuple]
    aromatic_rings = sum(all(atoms[index].GetIsAromatic() for index in ring) for ring in rings)
    ring_atoms = set().union(*rings) if rings else set()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in atoms)
    fused_pairs = 0
    spiro_pairs = 0
    for i, left in enumerate(rings):
        for right in rings[i + 1 :]:
            shared = len(left & right)
            fused_pairs += shared >= 2
            spiro_pairs += shared == 1
    components = ring_components(rings)
    cumulene_centers = sum(
        sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds()) >= 2
        for atom in atoms
    )
    element_counts = Counter(atom.GetSymbol() for atom in atoms if atom.GetAtomicNum() > 1)
    ring_sizes = Counter(len(ring) for ring in rings)
    net_charge = sum(atom.GetFormalCharge() for atom in atoms)
    positive_atoms = sum(atom.GetFormalCharge() > 0 for atom in atoms)
    negative_atoms = sum(atom.GetFormalCharge() < 0 for atom in atoms)
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold = canonical_smiles(scaffold_mol) if scaffold_mol.GetNumAtoms() else ""
    return {
        "canonical_smiles": canonical_smiles(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "hetero_atoms": sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms),
        "hetero_fraction": sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms) / heavy,
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "formal_charge": net_charge,
        "absolute_formal_charge": sum(abs(atom.GetFormalCharge()) for atom in atoms),
        "positive_atoms": positive_atoms,
        "negative_atoms": negative_atoms,
        "zwitterionic": int(positive_atoms > 0 and negative_atoms > 0 and net_charge == 0),
        "charged": int(any(atom.GetFormalCharge() for atom in atoms)),
        "explicit_radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in atoms),
        "total_hydrogens": sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in atoms),
        "fragments": fragments,
        "bonds": len(bonds),
        "single_bonds": sum(bond.GetBondType() == Chem.BondType.SINGLE for bond in bonds),
        "double_bonds": sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds),
        "triple_bonds": sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds),
        "aromatic_bonds": sum(bond.GetIsAromatic() for bond in bonds),
        "cycle_rank": len(bonds) - len(atoms) + fragments,
        "rings": len(rings),
        "aromatic_rings": aromatic_rings,
        "ring_atom_fraction": len(ring_atoms) / heavy,
        "aromatic_atom_fraction": aromatic_atoms / heavy,
        "small_rings_3_4": sum(len(ring) in (3, 4) for ring in rings),
        "large_rings_ge_9": sum(len(ring) >= 9 for ring in rings),
        "largest_ring": max((len(ring) for ring in rings), default=0),
        "ring_systems": len(components),
        "largest_ring_system": max((len(component) for component in components), default=0),
        "fused_ring_pairs": fused_pairs,
        "spiro_ring_pairs": spiro_pairs,
        "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "cumulene_centers": cumulene_centers,
        "scaffold": scaffold,
        "ring_sizes": dict(sorted(ring_sizes.items())),
        "element_counts": dict(sorted(element_counts.items())),
    }


def describe(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for key in NUMERIC_METRICS:
        values = np.asarray([finite_float(row.get(key)) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        output[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return output


def warning_counts(rows: list[dict[str, str]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = str(row.get(field, "")).strip()
        if not raw or raw.lower() in {"nan", "none", "[]"}:
            continue
        for token in re.split(r"[|;,]+", raw.strip("[]'\" ")):
            token = token.strip(" '[]\"")
            if token:
                counts[token] += 1
    return counts


def distribution_counts(rows: list[dict[str, object]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get(field, {})
        if isinstance(values, dict):
            counts.update({str(key): int(value) for key, value in values.items()})
    return counts


def incidence_counts(rows: list[dict[str, object]], field: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        values = row.get(field, {})
        if isinstance(values, dict):
            counts.update(str(key) for key, value in values.items() if int(value) > 0)
    return counts


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def make_plot(
    output: Path,
    generated: list[dict[str, object]],
    train: list[dict[str, object]],
    near_zero: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = (("generated", generated), ("train", train), ("train |gap|<=0.05", near_zero))
    colors = ("#d62728", "#4c78a8", "#2ca02c")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    panels = (
        ("heavy_atoms", np.arange(0.5, 31.5, 1), "Heavy atoms"),
        ("rings", np.arange(-0.5, 9.5, 1), "SSSR rings"),
        ("aromatic_atom_fraction", np.linspace(0, 1, 16), "Aromatic atom fraction"),
        ("hetero_fraction", np.linspace(0, 1, 16), "Hetero-atom fraction"),
    )
    for axis, (field, bins, title) in zip(axes.flat[:4], panels):
        for (label, rows), color in zip(groups, colors):
            axis.hist(
                [float(row[field]) for row in rows],
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=label,
                color=color,
            )
        axis.set_title(title)
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    ring_axis = axes.flat[4]
    width = 0.25
    ring_sizes = np.arange(3, 13)
    for offset, ((label, rows), color) in enumerate(zip(groups, colors)):
        counts = distribution_counts(rows, "ring_sizes")
        total = max(sum(counts.values()), 1)
        ring_axis.bar(
            ring_sizes + (offset - 1) * width,
            [counts[str(size)] / total for size in ring_sizes],
            width=width,
            label=label,
            color=color,
            alpha=0.8,
        )
    ring_axis.set_title("Ring-size share")
    ring_axis.set_xticks(ring_sizes)
    ring_axis.set_ylabel("fraction of all SSSR rings")
    ring_axis.grid(axis="y", alpha=0.2)
    element_axis = axes.flat[5]
    element_order = sorted(
        set(incidence_counts(train, "element_counts")) | set(incidence_counts(generated, "element_counts")),
        key=lambda element: (-incidence_counts(train, "element_counts")[element], element),
    )[:14]
    x = np.arange(len(element_order))
    for offset, ((label, rows), color) in enumerate(zip(groups, colors)):
        counts = incidence_counts(rows, "element_counts")
        element_axis.bar(
            x + (offset - 1) * width,
            [counts[element] / max(len(rows), 1) for element in element_order],
            width=width,
            label=label,
            color=color,
            alpha=0.8,
        )
    element_axis.set_xticks(x, element_order)
    element_axis.set_title("Element incidence per molecule")
    element_axis.set_ylabel("molecule fraction")
    element_axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat:
        axis.legend(fontsize=8)
    fig.suptitle("Generated structures vs HP1KG training molecules", fontsize=15)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def draw_candidates(path: Path, rows: list[dict[str, object]], mols: list[Chem.Mol]) -> None:
    if not rows:
        return
    legends = []
    selected = []
    for row in rows:
        selected.append(mols[int(row["generated_order"])])
        legends.append(
            f"{row['generated_name']} | XTB S1/T1 {float(row['xtb_s1_eV']):.2f}/{float(row['xtb_t1_eV']):.2f}\n"
            f"gap {float(row['xtb_delta_est_eV']):+.3f} | support {int(truthy(row['support_pass']))}"
        )
    image = Draw.MolsToGridImage(selected, molsPerRow=4, subImgSize=(420, 300), legends=legends)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--db-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    generation_dir = args.generation_dir.resolve()
    output_dir = (args.output_dir or generation_dir / "structure_analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_rows = read_csv(generation_dir / "molecules.csv")
    generation_by_index = {int(row["index"]): row for row in generation_rows}
    xtb_rows = read_csv(generation_dir / "xtb_stda_results_226.csv")
    supplier = Chem.SDMolSupplier(str(generation_dir / "unique_novel_connected.sdf"), removeHs=True)
    generated_mols = [mol for mol in supplier if mol is not None]
    if len(generated_mols) != len(xtb_rows):
        raise RuntimeError(f"SDF/XTB row mismatch: {len(generated_mols)} vs {len(xtb_rows)}")

    generated_metrics: list[dict[str, object]] = []
    for order, (mol, xtb) in enumerate(zip(generated_mols, xtb_rows)):
        source_index = int(xtb["source_generation_index"])
        source = generation_by_index[source_index]
        metrics = molecule_metrics(mol)
        if metrics["canonical_smiles"] != canonical_smiles(Chem.MolFromSmiles(xtb["smiles"])):
            raise RuntimeError(f"SMILES/SDF mismatch at generated order {order}")
        metrics.update(
            {
                "generated_order": order,
                "generated_name": xtb["generated_name"],
                "source_generation_index": source_index,
                "smiles": xtb["smiles"],
                "chemical_clean": int(truthy(source["chemical_clean"])),
                "chemical_warnings": source["chemical_warnings"],
                "support_pass": int(truthy(source["support_pass"])),
                "support_warnings": source["support_warnings"],
                "generated_bond_z_rms": finite_float(source["generated_bond_z_rms"]),
                "generated_bond_outliers": finite_float(source["generated_bond_outliers"]),
                "generated_steric_clashes": finite_float(source["generated_steric_clashes"]),
                "generated_geometry_warnings": source["generated_geometry_warnings"],
                "topology_prior_score": finite_float(source["topology_prior_score"]),
                "requested_s1_eV": finite_float(xtb["requested_s1_eV"]),
                "requested_t1_eV": finite_float(xtb["requested_t1_eV"]),
                "requested_delta_est_eV": finite_float(xtb["requested_delta_est_eV"]),
                "xtb_s1_eV": finite_float(xtb["xtb_s1_eV"]),
                "xtb_t1_eV": finite_float(xtb["xtb_t1_eV"]),
                "xtb_delta_est_eV": finite_float(xtb["xtb_delta_est_eV"]),
                "s1_error_eV": finite_float(xtb["s1_error_eV"]),
                "t1_error_eV": finite_float(xtb["t1_error_eV"]),
                "delta_est_error_eV": finite_float(xtb["delta_est_error_eV"]),
                "s1_f_osc": finite_float(xtb["s1_f_osc"]),
                "conformer_retry": int(xtb["conformer_retry"]),
            }
        )
        generated_metrics.append(metrics)

    db_rows = read_csv(args.db_results.resolve())
    train_rows: list[dict[str, object]] = []
    invalid_db = 0
    for row in db_rows:
        if row.get("split") != "train":
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            invalid_db += 1
            continue
        metrics = molecule_metrics(mol)
        metrics.update(
            {
                "database_id": row["database_id"],
                "smiles": row["smiles"],
                "reference_s1_eV": finite_float(row["reference_s1_eV"]),
                "reference_t1_eV": finite_float(row["reference_t1_eV"]),
                "reference_delta_est_eV": finite_float(row["reference_s1_eV"])
                - finite_float(row["reference_t1_eV"]),
            }
        )
        train_rows.append(metrics)
    near_zero = [row for row in train_rows if abs(float(row["reference_delta_est_eV"])) <= 0.05]

    train_scaffolds = {str(row["scaffold"]) for row in train_rows if row["scaffold"]}
    for row in generated_metrics:
        row["scaffold_novel_vs_train"] = int(bool(row["scaffold"]) and row["scaffold"] not in train_scaffolds)

    clean_gap = [
        row
        for row in generated_metrics
        if truthy(row["chemical_clean"]) and abs(float(row["xtb_delta_est_eV"])) <= 0.05
    ]
    strict_condition = [
        row
        for row in clean_gap
        if abs(float(row["s1_error_eV"])) <= 1.0 and abs(float(row["t1_error_eV"])) <= 1.0
    ]
    supported_clean_gap = [row for row in clean_gap if truthy(row["support_pass"])]

    generated_scaffolds = Counter(str(row["scaffold"]) for row in generated_metrics if row["scaffold"])
    summary = {
        "counts": {
            "generated": len(generated_metrics),
            "train": len(train_rows),
            "train_near_zero_gap_le_0p05": len(near_zero),
            "invalid_train_smiles": invalid_db,
            "chemical_clean": sum(truthy(row["chemical_clean"]) for row in generated_metrics),
            "support_pass": sum(truthy(row["support_pass"]) for row in generated_metrics),
            "clean_and_support_pass": sum(
                truthy(row["chemical_clean"]) and truthy(row["support_pass"]) for row in generated_metrics
            ),
            "xtb_gap_abs_le_0p05": sum(
                abs(float(row["xtb_delta_est_eV"])) <= 0.05 for row in generated_metrics
            ),
            "clean_xtb_gap_abs_le_0p05": len(clean_gap),
            "clean_supported_xtb_gap_abs_le_0p05": len(supported_clean_gap),
            "clean_gap_and_s1_t1_errors_abs_le_1": len(strict_condition),
            "charged": sum(int(row["charged"]) for row in generated_metrics),
            "zwitterionic": sum(int(row["zwitterionic"]) for row in generated_metrics),
            "radical": sum(float(row["explicit_radical_electrons"]) > 0 for row in generated_metrics),
            "small_ring_3_4": sum(float(row["small_rings_3_4"]) > 0 for row in generated_metrics),
            "large_ring_ge_9": sum(float(row["large_rings_ge_9"]) > 0 for row in generated_metrics),
            "bridgehead": sum(float(row["bridgehead_atoms"]) > 0 for row in generated_metrics),
            "spiro": sum(float(row["spiro_atoms"]) > 0 for row in generated_metrics),
            "cumulene": sum(float(row["cumulene_centers"]) > 0 for row in generated_metrics),
            "scaffold_bearing": sum(bool(row["scaffold"]) for row in generated_metrics),
            "unique_scaffolds": len(generated_scaffolds),
            "novel_scaffolds_vs_train": sum(int(row["scaffold_novel_vs_train"]) for row in generated_metrics),
        },
        "generated": describe(generated_metrics),
        "train": describe(train_rows),
        "train_near_zero_gap_le_0p05": describe(near_zero),
        "element_atom_counts": {
            "generated": dict(distribution_counts(generated_metrics, "element_counts")),
            "train": dict(distribution_counts(train_rows, "element_counts")),
            "train_near_zero": dict(distribution_counts(near_zero, "element_counts")),
        },
        "element_molecule_incidence": {
            "generated": dict(incidence_counts(generated_metrics, "element_counts")),
            "train": dict(incidence_counts(train_rows, "element_counts")),
            "train_near_zero": dict(incidence_counts(near_zero, "element_counts")),
        },
        "ring_size_counts": {
            "generated": dict(distribution_counts(generated_metrics, "ring_sizes")),
            "train": dict(distribution_counts(train_rows, "ring_sizes")),
            "train_near_zero": dict(distribution_counts(near_zero, "ring_sizes")),
        },
        "warnings": {
            "chemical": dict(warning_counts(generation_rows, "chemical_warnings")),
            "support": dict(warning_counts(generation_rows, "support_warnings")),
            "generated_geometry": dict(warning_counts(generation_rows, "generated_geometry_warnings")),
        },
        "scaffolds": {
            "most_common_generated": generated_scaffolds.most_common(20),
        },
    }

    write_csv(output_dir / "generated_structure_metrics.csv", generated_metrics)
    write_csv(output_dir / "clean_xtb_gap_le_0p05.csv", clean_gap)
    write_csv(output_dir / "clean_supported_xtb_gap_le_0p05.csv", supported_clean_gap)
    write_csv(output_dir / "clean_gap_and_s1_t1_errors_le_1eV.csv", strict_condition)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    make_plot(output_dir / "structure_distributions.png", generated_metrics, train_rows, near_zero)
    draw_candidates(output_dir / "clean_xtb_gap_le_0p05.png", clean_gap, generated_mols)

    print(json.dumps(summary["counts"], indent=2, sort_keys=True))
    print(f"outputs={output_dir}")


if __name__ == "__main__":
    main()
