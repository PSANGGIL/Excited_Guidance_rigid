#!/usr/bin/env python3
"""Detailed, reproducible structural and XTB/sTDA analysis of a generation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdDepictor, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def safe_int(value: object, default: int = -1) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def bond_label(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "AROMATIC"
    return str(bond.GetBondType()).upper()


def atom_h_count(atom: Chem.Atom) -> int:
    if atom.GetAtomicNum() == 1:
        return 0
    return int(atom.GetTotalNumHs(includeNeighbors=True))


def atom_signature(atom: Chem.Atom) -> str:
    neighbors = []
    for bond in atom.GetBonds():
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() == 1:
            continue
        neighbors.append(
            (
                other.GetSymbol(),
                bond_label(bond),
                int(other.GetIsAromatic()),
                other.GetFormalCharge(),
            )
        )
    neighbors.sort()
    center = (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetIsAromatic()),
        atom_h_count(atom),
        atom.GetDegree(),
    )
    return repr((center, tuple(neighbors)))


def bond_signature(bond: Chem.Bond) -> str:
    def endpoint(atom: Chem.Atom) -> tuple[Any, ...]:
        return (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom_h_count(atom),
            atom.GetDegree(),
        )

    endpoints = tuple(sorted((endpoint(bond.GetBeginAtom()), endpoint(bond.GetEndAtom()))))
    return repr((endpoints, bond_label(bond), int(bond.IsInRing())))


def ring_system_stats(rings: list[tuple[int, ...]]) -> tuple[int, int, int, int]:
    if not rings:
        return 0, 0, 0, 0
    parents = list(range(len(rings)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    fused_pairs = 0
    spiro_pairs = 0
    ring_sets = [set(ring) for ring in rings]
    for left in range(len(rings)):
        for right in range(left + 1, len(rings)):
            overlap = len(ring_sets[left] & ring_sets[right])
            if overlap:
                union(left, right)
            if overlap >= 2:
                fused_pairs += 1
            elif overlap == 1:
                spiro_pairs += 1
    component_sizes = Counter(find(index) for index in range(len(rings)))
    return len(component_sizes), max(component_sizes.values()), fused_pairs, spiro_pairs


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return ""
    return canonical_smiles(scaffold)


def base_properties(mol: Chem.Mol) -> dict[str, Any]:
    mol = Chem.RemoveHs(Chem.Mol(mol))
    Chem.SanitizeMol(mol)
    rings = [tuple(int(index) for index in ring) for ring in mol.GetRingInfo().AtomRings()]
    ring_sizes = [len(ring) for ring in rings]
    ring_systems, max_system, fused_pairs, spiro_pairs = ring_system_stats(rings)
    aromatic_rings = sum(
        all(mol.GetAtomWithIdx(index).GetIsAromatic() for index in ring)
        for ring in rings
    )
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
    elements = Counter(atom.GetSymbol() for atom in atoms)
    bonds = Counter(bond_label(bond) for bond in mol.GetBonds())
    h_counts = Counter(min(atom_h_count(atom), 3) for atom in atoms)
    charge = sum(atom.GetFormalCharge() for atom in atoms)
    radicals = sum(atom.GetNumRadicalElectrons() for atom in atoms)
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in atoms)
    heavy_atoms = len(atoms)
    hetero_atoms = sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
    components = len(Chem.GetMolFrags(mol))
    cycle_rank = mol.GetNumBonds() - mol.GetNumAtoms() + components
    has_small = any(size <= 4 for size in ring_sizes)
    has_large = any(size >= 9 for size in ring_sizes)
    closed_shell_neutral = charge == 0 and radicals == 0
    kekule_ok = True
    try:
        candidate = Chem.Mol(mol)
        Chem.Kekulize(candidate, clearAromaticFlags=False)
    except Exception:
        kekule_ok = False
    return {
        "smiles": canonical_smiles(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atoms": heavy_atoms,
        "total_hydrogens": sum(atom_h_count(atom) for atom in atoms),
        "attached_h_0": h_counts[0],
        "attached_h_1": h_counts[1],
        "attached_h_2": h_counts[2],
        "attached_h_3plus": h_counts[3],
        "bonds": mol.GetNumBonds(),
        "cycle_rank": cycle_rank,
        "num_rings": len(rings),
        "ring_sizes": ring_sizes,
        "small_ring_count": sum(size <= 4 for size in ring_sizes),
        "normal_ring_count": sum(size in (5, 6) for size in ring_sizes),
        "large_ring_count": sum(size >= 9 for size in ring_sizes),
        "aromatic_rings": aromatic_rings,
        "aromatic_atoms": aromatic_atoms,
        "aromatic_atom_fraction": aromatic_atoms / heavy_atoms if heavy_atoms else 0.0,
        "ring_systems": ring_systems,
        "max_ring_system_rings": max_system,
        "fused_ring_pairs": fused_pairs,
        "spiro_ring_pairs": spiro_pairs,
        "bridgehead_atoms": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "spiro_atoms": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "hetero_atoms": hetero_atoms,
        "hetero_fraction": hetero_atoms / heavy_atoms if heavy_atoms else 0.0,
        "formal_charge": charge,
        "radical_electrons": radicals,
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "mol_wt": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "elements_counter": elements,
        "bond_counter": bonds,
        "atom_signatures": [atom_signature(atom) for atom in atoms],
        "bond_signatures": [bond_signature(bond) for bond in mol.GetBonds()],
        "ringless": not rings,
        "only_5_6_rings": bool(rings) and all(size in (5, 6) for size in ring_sizes),
        "has_small_ring": has_small,
        "has_large_ring": has_large,
        "has_aromatic_ring": aromatic_rings > 0,
        "polyaromatic": aromatic_rings >= 2,
        "closed_shell_neutral": closed_shell_neutral,
        "topology_clean": closed_shell_neutral and not has_small and not has_large,
        "kekule_ok": kekule_ok,
        "scaffold": scaffold_smiles(mol),
        "mol": mol,
    }


def numeric_summary(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def js_divergence(left: Counter[Any], right: Counter[Any]) -> float | None:
    keys = sorted(set(left) | set(right), key=str)
    left_total, right_total = sum(left.values()), sum(right.values())
    if not keys or not left_total or not right_total:
        return None
    p = np.asarray([left[key] / left_total for key in keys], dtype=np.float64)
    q = np.asarray([right[key] / right_total for key in keys], dtype=np.float64)
    midpoint = 0.5 * (p + q)

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return 0.5 * kl(p) + 0.5 * kl(q)


def category_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    predicates: dict[str, Callable[[dict[str, Any]], bool]] = {
        "ringless": lambda row: row["ringless"],
        "only_5_6_rings": lambda row: row["only_5_6_rings"],
        "has_3_4_ring": lambda row: row["has_small_ring"],
        "has_9plus_ring": lambda row: row["has_large_ring"],
        "has_aromatic_ring": lambda row: row["has_aromatic_ring"],
        "polyaromatic_2plus": lambda row: row["polyaromatic"],
        "fused_ring_pair": lambda row: row["fused_ring_pairs"] > 0,
        "spiro_atom": lambda row: row["spiro_atoms"] > 0,
        "bridgehead_atom": lambda row: row["bridgehead_atoms"] > 0,
        "charged": lambda row: row["formal_charge"] != 0,
        "radical": lambda row: row["radical_electrons"] != 0,
        "closed_shell_neutral": lambda row: row["closed_shell_neutral"],
        "topology_clean": lambda row: row["topology_clean"],
        "kekule_ok": lambda row: row["kekule_ok"],
        "all_local_env_supported": lambda row: row.get("all_local_environments_supported", False),
        "scaffold_seen_in_db": lambda row: row.get("scaffold_in_db", False),
        "nearest_db_similarity_ge_0p5": lambda row: row.get("max_db_tanimoto", 0.0) >= 0.5,
    }
    denominator = len(rows)
    return {
        name: {
            "count": sum(bool(predicate(row)) for row in rows),
            "fraction": (
                sum(bool(predicate(row)) for row in rows) / denominator
                if denominator else 0.0
            ),
        }
        for name, predicate in predicates.items()
    }


def est_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("xtb_status") == "ok"]
    if not selected:
        return {"n": 0}
    values = np.asarray([row["est_eV"] for row in selected], dtype=np.float64)
    oscillators = np.asarray([row["s1_f_osc"] for row in selected], dtype=np.float64)
    return {
        "n": len(selected),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "mae_to_0": float(np.abs(values).mean()),
        "abs_le_0p05": int(np.sum(np.abs(values) <= 0.05)),
        "abs_le_0p10": int(np.sum(np.abs(values) <= 0.10)),
        "zero_to_0p5": int(np.sum((values >= 0.0) & (values <= 0.5))),
        "negative": int(np.sum(values < 0.0)),
        "median_f_osc": float(np.median(oscillators)),
    }


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(left: list[float], right: list[float], ranked: bool = False) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if ranked:
        x, y = rank_values(x), rank_values(y)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def draw_panel(path: Path, rows: list[dict[str, Any]], limit: int = 40) -> str | None:
    selected = rows[:limit]
    molecules: list[Chem.Mol] = []
    legends: list[str] = []
    for row in selected:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue
        rdDepictor.Compute2DCoords(mol)
        est = row.get("est_eV")
        est_label = "NA" if est is None else f"{float(est):.3f}"
        rings = row.get("ring_sizes_text", "-") or "-"
        legends.append(
            f"idx={row['generation_index']} EST={est_label} sim={row.get('max_db_tanimoto', 0):.2f}\n"
            f"rings={rings} q={row['formal_charge']} rad={row['radical_electrons']}"
        )
        molecules.append(mol)
    if not molecules:
        return None
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=5,
        subImgSize=(330, 270),
        legends=legends,
        useSVG=False,
    )
    image.save(path)
    return str(path.resolve())


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    excluded = {"mol", "elements_counter", "bond_counter", "atom_signatures", "bond_signatures"}
    fieldnames = [key for key in rows[0] if key not in excluded]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key, value in list(output.items()):
                if isinstance(value, list):
                    output[key] = ";".join(map(str, value))
            writer.writerow({key: output.get(key, "") for key in fieldnames})


def failure_class(error: str) -> str:
    if "open-shell" in error:
        return "open_shell"
    if "net-charged" in error:
        return "net_charged"
    if "ETKDG" in error:
        return "etkdg_embedding"
    if "RDKit read" in error:
        return "rdkit_read"
    return error or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generation_summary = json.loads((args.run_dir / "analysis/summary.json").read_text())
    all_generation_rows = read_csv(args.run_dir / "analysis/molecules.csv")
    condition_rows = read_csv(args.run_dir / "condition_manifest.csv")
    strict_rows = read_csv(args.run_dir / "xtb/inputs/manifest.csv")
    xtb_rows = read_csv(args.run_dir / "xtb/work/results.csv")

    database_by_smiles: dict[str, Chem.Mol] = {}
    for split in ("train", "val", "test"):
        with (args.db_dir / f"{split}_data_smiles.pkl").open("rb") as handle:
            for smiles in pickle.load(handle):
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    database_by_smiles.setdefault(canonical_smiles(mol), mol)
    database_rows = [base_properties(mol) for mol in database_by_smiles.values()]
    db_atom_signatures = {
        signature for row in database_rows for signature in row["atom_signatures"]
    }
    db_bond_signatures = {
        signature for row in database_rows for signature in row["bond_signatures"]
    }
    db_scaffolds = {row["scaffold"] for row in database_rows if row["scaffold"]}
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    db_fingerprints = [fingerprint_generator.GetFingerprint(row["mol"]) for row in database_rows]
    db_smiles_ordered = [row["smiles"] for row in database_rows]

    generated_rows: list[dict[str, Any]] = []
    supplier = Chem.SDMolSupplier(
        str(args.run_dir / "analysis/unique_novel_connected.sdf"),
        sanitize=False,
        removeHs=False,
    )
    unreadable = 0
    for source_index, raw in enumerate(supplier):
        if raw is None:
            unreadable += 1
            continue
        generation_index = (
            raw.GetIntProp("generation_index")
            if raw.HasProp("generation_index") else source_index
        )
        smiles = (
            raw.GetProp("canonical_smiles")
            if raw.HasProp("canonical_smiles") else ""
        )
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            try:
                Chem.SanitizeMol(raw)
                mol = Chem.Mol(raw)
            except Exception:
                unreadable += 1
                continue
        row = base_properties(mol)
        row.update(source_index=source_index, generation_index=generation_index)
        atom_unseen = [sig for sig in row["atom_signatures"] if sig not in db_atom_signatures]
        bond_unseen = [sig for sig in row["bond_signatures"] if sig not in db_bond_signatures]
        row["unseen_atom_instances"] = len(atom_unseen)
        row["unseen_atom_signature_count"] = len(set(atom_unseen))
        row["unseen_bond_instances"] = len(bond_unseen)
        row["unseen_bond_signature_count"] = len(set(bond_unseen))
        row["all_local_environments_supported"] = not atom_unseen and not bond_unseen
        row["scaffold_in_db"] = bool(row["scaffold"]) and row["scaffold"] in db_scaffolds
        fingerprint = fingerprint_generator.GetFingerprint(row["mol"])
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, db_fingerprints)
        nearest = int(np.argmax(similarities))
        row["max_db_tanimoto"] = float(similarities[nearest])
        row["nearest_db_smiles"] = db_smiles_ordered[nearest]
        row["elements"] = ";".join(
            f"{element}:{count}" for element, count in sorted(row["elements_counter"].items())
        )
        row["bond_types"] = ";".join(
            f"{kind}:{count}" for kind, count in sorted(row["bond_counter"].items())
        )
        row["ring_sizes_text"] = ";".join(map(str, row["ring_sizes"]))
        generated_rows.append(row)

    generated_by_index = {row["generation_index"]: row for row in generated_rows}
    strict_by_index: dict[int, dict[str, str]] = {}
    for strict in strict_rows:
        source_index = safe_int(strict.get("source_index"))
        generation_index = safe_int(strict.get("generation_index"))
        if generation_index < 0 and 0 <= source_index < len(generated_rows):
            generation_index = generated_rows[source_index]["generation_index"]
        if generation_index >= 0:
            strict_by_index[generation_index] = strict
    xtb_by_index = {
        safe_int(row.get("generation_index")): row
        for row in xtb_rows
        if safe_int(row.get("generation_index")) >= 0
    }
    for row in generated_rows:
        strict = strict_by_index.get(row["generation_index"], {})
        xtb = xtb_by_index.get(row["generation_index"], {})
        row["strict_prep_status"] = strict.get("status", "not_recorded")
        row["strict_prep_error"] = strict.get("error", "")
        row["xtb_status"] = xtb.get("status", "not_attempted")
        if xtb.get("status") == "ok":
            row["s1_eV"] = float(xtb["s1_energy_eV"])
            row["t1_eV"] = float(xtb["t1_energy_eV"])
            row["est_eV"] = row["s1_eV"] - row["t1_eV"]
            row["s1_f_osc"] = float(xtb["s1_f_osc"])
        else:
            row.update(s1_eV=None, t1_eV=None, est_eV=None, s1_f_osc=None)
        row["clean_and_supported"] = (
            row["topology_clean"] and row["all_local_environments_supported"]
        )

    generated_categories = category_counts(generated_rows)
    database_categories = category_counts(database_rows)
    distribution_fields = [
        "heavy_atoms", "total_hydrogens", "num_rings", "aromatic_rings",
        "aromatic_atom_fraction", "cycle_rank", "hetero_atoms", "hetero_fraction",
        "rotatable_bonds", "fraction_csp3", "bridgehead_atoms", "spiro_atoms",
        "mol_wt", "logp", "tpsa",
    ]
    distribution_comparison = {
        field: {
            "generated": numeric_summary([float(row[field]) for row in generated_rows]),
            "database": numeric_summary([float(row[field]) for row in database_rows]),
        }
        for field in distribution_fields
    }
    distribution_comparison["max_db_tanimoto"] = {
        "generated": numeric_summary([row["max_db_tanimoto"] for row in generated_rows]),
        "database": None,
    }

    generated_ring_sizes = Counter(size for row in generated_rows for size in row["ring_sizes"])
    database_ring_sizes = Counter(size for row in database_rows for size in row["ring_sizes"])
    generated_ring_counts = Counter(row["num_rings"] for row in generated_rows)
    database_ring_counts = Counter(row["num_rings"] for row in database_rows)
    generated_heavy_counts = Counter(row["heavy_atoms"] for row in generated_rows)
    database_heavy_counts = Counter(row["heavy_atoms"] for row in database_rows)
    generated_h_states = Counter()
    database_h_states = Counter()
    for row in generated_rows:
        for state in range(4):
            generated_h_states[state] += row[f"attached_h_{state if state < 3 else '3plus'}"]
    for row in database_rows:
        for state in range(4):
            database_h_states[state] += row[f"attached_h_{state if state < 3 else '3plus'}"]

    generated_element_counts = Counter()
    database_element_counts = Counter()
    for row in generated_rows:
        generated_element_counts.update(row["elements_counter"])
    for row in database_rows:
        database_element_counts.update(row["elements_counter"])

    est_rows = [row for row in generated_rows if row["xtb_status"] == "ok"]
    est_groups = {
        "all": est_group(est_rows),
        "topology_clean": est_group([row for row in est_rows if row["topology_clean"]]),
        "all_local_env_supported": est_group(
            [row for row in est_rows if row["all_local_environments_supported"]]
        ),
        "clean_and_supported": est_group(
            [row for row in est_rows if row["clean_and_supported"]]
        ),
        "has_3_4_ring": est_group([row for row in est_rows if row["has_small_ring"]]),
        "has_9plus_ring": est_group([row for row in est_rows if row["has_large_ring"]]),
        "has_aromatic_ring": est_group([row for row in est_rows if row["has_aromatic_ring"]]),
        "polyaromatic": est_group([row for row in est_rows if row["polyaromatic"]]),
        "ringless": est_group([row for row in est_rows if row["ringless"]]),
    }
    for element in sorted(generated_element_counts):
        est_groups[f"contains_{element}"] = est_group(
            [row for row in est_rows if row["elements_counter"].get(element, 0)]
        )

    correlation_fields = [
        "heavy_atoms", "total_hydrogens", "num_rings", "aromatic_rings",
        "aromatic_atom_fraction", "cycle_rank", "hetero_atoms", "fraction_csp3",
        "max_db_tanimoto", "s1_f_osc",
    ]
    est_correlations = {}
    est_values = [row["est_eV"] for row in est_rows]
    for field in correlation_fields:
        values = [float(row[field]) for row in est_rows]
        est_correlations[field] = {
            "pearson": correlation(values, est_values),
            "spearman": correlation(values, est_values, ranked=True),
        }

    invalid_reasons = Counter(
        row.get("invalid_reason", "unknown") or "unknown"
        for row in all_generation_rows if not truthy(row.get("valid"))
    )
    strict_failures = Counter(
        failure_class(row.get("error", ""))
        for row in strict_rows if row.get("status") != "ok"
    )
    strict_failure_details = Counter(
        row.get("error", "") or "unknown"
        for row in strict_rows if row.get("status") != "ok"
    )

    condition_by_index = {
        safe_int(row["generation_index"]): row for row in condition_rows
    }
    valid_indices = {
        safe_int(row.get("generation_index"))
        for row in all_generation_rows if truthy(row.get("valid"))
    }
    connected_indices = {
        safe_int(row.get("generation_index"))
        for row in all_generation_rows if truthy(row.get("connected"))
    }
    novel_indices = set(generated_by_index)
    prep_indices = {
        index for index, row in strict_by_index.items() if row.get("status") == "ok"
    }
    xtb_indices = set(xtb_by_index)
    size_bins = [
        ("6-15", 6, 15), ("16-25", 16, 25), ("26-35", 26, 35),
        ("36-45", 36, 45), ("46+", 46, 10**9),
    ]
    size_stage_rates = {}
    for label, lower, upper in size_bins:
        indices = {
            index for index, row in condition_by_index.items()
            if lower <= safe_int(row.get("requested_atom_count")) <= upper
        }
        requested = len(indices)
        size_stage_rates[label] = {
            "requested": requested,
            "rdkit_valid": len(indices & valid_indices),
            "connected": len(indices & connected_indices),
            "novel_unique_connected": len(indices & novel_indices),
            "strict_prep_ok": len(indices & prep_indices),
            "xtb_stda_ok": len(indices & xtb_indices),
            "rdkit_valid_rate": len(indices & valid_indices) / requested if requested else None,
            "xtb_stda_rate": len(indices & xtb_indices) / requested if requested else None,
        }

    ranked_est = sorted(est_rows, key=lambda row: (abs(row["est_eV"]), row["generation_index"]))
    clean_ranked = [row for row in ranked_est if row["topology_clean"]]
    supported_ranked = [row for row in ranked_est if row["clean_and_supported"]]
    low_clean = [row for row in supported_ranked if abs(row["est_eV"]) <= 0.10]
    write_rows(args.out_dir / "molecule_structure_details.csv", generated_rows)
    write_rows(args.out_dir / "ranked_low_est_supported_candidates.csv", supported_ranked)
    write_rows(args.out_dir / "low_est_clean_supported_le_0p10.csv", low_clean)

    panels = {
        "lowest_abs_est": draw_panel(args.out_dir / "lowest_abs_est.png", ranked_est),
        "lowest_abs_est_topology_clean": draw_panel(
            args.out_dir / "lowest_abs_est_topology_clean.png", clean_ranked
        ),
        "lowest_abs_est_clean_supported": draw_panel(
            args.out_dir / "lowest_abs_est_clean_supported.png", supported_ranked
        ),
        "small_3_4_member_rings": draw_panel(
            args.out_dir / "small_3_4_member_rings.png",
            sorted(
                [row for row in generated_rows if row["has_small_ring"]],
                key=lambda row: (-row["small_ring_count"], row["generation_index"]),
            ),
        ),
        "large_9plus_member_rings": draw_panel(
            args.out_dir / "large_9plus_member_rings.png",
            sorted(
                [row for row in generated_rows if row["has_large_ring"]],
                key=lambda row: (-max(row["ring_sizes"]), row["generation_index"]),
            ),
        ),
        "polyaromatic": draw_panel(
            args.out_dir / "polyaromatic.png",
            sorted(
                [row for row in generated_rows if row["polyaromatic"]],
                key=lambda row: (-row["aromatic_rings"], row["generation_index"]),
            ),
        ),
        "radical_or_charged_rejected": draw_panel(
            args.out_dir / "radical_or_charged_rejected.png",
            [
                row for row in generated_rows
                if row["radical_electrons"] or row["formal_charge"]
            ],
        ),
        "lowest_db_similarity": draw_panel(
            args.out_dir / "lowest_db_similarity.png",
            sorted(generated_rows, key=lambda row: row["max_db_tanimoto"]),
        ),
    }

    reference = None
    if args.reference_summary and args.reference_summary.exists():
        reference = json.loads(args.reference_summary.read_text())

    total_generated_atoms = sum(generated_element_counts.values())
    total_database_atoms = sum(database_element_counts.values())
    summary: dict[str, Any] = {
        "scope": {
            "requested": int(generation_summary.get("requested", 0)),
            "rdkit_valid": int(generation_summary.get("valid", 0)),
            "connected": int(generation_summary.get("connected", 0)),
            "novel_unique_connected": int(generation_summary.get("novel_unique_connected", 0)),
            "structurally_analyzed": len(generated_rows),
            "unreadable_novel_sdf_records": unreadable,
            "strict_prep_ok": sum(row.get("status") == "ok" for row in strict_rows),
            "xtb_stda_ok": len(est_rows),
            "database_molecules": len(database_rows),
        },
        "invalid_reason_counts": dict(invalid_reasons),
        "strict_preparation_failure_classes": dict(strict_failures),
        "strict_preparation_failure_details": dict(strict_failure_details),
        "generated_category_counts": generated_categories,
        "database_category_counts": database_categories,
        "distribution_comparison": distribution_comparison,
        "ring_size_counts": {
            "generated": {str(key): value for key, value in sorted(generated_ring_sizes.items())},
            "database": {str(key): value for key, value in sorted(database_ring_sizes.items())},
        },
        "distribution_jsd_bits": {
            "ring_size": js_divergence(generated_ring_sizes, database_ring_sizes),
            "ring_count": js_divergence(generated_ring_counts, database_ring_counts),
            "heavy_atom_count": js_divergence(generated_heavy_counts, database_heavy_counts),
            "attached_h_state": js_divergence(generated_h_states, database_h_states),
            "element_atom_fraction": js_divergence(generated_element_counts, database_element_counts),
        },
        "attached_h_state_counts": {
            "generated": {str(key): value for key, value in sorted(generated_h_states.items())},
            "database": {str(key): value for key, value in sorted(database_h_states.items())},
        },
        "element_atom_fractions": {
            "generated": {
                key: value / total_generated_atoms for key, value in sorted(generated_element_counts.items())
            },
            "database": {
                key: value / total_database_atoms for key, value in sorted(database_element_counts.items())
            },
        },
        "fingerprint_similarity_to_db": numeric_summary(
            [row["max_db_tanimoto"] for row in generated_rows]
        ),
        "exact_scaffold_support": generated_categories["scaffold_seen_in_db"],
        "local_environment_support": generated_categories["all_local_env_supported"],
        "est_groups": est_groups,
        "est_correlations": est_correlations,
        "stage_rates_by_requested_heavy_atoms": size_stage_rates,
        "joint_candidate_filter": {
            "xtb_all": len(est_rows),
            "clean_and_supported": len(supported_ranked),
            "abs_est_le_0p05": sum(abs(row["est_eV"]) <= 0.05 for row in est_rows),
            "abs_est_le_0p10": sum(abs(row["est_eV"]) <= 0.10 for row in est_rows),
            "clean_supported_abs_est_le_0p05": sum(
                abs(row["est_eV"]) <= 0.05 for row in supported_ranked
            ),
            "clean_supported_abs_est_le_0p10": len(low_clean),
            "clean_supported_abs_est_le_0p10_fosc_ge_0p01": sum(
                row["s1_f_osc"] >= 0.01 for row in low_clean
            ),
            "clean_supported_abs_est_le_0p10_fosc_ge_0p05": sum(
                row["s1_f_osc"] >= 0.05 for row in low_clean
            ),
        },
        "representative_panels": panels,
        "reference_epoch626_summary": str(args.reference_summary.resolve()) if reference else None,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    scope = summary["scope"]
    generated = summary["generated_category_counts"]
    database = summary["database_category_counts"]
    similarity = summary["fingerprint_similarity_to_db"] or {}
    all_est = est_groups["all"]
    report = [
        "# Detailed structure and XTB/sTDA analysis",
        "",
        "## Scope",
        "",
        f"- Requested: {scope['requested']}",
        f"- Strict RDKit-valid: {scope['rdkit_valid']} ({scope['rdkit_valid']/scope['requested']:.1%})",
        f"- Connected: {scope['connected']}",
        f"- Novel, unique, connected: {scope['novel_unique_connected']}",
        f"- Neutral closed-shell structures entering XTB: {scope['strict_prep_ok']}",
        f"- XTB/sTDA successful: {scope['xtb_stda_ok']}",
        f"- Database structures: {scope['database_molecules']}",
        "",
        "## Generated topology (novel unique connected subset)",
        "",
    ]
    labels = [
        ("has_aromatic_ring", "Aromatic ring present"),
        ("polyaromatic_2plus", "Polyaromatic (>=2 aromatic rings)"),
        ("only_5_6_rings", "Only 5/6-member rings"),
        ("has_3_4_ring", "Any 3/4-member ring"),
        ("has_9plus_ring", "Any >=9-member ring"),
        ("fused_ring_pair", "Fused ring pair"),
        ("spiro_atom", "Spiro atom"),
        ("bridgehead_atom", "Bridgehead atom"),
        ("radical", "Radical"),
        ("charged", "Net charged"),
        ("topology_clean", "Topology-clean proxy"),
        ("all_local_env_supported", "All local environments DB-supported"),
    ]
    for key, label in labels:
        item = generated[key]
        report.append(f"- {label}: {item['count']} ({item['fraction']:.1%})")
    report += [
        "",
        "## Database reference",
        "",
        f"- DB aromatic-ring molecules: {database['has_aromatic_ring']['fraction']:.1%}",
        f"- DB only-5/6-ring molecules: {database['only_5_6_rings']['fraction']:.1%}",
        f"- DB 3/4-ring molecules: {database['has_3_4_ring']['fraction']:.1%}",
        f"- DB >=9-ring molecules: {database['has_9plus_ring']['fraction']:.1%}",
        f"- Morgan radius-2 nearest-DB similarity: mean {similarity.get('mean', math.nan):.3f}, median {similarity.get('median', math.nan):.3f}",
        f"- Exact DB scaffold support: {generated['scaffold_seen_in_db']['count']} ({generated['scaffold_seen_in_db']['fraction']:.1%})",
        f"- Ring-size JSD: {summary['distribution_jsd_bits']['ring_size']:.4f} bits",
        f"- Ring-count JSD: {summary['distribution_jsd_bits']['ring_count']:.4f} bits",
        f"- Attached-H state JSD: {summary['distribution_jsd_bits']['attached_h_state']:.4f} bits",
        "",
        "## XTB/sTDA EST (target 0 eV)",
        "",
        f"- Successful: {all_est['n']}",
        f"- Mean / median EST: {all_est['mean']:.3f} / {all_est['median']:.3f} eV",
        f"- MAE to target: {all_est['mae_to_0']:.3f} eV",
        f"- |EST| <= 0.05 eV: {all_est['abs_le_0p05']}",
        f"- |EST| <= 0.10 eV: {all_est['abs_le_0p10']}",
        f"- Clean + DB-local-supported and |EST| <= 0.10 eV: {len(low_clean)}",
        "",
        "## Strict XTB-input exclusions",
        "",
    ]
    for key, value in sorted(strict_failures.items()):
        report.append(f"- {key}: {value}")
    if reference:
        ref_scope = reference.get("scope", {})
        ref_gen = reference.get("generated_category_counts", {})
        ref_est = reference.get("est_groups", {}).get("all", {})
        report += [
            "",
            "## Comparison with epoch 626",
            "",
            "| Metric | epoch 626 | epoch 935 |",
            "|---|---:|---:|",
            f"| RDKit valid | {ref_scope.get('rdkit_valid', 'NA')} | {scope['rdkit_valid']} |",
            f"| Novel unique connected | {ref_scope.get('novel_unique_connected', 'NA')} | {scope['novel_unique_connected']} |",
            f"| XTB/sTDA success | {ref_scope.get('xtb_stda_ok', 'NA')} | {scope['xtb_stda_ok']} |",
            f"| 3/4-ring fraction | {ref_gen.get('has_3_4_ring', {}).get('fraction', math.nan):.1%} | {generated['has_3_4_ring']['fraction']:.1%} |",
            f"| >=9-ring fraction | {ref_gen.get('has_9plus_ring', {}).get('fraction', math.nan):.1%} | {generated['has_9plus_ring']['fraction']:.1%} |",
            f"| Only 5/6-ring fraction | {ref_gen.get('only_5_6_rings', {}).get('fraction', math.nan):.1%} | {generated['only_5_6_rings']['fraction']:.1%} |",
            f"| Topology-clean fraction | {ref_gen.get('topology_clean', {}).get('fraction', math.nan):.1%} | {generated['topology_clean']['fraction']:.1%} |",
            f"| Mean EST | {ref_est.get('mean', math.nan):.3f} | {all_est['mean']:.3f} |",
            f"| EST MAE | {ref_est.get('mae_to_0', math.nan):.3f} | {all_est['mae_to_0']:.3f} |",
        ]
    report += [
        "",
        "See `summary.json`, `molecule_structure_details.csv`, candidate CSVs, and PNG panels.",
    ]
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
