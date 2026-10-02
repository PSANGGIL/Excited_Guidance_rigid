#!/usr/bin/env python3
"""Analyze MolGuidance SDF output and render novel connected molecules."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor


def canonical_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def load_database_smiles(data_dir: Path) -> set[str]:
    smiles: set[str] = set()
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_data_smiles.pkl"
        with path.open("rb") as handle:
            entries = pickle.load(handle)
        for entry in entries:
            mol = Chem.MolFromSmiles(entry)
            if mol is not None:
                smiles.add(canonical_smiles(mol))
    return smiles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--db-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--condition-manifest", type=Path, default=None,
        help="Optional CSV keyed by generation_index for per-condition metadata",
    )
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Requested sample count when invalid molecules were omitted from the SDF",
    )
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_smiles = load_database_smiles(args.db_dir)
    condition_rows: dict[int, dict[str, str]] = {}
    if args.condition_manifest is not None:
        with args.condition_manifest.open(newline="") as handle:
            condition_rows = {
                int(row["generation_index"]): row for row in csv.DictReader(handle)
            }
    # Delay sanitization until after reading SDF metadata. This preserves the
    # original generation index even for chemically invalid but parseable rows.
    supplier = Chem.SDMolSupplier(str(args.sdf), sanitize=False, removeHs=False)

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    novel_molecules: list[tuple[int, int, str, Chem.Mol]] = []
    element_counts: Counter[str] = Counter()
    ring_size_counts: Counter[int] = Counter()
    valid = connected = unique = novel = aromatic = radicals = 0

    for index, mol in enumerate(supplier):
        if mol is None:
            rows.append(
                {"index": index, "valid": False, "invalid_reason": "SDF parse/sanitize failure"}
            )
            continue

        generation_index = (
            mol.GetIntProp("generation_index")
            if mol.HasProp("generation_index")
            else index
        )
        condition = condition_rows.get(generation_index, {})
        if condition:
            condition_range = condition.get(
                "condition_range", condition.get("s1_range", "")
            )
            target_value = condition.get(
                "target_value", condition.get("target_s1_eV", "")
            )
            condition_name = condition.get(
                "condition_name", "s1" if "target_s1_eV" in condition else ""
            )
            mol.SetProp("condition_range", condition_range)
            mol.SetProp("condition_name", condition_name)
            mol.SetDoubleProp("target_property", float(target_value))
            if "s1_range" in condition:
                mol.SetProp("s1_range", condition["s1_range"])
            mol.SetIntProp(
                "requested_atom_count", int(condition["requested_atom_count"])
            )

        try:
            Chem.SanitizeMol(mol)
            allowed_bond_types = {
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            }
            if any(bond.GetBondType() not in allowed_bond_types for bond in mol.GetBonds()):
                raise ValueError("unsupported bond type")
            smiles = canonical_smiles(mol)
        except Exception as exc:
            rows.append(
                {
                    "index": index,
                    "generation_index": generation_index,
                    "condition_name": condition.get("condition_name", ""),
                    "condition_range": condition.get(
                        "condition_range", condition.get("s1_range", "")
                    ),
                    "target_value": condition.get(
                        "target_value", condition.get("target_s1_eV", "")
                    ),
                    "s1_range": condition.get("s1_range", ""),
                    "target_s1_eV": condition.get("target_s1_eV", ""),
                    "requested_atom_count": condition.get(
                        "requested_atom_count", ""
                    ),
                    "valid": False,
                    "invalid_reason": f"canonicalization failure: {type(exc).__name__}",
                }
            )
            continue

        valid += 1
        is_connected = len(Chem.GetMolFrags(mol)) == 1
        is_unique = smiles not in seen
        in_database = smiles in db_smiles
        if is_connected:
            connected += 1
        if is_unique:
            unique += 1
            seen.add(smiles)

        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        ring_sizes = [len(ring) for ring in atom_rings]
        aromatic_rings = sum(
            all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
            for ring in atom_rings
        )
        if aromatic_rings:
            aromatic += 1
        radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        if radical_electrons:
            radicals += 1

        for atom in mol.GetAtoms():
            element_counts[atom.GetSymbol()] += 1
        ring_size_counts.update(ring_sizes)

        is_novel = is_unique and is_connected and not in_database
        if is_novel:
            novel += 1
            novel_molecules.append((index, generation_index, smiles, Chem.Mol(mol)))

        rows.append(
            {
                "index": index,
                "generation_index": generation_index,
                "condition_name": condition.get("condition_name", ""),
                "condition_range": condition.get(
                    "condition_range", condition.get("s1_range", "")
                ),
                "target_value": condition.get(
                    "target_value", condition.get("target_s1_eV", "")
                ),
                "s1_range": condition.get("s1_range", ""),
                "target_s1_eV": condition.get("target_s1_eV", ""),
                "requested_atom_count": condition.get("requested_atom_count", ""),
                "valid": True,
                "connected": is_connected,
                "unique_first": is_unique,
                "in_database": in_database,
                "novel_unique_connected": is_novel,
                "smiles": smiles,
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
                "radical_electrons": radical_electrons,
                "num_rings": len(atom_rings),
                "num_aromatic_rings": aromatic_rings,
                "ring_sizes": ";".join(map(str, ring_sizes)),
                "elements": ";".join(sorted({atom.GetSymbol() for atom in mol.GetAtoms()})),
            }
        )

    csv_path = args.out_dir / "molecules.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    novel_sdf = args.out_dir / "unique_novel_connected.sdf"
    writer = Chem.SDWriter(str(novel_sdf))
    writer.SetKekulize(False)
    for source_sdf_index, generation_index, smiles, mol in novel_molecules:
        mol.SetIntProp("source_sdf_index", source_sdf_index)
        mol.SetIntProp("generation_index", generation_index)
        mol.SetProp("canonical_smiles", smiles)
        writer.write(mol)
    writer.close()

    png_paths: list[str] = []
    for page_index, start in enumerate(range(0, len(novel_molecules), args.page_size)):
        page = novel_molecules[start : start + args.page_size]
        page_mols: list[Chem.Mol] = []
        legends: list[str] = []
        for source_sdf_index, generation_index, smiles, mol in page:
            draw_mol = Chem.Mol(mol)
            rdDepictor.Compute2DCoords(draw_mol)
            page_mols.append(draw_mol)
            legends.append(
                f"#{generation_index} (sdf {source_sdf_index}) {smiles[:42]}"
            )
        image = Draw.MolsToGridImage(
            page_mols,
            molsPerRow=5,
            subImgSize=(300, 240),
            legends=legends,
            useSVG=False,
        )
        png_path = args.out_dir / f"novel_valid_{page_index:03d}.png"
        image.save(png_path)
        png_paths.append(str(png_path.resolve()))

    total_records = len(rows)
    summary = {
        "source_sdf": str(args.sdf.resolve()),
        "database_unique_smiles": len(db_smiles),
        "records": total_records,
        "requested": args.expected_count,
        "valid": valid,
        "valid_fraction": valid / total_records if total_records else 0.0,
        "valid_fraction_of_requested": (
            valid / args.expected_count if args.expected_count else None
        ),
        "connected": connected,
        "connected_fraction_of_valid": connected / valid if valid else 0.0,
        "unique": unique,
        "unique_fraction_of_valid": unique / valid if valid else 0.0,
        "novel_unique_connected": novel,
        "novel_fraction_of_valid": novel / valid if valid else 0.0,
        "molecules_with_aromatic_ring": aromatic,
        "aromatic_fraction_of_valid": aromatic / valid if valid else 0.0,
        "molecules_with_radicals": radicals,
        "element_counts": dict(sorted(element_counts.items())),
        "ring_size_counts": {str(key): value for key, value in sorted(ring_size_counts.items())},
        "csv": str(csv_path.resolve()),
        "novel_sdf": str(novel_sdf.resolve()),
        "pngs": png_paths,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
