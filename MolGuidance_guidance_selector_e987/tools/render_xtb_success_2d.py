#!/usr/bin/env python3
"""Render successful xTB/sTDA result molecules as paginated RDKit 2D PNGs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-s1", type=float, required=True)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--mols-per-row", type=int, default=5)
    return parser.parse_args()


def finite_label(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "NA"


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.error")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.results.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "ok"]

    drawable: list[tuple[dict[str, str], Chem.Mol]] = []
    parse_failures: list[dict[str, str]] = []
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            parse_failures.append(row)
            continue
        try:
            rdDepictor.Compute2DCoords(mol)
        except Exception as exc:
            failed = dict(row)
            failed["depiction_error"] = f"{type(exc).__name__}: {exc}"
            parse_failures.append(failed)
            continue
        drawable.append((row, mol))

    page_paths: list[str] = []
    manifest_rows: list[dict[str, str | int]] = []
    for page_index, start in enumerate(range(0, len(drawable), args.page_size)):
        page = drawable[start : start + args.page_size]
        mols: list[Chem.Mol] = []
        legends: list[str] = []
        for position, (row, mol) in enumerate(page):
            mols.append(mol)
            legends.append(
                f"{row['name']}  S1={finite_label(row.get('s1_energy_eV', ''))} "
                f"T1={finite_label(row.get('t1_energy_eV', ''))}\n"
                f"f={finite_label(row.get('s1_f_osc', ''), 4)}"
            )
            manifest_rows.append(
                {
                    **row,
                    "png_page": page_index,
                    "position_on_page": position,
                }
            )
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=args.mols_per_row,
            subImgSize=(320, 260),
            legends=legends,
            useSVG=False,
        )
        png_path = args.out_dir / f"xtb_success_2d_{page_index:03d}.png"
        image.save(png_path)
        page_paths.append(str(png_path.resolve()))

    manifest_path = args.out_dir / "xtb_success_2d_manifest.csv"
    if manifest_rows:
        with manifest_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

    summary = {
        "target_s1_eV": args.target_s1,
        "source_results": str(args.results.resolve()),
        "successful_rows": len(rows),
        "rendered_molecules": len(drawable),
        "smiles_parse_failures": len(parse_failures),
        "page_size": args.page_size,
        "pages": len(page_paths),
        "first_page": page_paths[0] if page_paths else None,
        "manifest": str(manifest_path.resolve()),
        "png_paths": page_paths,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
