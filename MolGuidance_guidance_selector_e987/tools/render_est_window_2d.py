#!/usr/bin/env python3
"""Render XTB-success molecules whose calculated EST lies in a given window."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-est", type=float, default=0.0)
    parser.add_argument("--max-est", type=float, default=0.5)
    parser.add_argument("--page-size", type=int, default=40)
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.error")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.results.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    selected: list[tuple[float, dict[str, str], Chem.Mol]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        est = float(row["s1_energy_eV"]) - float(row["t1_energy_eV"])
        if args.min_est <= est <= args.max_est:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is not None:
                rdDepictor.Compute2DCoords(mol)
                selected.append((est, row, mol))
    selected.sort(key=lambda item: (abs(item[0]), int(item[1]["generation_index"])))

    manifest_rows: list[dict[str, object]] = []
    pages: list[str] = []
    for page_index, start in enumerate(range(0, len(selected), args.page_size)):
        page = selected[start : start + args.page_size]
        mols = [item[2] for item in page]
        legends = [
            f"idx={row['generation_index']} EST={est:.3f}\n"
            f"S1={float(row['s1_energy_eV']):.3f} T1={float(row['t1_energy_eV']):.3f}"
            for est, row, _ in page
        ]
        for position, (est, row, _) in enumerate(page):
            manifest_rows.append(
                {
                    **row,
                    "calculated_est_eV": est,
                    "png_page": page_index,
                    "position_on_page": position,
                }
            )
        image = Draw.MolsToGridImage(
            mols,
            molsPerRow=5,
            subImgSize=(320, 260),
            legends=legends,
            useSVG=False,
        )
        path = args.out_dir / f"est_window_{page_index:03d}.png"
        image.save(path)
        pages.append(str(path.resolve()))

    manifest = args.out_dir / "selected_molecules.csv"
    if manifest_rows:
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
    summary = {
        "min_est_eV": args.min_est,
        "max_est_eV": args.max_est,
        "selected": len(selected),
        "pages": len(pages),
        "manifest": str(manifest.resolve()),
        "first_page": pages[0] if pages else None,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
