#!/usr/bin/env python3
"""Export manifest-approved hybrid molecules as 3-D SDF inputs for xTB."""
import argparse
import csv
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

from export_hybrid_png import decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in args.samples.read_text().splitlines()]
    approved = []
    with args.manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "ok":
                approved.append(row)

    results = []
    for row in approved:
        index = int(row["generation_index"])
        name = row["name"]
        try:
            mol, smiles = decode(records[index])
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 20260911 + index
            params.useRandomCoords = True
            params.maxIterations = 1000
            if AllChem.EmbedMolecule(mol, params) != 0:
                raise ValueError("ETKDG embedding failed")
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
            mol.SetProp("source_generation_index", str(index))
            mol.SetProp("canonical_smiles", smiles)
            writer = Chem.SDWriter(str(args.output_dir / f"{name}.sdf"))
            writer.write(mol)
            writer.close()
            results.append({"name": name, "generation_index": index, "status": "ok", "error": ""})
        except Exception as exc:
            results.append({"name": name, "generation_index": index, "status": "failed", "error": str(exc)})

    with (args.output_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("name", "generation_index", "status", "error"))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "requested": len(approved),
        "exported": sum(row["status"] == "ok" for row in results),
        "failed": sum(row["status"] != "ok" for row in results),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
