#!/usr/bin/env python3
"""Strict v5 xTB/sTDA pipeline with one-time hydrogen materialization."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from joint_xtb_stda_pipeline import aggregate, run_one


def prepare_strict(source: Path, output: Path, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    config_path = output / "preparation_config.json"
    config = {
        "schema_version": 1,
        "hydrogen_policy": "single_Chem.AddHs_without_SMILES_roundtrip",
        "require_closed_shell": True,
        "require_net_charge_zero": True,
        "etkdg_timeout_seconds_per_attempt": 15,
    }
    if manifest.exists() and any(output.glob("generated_*.sdf")):
        if not config_path.exists() or json.loads(config_path.read_text()) != config:
            raise RuntimeError("existing input cache does not use the v5 strict hydrogen policy; choose a new output directory")
        with manifest.open(newline="") as handle:
            return list(csv.DictReader(handle))

    rows = []
    supplier = Chem.SDMolSupplier(str(source), removeHs=False, sanitize=True)
    for index, molecule in enumerate(supplier):
        name = f"generated_{index:06d}"
        row = {
            "name": name, "source_index": index, "status": "", "smiles": "",
            "formal_charge": "", "uhf": "", "heavy_atoms": "", "rings": "",
            "aromatic_rings": "", "elements": "", "hydrogens": "", "error": "",
        }
        try:
            if molecule is None:
                raise ValueError("RDKit read failed")
            base = Chem.Mol(molecule)
            if any(atom.GetAtomicNum() == 1 for atom in base.GetAtoms()):
                raise ValueError("v5 sampler output must be a hydrogen-free heavy-atom graph")
            Chem.SanitizeMol(base)
            charge = sum(atom.GetFormalCharge() for atom in base.GetAtoms())
            uhf = sum(atom.GetNumRadicalElectrons() for atom in base.GetAtoms())
            if charge:
                raise ValueError(f"net-charged molecule rejected before xTB: charge={charge}")
            if uhf:
                raise ValueError(f"open-shell molecule rejected before xTB: UHF={uhf}")
            smiles = Chem.MolToSmiles(base, canonical=True, isomericSmiles=False)
            # This is the only heavy->explicit-H conversion in the v5 path.
            molecule_3d = None
            for attempt in range(8):
                candidate = Chem.AddHs(
                    base, explicitOnly=False, addCoords=False,
                )
                params = AllChem.ETKDGv3()
                params.randomSeed = seed + index + attempt * 104729
                params.useRandomCoords = True
                params.maxIterations = 1000 if attempt == 0 else 5000
                params.boxSizeMult = 2.0 + 0.5 * attempt
                params.numThreads = 1
                params.timeout = 15
                if AllChem.EmbedMolecule(candidate, params) == 0:
                    molecule_3d = candidate
                    break
            if molecule_3d is None:
                raise ValueError("ETKDGv3 failed after 8 seeded attempts")
            if AllChem.MMFFHasAllMoleculeParams(molecule_3d):
                AllChem.MMFFOptimizeMolecule(molecule_3d, maxIters=1000)
            elif AllChem.UFFHasAllMoleculeParams(molecule_3d):
                AllChem.UFFOptimizeMolecule(molecule_3d, maxIters=1000)
            molecule_3d.SetProp("_Name", name)
            molecule_3d.SetProp("SMILES", smiles)
            molecule_3d.SetIntProp("formal_charge", charge)
            molecule_3d.SetIntProp("uhf", uhf)
            writer = Chem.SDWriter(str(output / f"{name}.sdf"))
            writer.write(molecule_3d)
            writer.close()
            row.update(
                status="ok", smiles=smiles, formal_charge=charge, uhf=uhf,
                heavy_atoms=base.GetNumHeavyAtoms(),
                rings=rdMolDescriptors.CalcNumRings(base),
                aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(base),
                elements=",".join(sorted({atom.GetSymbol() for atom in base.GetAtoms()})),
                hydrogens=molecule_3d.GetNumAtoms() - base.GetNumAtoms(),
            )
        except Exception as exc:
            row.update(status="failed", error=str(exc))
        rows.append(row)
        if (index + 1) % 50 == 0:
            print(json.dumps({"event": "prepare", "completed": index + 1}), flush=True)
    if not rows:
        raise RuntimeError("no molecules in source SDF")
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    config_path.write_text(json.dumps(config, indent=2))
    (output / "summary.json").write_text(json.dumps({
        "total": len(rows),
        "exported": sum(row["status"] == "ok" for row in rows),
        "failed": sum(row["status"] != "ok" for row in rows),
        **config,
    }, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xtb-env", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--target-s1", type=float, required=True)
    parser.add_argument("--target-t1", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    args = parser.parse_args()
    if args.workers < 1 or args.threads != 1:
        raise ValueError("workers must be positive and threads must remain 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.output_dir / "inputs"
    work = args.output_dir / "work"
    work.mkdir(exist_ok=True)
    metadata = prepare_strict(args.input_sdf, inputs, args.seed)
    files = sorted(inputs.glob("generated_*.sdf"))
    environment = os.environ.copy()
    environment["PATH"] = str(args.xtb_env / "bin") + ":/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe:" + environment["PATH"]
    environment["LD_LIBRARY_PATH"] = str(args.xtb_env / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    environment["XTB4STDAHOME"] = "/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home"
    environment["OMP_STACKSIZE"] = "2G"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = "1"
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, path, work, environment): path for path in files}
        with (work / "progress.jsonl").open("a") as progress:
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                print(json.dumps({"event": "result", "completed": completed, "total": len(files), "name": result["name"], "status": result["status"], "s1": result.get("s1_energy_eV"), "t1": result.get("t1_energy_eV"), "elapsed_s": time.time() - started}), flush=True)
    print(json.dumps({"event": "complete", **aggregate(work, metadata, args.target_s1, args.target_t1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
