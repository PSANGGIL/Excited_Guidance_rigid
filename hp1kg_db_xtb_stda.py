#!/usr/bin/env python3
"""Restartable HP1KG reference xTB/sTDA calculation and NEVPT2 comparison.

The source database has no reference Cartesian geometry.  Each SMILES is
therefore embedded with ETKDGv3, optimized with MMFF/UFF, optimized once more
by xTB, and evaluated by sTDA-xTB.  Four one-thread workers are used by
default, matching the generated-molecule evaluation pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from joint_xtb_stda_pipeline import run_one


SPLITS = ("train", "val", "test")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def load_database(root: Path) -> list[dict]:
    records: list[dict] = []
    global_index = 0
    for split in SPLITS:
        split_path = root / f"{split}.pt"
        rows = torch.load(split_path, map_location="cpu", weights_only=False)
        for split_index, row in enumerate(rows):
            records.append(
                {
                    "name": f"db_{global_index:06d}",
                    "global_index": global_index,
                    "split": split,
                    "split_index": split_index,
                    "database_id": str(row["id"]),
                    "smiles": str(row["smiles"]),
                    "reference_s1_eV": float(row["s1"]),
                    "reference_t1_eV": float(row["t1"]),
                    "training_geometry_source": str(row.get("geometry_source", "")),
                }
            )
            global_index += 1
    return records


def prepare_inputs(records: list[dict], output: Path, seed: int) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.csv"
    config_path = output / "preparation_config.json"
    config = {
        "schema_version": 1,
        "source_geometry_available": False,
        "geometry_pipeline": "SMILES -> AddHs -> ETKDGv3 -> MMFF/UFF",
        "seed": seed,
    }
    if manifest_path.exists() and config_path.exists():
        existing_config = json.loads(config_path.read_text())
        if existing_config != config:
            raise RuntimeError("input cache configuration mismatch; use a new output directory")
        with manifest_path.open(newline="") as handle:
            cached = list(csv.DictReader(handle))
        if len(cached) == len(records):
            return cached

    prepared: list[dict] = []
    for record in records:
        row = dict(record)
        row.update(
            status="",
            formal_charge="",
            uhf="",
            heavy_atoms="",
            rings="",
            aromatic_rings="",
            elements="",
            hydrogens="",
            embedding_geometry_source="",
            error="",
        )
        try:
            molecule = Chem.MolFromSmiles(record["smiles"])
            if molecule is None:
                raise ValueError("SMILES parse failed")
            Chem.SanitizeMol(molecule)
            formal_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
            uhf = sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
            molecule_3d = Chem.AddHs(molecule, explicitOnly=False, addCoords=False)
            parameters = AllChem.ETKDGv3()
            parameters.randomSeed = seed + int(record["global_index"])
            parameters.useRandomCoords = True
            parameters.maxIterations = 1000
            if AllChem.EmbedMolecule(molecule_3d, parameters) != 0:
                raise ValueError("ETKDGv3 failed")
            geometry_source = "ETKDGv3"
            if AllChem.MMFFHasAllMoleculeParams(molecule_3d):
                AllChem.MMFFOptimizeMolecule(molecule_3d, maxIters=1000)
                geometry_source += "+MMFF"
            elif AllChem.UFFHasAllMoleculeParams(molecule_3d):
                AllChem.UFFOptimizeMolecule(molecule_3d, maxIters=1000)
                geometry_source += "+UFF"
            molecule_3d.SetProp("_Name", record["name"])
            molecule_3d.SetProp("database_id", record["database_id"])
            molecule_3d.SetProp("SMILES", record["smiles"])
            molecule_3d.SetDoubleProp("reference_s1_eV", record["reference_s1_eV"])
            molecule_3d.SetDoubleProp("reference_t1_eV", record["reference_t1_eV"])
            writer = Chem.SDWriter(str(output / f"{record['name']}.sdf"))
            writer.write(molecule_3d)
            writer.close()
            row.update(
                status="ok",
                formal_charge=formal_charge,
                uhf=uhf,
                heavy_atoms=molecule.GetNumHeavyAtoms(),
                rings=rdMolDescriptors.CalcNumRings(molecule),
                aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(molecule),
                elements=",".join(sorted({atom.GetSymbol() for atom in molecule.GetAtoms()})),
                hydrogens=molecule_3d.GetNumAtoms() - molecule.GetNumAtoms(),
                embedding_geometry_source=geometry_source,
            )
        except Exception as exc:  # preserve every failed DB row in the manifest
            row.update(status="failed", error=str(exc))
        prepared.append(row)
        completed = int(record["global_index"]) + 1
        if completed % 50 == 0 or completed == len(records):
            print(json.dumps({"event": "prepare", "completed": completed, "total": len(records)}), flush=True)

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prepared[0].keys())
        writer.writeheader()
        writer.writerows(prepared)
    atomic_json(config_path, config)
    atomic_json(
        output / "summary.json",
        {
            "total": len(prepared),
            "exported": sum(row["status"] == "ok" for row in prepared),
            "failed": sum(row["status"] != "ok" for row in prepared),
            **config,
        },
    )
    return prepared


def correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def metric(reference: list[float], calculated: list[float]) -> dict:
    errors = [value - target for target, value in zip(reference, calculated)]
    return {
        "n": len(errors),
        "reference_mean_eV": statistics.fmean(reference),
        "calculated_mean_eV": statistics.fmean(calculated),
        "bias_calculated_minus_reference_eV": statistics.fmean(errors),
        "mae_eV": statistics.fmean(abs(error) for error in errors),
        "rmse_eV": math.sqrt(statistics.fmean(error * error for error in errors)),
        "pearson_r": correlation(reference, calculated),
    }


def aggregate(work: Path, metadata: list[dict], output: Path) -> dict:
    by_name = {row["name"]: row for row in metadata}
    raw_results: list[dict] = []
    rows: list[dict] = []
    for result_path in sorted(work.glob("db_*/result.json")):
        try:
            result = json.loads(result_path.read_text())
        except Exception:
            continue
        raw_results.append(result)
        source = by_name.get(result["name"], {})
        row = {key: source.get(key, "") for key in source}
        row.update(
            status=result.get("status", ""),
            calculated_s1_eV=result.get("s1_energy_eV", ""),
            calculated_t1_eV=result.get("t1_energy_eV", ""),
            calculated_s1_f_osc=result.get("s1_f_osc", ""),
            strongest_singlet_state=result.get("strongest_singlet_state", ""),
            strongest_singlet_energy_eV=result.get("strongest_singlet_energy_eV", ""),
            strongest_singlet_f_osc=result.get("strongest_singlet_f_osc", ""),
        )
        if result.get("status") == "ok":
            row["s1_error_eV"] = float(result["s1_energy_eV"]) - float(source["reference_s1_eV"])
            row["t1_error_eV"] = float(result["t1_energy_eV"]) - float(source["reference_t1_eV"])
        else:
            row["s1_error_eV"] = ""
            row["t1_error_eV"] = ""
        rows.append(row)

    if rows:
        with (output / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(raw_results, indent=2))
    counts: dict[str, int] = {}
    for result in raw_results:
        status = result.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    valid = [row for row in rows if row["status"] == "ok"]
    summary = {
        "database_total": len(metadata),
        "prepared": sum(row.get("status") == "ok" for row in metadata),
        "calculation_completed": len(raw_results),
        "status_counts": counts,
        "comparison_note": "Reference NEVPT2 geometry is unavailable; calculated values use independently embedded ETKDGv3/MMFF-or-UFF/xTB geometries.",
    }
    if valid:
        summary["s1"] = metric(
            [float(row["reference_s1_eV"]) for row in valid],
            [float(row["calculated_s1_eV"]) for row in valid],
        )
        summary["t1"] = metric(
            [float(row["reference_t1_eV"]) for row in valid],
            [float(row["calculated_t1_eV"]) for row in valid],
        )
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("joint_kekule_diffusion/data_hp1kg_nevpt2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xtb-env", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    if args.workers < 1 or args.threads != 1:
        raise ValueError("threads must remain 1 and workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.output_dir / "inputs"
    work = args.output_dir / "work"
    work.mkdir(exist_ok=True)
    records = load_database(args.root)
    metadata = prepare_inputs(records, inputs, args.seed)
    input_files = sorted(inputs.glob("db_*.sdf"))

    environment = os.environ.copy()
    environment["PATH"] = str(args.xtb_env / "bin") + ":/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe:" + environment["PATH"]
    environment["LD_LIBRARY_PATH"] = str(args.xtb_env / "lib") + ":" + environment.get("LD_LIBRARY_PATH", "")
    environment["XTB4STDAHOME"] = "/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home"
    environment["OMP_STACKSIZE"] = "2G"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[key] = str(args.threads)

    started = time.time()
    status_path = args.output_dir / "status.json"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, path, work, environment): path for path in input_files}
        with (work / "progress.jsonl").open("a") as progress:
            for completed, future in enumerate(as_completed(futures), 1):
                result = future.result()
                elapsed = time.time() - started
                eta = elapsed * (len(input_files) - completed) / completed
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                status = {
                    "state": "running",
                    "completed": completed,
                    "total": len(input_files),
                    "last_name": result["name"],
                    "last_status": result["status"],
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                    "workers": args.workers,
                    "threads_per_worker": args.threads,
                }
                atomic_json(status_path, status)
                print(json.dumps({"event": "result", **status}), flush=True)

    summary = aggregate(work, metadata, args.output_dir)
    atomic_json(
        status_path,
        {
            "state": "complete",
            "completed": len(input_files),
            "total": len(input_files),
            "elapsed_s": time.time() - started,
            "summary": summary,
        },
    )
    print(json.dumps({"event": "complete", **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
