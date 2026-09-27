#!/usr/bin/env python3
"""Generate V7 molecules over a near-zero singlet-triplet gap range."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor

from joint_kekule_diffusion.chemistry_v5 import AtomStateCodec
from joint_kekule_diffusion.chemistry_v7 import (
    ChemistryReferenceV7,
    ConditionSurrogateV7,
    GeometryReferenceV7,
)
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v7 import HeavyAtomJointDDPMV7
from joint_kekule_diffusion.sample_v7 import (
    evaluate_sample_v7,
    load_known_hp1kg,
    molecule_with_coordinates,
    sample_batch_v7,
    summarize_records_v7,
    validate_v7_artifacts,
)


def reference_midpoints(path: Path, threshold: float) -> np.ndarray:
    values = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train":
                continue
            s1 = float(row["reference_s1_eV"])
            t1 = float(row["reference_t1_eV"])
            if abs(s1 - t1) <= threshold:
                values.append((s1 + t1) * .5)
    if not values:
        raise ValueError("no train reference molecules in requested delta-EST range")
    return np.asarray(values, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--delta-min", type=float, default=-.05)
    parser.add_argument("--delta-max", type=float, default=.05)
    parser.add_argument("--reference-threshold", type=float, default=.05)
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=2209)
    args = parser.parse_args()
    if args.num <= 0 or args.batch_size <= 0 or args.delta_min > args.delta_max:
        raise ValueError("invalid generation configuration")

    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_v7_artifacts(metadata, checkpoint, args.root)
    codec = AtomStateCodec.from_metadata(metadata["atom_states"])
    chemistry = ChemistryReferenceV7.from_metadata(metadata["chemistry_reference"])
    geometry = GeometryReferenceV7.from_metadata(metadata["geometry_reference"])
    surrogate = ConditionSurrogateV7.from_metadata(metadata["condition_surrogate"])
    model_args = metadata["model_args"]
    model = HeavyAtomJointDDPMV7(
        codec, model_args["hidden"], metadata["max_atoms"], model_args["layers"],
        model_args["condition_dropout"], model_args["s1_rbf_bins"],
        model_args["t1_rbf_bins"], model_args["condition_rbf_min"],
        model_args["condition_rbf_max"], model_args["condition_rbf_width_scale"],
        model_args["model_dropout"], model_args["count_dropout"],
    ).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    diffusion = JointDiffusion(marginals=metadata["diffusion_marginals"]).cuda()
    stats = metadata["condition_stats"]
    midpoints = reference_midpoints(Path(args.reference_csv), args.reference_threshold)
    selected_midpoints = rng.choice(midpoints, size=args.num, replace=True)
    deltas = rng.uniform(args.delta_min, args.delta_max, size=args.num)
    requested_s1 = selected_midpoints + deltas * .5
    requested_t1 = selected_midpoints - deltas * .5
    normalized = torch.tensor(np.column_stack((
        (requested_s1 - stats["s1_energy_mean"]) / stats["s1_energy_std"],
        (requested_t1 - stats["t1_energy_mean"]) / stats["t1_energy_std"],
    )), dtype=torch.float32, device="cuda")

    projection = metadata["topology_projection"]
    known = load_known_hp1kg(args.root)
    records, unique_smiles, unique_molecules = [], set(), []
    started = time.time()
    for begin in range(0, args.num, args.batch_size):
        size = min(args.batch_size, args.num - begin)
        sampled = sample_batch_v7(
            model, diffusion, normalized[begin:begin + size], size, 0, args.cfg,
            metadata["fixed_atom_count"], 25., 8., 4, chemistry,
            projection["prior_weight"], projection["reproject_threshold"],
            args.seed + begin,
        )
        for local in range(size):
            index = begin + local
            item, molecule = evaluate_sample_v7(
                sampled, local, codec, chemistry, geometry, surrogate,
                float(requested_s1[index]), float(requested_t1[index]), False,
                args.seed + index,
            )
            smiles = item.get("smiles", "")
            novel = bool(smiles and smiles not in known)
            unique = bool(novel and smiles not in unique_smiles)
            item.update({
                "index": index,
                "atom_count": int(sampled["atom_count"][local]),
                "requested_s1": float(requested_s1[index]),
                "requested_t1": float(requested_t1[index]),
                "requested_delta_est": float(deltas[index]),
                "novel_vs_db": int(novel),
                "unique_novel": int(unique),
            })
            records.append(item)
            if unique:
                unique_smiles.add(smiles)
                unique_molecules.append(molecule_with_coordinates(
                    molecule, sampled["positions"][local, :item["atom_count"]],
                ))
        with (output / "molecules.csv").open("w", newline="") as handle:
            fields = sorted({key for row in records for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(records)
        print(json.dumps({
            "event": "generation", "completed": len(records),
            "projected_valid": sum(row["projection_valid"] for row in records),
            "chemical_clean": sum(row.get("chemical_clean", 0) for row in records),
        }), flush=True)

    summary = summarize_records_v7(records)
    summary.update({
        "checkpoint_epoch": checkpoint["epoch"], "cfg": args.cfg,
        "requested_delta_est_min": args.delta_min,
        "requested_delta_est_max": args.delta_max,
        "reference_midpoint_count": int(len(midpoints)),
        "requested_s1_min": float(requested_s1.min()),
        "requested_s1_max": float(requested_s1.max()),
        "requested_t1_min": float(requested_t1.min()),
        "requested_t1_max": float(requested_t1.max()),
        "novel_vs_db": sum(row["novel_vs_db"] for row in records),
        "unique_novel": len(unique_molecules),
        "elapsed_s": time.time() - started,
    })
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    writer = Chem.SDWriter(str(output / "unique_novel_connected.sdf"))
    for index, molecule in enumerate(unique_molecules):
        molecule.SetProp("_Name", f"generated_{index:06d}")
        molecule.SetProp("SMILES", Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
        writer.write(molecule)
    writer.close()
    for page_start in range(0, len(unique_molecules), 100):
        page, legends = [], []
        for molecule in unique_molecules[page_start:page_start + 100]:
            depicted = Chem.Mol(molecule); depicted.RemoveAllConformers()
            rdDepictor.Compute2DCoords(depicted, canonOrient=True, clearConfs=True)
            page.append(depicted)
            legends.append(Chem.MolToSmiles(depicted, canonical=True, isomericSmiles=False))
        Draw.MolsToGridImage(page, molsPerRow=10, subImgSize=(300, 240), legends=legends).save(
            str(output / f"novel_valid_{page_start // 100:03d}.png")
        )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
