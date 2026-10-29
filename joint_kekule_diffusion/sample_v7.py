#!/usr/bin/env python3
"""V7 sampler: unconstrained joint-edge reverse diffusion, one final joint solve."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors

from .chemistry_v5 import AtomStateCodec, decode_heavy_graph
from .chemistry_v7 import (
    ChemistryReferenceV7,
    ConditionSurrogateV7,
    GeometryReferenceV7,
    assess_generated_coordinates,
    assess_structure,
    dataset_fingerprint,
    decode_joint_assignment_v7,
    metadata_fingerprint,
    project_joint_graph_v7,
    source_fingerprints_v7,
    support_warnings_v7,
    topology_assessment_v7,
)
from .model import JointDiffusion, cfg_combine
from .model_v7 import (
    JOINT_EDGE_CLASSES,
    HeavyAtomJointDDPMV7,
    decode_joint_edge_state,
)
from .sample import posterior_step, reverse_coords


def _condition_logits_on_sample(logits, sampled_state, strength=4.):
    one_hot = torch.nn.functional.one_hot(sampled_state.long(), logits.shape[-1]).to(logits.dtype)
    return logits + float(strength) * (2 * one_hot - 1)


def validate_v7_artifacts(metadata, checkpoint, root):
    """Fail closed if metadata, checkpoint or DB bytes differ."""
    actual_metadata_hash = metadata_fingerprint(metadata)
    if metadata.get("metadata_hash") != actual_metadata_hash:
        raise ValueError("metadata hash mismatch")
    if checkpoint.get("metadata_hash") != actual_metadata_hash:
        raise ValueError("checkpoint/metadata hash mismatch")
    embedded = checkpoint.get("metadata")
    if embedded is not None and metadata_fingerprint(embedded) != actual_metadata_hash:
        raise ValueError("embedded checkpoint metadata mismatch")
    actual_dataset_hash = dataset_fingerprint(root)
    if metadata.get("dataset_hash") != actual_dataset_hash:
        raise ValueError("dataset manifest/cache hash mismatch")
    if metadata.get("source_hashes") != source_fingerprints_v7():
        raise ValueError("v7 source hash mismatch")
    return actual_metadata_hash


@torch.inference_mode()
def sample_batch_v7(
    model,
    diffusion,
    condition,
    batch_size,
    atoms=0,
    cfg=1.,
    observed_counts=None,
    coordinate_clip=25.,
    solver_time_limit=8.,
    solver_attempts=4,
    chemistry_reference=None,
    topology_prior_weight=1.5,
    topology_reproject_threshold=.2,
    projection_seed=0,
):
    """Sample the model kernel exactly, then project once at x0.

    There is deliberately no aromatic, valence, connectivity, support or
    topology projection inside the reverse chain. This keeps p(x_{t-1}|x_t)
    aligned with the categorical forward kernel used during training.
    """
    if batch_size <= 0 or not 0 <= atoms <= model.max_atoms:
        raise ValueError(f"atoms must be zero or in [1,{model.max_atoms}]")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("CFG scale must be finite and non-negative")
    device = condition.device
    conditional_count = model.count_head(model.condition_context(condition))
    if cfg == 1:
        count_logits = conditional_count
    else:
        unconditional_count = model.count_head(model.condition_context(condition, unconditional=True))
        count_logits = unconditional_count + cfg * (conditional_count - unconditional_count)
    if not atoms and observed_counts:
        support = torch.zeros(model.max_atoms, dtype=torch.bool, device=device)
        support[torch.as_tensor(observed_counts, device=device) - 1] = True
        count_logits = count_logits.masked_fill(~support[None], -torch.inf)
    counts = (
        torch.full((batch_size,), atoms, dtype=torch.long, device=device)
        if atoms else torch.distributions.Categorical(logits=count_logits).sample() + 1
    )
    index = torch.arange(model.max_atoms, device=device)
    mask = index[None] < counts[:, None]
    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(model.max_atoms, dtype=torch.bool, device=device)[None]

    atom_probability = diffusion.marginal("atom", model.state_types, device)
    atom_state = torch.multinomial(
        atom_probability, batch_size * model.max_atoms, replacement=True,
    ).reshape(batch_size, model.max_atoms)
    edge_probability = diffusion.marginal("joint_edge", JOINT_EDGE_CLASSES, device)
    edge_state = torch.multinomial(
        edge_probability, batch_size * model.max_atoms * model.max_atoms, replacement=True,
    ).reshape(batch_size, model.max_atoms, model.max_atoms)
    upper = torch.triu(torch.ones_like(edge_state, dtype=torch.bool), 1)
    edge_state = torch.where(upper, edge_state, edge_state.transpose(1, 2)).masked_fill(diagonal, 0)
    positions = torch.randn(batch_size, model.max_atoms, 3, device=device) * mask[..., None]
    denominator = mask.sum(1, keepdim=True).clamp_min(1)[..., None]
    positions -= (positions.sum(1, keepdim=True) / denominator) * mask[..., None]
    state = {
        "atom_mask": mask,
        "atom_state": atom_state,
        "edge_state": edge_state,
        "positions": positions,
    }

    final_outputs = None
    for time_index in reversed(range(diffusion.timesteps)):
        timestep = torch.full((batch_size,), time_index, dtype=torch.long, device=device)
        conditional = model(condition, timestep, state)
        outputs = (
            cfg_combine(conditional, model(condition, timestep, state, unconditional=True), cfg)
            if cfg != 1 else conditional
        )
        state["atom_state"] = posterior_step(
            state["atom_state"], outputs["state_logits"], time_index, diffusion,
            model.state_types, mask, False, "atom",
        )
        state["edge_state"] = posterior_step(
            state["edge_state"], outputs["joint_edge_logits"], time_index, diffusion,
            JOINT_EDGE_CLASSES, pair & ~diagonal, True, "joint_edge",
        )
        state["positions"] = reverse_coords(
            state["positions"], outputs["coordinate_eps"], time_index,
            diffusion, mask, coordinate_clip,
        )
        final_outputs = outputs

    projected_states = torch.zeros(batch_size, model.max_atoms, dtype=torch.long)
    projected_edges = torch.zeros(batch_size, model.max_atoms, model.max_atoms, dtype=torch.long)
    projected_bonds = torch.zeros_like(projected_edges)
    projected_aromatic = torch.zeros_like(projected_edges, dtype=torch.bool)
    projected_hydrogens = torch.zeros(batch_size, model.max_atoms, dtype=torch.long)
    valid = torch.zeros(batch_size, dtype=torch.bool)
    solver_used = torch.zeros(batch_size, dtype=torch.bool)
    projection_topology_score = torch.zeros(batch_size, dtype=torch.float32)
    topology_reprojections = torch.zeros(batch_size, dtype=torch.long)
    reasons, statuses = [], []
    for batch_index, count in enumerate(counts.cpu().tolist()):
        state_logits = _condition_logits_on_sample(
            final_outputs["state_logits"][batch_index, :count],
            state["atom_state"][batch_index, :count],
        ).cpu()
        edge_logits = _condition_logits_on_sample(
            final_outputs["joint_edge_logits"][batch_index, :count, :count],
            state["edge_state"][batch_index, :count, :count],
        ).cpu()
        result = project_joint_graph_v7(
            state_logits, edge_logits, model.codec, 0,
            solver_time_limit, solver_attempts, chemistry_reference,
            topology_prior_weight, topology_reproject_threshold,
            projection_seed + batch_index,
        )
        projected_states[batch_index, :count] = result.state_ids
        projected_edges[batch_index, :count, :count] = result.edge_state
        projected_bonds[batch_index, :count, :count] = result.bonds
        projected_aromatic[batch_index, :count, :count] = result.aromatic_edges
        projected_hydrogens[batch_index, :count] = result.hydrogens
        valid[batch_index] = result.valid
        solver_used[batch_index] = result.solver_used
        projection_topology_score[batch_index] = result.topology_score
        topology_reprojections[batch_index] = result.topology_reprojections
        reasons.append(result.reason)
        statuses.append(result.solver_status)
    return {
        "atom_count": counts.cpu(),
        "raw_atom_state": state["atom_state"].cpu(),
        "raw_edge_state": state["edge_state"].cpu(),
        "atom_state": projected_states,
        "joint_edge_state": projected_edges,
        "hydrogen": projected_hydrogens,
        "bond": projected_bonds,
        "aromatic_edge": projected_aromatic,
        "positions": state["positions"].cpu(),
        "projection_valid": valid,
        "projection_reason": reasons,
        "solver_used": solver_used,
        "solver_status": statuses,
        "projection_topology_score": projection_topology_score,
        "topology_reprojections": topology_reprojections,
    }


def molecule_with_coordinates(mol, positions):
    output = Chem.Mol(mol)
    output.RemoveAllConformers()
    conformer = Chem.Conformer(output.GetNumAtoms())
    conformer.Set3D(True)
    for index, xyz in enumerate(torch.as_tensor(positions).tolist()):
        conformer.SetAtomPosition(index, tuple(map(float, xyz)))
    output.AddConformer(conformer, assignId=True)
    return output


def load_known_hp1kg(root):
    known = set()
    for split in ("train", "val", "test"):
        rows = torch.load(Path(root) / f"{split}.pt", map_location="cpu", weights_only=False)
        for row in rows:
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is not None:
                known.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    return known


def evaluate_sample_v7(
    sampled,
    index,
    codec,
    chemistry_reference,
    geometry_reference,
    condition_surrogate,
    requested_s1,
    requested_t1,
    evaluate_3d=False,
    seed=1701,
):
    count = int(sampled["atom_count"][index])
    raw_states = sampled["raw_atom_state"][index, :count]
    raw_edges = sampled["raw_edge_state"][index, :count, :count]
    raw_bonds, raw_aromatic = decode_joint_edge_state(raw_edges)
    raw_decode = decode_heavy_graph(raw_states.tolist(), raw_bonds, codec)
    raw_joint = decode_joint_assignment_v7(raw_states.tolist(), raw_edges, codec)
    projected_decode = None
    if bool(sampled["projection_valid"][index]):
        projected_decode = decode_heavy_graph(
            sampled["atom_state"][index, :count].tolist(),
            sampled["bond"][index, :count, :count],
            codec,
        )
    valid = bool(projected_decode and projected_decode.valid)
    result = {
        "raw_rdkit_valid": int(raw_decode.valid),
        "raw_joint_valid": int(raw_joint[0] is not None),
        "projection_valid": int(valid),
        "projection_error": "" if valid else sampled["projection_reason"][index],
        "solver_used": int(sampled["solver_used"][index]),
        "solver_status": sampled["solver_status"][index],
        "projection_topology_score": float(sampled["projection_topology_score"][index]),
        "topology_reprojections": int(sampled["topology_reprojections"][index]),
    }
    if not valid:
        return result, None
    molecule = projected_decode.molecule
    states = sampled["atom_state"][index, :count].tolist()
    bonds = sampled["bond"][index, :count, :count]
    aromatic = sampled["aromatic_edge"][index, :count, :count]
    support_warnings = support_warnings_v7(states, bonds, aromatic, chemistry_reference)
    topology, chemical_warnings = topology_assessment_v7(molecule)
    prior_score = chemistry_reference.topology_score(count, topology)
    geometry = assess_generated_coordinates(
        molecule, states, bonds, aromatic, sampled["positions"][index, :count],
        codec, geometry_reference, optimize=evaluate_3d,
    )
    reembedded = assess_structure(molecule, (), evaluate_3d, seed + index)
    predicted_s1, predicted_t1 = condition_surrogate.predict(molecule)
    result.update({
        "smiles": projected_decode.smiles,
        "rings": int(rdMolDescriptors.CalcNumRings(molecule)),
        "aromatic_rings": projected_decode.aromatic_rings,
        "support_pass": int(not support_warnings),
        "support_warnings": ";".join(support_warnings),
        "topology_prior_score": prior_score,
        "chemical_clean": int(not chemical_warnings),
        "chemical_warnings": ";".join(chemical_warnings),
        "generated_geometry_finite": int(geometry.finite),
        "generated_bond_z_rms": geometry.bond_z_rms,
        "generated_bond_outliers": geometry.bond_outliers,
        "generated_steric_clashes": geometry.steric_clashes,
        "generated_forcefield": geometry.forcefield,
        "generated_forcefield_converged": int(geometry.forcefield_converged),
        "generated_relaxation_energy_per_heavy_atom": geometry.relaxation_energy_per_heavy_atom,
        "generated_geometry_warnings": ";".join(geometry.warnings),
        "reembedded_etkdg_success": int(reembedded.etkdg_success) if evaluate_3d else -1,
        "reembedded_forcefield_converged": int(reembedded.forcefield_converged) if evaluate_3d else -1,
        "reembedded_relaxation_energy_per_heavy_atom": reembedded.strain_energy_per_heavy_atom,
        "predicted_s1": float(predicted_s1),
        "predicted_t1": float(predicted_t1),
        "s1_absolute_error": abs(float(predicted_s1) - requested_s1),
        "t1_absolute_error": abs(float(predicted_t1) - requested_t1),
    })
    return result, molecule


def summarize_records_v7(records):
    count = max(1, len(records))
    valid = [row for row in records if row.get("projection_valid")]
    clean_count = sum(row.get("chemical_clean", 0) for row in records)
    mean = lambda key, rows: (
        float(np.mean([row[key] for row in rows if row.get(key) is not None]))
        if any(row.get(key) is not None for row in rows) else None
    )
    # Wilson lower bound is less vulnerable than the raw rate when rollout
    # validation uses a small number of expensive reverse samples.
    z = 1.959963984540054
    n = len(records)
    p = clean_count / max(1, n)
    denominator = 1 + z * z / max(1, n)
    clean_lower = (
        p + z * z / (2 * max(1, n))
        - z * math.sqrt((p * (1 - p) + z * z / (4 * max(1, n))) / max(1, n))
    ) / denominator
    return {
        "requested": len(records),
        "raw_rdkit_valid_rate": sum(row["raw_rdkit_valid"] for row in records) / count,
        "raw_joint_valid_rate": sum(row["raw_joint_valid"] for row in records) / count,
        "projection_valid_rate": len(valid) / count,
        "chemical_clean_count": clean_count,
        "chemical_clean_rate": clean_count / count,
        "chemical_clean_wilson_lower_bound": clean_lower,
        "train_support_pass_given_valid_rate": sum(row.get("support_pass", 0) for row in valid) / max(1, len(valid)),
        "chemical_clean_given_valid_rate": sum(row.get("chemical_clean", 0) for row in valid) / max(1, len(valid)),
        "mean_topology_prior_score": mean("topology_prior_score", valid),
        "topology_reprojection_rate": sum(row.get("topology_reprojections", 0) > 0 for row in records) / count,
        "mean_topology_reprojections": mean("topology_reprojections", records),
        "generated_geometry_finite_rate": sum(row.get("generated_geometry_finite", 0) for row in valid) / max(1, len(valid)),
        "generated_clash_free_rate": sum(row.get("generated_steric_clashes") == 0 for row in valid) / max(1, len(valid)),
        "mean_generated_bond_z_rms": mean("generated_bond_z_rms", valid),
        "mean_generated_relaxation_energy_per_heavy_atom": mean("generated_relaxation_energy_per_heavy_atom", valid),
        "reembedded_etkdg_success_rate": (
            sum(row.get("reembedded_etkdg_success") == 1 for row in valid) / max(1, len(valid))
            if any(row.get("reembedded_etkdg_success", -1) >= 0 for row in valid) else None
        ),
        "condition_s1_mae": mean("s1_absolute_error", valid),
        "condition_t1_mae": mean("t1_absolute_error", valid),
        "projection_failures": dict(collections.Counter(
            row["projection_error"] for row in records if row["projection_error"]
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root", default="joint_kekule_diffusion/data_hp1kg_nevpt2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=0)
    parser.add_argument("--s1", type=float, default=3.640037775)
    parser.add_argument("--t1", type=float, default=3.111467600)
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--coordinate-clip", type=float, default=25.)
    parser.add_argument("--solver-time-limit", type=float, default=8.)
    parser.add_argument("--solver-attempts", type=int, default=4)
    parser.add_argument("--topology-prior-weight", type=float)
    parser.add_argument("--topology-reproject-threshold", type=float)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--evaluate-3d", action="store_true")
    args = parser.parse_args()
    if args.num <= 0 or args.batch_size <= 0 or args.atoms < 0:
        raise ValueError("invalid generation size")
    if args.solver_time_limit <= 0 or args.solver_attempts <= 0:
        raise ValueError("solver time limit and attempts must be positive")
    if args.topology_prior_weight is not None and args.topology_prior_weight < 0:
        raise ValueError("topology prior weight must be non-negative")
    if (args.topology_reproject_threshold is not None
            and not 0 <= args.topology_reproject_threshold <= 1):
        raise ValueError("topology reproject threshold must be in [0,1]")
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text())
    if metadata.get("model_kind") != "heavy_atom_joint_edge_d3pm_v7_topology_regularized":
        raise ValueError("metadata is not topology-regularized v7 joint-edge diffusion")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_v7_artifacts(metadata, checkpoint, args.root)
    codec = AtomStateCodec.from_metadata(metadata["atom_states"])
    chemistry_reference = ChemistryReferenceV7.from_metadata(metadata["chemistry_reference"])
    geometry_reference = GeometryReferenceV7.from_metadata(metadata["geometry_reference"])
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
    condition = torch.tensor([[
        (args.s1 - stats["s1_energy_mean"]) / stats["s1_energy_std"],
        (args.t1 - stats["t1_energy_mean"]) / stats["t1_energy_std"],
    ]], device="cuda")
    projection = metadata["topology_projection"]
    topology_prior_weight = (
        projection["prior_weight"] if args.topology_prior_weight is None
        else args.topology_prior_weight
    )
    topology_reproject_threshold = (
        projection["reproject_threshold"] if args.topology_reproject_threshold is None
        else args.topology_reproject_threshold
    )
    known = load_known_hp1kg(args.root)
    records, unique_smiles, unique_molecules = [], set(), []
    started = time.time()
    for begin in range(0, args.num, args.batch_size):
        size = min(args.batch_size, args.num - begin)
        sampled = sample_batch_v7(
            model, diffusion, condition.expand(size, -1), size, args.atoms, args.cfg,
            metadata["fixed_atom_count"], args.coordinate_clip,
            args.solver_time_limit, args.solver_attempts,
            chemistry_reference, topology_prior_weight,
            topology_reproject_threshold, args.seed + begin,
        )
        for row_index in range(size):
            item, molecule = evaluate_sample_v7(
                sampled, row_index, codec, chemistry_reference, geometry_reference,
                surrogate, args.s1, args.t1, args.evaluate_3d, args.seed + begin,
            )
            smiles = item.get("smiles", "")
            novel = bool(smiles and smiles not in known)
            unique = bool(novel and smiles not in unique_smiles)
            item.update({
                "index": begin + row_index,
                "atom_count": int(sampled["atom_count"][row_index]),
                "novel_vs_db": int(novel),
                "unique_novel": int(unique),
            })
            records.append(item)
            if unique:
                unique_smiles.add(smiles)
                unique_molecules.append(molecule_with_coordinates(
                    molecule, sampled["positions"][row_index, :item["atom_count"]],
                ))
        with (output_dir / "molecules.csv").open("w", newline="") as handle:
            fields = sorted({key for row in records for key in row})
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({
            "event": "generation", "completed": len(records),
            "raw_valid": sum(row["raw_rdkit_valid"] for row in records),
            "projected_valid": sum(row["projection_valid"] for row in records),
        }), flush=True)

    summary = summarize_records_v7(records)
    summary.update({
        "checkpoint_epoch": checkpoint["epoch"],
        "s1": args.s1, "t1": args.t1, "cfg": args.cfg, "atoms": args.atoms,
        "novel_vs_db": sum(row["novel_vs_db"] for row in records),
        "unique_novel": len(unique_molecules),
        "surrogate_validation": surrogate.validation_metrics,
        "metadata_hash": metadata["metadata_hash"],
        "dataset_hash": metadata["dataset_hash"],
        "elapsed_s": time.time() - started,
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    writer = Chem.SDWriter(str(output_dir / "unique_novel_connected.sdf"))
    for index, molecule in enumerate(unique_molecules):
        molecule.SetProp("_Name", f"generated_{index:06d}")
        molecule.SetProp("SMILES", Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
        writer.write(molecule)
    writer.close()
    for page_start in range(0, len(unique_molecules), 100):
        page, legends = [], []
        for molecule in unique_molecules[page_start:page_start + 100]:
            depicted = Chem.Mol(molecule)
            depicted.RemoveAllConformers()
            rdDepictor.Compute2DCoords(depicted, canonOrient=True, clearConfs=True)
            page.append(depicted)
            legends.append(Chem.MolToSmiles(depicted, canonical=True, isomericSmiles=False))
        Draw.MolsToGridImage(
            page, molsPerRow=10, subImgSize=(300, 240), legends=legends,
        ).save(str(output_dir / f"novel_valid_{page_start // 100:03d}.png"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "sample_batch_v7", "evaluate_sample_v7", "summarize_records_v7",
    "validate_v7_artifacts", "main",
]

