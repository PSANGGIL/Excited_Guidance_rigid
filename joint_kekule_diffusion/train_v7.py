#!/usr/bin/env python3
"""Trainer for v7 joint atom-state/joint-edge/Cartesian diffusion."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .chemistry_v5 import hp1kg_records
from .chemistry_v7 import (
    HP1KGV7Dataset,
    build_train_codec_v7,
    build_v7_references,
    collate_hp1kg_v7,
    dataset_fingerprint,
    metadata_fingerprint,
    source_fingerprints_v7,
)
from .model import JointDiffusion
from .model_v7 import (
    JOINT_EDGE_CLASSES,
    HeavyAtomJointDDPMV7,
    encode_joint_edge_state,
    joint_training_loss_v7,
)
from .sample_v7 import evaluate_sample_v7, sample_batch_v7, summarize_records_v7


def compute_marginals_v7(dataset, codec):
    atom = torch.zeros(len(codec), dtype=torch.float64)
    joint_edge = torch.zeros(JOINT_EDGE_CLASSES, dtype=torch.float64)
    for index in range(len(dataset)):
        row = dataset[index]
        states = codec.encode_record(row, dataset.atom_map)
        edges = encode_joint_edge_state(row["kekule_bonds"], row["aromatic_edges"])
        count = len(states)
        upper = torch.triu(torch.ones(count, count, dtype=torch.bool), 1)
        atom += torch.bincount(states, minlength=len(codec))
        joint_edge += torch.bincount(edges[upper], minlength=JOINT_EDGE_CLASSES)
    return {
        "atom": ((atom + 1e-6) / (atom + 1e-6).sum()).float().tolist(),
        "joint_edge": ((joint_edge + 1e-6) / (joint_edge + 1e-6).sum()).float().tolist(),
    }


def device_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def validate_v7(model, diffusion, loader, device, max_batches=8, seed=1701):
    model.eval()
    totals, batches, probe = {}, 0, None
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for batch in loader:
            batch = device_batch(batch, device)
            probe = batch if probe is None else probe
            loss, terms = joint_training_loss_v7(model, diffusion, batch)
            if not torch.isfinite(loss):
                model.train()
                return {"loss": float("nan")}
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.) + float(value)
            batches += 1
            if batches >= max_batches:
                break
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 1)
        correct = float(joint_training_loss_v7(model, diffusion, probe)[0])
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 1)
        shuffled = {**probe, "condition": probe["condition"].roll(1, 0)}
        shuffled_loss = float(joint_training_loss_v7(model, diffusion, shuffled)[0])
    model.train()
    output = {name: value / max(1, batches) for name, value in totals.items()}
    output["loss"] = sum(output.values())
    output["condition_shuffle_delta"] = shuffled_loss - correct
    return output


@torch.no_grad()
def rollout_validate_v7(
    model,
    diffusion,
    loader,
    device,
    chemistry_reference,
    geometry_reference,
    condition_surrogate,
    condition_stats,
    observed_counts,
    coordinate_clip,
    samples=16,
    seed=901,
    evaluate_3d=False,
    topology_prior_weight=1.5,
    topology_reproject_threshold=.2,
):
    model.eval()
    batch = device_batch(next(iter(loader)), device)
    count = min(samples, len(batch["condition"]))
    sampled = sample_batch_v7(
        model, diffusion, batch["condition"][:count], count, 0, 1.,
        observed_counts, coordinate_clip, 2., 2,
        chemistry_reference, topology_prior_weight,
        topology_reproject_threshold, seed,
    )
    records = []
    for index in range(count):
        raw_condition = batch["condition"][index].detach().cpu()
        requested_s1 = float(raw_condition[0] * condition_stats["s1_energy_std"] + condition_stats["s1_energy_mean"])
        requested_t1 = float(raw_condition[1] * condition_stats["t1_energy_std"] + condition_stats["t1_energy_mean"])
        record, _ = evaluate_sample_v7(
            sampled, index, model.codec, chemistry_reference, geometry_reference,
            condition_surrogate, requested_s1, requested_t1, evaluate_3d, seed,
        )
        records.append(record)
    output = summarize_records_v7(records)
    output["samples"] = count
    model.train()
    return output


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def build_metadata(args, manifest, codec, references, marginals, train, validation, max_atoms):
    chemistry, geometry, surrogate = references
    model_args = {
        key: getattr(args, key)
        for key in (
            "hidden", "layers", "condition_dropout", "model_dropout", "count_dropout",
            "s1_rbf_bins", "t1_rbf_bins", "condition_rbf_min",
            "condition_rbf_max", "condition_rbf_width_scale",
        )
    }
    metadata = {
        "schema_version": 10,
        "model_kind": "heavy_atom_joint_edge_d3pm_v7_topology_regularized",
        "model_args": model_args,
        "diffusion": {"timesteps": 500, "beta_start": 1e-4, "beta_end": .05},
        "diffusion_marginals": marginals,
        "atom_states": codec.to_metadata(),
        "atom_state_schema_splits": ["train"],
        "chemistry_reference": chemistry.to_metadata(),
        "chemistry_support_splits": ["train"],
        "geometry_reference": geometry.to_metadata(),
        "geometry_reference_splits": ["train"],
        "condition_surrogate": surrogate.to_metadata(),
        "condition_surrogate_train_split": "train",
        "condition_surrogate_validation_split": "val",
        "condition_stats": train.condition_stats,
        "condition_stats_split": "train",
        "atom_map": list(train.atom_map),
        "max_atoms": max_atoms,
        "fixed_atom_count": sorted({train.heavy_count_at(i) for i in range(len(train))}),
        "train_size": len(train),
        "val_size": len(validation),
        "dataset_hash": dataset_fingerprint(args.root),
        "source_hashes": source_fingerprints_v7(),
        "dataset_manifest": manifest,
        "hydrogen_policy": "implicit_residual_from_joint_atom_state_and_final_bonds",
        "edge_policy": "single_7_class_state_none_regular123_aromatic123",
        "reverse_policy": "exact_d3pm_no_intermediate_projection_final_joint_milp_only",
        "projection_policy": "joint_atom_state_edge_valence_charge_aromatic_connectivity_topology_soft_milp",
        "topology_loss_policy": "adjacency_3_4_cycle_cycle_rank_soft_2core_ring_system_bridgehead",
        "topology_projection": {
            "prior_weight": args.topology_prior_weight,
            "reproject_threshold": args.topology_reproject_threshold,
            "hard_topology_cap": False,
        },
        "metric_policy": "raw_projection_support_prior_clean_overall_wilson_generated3d_reembedded3d_separate",
        "stereochemistry_policy": "not_modelled_dataset_scope",
    }
    metadata["metadata_hash"] = metadata_fingerprint(metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="joint_kekule_diffusion/data_hp1kg_nevpt2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=208)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--s1-rbf-bins", type=int, default=24)
    parser.add_argument("--t1-rbf-bins", type=int, default=24)
    parser.add_argument("--condition-rbf-min", type=float, default=-5.)
    parser.add_argument("--condition-rbf-max", type=float, default=5.)
    parser.add_argument("--condition-rbf-width-scale", type=float, default=1.)
    parser.add_argument("--condition-dropout", type=float, default=.1)
    parser.add_argument("--model-dropout", type=float, default=.1)
    parser.add_argument("--count-dropout", type=float, default=.2)
    parser.add_argument("--topology-prior-weight", type=float, default=1.5)
    parser.add_argument("--topology-reproject-threshold", type=float, default=.2)
    parser.add_argument("--max-atoms", type=int, default=0)
    parser.add_argument("--rollout-every", type=int, default=10)
    parser.add_argument("--rollout-samples", type=int, default=16)
    parser.add_argument("--rollout-3d", action="store_true")
    parser.add_argument("--coordinate-clip", type=float, default=25.)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0 or args.hidden % 2:
        raise ValueError("invalid training configuration")
    if not all(0 <= value < 1 for value in (
        args.condition_dropout, args.model_dropout, args.count_dropout,
    )) or args.rollout_every <= 0 or args.rollout_samples <= 0:
        raise ValueError("invalid dropout or rollout configuration")
    if args.topology_prior_weight < 0 or not 0 <= args.topology_reproject_threshold <= 1:
        raise ValueError("invalid topology projection configuration")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    run_dir = Path(args.run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.root) / "manifest.json").read_text())
    if manifest.get("dataset_type") != "hp1kg_nevpt2":
        raise ValueError("v7 requires the HP1KG NEVPT2 heavy-atom cache")

    codec = build_train_codec_v7(args.root)
    chemistry, geometry, surrogate = build_v7_references(args.root, codec)
    train = HP1KGV7Dataset(args.root, "train")
    validation = HP1KGV7Dataset(
        args.root, "val", condition_stats=train.condition_stats, atom_map=train.atom_map,
    )
    observed_max = max(train.heavy_count_at(i) for i in range(len(train)))
    max_atoms = args.max_atoms or int(manifest.get("heavy_atom_max", observed_max))
    if max_atoms < observed_max:
        raise ValueError(f"max_atoms={max_atoms} below training maximum {observed_max}")
    collate = partial(
        collate_hp1kg_v7, max_atoms=max_atoms, codec=codec, atom_map=train.atom_map,
    )
    loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate,
    )
    val_workers = max(1, args.workers // 2) if args.workers else 0
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=val_workers,
        pin_memory=True, persistent_workers=val_workers > 0, collate_fn=collate,
    )
    marginals = compute_marginals_v7(train, codec)
    model = HeavyAtomJointDDPMV7(
        codec, args.hidden, max_atoms, args.layers, args.condition_dropout,
        args.s1_rbf_bins, args.t1_rbf_bins, args.condition_rbf_min,
        args.condition_rbf_max, args.condition_rbf_width_scale,
        args.model_dropout, args.count_dropout,
    ).to(device)
    diffusion = JointDiffusion(marginals=marginals).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    metadata = build_metadata(
        args, manifest, codec, (chemistry, geometry, surrogate),
        marginals, train, validation, max_atoms,
    )
    start_epoch = step = 0
    best_checkpoint_score = (-1., -1., -1., -1., -1.)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("metadata_hash") != metadata["metadata_hash"]:
            raise ValueError("resume checkpoint metadata/DB/config hash mismatch")
        if metadata_fingerprint(checkpoint["metadata"]) != metadata["metadata_hash"]:
            raise ValueError("embedded resume metadata hash mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
        best_checkpoint_score = tuple(checkpoint.get(
            "best_checkpoint_score", best_checkpoint_score,
        ))
        rng = checkpoint.get("rng")
        if rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state_all(rng["cuda"])
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    model.train()
    started = time.time()
    observed_counts = metadata["fixed_atom_count"]
    for epoch in range(start_epoch, args.epochs):
        totals, batches = {}, 0
        for batch in loader:
            batch = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                loss, terms = joint_training_loss_v7(model, diffusion, batch)
            if not torch.isfinite(loss):
                atomic_save({
                    "epoch": epoch, "step": step, "model": model.state_dict(),
                    "batch": {key: value.cpu() for key, value in batch.items()},
                    "metadata": metadata, "metadata_hash": metadata["metadata_hash"],
                }, run_dir / "nan_checkpoint.pt")
                raise FloatingPointError(f"non-finite loss epoch={epoch} step={step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient epoch={epoch} step={step}")
            optimizer.step()
            step += 1
            batches += 1
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.) + float(value.detach())
            if step % 20 == 0:
                print(json.dumps({
                    "event": "step", "epoch": epoch, "step": step,
                    "loss": float(loss), "grad_norm": float(gradient_norm),
                    "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                    "elapsed_s": time.time() - started,
                }), flush=True)
        scheduler.step()
        train_metrics = {name: value / max(1, batches) for name, value in totals.items()}
        validation_metrics = validate_v7(model, diffusion, validation_loader, device)
        rollout = None
        if (epoch + 1) % args.rollout_every == 0:
            try:
                rollout = rollout_validate_v7(
                    model, diffusion, validation_loader, device, chemistry, geometry,
                    surrogate, train.condition_stats, observed_counts,
                    args.coordinate_clip, args.rollout_samples, 901 + epoch,
                    args.rollout_3d, args.topology_prior_weight,
                    args.topology_reproject_threshold,
                )
            except Exception as exc:
                model.train()
                rollout = {"failed": True, "error": f"{type(exc).__name__}: {exc}"}
        candidate_score = None
        is_best_clean = False
        if rollout and not rollout.get("failed"):
            candidate_score = (
                float(rollout.get("chemical_clean_wilson_lower_bound", 0.)),
                float(rollout.get("chemical_clean_rate", 0.)),
                float(rollout.get("projection_valid_rate", 0.)),
                float(rollout.get("raw_joint_valid_rate", 0.)),
                float(rollout.get("mean_topology_prior_score") or 0.),
            )
            if candidate_score > best_checkpoint_score:
                best_checkpoint_score = candidate_score
                is_best_clean = True
        event = {
            "event": "epoch", "epoch": epoch, "step": step,
            "train": train_metrics, "val": validation_metrics, "rollout": rollout,
            "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
            "checkpoint_selection": {
                "criterion": "chemical_clean_wilson_then_clean_projection_raw_joint_prior",
                "candidate": candidate_score,
                "best": best_checkpoint_score,
                "is_best": is_best_clean,
            },
        }
        print(json.dumps(event), flush=True)
        payload = {
            "epoch": epoch, "step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "metadata": metadata, "metadata_hash": metadata["metadata_hash"],
            "dataset_hash": metadata["dataset_hash"],
            "evaluation": {"validation": validation_metrics, "rollout": rollout},
            "best_checkpoint_score": best_checkpoint_score,
            "checkpoint_selection_criterion": "chemical_clean_wilson_then_clean_projection_raw_joint_prior",
            "rng": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            },
        }
        atomic_save(payload, run_dir / "checkpoints" / "latest.pt")
        if is_best_clean:
            atomic_save(payload, run_dir / "checkpoints" / "best_chemical_clean.pt")
        if (epoch + 1) % args.save_every == 0:
            atomic_save(payload, run_dir / "checkpoints" / f"epoch_{epoch + 1:05d}.pt")


if __name__ == "__main__":
    main()


__all__ = [
    "compute_marginals_v7", "validate_v7", "rollout_validate_v7",
    "build_metadata", "main",
]

