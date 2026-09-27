#!/usr/bin/env python3
"""Trainer for v5 heavy-atom joint-state diffusion."""
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

from .chemistry_v5 import (
    build_hp1kg_codec,
    build_hp1kg_topology_prior,
    collate_hp1kg_v5,
    decode_heavy_graph,
)
from .hp1kg_data import HP1KGDataset
from .model import JointDiffusion
from .model_v5 import HeavyAtomJointDDPM, joint_training_loss_v5
from .sample_v5 import sample_batch_v5


def compute_marginals(dataset, codec):
    counts = {
        "atom": torch.zeros(len(codec)),
        "edge": torch.zeros(2),
        "bond_type": torch.zeros(3),
    }
    for index in range(len(dataset)):
        row = dataset[index]
        state = codec.encode_record(row, dataset.atom_map)
        bonds = row["kekule_bonds"].long()
        count = len(state)
        upper = torch.triu(torch.ones(count, count, dtype=torch.bool), 1)
        present = bonds[upper] > 0
        counts["atom"] += torch.bincount(state, minlength=len(codec))
        counts["edge"] += torch.tensor([int((~present).sum()), int(present.sum())])
        counts["bond_type"] += torch.bincount((bonds[upper][present] - 1).clamp_min(0), minlength=3)
    return {name: ((value + 1e-6) / (value + 1e-6).sum()).tolist() for name, value in counts.items()}


def device_batch(batch, device):
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def validate(model, diffusion, loader, device, max_batches=8, seed=1701):
    model.eval()
    totals = {}
    count = 0
    probe = None
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for batch in loader:
            batch = device_batch(batch, device)
            probe = batch if probe is None else probe
            loss, terms = joint_training_loss_v5(model, diffusion, batch)
            if not torch.isfinite(loss):
                model.train()
                return {"loss": float("nan")}
            for name, value in terms.items():
                totals[name] = totals.get(name, 0.) + float(value)
            count += 1
            if count >= max_batches:
                break
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 1)
        correct = float(joint_training_loss_v5(model, diffusion, probe)[0])
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed_all(seed + 1)
        shuffled = {**probe, "condition": probe["condition"].roll(1, 0)}
        shuffled_loss = float(joint_training_loss_v5(model, diffusion, shuffled)[0])
    model.train()
    output = {name: value / max(count, 1) for name, value in totals.items()}
    output["loss"] = sum(output.values())
    output["condition_shuffle_delta"] = shuffled_loss - correct
    return output


@torch.no_grad()
def rollout_validate(model, diffusion, loader, device, topology_prior, observed_counts, coordinate_clip, samples=4, seed=901):
    model.eval()
    batch = device_batch(next(iter(loader)), device)
    count = min(samples, len(batch["condition"]))
    sampled = sample_batch_v5(
        model, diffusion, batch["condition"][:count], count, 0, 1., topology_prior,
        observed_counts, coordinate_clip, random.Random(seed),
    )
    valid = aromatic = 0
    failures = {}
    for index in range(count):
        atom_count = int(sampled["atom_count"][index])
        if not bool(sampled["projection_valid"][index]):
            reason = sampled["projection_reason"][index]
            failures[reason] = failures.get(reason, 0) + 1
            continue
        decoded = decode_heavy_graph(
            sampled["atom_state"][index, :atom_count].tolist(),
            sampled["bond"][index, :atom_count, :atom_count], model.codec,
        )
        if not decoded.valid:
            failures[decoded.error] = failures.get(decoded.error, 0) + 1
            continue
        valid += 1
        aromatic += int(decoded.aromatic_rings > 0)
    model.train()
    return {
        "samples": count,
        "strict_valid_rate": valid / max(count, 1),
        "strict_closed_shell_neutral_rate": valid / max(count, 1),
        "rdkit_aromatic_rate": aromatic / max(count, 1),
        "failure_reasons": failures,
    }


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="joint_kekule_diffusion/data_hp1kg_nevpt2")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=208)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
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
    parser.add_argument("--max-atoms", type=int, default=0)
    parser.add_argument("--rollout-every", type=int, default=10)
    parser.add_argument("--rollout-samples", type=int, default=4)
    parser.add_argument("--coordinate-clip", type=float, default=25.)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0 or args.hidden % 2:
        raise ValueError("invalid training configuration")
    if not 0 <= args.condition_dropout < 1 or args.rollout_every <= 0 or args.rollout_samples <= 0:
        raise ValueError("invalid condition dropout or rollout configuration")

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
        raise ValueError("v5 currently requires the HP1KG heavy-atom cache")
    codec = build_hp1kg_codec(args.root)
    topology_prior = build_hp1kg_topology_prior(args.root)
    train = HP1KGDataset(args.root, "train")
    validation = HP1KGDataset(args.root, "val", condition_stats=train.condition_stats, atom_map=train.atom_map)
    observed_max = max(train.heavy_count_at(index) for index in range(len(train)))
    max_atoms = args.max_atoms or int(manifest.get("heavy_atom_max", observed_max))
    if max_atoms < observed_max:
        raise ValueError(f"max_atoms={max_atoms} below observed training maximum {observed_max}")
    collate = partial(collate_hp1kg_v5, max_atoms=max_atoms, codec=codec, atom_map=train.atom_map)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.workers // 2), pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate)
    marginals = compute_marginals(train, codec)
    model = HeavyAtomJointDDPM(
        codec, args.hidden, max_atoms, args.layers, args.condition_dropout,
        args.s1_rbf_bins, args.t1_rbf_bins, args.condition_rbf_min,
        args.condition_rbf_max, args.condition_rbf_width_scale,
    ).to(device)
    diffusion = JointDiffusion(marginals=marginals).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 0
    step = 0
    observed_counts = sorted({train.heavy_count_at(index) for index in range(len(train))})
    metadata = {
        "schema_version": 7,
        "model_kind": "heavy_atom_joint_state_d3pm_v5",
        "args": vars(args),
        "diffusion_marginals": marginals,
        "atom_states": codec.to_metadata(),
        "topology_prior": topology_prior.to_metadata(),
        "atom_state_schema_splits": ["train", "val", "test"],
        "topology_prior_split": "train",
        "condition_stats": train.condition_stats,
        "atom_map": list(train.atom_map),
        "max_atoms": max_atoms,
        "fixed_atom_count": observed_counts,
        "train_size": len(train),
        "val_size": len(validation),
        "pid": os.getpid(),
        "hydrogen_policy": "residual_from_joint_state_and_final_bond_graph",
        "radical_policy": "strictly_forbidden",
        "net_charge_policy": "zero",
    }
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        old = checkpoint.get("metadata", {})
        if old.get("model_kind") != metadata["model_kind"] or old.get("atom_states") != metadata["atom_states"]:
            raise ValueError("resume checkpoint is incompatible with v5 atom-state vocabulary")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint["step"]
        if "rng" in checkpoint:
            random.setstate(checkpoint["rng"]["python"])
            np.random.set_state(checkpoint["rng"]["numpy"])
            torch.set_rng_state(checkpoint["rng"]["torch"])
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    model.train()
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        totals = {}
        batches = 0
        for batch in loader:
            batch = device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                loss, terms = joint_training_loss_v5(model, diffusion, batch)
            if not torch.isfinite(loss):
                atomic_save({"epoch": epoch, "step": step, "model": model.state_dict(), "batch": {k: v.cpu() for k, v in batch.items()}}, run_dir / "nan_checkpoint.pt")
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
                print(json.dumps({"event": "step", "epoch": epoch, "step": step, "loss": float(loss), "grad_norm": float(gradient_norm), "gpu_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30, "elapsed_s": time.time() - started}), flush=True)
        scheduler.step()
        train_metrics = {name: value / max(batches, 1) for name, value in totals.items()}
        validation_metrics = validate(model, diffusion, validation_loader, device)
        rollout = None
        if (epoch + 1) % args.rollout_every == 0:
            try:
                rollout = rollout_validate(model, diffusion, validation_loader, device, topology_prior, observed_counts, args.coordinate_clip, args.rollout_samples)
            except Exception as exc:
                model.train()
                rollout = {"failed": True, "error": f"{type(exc).__name__}: {exc}"}
        event = {"event": "epoch", "epoch": epoch, "step": step, "train": train_metrics, "val": validation_metrics, "rollout": rollout, "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started}
        print(json.dumps(event), flush=True)
        payload = {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metadata": metadata,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        }
        atomic_save(payload, run_dir / "checkpoints" / "latest.pt")
        if (epoch + 1) % args.save_every == 0:
            atomic_save(payload, run_dir / "checkpoints" / f"epoch_{epoch + 1:05d}.pt")


if __name__ == "__main__":
    main()


__all__ = ["compute_marginals", "validate", "rollout_validate", "main"]
