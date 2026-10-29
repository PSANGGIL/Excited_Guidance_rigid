#!/usr/bin/env python3
"""Strict-FP32 optimizer smoke gate for the production V8 configuration."""
from __future__ import annotations

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from joint_kekule_diffusion.chemistry_v8 import (
    HP1KGV8Dataset,
    build_train_codec_v8,
    collate_hp1kg_v8,
    dataset_fingerprint,
    runtime_versions_v8,
    source_fingerprints_v8,
)
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v8 import (
    HeavyAtomJointDDPMV8,
    joint_training_loss_v8,
)
from joint_kekule_diffusion.train_v8 import compute_marginals_v8, device_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default="joint_kekule_diffusion/data_hp1kg_nevpt2",
    )
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument(
        "--output", default="validation/v8_gpu_smoke_report.json",
    )
    args = parser.parse_args()
    if min(args.batch_size, args.steps, args.hidden, args.layers) <= 0:
        raise ValueError("all numeric smoke-test arguments must be positive")
    if args.hidden % 2:
        raise ValueError("hidden width must be even")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the production smoke gate")

    torch.manual_seed(20260925)
    torch.cuda.manual_seed_all(20260925)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda")
    codec = build_train_codec_v8(args.root)
    dataset = HP1KGV8Dataset(args.root, "train")
    max_atoms = max(dataset.heavy_count_at(index) for index in range(len(dataset)))
    collate = partial(
        collate_hp1kg_v8, max_atoms=max_atoms, codec=codec,
        atom_map=dataset.atom_map,
    )
    generator = torch.Generator().manual_seed(20260925)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0, collate_fn=collate, drop_last=True,
    )
    marginals = compute_marginals_v8(dataset, codec)
    diffusion = JointDiffusion(
        timesteps=500, beta_start=1e-4, beta_end=.05,
        marginals=marginals,
    ).to(device)
    model = HeavyAtomJointDDPMV8(
        codec, hidden=args.hidden, max_atoms=max_atoms, layers=args.layers,
        condition_dropout=.1, s1_rbf_bins=24, t1_rbf_bins=24,
        condition_rbf_min=-6., condition_rbf_max=6.,
        condition_rbf_width_scale=1., model_dropout=.1, count_dropout=.2,
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_rows = []
    iterator = iter(loader)
    required_gradients = (
        "state_head.weight", "joint_edge_head.weight",
        "ring_closure_head.weight", "ring_size_head.weight",
        "coordinate_head.3.weight", "count_head.3.weight",
    )
    for step in range(args.steps):
        step_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = device_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = joint_training_loss_v8(model, diffusion, batch)
        if not torch.isfinite(loss) or not all(
            torch.isfinite(value) for value in terms.values()
        ):
            raise FloatingPointError(f"non-finite forward value at step {step}")
        loss.backward()
        named = dict(model.named_parameters())
        missing = [
            name for name in required_gradients
            if name not in named or named[name].grad is None
            or not torch.isfinite(named[name].grad).all()
        ]
        if missing:
            raise AssertionError(f"missing/non-finite required gradients: {missing}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        if not torch.isfinite(gradient_norm) or float(gradient_norm) <= 0:
            raise FloatingPointError(f"invalid gradient norm at step {step}")
        optimizer.step()
        torch.cuda.synchronize()
        step_rows.append({
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "elapsed_s": time.perf_counter() - step_started,
            "largest_term": max(
                ((name, float(value.detach())) for name, value in terms.items()),
                key=lambda item: item[1],
            ),
            "all_terms": {name: float(value.detach()) for name, value in terms.items()},
        })

    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    report = {
        "passed": True,
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "precision": "fp32",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "hidden": args.hidden,
        "layers": args.layers,
        "max_atoms": max_atoms,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset_hash": dataset_fingerprint(args.root),
        "source_hashes": source_fingerprints_v8(),
        "runtime_versions": runtime_versions_v8(),
        "peak_allocated_gib": allocated / 2 ** 30,
        "peak_reserved_gib": reserved / 2 ** 30,
        "total_memory_gib": total_memory / 2 ** 30,
        "peak_allocated_fraction": allocated / total_memory,
        "steps_detail": step_rows,
    }
    assert not report["tf32_matmul"] and not report["tf32_cudnn"]
    assert all(math.isfinite(row["loss"]) for row in step_rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
