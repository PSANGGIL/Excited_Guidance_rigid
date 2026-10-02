#!/usr/bin/env python3
"""Build reproducible fixed-EST targets paired with training atom counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-mols", type=int, required=True)
    parser.add_argument("--target-est", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    if args.n_mols <= 0:
        raise ValueError("--n-mols must be positive")

    records = torch.load(args.train_data, map_location="cpu", weights_only=False)
    if not records:
        raise ValueError("training data is empty")

    rng = np.random.default_rng(args.seed)
    source_indices = rng.integers(0, len(records), size=args.n_mols)
    targets = np.full(args.n_mols, args.target_est, dtype=np.float32)
    atom_counts = np.asarray(
        [len(records[int(index)]["atom_types"]) for index in source_indices],
        dtype=np.int64,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.out_dir / "targets_est_fixed.npy"
    count_path = args.out_dir / "atom_counts_train_bootstrap.npy"
    manifest_path = args.out_dir / "condition_manifest.csv"
    np.save(target_path, targets)
    np.save(count_path, atom_counts)

    fieldnames = [
        "generation_index",
        "condition_name",
        "condition_range",
        "target_value",
        "target_est_eV",
        "requested_atom_count",
        "source_train_index",
        "source_id",
        "source_smiles",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for generation_index, source_index in enumerate(source_indices):
            record = records[int(source_index)]
            writer.writerow(
                {
                    "generation_index": generation_index,
                    "condition_name": "est",
                    "condition_range": f"fixed_{args.target_est:.6f}",
                    "target_value": args.target_est,
                    "target_est_eV": args.target_est,
                    "requested_atom_count": len(record["atom_types"]),
                    "source_train_index": int(source_index),
                    "source_id": str(record.get("id", source_index)),
                    "source_smiles": str(record.get("smiles", "")),
                }
            )

    unique_counts, frequencies = np.unique(atom_counts, return_counts=True)
    summary = {
        "schema_version": 1,
        "condition": "est",
        "definition": "S1 - T1",
        "target_est_eV": args.target_est,
        "n_mols": args.n_mols,
        "seed": args.seed,
        "atom_count_sampling": "bootstrap with replacement from training records",
        "source_train_data": str(args.train_data.resolve()),
        "source_train_count": len(records),
        "atom_count_min": int(atom_counts.min()),
        "atom_count_max": int(atom_counts.max()),
        "atom_count_mean": float(atom_counts.mean()),
        "atom_count_histogram": {
            str(int(count)): int(frequency)
            for count, frequency in zip(unique_counts, frequencies)
        },
        "target_file": str(target_path.resolve()),
        "atom_count_file": str(count_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    (args.out_dir / "condition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
