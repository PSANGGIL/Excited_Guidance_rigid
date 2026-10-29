#!/usr/bin/env python3
"""Build a paired S1/atom-count benchmark stratified by training-set rank."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


RANGE_NAMES = ("Q1_low", "Q2_low_mid", "Q3_mid", "Q4_high_mid", "Q5_high")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-range", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()

    records = torch.load(args.train_data, map_location="cpu", weights_only=False)
    if len(records) < args.per_range * len(RANGE_NAMES):
        raise ValueError("training set is too small for sampling without replacement")
    values = np.asarray([float(record["s1"]) for record in records], dtype=np.float64)
    order = np.argsort(values, kind="stable")
    rank_groups = np.array_split(order, len(RANGE_NAMES))
    rng = np.random.default_rng(args.seed)

    selected: list[dict[str, object]] = []
    band_summaries: list[dict[str, object]] = []
    for band_index, (name, group) in enumerate(zip(RANGE_NAMES, rank_groups), start=1):
        chosen = rng.choice(group, size=args.per_range, replace=False)
        band_values = values[group]
        band_summaries.append(
            {
                "band_index": band_index,
                "range_name": name,
                "database_count": int(len(group)),
                "database_min_eV": float(band_values.min()),
                "database_max_eV": float(band_values.max()),
                "database_mean_eV": float(band_values.mean()),
                "selected_count": args.per_range,
            }
        )
        for source_index in chosen:
            record = records[int(source_index)]
            selected.append(
                {
                    "band_index": band_index,
                    "s1_range": name,
                    "target_s1_eV": float(record["s1"]),
                    "requested_atom_count": int(len(record["atom_types"])),
                    "source_train_index": int(source_index),
                    "source_id": str(record.get("id", source_index)),
                    "source_smiles": str(record.get("smiles", "")),
                }
            )

    # Mix ranges so no method gets a favorable/unfavorable range only because
    # it happened to be sampled earlier in a stochastic trajectory.
    permutation = rng.permutation(len(selected))
    selected = [selected[int(index)] for index in permutation]
    for generation_index, row in enumerate(selected):
        row["generation_index"] = generation_index

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray([row["target_s1_eV"] for row in selected], dtype=np.float32)
    atom_counts = np.asarray(
        [row["requested_atom_count"] for row in selected], dtype=np.int64
    )
    target_path = args.out_dir / "targets_s1_seed1701.npy"
    counts_path = args.out_dir / "atom_counts_seed1701.npy"
    np.save(target_path, targets)
    np.save(counts_path, atom_counts)

    manifest_path = args.out_dir / "condition_manifest.csv"
    fieldnames = [
        "generation_index", "band_index", "s1_range", "target_s1_eV",
        "requested_atom_count", "source_train_index", "source_id", "source_smiles",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    quantile_edges = np.quantile(values, np.linspace(0.0, 1.0, 6)).tolist()
    summary = {
        "schema_version": 1,
        "source_train_data": str(args.train_data.resolve()),
        "seed": args.seed,
        "train_count": len(records),
        "requested_per_range": args.per_range,
        "total_conditions": len(selected),
        "split_rule": "five equal-count groups after stable S1 sort",
        "quantile_edges_eV": quantile_edges,
        "bands": band_summaries,
        "target_file": str(target_path.resolve()),
        "atom_count_file": str(counts_path.resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    (args.out_dir / "condition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
