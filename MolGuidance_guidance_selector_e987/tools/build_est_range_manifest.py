#!/usr/bin/env python3
"""Build a paired EST/atom-count benchmark stratified by training-set rank."""

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
    values = np.asarray(
        [float(record["s1"]) - float(record["t1"]) for record in records],
        dtype=np.float64,
    )
    order = np.argsort(values, kind="stable")
    groups = np.array_split(order, len(RANGE_NAMES))
    if min(map(len, groups)) < args.per_range:
        raise ValueError("training split is too small for sampling without replacement")
    rng = np.random.default_rng(args.seed)

    selected: list[dict[str, object]] = []
    band_summaries: list[dict[str, object]] = []
    for band_index, (name, group) in enumerate(zip(RANGE_NAMES, groups), start=1):
        chosen = rng.choice(group, args.per_range, replace=False)
        group_values = values[group]
        band_summaries.append(
            {
                "band_index": band_index,
                "condition_range": name,
                "database_count": int(len(group)),
                "database_min_eV": float(group_values.min()),
                "database_max_eV": float(group_values.max()),
                "database_mean_eV": float(group_values.mean()),
                "selected_count": args.per_range,
            }
        )
        for source_index in chosen:
            record = records[int(source_index)]
            selected.append(
                {
                    "band_index": band_index,
                    "condition_name": "est",
                    "condition_range": name,
                    "target_value": float(values[int(source_index)]),
                    "target_est_eV": float(values[int(source_index)]),
                    "requested_atom_count": int(len(record["atom_types"])),
                    "source_train_index": int(source_index),
                    "source_id": str(record.get("id", source_index)),
                    "source_smiles": str(record.get("smiles", "")),
                }
            )

    selected = [selected[int(index)] for index in rng.permutation(len(selected))]
    for generation_index, row in enumerate(selected):
        row["generation_index"] = generation_index

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = np.asarray([row["target_value"] for row in selected], dtype=np.float32)
    atom_counts = np.asarray(
        [row["requested_atom_count"] for row in selected], dtype=np.int64
    )
    target_path = args.out_dir / "targets_est_seed1701.npy"
    counts_path = args.out_dir / "atom_counts_seed1701.npy"
    np.save(target_path, targets)
    np.save(counts_path, atom_counts)

    manifest_path = args.out_dir / "condition_manifest.csv"
    fieldnames = [
        "generation_index", "band_index", "condition_name", "condition_range",
        "target_value", "target_est_eV", "requested_atom_count",
        "source_train_index", "source_id", "source_smiles",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "schema_version": 1,
        "condition": "est",
        "definition": "S1 - T1",
        "source_train_data": str(args.train_data.resolve()),
        "seed": args.seed,
        "train_count": len(records),
        "total_conditions": len(selected),
        "split_rule": "five equal-count groups after stable EST sort",
        "quantile_edges_eV": np.quantile(values, np.linspace(0, 1, 6)).tolist(),
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
