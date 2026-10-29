#!/usr/bin/env python3
"""Summarize and plot the HP1KG ΔE_ST = S1 - T1 distribution."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def describe(values: np.ndarray) -> dict[str, float | int]:
    percentiles = np.percentile(values, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {
        "count": int(values.size),
        "mean_eV": float(values.mean()),
        "std_eV": float(values.std(ddof=1)),
        "min_eV": float(values.min()),
        "max_eV": float(values.max()),
        "negative_count": int((values < 0).sum()),
        "negative_fraction": float((values < 0).mean()),
        "abs_le_0p05_count": int((np.abs(values) <= 0.05).sum()),
        "abs_le_0p1_count": int((np.abs(values) <= 0.1).sum()),
        "abs_le_0p25_count": int((np.abs(values) <= 0.25).sum()),
        "abs_le_0p5_count": int((np.abs(values) <= 0.5).sum()),
        "percentiles_eV": {
            label: float(value)
            for label, value in zip(
                ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"),
                percentiles,
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bin-width", type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_values: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        records = torch.load(
            args.data_dir / f"{split}.pt", map_location="cpu", weights_only=False
        )
        values = np.asarray(
            [float(record["s1"]) - float(record["t1"]) for record in records],
            dtype=np.float64,
        )
        split_values[split] = values
        for record, est in zip(records, values):
            rows.append(
                {
                    "split": split,
                    "id": record.get("id", ""),
                    "smiles": record["smiles"],
                    "s1_eV": float(record["s1"]),
                    "t1_eV": float(record["t1"]),
                    "est_eV": float(est),
                }
            )

    all_values = np.concatenate(list(split_values.values()))
    lower = np.floor(all_values.min() / args.bin_width) * args.bin_width
    upper = np.ceil(all_values.max() / args.bin_width) * args.bin_width
    bins = np.arange(lower, upper + args.bin_width * 1.01, args.bin_width)

    summary = {
        "definition": "est_eV = s1_eV - t1_eV",
        "bin_width_eV": args.bin_width,
        "all": describe(all_values),
        "splits": {name: describe(values) for name, values in split_values.items()},
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    with (args.out_dir / "molecules.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (args.out_dir / "histogram_bins.csv").open("w", newline="") as handle:
        fieldnames = ["left_eV", "right_eV", "all", "train", "val", "test"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        all_counts, _ = np.histogram(all_values, bins=bins)
        split_counts = {
            name: np.histogram(values, bins=bins)[0]
            for name, values in split_values.items()
        }
        for index in range(len(bins) - 1):
            writer.writerow(
                {
                    "left_eV": bins[index],
                    "right_eV": bins[index + 1],
                    "all": int(all_counts[index]),
                    **{name: int(counts[index]) for name, counts in split_counts.items()},
                }
            )

    plt.figure(figsize=(10, 6))
    plt.hist(all_values, bins=bins, color="#315f9b", edgecolor="white")
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2)
    plt.axvline(all_values.mean(), color="#b22222", linewidth=1.6, label=f"mean={all_values.mean():.3f} eV")
    plt.xlabel("ΔE_ST = S1 − T1 (eV)")
    plt.ylabel("Molecule count")
    plt.title(f"HP1KG ΔE_ST distribution (N={all_values.size}, bin={args.bin_width:g} eV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "est_histogram_all.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    colors = {"train": "#315f9b", "val": "#d0742c", "test": "#39935c"}
    for name, values in split_values.items():
        plt.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=colors[name],
            label=f"{name} (N={len(values)})",
        )
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2)
    plt.xlabel("ΔE_ST = S1 − T1 (eV)")
    plt.ylabel("Density")
    plt.title("HP1KG ΔE_ST distribution by split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "est_histogram_splits.png", dpi=180)
    plt.close()

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
