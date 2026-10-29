#!/usr/bin/env python3
"""Compare generated S1 splits against the empirical DB conditional splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


METHODS = ("cfg", "mg", "cg", "ag", "conditional")
RANGES = ("Q1_low", "Q2_low_mid", "Q3_mid", "Q4_high_mid", "Q5_high")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str | None) -> bool:
    return str(value).lower() == "true"


def distribution_jsd(left: list[int], right: list[int]) -> float | None:
    if not left or not right:
        return None
    maximum = max(max(left), max(right))
    left_counts = np.bincount(left, minlength=maximum + 1).astype(float)
    right_counts = np.bincount(right, minlength=maximum + 1).astype(float)
    left_counts /= left_counts.sum()
    right_counts /= right_counts.sum()
    return float(jensenshannon(left_counts, right_counts, base=2.0) ** 2)


def stats(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {f"{prefix}_{key}": None for key in ("n", "mean", "p25", "p50", "p75")}
    q25, q50, q75 = np.quantile(values, (0.25, 0.5, 0.75))
    return {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_p25": float(q25),
        f"{prefix}_p50": float(q50),
        f"{prefix}_p75": float(q75),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_csv(args.manifest)
    bounds = {}
    for name in RANGES:
        targets = [float(row["target_s1_eV"]) for row in manifest if row["s1_range"] == name]
        bounds[name] = (min(targets), max(targets))

    db_all = [
        row for row in read_csv(args.db_results)
        if row["split"] == "train" and row["status"] == "ok"
    ]
    db_by_range: dict[str, list[dict[str, str]]] = {}
    for name in RANGES:
        lower, upper = bounds[name]
        db_by_range[name] = [
            row for row in db_all
            if lower <= float(row["reference_s1_eV"]) <= upper
        ]

    output_rows: list[dict[str, object]] = []
    for method in METHODS:
        structures = read_csv(args.root / method / "analysis" / "molecules.csv")
        calculated = [
            row for row in read_csv(args.root / "xtb" / method / "work" / "results.csv")
            if row["status"] == "ok"
        ]
        for name in RANGES:
            lower, upper = bounds[name]
            db_rows = db_by_range[name]
            gen_struct = [
                row for row in structures
                if row.get("s1_range") == name and truth(row.get("valid"))
            ]
            gen_xtb = [row for row in calculated if row.get("s1_range") == name]

            db_reference = np.asarray([float(row["reference_s1_eV"]) for row in db_rows])
            db_calculated = np.asarray([float(row["calculated_s1_eV"]) for row in db_rows])
            gen_reference = np.asarray([float(row["target_s1_eV"]) for row in gen_xtb])
            gen_calculated = np.asarray([float(row["s1_energy_eV"]) for row in gen_xtb])
            db_error = db_calculated - db_reference
            gen_error = gen_calculated - gen_reference

            db_rings = [int(row["rings"]) for row in db_rows]
            db_aromatic_rings = [int(row["aromatic_rings"]) for row in db_rows]
            db_atoms = [int(row["heavy_atoms"]) for row in db_rows]
            gen_rings = [int(float(row.get("num_rings") or 0)) for row in gen_struct]
            gen_aromatic_rings = [int(float(row.get("num_aromatic_rings") or 0)) for row in gen_struct]
            gen_atoms = [int(float(row.get("num_heavy_atoms") or 0)) for row in gen_struct]

            row: dict[str, object] = {
                "method": method,
                "s1_range": name,
                "range_min_eV": lower,
                "range_max_eV": upper,
                "requested": 200,
                "valid": len(gen_struct),
                "valid_rate": len(gen_struct) / 200,
                "xtb_ok": len(gen_xtb),
                "db_n": len(db_rows),
                "db_reference_mae_eV": float(np.abs(db_error).mean()),
                "generated_mae_eV": float(np.abs(gen_error).mean()),
                "excess_mae_vs_db_eV": float(np.abs(gen_error).mean() - np.abs(db_error).mean()),
                "db_bias_eV": float(db_error.mean()),
                "generated_bias_eV": float(gen_error.mean()),
                "calculated_s1_w1_to_db_eV": float(wasserstein_distance(db_calculated, gen_calculated)),
                "db_calculated_inside_range": float(np.mean((db_calculated >= lower) & (db_calculated <= upper))),
                "generated_calculated_inside_range": float(np.mean((gen_calculated >= lower) & (gen_calculated <= upper))),
                "db_aromatic_molecule_fraction": float(np.mean(np.asarray(db_aromatic_rings) > 0)),
                "generated_aromatic_molecule_fraction": float(np.mean(np.asarray(gen_aromatic_rings) > 0)) if gen_aromatic_rings else None,
                "db_mean_rings": float(np.mean(db_rings)),
                "generated_mean_rings": float(np.mean(gen_rings)) if gen_rings else None,
                "db_mean_aromatic_rings": float(np.mean(db_aromatic_rings)),
                "generated_mean_aromatic_rings": float(np.mean(gen_aromatic_rings)) if gen_aromatic_rings else None,
                "db_mean_heavy_atoms": float(np.mean(db_atoms)),
                "generated_mean_heavy_atoms": float(np.mean(gen_atoms)) if gen_atoms else None,
                "ring_count_jsd": distribution_jsd(db_rings, gen_rings),
                "aromatic_ring_count_jsd": distribution_jsd(db_aromatic_rings, gen_aromatic_rings),
                "heavy_atom_count_jsd": distribution_jsd(db_atoms, gen_atoms),
                **stats(db_calculated, "db_calculated_s1"),
                **stats(gen_calculated, "generated_calculated_s1"),
            }
            output_rows.append(row)

    csv_path = args.out_dir / "db_conditional_range_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    best = {}
    for name in RANGES:
        candidates = [row for row in output_rows if row["s1_range"] == name]
        best[name] = {
            "lowest_calculated_s1_w1": min(candidates, key=lambda row: row["calculated_s1_w1_to_db_eV"])["method"],
            "lowest_excess_mae": min(candidates, key=lambda row: row["excess_mae_vs_db_eV"])["method"],
            "lowest_ring_count_jsd": min(candidates, key=lambda row: row["ring_count_jsd"])["method"],
        }

    x = np.arange(len(RANGES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for method in METHODS:
        rows = [row for row in output_rows if row["method"] == method]
        axes[0, 0].plot(x, [row["calculated_s1_w1_to_db_eV"] for row in rows], marker="o", label=method)
        axes[0, 1].plot(x, [row["excess_mae_vs_db_eV"] for row in rows], marker="o", label=method)
        axes[1, 0].plot(x, [row["generated_aromatic_molecule_fraction"] for row in rows], marker="o", label=method)
        axes[1, 1].plot(x, [row["valid_rate"] for row in rows], marker="o", label=method)
    db_aromatic = [output_rows[index]["db_aromatic_molecule_fraction"] for index in range(len(RANGES))]
    axes[1, 0].plot(x, db_aromatic, color="black", linestyle="--", marker="x", label="DB")
    axes[0, 0].set_ylabel("W1(gen calc S1, DB calc S1), eV")
    axes[0, 1].set_ylabel("Generated MAE - DB oracle MAE, eV")
    axes[1, 0].set_ylabel("Aromatic molecule fraction")
    axes[1, 1].set_ylabel("RDKit valid / requested")
    for axis in axes.flat:
        axis.set_xticks(x, RANGES, rotation=18)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=3)
    fig.suptitle("S1 split: generated distributions vs empirical DB conditional distributions")
    fig.tight_layout()
    png_path = args.out_dir / "db_conditional_comparison.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    summary = {
        "interpretation": "Compare each generated range to the corresponding empirical train-DB conditional range; lower W1/JSD/excess MAE is better.",
        "best_by_range": best,
        "metrics_csv": str(csv_path.resolve()),
        "comparison_png": str(png_path.resolve()),
        "db_results": str(args.db_results.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
