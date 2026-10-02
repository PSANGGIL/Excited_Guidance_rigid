#!/usr/bin/env python3
"""Analyze EST-conditioned generations against train-DB conditional ranges."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


METHODS = ("cfg", "mg", "cg", "ag")
RANGES = ("Q1_low", "Q2_low_mid", "Q3_mid", "Q4_high_mid", "Q5_high")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: str | None) -> bool:
    return str(value).lower() == "true"


def jsd(left: list[int], right: list[int]) -> float | None:
    if not left or not right:
        return None
    size = max(max(left), max(right)) + 1
    p = np.bincount(left, minlength=size).astype(float)
    q = np.bincount(right, minlength=size).astype(float)
    p /= p.sum(); q /= q.sum()
    return float(jensenshannon(p, q, base=2.0) ** 2)


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {f"{prefix}_{key}": None for key in ("n", "mean", "std", "p25", "p50", "p75")}
    p25, p50, p75 = np.quantile(values, (0.25, 0.5, 0.75))
    return {
        f"{prefix}_n": int(len(values)), f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()), f"{prefix}_p25": float(p25),
        f"{prefix}_p50": float(p50), f"{prefix}_p75": float(p75),
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
    manifest_by_index = {int(row["generation_index"]): row for row in manifest}
    bounds = {
        name: (
            min(float(row["target_value"]) for row in manifest if row["condition_range"] == name),
            max(float(row["target_value"]) for row in manifest if row["condition_range"] == name),
        )
        for name in RANGES
    }
    db_all = [row for row in read_csv(args.db_results) if row["split"] == "train" and row["status"] == "ok"]
    db_by_range = {}
    for name, (lower, upper) in bounds.items():
        db_by_range[name] = [
            row for row in db_all
            if lower <= float(row["reference_s1_eV"]) - float(row["reference_t1_eV"]) <= upper
        ]

    range_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for method in METHODS:
        structures = read_csv(args.root / method / "analysis" / "molecules.csv")
        results = [row for row in read_csv(args.root / "xtb" / method / "work" / "results.csv") if row["status"] == "ok"]
        joined_results = []
        for row in results:
            generation_index = int(row["generation_index"])
            condition = manifest_by_index[generation_index]
            joined_results.append({**row, **condition})

        all_target = np.asarray([float(row["target_value"]) for row in joined_results])
        all_calculated = np.asarray([float(row["s1_energy_eV"]) - float(row["t1_energy_eV"]) for row in joined_results])
        slope, intercept = np.polyfit(all_target, all_calculated, 1) if len(all_target) >= 2 else (np.nan, np.nan)
        correlation = np.corrcoef(all_target, all_calculated)[0, 1] if len(all_target) >= 2 else np.nan
        overall_rows.append({
            "method": method,
            "requested": len(manifest),
            "rdkit_valid": sum(truth(row.get("valid")) for row in structures),
            "xtb_ok": len(joined_results),
            "est_response_slope": float(slope),
            "est_response_intercept": float(intercept),
            "est_pearson_r": float(correlation),
            "est_r2": float(correlation ** 2),
            "est_bias_eV": float((all_calculated - all_target).mean()),
            "est_mae_eV": float(np.abs(all_calculated - all_target).mean()),
            "est_rmse_eV": float(np.sqrt(np.mean((all_calculated - all_target) ** 2))),
        })

        for name in RANGES:
            lower, upper = bounds[name]
            db_rows = db_by_range[name]
            gen_struct = [row for row in structures if row.get("condition_range") == name and truth(row.get("valid"))]
            gen_rows = [row for row in joined_results if row["condition_range"] == name]
            db_reference = np.asarray([float(row["reference_s1_eV"]) - float(row["reference_t1_eV"]) for row in db_rows])
            db_calculated = np.asarray([float(row["calculated_s1_eV"]) - float(row["calculated_t1_eV"]) for row in db_rows])
            gen_target = np.asarray([float(row["target_value"]) for row in gen_rows])
            gen_calculated = np.asarray([float(row["s1_energy_eV"]) - float(row["t1_energy_eV"]) for row in gen_rows])
            db_error = db_calculated - db_reference
            gen_error = gen_calculated - gen_target
            db_rings = [int(row["rings"]) for row in db_rows]
            db_aromatic = [int(row["aromatic_rings"]) for row in db_rows]
            db_atoms = [int(row["heavy_atoms"]) for row in db_rows]
            gen_rings = [int(float(row.get("num_rings") or 0)) for row in gen_struct]
            gen_aromatic = [int(float(row.get("num_aromatic_rings") or 0)) for row in gen_struct]
            gen_atoms = [int(float(row.get("num_heavy_atoms") or 0)) for row in gen_struct]
            range_rows.append({
                "method": method, "est_range": name, "range_min_eV": lower, "range_max_eV": upper,
                "requested": 200, "valid": len(gen_struct), "valid_rate": len(gen_struct) / 200,
                "xtb_ok": len(gen_rows), "db_n": len(db_rows),
                "db_oracle_mae_eV": float(np.abs(db_error).mean()),
                "generated_mae_eV": float(np.abs(gen_error).mean()),
                "excess_mae_vs_db_eV": float(np.abs(gen_error).mean() - np.abs(db_error).mean()),
                "db_bias_eV": float(db_error.mean()), "generated_bias_eV": float(gen_error.mean()),
                "calculated_est_w1_to_db_eV": float(wasserstein_distance(db_calculated, gen_calculated)),
                "db_calculated_inside_range": float(np.mean((db_calculated >= lower) & (db_calculated <= upper))),
                "generated_calculated_inside_range": float(np.mean((gen_calculated >= lower) & (gen_calculated <= upper))),
                "hit_within_0p05_eV": float(np.mean(np.abs(gen_error) <= 0.05)),
                "hit_within_0p1_eV": float(np.mean(np.abs(gen_error) <= 0.1)),
                "hit_within_0p25_eV": float(np.mean(np.abs(gen_error) <= 0.25)),
                "db_aromatic_fraction": float(np.mean(np.asarray(db_aromatic) > 0)),
                "generated_aromatic_fraction": float(np.mean(np.asarray(gen_aromatic) > 0)) if gen_aromatic else None,
                "ring_count_jsd": jsd(db_rings, gen_rings),
                "aromatic_ring_count_jsd": jsd(db_aromatic, gen_aromatic),
                "heavy_atom_count_jsd": jsd(db_atoms, gen_atoms),
                **quantiles(db_calculated, "db_calculated_est"),
                **quantiles(gen_calculated, "generated_calculated_est"),
            })

    range_path = args.out_dir / "est_range_metrics.csv"
    overall_path = args.out_dir / "est_overall_metrics.csv"
    for path, rows in ((range_path, range_rows), (overall_path, overall_rows)):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    x = np.arange(len(RANGES), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    for method in METHODS:
        rows = [row for row in range_rows if row["method"] == method]
        axes[0, 0].plot(x, [row["calculated_est_w1_to_db_eV"] for row in rows], marker="o", label=method)
        axes[0, 1].plot(x, [row["excess_mae_vs_db_eV"] for row in rows], marker="o", label=method)
        axes[1, 0].plot(x, [row["generated_aromatic_fraction"] for row in rows], marker="o", label=method)
        axes[1, 1].plot(x, [row["valid_rate"] for row in rows], marker="o", label=method)
    db_aromatic = [range_rows[index]["db_aromatic_fraction"] for index in range(len(RANGES))]
    axes[1, 0].plot(x, db_aromatic, "k--x", label="DB")
    axes[0, 0].set_ylabel("W1(gen calc EST, DB calc EST), eV")
    axes[0, 1].set_ylabel("Generated MAE - DB oracle MAE, eV")
    axes[1, 0].set_ylabel("Aromatic molecule fraction")
    axes[1, 1].set_ylabel("RDKit valid / requested")
    for axis in axes.flat:
        axis.set_xticks(x, RANGES, rotation=18); axis.grid(alpha=0.25)
    axes[0, 0].legend(ncol=2)
    fig.suptitle("EST guidance by train-DB conditional quintile")
    fig.tight_layout()
    png_path = args.out_dir / "est_range_comparison.png"
    fig.savefig(png_path, dpi=180); plt.close(fig)

    best = {}
    for name in RANGES:
        candidates = [row for row in range_rows if row["est_range"] == name]
        best[name] = {
            "lowest_est_w1": min(candidates, key=lambda row: row["calculated_est_w1_to_db_eV"])["method"],
            "lowest_excess_mae": min(candidates, key=lambda row: row["excess_mae_vs_db_eV"])["method"],
            "highest_valid_rate": max(candidates, key=lambda row: row["valid_rate"])["method"],
        }
    summary = {
        "definition": "EST = S1 - T1",
        "best_by_range": best,
        "range_metrics": str(range_path.resolve()),
        "overall_metrics": str(overall_path.resolve()),
        "comparison_png": str(png_path.resolve()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
