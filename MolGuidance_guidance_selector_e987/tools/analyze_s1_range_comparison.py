#!/usr/bin/env python3
"""Aggregate structural and xTB/sTDA performance over S1 rank ranges."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("cfg", "mg", "cg", "ag", "conditional")
RANGES = ("Q1_low", "Q2_low_mid", "Q3_mid", "Q4_high_mid", "Q5_high")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_block(
    requested: list[dict[str, str]],
    structural: list[dict[str, str]],
    xtb: list[dict[str, str]],
) -> dict[str, object]:
    count = len(requested)
    valid = [row for row in structural if truth(row.get("valid"))]
    connected = [row for row in valid if truth(row.get("connected"))]
    novel = [row for row in valid if truth(row.get("novel_unique_connected"))]
    aromatic = [row for row in valid if int(float(row.get("num_aromatic_rings") or 0)) > 0]
    radicals = [row for row in valid if int(float(row.get("radical_electrons") or 0)) > 0]
    xtb_ok = [row for row in xtb if row.get("status") == "ok"]
    errors: list[float] = []
    predicted: list[float] = []
    targets: list[float] = []
    for row in xtb_ok:
        prediction = finite(row.get("s1_energy_eV"))
        target = finite(row.get("target_s1_eV"))
        if prediction is not None and target is not None:
            predicted.append(prediction)
            targets.append(target)
            errors.append(prediction - target)
    array = np.asarray(errors, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    requested_targets = np.asarray(
        [float(row["target_s1_eV"]) for row in requested], dtype=np.float64
    )
    band_min = float(requested_targets.min()) if len(requested_targets) else None
    band_max = float(requested_targets.max()) if len(requested_targets) else None
    quantile_labels = ("p25", "p50", "p75")
    target_quantiles = (
        np.quantile(target_array, (0.25, 0.5, 0.75))
        if len(target_array) else (None, None, None)
    )
    predicted_quantiles = (
        np.quantile(predicted_array, (0.25, 0.5, 0.75))
        if len(predicted_array) else (None, None, None)
    )
    # With equal empirical weights and paired sample counts, the 1-D
    # Wasserstein-1 distance is the mean difference between sorted samples.
    distribution_w1 = (
        float(np.mean(np.abs(np.sort(predicted_array) - np.sort(target_array))))
        if len(target_array) else None
    )
    result: dict[str, object] = {
        "requested": count,
        "sdf_written": len(structural),
        "builder_or_sdf_missing": count - len(structural),
        "rdkit_valid": len(valid),
        "valid_rate": len(valid) / count if count else None,
        "connected": len(connected),
        "connected_rate": len(connected) / count if count else None,
        "novel_unique_connected": len(novel),
        "novel_rate": len(novel) / count if count else None,
        "aromatic_valid": len(aromatic),
        "aromatic_fraction_of_valid": len(aromatic) / len(valid) if valid else None,
        "radical_valid": len(radicals),
        "xtb_attempted_novel": len(xtb),
        "xtb_ok": len(array),
        "xtb_success_rate_of_requested": len(array) / count if count else None,
        "s1_target_mean_eV": float(np.mean([float(r["target_s1_eV"]) for r in requested])) if requested else None,
        "s1_pred_mean_eV": float(np.mean(predicted)) if predicted else None,
        "s1_requested_band_min_eV": band_min,
        "s1_requested_band_max_eV": band_max,
        "s1_target_std_eV": float(target_array.std()) if len(target_array) else None,
        "s1_pred_std_eV": float(predicted_array.std()) if len(predicted_array) else None,
        **{
            f"s1_target_{label}_eV": None if value is None else float(value)
            for label, value in zip(quantile_labels, target_quantiles)
        },
        **{
            f"s1_pred_{label}_eV": None if value is None else float(value)
            for label, value in zip(quantile_labels, predicted_quantiles)
        },
        "s1_pred_inside_requested_band": (
            float(np.mean((predicted_array >= band_min) & (predicted_array <= band_max)))
            if len(predicted_array) and band_min is not None and band_max is not None
            else None
        ),
        "s1_distribution_w1_eV": distribution_w1,
        "s1_bias_eV": float(array.mean()) if len(array) else None,
        "s1_mae_eV": float(np.abs(array).mean()) if len(array) else None,
        "s1_rmse_eV": float(np.sqrt(np.mean(array ** 2))) if len(array) else None,
        "hit_within_0p25_eV": float(np.mean(np.abs(array) <= 0.25)) if len(array) else None,
        "hit_within_0p5_eV": float(np.mean(np.abs(array) <= 0.5)) if len(array) else None,
        "hit_within_1p0_eV": float(np.mean(np.abs(array) <= 1.0)) if len(array) else None,
    }
    if len(array) >= 2 and np.std(targets) > 0 and np.std(predicted) > 0:
        result["s1_pearson_r"] = float(np.corrcoef(targets, predicted)[0, 1])
    else:
        result["s1_pearson_r"] = None
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_range_metrics(rows: list[dict[str, object]], output: Path) -> None:
    by_key = {(row["method"], row["s1_range"]): row for row in rows}
    x = np.arange(len(RANGES), dtype=float)
    width = 0.15
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
    for method_index, method in enumerate(METHODS):
        offset = (method_index - 2) * width
        valid = [by_key[(method, band)]["valid_rate"] or 0.0 for band in RANGES]
        mae = [by_key[(method, band)]["s1_mae_eV"] for band in RANGES]
        hit = [by_key[(method, band)]["hit_within_0p5_eV"] for band in RANGES]
        axes[0].bar(x + offset, valid, width, label=method)
        axes[1].bar(x + offset, [np.nan if v is None else v for v in mae], width)
        axes[2].bar(x + offset, [0.0 if v is None else v for v in hit], width)
    axes[0].set_ylabel("RDKit valid / requested")
    axes[1].set_ylabel("xTB/sTDA S1 MAE (eV)\nnovel connected only")
    axes[2].set_ylabel("|error| <= 0.5 eV")
    axes[2].set_xticks(x, RANGES, rotation=18)
    axes[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("S1 guidance performance by training-distribution quintile")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_calibration(root: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    axes_flat = list(axes.flat)
    all_values: list[float] = []
    for axis, method in zip(axes_flat, METHODS):
        rows = [row for row in read_csv(root / "xtb" / method / "work" / "results.csv") if row.get("status") == "ok"]
        target = np.asarray([float(row["target_s1_eV"]) for row in rows], dtype=float) if rows else np.asarray([])
        prediction = np.asarray([float(row["s1_energy_eV"]) for row in rows], dtype=float) if rows else np.asarray([])
        if len(target):
            all_values.extend(target.tolist()); all_values.extend(prediction.tolist())
            axis.scatter(target, prediction, s=12, alpha=0.45)
            if len(target) >= 2 and np.std(target) > 0:
                slope, intercept = np.polyfit(target, prediction, 1)
                xx = np.linspace(target.min(), target.max(), 100)
                axis.plot(xx, slope * xx + intercept, color="tab:red", linewidth=1.5,
                          label=f"slope={slope:.2f}")
                axis.legend()
        axis.set_title(f"{method} (n={len(target)})")
        axis.grid(alpha=0.2)
    axes_flat[-1].axis("off")
    if all_values:
        low, high = min(all_values), max(all_values)
        for axis in axes_flat[:-1]:
            axis.plot([low, high], [low, high], "k--", linewidth=1)
    for axis in axes[-1, :2]: axis.set_xlabel("target S1 (eV)")
    for axis in axes[:, 0]: axis.set_ylabel("xTB/sTDA S1 (eV)")
    fig.suptitle("Per-molecule S1 calibration (unique novel connected molecules)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = read_csv(args.manifest)
    requested_by_range: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        requested_by_range[row["s1_range"]].append(row)

    range_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for method in METHODS:
        structural = read_csv(args.root / method / "analysis" / "molecules.csv")
        xtb = read_csv(args.root / "xtb" / method / "work" / "results.csv")
        structural_by_range: dict[str, list[dict[str, str]]] = defaultdict(list)
        xtb_by_range: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in structural:
            structural_by_range[row.get("s1_range", "")].append(row)
        for row in xtb:
            xtb_by_range[row.get("s1_range", "")].append(row)
        for band in RANGES:
            range_rows.append({
                "method": method,
                "s1_range": band,
                **metric_block(requested_by_range[band], structural_by_range[band], xtb_by_range[band]),
            })
        overall_rows.append({
            "method": method,
            "s1_range": "ALL",
            **metric_block(manifest, structural, xtb),
        })

    args.root.mkdir(parents=True, exist_ok=True)
    write_rows(args.root / "range_metrics.csv", range_rows)
    write_rows(args.root / "overall_metrics.csv", overall_rows)
    plot_range_metrics(range_rows, args.root / "s1_range_performance.png")
    plot_calibration(args.root, args.root / "s1_calibration.png")
    best_by_range = {}
    for band in RANGES:
        candidates = [row for row in range_rows if row["s1_range"] == band and row["s1_mae_eV"] is not None]
        best_by_range[band] = {
            "lowest_s1_mae": min(candidates, key=lambda row: row["s1_mae_eV"])["method"] if candidates else None,
            "highest_valid_rate": max(
                (row for row in range_rows if row["s1_range"] == band),
                key=lambda row: row["valid_rate"] or 0.0,
            )["method"],
        }
    summary = {
        "scope_note": "xTB/sTDA metrics use unique novel connected structures; structural rates use all 200 requests per range",
        "manifest": str(args.manifest.resolve()),
        "best_by_range": best_by_range,
        "overall": overall_rows,
        "range_metrics_csv": str((args.root / "range_metrics.csv").resolve()),
        "overall_metrics_csv": str((args.root / "overall_metrics.csv").resolve()),
        "performance_png": str((args.root / "s1_range_performance.png").resolve()),
        "calibration_png": str((args.root / "s1_calibration.png").resolve()),
    }
    (args.root / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
