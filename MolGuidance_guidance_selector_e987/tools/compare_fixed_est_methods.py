#!/usr/bin/env python3
"""Aggregate fixed-EST generation summaries across guidance methods."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for method in args.methods:
        path = args.root / method / "est_analysis" / "summary.json"
        summary = json.loads(path.read_text())
        requested = int(summary["requested"])
        xtb_ok = int(summary["xtb_ok"])
        hits = summary["hits"]
        row = {
            "method": method,
            "requested": requested,
            "rdkit_valid": int(summary["rdkit_valid"]),
            "rdkit_valid_rate": float(summary["rdkit_valid_rate"]),
            "novel_unique_connected": int(summary["novel_unique_connected"]),
            "novel_unique_connected_rate": (
                int(summary["novel_unique_connected"]) / requested
            ),
            "xtb_ok": xtb_ok,
            "xtb_ok_rate": float(summary["xtb_ok_rate"]),
            "chemical_clean_proxy": int(summary["chemical_clean_proxy"]),
            "chemical_clean_proxy_rate": float(
                summary["chemical_clean_proxy_rate"]
            ),
            "calculated_est_mean_eV": float(summary["calculated_est_mean_eV"]),
            "calculated_est_std_eV": float(summary["calculated_est_std_eV"]),
            "est_mae_eV": float(summary["est_mae_eV"]),
            "abs_error_le_0p05": int(hits["abs_error_le_0p05"]),
            "abs_error_le_0p05_rate_of_xtb_ok": (
                int(hits["abs_error_le_0p05"]) / xtb_ok if xtb_ok else None
            ),
            "abs_error_le_0p10": int(hits["abs_error_le_0p10"]),
            "abs_error_le_0p10_rate_of_xtb_ok": (
                int(hits["abs_error_le_0p10"]) / xtb_ok if xtb_ok else None
            ),
            "zero_to_0p5": int(hits["zero_to_0p5"]),
            "zero_to_0p5_rate_of_xtb_ok": (
                int(hits["zero_to_0p5"]) / xtb_ok if xtb_ok else None
            ),
            "negative": int(hits["negative"]),
        }
        rows.append(row)

    csv_path = args.out_dir / "method_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = {
        "lowest_est_mae": min(rows, key=lambda row: row["est_mae_eV"])["method"],
        "highest_rdkit_valid_rate": max(
            rows, key=lambda row: row["rdkit_valid_rate"]
        )["method"],
        "highest_xtb_ok_rate": max(rows, key=lambda row: row["xtb_ok_rate"])[
            "method"
        ],
        "highest_chemical_clean_proxy_rate": max(
            rows, key=lambda row: row["chemical_clean_proxy_rate"]
        )["method"],
        "highest_abs_error_le_0p10_rate_of_xtb_ok": max(
            rows, key=lambda row: row["abs_error_le_0p10_rate_of_xtb_ok"]
        )["method"],
    }
    output = {
        "definition": "EST = calculated S1 - calculated T1",
        "target_est_eV": 0.0,
        "methods": rows,
        "best": best,
        "method_summary_csv": str(csv_path.resolve()),
    }
    output_path = args.out_dir / "summary.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
