#!/usr/bin/env python3
"""Summarize a fixed-EST generation and XTB/sTDA evaluation run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument("--molecules-csv", type=Path, required=True)
    parser.add_argument("--xtb-results", type=Path, required=True)
    parser.add_argument("--target-est", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    generation = json.loads(args.generation_summary.read_text())
    molecules = {
        int(row["generation_index"]): row
        for row in read_csv(args.molecules_csv)
        if row.get("generation_index", "") != ""
    }
    xtb_rows = read_csv(args.xtb_results)
    status_counts = Counter(row.get("status", "unknown") for row in xtb_rows)
    successful = [row for row in xtb_rows if row.get("status") == "ok"]

    enriched: list[dict[str, object]] = []
    gaps: list[float] = []
    clean = 0
    for row in successful:
        generation_index = int(row["generation_index"])
        gap = float(row["s1_energy_eV"]) - float(row["t1_energy_eV"])
        error = gap - args.target_est
        molecule = molecules.get(generation_index, {})
        ring_sizes = [
            int(value)
            for value in molecule.get("ring_sizes", "").split(";")
            if value
        ]
        topology_clean = (
            not any(size <= 4 or size >= 9 for size in ring_sizes)
            and int(float(molecule.get("radical_electrons", 0) or 0)) == 0
        )
        is_clean = as_bool(molecule.get("novel_unique_connected")) and topology_clean
        clean += int(is_clean)
        gaps.append(gap)
        enriched.append(
            {
                **row,
                "calculated_est_eV": gap,
                "est_error_eV": error,
                "abs_est_error_eV": abs(error),
                "ring_sizes": ";".join(map(str, ring_sizes)),
                "topology_clean": topology_clean,
                "chemical_clean_proxy": is_clean,
            }
        )

    output_csv = args.out_dir / "xtb_est_results_enriched.csv"
    if enriched:
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
            writer.writeheader()
            writer.writerows(enriched)

    array = np.asarray(gaps, dtype=np.float64)
    requested = int(generation.get("requested") or 0)
    summary = {
        "definition": "EST = calculated S1 - calculated T1",
        "target_est_eV": args.target_est,
        "requested": requested,
        "rdkit_valid": int(generation.get("valid", 0)),
        "connected": int(generation.get("connected", 0)),
        "novel_unique_connected": int(
            generation.get("novel_unique_connected", 0)
        ),
        "xtb_attempted": len(xtb_rows),
        "xtb_status_counts": dict(status_counts),
        "xtb_ok": len(successful),
        "chemical_clean_proxy": clean,
        "rdkit_valid_rate": (
            float(generation.get("valid", 0)) / requested if requested else None
        ),
        "xtb_ok_rate": len(successful) / requested if requested else None,
        "chemical_clean_proxy_rate": clean / requested if requested else None,
        "calculated_est_mean_eV": float(array.mean()) if len(array) else None,
        "calculated_est_std_eV": float(array.std()) if len(array) else None,
        "calculated_est_quantiles_eV": (
            {
                str(q): float(np.quantile(array, q))
                for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
            }
            if len(array)
            else {}
        ),
        "est_mae_eV": float(np.mean(np.abs(array - args.target_est)))
        if len(array)
        else None,
        "hits": {
            "abs_error_le_0p05": int(np.sum(np.abs(array - args.target_est) <= 0.05)),
            "abs_error_le_0p10": int(np.sum(np.abs(array - args.target_est) <= 0.10)),
            "abs_error_le_0p25": int(np.sum(np.abs(array - args.target_est) <= 0.25)),
            "zero_to_0p5": int(np.sum((array >= 0.0) & (array <= 0.5))),
            "negative": int(np.sum(array < 0.0)),
        },
        "enriched_results": str(output_csv.resolve()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
