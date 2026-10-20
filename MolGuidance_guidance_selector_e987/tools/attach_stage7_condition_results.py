"""Attach externally computed S1/T1 to an immutable generation panel.

Input CSV: generation_index,status,s1_eV,t1_eV (status='success').
Required calculation manifest: model_state_sha256, source_sdf_sha256, method.
No calibration offset or missing-result imputation is applied.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from molguidance.model_utils.provenance import sha256_file


def attach(report, rows, manifest, tolerance=.1):
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("EST tolerance must be finite and non-negative")
    for key, expected in (
        ("model_state_sha256", report["model_state_sha256"]),
        ("source_sdf_sha256", report["raw_sdf_sha256"]),
    ):
        if manifest.get(key) != expected:
            raise ValueError(f"calculation provenance mismatch: {key}")
    if not manifest.get("method"):
        raise ValueError("calculation method must be recorded")
    records = {row["generation_index"]: dict(row) for row in report["records"]}
    seen, errors = set(), []
    for row in rows:
        index = int(row["generation_index"])
        if index not in records or index in seen:
            raise ValueError(f"unknown or duplicate generation_index: {index}")
        seen.add(index)
        if row["status"] != "success":
            records[index]["calculation_status"] = row["status"]
            continue
        if not records[index]["eligible"]:
            raise ValueError("neutral-closed-shell evaluation includes an ineligible graph")
        s1, t1 = float(row["s1_eV"]), float(row["t1_eV"])
        if not all(math.isfinite(value) for value in (s1, t1)):
            raise ValueError("non-finite successful S1/T1 calculation")
        est = s1 - t1
        error = abs(est - records[index]["target_est_eV"])
        records[index].update(calculation_status="success", s1_eV=s1, t1_eV=t1,
                              est_eV=est, condition_abs_error_eV=error,
                              condition_hit=error <= tolerance + 1e-6)
        errors.append(error)
    evaluated = list(records.values())
    hits = [row for row in evaluated if row.get("condition_hit", False)]
    n = report["requested"]
    return {
        "model_state_sha256": report["model_state_sha256"],
        "raw_sdf_sha256": report["raw_sdf_sha256"],
        "method": manifest["method"], "requested": n, "calculation_success": len(errors),
        "calculation_coverage": len(errors) / max(n, 1),
        "condition_mae_eV_calculated_only": sum(errors) / len(errors) if errors else None,
        "condition_hit_yield": len(hits) / max(n, 1),
        "aromatic_condition_hit_yield": sum(row["aromatic"] for row in hits) / max(n, 1),
        "geometry_aromatic_condition_hit_yield": sum(row["aromatic"] and row["geometry_screen"] for row in hits) / max(n, 1),
        "tolerance_eV": tolerance, "records": evaluated,
        "selection_note": "compare identical condition/count panels; missing calculations are not property hits",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--calculation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=.1)
    args = parser.parse_args()
    with args.report.open() as stream:
        report = json.load(stream)
    with args.calculation_manifest.open() as stream:
        manifest = json.load(stream)
    with args.results.open() as stream:
        results = list(csv.DictReader(stream))
    result = attach(report, results, manifest, args.tolerance)
    result.update(generation_report_sha256=sha256_file(args.report),
                  calculation_results_sha256=sha256_file(args.results),
                  calculation_manifest_sha256=sha256_file(args.calculation_manifest))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
