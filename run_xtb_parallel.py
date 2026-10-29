#!/usr/bin/env python3
"""Run xTB optimizations with bounded process and thread parallelism."""
import argparse
import csv
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run_one(sdf, output_dir, xtb_bin, env):
    work = output_dir / sdf.stem
    work.mkdir(parents=True, exist_ok=True)
    local = work / sdf.name
    shutil.copy2(sdf, local)
    result = {"name": sdf.stem, "status": "", "returncode": "", "error": ""}
    try:
        with (work / "xtb_opt.log").open("w") as log:
            proc = subprocess.run(
                [str(xtb_bin), sdf.name, "--opt", "loose", "--cycles", "500"],
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=7200,
            )
        result["returncode"] = proc.returncode
        result["status"] = "ok" if proc.returncode == 0 and (work / "xtbopt.sdf").exists() else "failed"
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xtb-bin", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    env["OMP_STACKSIZE"] = "1G"

    files = sorted(args.input_dir.glob("*.sdf"))
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, sdf, args.output_dir, args.xtb_bin, env): sdf for sdf in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)

    results.sort(key=lambda row: row["name"])
    fields = ("name", "status", "returncode", "error")
    with (args.output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "total": len(results),
        "ok": sum(row["status"] == "ok" for row in results),
        "failed": sum(row["status"] != "ok" for row in results),
        "workers": args.workers,
        "threads_per_worker": args.threads,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
