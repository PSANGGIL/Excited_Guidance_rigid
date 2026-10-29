#!/usr/bin/env python3
"""Detached scalar-condition generation -> original-coordinate XTB/sTDA -> PNGs.

Existing training, model code, checkpoints, and earlier outputs are never changed.
The launcher freezes the required Python sources; each stage has its own log and
status. Quantum calculations use the existing audited driver, not ETKDG inputs.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from excitation_conditions import FORMULAS, target_metadata, energy_values, require_property
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_file(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def snapshot_sources(destination):
    sources = list(sorted((PACKAGE / "molguidance").rglob("*.py")))
    sources += [PACKAGE / "tools" / name for name in (
        Path(__file__).name, "generate_stage8_panel.py", "run_stage8_xtb_stda.py",
        "joint_xtb_stda_pipeline.py", "joint_xtb_stda_diffusion_coords.py", "excitation_conditions.py")]
    hashes = {}
    for source in sources:
        relative = source.relative_to(PACKAGE)
        target = destination / relative
        before = digest(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if digest(source) != before or digest(target) != before:
            raise RuntimeError(f"Source changed while being frozen: {source}")
        hashes[str(relative)] = before
    return hashes


def stage(command, output, state, phase, log_path, env):
    """Supervise a foreground worker inside an independent OS session."""
    state.update(phase=phase, updated_utc=now(), stage_command=command)
    with log_path.open("xb") as log:
        process = subprocess.Popen(command, cwd=output / "pipeline_source_snapshot",
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   close_fds=True, start_new_session=True, env=env)
        state.update(stage_pid=process.pid, stage_log=str(log_path))
        json_file(output / "pipeline_status.json", state)
        print(json.dumps({"event": "stage_started", **state}), flush=True)
        code = process.wait()
    state.update(stage_exit_code=code, updated_utc=now())
    json_file(output / "pipeline_status.json", state)
    if code != 0:
        raise RuntimeError(f"{phase} failed with exit code {code}; see {log_path}")


def render_spaced(rows, destination, title, condition_name="est", target=0.):
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Geometry import Point2D

    columns, cell_width, cell_height, structure_height = 5, 680, 620, 410
    header, footer = 130, 90
    width = columns * cell_width
    height = header + ((len(rows) + columns - 1) // columns) * cell_height + footer
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height, cell_width, structure_height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding, options.minFontSize, options.maxFontSize = .10, 14, 32
    options.bondLineWidth = 2.2
    drawer.ClearDrawing()
    for i, row in enumerate(rows):
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            raise ValueError("Invalid successful graph during depiction")
        rdDepictor.Compute2DCoords(mol)
        drawer.SetOffset((i % columns) * cell_width, header + (i // columns) * cell_height)
        drawer.DrawMolecule(mol)
    drawer.SetOffset(0, 0)
    drawer.SetColour((.10, .13, .18))
    drawer.SetFontSize(32)
    drawer.DrawString(title, Point2D(width / 2, 42), 0, True)
    drawer.SetFontSize(24)
    drawer.DrawString(f"XTB/sTDA successes | DB matches and duplicates excluded | Target {FORMULAS[condition_name]} = {target:g} eV",
                      Point2D(width / 2, 91), 0, True)
    for i, row in enumerate(rows):
        center_x = (i % columns) * cell_width + cell_width / 2
        top_y = header + (i // columns) * cell_height
        value = energy_values(row["s1_energy_eV"],row["t1_energy_eV"])[condition_name]
        error = abs(value-Decimal(str(target)))
        labels = (
            f"Molecule #{int(row['generation_index']):06d}",
            f"S1 = {float(row['s1_energy_eV']):.3f} eV | T1 = {float(row['t1_energy_eV']):.3f} eV",
            f"{FORMULAS[condition_name]} = {float(value):+.3f} eV | Error = {float(error):.3f} eV",
        )
        for line, (label, offset) in enumerate(zip(labels, (455, 507, 559))):
            drawer.SetFontSize(29 if line == 0 else 27)
            drawer.DrawString(label, Point2D(center_x, top_y + offset), 0, True)
    drawer.SetFontSize(22)
    drawer.SetColour((.30, .33, .38))
    drawer.DrawString("Original generated graphs in RDKit 2D; energies after XTB optimization, uncalibrated.",
                      Point2D(width / 2, height - 59), 0, True)
    drawer.DrawString("Calculation success does not certify physical stability or unchanged optimized connectivity.",
                      Point2D(width / 2, height - 25), 0, True)
    drawer.FinishDrawing()
    with destination.open("xb") as handle:
        handle.write(drawer.GetDrawingText())


def postprocess(output, calculation, requested, condition_name="est", target=0.):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.error")
    records = json.loads((output / "raw_analysis/records.json").read_text())
    indexed = {int(row["generation_index"]): row for row in records}
    if len(records) != requested or len(indexed) != requested:
        raise ValueError("Generation ledger is incomplete")
    summary = json.loads((calculation / "summary.json").read_text())
    target_metadata(condition_name,target)
    if summary.get("condition_name", "est") != condition_name or summary.get("target_value_eV", 0.) != target:
        raise ValueError("postprocessing condition disagrees with quantum summary")
    if not summary["complete"] or sum(summary["full_outcome_counts"].values()) != requested:
        raise ValueError("Quantum outcome ledger is incomplete")
    # Keep the old driver's compatibility names and add correctly sized aliases.
    if requested != 1000:
        shutil.copy2(calculation / "all_1000_outcomes.csv", calculation / f"all_{requested}_outcomes.csv")
        shutil.copy2(calculation / "source_all_1000_records.json", calculation / f"source_all_{requested}_records.json")
    with (calculation / "results.csv").open(newline="") as handle:
        results = list(csv.DictReader(handle))
    successful = [row for row in results if row["status"] == "ok"]
    if len(successful) != summary["xtb_stda_successes"]:
        raise ValueError("Calculation success count mismatch")
    rows, seen, excluded_db, duplicate_count = [], set(), 0, 0
    for row in sorted(successful, key=lambda item: int(item["generation_index"])):
        original = indexed[int(row["generation_index"])]
        if not original["eligible"]:
            raise ValueError("Successful calculation of an ineligible raw graph")
        if row.get("condition_name", "est") != condition_name:
            raise ValueError("per-molecule condition disagrees with postprocessing target")
        if original["db_member"] is True:
            excluded_db += 1
            continue
        if original["db_member"] is not False:
            raise ValueError("Missing DB membership flag")
        mol = Chem.MolFromSmiles(row["smiles"])
        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        reference = Chem.MolToSmiles(Chem.MolFromSmiles(original["smiles"]), isomericSmiles=False)
        if canonical != reference:
            raise ValueError("Generated and calculated input graphs differ")
        if canonical in seen:
            duplicate_count += 1
            continue
        seen.add(canonical)
        s1, t1 = Decimal(row["s1_energy_eV"]), Decimal(row["t1_energy_eV"])
        if not s1.is_finite() or not t1.is_finite() or s1 - t1 != Decimal(row["calculated_est_eV"]):
            raise ValueError("Inconsistent result energies")
        values=energy_values(s1,t1)
        value=values[condition_name]
        error=abs(value-Decimal(str(target)))
        if row.get("calculated_condition_eV") and Decimal(row["calculated_condition_eV"]) != value:
            raise ValueError("Calculated condition does not match S1/T1")
        rows.append(dict(generation_index=int(row["generation_index"]), canonical_smiles=canonical,
                         s1_energy_eV=str(s1), t1_energy_eV=str(t1), calculated_est_eV=str(s1 - t1),
                         calculated_s1_minus_2t1_eV=str(values["s1_minus_2t1"]),
                         calculated_condition_eV=str(value), abs_condition_error_eV=str(error),
                         condition_name=condition_name,target_value_eV=target,
                         abs_est_error_eV=str(error) if condition_name == "est" else None,
                         aromatic=original["aromatic"],
                         ring_sizes=json.dumps(original.get("ring_sizes", [])),
                         topology_screen=original.get("topology_screen"), geometry_screen=original.get("geometry_screen")))
    directory = calculation / "rdkit_2d_novel_unique_spaced"
    directory.mkdir(exist_ok=False)
    fields = ["generation_index", "canonical_smiles", "s1_energy_eV", "t1_energy_eV",
              "calculated_est_eV", "abs_est_error_eV", "calculated_s1_minus_2t1_eV",
              "calculated_condition_eV", "abs_condition_error_eV", "condition_name", "target_value_eV",
              "aromatic", "ring_sizes", "topology_screen", "geometry_screen"]
    with (directory / "molecules.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (directory / "molecules.smi").open("x") as handle:
        for row in rows:
            handle.write(f"{row['canonical_smiles']} generated_{row['generation_index']:06d}\n")
    pngs = []
    for start in range(0, len(rows), 20):
        png = directory / f"novel_xtb_stda_{start // 20:03d}.png"
        render_spaced(rows[start:start + 20], png, f"HP1KG | Novel successful molecules | Page {start // 20 + 1}", condition_name, target)
        pngs.append(str(png))
    top = sorted(rows, key=lambda row: (Decimal(row["abs_condition_error_eV"]), row["generation_index"]))[:20]
    if top:
        render_spaced(top, directory / f"top20_{condition_name}_target_match_spaced.png",
                      f"HP1KG | Top 20 by absolute {FORMULAS[condition_name]} target error",condition_name,target)
        with (directory / "top20_molecules.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(top)
    report = dict(requested_generation=requested, calculation_successes=len(successful),
                  excluded_db_matches=excluded_db, excluded_duplicate_successes=duplicate_count,
                  novel_unique_successes=len(rows), label_line_separation_px=52, pngs=pngs,
                  condition_name=condition_name,condition_formula=FORMULAS[condition_name],target_value_eV=target,
                  exact_abs_condition_error_le_0p1_novel_unique=sum(Decimal(row["abs_condition_error_eV"]) <= Decimal("0.1") for row in rows),
                  **{f"exact_abs_{condition_name}_le_0p1_novel_unique":sum(Decimal(row["abs_condition_error_eV"]) <= Decimal("0.1") for row in rows)},
                  depiction="Original generated graphs; optimized XTB/sTDA energies; no topology prefilter.")
    json_file(directory / "manifest.json", report)
    return report


def worker(output):
    request = json.loads((output / "pipeline_request.json").read_text())
    condition_name=request.get("property","est")
    target=float(request["target_eV"])
    snapshot = output / "pipeline_source_snapshot"
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / "tools"))
    state = dict(phase="preparing", pid=os.getpid(), sid=os.getsid(0), started_utc=now(),
                 requested=request["n"], condition=FORMULAS[condition_name],target_eV=target)
    json_file(output / "pipeline_status.json", state)
    try:
        import numpy as np
        import torch
        from generate_stage8_panel import freeze_checkpoint
        from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
        from molguidance.model_utils.provenance import data_manifest, verify_manifest
        torch.set_num_threads(1)
        frozen = output / "pipeline_checkpoint.ckpt"
        checkpoint, checkpoint_hash = freeze_checkpoint(Path(request["checkpoint"]), frozen)
        epoch = int(checkpoint["epoch"])
        model = ClassifierFreeGuidance.load_from_checkpoint(str(frozen), map_location="cpu", weights_only=False)
        verify_manifest(model.data_manifest, data_manifest(Path(request["data"])))
        norm=torch.load(Path(request["data"])/"train_data_property_normalization.pt",map_location="cpu",weights_only=False)
        require_property(condition_name,norm.get("property_names"))
        generator = torch.Generator(device="cpu").manual_seed(request["seed"])
        counts = model.sample_n_atoms(request["n"], properties=target, generator=generator).cpu().numpy()
        if counts.shape != (request["n"],) or not np.issubdtype(counts.dtype, np.integer) or (counts <= 0).any():
            raise ValueError("Invalid atom-count manifest")
        with (output / "sampled_atom_counts.npy").open("xb") as handle:
            np.save(handle, counts)
        json_file(output / "atom_count_manifest.json", dict(
            source="checkpoint atom-count sampler; training-data distribution only", **target_metadata(condition_name,target),
            model_config=model.atom_count_conditioning, diagnostics=model.last_atom_count_diagnostics,
            seed=request["seed"], n=len(counts), min=int(counts.min()), max=int(counts.max()),
            mean=float(counts.mean()), sha256=digest(output / "sampled_atom_counts.npy")))
        state.update(checkpoint_epoch_zero_based=epoch, checkpoint_sha256=checkpoint_hash)
        del model, checkpoint
        env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        for key in THREAD_VARIABLES:
            env[key] = "1"
        command = [sys.executable, "-u", str(snapshot / "tools/generate_stage8_panel.py"),
                   "--checkpoint", str(frozen), "--data", request["data"],
                   "--counts", str(output / "sampled_atom_counts.npy"), "--output", str(output),
                   "--n", str(request["n"]), "--batch", str(request["batch"]), "--target", str(target),
                   "--property",condition_name,
                   "--cfg", "1", "--timesteps", "100", "--seed", str(request["seed"]),
                   "--memory-fraction", str(request["memory_fraction"]),
                   "--joint-workers", "2", "--joint-seconds", "2", "--worker"]
        stage(command, output, state, "generating", output / "worker.log", env)
        generated = json.loads((output / "status.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        if (generated["phase"] != "complete" or generated["generated"] != request["n"]
                or manifest["checkpoint_sha256"] != checkpoint_hash
                or manifest[f"target_{condition_name}_eV"] != target
                or manifest.get("property") != condition_name or manifest["requested"] != request["n"]):
            raise ValueError("Generation did not finish with the requested checkpoint/count/condition")
        calculation = output / "xtb_stda_diffusion_coords_24x1"
        calculation.mkdir(exist_ok=False)
        command = [sys.executable, "-u", str(snapshot / "tools/run_stage8_xtb_stda.py"),
                   "--generation", str(output), "--output", str(calculation), "--workers", "24",
                   "--target-"+condition_name.replace("_","-"),str(target),"--worker"]
        env["CUDA_VISIBLE_DEVICES"] = ""
        stage(command, output, state, "xtb_stda", calculation / "supervisor.log", env)
        if json.loads((calculation / "status.json").read_text())["phase"] != "complete":
            raise ValueError("Quantum calculations did not finish")
        state.update(phase="postprocessing", updated_utc=now())
        json_file(output / "pipeline_status.json", state)
        drawing_report = postprocess(output, calculation, request["n"],condition_name,target)
        summary = dict(generation=str(output), calculation=str(calculation),
                       condition=FORMULAS[condition_name],condition_name=condition_name,target_eV=target,requested=request["n"],
                       checkpoint_epoch_zero_based=epoch, checkpoint_sha256=checkpoint_hash,
                       generation_summary=json.loads((output / "raw_analysis/summary.json").read_text()),
                       quantum_summary=json.loads((calculation / "summary.json").read_text()),
                       depiction_summary=drawing_report, finished_utc=now())
        json_file(output / "pipeline_summary.json", summary)
        state.update(phase="complete", exit_code=0, finished_utc=now(), summary=str(output / "pipeline_summary.json"))
        json_file(output / "pipeline_status.json", state)
        print(json.dumps(state), flush=True)
    except BaseException:
        error = traceback.format_exc()
        state.update(phase="failed", error=error, exit_code=1, finished_utc=now())
        json_file(output / "pipeline_status.json", state)
        print(error, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--property",choices=tuple(FORMULAS),default="est")
    parser.add_argument("--target",type=float,default=0.)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--memory-fraction", type=float, default=0.08)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        worker(output)
        return
    if args.checkpoint is None or args.data is None:
        parser.error("--checkpoint and --data are required when launching")
    if args.n < 1 or args.batch < 1 or not math.isfinite(args.memory_fraction) or not 0 < args.memory_fraction <= 1:
        parser.error("positive n, batch and memory-fraction in (0,1] required")
    target_metadata(args.property,args.target)
    if not args.checkpoint.is_file() or not args.data.is_dir():
        parser.error("checkpoint or data directory does not exist")
    for binary in ("/home/jovyan/cpu-only-1-datavol-1/xtb_env/bin/xtb",
                   "/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe/xtb4stda",
                   "/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe/stda"):
        if not os.access(binary, os.X_OK):
            parser.error(f"Missing executable: {binary}")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "pipeline_source_snapshot"
    source_hashes = snapshot_sources(snapshot)
    request = dict(checkpoint=str(args.checkpoint.resolve()), data=str(args.data.resolve()),
                   n=args.n, batch=args.batch, seed=args.seed, memory_fraction=args.memory_fraction,
                   condition=FORMULAS[args.property],property=args.property,target_eV=args.target,cfg=1.0,time_grid_points=100,
                   precision="float32; TF32 disabled", xtb_workers=24, threads_per_calculation=1,
                   created_utc=now(), source_sha256=source_hashes,
                   coordinate_policy="diffusion heavy XYZ + RDKit hydrogen coordinates; no ETKDG/MMFF/UFF")
    json_file(output / "pipeline_request.json", request)
    command = [sys.executable, "-u", str(snapshot / "tools" / Path(__file__).name),
               "--output", str(output), "--worker"]
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    for key in THREAD_VARIABLES:
        env[key] = "1"
    with (output / "pipeline.log").open("xb") as log:
        child = subprocess.Popen(command, cwd=snapshot, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True, close_fds=True)
    launch = dict(pid=child.pid, detached=True, command=command, output=str(output), started_utc=now())
    json_file(output / "pipeline_launch.json", launch)
    print(json.dumps(launch, indent=2))


if __name__ == "__main__":
    main()
