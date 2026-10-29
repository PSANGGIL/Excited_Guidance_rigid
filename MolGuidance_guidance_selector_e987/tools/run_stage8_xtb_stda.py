"""Detached XTB/sTDA evaluation of a completed raw Stage8 generation panel.

Uses the existing calculation protocol, retains diffusion heavy-atom coordinates
until XTB optimization, and records EST or S1 targets without inventing targets
for the other properties. Both singlet and triplet energies are calculated.
All requested generation indices remain in the outcome ledger, including skips.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import csv
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

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from excitation_conditions import FORMULAS, energy_values


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_file(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def csv_file(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    tmp.replace(path)


def generation_spec(manifest, target_est=None, target_s1=None, target_s1_minus_2t1=None):
    """Accept both panel schemas; reject accidental relabeling of conditions."""
    names = [name for name in FORMULAS if manifest.get(f"target_{name}_eV") is not None]
    if len(names) != 1:
        raise ValueError("generation must specify exactly one scalar target")
    name = names[0]
    target = float(manifest[f"target_{name}_eV"])
    if not math.isfinite(target):
        raise ValueError("nonfinite generation target")
    if manifest.get("property", name) != name:
        raise ValueError("generation property and target disagree")
    if manifest.get("condition_name", name) != name:
        raise ValueError("generation condition name and target disagree")
    for requested_name, value in (("est", target_est), ("s1", target_s1), ("s1_minus_2t1", target_s1_minus_2t1)):
        if value is not None and (requested_name != name or not math.isfinite(value)
                                  or not math.isclose(value, target, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("calculation target disagrees with generation manifest")
    requested = manifest.get("requested", manifest.get("n"))
    epoch = manifest.get("checkpoint_epoch_zero_based", manifest.get("checkpoint_epoch"))
    if not isinstance(requested, int) or requested <= 0 or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("missing or invalid generation count/checkpoint epoch")
    return dict(condition_name=name, target=target, requested=requested, checkpoint_epoch=epoch)


def exact_energies(row):
    """Use reported decimal energies so exact hit-window boundaries are stable."""
    s1, t1 = (Decimal(str(row[k])) for k in ("s1_energy_eV", "t1_energy_eV"))
    return s1, t1, s1 - t1


def summarize(out, records, results, target, finished=False, condition_name="est"):
    import numpy as np
    if condition_name not in FORMULAS or not math.isfinite(target):
        raise ValueError("invalid calculation target")
    ordered = [results[k] for k in sorted(results)]
    good = [r for r in ordered if r["status"] == "ok"]
    energies = [exact_energies(r) for r in good]
    exact_gaps = [v[2] for v in energies]
    exact_conditions = [energy_values(v[0], v[1])[condition_name] for v in energies]
    exact_errors = [abs(v - Decimal(str(target))) for v in exact_conditions]
    gaps = np.array([float(v) for v in exact_gaps])
    errors = np.array([float(v) for v in exact_errors])
    s1_values = np.array([float(v[0]) for v in energies])
    conditions = np.array([float(v) for v in exact_conditions])
    sf_values = np.array([float(energy_values(v[0],v[1])["s1_minus_2t1"]) for v in energies])
    summary = dict(requested_generation=len(records), xtb_eligible=sum(r["eligible"] for r in records),
                   completed_calculations=len(ordered), status_counts=dict(Counter(r["status"] for r in ordered)),
                   xtb_stda_successes=len(good), condition_name=condition_name, target_value_eV=target,
                   target_est_eV=target if condition_name == "est" else None,
                   target_s1_eV=target if condition_name == "s1" else None, target_t1_eV=None,
                   target_s1_minus_2t1_eV=target if condition_name == "s1_minus_2t1" else None,
                   condition_formula=FORMULAS[condition_name],
                   success_rate_of_requested=len(good) / len(records),
                   success_rate_of_eligible=len(good) / max(1, sum(r["eligible"] for r in records)),
                   est_mean_eV=float(gaps.mean()) if len(gaps) else None,
                   est_mae_eV=float(errors.mean()) if len(gaps) and condition_name == "est" else None,
                   s1_mean_eV=float(s1_values.mean()) if len(gaps) else None,
                   s1_mae_eV=float(errors.mean()) if len(gaps) and condition_name == "s1" else None,
                   condition_mae_eV=float(errors.mean()) if len(gaps) else None,
                   condition_mean_eV=float(conditions.mean()) if len(gaps) else None,
                   condition_std_eV=float(conditions.std()) if len(gaps) else None,
                   s1_minus_2t1_mean_eV=float(sf_values.mean()) if len(gaps) else None,
                   s1_minus_2t1_mae_eV=float(errors.mean()) if len(gaps) and condition_name == "s1_minus_2t1" else None,
                   condition_quantiles_eV={str(q):float(np.quantile(conditions,q)) for q in (0,.05,.25,.5,.75,.95,1)} if len(gaps) else {},
                   condition_negative_count=sum(v<0 for v in exact_conditions),
                   est_quantiles_eV={str(q): float(np.quantile(gaps, q)) for q in (0, .05, .25, .5, .75, .95, 1)} if len(gaps) else {},
                   hits=dict(abs_error_le_0p05=sum(e <= Decimal("0.05") for e in exact_errors),
                             abs_error_le_0p1=sum(e <= Decimal("0.1") for e in exact_errors),
                             abs_error_le_0p25=sum(e <= Decimal("0.25") for e in exact_errors),
                             negative=sum(g < 0 for g in exact_gaps),
                             zero_to_0p5=sum(0 <= g <= Decimal("0.5") for g in exact_gaps)),
                   hit_definitions="abs_error uses condition_name; negative and zero_to_0p5 refer to calculated EST",
                   calibration_applied=False, complete=finished, updated_utc=stamp())
    ledger = []
    for row in records:
        index = row["generation_index"]
        if index in results:
            ledger.append({**row, **results[index]})
            continue
        if not row["raw_valid"]:
            status = "not_attempted_invalid_graph"
        elif not row["connected"]:
            status = "not_attempted_disconnected"
        elif row.get("radicals", 0):
            status = "not_attempted_open_shell"
        elif row.get("net_charge", 0):
            status = "not_attempted_nonzero_net_charge"
        else:
            status = "pending"
        ledger.append({**row, "status": status})
    summary["full_outcome_counts"] = dict(Counter(r["status"] for r in ledger))
    json_file(out / "summary.json", summary)
    csv_file(out / "results.csv", ordered)
    csv_file(out / "all_1000_outcomes.csv", ledger)
    return summary


def prepare_saved_endpoints(source, output, records, manifest):
    """Reconstruct the SAME sampled graph/H states; never infer bonds from XYZ.

    Aromatic heavy-only SDF can lose explicit-H semantics on reread. Therefore
    hydrogenation precedes serialization and both chemical identity and heavy
    coordinates are checked through the fully hydrogenated SDF round trip.
    """
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    torch.set_num_threads(1)
    spec = generation_spec(manifest)
    metadata_path = Path(manifest["data_manifest"]["directory"]) / "joint_atom_state_metadata.json"
    if digest(metadata_path) != manifest["data_manifest"]["sha256"][metadata_path.name]:
        raise ValueError("atom-state metadata changed since generation")
    vocab = json.loads(metadata_path.read_text())["states"]
    wanted = {r["generation_index"]: r for r in records if r["eligible"]}
    output.mkdir(exist_ok=False)
    rows, visited = [], set()
    bond_types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    for path in sorted((source / "raw_endpoints").glob("*.pt")):
        for endpoint in torch.load(path, map_location="cpu", weights_only=False):
            index = endpoint["generation_index"]
            if index not in wanted:
                continue
            if index in visited:
                raise ValueError("duplicate endpoint generation index")
            visited.add(index)
            name = f"generated_{index:06d}"
            row = dict(name=name, source_index=index, generation_index=index, status="failed", error="",
                       smiles=wanted[index]["smiles"], heavy_atoms=wanted[index]["heavy_atoms"],
                       aromatic_rings=wanted[index]["aromatic_ring_count"],
                       condition_name=spec["condition_name"], target_value=spec["target"])
            try:
                states = endpoint["nodes"]["s_1"].argmax(-1).tolist()
                if any(s > len(vocab) for s in states):
                    raise ValueError("unresolved atom mask")
                keep = [i for i, s in enumerate(states) if s < len(vocab)]
                mapping = {old: new for new, old in enumerate(keep)}
                xyz = endpoint["nodes"]["x_1"][keep].numpy().astype(np.float64)
                if not np.isfinite(xyz).all():
                    raise ValueError("nonfinite source coordinates")
                rw = Chem.RWMol()
                for old in keep:
                    state = vocab[states[old]]
                    atom = Chem.Atom(state["symbol"])
                    atom.SetFormalCharge(int(state["formal_charge"]))
                    atom.SetNumExplicitHs(int(state["hydrogens"]))
                    atom.SetNoImplicit(True)
                    rw.AddAtom(atom)
                src, dst = endpoint["edge_index"]
                orders = endpoint["edges"]["e_1"].argmax(-1)
                for a, b, order in zip(src.tolist(), dst.tolist(), orders.tolist()):
                    if a >= b or a not in mapping or b not in mapping or order == 0:
                        continue
                    if order >= len(bond_types):
                        raise ValueError("unresolved bond mask")
                    rw.AddBond(mapping[a], mapping[b], bond_types[order])
                mol = rw.GetMol()
                conf = Chem.Conformer(len(keep))
                conf.Set3D(True)
                for i, point in enumerate(xyz):
                    conf.SetAtomPosition(i, point.tolist())
                mol.AddConformer(conf)
                Chem.SanitizeMol(mol)
                if Chem.MolToSmiles(mol, isomericSmiles=False) != wanted[index]["smiles"]:
                    raise ValueError("endpoint reconstruction changed molecular identity")
                if len(Chem.GetMolFrags(mol)) != 1 or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                    raise ValueError("endpoint not connected/net-neutral/closed-shell")
                hydrogenated = Chem.AddHs(mol, addCoords=True)
                all_xyz = hydrogenated.GetConformer().GetPositions()
                if not np.isfinite(all_xyz).all():
                    raise ValueError("hydrogen placement produced nonfinite coordinates")
                delta = float(np.abs(all_xyz[:len(keep)] - xyz).max())
                if delta > 1e-7:
                    raise ValueError("hydrogen placement changed heavy-atom coordinates")
                hydrogenated.SetProp("_Name", name)
                hydrogenated.SetIntProp("generation_index", index)
                hydrogenated.SetProp("coordinate_source", "original diffusion endpoint; heavy atoms unchanged")
                block = Chem.MolToMolBlock(hydrogenated, kekulize=True)
                reread = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
                if reread is None or Chem.MolToSmiles(Chem.RemoveHs(reread), isomericSmiles=False) != wanted[index]["smiles"]:
                    raise ValueError("fully hydrogenated SDF round trip changed molecular identity")
                rounding = float(np.abs(reread.GetConformer().GetPositions()[:len(keep)] - xyz).max())
                if rounding > 5.1e-5:
                    raise ValueError("SDF changed coordinates beyond decimal rounding")
                with Chem.SDWriter(str(output / f"{name}.sdf")) as writer:
                    writer.write(hydrogenated)
                row.update(status="ok", formal_charge=0, uhf=0, heavy_atoms=len(keep),
                           rings=rdMolDescriptors.CalcNumRings(mol),
                           aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(mol),
                           hydrogens=hydrogenated.GetNumAtoms() - len(keep),
                           heavy_coord_max_abs_delta=delta, sdf_coordinate_rounding_delta=rounding,
                           identity_roundtrip=True)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    if visited != set(wanted):
        raise ValueError("missing saved endpoint for eligible molecule")
    rows.sort(key=lambda row: row["generation_index"])
    csv_file(output / "manifest.csv", rows)
    json_file(output / "summary.json", dict(total=len(rows), exported=sum(r["status"] == "ok" for r in rows),
              failed=sum(r["status"] != "ok" for r in rows),
              max_heavy_coord_abs_delta=max((r["heavy_coord_max_abs_delta"] for r in rows if r["status"] == "ok"), default=None),
              method="original sampled atom-state/bond/XYZ tensors; hydrogenation BEFORE Kekule SDF serialization",
              rdkit_embedding=False, force_field_optimization=False))
    return rows


def worker(args):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Draw, rdDepictor
    from joint_xtb_stda_pipeline import run_one

    RDLogger.DisableLog("rdApp.error")
    source, out = Path(args.generation).resolve(), Path(args.output).resolve()
    started = time.monotonic()
    status = dict(phase="preparing", pid=os.getpid(), sid=os.getsid(0), started_utc=stamp(), workers=args.workers, threads=1)
    json_file(out / "status.json", status)
    env = dict(os.environ)
    home = Path(args.xtb4stda_home).resolve()
    xtb_env = Path(args.xtb_env).resolve()
    env["PATH"] = f"{xtb_env / 'bin'}:{home / 'exe'}:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(xtb_env / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["XTB4STDAHOME"] = str(home)
    env["OMP_STACKSIZE"] = "2G"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    binaries = {name: shutil.which(name, path=env["PATH"]) for name in ("xtb", "xtb4stda", "stda")}
    if not all(binaries.values()):
        raise FileNotFoundError(binaries)
    records = json.loads((source / "raw_analysis/records.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    spec = generation_spec(manifest, args.target_est, args.target_s1, args.target_s1_minus_2t1)
    condition_name, target = spec["condition_name"], spec["target"]
    if len(records) != spec["requested"] or len({r["generation_index"] for r in records}) != len(records):
        raise ValueError("raw panel ledger mismatch")
    checkpoint = source / "checkpoint_snapshot.ckpt"
    if digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("checkpoint snapshot differs from generation manifest")
    record_map = {r["generation_index"]: r for r in records}
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in (Path(__file__), TOOLS / "joint_xtb_stda_diffusion_coords.py", TOOLS / "joint_xtb_stda_pipeline.py", TOOLS / "excitation_conditions.py"):
        shutil.copy2(path, snapshot / path.name)
        hashes[path.name] = digest(path)
    source_sdf = source / "raw_analysis/unique_novel_connected.sdf"
    copied = out / "source_diffusion_heavy_atoms.sdf"
    shutil.copy2(source_sdf, copied)
    shutil.copy2(source / "raw_analysis/records.json", out / "source_all_1000_records.json")
    json_file(out / "protocol.json", dict(
        generation=str(source), checkpoint_epoch=spec["checkpoint_epoch"],
        checkpoint_sha256=manifest["checkpoint_sha256"], input_sdf=str(source_sdf), input_sha256=digest(copied),
        generation_manifest_sha256=digest(source / "manifest.json"),
        raw_records_sha256=digest(source / "raw_analysis/records.json"),
        workers=args.workers, threads_per_calculation=1, condition_name=condition_name,
        target_est_eV=target if condition_name == "est" else None,
        target_s1_eV=target if condition_name == "s1" else None, target_t1_eV=None, binaries=binaries,
        target_s1_minus_2t1_eV=target if condition_name == "s1_minus_2t1" else None,
        condition_formula=FORMULAS[condition_name],
        binary_sha256={k: digest(v) for k, v in binaries.items()}, source_sha256=hashes,
        coordinate_policy="original saved endpoint graph/H/XYZ + RDKit AddHs(addCoords=True) before SDF serialization; no ETKDG/MMFF/UFF",
        input_source="raw_endpoints/*.pt; copied heavy-only SDF is archived for provenance, NOT reread as the calculation input",
        optimization="existing protocol: xtb --opt loose --cycles 500 --chrg 0 --uhf 0",
        excitation="xtb4stda, then stda -xtb -e 10.0 for singlet and with -t for triplet",
        scope="raw samples only; no joint-decoder molecules and no topology/geometry prefilter",
        note="XTB optimization may move atoms. Heavy atom preservation is verified at the XTB INPUT, not after optimization."))
    inputs, work = out / "inputs", out / "work"
    work.mkdir()
    metadata = prepare_saved_endpoints(source, inputs, records, manifest)
    meta_by_name = {r["name"]: r for r in metadata}
    ok_ids = set()
    results = {}
    for row in metadata:
        index = int(row["generation_index"])
        original = record_map[index]
        if row["status"] == "ok":
            if not original["eligible"] or "." in row["smiles"]:
                raise ValueError("preparation admitted a disconnected/charged/open-shell graph")
            ok_ids.add(index)
        elif original["eligible"]:
            results[index] = dict(generation_index=index, name=row["name"], status="preparation_failed", error=row["error"])
    if ok_ids | set(results) != {r["generation_index"] for r in records if r["eligible"]}:
        raise ValueError("some eligible raw molecules were lost during input preparation")
    json_file(out / "preparation_accounting.json", dict(requested=len(records), connected_raw_entries=sum(r["connected"] for r in records), preparation_rows=len(metadata),
              eligible=sum(r["eligible"] for r in records), prepared=len(ok_ids), preparation_failures=len(results),
              exact_heavy_atom_preservation_max_delta=json.loads((inputs / "summary.json").read_text())["max_heavy_coord_abs_delta"]))
    status.update(phase="calculating", requested=len(records), eligible=len(ok_ids), completed=0)
    json_file(out / "status.json", status)
    summarize(out, records, results, target, condition_name=condition_name)
    files = [inputs / f"{row['name']}.sdf" for row in metadata if row["status"] == "ok"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (out / "progress.jsonl").open("w") as log:
        futures = {pool.submit(run_one, path, work, env): path.stem for path in files}
        for completed, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            meta = meta_by_name[name]
            index = int(meta["generation_index"])
            try:
                result = dict(future.result())
            except Exception:
                result = dict(name=name, status="driver_exception", error=traceback.format_exc())
            result.update(generation_index=index, smiles=meta["smiles"], heavy_atoms=int(meta["heavy_atoms"]),
                          aromatic_rings=int(meta["aromatic_rings"]), condition_name=condition_name,
                          target_est_eV=target if condition_name == "est" else None,
                          target_s1_eV=target if condition_name == "s1" else None,
                          target_s1_minus_2t1_eV=target if condition_name == "s1_minus_2t1" else None)
            if result["status"] == "ok":
                if any(result.get(k) != 0 for k in ("xtb_returncode", "xtb4stda_returncode", "singlet_returncode", "triplet_returncode")):
                    result.update(status="nonzero_returncode_with_parsed_states")
                elif not all(math.isfinite(result[k]) for k in ("s1_energy_eV", "t1_energy_eV")):
                    result.update(status="nonfinite_excitation_energy")
                else:
                    s1, t1, gap = exact_energies(result)
                    values = energy_values(s1,t1)
                    error = values[condition_name] - Decimal(str(target))
                    result.update(calculated_est_eV=float(gap), condition_error_eV=float(error),
                                  calculated_s1_minus_2t1_eV=float(values["s1_minus_2t1"]),
                                  calculated_condition_eV=float(values[condition_name]),
                                  abs_condition_error_eV=float(abs(error)),
                                  **{f"{condition_name}_error_eV": float(error),
                                     f"abs_{condition_name}_error_eV": float(abs(error))})
            results[index] = result
            json_file(work / name / "audited_result.json", result)
            log.write(json.dumps(result, allow_nan=False) + "\n")
            log.flush()
            status.update(completed=completed, successes=sum(r["status"] == "ok" for r in results.values()),
                          updated_utc=stamp(), elapsed_seconds=time.monotonic() - started)
            json_file(out / "status.json", status)
            print(json.dumps(status), flush=True)
            if completed % 24 == 0:
                summarize(out, records, results, target, condition_name=condition_name)
    summary = summarize(out, records, results, target, finished=True, condition_name=condition_name)
    drawings = []
    for index in sorted(results):
        row = results[index]
        if row["status"] != "ok":
            continue
        mol = Chem.MolFromSmiles(row["smiles"])
        rdDepictor.Compute2DCoords(mol)
        value=energy_values(row["s1_energy_eV"],row["t1_energy_eV"])[condition_name]
        legend = f"#{index} S1={row['s1_energy_eV']:.3f} T1={row['t1_energy_eV']:.3f}\n{FORMULAS[condition_name]}={float(value):.3f} eV"
        drawings.append((mol, legend))
    png_dir = out / "rdkit_2d"
    png_dir.mkdir()
    png_paths = []
    for start in range(0, len(drawings), 24):
        page = drawings[start:start + 24]
        path = png_dir / f"xtb_stda_success_{start // 24:03d}.png"
        Draw.MolsToGridImage([m for m, _ in page], legends=[label for _, label in page],
                            molsPerRow=4, subImgSize=(300, 260)).save(path)
        png_paths.append(str(path))
    json_file(png_dir / "manifest.json", dict(rendered=len(drawings), png_paths=png_paths,
              depiction="original generated 2D graph annotated with optimized XTB/sTDA energies; not a connectivity-preservation certificate"))
    status.update(phase="complete", finished_utc=stamp(), exit_code=0, elapsed_seconds=time.monotonic() - started,
                  summary=str(out / "summary.json"), png_pages=len(png_paths))
    json_file(out / "status.json", status)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xtb-env", default="/home/jovyan/cpu-only-1-datavol-1/xtb_env")
    parser.add_argument("--xtb4stda-home", default="/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home")
    parser.add_argument("--workers", type=int, default=24)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--target-est", type=float, default=None)
    targets.add_argument("--target-s1", type=float, default=None)
    targets.add_argument("--target-s1-minus-2t1", type=float, default=None)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    source = Path(args.generation).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    generation_spec(manifest, args.target_est, args.target_s1, args.target_s1_minus_2t1)
    if json.loads((source / "status.json").read_text()).get("phase") != "complete":
        parser.error("generation must be complete before calculation")
    out = Path(args.output).resolve()
    if not args.worker:
        out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        with (out / "supervisor.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=TOOLS, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        launch = dict(pid=process.pid, detached=True, command=command, output=str(out), started_utc=stamp())
        json_file(out / "launch.json", launch)
        print(json.dumps(launch, indent=2))
        return
    try:
        worker(args)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        state = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
        state.update(phase="failed", error=error, exit_code=1, finished_utc=stamp())
        json_file(out / "status.json", state)
        raise


if __name__ == "__main__":
    main()
