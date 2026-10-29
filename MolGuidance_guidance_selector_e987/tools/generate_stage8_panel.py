"""Detached, auditable generation panel; does not modify training or repair raw outputs.

Preserves every endpoint, evaluates raw and optional final MAP decoding separately,
and draws only unique DB-novel connected molecules. No ETKDG or XTB is run here.
"""
import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from excitation_conditions import FORMULAS, target_metadata, require_property


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def worker_init():
    import torch
    from rdkit import RDLogger
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def decode_one(payload):
    import numpy as np
    from molguidance.analysis.joint_decode import joint_decode
    started = time.monotonic()
    index, nodes, edges, xyz, states, bonds, metadata, count, options = payload
    try:
        result = joint_decode(nodes, edges, xyz, metadata, required_atoms=count, **options)
        report = dict(status=result.status, optimal=result.optimal,
                      iterations=result.iterations, reason=result.reason)
        if result.molecule is not None:
            report.update(atom_state_edits=int(np.sum(result.states != states)),
                          bond_edits=int(np.sum(result.bonds != bonds)),
                          retained_indices=list(map(int, result.retained_indices)),
                          coordinates_reembedded=False)
        return index, None if result.molecule is None else result.molecule.ToBinary(), {
            **report, "wall_seconds": time.monotonic() - started}
    except Exception:
        return index, None, dict(status="exception", reason=traceback.format_exc(),
                                 wall_seconds=time.monotonic() - started)


def detailed_record(mol):
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from molguidance.analysis.strict_generation import strict_molecule_record
    row = strict_molecule_record(mol)
    if not row["raw_valid"]:
        return row
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    atoms = list(mol.GetAtoms())
    charges = [a.GetFormalCharge() for a in atoms]
    rings = list(mol.GetRingInfo().AtomRings())
    row.update(
        fragments=len(Chem.GetMolFrags(mol)),
        charged_atoms=sum(c != 0 for c in charges),
        zwitterionic=any(c > 0 for c in charges) and any(c < 0 for c in charges),
        total_hydrogens=sum(a.GetTotalNumHs() for a in atoms),
        elements=dict(Counter(a.GetSymbol() for a in atoms)),
        ring_count=len(rings), aromatic_ring_count=rdMolDescriptors.CalcNumAromaticRings(mol),
        bridgeheads=rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        spiro_atoms=rdMolDescriptors.CalcNumSpiroAtoms(mol),
        cumulene_atoms=sum(sum(b.GetBondTypeAsDouble() == 2 for b in a.GetBonds()) >= 2 for a in atoms),
        cycle_rank=mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol)),
    )
    systems = []
    for ring in rings:
        group = set(ring)
        overlap = [s for s in systems if s & group]
        for s in overlap:
            group |= s
            systems.remove(s)
        systems.append(group)
    row["ring_system_sizes"] = [len(s) for s in systems]
    if mol.GetNumConformers():
        xyz = mol.GetConformer().GetPositions()
        if np.isfinite(xyz).all():
            radii = np.array([Chem.GetPeriodicTable().GetRcovalent(a.GetAtomicNum()) for a in atoms])
            dist = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
            ratio = dist / np.maximum(radii[:, None] + radii[None, :], 1e-8)
            adj = Chem.GetAdjacencyMatrix(mol).astype(bool)
            upper = np.triu(np.ones_like(adj), 1).astype(bool)
            br = ratio[adj & upper]
            nr = ratio[~adj & upper]
            row.update(short_bonds=int(np.sum(br < .65)), long_bonds=int(np.sum(br > 1.4)),
                       nonbonded_clashes=int(np.sum(nr < .65)),
                       min_bond_radius_ratio=float(br.min()) if len(br) else None,
                       max_bond_radius_ratio=float(br.max()) if len(br) else None)
    return row


def annotate_novelty(rows, reference):
    seen = set()
    for row in rows:
        smi = row["smiles"]
        row["db_member"] = bool(smi and smi in reference)
        row["novel_unique"] = bool(smi and smi not in reference and smi not in seen)
        if smi:
            seen.add(smi)


def aggregate(rows):
    import numpy as np
    from molguidance.analysis.strict_generation import summarize_records
    result = summarize_records(rows)
    valid = [r for r in rows if r["raw_valid"]]
    rings = Counter(size for r in valid for size in r.get("ring_sizes", []))
    result.update(
        unique_valid=len({r["smiles"] for r in valid}),
        db_member_count=sum(r.get("db_member", False) for r in rows),
        novel_unique_connected_count=sum(r.get("novel_unique", False) and r["connected"] for r in rows),
        novel_unique_eligible_count=sum(r.get("novel_unique", False) and r["eligible"] for r in rows),
        radicals_count=sum(r.get("radicals", 0) > 0 for r in valid),
        nonzero_net_charge_count=sum(r.get("net_charge", 0) != 0 for r in valid),
        charged_atoms_molecule_count=sum(r.get("charged_atoms", 0) > 0 for r in valid),
        zwitterionic_count=sum(r.get("zwitterionic", False) for r in valid),
        aromatic_fraction_of_valid=sum(r["aromatic"] for r in valid) / max(1, len(valid)),
        ring_size_counts=dict(sorted(rings.items())),
        five_six_ring_fraction=sum(rings[k] for k in (5, 6)) / max(1, sum(rings.values())),
        ringless_count=sum(not r.get("ring_sizes") for r in valid),
        small_ring_molecule_count=sum(any(s <= 4 for s in r.get("ring_sizes", [])) for r in valid),
        seven_plus_ring_molecule_count=sum(any(s >= 7 for s in r.get("ring_sizes", [])) for r in valid),
        nine_plus_ring_molecule_count=sum(any(s >= 9 for s in r.get("ring_sizes", [])) for r in valid),
        bridgehead_molecule_count=sum(r.get("bridgeheads", 0) > 0 for r in valid),
        cumulene_molecule_count=sum(r.get("cumulene_atoms", 0) > 0 for r in valid),
        geometry_long_bond_molecule_count=sum(r.get("long_bonds", 0) > 0 for r in valid),
        geometry_short_bond_molecule_count=sum(r.get("short_bonds", 0) > 0 for r in valid),
        geometry_clash_molecule_count=sum(r.get("nonbonded_clashes", 0) > 0 for r in valid),
        errors=dict(Counter(r["error"] for r in rows if r["error"])),
        heavy_atom_count_histogram=dict(sorted(Counter(r["heavy_atoms"] for r in valid).items())),
        ring_count_histogram=dict(sorted(Counter(r.get("ring_count", 0) for r in valid).items())),
        element_counts=dict(sum((Counter(r.get("elements", {})) for r in valid), Counter())),
        mean_heavy_atoms_valid=float(np.mean([r["heavy_atoms"] for r in valid])) if valid else None,
    )
    retained = [r for r in rows if "sampled_real_atoms" in r]
    if retained:
        result["raw_count_exact_count"] = sum(r["sampled_real_atoms"] == r["requested_heavy_atoms"] for r in retained)
        result["mean_actual_over_requested_atoms"] = float(np.mean([
            r["sampled_real_atoms"] / r["requested_heavy_atoms"] for r in retained]))
    result["warning"] = "RDKit validity and geometric heuristics do not establish physical or synthetic feasibility."
    return result


def export_panel(folder, rows, molecules, reference):
    from rdkit import Chem
    from rdkit.Chem import Draw, rdDepictor
    folder.mkdir(parents=True, exist_ok=True)
    annotate_novelty(rows, reference)
    write_json(folder / "records.json", rows)
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with (folder / "molecules.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    drawings, drawing_errors = [], []
    with Chem.SDWriter(str(folder / "unique_novel_connected.sdf")) as writer:
        writer.SetKekulize(False)
        for row in rows:
            if not (row["novel_unique"] and row["connected"]):
                continue
            index = row["generation_index"]
            mol = Chem.Mol(molecules[index])
            Chem.SanitizeMol(mol)
            mol.SetIntProp("generation_index", index)
            for name in FORMULAS:
                if f"target_{name}_eV" in row:
                    mol.SetDoubleProp(f"target_{name}_eV", row[f"target_{name}_eV"])
            writer.write(mol)  # original diffusion conformer; drawing uses another copy
            try:
                drawing = Chem.Mol(mol)
                rdDepictor.Compute2DCoords(drawing)
                legend = f"#{index}  rings={row['ring_count']}  arom={row['aromatic_ring_count']}"
                if row.get("radicals", 0):
                    legend += f"  rad={row['radicals']}"
                drawings.append((drawing, legend))
            except Exception as exc:
                drawing_errors.append(dict(generation_index=index, error=str(exc)))
    pngs = []
    for start in range(0, len(drawings), 24):
        page = drawings[start:start + 24]
        path = folder / f"novel_rdkit2d_{start // 24:03d}.png"
        Draw.MolsToGridImage([m for m, _ in page], legends=[label for _, label in page],
                            molsPerRow=4, subImgSize=(300, 250)).save(path)
        pngs.append(str(path))
    report = aggregate(rows)
    report.update(pngs=pngs, depiction_errors=drawing_errors)
    write_json(folder / "summary.json", report)
    return report


def freeze_checkpoint(source, destination):
    import torch
    for attempt in range(5):
        before = source.stat()
        data = source.read_bytes()
        after = source.stat()
        if (before.st_ino, before.st_mtime_ns, before.st_size) != (after.st_ino, after.st_mtime_ns, after.st_size):
            continue
        checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        with destination.open("xb") as out:
            out.write(data)
        return checkpoint, hashlib.sha256(data).hexdigest()
    raise RuntimeError("live checkpoint changed repeatedly; nothing was overwritten")


def run(args):
    import numpy as np
    import torch
    from rdkit import Chem, RDLogger
    from molguidance.model_utils.provenance import state_dict_sha256, sha256_file
    from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
    worker_init()
    RDLogger.DisableLog("rdApp.error")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    out = Path(args.output).resolve()
    started = time.monotonic()
    status = dict(phase="loading", pid=os.getpid(), sid=os.getsid(0), started_utc=now(), requested=args.n,
                  generated=0, joint_completed=0)
    write_json(out / "status.json", status)
    frozen = out / "checkpoint_snapshot.ckpt"
    checkpoint, checkpoint_sha = freeze_checkpoint(Path(args.checkpoint), frozen)
    expected_state_hash = state_dict_sha256(checkpoint["state_dict"])
    epoch, step = int(checkpoint["epoch"]), int(checkpoint["global_step"])
    model = ClassifierFreeGuidance.load_from_checkpoint(str(frozen), map_location="cpu", weights_only=False)
    if state_dict_sha256(model.state_dict()) != expected_state_hash:
        raise RuntimeError("loaded model weights differ from frozen checkpoint")
    data_dir = Path(args.data).resolve()
    from molguidance.model_utils.provenance import data_manifest, verify_manifest
    verify_manifest(model.data_manifest, data_manifest(data_dir))
    property_name = getattr(args, "property", "est")
    target_fields = target_metadata(property_name, args.target)
    norm = torch.load(data_dir / "train_data_property_normalization.pt", map_location="cpu", weights_only=False)
    require_property(property_name, norm.get("property_names"))
    counts = np.load(args.counts)
    if counts.shape != (args.n,) or not np.issubdtype(counts.dtype, np.integer) or (counts <= 0).any():
        raise ValueError("fixed count manifest must contain exactly n positive integers")
    shutil.copy2(args.counts, out / "fixed_atom_counts.npy")
    source_dir = out / "source_snapshot"
    source_dir.mkdir()
    source_hashes = {}
    for source in [Path(__file__), *sorted((PACKAGE / "molguidance").rglob("*.py"))]:
        relative = source.relative_to(PACKAGE)
        dest = source_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        source_hashes[str(relative)] = sha256_file(source)
    joint_options = dict(time_limit=args.joint_seconds, max_rounds=8)
    manifest = dict(created_utc=now(), checkpoint_source=str(Path(args.checkpoint).resolve()),
                    checkpoint_snapshot=str(frozen), checkpoint_sha256=checkpoint_sha,
                    checkpoint_epoch_zero_based=epoch, optimizer_step=step,
                    model_state_sha256=expected_state_hash, data_manifest=model.data_manifest,
                    source_sha256=source_hashes, requested=args.n, property=property_name,
                    condition_formula=FORMULAS[property_name], **target_fields,
                    cfg_scale=args.cfg, seed=args.seed, time_grid_points=args.timesteps,
                    batch_size=args.batch, precision="float32; TF32 off",
                    atom_count_manifest_source=str(Path(args.counts).resolve()),
                    atom_count_manifest_sha256=sha256_file(out / "fixed_atom_counts.npy"),
                    fake_atom_capacity_mode=model.fake_atom_capacity_mode,
                    joint_decoding={**joint_options, "workers": args.joint_workers, "required_atoms": "fixed requested heavy count"},
                    sampling_path="diagonal", coordinates="original diffusion; no ETKDG, optimization or repair",
                    condition_fidelity_status="not_measured_requires_external_calculation")
    write_json(out / "manifest.json", manifest)
    del checkpoint
    reference, database_rows = set(), []
    for split in ("train", "val", "test"):
        data = torch.load(data_dir / f"{split}_data_processed.pt", map_location="cpu", weights_only=False)
        require_property(property_name, data.get("property_names"))
        for i, smi in enumerate(data["smiles"]):
            row = detailed_record(Chem.MolFromSmiles(smi))
            if row["smiles"]:
                reference.add(row["smiles"])
            row.update(split=split, split_index=i,
                       **target_metadata(property_name, float(data["properties"][i].reshape(-1)[0])))
            database_rows.append(row)
    write_json(out / "db_graph_reference.json", dict(
        all_db=aggregate(database_rows), train=aggregate([r for r in database_rows if r["split"] == "train"]),
        **{f"train_abs_{property_name}_le_0p5": aggregate([r for r in database_rows
             if r["split"] == "train" and abs(r["target_value_eV"]) <= .5])},
        note="graph statistics from original DB SMILES; no DB coordinate evaluation here", records=database_rows))
    model.to("cuda:0").eval()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, molecules, futures, joint_results = [], {}, [], {}
    (out / "raw_endpoints").mkdir()
    status.update(phase="generating", checkpoint_epoch_zero_based=epoch, optimizer_step=step)
    write_json(out / "status.json", status)
    print(json.dumps({"event": "generation_started", **manifest}, default=str), flush=True)
    with ProcessPoolExecutor(max_workers=args.joint_workers, mp_context=mp.get_context("spawn"),
                             initializer=worker_init) as pool, Chem.SDWriter(str(out / "raw_generated.sdf")) as writer:
        writer.SetKekulize(False)
        with (out / "raw_records.jsonl").open("w") as row_stream, torch.no_grad():
            for start in range(0, args.n, args.batch):
                length = min(args.batch, args.n - start)
                tick = time.monotonic()
                sampled = model.sample_random_sizes(
                    length, device=model.device, n_timesteps=args.timesteps,
                    number_of_atoms=counts[start:start + length].tolist(),
                    properties_for_sampling=args.target, property_name=property_name, dataset_name=model.dataset_name,
                    properties_handle_method="concatenate_sum", training_mode=False,
                    normalization_file_path=str(data_dir / "train_data_property_normalization.pt"),
                    guide_w={key: args.cfg for key in model.canonical_feat_order},
                    dfm_type="campbell", guidance_format="linear", where_to_apply_guide="probabilities")
                if len(sampled) != length:
                    raise RuntimeError("sampler changed the sample count")
                endpoints = []
                for offset, sample in enumerate(sampled):
                    index = start + offset
                    graph = sample.slot_graph
                    src, dst = graph.edges()
                    selected = torch.where(src < dst)[0]
                    selected = selected[torch.argsort(src[selected] * graph.num_nodes() + dst[selected])]
                    endpoint = dict(generation_index=index, edge_index=torch.stack((src, dst)).cpu(),
                                    requested_heavy_atoms=int(counts[index]), **target_fields,
                                    nodes={k: v.detach().cpu() for k, v in graph.ndata.items() if k.endswith(("_1", "_endpoint_probs"))},
                                    edges={k: v.detach().cpu() for k, v in graph.edata.items() if k.endswith(("_1", "_endpoint_probs"))})
                    endpoints.append(endpoint)
                    mol = sample.rdkit_mol
                    row = detailed_record(mol)
                    row.update(generation_index=index, **target_fields,
                               requested_heavy_atoms=int(counts[index]), sampled_real_atoms=sample.num_atoms,
                               total_slots=sample.num_slots)
                    rows.append(row)
                    if row["raw_valid"]:
                        molecules[index] = Chem.Mol(mol)
                        molecules[index].SetIntProp("generation_index", index)
                        molecules[index].SetDoubleProp(f"target_{property_name}_eV", args.target)
                        writer.write(molecules[index])
                    row_stream.write(json.dumps(row, allow_nan=False) + "\n")
                    payload = (index, graph.ndata["s_endpoint_probs"].numpy(),
                               graph.edata["e_endpoint_probs"][selected].numpy(), graph.ndata["x_1"].numpy(),
                               graph.ndata["s_1"].argmax(-1).numpy(),
                               graph.edata["e_1"][selected].argmax(-1).numpy(),
                               model.atom_state_vocab, int(counts[index]), joint_options)
                    futures.append(pool.submit(decode_one, payload))
                torch.save(endpoints, out / "raw_endpoints" / f"batch_{start:05d}.pt")
                row_stream.flush()
                writer.flush()
                for future in futures:
                    if future.done() and future not in joint_results:
                        joint_results[future] = future.result()
                status.update(generated=start + length, joint_completed=len(joint_results),
                              last_batch_seconds=time.monotonic() - tick,
                              raw_valid_so_far=sum(r["raw_valid"] for r in rows),
                              elapsed_seconds=time.monotonic() - started, updated_utc=now(),
                              cuda_reserved_MiB=torch.cuda.memory_reserved() / 2**20)
                write_json(out / "status.json", status)
                print(json.dumps(status), flush=True)
                del sampled, endpoints
        raw_summary = export_panel(out / "raw_analysis", rows, molecules, reference)
        del model
        torch.cuda.empty_cache()
        status.update(phase="joint_decoding_and_analysis", raw_summary=raw_summary)
        write_json(out / "status.json", status)
        for future in futures:
            if future not in joint_results:
                joint_results[future] = future.result()
            if len(joint_results) % 25 == 0:
                status.update(joint_completed=len(joint_results), updated_utc=now())
                write_json(out / "status.json", status)
    joint_rows, joint_mols = [], {}
    with Chem.SDWriter(str(out / "joint_decoded.sdf")) as writer:
        writer.SetKekulize(False)
        for index, binary, decoder in sorted(joint_results.values(), key=lambda x: x[0]):
            mol = Chem.Mol(binary) if binary is not None else None
            row = detailed_record(mol)
            row.update(generation_index=index, **target_fields, decoder=decoder,
                       requested_heavy_atoms=int(counts[index]))
            joint_rows.append(row)
            if mol is not None:
                mol.SetIntProp("generation_index", index)
                mol.SetDoubleProp(f"target_{property_name}_eV", args.target)
                writer.write(mol)
                joint_mols[index] = mol
    joint_summary = export_panel(out / "joint_analysis", joint_rows, joint_mols, reference)
    joint_summary["solver_status_counts"] = dict(Counter(r["decoder"]["status"] for r in joint_rows))
    write_json(out / "joint_analysis" / "summary.json", joint_summary)
    previous_summary = None
    if args.previous_sdf:
        previous_rows = []
        for index, mol in enumerate(Chem.SDMolSupplier(args.previous_sdf, sanitize=False, removeHs=False)):
            row = detailed_record(mol)
            row["generation_index"] = index
            previous_rows.append(row)
        if len(previous_rows) != args.n:
            raise ValueError("previous SDF lacks requested count; cannot compare without missing-record accounting")
        annotate_novelty(previous_rows, reference)
        previous_summary = aggregate(previous_rows)
        write_json(out / "previous_same_metrics.json", dict(source=args.previous_sdf, summary=previous_summary,
                                                           records=previous_rows))
    report = dict(manifest=manifest, raw=raw_summary, joint_decoded=joint_summary,
                  previous_same_metrics=previous_summary,
                  limitations=["Joint MAP is postprocessing, not raw model validity.",
                               "Generated heavy-atom coordinates were retained without reembedding.",
                               "Geometry screen is a heuristic, not a physical-stability calculation.",
                               "Condition fidelity is unmeasured: no XTB/STDA or calibrated property predictor was run.",
                               "Older and current checkpoints differ in architecture/loss/training age; comparison is not a controlled ablation."],
                  finished_utc=now(), elapsed_seconds=time.monotonic() - started)
    write_json(out / "summary.json", report)
    status.update(phase="complete", joint_completed=args.n, finished_utc=now(), exit_code=0,
                  elapsed_seconds=time.monotonic() - started, summary=str(out / "summary.json"))
    write_json(out / "status.json", status)
    print(json.dumps({"event": "complete", "raw": raw_summary, "joint": joint_summary}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--counts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--previous-sdf")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--target", type=float, default=0.)
    parser.add_argument("--property", choices=tuple(FORMULAS), default="est")
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--memory-fraction", type=float, default=.15)
    parser.add_argument("--joint-workers", type=int, default=4)
    parser.add_argument("--joint-seconds", type=float, default=2.)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if not args.worker:
        out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        with (out / "worker.log").open("xb") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=PACKAGE, env=env, start_new_session=True, close_fds=True)
        launch = dict(pid=child.pid, detached=True, command=command, output=str(out), started_utc=now())
        write_json(out / "launch.json", launch)
        print(json.dumps(launch, indent=2))
        return
    try:
        run(args)
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
        status.update(phase="failed", error=error, failed_utc=now(), exit_code=1)
        write_json(out / "status.json", status)
        raise


if __name__ == "__main__":
    main()
