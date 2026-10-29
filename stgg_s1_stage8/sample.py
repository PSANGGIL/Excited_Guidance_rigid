"""One fixed-S1 raw generation panel; no projection or coordinate re-embedding."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get("STGG_WORKSPACE_ROOT", Path(__file__).resolve().parents[1]))
PACKAGE = Path(os.environ.get("STGG_MODEL_PACKAGE", ROOT / "MolGuidance_guidance_selector_e987"))
sys.path.insert(0, str(PACKAGE))
from prepare import write_json


def run(args):
    import numpy as np
    import pandas as pd
    import torch
    from rdkit import Chem, RDLogger
    from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
    from molguidance.model_utils.provenance import data_manifest, verify_manifest, state_dict_sha256
    from tools.generate_stage8_panel import freeze_checkpoint, detailed_record, export_panel
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = dict(phase="loading", pid=os.getpid(), sid=os.getsid(0), generated=0, requested=args.n)
    write_json(out / "status.json", status)
    ckpt, checkpoint_hash = freeze_checkpoint(args.checkpoint, out / "checkpoint_snapshot.ckpt")
    model = ClassifierFreeGuidance.load_from_checkpoint(out / "checkpoint_snapshot.ckpt", map_location="cpu", weights_only=False)
    assert state_dict_sha256(model.state_dict()) == state_dict_sha256(ckpt["state_dict"])
    verify_manifest(model.data_manifest, data_manifest(args.data))
    normalization = torch.load(args.data / "train_data_property_normalization.pt", weights_only=False)
    if normalization["property_names"] != "s1":
        raise ValueError("refusing EST/OS checkpoint or normalization for S1 sampling")
    generator = np.random.default_rng(args.seed)
    train = torch.load(args.data / "train_data_processed.pt", weights_only=False, map_location="cpu")
    # Explicit train-only P(N | S1), not an arbitrary fixed molecule size.
    energy = train["properties"].numpy().reshape(-1).astype(float)
    logits = -.5 * ((energy - args.target) / .15) ** 2
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    n_atoms = (train["node_idx_array"][:, 1] - train["node_idx_array"][:, 0]).numpy()
    counts = n_atoms[generator.choice(len(energy), size=args.n, p=probabilities)]
    np.save(out / "fixed_atom_counts.npy", counts)
    reference = set(train["smiles"])
    for split in ("val", "test"):
        reference.update(torch.load(args.data / f"{split}_data_processed.pt", weights_only=False)["smiles"])
    del train
    manifest = dict(target_s1_eV=args.target, property="s1", n=args.n, cfg=args.cfg,
                    checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                    checkpoint_epoch=int(ckpt["epoch"]), optimizer_step=int(ckpt["global_step"]),
                    seed=args.seed, timesteps=args.timesteps, batch=args.batch,
                    counts_policy="train-only Gaussian S1 kernel, bandwidth=0.15 eV",
                    precision="float32, TF32 disabled", data_manifest=model.data_manifest,
                    novelty_scope="used STGG 100k-derived train+val+test; full 2.9M membership NOT checked",
                    coordinates="original generated heavy-atom coordinates; no ETKDG or force-field repair",
                    property_fidelity="unmeasured until independent excited-state calculation")
    write_json(out / "manifest.json", manifest)
    del ckpt
    model.cuda().eval()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows, molecules = [], {}
    (out / "raw_endpoints").mkdir()
    with torch.inference_mode(), (out / "raw_records.jsonl").open("x") as log:
        for offset in range(0, args.n, args.batch):
            size = min(args.batch, args.n - offset)
            sampled = model.sample_random_sizes(
                size, device=model.device, n_timesteps=args.timesteps,
                number_of_atoms=counts[offset:offset+size].tolist(),
                properties_for_sampling=args.target, property_name="s1", dataset_name=model.dataset_name,
                properties_handle_method="concatenate_sum", training_mode=False,
                normalization_file_path=str(args.data / "train_data_property_normalization.pt"),
                guide_w={k: args.cfg for k in model.canonical_feat_order},
                dfm_type="campbell", guidance_format="linear", where_to_apply_guide="probabilities")
            if len(sampled) != size:
                raise RuntimeError("sampler altered the denominator")
            endpoints = []
            for j, sample in enumerate(sampled):
                index = offset + j
                graph = sample.slot_graph
                src, dst = graph.edges()
                endpoints.append(dict(generation_index=index, target_s1_eV=args.target,
                    requested_heavy_atoms=int(counts[index]), edge_index=torch.stack((src, dst)).cpu(),
                    nodes={k: v.detach().cpu() for k, v in graph.ndata.items() if k.endswith(("_1", "_endpoint_probs"))},
                    edges={k: v.detach().cpu() for k, v in graph.edata.items() if k.endswith(("_1", "_endpoint_probs"))}))
                row = detailed_record(sample.rdkit_mol)
                row.update(generation_index=index, target_s1_eV=args.target, requested_heavy_atoms=int(counts[index]),
                           sampled_real_atoms=int(sample.num_atoms), total_slots=int(sample.num_slots))
                rows.append(row)
                if row["raw_valid"]:
                    molecules[index] = Chem.Mol(sample.rdkit_mol)
                    molecules[index].SetDoubleProp("target_s1_eV", args.target)
                log.write(json.dumps(row, allow_nan=False) + "\n")
            torch.save(endpoints, out / "raw_endpoints" / f"batch_{offset:05d}.pt")
            log.flush()
            status.update(phase="generating", generated=len(rows), raw_valid=sum(r["raw_valid"] for r in rows),
                          elapsed_seconds=time.monotonic()-start)
            write_json(out / "status.json", status)
            print(json.dumps(status), flush=True)
    del model
    torch.cuda.empty_cache()
    summary = export_panel(out / "raw_analysis", rows, molecules, reference)
    db = pd.read_csv(args.data / "db_structure_properties.csv")
    db = db[db.split == "train"]
    baselines = {}
    for name, subset in [("train_all", db), ("train_target_plusminus_0p1_eV", db[abs(db.s1_eV-args.target) <= .1])]:
        rings = Counter(size for value in subset.ring_sizes for size in json.loads(value))
        baselines[name] = dict(molecules=len(subset), aromatic_molecules=int(subset.aromatic.sum()),
            aromatic_fraction=float(subset.aromatic.mean()) if len(subset) else None,
            ring_size_counts=dict(rings), five_six_ring_fraction=(rings[5]+rings[6])/max(1,sum(rings.values())),
            mean_heavy_atoms=float(subset.heavy_atoms.mean()) if len(subset) else None)
    write_json(out / "summary.json", dict(manifest=manifest, raw=summary, database=baselines,
        limitations=["This measures generated graphs, not whether their actual S1 is 4 eV.",
                     "Not a causal aromaticity-only ablation: data source, molecule sizes, chemistry and geometry differ.",
                     "Novelty checked against the used 100k-derived corpus, not all 2.9M STGG rows.",
                     "First scheduled panel is an interim training diagnostic, not a claim of convergence."],
        elapsed_seconds=time.monotonic()-start))
    status.update(phase="complete", elapsed_seconds=time.monotonic()-start)
    write_json(out / "status.json", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=float, default=4.0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--memory-fraction", type=float, default=.08)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not args.worker:
        args.output = args.output.resolve()
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
        with args.output.with_suffix(".log").open("xb") as log:
            child = subprocess.Popen(command, env=os.environ.copy(), cwd=ROOT,
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        write_json(args.output.with_name(args.output.name + "_launch.json"), dict(pid=child.pid, detached=True, command=command))
        print(json.dumps(dict(pid=child.pid, detached=True, output=str(args.output))), flush=True)
        sys.exit(0)
    try:
        run(args)
    except BaseException:
        if args.output.is_dir():
            write_json(args.output / "status.json", dict(phase="failed", traceback=traceback.format_exc()))
        raise
