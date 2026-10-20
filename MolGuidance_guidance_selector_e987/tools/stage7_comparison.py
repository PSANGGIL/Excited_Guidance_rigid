#!/usr/bin/env python3
"""Detached, bounded, from-scratch Stage 7 ablation screening.

The test uses the production architecture in FP32, not the tiny unit-test
network. Every arm receives identical batches, example priors and step seeds.
It is an early screening experiment, not a claim about converged performance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import Dataset, DataLoader, Sampler
import yaml

from molguidance.data_processing.dataset import collate
from molguidance.model_utils.load import read_config_file, data_module_from_config
from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
from molguidance.model_utils.provenance import data_manifest, verify_manifest, sha256_file
from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
from molguidance.models.correction import correction_losses
from molguidance.analysis.generation_callback import GenerationValidation
from tools.attach_stage7_condition_results import attach


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
DATA = REPO / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz"
STAGES = (
    "00_stage6_control", "01a_legacy_loss_control", "01_contracts",
    "01b_unweighted_loss_control", "02_feedback", "03_distribution", "04_correction",
)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    """Atomic replacement is restricted to this NEW experiment's own status."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temp.replace(path)


def event(root, **values):
    row = dict(time=now(), pid=os.getpid(), **values)
    with (root / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row, allow_nan=False), flush=True)


def source_hashes():
    paths = list((PROJECT / "molguidance").rglob("*.py"))
    paths += list((PROJECT / "configs/local/stage7").glob("*.yaml"))
    paths += [Path(__file__), PROJECT / "tools/attach_stage7_condition_results.py",
              PROJECT / "tools/joint_xtb_stda_pipeline.py",
              PROJECT / "tools/joint_xtb_stda_diffusion_coords.py"]
    return {str(path.relative_to(PROJECT)): sha256_file(path) for path in sorted(paths)}


def check_contract(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if source_hashes() != manifest["source_sha256"]:
        raise RuntimeError("source changed during comparison; do not mix different code versions")
    for name, expected in manifest["frozen_config_sha256"].items():
        if sha256_file(root / "config_sources" / name) != expected:
            raise RuntimeError(f"frozen experiment config changed: {name}")
    verify_manifest(manifest["data_manifest"], data_manifest(DATA))
    return manifest


class PairedDataset(Dataset):
    """Same augmentation/prior per (epoch,index), independent of model RNG."""
    def __init__(self, dataset, seed):
        self.dataset, self.seed = dataset, int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, int(key))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(
                self.seed + int(epoch) * len(self) + int(index)
            )
            return self.dataset[index]


class PairedBatches(Sampler):
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        self.epoch = 0

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        epoch = self.epoch
        self.epoch += 1
        order = torch.randperm(
            self.size, generator=torch.Generator().manual_seed(self.seed + epoch)
        ).tolist()
        for start in range(0, self.size, self.batch_size):
            yield [(epoch, index) for index in order[start:start + self.batch_size]]


class Progress(pl.Callback):
    def __init__(self, directory, seed):
        self.directory, self.seed = directory, int(seed)
        self.started = time.monotonic()
        self.gradient_norm = None

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        seed = self.seed + 1_000_000 + trainer.global_step
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        if trainer.global_step < 3:
            digest = hashlib.sha256()
            for value in (batch.ndata["x_0"], batch.ndata["s_1_true"], batch.prop):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            with (self.directory / "paired_batch_hashes.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(step=trainer.global_step, sha256=digest.hexdigest())) + "\n")

    def on_before_backward(self, trainer, model, loss):
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss; comparison arm stopped")

    def on_before_optimizer_step(self, trainer, model, optimizer):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        norms = torch.stack(torch._foreach_norm(gradients))
        if not torch.isfinite(norms).all():
            raise FloatingPointError("non-finite training gradients; comparison arm stopped")
        self.gradient_norm = float(torch.linalg.vector_norm(norms))

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        row = dict(time=now(), step=trainer.global_step, epoch=trainer.current_epoch,
                   elapsed_s=time.monotonic() - self.started,
                   learning_rate=trainer.optimizers[0].param_groups[0]["lr"],
                   gradient_norm_before_clip=self.gradient_norm,
                   peak_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        for key, value in trainer.callback_metrics.items():
            if "train_loss" in key or key == "train_total_loss":
                row[key] = float(value)
        with (self.directory / "training_steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_json(self.directory / "progress.json", dict(status="training", **row))
        if trainer.global_step % 25 == 0 or trainer.global_step <= 3:
            print(json.dumps(row, allow_nan=False), flush=True)

    def on_validation_epoch_end(self, trainer, model):
        row = dict(time=now(), step=trainer.global_step, epoch=trainer.current_epoch)
        for key, value in trainer.callback_metrics.items():
            if "val" in key:
                row[key] = float(value)
        with (self.directory / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def configure_cuda(fraction=.75):
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU comparison requires CUDA")
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats()


def config_for(root, stage, batch_size):
    spec = json.loads((root / "manifest.json").read_text())
    config = read_config_file(root / "config_sources" / f"{stage}.yaml")
    config["dataset"]["processed_data_dir"] = str(DATA)
    config["training"].update(batch_size=batch_size, num_workers=spec["data_workers"])
    config["training"]["evaluation"]["generation"]["enabled"] = False
    # Common step-based warmup and LR for every arm; documented screening override.
    batches_per_epoch = math.ceil(1264 / batch_size)
    config["lr_scheduler"]["warmup_length"] = spec["warmup_steps"] / batches_per_epoch
    config["lr_scheduler"]["base_lr"] = .0014 * batch_size / 640.
    config["training"]["output_dir"] = str(root / "stages" / stage)
    return config


def probe(root, batch_size):
    check_contract(root)
    configure_cuda()
    torch.manual_seed(17041)
    config = config_for(root, "04_correction", batch_size)
    model = model_from_config_with_guidance(config).cuda().train()
    dm = data_module_from_config(config)
    dm.setup("fit")
    dataset = dm.train_dataset
    sizes = dataset.node_idx_array[:, 1] - dataset.node_idx_array[:, 0]
    # A conservative batch: largest real molecules, not an easy random batch.
    indices = torch.argsort(sizes, descending=True)[:batch_size].tolist()
    paired = PairedDataset(dataset, 17041)
    batch = collate([paired[(0, index)] for index in indices]).to("cuda")
    # DGL's .to() does not copy arbitrary Python graph attributes.
    batch.prop = torch.stack([paired[(0, index)].prop for index in indices]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_scheduler"]["base_lr"])
    start = time.monotonic()
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = model(
            batch, t=torch.full((batch.batch_size,), .85, device="cuda"),
            graph_time=torch.full((batch.batch_size,), .85, device="cuda"),
            self_conditioning=True,
        )
        losses.update(correction_losses(model, batch, force=True))
        loss = model._total_from_feature_losses(losses)
        loss.backward()
        norms = torch.stack(torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None]))
        if not torch.isfinite(loss) or not torch.isfinite(norms).all():
            raise FloatingPointError("GPU probe produced non-finite loss/gradient")
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.)
        optimizer.step()
    torch.cuda.synchronize()
    result = dict(
        status="passed", batch_size=batch_size, iterations=2, loss=float(loss.detach()),
        total_slots=batch.num_nodes(), total_directed_edges=batch.num_edges(),
        max_real_atoms=int(sizes.max()), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        elapsed_s=time.monotonic() - start, dtype="float32", forced_correction=True,
        interpretation="largest-molecule preflight; not a steady-state training-utilization target",
    )
    write_json(root / "probes" / f"batch_{batch_size}.json", result)
    event(root, phase="gpu_probe", **result)


def train(root, stage, batch_size):
    spec = check_contract(root)
    configure_cuda()
    directory = root / "stages" / stage
    directory.mkdir(parents=True, exist_ok=False)
    config = config_for(root, stage, batch_size)
    with (directory / "config.yaml").open("x") as stream:
        yaml.safe_dump(config, stream)
    pl.seed_everything(spec["seed"], workers=True)
    model = model_from_config_with_guidance(config)
    dm = data_module_from_config(config)
    dm.setup("fit")
    worker_kwargs = dict(num_workers=spec["data_workers"])
    if spec["data_workers"]:
        worker_kwargs.update(multiprocessing_context="spawn", persistent_workers=True)
    train_loader = DataLoader(
        PairedDataset(dm.train_dataset, spec["seed"]),
        batch_sampler=PairedBatches(len(dm.train_dataset), batch_size, spec["seed"]),
        collate_fn=collate, generator=torch.Generator().manual_seed(spec["seed"]),
        **worker_kwargs,
    )
    val_loader = DataLoader(
        PairedDataset(dm.val_dataset, spec["seed"] + 20_000_000),
        batch_size=min(batch_size, 32), shuffle=False, collate_fn=collate,
        generator=torch.Generator().manual_seed(spec["seed"]), **worker_kwargs,
    )
    best = ModelCheckpoint(
        dirpath=directory / "checkpoints", monitor="val_total_loss", mode="min",
        save_top_k=1, save_last=True, save_on_train_epoch_end=False,
        filename="best-epoch{epoch:04d}-step{step:06d}", auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, precision="32-true", max_steps=spec["steps"],
        max_epochs=-1, logger=CSVLogger(str(directory), name="metrics", version="run"),
        callbacks=[best, Progress(directory, spec["seed"])],
        gradient_clip_algorithm="value", gradient_clip_val=1.,
        log_every_n_steps=1, num_sanity_val_steps=0, enable_progress_bar=False,
        enable_model_summary=False, check_val_every_n_epoch=None,
        val_check_interval=min(200, spec["steps"]), use_distributed_sampler=False,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    if trainer.global_step != spec["steps"]:
        raise RuntimeError("training did not reach the common step budget")
    trainer.save_checkpoint(directory / "checkpoints" / "matched_steps_final.ckpt")
    write_json(directory / "training_complete.json", dict(
        time=now(), optimizer_steps=trainer.global_step, epoch=trainer.current_epoch,
        matched_checkpoint=str(directory / "checkpoints/matched_steps_final.ckpt"),
        best_loss_checkpoint=best.best_model_path, parameters=sum(p.numel() for p in model.parameters()),
        batch_size=batch_size, learning_rate=config["lr_scheduler"]["base_lr"],
        warmup_steps=spec["warmup_steps"], peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
    ))


def evaluate(root, stage):
    spec = check_contract(root)
    configure_cuda()
    directory = root / "stages" / stage
    ckpt = directory / "checkpoints/matched_steps_final.ckpt"
    model = ClassifierFreeGuidance.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False).cuda().eval()
    panels = []
    for seed in (spec["seed"], spec["seed"] + 1):
        for label, target in (("validation", None), ("est0", 0.)):
            output = directory / "generation" / f"{label}_seed{seed}"
            callback = GenerationValidation(
                output, DATA, n_samples=spec["samples_per_panel"], batch_size=8,
                n_timesteps=spec["time_points"], seed=seed, target_est_eV=target,
            )
            report = callback.evaluate(model, epoch=0, step=spec["steps"])
            report_path = next(output.rglob("report.json"))
            panels.append(dict(panel=label, seed=seed, report=str(report_path)))
            event(root, phase="generation", stage=stage, panel=label, seed=seed,
                  requested=report["requested"], raw_valid=report["raw_valid_count"],
                  eligible=report["eligible_count"], aromatic=report["aromatic_count"],
                  geometry_eligible=report["geometry_eligible_count"])
    write_json(directory / "generation_complete.json", dict(
        checkpoint_sha256=sha256_file(ckpt), panels=panels, time=now(),
        checkpoint_selection="final identical optimizer-step budget, NOT each arm's best loss",
    ))


def xtb_environment(spec):
    env = os.environ.copy()
    xtb_env = Path(spec["xtb_env"])
    env["PATH"] = f"{xtb_env}/bin:/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe:" + env["PATH"]
    env["LD_LIBRARY_PATH"] = str(xtb_env / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["XTB4STDAHOME"] = "/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home"
    env["OMP_STACKSIZE"] = "2G"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    for program in ("xtb", "xtb4stda", "stda"):
        if shutil.which(program, path=env["PATH"]) is None:
            raise FileNotFoundError(f"missing calculation executable: {program}")
    return env


def xtb(root, stage):
    spec = check_contract(root)
    from tools import joint_xtb_stda_pipeline as pipeline
    # The existing coordinate preparation imports this module by its script name.
    sys.modules["joint_xtb_stda_pipeline"] = pipeline
    from tools.joint_xtb_stda_diffusion_coords import prepare_diffusion_coordinates
    env = xtb_environment(spec)

    def bounded_call(cmd, work, environment, log):
        with (work / log).open("w") as stream:
            return subprocess.run(
                cmd, cwd=work, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                timeout=spec["calculation_command_timeout_s"], check=False,
            ).returncode
    pipeline.call = bounded_call
    directory = root / "stages" / stage
    panels = json.loads((directory / "generation_complete.json").read_text())["panels"]
    for panel in panels:
        report_path = Path(panel["report"])
        report = json.loads(report_path.read_text())
        output = report_path.parent / "xtb_stda"
        output.mkdir(exist_ok=False)
        rows = []
        source = report_path.parent / "raw_generated.sdf"
        if source.stat().st_size:
            metadata = prepare_diffusion_coordinates(source, output / "inputs")
            work = output / "work"
            work.mkdir()
            by_name = {row["name"]: row for row in metadata}
            eligible = {row["generation_index"] for row in report["records"] if row["eligible"]}
            jobs = [path for path in sorted((output / "inputs").glob("generated_*.sdf"))
                    if int(by_name[path.stem]["generation_index"]) in eligible]
            for item in metadata:
                if item["status"] != "ok":
                    rows.append(dict(generation_index=item["generation_index"], status="preparation_failed",
                                     s1_eV="", t1_eV=""))
            with ThreadPoolExecutor(max_workers=spec["xtb_workers"]) as pool:
                futures = {pool.submit(pipeline.run_one, path, work, env): path for path in jobs}
                for future in as_completed(futures):
                    result = future.result()
                    path = futures[future]
                    rows.append(dict(
                        generation_index=by_name[path.stem]["generation_index"],
                        status="success" if result["status"] == "ok" else result["status"],
                        s1_eV=result.get("s1_energy_eV", ""), t1_eV=result.get("t1_energy_eV", ""),
                    ))
                    event(root, phase="xtb", stage=stage, panel=panel["panel"],
                          seed=panel["seed"], completed=len(rows), jobs=len(jobs), status=result["status"])
        manifest = dict(
            model_state_sha256=report["model_state_sha256"], source_sdf_sha256=sha256_file(source),
            method="diffusion heavy XYZ + RDKit H placement -> GFN2-xTB opt loose -> xTB4sTDA -> sTDA S1/T1",
            cpu_workers=spec["xtb_workers"], threads_per_calculation=1,
            timeout_per_external_command_s=spec["calculation_command_timeout_s"],
        )
        write_json(output / "calculation_manifest.json", manifest)
        with (output / "results.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["generation_index", "status", "s1_eV", "t1_eV"])
            writer.writeheader()
            writer.writerows(rows)
        combined = attach(report, rows, manifest)
        write_json(output / "condition_results.json", combined)
    write_json(directory / "xtb_complete.json", dict(time=now(), panel_count=len(panels)))


def collect_summary(root):
    summary = []
    spec = json.loads((root / "manifest.json").read_text())
    for stage in spec["stages"]:
        directory = root / "stages" / stage
        generation = directory / "generation_complete.json"
        if not generation.exists():
            continue
        for panel in json.loads(generation.read_text())["panels"]:
            path = Path(panel["report"])
            report = json.loads(path.read_text())
            row = dict(stage=stage, panel=panel["panel"], seed=panel["seed"],
                       requested=report["requested"])
            row.update({key: value for key, value in report.items() if key.endswith("_rate")})
            condition = path.parent / "xtb_stda/condition_results.json"
            if condition.exists():
                result = json.loads(condition.read_text())
                row.update({key: result[key] for key in (
                    "calculation_success", "calculation_coverage", "condition_hit_yield",
                    "condition_mae_eV_calculated_only", "aromatic_condition_hit_yield",
                    "geometry_aromatic_condition_hit_yield",
                )})
            summary.append(row)
    write_json(root / "comparison.json", dict(
        time=now(), records=summary, interpretation="early matched-step screening; not converged performance",
        caveat="two sampling seeds; one training seed. Small counts and survivors-only MAE are not definitive.",
    ))


def child(root, mode, stage=None, batch_size=None):
    check_contract(root)
    label = f"{mode}_{stage or batch_size}"
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", mode, "--output", str(root)]
    if stage:
        command += ["--stage", stage]
    if batch_size:
        command += ["--batch-size", str(batch_size)]
    with (root / "logs" / f"{label}.log").open("x") as stream:
        process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT)
        write_json(root / "status.json", dict(time=now(), phase=mode, stage=stage,
                                            worker_pid=os.getpid(), child_pid=process.pid))
        return process.wait()


def orchestrate(root):
    spec = check_contract(root)
    xtb_environment(spec)  # fail early, not after all GPU work
    selected = None
    for batch_size in spec["batch_candidates"]:
        code = child(root, "probe", batch_size=batch_size)
        event(root, phase="probe_exit", batch_size=batch_size, returncode=code)
        if code == 0:
            result = json.loads((root / "probes" / f"batch_{batch_size}.json").read_text())
            if result["peak_allocated_bytes"] / result["device_total_bytes"] <= .70:
                selected = batch_size
                break
    if selected is None:
        raise RuntimeError("no batch size passed the conservative GPU preflight")
    write_json(root / "selected_batch.json", dict(batch_size=selected,
               learning_rate=.0014 * selected / 640., same_for_every_arm=True))
    reference_batches = None
    for stage in spec["stages"]:
        if child(root, "train", stage, selected):
            raise RuntimeError(f"training failed in {stage}; see its log")
        batch_hashes = (root / "stages" / stage / "paired_batch_hashes.jsonl").read_text()
        if reference_batches is None:
            reference_batches = batch_hashes
        elif batch_hashes != reference_batches:
            raise RuntimeError("arms did not receive identical first training batches")
        if child(root, "evaluate", stage):
            raise RuntimeError(f"generation failed in {stage}; see its log")
        collect_summary(root)
    # All GPU arms finish before the bounded, four-worker CPU calculation phase.
    for stage in spec["stages"]:
        if child(root, "xtb", stage):
            raise RuntimeError(f"XTB pipeline failed in {stage}; see its log")
        collect_summary(root)
    verify_manifest(spec["data_manifest"], data_manifest(DATA))
    write_json(root / "status.json", dict(time=now(), phase="complete", worker_pid=os.getpid(),
                                        data_unchanged=True, stages=spec["stages"]))
    event(root, phase="complete", data_unchanged=True)


def launch(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "config_sources", "probes"):
        (root / name).mkdir()
    for stage in STAGES:
        config = read_config_file(PROJECT / "configs/local/stage7" / f"{stage}.yaml")
        with (root / "config_sources" / f"{stage}.yaml").open("x") as stream:
            yaml.safe_dump(config, stream)
    spec = dict(
        created=now(), source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        source_sha256=source_hashes(), data_manifest=data_manifest(DATA), steps=args.steps,
        frozen_config_sha256={path.name: sha256_file(path) for path in sorted((root / "config_sources").glob("*.yaml"))},
        seed=args.seed, warmup_steps=min(100, max(args.steps // 4, 1)),
        batch_candidates=args.batch_candidates, data_workers=args.data_workers,
        samples_per_panel=args.samples_per_panel, time_points=args.time_points,
        stages=args.stages, xtb_env=str(args.xtb_env), xtb_workers=4,
        calculation_command_timeout_s=600, precision="32-true",
        intended_samples_per_arm=4 * args.samples_per_panel,
        purpose="bounded from-scratch ablation screening; not convergence/production training",
    )
    write_json(root / "manifest.json", spec)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        PYTHONPATH=f"{REPO}:{PROJECT}", OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4", WANDB_MODE="disabled",
    )
    with (root / "worker.log").open("x") as stream:
        process = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).resolve()),
             "--mode", "orchestrate", "--output", str(root)],
            cwd=PROJECT, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        )
    write_json(root / "launch.json", dict(time=now(), pid=process.pid, session_id=process.pid,
                                        output=str(root), detached=True))
    print(json.dumps(dict(pid=process.pid, output=str(root), detached=True)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["launch", "orchestrate", "probe", "train", "evaluate", "xtb"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--stages", choices=STAGES, nargs="+", default=list(STAGES))
    parser.add_argument("--batch-candidates", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--seed", type=int, default=17041)
    parser.add_argument("--samples-per-panel", type=int, default=32)
    parser.add_argument("--time-points", type=int, default=100)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--xtb-env", type=Path, default=Path("/home/jovyan/cpu-only-1-datavol-1/xtb_env"))
    args = parser.parse_args()
    if (args.steps < 1 or args.samples_per_panel < 1 or args.time_points < 2
            or args.data_workers < 0 or min(args.batch_candidates) < 1):
        parser.error("invalid test dimensions")
    root = args.output.resolve()
    try:
        if args.mode == "launch":
            launch(args)
        elif args.mode == "orchestrate":
            orchestrate(root)
        elif args.mode == "probe":
            if not args.batch_size:
                parser.error("--batch-size is required for probe")
            probe(root, args.batch_size)
        elif args.mode == "train":
            if not args.stage or not args.batch_size:
                parser.error("--stage and --batch-size are required for train")
            train(root, args.stage, args.batch_size)
        elif args.mode == "evaluate":
            evaluate(root, args.stage)
        else:
            xtb(root, args.stage)
    except Exception as exc:
        if root.exists():
            write_json(root / f"failure_{args.mode}_{args.stage or args.batch_size}.json",
                       dict(time=now(), error=repr(exc), traceback=traceback.format_exc()))
            if args.mode == "orchestrate":
                write_json(root / "status.json", dict(time=now(), phase="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
