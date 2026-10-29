#!/usr/bin/env python3
"""Detached fresh training, gated by completed Stage8 integration verification.

No implicit checkpoint warm-start. Saves expanded config, data/source manifests,
CSV metrics, last/best checkpoints, strict generation panels, and live status.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import yaml

from molguidance.model_utils.load import read_config_file, data_module_from_config
from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
from molguidance.model_utils.provenance import data_manifest, verify_manifest, sha256_file
from molguidance.analysis.generation_callback import GenerationValidation, GeneratedQualityCheckpoint
from molguidance.data_processing.dataset import collate
from molguidance.models.correction import correction_losses

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz"


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


class TrainingStatus(pl.Callback):
    def __init__(self, output):
        self.output = output
        self.started = time.monotonic()
        self.active_batch = None
        self.failure_saved = False
        self.step_losses = []
        self.molecule_counts = []
        self.epoch_started = self.started
        self.last_loss = None

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.active_batch = batch
        self.batch_idx = int(batch_idx)
        self.last_loss = None
        self.pre_forward_rng = dict(
            torch_cpu=torch.get_rng_state(),
            torch_cuda=torch.cuda.get_rng_state_all() if model.device.type == "cuda" else [],
            python=random.getstate(), numpy=np.random.get_state())

    def record_failure(self, trainer, model, reason):
        if self.failure_saved:
            return
        self.failure_saved = True
        try:
            graph = None if self.active_batch is None else self.active_batch.to("cpu")
            if graph is not None and hasattr(self.active_batch, "prop"):
                graph.prop = self.active_batch.prop.detach().cpu()
            bad_gradients = [name for name, p in model.named_parameters()
                             if p.grad is not None and not torch.isfinite(p.grad).all()]
            torch.save(dict(
                reason=reason, epoch=int(trainer.current_epoch), global_step=int(trainer.global_step),
                batch_index=getattr(self, "batch_idx", None), batch=graph,
                pre_forward_rng=getattr(self, "pre_forward_rng", None),
                loss=self.last_loss, nonfinite_gradient_parameters=bad_gradients,
            ), self.output / "failure_diagnostic.pt")
            # Forensics only: never automatically resume a failed batch. This
            # captures weights and optimizer BEFORE the guarded update occurs.
            trainer.save_checkpoint(self.output / "failure_before_optimizer.ckpt")
        except Exception:
            print("FAILURE DIAGNOSTIC SAVE ERROR", traceback.format_exc(), flush=True)

    def on_before_backward(self, trainer, model, loss):
        self.last_loss = loss.detach().cpu()
        if not torch.isfinite(loss):
            self.record_failure(trainer, model, "nonfinite loss before backward")
            raise FloatingPointError("nonfinite training loss before backward")

    def on_train_epoch_start(self, trainer, model):
        self.step_losses = []
        self.molecule_counts = []
        self.epoch_started = time.monotonic()

    def update(self, trainer, phase):
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                number = float(value.detach().cpu())
                metrics[key] = number if torch.isfinite(value) else None
        write(self.output / "status.json", dict(
            phase=phase, pid=os.getpid(), epoch=int(trainer.current_epoch),
            optimizer_step=int(trainer.global_step), elapsed_s=time.monotonic() - self.started,
            metrics=metrics, cuda_allocated_MiB=torch.cuda.memory_allocated() / 2**20,
            cuda_reserved_MiB=torch.cuda.memory_reserved() / 2**20))

    def on_before_optimizer_step(self, trainer, model, optimizer):
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                self.record_failure(trainer, model, f"nonfinite gradient: {name}")
                raise FloatingPointError(f"nonfinite gradient: {name}")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        self.step_losses.append(float(loss.detach()))
        self.molecule_counts.append(int(batch.batch_size))
        if trainer.global_step % 10 == 0:
            self.update(trainer, "training")

    def on_train_epoch_end(self, trainer, model):
        # A validation-end checkpoint is saved before the epoch-end callbacks.
        # Lightning replays that boundary on resume without a train-epoch-start
        # hook or new batches. Do not crash or write a fabricated zero-loss row.
        if not self.step_losses:
            return
        record = dict(
            epoch=int(trainer.current_epoch), optimizer_step=int(trainer.global_step),
            mean_step_train_loss=sum(self.step_losses) / max(len(self.step_losses), 1),
            molecule_weighted_train_loss=sum(v * n for v, n in zip(self.step_losses, self.molecule_counts)) /
                                         max(sum(self.molecule_counts), 1),
            training_graphs=sum(self.molecule_counts), batches=len(self.step_losses),
            epoch_seconds=time.monotonic() - self.epoch_started,
            val_total_loss=(float(trainer.callback_metrics["val_total_loss"])
                            if "val_total_loss" in trainer.callback_metrics else None),
        )
        with (self.output / "epoch_history.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.update(trainer, "training")

    def on_validation_end(self, trainer, model):
        for key, value in trainer.callback_metrics.items():
            if key.startswith("val_") and isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError(f"nonfinite validation metric: {key}")
        self.update(trainer, "sanity_validation" if trainer.sanity_checking else "validation")
        if not trainer.sanity_checking:
            print("EPOCH", trainer.current_epoch, "STEP", trainer.global_step,
                  "VAL", float(trainer.callback_metrics["val_total_loss"]), flush=True)


def training_checkpoints(output):
    # The installed Lightning version updates save_last=True only when its
    # monitored top-k callback saves. An independent, unmonitored checkpoint
    # guarantees a real latest completed epoch even when validation worsens.
    return [
        ModelCheckpoint(
            dirpath=output / "checkpoints", monitor="val_total_loss", mode="min",
            save_top_k=5, save_last=False, every_n_epochs=1,
            save_on_train_epoch_end=False, auto_insert_metric_name=False,
            filename="epoch{epoch:05d}-val{val_total_loss:.6f}"),
        ModelCheckpoint(
            dirpath=output / "checkpoints", monitor=None,
            save_top_k=1, save_last=False, every_n_epochs=1,
            save_on_train_epoch_end=False, auto_insert_metric_name=False,
            enable_version_counter=False, filename="last"),
    ]


def train(args, output):
    data_dir = Path(getattr(args, "data", DATA)).resolve()
    verification = json.loads((args.verification / "verification.json").read_text())
    if not verification.get("complete") or not verification.get("data_unchanged"):
        raise ValueError("complete integration verification is required before long training")
    if args.config.name not in {entry["config"] for entry in verification["configs"]}:
        raise ValueError("production config was not integration-tested")
    for relative, expected in verification["source_sha256"].items():
        if sha256_file(PROJECT / relative) != expected:
            raise ValueError(f"code changed after verification: {relative}")
    verify_manifest(verification["data_before"], data_manifest(data_dir))
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    pl.seed_everything(args.seed, workers=True)
    config = read_config_file(args.config)
    if config != verification["expanded_configs"][args.config.name]:
        raise ValueError("production config differs from integration-tested settings")
    config["dataset"]["processed_data_dir"] = str(data_dir)
    config["training"]["output_dir"] = str(output)
    config["wandb"]["mode"] = "disabled"
    # Keep this record usable by the ordinary checkpoint loaders.
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    source = output / "source_snapshot"
    source.mkdir()
    for name in ("molguidance", "configs", "tests", "tools", "docs"):
        shutil.copytree(PROJECT / name, source / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.orig"))
    shutil.copy2(PROJECT / "README.md", source / "README.md")
    diff = subprocess.check_output(["git", "diff", "--", str(PROJECT)], cwd=PROJECT, text=True)
    (output / "tracked_changes.patch").write_text(diff)
    write(output / "manifest.json", dict(
        fresh_training=args.resume is None, resume_checkpoint=None if args.resume is None else str(args.resume),
        resume_sha256=None if args.resume is None else sha256_file(args.resume),
        seed=args.seed, precision="float32_true_no_tf32",
        verification=str(args.verification), verification_sha256=sha256_file(args.verification / "verification.json"),
        source_sha256={str(p.relative_to(source)): sha256_file(p) for p in source.rglob("*") if p.is_file()},
        data=verification["data_before"], gpu_memory_fraction_limit=args.memory_fraction,
        condition_name=config["dataset"]["conditioning"]["property"],
        base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip()))

    data_module = data_module_from_config(config)
    data_module.setup("fit")
    loader = data_module.train_dataloader()
    sampler = loader.batch_sampler
    batches = list(sampler)
    worst = max(batches, key=lambda batch: int(sampler.costs[batch].sum()))
    # Full production-width, production-budget forward/backward/update on the
    # largest planned batch. The probe model is DISCARDED before fresh training.
    probe = model_from_config_with_guidance(config).cuda().train()
    examples = [data_module.train_dataset[i] for i in worst]
    batch = collate(examples).to("cuda")
    batch.prop = torch.stack([graph.prop for graph in examples]).cuda()
    torch.cuda.reset_peak_memory_stats()
    losses = probe(batch)
    losses.update(correction_losses(probe, batch, force=True))
    total = probe._total_from_feature_losses(losses)
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite production-batch probe")
    total.backward()
    parameters = [p for p in probe.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("nonfinite production-batch probe gradients")
    torch.nn.utils.clip_grad_value_(parameters, 1.)
    optimizer = torch.optim.Adam(parameters, lr=config["lr_scheduler"]["base_lr"])
    optimizer.step()
    torch.cuda.synchronize()
    write(output / "capacity_probe.json", dict(
        passed=True, molecules=len(worst), directed_edges=int(batch.num_edges()),
        maximum_estimated_directed_edges=int(sampler.costs[worst].sum()),
        edge_budget=int(sampler.max_edges), batches_per_epoch=len(batches),
        batch_sizes=[len(b) for b in batches], loss=float(total.detach()),
        peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
        total_gpu_MiB=torch.cuda.get_device_properties(0).total_memory / 2**20))
    del probe, batch, losses, total, parameters, optimizer
    torch.cuda.empty_cache()
    pl.seed_everything(args.seed, workers=True)
    model = model_from_config_with_guidance(config)
    generation_args = copy.deepcopy(config["training"]["evaluation"]["generation"])
    generation_args.pop("enabled", None)
    generation_args.setdefault("property_name", config["dataset"]["conditioning"]["property"])
    generation = GenerationValidation(output / "generation_validation", data_dir, **generation_args)
    status = TrainingStatus(output)
    callbacks = [
        *training_checkpoints(output), LearningRateMonitor(logging_interval="step"), generation,
        GeneratedQualityCheckpoint(
            dirpath=output / "checkpoints/generation_quality",
            monitor="gen_geometry_eligible_rate", mode="max", save_top_k=3,
            every_n_epochs=generation.every_n_epochs, save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
            filename="epoch{epoch:05d}-geometry{gen_geometry_eligible_rate:.4f}"),
        status,
    ]
    trainer_options = copy.deepcopy(config["training"]["trainer_args"])
    trainer_options.update(enable_progress_bar=False, num_sanity_val_steps=2,
                           use_distributed_sampler=False, check_val_every_n_epoch=1)
    trainer = pl.Trainer(
        **trainer_options, default_root_dir=output, callbacks=callbacks,
        logger=CSVLogger(save_dir=output, name="metrics", version="train",
                         flush_logs_every_n_steps=10))
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=data_module.val_dataloader(),
                ckpt_path=None if args.resume is None else str(args.resume),
                # These are this project's trusted full Lightning checkpoints,
                # including Path metadata and optimizer/loop state, not tensor-only weights.
                weights_only=False if args.resume is not None else None)
    status.update(trainer, "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17041)
    parser.add_argument("--resume", type=Path, help="Explicit compatible checkpoint; output must still be NEW")
    parser.add_argument("--memory-fraction", type=float, default=.7)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.config, args.verification, output = args.config.resolve(), args.verification.resolve(), args.output.resolve()
    args.data = args.data.resolve(strict=True)
    if args.resume is not None:
        args.resume = args.resume.resolve(strict=True)
    if not 0 < args.memory_fraction <= .9:
        parser.error("memory-fraction must be in (0, .9]")
    if not args.worker:
        # Check before launch as well as in the detached worker.
        verification = json.loads((args.verification / "verification.json").read_text())
        if not verification.get("complete"):
            raise ValueError("verification is not complete")
        output.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(PYTHONPATH=f"{PROJECT.parent}:{PROJECT}",
                           WANDB_MODE="disabled", OMP_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="1",
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                   "--config", str(args.config), "--verification", str(args.verification),
                   "--data", str(args.data),
                   "--output", str(output), "--seed", str(args.seed),
                   "--memory-fraction", str(args.memory_fraction)]
        if args.resume is not None:
            command += ["--resume", str(args.resume)]
        with (output / "train.log").open("x") as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=environment,
                                       start_new_session=True, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT)
        launch = dict(pid=process.pid, detached=True, command=command, output=str(output))
        write(output / "launch.json", launch)
        print(json.dumps(launch), flush=True)
    else:
        try:
            train(args, output)
        except BaseException:
            write(output / "status.json", dict(phase="failed", traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
