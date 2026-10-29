"""Detached STGG-S1 comparison, preserving the separate HP1KG/EST process.

Freeze source -> convert and audit -> real GPU probe -> fresh CFG training.
After >=10k optimizer steps, launch a detached 1,000-sample S1=4eV panel from
the best completed-validation checkpoint. Training continues during sampling.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ.get("STGG_WORKSPACE_ROOT", Path(__file__).resolve().parents[1]))
PACKAGE = Path(os.environ.get("STGG_MODEL_PACKAGE", ROOT / "MolGuidance_guidance_selector_e987"))
sys.path.insert(0, str(PACKAGE))
from prepare import prepare, write_json, sha256


def train(args, data_dir, conversion):
    import numpy as np
    import torch
    import pytorch_lightning as pl
    import yaml
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.loggers import CSVLogger
    from molguidance.model_utils.load import read_config_file, data_module_from_config
    from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
    from molguidance.model_utils.provenance import data_manifest, state_dict_sha256
    from molguidance.data_processing.dataset import collate
    from molguidance.models.correction import correction_losses
    from tools.train_stage8 import TrainingStatus, training_checkpoints

    output = args.output / "train"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    pl.seed_everything(args.seed, workers=True)
    if conversion["oracle_identity_total"] != conversion["oracle_identity_expected"] or not conversion["complete"]:
        raise ValueError("full accepted-corpus codec identity must pass before training")
    config = read_config_file(PACKAGE / "configs/local/stage8/08_longtrain_cfg.yaml")
    original_config = copy.deepcopy(config)
    config["dataset"].update(processed_data_dir=str(data_dir), raw_data_dir=str(args.source),
        atom_map=conversion["atom_map"], n_atom_states=conversion["atom_state_count"] + 1,
        training_split="train", use_first_half_training_set=False)
    config["dataset"]["conditioning"]["property"] = "s1"
    config["mol_fm"]["n_atom_states"] = conversion["atom_state_count"] + 1
    config["training"]["output_dir"] = str(output)
    config["training"]["evaluation"]["generation"]["enabled"] = False
    config["wandb"].update(mode="disabled", name="stgg-s1-stage8-cfg", save_dir=str(output))
    write_json(args.output / "status.json", dict(phase="building_dataset_and_kekule_cache", pid=os.getpid(),
               dataset_molecules=conversion["oracle_identity_total"], target_s1_eV=args.target))
    dm = data_module_from_config(config)
    dm.setup("fit")
    print("DATASET READY", len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset), flush=True)
    train_loader, val_loader = dm.train_dataloader(), dm.val_dataloader()
    sampler = train_loader.batch_sampler
    batches = list(sampler)
    flat = [i for batch in batches for i in batch]
    if sorted(flat) != list(range(len(dm.train_dataset))):
        raise ValueError("batch sampler omitted or duplicated an example")
    # The original 1,264-molecule run warmed up over 5*43 optimizer steps.
    # Retain that step scale, rather than lengthening warmup ~60x on STGG.
    config["lr_scheduler"]["warmup_length"] = args.warmup_steps / len(batches)
    config["training"]["trainer_args"]["max_epochs"] = args.max_epochs
    if args.smoke:
        config["training"]["trainer_args"].update(max_epochs=1, limit_train_batches=3, limit_val_batches=2)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "reference_hp1kg_config.yaml").write_text(yaml.safe_dump(original_config, sort_keys=False))
    write_json(args.output / "status.json", dict(phase="gpu_capacity_and_gradient_probe", pid=os.getpid(),
               batches_per_epoch=len(batches), target_s1_eV=args.target))
    worst = max(batches, key=lambda b: int(sampler.costs[b].sum()))
    model = model_from_config_with_guidance(config).cuda().train()
    initial = state_dict_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr_scheduler"]["base_lr"])
    torch.cuda.reset_peak_memory_stats()
    losses_seen = []
    for _ in range(3):
        # Dataset properties must be normalized exactly as __getitem__, not twice.
        batch = collate([dm.train_dataset[i] for i in worst])
        prop = batch.prop
        batch = batch.to("cuda")
        batch.prop = prop.cuda()
        optimizer.zero_grad(set_to_none=True)
        losses = model(batch)
        losses.update(correction_losses(model, batch, force=True))
        loss = model._total_from_feature_losses(losses)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite STGG probe loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite STGG probe gradient")
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        losses_seen.append(float(loss.detach()))
    if state_dict_sha256(model.state_dict()) == initial:
        raise AssertionError("optimizer did not change model weights")
    torch.cuda.synchronize()
    normalization = torch.load(data_dir / "train_data_property_normalization.pt", weights_only=False)
    probe = dict(passed=True, losses=losses_seen, optimizer_steps=3, probe_weights_discarded=True,
                 worst_batch_molecules=len(worst), worst_estimated_directed_edges=int(sampler.costs[worst].sum()),
                 edge_budget=int(sampler.max_edges), batches_per_epoch=len(batches),
                 batch_size_min=min(map(len,batches)), batch_size_max=max(map(len,batches)),
                 peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                 peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
                 target_s1_eV=args.target,
                 standardized_target=float((args.target-normalization["mean"])/normalization["std"]))
    write_json(output / "verification.json", probe)
    del model, optimizer, batch, loss, losses
    gc.collect()
    torch.cuda.empty_cache()
    pl.seed_everything(args.seed, workers=True)
    model = model_from_config_with_guidance(config)
    write_json(output / "manifest.json", dict(
        fresh_training=True, resumed_checkpoint=None, model_family="unchanged Stage8 CFG; new dataset-dependent vocabulary",
        source_package=str(PACKAGE), precision="32-true; TF32 disabled", data=data_manifest(data_dir),
        property="s1", target_s1_eV=args.target, seed=args.seed, gpu_memory_fraction_limit=args.memory_fraction,
        existing_hp1kg_process="not stopped or modified", warmup_optimizer_steps=args.warmup_steps,
        panel_after_minimum_optimizer_steps=args.panel_after_steps,
        panel_note="Interim diagnostic from best completed validation checkpoint; not proof of convergence.",
        comparison_limitations=conversion["comparison_limitations"]))

    class AutoPanel(Callback):
        def __init__(self):
            self.launched = False

        def on_train_epoch_end(self, trainer, module):
            if trainer.sanity_checking or self.launched or args.smoke or trainer.global_step < args.panel_after_steps:
                return
            best = next(c for c in trainer.checkpoint_callbacks if c.monitor == "val_total_loss")
            checkpoint = Path(best.best_model_path)
            if not checkpoint.is_file():
                return
            destination = args.output / "generation_s1_4ev_n1000"
            frozen = args.output / "panel_best_checkpoint.ckpt"
            # Immutable copy made in the checkpoint-save callback sequence: no racing reader.
            shutil.copy2(checkpoint, frozen)
            command = [sys.executable, "-u", str(Path(__file__).with_name("sample.py")),
                       "--worker",
                       "--checkpoint", str(frozen), "--data", str(data_dir), "--output", str(destination),
                       "--target", str(args.target), "--n", str(args.generate_n), "--batch", "4",
                       "--memory-fraction", ".08", "--seed", str(args.seed + 1)]
            with (args.output / "generation.log").open("xb") as log:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    env=os.environ.copy(), cwd=ROOT, start_new_session=True, close_fds=True)
            write_json(args.output / "generation_launch.json", dict(pid=process.pid, detached=True, command=command,
                       selection="best val_total_loss so far", selected_checkpoint=str(checkpoint),
                       selected_epoch=int(torch.load(frozen, map_location="cpu", weights_only=False)["epoch"]),
                       trigger_optimizer_step=int(trainer.global_step)))
            self.launched = True

    checkpoints = training_checkpoints(output)
    status = TrainingStatus(output)
    options = copy.deepcopy(config["training"]["trainer_args"])
    options.update(enable_progress_bar=False, num_sanity_val_steps=2,
                   use_distributed_sampler=False, check_val_every_n_epoch=1)
    trainer = pl.Trainer(**options, default_root_dir=output,
        callbacks=[*checkpoints, status, AutoPanel()],
        logger=CSVLogger(save_dir=output, name="metrics", version="train", flush_logs_every_n_steps=10))
    write_json(args.output / "status.json", dict(phase="training", pid=os.getpid(), sid=os.getsid(0),
               live_status=str(output / "status.json"), target_s1_eV=args.target,
               generation_scheduled_after_min_optimizer_steps=args.panel_after_steps))
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    status.update(trainer, "complete")
    if args.smoke:
        saved = torch.load(output / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
        restored = type(model).load_from_checkpoint(output / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
        if state_dict_sha256(restored.state_dict()) != state_dict_sha256(saved["state_dict"]):
            raise ValueError("smoke checkpoint load mismatch")
        write_json(output / "smoke_checkpoint_roundtrip.json", dict(passed=True, optimizer_steps=int(saved["global_step"])))
    write_json(args.output / "status.json", dict(phase="complete", pid=os.getpid()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "STGG-AL/data/data_kekule")
    parser.add_argument("--raw-csv", type=Path, default=ROOT.parent / "STGG-AL/data/dataset_train.csv")
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--target", type=float, default=4.0)
    parser.add_argument("--generate-n", type=int, default=1000)
    parser.add_argument("--panel-after-steps", type=int, default=10000)
    parser.add_argument("--max-epochs", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=215)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--memory-fraction", type=float, default=.20)
    parser.add_argument("--seed", type=int, default=17042)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.source, args.raw_csv = args.source.resolve(), args.raw_csv.resolve()
    if args.prepared_data:
        args.prepared_data = args.prepared_data.resolve()
    if not 0 < args.memory_fraction <= .20 or args.workers > 12:
        parser.error("concurrent experiment limited to 20% GPU allocation and 12 CPU workers")
    if not args.worker:
        args.output.mkdir(parents=True, exist_ok=False)
        snapshot = args.output / "model_source_snapshot"
        snapshot.mkdir()
        for name in ("molguidance", "configs", "tools", "tests", "docs"):
            shutil.copytree(PACKAGE / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.orig"))
        scripts = args.output / "experiment_code"
        scripts.mkdir()
        for name in ("workflow.py", "prepare.py", "sample.py"):
            shutil.copy2(Path(__file__).with_name(name), scripts / name)
        env = dict(os.environ, STGG_WORKSPACE_ROOT=str(ROOT), STGG_MODEL_PACKAGE=str(snapshot),
                   PYTHONPATH=f"{snapshot}:{ROOT}", WANDB_MODE="disabled", CUDA_VISIBLE_DEVICES="0",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", str(scripts / "workflow.py"), *sys.argv[1:], "--worker"]
        with (args.output / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, env=env, cwd=ROOT, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, close_fds=True)
        write_json(args.output / "launch.json", dict(pid=process.pid, detached=True, command=command,
                   source_sha256={str(p.relative_to(snapshot)): sha256(p) for p in snapshot.rglob("*.py")},
                   experiment_sha256={p.name: sha256(p) for p in scripts.glob("*.py")}))
        print(json.dumps(dict(pid=process.pid, output=str(args.output), detached=True)), flush=True)
        return
    try:
        write_json(args.output / "status.json", dict(phase="preparing", pid=os.getpid(), sid=os.getsid(0)))
        data_dir = args.prepared_data or args.output / "data_s1_joint_kekule"
        if args.prepared_data:
            conversion = json.loads((data_dir / "conversion_report.json").read_text())
        else:
            conversion = prepare(args.source, args.raw_csv, data_dir, args.workers, 16 if args.smoke else None)
        train(args, data_dir, conversion)
    except BaseException:
        write_json(args.output / "status.json", dict(phase="failed", pid=os.getpid(), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
