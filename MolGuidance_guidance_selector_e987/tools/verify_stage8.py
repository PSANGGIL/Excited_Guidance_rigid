#!/usr/bin/env python3
"""Detached bounded integration verification, never a production training run."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import torch
import pytorch_lightning as pl
from rdkit import Chem
from torch.utils.data import DataLoader

from molguidance.model_utils.load import read_config_file, data_module_from_config
from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
from molguidance.model_utils.provenance import data_manifest, verify_manifest, state_dict_sha256, sha256_file
from molguidance.data_processing.dataset import collate
from molguidance.data_processing.utils import get_upper_edge_mask
from molguidance.models.correction import correction_losses
from molguidance.models.model_guidance_scale import ModelGuidance
from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.analysis.joint_decode import joint_decode
from molguidance.analysis.generation_callback import GenerationValidation

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def full_config(path):
    config = read_config_file(path)
    config["dataset"]["processed_data_dir"] = str(DATA)
    config["training"].update(batch_size=4, num_workers=0)
    return config


def verify(output, device):
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.7)
    report = dict(purpose="bounded integration, not converged performance",
                  data_before=data_manifest(DATA), configs=[], codec={},
                  source_sha256={str(p.relative_to(PROJECT)): sha256_file(p)
                                 for p in (PROJECT / "molguidance").rglob("*.py")})
    paths = sorted((PROJECT / "configs/local/stage8").glob("*.yaml"))
    report["expanded_configs"] = {p.name: read_config_file(p) for p in paths}
    reference_config = full_config(paths[0])
    reference = model_from_config_with_guidance(reference_config)
    dm = data_module_from_config(reference_config)
    dm.setup("fit")
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset), ("test", dm.test_dataset)):
        original = torch.load(DATA / f"{split}_data_processed.pt", weights_only=False, map_location="cpu")
        passed, joint_passed, unsupported = 0, 0, []
        for index, expected in enumerate(original["smiles"]):
            graph = dataset[index]
            for key in ("x", "s"):
                graph.ndata[f"{key}_1"] = graph.ndata[f"{key}_1_true"]
            graph.edata["e_1"] = graph.edata["e_1_true"]
            graph.edata["ue_mask"] = get_upper_edge_mask(graph)
            sampled = SampledMolecule(graph, reference.atom_type_map, ctmc_mol=False,
                                      atom_state_vocab=reference.atom_state_vocab)
            Chem.SanitizeMol(sampled.rdkit_mol)
            actual = Chem.MolToSmiles(sampled.rdkit_mol, isomericSmiles=False)
            expected = Chem.MolToSmiles(Chem.MolFromSmiles(expected), isomericSmiles=False)
            assert actual == expected, (split, index, actual, expected)
            passed += 1
            src, dst = graph.edges()
            upper = torch.where(src < dst)[0]
            upper = upper[torch.argsort(src[upper] * graph.num_nodes() + dst[upper])]
            states = graph.ndata["s_1"].argmax(-1).tolist()
            # The inference solver must NOT learn allowed valences from held-out
            # targets. Record out-of-train support separately, never alter DB.
            valence = torch.zeros(graph.num_nodes(), dtype=torch.long)
            orders = graph.edata["e_1"][upper].argmax(-1)
            valence.index_add_(0, src[upper], orders)
            valence.index_add_(0, dst[upper], orders)
            supported = all(reference.atom_state_vocab[s].get("is_fake") or
                            reference.atom_state_vocab[s].get("train_heavy_valence_counts", {}).get(str(int(v)), 0) > 0
                            for s, v in zip(states, valence))
            if not supported:
                unsupported.append(dict(index=index, smiles=expected))
                continue
            charge = sum(a.GetFormalCharge() for a in sampled.rdkit_mol.GetAtoms())
            result = joint_decode(graph.ndata["s_1"], graph.edata["e_1"][upper],
                                  graph.ndata["x_1"], reference.atom_state_vocab,
                                  net_charge=charge)
            assert result.molecule is not None, (split, index, result)
            assert Chem.MolToSmiles(result.molecule, isomericSmiles=False) == expected
            joint_passed += 1
        report["codec"][split] = dict(total=len(dataset), identity=passed,
                                     train_supported_joint_identity=joint_passed,
                                     out_of_train_valence_support=unsupported)
        write(output / "verification.json", report)
        print("CODEC", split, passed, joint_passed, len(unsupported), flush=True)
    del reference

    for path in paths:
        write(output / "status.json", dict(phase="integration", config=path.name))
        config = full_config(path)
        # Exercise the enabled auxiliary/MG branches immediately in a 3-step
        # test; do NOT misreport these overrides as a performance comparison.
        if config["mol_fm"]["correction"]["weight"] > 0:
            config["mol_fm"]["correction"].update(start_step=0, warmup_steps=0, probability=1.)
        if config["guidance"]["guidance_type"] == "model_guidance":
            config["guidance"]["mg_start_step"] = 0
        stage = output / path.stem
        stage.mkdir()
        pl.seed_everything(17041, workers=True)
        model = model_from_config_with_guidance(config)
        data_module = data_module_from_config(config)
        data_module.setup("fit")
        augmentation = data_module.train_dataset.kekule_augmentation_report
        if augmentation:
            assert data_module.val_dataset.kekule_pools is None
            assert data_module.test_dataset.kekule_pools is None
            # Every member was checked against the original full RDKit
            # identity during construction; report coverage and rejections.
            assert len(augmentation) == len(data_module.train_dataset)
        for loader in (data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()):
            sampler = loader.batch_sampler
            batches = list(sampler)
            assert len(batches) == len(sampler)
            assert sorted(i for batch in batches for i in batch) == list(range(len(loader.dataset)))
            assert all(int(sampler.costs[batch].sum()) <= sampler.max_edges for batch in batches)
        train_loader = data_module.train_dataloader()
        trainer = pl.Trainer(
            accelerator="gpu" if device == "cuda" else "cpu", devices=1,
            precision="32-true", max_steps=3, max_epochs=1, limit_train_batches=3,
            limit_val_batches=1, num_sanity_val_steps=0, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            gradient_clip_val=1., gradient_clip_algorithm="value", default_root_dir=stage,
            use_distributed_sampler=False,
        )
        initial = state_dict_sha256(model.state_dict())
        teacher_before = state_dict_sha256(model.ema_vector_field.state_dict()) if isinstance(model, ModelGuidance) else None
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=data_module.val_dataloader())
        assert trainer.global_step == 3 and state_dict_sha256(model.state_dict()) != initial
        assert all(torch.isfinite(p).all() for p in model.parameters())
        if teacher_before:
            assert state_dict_sha256(model.ema_vector_field.state_dict()) != teacher_before
            assert not model.ema_vector_field.training
        ckpt = stage / "smoke_final.ckpt"
        trainer.save_checkpoint(ckpt)
        restored = type(model).load_from_checkpoint(ckpt, map_location="cpu", weights_only=False).to(device).eval()
        assert state_dict_sha256(restored.state_dict()) == state_dict_sha256(model.state_dict())
        mg_sampling_contract = None
        if isinstance(restored, ModelGuidance):
            common = dict(n_atoms=torch.tensor([6], device=device), n_timesteps=4,
                          device=device, property_name="est", training_mode=False,
                          dataset_name=restored.dataset_name)
            outputs = []
            for target in (-2., 10.):
                torch.manual_seed(19)
                outputs.append(restored.sample(
                    **common, properties_for_sampling=target, guidance_scale=0.)[0].slot_graph)
            for key in ("x_1", "s_1"):
                torch.testing.assert_close(outputs[0].ndata[key], outputs[1].ndata[key], rtol=0, atol=0)
            torch.testing.assert_close(outputs[0].edata["e_1"], outputs[1].edata["e_1"], rtol=0, atol=0)
            guided = restored.sample(**common, properties_for_sampling=0., guidance_scale=2.)[0].slot_graph
            assert torch.isfinite(guided.ndata["x_1"]).all()
            mg_sampling_contract = dict(unconditional_target_independent=True, max_scale_finite=True)
        if path.stem == "08_longtrain_cfg":
            for location in ("probabilities", "rate_matrix"):
                sample = restored.sample(
                    n_atoms=torch.tensor([6], device=device), n_timesteps=4, device=device,
                    property_name="est", properties_for_sampling=0., training_mode=False,
                    dataset_name=restored.dataset_name, dfm_type="campbell_rate_matrix",
                    where_to_apply_guide=location,
                    guide_w={key: 1. for key in restored.canonical_feat_order})[0]
                assert torch.isfinite(sample.slot_graph.ndata["x_1"]).all()
        panels = []
        for name, target in (("validation", None), ("est0", 0.)):
            callback = GenerationValidation(stage / name, DATA, n_samples=2, batch_size=2,
                                            n_timesteps=12, target_est_eV=target,
                                            joint_decoding=dict(time_limit=2., max_rounds=4))
            result = callback.evaluate(restored, 0, 3)
            panels.append(dict(panel=name, raw=result["raw_valid_count"],
                               requested=result["requested"], joint=result["joint_decoding"]["raw_valid_count"]))
        report["configs"].append(dict(config=path.name, optimizer_steps=3,
                                      checkpoint_roundtrip=True, teacher_updated=bool(teacher_before),
                                      mg_sampling_contract=mg_sampling_contract,
                                      batch_budget=True, panels=panels,
                                      kekule_augmentation=dict(
                                          verified_records=len(augmentation),
                                          multiple_variants=sum(r["variants"] > 1 for r in augmentation),
                                          cached_variants=sum(r["variants"] for r in augmentation),
                                          identity_rejections=sum(r["identity_rejections"] for r in augmentation),
                                          unsupported_components_preserved=sum(r["unsupported_components_preserved"] for r in augmentation)),
                                      parameters=sum(p.numel() for p in model.parameters()),
                                      explicit_smoke_overrides=config))
        write(output / "verification.json", report)
        print("INTEGRATION PASS", path.name, panels, flush=True)
        del model, restored, trainer, train_loader
        if device == "cuda":
            torch.cuda.empty_cache()
    verify_manifest(report["data_before"], data_manifest(DATA))
    for relative, expected in report["source_sha256"].items():
        assert sha256_file(PROJECT / relative) == expected, ("source changed during verification", relative)
    for p in paths:
        assert read_config_file(p) == report["expanded_configs"][p.name]
    report.update(data_unchanged=True, complete=True)
    write(output / "verification.json", report)
    write(output / "status.json", dict(phase="complete", data_unchanged=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.worker:
        output.mkdir(parents=True, exist_ok=False)
        environment = os.environ.copy()
        environment.update(PYTHONPATH=f"{PROJECT.parent}:{PROJECT}", OMP_NUM_THREADS="2",
                           OPENBLAS_NUM_THREADS="1", WANDB_MODE="disabled",
                           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        with (output / "worker.log").open("x") as stream:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                 "--output", str(output), "--device", args.device],
                cwd=PROJECT.parent, env=environment, start_new_session=True,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        write(output / "launch.json", dict(pid=process.pid, detached=True))
        print(json.dumps(dict(pid=process.pid, output=str(output))), flush=True)
    else:
        try:
            verify(output, args.device)
        except Exception:
            write(output / "status.json", dict(phase="failed", traceback=traceback.format_exc()))
            raise


if __name__ == "__main__":
    main()
