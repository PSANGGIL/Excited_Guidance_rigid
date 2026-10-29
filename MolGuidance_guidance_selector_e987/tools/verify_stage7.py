"""CPU integration checks on the unchanged HP1KG DB; never launch long training.

Writes only a fresh verification directory. Small-network optimizer steps are
disposable tests, not production checkpoints or evidence of better generation.
"""
import argparse
import copy
import json
from pathlib import Path

import torch
import pytorch_lightning as pl
from rdkit import Chem

from molguidance.model_utils.load import read_config_file, data_module_from_config
from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
from molguidance.model_utils.provenance import data_manifest, state_dict_sha256, verify_manifest
from molguidance.data_processing.dataset import collate
from molguidance.data_processing.utils import get_upper_edge_mask
from molguidance.models.correction import correction_losses
from molguidance.analysis.generation_callback import GenerationValidation, GeneratedQualityCheckpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    data = project.parent / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz"
    before_manifest = data_manifest(data)
    report = {"data_before": before_manifest, "stages": [], "production_training_started": False}
    torch.set_num_threads(2)
    paths = sorted((project / "configs/local/stage7").glob("*.yaml"))
    final_model = final_config = final_batch = None
    reference_initial = None
    for path in paths:
        config = read_config_file(path)
        config["dataset"]["processed_data_dir"] = str(data)
        # Preserve feature flags/losses; reduce ONLY network width/depth for smoke.
        config["vector_field"].update(
            n_hidden_scalars=16, n_hidden_edge_feats=12, n_vec_channels=4,
            n_molecule_updates=2, n_message_gvps=1, n_update_gvps=1, rbf_dim=8,
        )
        config["model_setting"]["property_embedding_dim"] = 16
        config["training"].update(batch_size=2, num_workers=0)
        torch.manual_seed(17041)
        model = model_from_config_with_guidance(config)
        if path.name == "01_contracts.yaml":
            reference_initial = copy.deepcopy(model.state_dict())
        if path.name == "02_feedback.yaml":
            assert reference_initial is not None
            for key, value in reference_initial.items():
                assert torch.equal(value, model.state_dict()[key]), key
            report["feedback_preserves_common_initialization"] = True
        dm = data_module_from_config(config)
        dm.setup("fit")
        dataset = dm.train_dataset
        count = dataset.node_idx_array[:, 1] - dataset.node_idx_array[:, 0]
        ids = sorted(set([int(count.argmin()), int(count.argmax()), 0, 17]))
        batch = collate([dataset[index] for index in ids])
        model.train()
        initial = state_dict_sha256(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses = model(
            batch, t=torch.full((len(ids),), .85),
            graph_time=torch.full((len(ids),), .85),
            corruption_generator=torch.Generator().manual_seed(83),
        )
        target_before = batch.ndata["x_1_true"].clone()
        properties_before = batch.prop.clone()
        if model.correction_config["weight"] > 0:
            losses.update(correction_losses(
                model, batch, torch.Generator().manual_seed(120), force=True
            ))
        total = model._total_from_feature_losses(losses)
        assert torch.isfinite(total), (path, losses)
        total.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
        optimizer.step()
        assert initial != state_dict_sha256(model.state_dict())
        assert torch.equal(target_before, batch.ndata["x_1_true"])
        assert torch.equal(properties_before, batch.prop)
        model.eval()
        # Check diagonal and off-diagonal contracts, and clean graph / empty CE.
        validation = {}
        for tx, tg in ((.5, .5), (.3, .95), (.95, .3), (.5, 1.)):
            with torch.no_grad():
                values = model(
                    batch, t=torch.full((len(ids),), tx),
                    graph_time=torch.full((len(ids),), tg), condition_dropout=False,
                    corruption_generator=torch.Generator().manual_seed(123),
                )
            assert all(torch.isfinite(value) for value in values.values())
            if tg == 1.:
                assert float(values["s"]) == 0. and float(values["e"]) == 0.
            validation[f"{tx}/{tg}"] = {key: float(value) for key, value in values.items()}
        checkpoint = {"state_dict": copy.deepcopy(model.state_dict())}
        model.on_save_checkpoint(checkpoint)
        restored = model_from_config_with_guidance(config)
        restored.on_load_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint["state_dict"], strict=True)
        assert state_dict_sha256(restored.state_dict()) == state_dict_sha256(model.state_dict())
        # Wrong dataset and wrong parameters must both fail closed.
        broken = copy.deepcopy(checkpoint)
        broken["stage7_data_manifest"]["sha256"]["train_data_processed.pt"] = "wrong"
        try:
            restored.on_load_checkpoint(broken)
            raise AssertionError("bad data manifest accepted")
        except ValueError:
            pass
        broken = copy.deepcopy(checkpoint)
        broken["stage7_model_sha256"] = "wrong"
        try:
            restored.on_load_checkpoint(broken)
            raise AssertionError("bad checkpoint hash accepted")
        except ValueError:
            pass
        report["stages"].append(dict(
            config=path.name, train_indices=ids, original_atom_counts=count[ids].tolist(),
            loss=float(total.detach()), gradient_norm_before_clip=grad_norm,
            losses={key: float(value.detach()) for key, value in losses.items()},
            validation=validation, checkpoint_roundtrip=True,
        ))
        print(path.name, "PASS", float(total.detach()), "gradient", grad_norm, flush=True)
        final_model, final_config, final_batch = model, config, batch
    callback = GenerationValidation(
        args.output / "generation_smoke", data, n_samples=2, batch_size=1, n_timesteps=4,
    )
    rng_before = torch.random.get_rng_state().clone()
    generated = callback.evaluate(final_model, epoch=0, step=1)
    assert torch.equal(rng_before, torch.random.get_rng_state()), "generation changed training RNG"
    report["disposable_untrained_generation"] = generated
    # Exercise actual Lightning ordering: train -> validation -> generation
    # metrics -> quality checkpoint. This is a one-step disposable CPU smoke.
    tiny_eval = GenerationValidation(
        args.output / "lightning_generation_smoke", data, every_n_epochs=1,
        n_samples=2, batch_size=1, n_timesteps=4,
    )
    quality_checkpoint = GeneratedQualityCheckpoint(
        dirpath=args.output / "disposable_checkpoints",
        monitor="gen_geometry_eligible_rate", mode="max", save_top_k=1,
        every_n_epochs=1, save_on_train_epoch_end=False,
    )
    dm = data_module_from_config(final_config)
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, logger=False, max_steps=1, max_epochs=1,
        limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
        enable_progress_bar=False, enable_model_summary=False,
        default_root_dir=args.output / "lightning",
        callbacks=[tiny_eval, quality_checkpoint],
    )
    final_model.train()
    trainer.fit(final_model, datamodule=dm)
    assert trainer.global_step == 1 and quality_checkpoint.best_model_path
    from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
    loaded = ClassifierFreeGuidance.load_from_checkpoint(
        quality_checkpoint.best_model_path, map_location="cpu", weights_only=False,
    )
    assert state_dict_sha256(loaded.state_dict()) == state_dict_sha256(final_model.state_dict())
    report["lightning_one_step_and_quality_checkpoint"] = True
    # Exact production architecture, including dropout/DP/correction, with a
    # small real batch. This is not a batch-640 memory or speed benchmark.
    full_config = read_config_file(project / "configs/local/stage7/06_weight_decay.yaml")
    full_config["dataset"]["processed_data_dir"] = str(data)
    torch.manual_seed(17041)
    full_model = model_from_config_with_guidance(full_config).train()
    batch = final_batch
    full_losses = full_model(
        batch, t=torch.full((batch.batch_size,), .85),
        graph_time=torch.full((batch.batch_size,), .85),
        corruption_generator=torch.Generator().manual_seed(81),
    )
    full_losses.update(correction_losses(
        full_model, batch, torch.Generator().manual_seed(82), force=True,
    ))
    full_loss = full_model._total_from_feature_losses(full_losses)
    full_loss.backward()
    assert torch.isfinite(full_loss)
    assert all(torch.isfinite(p.grad).all() for p in full_model.parameters() if p.grad is not None)
    report["full_width_forward_backward"] = dict(
        parameters=sum(p.numel() for p in full_model.parameters()),
        loss=float(full_loss.detach()), graph_count=batch.batch_size,
        total_slots=batch.num_nodes(),
    )
    del full_model, full_losses, full_loss
    # Entire DB encoder/decoder unchanged: test the actual graph codec.
    from molguidance.analysis.molecule_builder import SampledMolecule
    dm = data_module_from_config(final_config)
    dm.setup("fit")
    identity = {}
    for split, dataset in (("train", dm.train_dataset), ("val", dm.val_dataset), ("test", dm.test_dataset)):
        original = torch.load(data / f"{split}_data_processed.pt", weights_only=False)
        passed = 0
        for index, expected_smiles in enumerate(original["smiles"]):
            graph = dataset[index]
            for key in ("x", "s"):
                graph.ndata[f"{key}_1"] = graph.ndata[f"{key}_1_true"]
            graph.edata["e_1"] = graph.edata["e_1_true"]
            graph.edata["ue_mask"] = get_upper_edge_mask(graph)
            mol = SampledMolecule(
                graph, final_model.atom_type_map, ctmc_mol=False,
                atom_state_vocab=final_model.atom_state_vocab,
            ).rdkit_mol
            Chem.SanitizeMol(mol)
            actual = Chem.MolToSmiles(mol, isomericSmiles=False)
            expected = Chem.MolToSmiles(Chem.MolFromSmiles(expected_smiles), isomericSmiles=False)
            assert actual == expected, (split, index, actual, expected)
            passed += 1
        identity[split] = passed
    report["codec_identity"] = identity
    verify_manifest(before_manifest, data_manifest(data))
    report["data_unchanged"] = True
    with (args.output / "verification.json").open("w") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print("ALL PASS", identity, flush=True)


if __name__ == "__main__":
    main()
