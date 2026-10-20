import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from rdkit import Chem

from molguidance.analysis.strict_generation import strict_molecule_record, summarize_records
from molguidance.analysis.generation_callback import GenerationValidation, GeneratedQualityCheckpoint
from molguidance.model_utils.load import read_config_file, resolve_training_resume
from molguidance.model_utils.provenance import verify_manifest, state_dict_sha256
from molguidance.models.classifier_free_guidance import ClassifierFreeGuidance
from tools.attach_stage7_condition_results import attach


def _ethane():
    mol = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, (0., 0., 0.))
    conformer.SetAtomPosition(1, (1.52, 0., 0.))
    mol.AddConformer(conformer)
    return mol


class _SamplingProbe(ClassifierFreeGuidance):
    def __init__(self):
        nn.Module.__init__(self)
        self.last_atom_count_diagnostics = {}
        self.kwargs = None

    def resolve_atom_counts(self, n_molecules, device, **kwargs):
        return torch.tensor(kwargs["number_of_atoms"])

    def add_fake_atom_capacity(self, counts, **kwargs):
        return counts

    def sample(self, counts, **kwargs):
        self.kwargs = kwargs
        return counts


class _PanelProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.))
        self.vector_field = nn.Identity()
        self.vector_field.training_mode = True
        self.dataset_name = "qme14s"
        self.canonical_feat_order = ["x", "s", "e"]
        self.data_manifest = None
        self.device = torch.device("cpu")
        self.arguments = []

    def sample_random_sizes(self, n, **kwargs):
        self.arguments.append(kwargs)
        torch.randn(8)
        self.vector_field.training_mode = False
        return [SimpleNamespace(rdkit_mol=_ethane()) for _ in range(n)]


class Stage7EvaluationTests(unittest.TestCase):
    def test_nested_config_and_cycle_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "base.yaml").write_text("model:\n  a: 1\n  b: 2\n")
            (path / "child.yaml").write_text("_base_: base.yaml\nmodel:\n  a: 3\n")
            self.assertEqual(read_config_file(path / "child.yaml"), {"model": {"a": 3, "b": 2}})
            (path / "base.yaml").write_text("_base_: child.yaml\n")
            with self.assertRaisesRegex(ValueError, "cyclic"):
                read_config_file(path / "child.yaml")

    def test_hash_contract_relocation_and_mutation(self):
        expected = dict(version=1, directory="/old", sha256={"data": "abc"})
        verify_manifest(expected, dict(version=1, directory="/new", sha256={"data": "abc"}))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            verify_manifest(expected, dict(version=1, sha256={"data": "different"}))
        state = {"x": torch.tensor([1., 2.])}
        before = state_dict_sha256(state)
        state["x"][1] = 3.
        self.assertNotEqual(before, state_dict_sha256(state))

    def test_resume_resolves_nested_quality_checkpoint_and_rejects_missing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "config.yaml").write_text("model: {}\n")
            quality = run / "checkpoints" / "generation_quality"
            quality.mkdir(parents=True)
            last = quality.parent / "last.ckpt"
            chosen = quality / "epoch0049.ckpt"
            last.touch()
            chosen.touch()
            self.assertEqual(resolve_training_resume(run), (run, last))
            self.assertEqual(resolve_training_resume(chosen), (run, chosen))
            (run / "config.yaml").unlink()
            with self.assertRaisesRegex(ValueError, "config.yaml"):
                resolve_training_resume(chosen)

    def test_sampler_time_points_are_forwarded(self):
        model = _SamplingProbe()
        model.sample_random_sizes(2, device="cpu", n_timesteps=173,
                                  number_of_atoms=[6, 11])
        self.assertEqual(model.kwargs["n_timesteps"], 173)

    def test_strict_validity_does_not_keep_largest_fragment_or_ignore_charge(self):
        fragment = strict_molecule_record(Chem.MolFromSmiles("CC.C"))
        charged = strict_molecule_record(Chem.MolFromSmiles("[NH4+]"))
        radical = strict_molecule_record(Chem.MolFromSmiles("[CH3]"))
        self.assertTrue(fragment["raw_valid"])
        self.assertFalse(fragment["connected"])
        self.assertFalse(charged["eligible"])
        self.assertFalse(radical["eligible"])
        self.assertEqual(summarize_records([])["requested"], 0)
        self.assertEqual(summarize_records([fragment, charged, radical])["eligible_rate"], 0.)

    def test_geometry_is_checked_before_any_coordinate_repair(self):
        mol = _ethane()
        self.assertTrue(strict_molecule_record(mol)["geometry_eligible"])
        mol.GetConformer().SetAtomPosition(1, (10., 0., 0.))
        record = strict_molecule_record(mol)
        self.assertTrue(record["eligible"])
        self.assertFalse(record["geometry_eligible"])
        self.assertEqual(mol.GetConformer().GetAtomPosition(1).x, 10.)

    def test_one_heavy_atom_collapse_is_not_a_geometry_success(self):
        mol = Chem.MolFromSmiles("C")
        mol.AddConformer(Chem.Conformer(1))
        record = strict_molecule_record(mol)
        self.assertTrue(record["raw_valid"])
        self.assertEqual(record["heavy_atoms"], 1)
        self.assertFalse(record["geometry_evaluable"])
        self.assertFalse(record["geometry_eligible"])

    def test_generation_panel_rng_png_units_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for split in ("train", "val", "test"):
                torch.save(
                    dict(smiles=["O", "CO"], node_idx_array=torch.tensor([[0, 1], [1, 3]]),
                         properties=torch.tensor([.1, -.2])),
                    path / f"{split}_data_processed.pt",
                )
            model = _PanelProbe().train()
            callback = GenerationValidation(path / "out", path, n_samples=2, batch_size=1, n_timesteps=5)
            state = torch.random.get_rng_state()
            report = callback.evaluate(model, 0, 1)
            self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
            self.assertTrue(model.training)
            self.assertTrue(model.vector_field.training_mode)
            self.assertEqual(report["eligible_count"], 2)
            self.assertIn("not_measured", report["condition_fidelity_status"])
            self.assertEqual(model.arguments[0]["n_timesteps"], 5)
            targets = sorted(row["target_est_eV"] for row in report["records"])
            self.assertAlmostEqual(targets[0], -.2, places=6)
            self.assertAlmostEqual(targets[1], .1, places=6)
            self.assertEqual(len(list((path / "out").rglob("*.png"))), 1)
            callback.evaluate(model, 0, 1)
            self.assertEqual(len(list((path / "out").rglob("report.json"))), 2)

    def test_failed_or_stale_generation_cannot_select_a_quality_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            callback = GenerationValidation(Path(tmp) / "out", tmp, every_n_epochs=1)
            logged = []
            model = SimpleNamespace(_stage7_generation_epoch=7,
                                    log=lambda *args, **kwargs: logged.append(args))
            trainer = SimpleNamespace(sanity_checking=False, current_epoch=8, global_step=17)
            with patch.object(callback, "evaluate", side_effect=FloatingPointError("bad rollout")):
                with self.assertWarnsRegex(UserWarning, "Generation validation failed"):
                    callback.on_validation_epoch_end(trainer, model)
            self.assertIsNone(model._stage7_generation_epoch)
            self.assertIn(("gen_evaluation_failed", 1.), logged)
            self.assertEqual(len(list((Path(tmp) / "out").glob("evaluation_error_*.json"))), 1)
            checkpointer = GeneratedQualityCheckpoint()
            with patch("pytorch_lightning.callbacks.ModelCheckpoint.on_validation_end") as parent:
                checkpointer.on_validation_end(trainer, model)
                self.assertFalse(parent.called)
                model._stage7_generation_epoch = 7
                checkpointer.on_validation_end(trainer, model)
                self.assertFalse(parent.called)
                model._stage7_generation_epoch = 8
                checkpointer.on_validation_end(trainer, model)
                parent.assert_called_once_with(trainer, model)

    def test_external_condition_results_keep_denominators_and_require_provenance(self):
        report = dict(
            model_state_sha256="model", raw_sdf_sha256="sdf", requested=3,
            records=[
                dict(generation_index=0, eligible=True, aromatic=True,
                     geometry_screen=True, target_est_eV=0.),
                dict(generation_index=1, eligible=True, aromatic=False,
                     geometry_screen=True, target_est_eV=0.),
                dict(generation_index=2, eligible=False, aromatic=False,
                     geometry_screen=False, target_est_eV=0.),
            ],
        )
        rows = [dict(generation_index="0", status="success", s1_eV="1.428", t1_eV="1.528")]
        manifest = dict(model_state_sha256="model", source_sdf_sha256="sdf", method="test-only")
        result = attach(report, rows, manifest)
        self.assertEqual(result["calculation_success"], 1)
        self.assertEqual(result["condition_hit_yield"], 1 / 3)
        self.assertEqual(result["aromatic_condition_hit_yield"], 1 / 3)
        self.assertAlmostEqual(result["condition_mae_eV_calculated_only"], .1)
        self.assertNotIn("calculation_status", report["records"][0])
        for bad_rows in (
            rows + rows, [dict(rows[0], generation_index="4")],
            [dict(rows[0], s1_eV="nan")], [dict(rows[0], generation_index="2")],
        ):
            with self.assertRaises(ValueError):
                attach(report, bad_rows, manifest)
        with self.assertRaisesRegex(ValueError, "provenance"):
            attach(report, rows, dict(manifest, source_sdf_sha256="different"))


if __name__ == "__main__":
    unittest.main()
