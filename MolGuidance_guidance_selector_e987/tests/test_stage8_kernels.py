import unittest
import copy
from pathlib import Path

import torch

from molguidance.models.mask_ctmc import frozen_mask_transition
from molguidance.models.classifier_free_guidance import CFGVectorField
from molguidance.data_processing.kekule_augmentation import kekule_variants, _identity
from test_stage8_joint_decode import encode


class Stage8KernelTests(unittest.TestCase):
    def test_frozen_ctmc_matches_matrix_exponential_and_conserves_support(self):
        for eta in (0., .1, 4.):
            for dt in (0., 1e-8, .1, 20.):
                rates = torch.tensor([[.1, .7, 0.]] * 4, dtype=torch.float64)
                xt = torch.arange(4)
                actual = frozen_mask_transition(rates, eta, xt, dt)
                q = torch.zeros(4, 4, dtype=torch.float64)
                q[3, :3] = rates[0]
                q[:3, 3] = eta
                q -= torch.diag(q.sum(-1))
                expected = torch.matrix_exp(q * dt)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        zero = frozen_mask_transition(torch.zeros(3, 2), 0., torch.arange(3), 1.)
        torch.testing.assert_close(zero, torch.eye(3))
        with self.assertRaises(ValueError):
            frozen_mask_transition(torch.tensor([[-1., 1.]]), 0, torch.tensor([2]), .1)

    def test_rate_cfg_final_step_samples_requested_distribution(self):
        n = 12000
        pu = torch.tensor([[.85, .15, 0.]]).expand(n, -1)
        pc = torch.tensor([[.25, .75, 0.]]).expand(n, -1)
        torch.manual_seed(17)
        for location in ("probabilities", "rate_matrix"):
            for weight, expected in ((0., .15), (1., .75)):
                out, endpoint = CFGVectorField.campbell_step_with_rate_matrix_cfg(
                    pu, pc, torch.full((n,), 3), 2., .999, 1., .1, weight, 3, 4,
                    pu.clamp_min(1e-9).log(), pc.clamp_min(1e-9).log(),
                    last_step=True, where_to_apply_guide=location,
                    class_support_mask=torch.tensor([True, True, False]))
                self.assertAlmostEqual(float(out[:, 1].mean()), expected, delta=.02)
                self.assertEqual(float(out[:, 2:].sum()), 0.)
                self.assertEqual(float(endpoint[:, 2:].sum()), 0.)

    def test_kekule_pool_identity_valence_and_no_inplace_mutation(self):
        for smiles in ("c1ccccc1", "c1ccc2ccccc2c1", "c1cc[nH]c1",
                       "c1ccoc1", "c1cc[nH+]cc1", "[O-][N+](=O)c1ccccc1"):
            p, q, _, metadata, aromatic = encode(smiles)
            pairs = torch.triu_indices(len(p), len(p), 1).T
            orders = torch.tensor(q.argmax(-1))
            flags = torch.zeros(len(q), dtype=torch.bool)
            flags[aromatic] = True
            occupied = orders > 0
            pairs, orders, flags = pairs[occupied], orders[occupied], flags[occupied]
            before = orders.clone()
            states = torch.tensor(p.argmax(-1))
            pool, report = kekule_variants(states, pairs, orders, flags, metadata, attempts=32)
            original = _identity(states, pairs, orders, metadata)
            for variant in pool:
                self.assertEqual(_identity(states, pairs, variant, metadata), original)
                original_valence = torch.zeros(len(p), dtype=torch.long)
                variant_valence = original_valence.clone()
                for column in (0, 1):
                    original_valence.index_add_(0, pairs[:, column], orders)
                    variant_valence.index_add_(0, pairs[:, column], variant)
                torch.testing.assert_close(original_valence, variant_valence)
            torch.testing.assert_close(before, orders)
            if smiles == "c1ccccc1":
                self.assertEqual(len(pool), 2)
            self.assertEqual(report["identity_rejections"], 0)

    def test_augmentation_preserves_paired_prior_rng_and_validation(self):
        from molguidance.model_utils.load import read_config_file, data_module_from_config
        project = Path(__file__).resolve().parents[1]
        config = read_config_file(project / "configs/local/stage8/00_control.yaml")
        config["dataset"]["processed_data_dir"] = str(
            project.parent / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz")
        plain = data_module_from_config(copy.deepcopy(config))
        plain.setup("fit")
        config["dataset"]["kekule_augmentation_variants"] = 2
        augmented = data_module_from_config(config)
        augmented.setup("fit")
        self.assertIsNone(augmented.val_dataset.kekule_pools)
        for index in (0, 1, 15):
            torch.manual_seed(812)
            a = plain.train_dataset[index]
            torch.manual_seed(812)
            b = augmented.train_dataset[index]
            for key in ("x_0", "x_1_true", "s_1_true"):
                torch.testing.assert_close(a.ndata[key], b.ndata[key])
            torch.testing.assert_close(a.edata["e_0"], b.edata["e_0"])
            torch.testing.assert_close(a.prop, b.prop)
        torch.testing.assert_close(plain.train_dataset.bond_types, augmented.train_dataset.bond_types)


if __name__ == "__main__":
    unittest.main()
