from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from molguidance.models.mg_targets import soft_guided_target
from molguidance.model_utils.load import read_config_file, data_module_from_config
from molguidance.model_utils.load_with_guidance import model_from_config_with_guidance
from molguidance.data_processing.dataset import collate
from molguidance.models.correction import correction_losses
from molguidance.models.flowmol import FlowMol


PROJECT = Path(__file__).resolve().parents[1]


def tiny_mg():
    config = read_config_file(PROJECT / "configs/local/stage7/08_condition_scale_control.yaml")
    config["dataset"]["processed_data_dir"] = str(PROJECT.parent / "molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz")
    config["vector_field"].update(n_hidden_scalars=16, n_hidden_edge_feats=12, n_vec_channels=4,
                                  n_molecule_updates=2, n_message_gvps=1, n_update_gvps=1, rbf_dim=8,
                                  dropout=.1)
    config["model_setting"]["property_embedding_dim"] = 16
    config["guidance"].update(guidance_type="model_guidance", min_scale=0., max_scale=2.,
                              mg_high=.75, data_ratio=1., mg_start_step=0, use_scale_aware=True,
                              p_uncond=0.)
    config["training"].update(batch_size=2, num_workers=0, max_num_edges=50000)
    return model_from_config_with_guidance(config), config


class MGTargetTests(unittest.TestCase):
    def test_soft_target_boundary_support_and_detachment(self):
        clean = F.one_hot(torch.tensor([0, 2]), 3).float()
        cond, uncond = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        for scale in (0., .5, 1., 1.5, 2., 20.):
            q = soft_guided_target(clean, cond, uncond, scale, torch.tensor([True, False, True]))
            self.assertTrue(torch.isfinite(q).all())
            self.assertFalse(q.requires_grad)
            torch.testing.assert_close(q.sum(-1), torch.ones(2))
            self.assertTrue((q[:, 1] == 0).all())
            if scale == 1:
                torch.testing.assert_close(q, clean)
        self.assertTrue((soft_guided_target(clean, cond, uncond, 1.5) > 0).all())

    def test_true_soft_ce_gradient_not_argmax(self):
        logits = torch.tensor([[.1, .2, .3]], requires_grad=True)
        q = torch.tensor([[.2, .3, .5]])
        F.cross_entropy(logits, q).backward()
        torch.testing.assert_close(logits.grad, logits.detach().softmax(-1) - q)

    def test_correction_schedule_and_detached_loss_cap(self):
        model = SimpleNamespace(canonical_feat_order=["x", "s", "e"],
                                total_loss_weights=dict(x=1., s=1., e=1.),
                                chemistry_loss_weights={},
                                correction_config=dict(weight=1., start_step=10,
                                                       warmup_steps=1, max_loss_ratio=.1),
                                global_step=0)
        x, aux = torch.tensor(1., requires_grad=True), torch.tensor(100., requires_grad=True)
        losses = dict(x=x, s=x*0, e=x*0, correction_x=aux)
        self.assertEqual(float(FlowMol._total_from_feature_losses(model, losses)), 1.)
        model.global_step = 10
        total = FlowMol._total_from_feature_losses(model, losses)
        self.assertAlmostEqual(float(total), 1.1, places=6)
        total.backward()
        self.assertAlmostEqual(float(x.grad), 1.)
        self.assertAlmostEqual(float(aux.grad), .001, places=6)

    def test_real_mg_forward_preserves_clean_targets_and_supports_correction(self):
        model, config = tiny_mg()
        dm = data_module_from_config(config)
        dm.setup("fit")
        batch = collate([dm.train_dataset[0], dm.train_dataset[17]])
        before = {k: batch.ndata[k].clone() for k in ("x_1_true", "s_1_true")}
        edge = batch.edata["e_1_true"].clone()
        prop = batch.prop.clone()
        model.train()
        self.assertFalse(model.ema_vector_field.training)
        losses = model(batch, t=torch.tensor([.2, .8]), graph_time=torch.tensor([.3, .4]),
                       corruption_generator=torch.Generator().manual_seed(9))
        losses.update(correction_losses(model, batch, torch.Generator().manual_seed(8), force=True))
        total = model._total_from_feature_losses(losses)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        self.assertTrue(all(p.grad is None for p in model.ema_vector_field.parameters()))
        for key, original in before.items():
            torch.testing.assert_close(batch.ndata[key], original, rtol=0, atol=0)
        torch.testing.assert_close(batch.edata["e_1_true"], edge, rtol=0, atol=0)
        torch.testing.assert_close(batch.prop.reshape(-1), prop.reshape(-1))
        with self.assertRaisesRegex(ValueError, "trained range"):
            model.sample(torch.tensor([6]), device="cpu", guidance_scale=3.)
        # Batch of one with no MG rows used to call dgl.batch([]).
        model.data_ratio = 0.
        single = collate([dm.train_dataset[0]])
        self.assertTrue(all(torch.isfinite(v) for v in model(single).values()))


if __name__ == "__main__":
    unittest.main()
