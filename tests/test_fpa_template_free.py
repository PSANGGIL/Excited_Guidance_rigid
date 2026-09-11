import math
import unittest
import torch

from polyring_diffusion.fpa_model_template_free import TemplateFreeFPAEntityDenoiser
from polyring_diffusion.fpa_projection_template_free import project_template_free_fpa
from sample_fpa_template_free import cfg_predict


class TemplateFreeFPATests(unittest.TestCase):
    def model(self):
        return TemplateFreeFPAEntityDenoiser(3, [.5, .6, .7], hidden_dim=24,
            num_layers=2, time_dim=8, condition_dim=4, min_ring_size=3,
            max_ring_size=10, max_atoms=12, max_cells=4).eval()

    def test_model_has_no_template_identity_and_cfg_null_is_exact(self):
        model = self.model(); b, na, nc = 2, 6, 2
        am = torch.ones(b, na, dtype=torch.bool); cm = torch.ones(b, nc, dtype=torch.bool)
        args = (torch.zeros(b, na, dtype=torch.long), torch.zeros(b, na, dtype=torch.long),
            torch.ones(b, na, dtype=torch.long), torch.zeros(b, na, dtype=torch.long), torch.randn(b, na, 3),
            torch.zeros(b, na, na, dtype=torch.long), am,
            torch.full((b, nc), 3), torch.ones(b, nc, dtype=torch.long),
            torch.randn(b, nc, 3), torch.eye(3).reshape(1, 1, 3, 3).repeat(b, nc, 1, 1),
            torch.zeros(b, nc), torch.zeros(b, nc, nc, dtype=torch.long),
            torch.ones(b, nc, na, dtype=torch.long), cm, torch.zeros(b, dtype=torch.long))
        condition = torch.randn(b, 4)
        guided = cfg_predict(model, args, condition, 0.)
        null = model(*args, condition, torch.ones(b, dtype=torch.bool))
        self.assertTrue(all(torch.equal(guided[key], null[key]) for key in guided))
        self.assertFalse(hasattr(model, "cell_type"))

    def test_single_heteroaromatic_ring_projection(self):
        n = 6; angle = torch.arange(n)*2*math.pi/n
        sample = {
            "atom_types": torch.tensor([[0, 0, 1, 0, 0, 0]]),
            "charges": torch.full((1, n), 2), "hydrogens": torch.ones(1, n, dtype=torch.long), "radicals": torch.zeros(1, n, dtype=torch.long),
            "positions": torch.stack((angle.cos(), angle.sin(), torch.ones(n)*.2), -1)[None],
            "bonds": torch.zeros(1, n, n, dtype=torch.long), "cell_sizes": torch.tensor([[6]]),
            "cell_aromatic": torch.ones(1, 1, dtype=torch.long), "cell_centers": torch.zeros(1, 1, 3),
            "cell_frames": torch.eye(3).reshape(1, 1, 3, 3), "cell_radii": torch.ones(1, 1),
            "cell_relations": torch.zeros(1, 1, 1, dtype=torch.long),
            "membership": torch.zeros(1, 1, n, dtype=torch.long),
            "atom_mask": torch.ones(1, n, dtype=torch.bool), "cell_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        member = torch.zeros(1, 1, n, 2); member[..., 1] = 9
        bond = torch.zeros(1, n, n, 5); bond[..., 0] = 9
        out = project_template_free_fpa(sample, {"membership_logits": member, "bond_logits": bond}, ["C", "N", "O"])
        self.assertEqual(int(out["membership"].sum()), 6)
        self.assertEqual(int((out["bonds"] == 4).sum()), 12)
        self.assertEqual(int(out["atom_types"][0, 2]), 1)
        self.assertTrue(bool(out["projection_valid"][0]))

    def test_fused_overlap_is_exact_and_edges_not_double_counted(self):
        n, c = 10, 2
        sample = {"atom_types": torch.zeros(1, n, dtype=torch.long), "charges": torch.full((1, n), 2),
            "hydrogens": torch.zeros(1, n, dtype=torch.long), "radicals": torch.zeros(1, n, dtype=torch.long), "positions": torch.randn(1, n, 3),
            "bonds": torch.zeros(1, n, n, dtype=torch.long), "cell_sizes": torch.full((1, c), 6),
            "cell_aromatic": torch.ones(1, c, dtype=torch.long), "cell_centers": torch.zeros(1, c, 3),
            "cell_frames": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, c, 1, 1),
            "cell_radii": torch.ones(1, c), "cell_relations": torch.tensor([[[0, 3], [3, 0]]]),
            "membership": torch.zeros(1, c, n, dtype=torch.long), "atom_mask": torch.ones(1, n, dtype=torch.bool),
            "cell_mask": torch.ones(1, c, dtype=torch.bool)}
        member = torch.zeros(1, c, n, 2)
        member[0, 0, :, 1] = torch.tensor([10, 9, 8, 7, 6, 5, 0, 0, 0, 0.])
        member[0, 1, :, 1] = torch.tensor([10, 9, 8, 0, 0, 0, 7, 6, 5, 4.])
        bond = torch.zeros(1, n, n, 5); bond[..., 0] = 9
        out = project_template_free_fpa(sample, {"membership_logits": member, "bond_logits": bond}, ["C"])
        shared = out["membership"][0, 0].bool() & out["membership"][0, 1].bool()
        self.assertEqual(int(shared.sum()), 2)
        self.assertEqual(out["membership"][0].sum(1).tolist(), [6, 6])
        ids = torch.where(shared)[0]
        self.assertEqual(int(out["bonds"][0, ids[0], ids[1]]), 4)


if __name__ == "__main__": unittest.main()
