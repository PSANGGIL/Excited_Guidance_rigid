import unittest

import torch
import torch.nn as nn

from molguidance.models.model_guidance_scale import (
    ModelGuidance,
    _mg_feature_slices,
)


class TestMGFeatureSlices(unittest.TestCase):
    def test_node_feature_slices_follow_node_counts(self):
        slices = _mg_feature_slices('a', [3, 5, 2])
        self.assertEqual([(s.start, s.stop) for s in slices], [(0, 3), (3, 8), (8, 10)])

    def test_edge_feature_slices_follow_upper_triangle_counts(self):
        slices = _mg_feature_slices('e', [3, 5, 2])
        # C(3, 2) + C(5, 2) + C(2, 2) = 3 + 10 + 1 = 14.
        self.assertEqual([(s.start, s.stop) for s in slices], [(0, 3), (3, 13), (13, 14)])

    def test_edge_slices_do_not_cross_graph_boundaries(self):
        slices = _mg_feature_slices('e', [4, 6])
        packed = torch.zeros(21, dtype=torch.int64)
        packed[slices[0]] = 1
        packed[slices[1]] = 2

        self.assertTrue(torch.equal(packed[:6], torch.ones(6, dtype=torch.int64)))
        self.assertTrue(torch.equal(packed[6:], torch.full((15,), 2, dtype=torch.int64)))


class TestEMAUpdateHook(unittest.TestCase):
    def test_one_optimizer_step_changes_ema(self):
        # Construct only the state required by the hook; the full molecular model
        # is intentionally unnecessary for this lifecycle regression test.
        model = ModelGuidance.__new__(ModelGuidance)
        nn.Module.__init__(model)
        model.vector_field = nn.Linear(1, 1, bias=False)
        model.ema_vector_field = nn.Linear(1, 1, bias=False)

        with torch.no_grad():
            model.vector_field.weight.fill_(1.0)
            model.ema_vector_field.weight.fill_(1.0)

        optimizer = torch.optim.SGD(model.vector_field.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss = model.vector_field(torch.ones(1, 1)).sum()
        loss.backward()
        ema_before = model.ema_vector_field.weight.detach().clone()
        model.optimizer_step(0, 0, optimizer)
        online_after_step = model.vector_field.weight.detach().clone()
        ema_after = model.ema_vector_field.weight.detach().clone()

        expected = ema_before * 0.9999 + online_after_step * 0.0001
        self.assertFalse(torch.equal(ema_before, ema_after))
        self.assertTrue(torch.allclose(ema_after, expected, atol=1e-7, rtol=0.0))


if __name__ == '__main__':
    unittest.main()
