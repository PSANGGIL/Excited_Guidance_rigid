import itertools
import unittest

import dgl
import torch
from torch.nn.functional import one_hot

from molguidance.models.chemistry_distribution import bounded_sum_distribution, compatibility_probabilities
from molguidance.models.chemistry_feedback import JointChemicalFeedback
from molguidance.models.correction import correction_options, corrupt_visible_state
from molguidance.data_processing.utils import build_edge_idxs, get_batch_idxs, get_upper_edge_mask
from test_joint_chemistry_auxiliary import _complete_benzene, _chemistry_loss, _oracle_logits
from test_independent_time_bond_first import _field, _graph


class Stage7ChemistryTests(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration_including_overflow(self):
        p = torch.softmax(torch.randn(2, 3, 4, dtype=torch.double), -1).requires_grad_()
        actual = bounded_sum_distribution(p, 4)
        expected = torch.zeros_like(actual)
        for values in itertools.product(range(4), repeat=3):
            mass = p[:, 0, values[0]] * p[:, 1, values[1]] * p[:, 2, values[2]]
            expected[:, min(sum(values), 5)] += mass
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(-1), torch.ones(2, dtype=torch.double))
        self.assertTrue(torch.autograd.gradcheck(lambda x: bounded_sum_distribution(x, 4), (p,)))
        empty = bounded_sum_distribution(p[:, :0], 0)
        torch.testing.assert_close(empty, torch.tensor([[1., 0.], [1., 0.]], dtype=p.dtype))

    def test_benzene_marginal_counterexample_is_penalized(self):
        g, classes = _complete_benzene()
        node, oracle = _oracle_logits(classes)
        edges = oracle.clone()
        edges[classes != 0] = torch.tensor([-40., 0., 0., -40.])
        edges.requires_grad_()
        module = _chemistry_loss()
        upper = get_upper_edge_mask(g)
        legacy = module(g, node, edges, upper)
        self.assertLess(float(legacy["chem_valence"]), 1e-6)
        self.assertLess(float(legacy["chem_aromatic"]), 1e-6)
        new = module.distribution_losses(g, node, edges, upper, torch.ones(1))
        self.assertAlmostEqual(float(new["chem_distribution_valence"]), .69314718, places=5)
        sum(new.values()).backward()
        self.assertTrue(torch.isfinite(edges.grad).all())
        perfect = module.distribution_losses(g, node, oracle, upper, torch.ones(1))
        self.assertLess(float(sum(perfect.values())), 1e-6)

    def test_charge_probability_and_zero_padding_are_not_mean_charge(self):
        graphs = []
        for n in (2, 3, 1):
            edges = build_edge_idxs(n)
            graphs.append(dgl.graph((edges[0], edges[1]), num_nodes=n))
        g = dgl.batch(graphs)
        upper = get_upper_edge_mask(g)
        # +/- states and neutral padding/fake state; mean Q=0 in each graph.
        node = torch.tensor([[.5, .5, 0.]] * 5 + [[0., 0., 1.]], requires_grad=True)
        edge = one_hot(torch.zeros(int(upper.sum()), dtype=torch.long), 4).float()
        _, charge = compatibility_probabilities(
            g, node, edge, upper, torch.ones(3, 1, dtype=torch.bool),
            torch.tensor([-1., 1., 0.]), torch.zeros(3),
        )
        torch.testing.assert_close(charge, torch.tensor([.5, 0., 1.]))
        charge.sum().backward()
        self.assertTrue(torch.isfinite(node.grad).all())
        # A definite zwitterion has net zero probability 1, not a penalty.
        node2 = node.detach().clone()
        node2[:2] = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
        _, charge2 = compatibility_probabilities(
            g, node2, edge, upper, torch.ones(3, 1, dtype=torch.bool),
            torch.tensor([-1., 1., 0.]), torch.zeros(3),
        )
        self.assertEqual(float(charge2[0]), 1.)

    def test_feedback_crosses_node_edge_boundary_without_target_access(self):
        g, classes = _complete_benzene()
        upper = get_upper_edge_mask(g)
        metadata = [
            dict(hydrogens=1, formal_charge=0, train_heavy_valence_counts={"3": 1}),
            dict(hydrogens=0, formal_charge=1, train_heavy_valence_counts={"4": 1}),
        ]
        layer = JointChemicalFeedback(metadata, 8, 8)
        previous = {"s": one_hot(torch.zeros(6, dtype=torch.long), 2).float(),
                    "e": one_hot(classes, 4).float()}
        node1, edge1 = layer.features(g, upper, previous)
        previous["e"] = one_hot(torch.zeros_like(classes), 4).float()
        node2, _ = layer.features(g, upper, previous)
        self.assertFalse(torch.equal(node1, node2))
        previous["e"] = one_hot(classes, 4).float()
        previous["s"] = previous["s"].flip(-1)
        _, edge2 = layer.features(g, upper, previous)
        self.assertFalse(torch.equal(edge1, edge2))
        # Zero initialization and no-history must both preserve the old field.
        nodes, edges = torch.randn(6, 8), torch.randn(g.num_edges(), 8)
        a, b = layer(g, nodes, edges, upper, previous)
        torch.testing.assert_close(a, nodes)
        torch.testing.assert_close(b, edges)
        self.assertIs(layer(g, nodes, edges, upper, None)[0], nodes)
        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)
        (a.square().mean() + b.square().mean()).backward()
        optimizer.step()
        self.assertGreater(float(layer.node_residual[-1].weight.abs().sum()), 0.)
        self.assertGreater(float(layer.edge_residual[-1].weight.abs().sum()), 0.)
        a2, b2 = layer(g, nodes, edges, upper, previous)
        self.assertFalse(torch.equal(a2, nodes))
        self.assertFalse(torch.equal(b2, edges))

    def test_multiple_allowed_valences_are_not_collapsed_to_one_mean(self):
        # P-like {3, 5} and S-like {2, 4, 6} heavy valences are all retained.
        # The peripheral state allows a single heavy bond.
        allowed = torch.zeros(3, 7, dtype=torch.bool)
        allowed[0, [3, 5]] = True
        allowed[1, [2, 4, 6]] = True
        allowed[2, 1] = True
        for state, degree, expected in (
            (0, 3, 1.), (0, 5, 1.), (0, 4, 0.),
            (1, 2, 1.), (1, 4, 1.), (1, 6, 1.), (1, 3, 0.),
        ):
            indices = build_edge_idxs(degree + 1)
            g = dgl.graph((indices[0], indices[1]), num_nodes=degree + 1)
            upper = get_upper_edge_mask(g)
            src, dst = g.edges()
            edge = one_hot(((src[upper] == 0) | (dst[upper] == 0)).long(), 4).float()
            node = one_hot(torch.tensor([state] + [2] * degree), 3).float()
            local, charge = compatibility_probabilities(
                g, node, edge, upper, allowed, torch.zeros(3), torch.zeros(1),
            )
            self.assertEqual(float(local[0]), expected)
            torch.testing.assert_close(local[1:], torch.ones(degree))
            self.assertEqual(float(charge[0]), 1.)

    def test_correction_preserves_targets_labels_symmetry_and_center(self):
        field = _field(independent=True)
        g = _graph(4)
        g.prop = torch.tensor([[.123]])
        node_batch, edge_batch = get_batch_idxs(g)
        upper = get_upper_edge_mask(g)
        time = torch.tensor([.85])
        field.sample_conditional_path(g, time, node_batch, edge_batch, upper, graph_time=time)
        original = {key: g.ndata[key].clone() for key in ("x_1_true", "s_1_true")}
        original["e_1_true"] = g.edata["e_1_true"].clone()
        center = g.ndata["x_t"].mean(0)
        opts = correction_options(dict(node_rate=1., edge_rate=1.))
        active = corrupt_visible_state(g, field, opts, torch.Generator().manual_seed(6))
        torch.testing.assert_close(g.ndata["x_t"].mean(0), center, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(g.edata["e_t"][upper], g.edata["e_t"][~upper])
        for key in ("x_1_true", "s_1_true"):
            self.assertTrue(torch.equal(g.ndata[key], original[key]))
        self.assertTrue(torch.equal(g.edata["e_1_true"], original["e_1_true"]))
        self.assertAlmostEqual(float(g.prop), .123, places=6)
        self.assertTrue(active["e"].any())


if __name__ == "__main__":
    unittest.main()
