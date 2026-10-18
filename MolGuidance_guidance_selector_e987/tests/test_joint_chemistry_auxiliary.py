import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dgl
import torch
from torch.nn.functional import one_hot

from molguidance.data_processing.dataset import MoleculeDataset
from molguidance.models.chemistry_auxiliary import JointChemistryAuxiliaryLoss
from molguidance.models.flowmol import FlowMol
from molguidance.data_processing.utils import get_upper_edge_mask
from molguidance_hp1kg_baseline.convert_hp1kg_joint_kekule import (
    uniform_spanning_tree_edge_weights,
)


def _complete_benzene(masked=True):
    n_atoms = 6
    upper_edges = torch.triu_indices(n_atoms, n_atoms, offset=1)
    edges = torch.cat((upper_edges, upper_edges.flip(0)), dim=1)
    graph = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
    ring_bonds = {
        (0, 1): 1,
        (1, 2): 2,
        (2, 3): 1,
        (3, 4): 2,
        (4, 5): 1,
        (0, 5): 2,
    }
    upper_classes = torch.tensor(
        [ring_bonds.get((int(i), int(j)), 0) for i, j in upper_edges.T]
    )
    all_classes = torch.cat((upper_classes, upper_classes))
    graph.edata["e_1_true"] = one_hot(all_classes, num_classes=4).float()
    graph.ndata["s_1_true"] = one_hot(
        torch.zeros(n_atoms, dtype=torch.long), num_classes=2
    ).float()

    ring_mask = upper_classes != 0
    upper_connectivity = torch.where(
        ring_mask,
        torch.full_like(upper_classes, 5.0 / 6.0, dtype=torch.float32),
        torch.zeros_like(upper_classes, dtype=torch.float32),
    )
    graph.edata["e_connectivity_weight"] = torch.cat(
        (upper_connectivity, upper_connectivity)
    )
    graph.edata["e_aromatic_true"] = torch.cat((ring_mask, ring_mask))

    if masked:
        node_t = torch.full((n_atoms,), 2, dtype=torch.long)
        edge_t = torch.full((all_classes.numel(),), 4, dtype=torch.long)
    else:
        node_t = torch.zeros(n_atoms, dtype=torch.long)
        edge_t = all_classes
    graph.ndata["s_t"] = one_hot(node_t, num_classes=3).float()
    graph.edata["e_t"] = one_hot(edge_t, num_classes=5).float()
    return graph, upper_classes


def _chemistry_loss():
    # C(0,H1) accepts heavy valence 3; a second supported charged state
    # provides a non-trivial node probability/charge gradient in the tests.
    valence_probabilities = torch.zeros(2, 5)
    valence_probabilities[0, 3] = 1.0
    valence_probabilities[1, 4] = 1.0
    return JointChemistryAuxiliaryLoss(
        state_charges=torch.tensor([0.0, 1.0]),
        state_valence_probabilities=valence_probabilities,
        state_support=torch.tensor([True, True]),
    )


def _oracle_logits(upper_classes):
    node_logits = torch.tensor([[20.0, -20.0]]).expand(6, -1).clone()
    edge_logits = torch.full((upper_classes.numel(), 4), -20.0)
    edge_logits.scatter_(1, upper_classes[:, None], 20.0)
    return node_logits, edge_logits


class JointChemistryAuxiliaryTests(unittest.TestCase):
    def test_oracle_benzene_has_near_zero_loss(self):
        graph, upper_classes = _complete_benzene(masked=True)
        node_logits, edge_logits = _oracle_logits(upper_classes)
        losses = _chemistry_loss()(
            graph,
            node_logits,
            edge_logits,
            get_upper_edge_mask(graph),
        )
        self.assertEqual(
            set(losses),
            {
                "chem_edge_existence",
                "chem_valence",
                "chem_charge",
                "chem_connectivity",
                "chem_aromatic",
            },
        )
        for name, value in losses.items():
            with self.subTest(loss=name):
                self.assertTrue(torch.isfinite(value))
                self.assertLess(float(value), 1.0e-6)

    def test_no_bond_prediction_is_penalized(self):
        graph, upper_classes = _complete_benzene(masked=True)
        node_logits, oracle_edge_logits = _oracle_logits(upper_classes)
        oracle = _chemistry_loss()(
            graph,
            node_logits,
            oracle_edge_logits,
            get_upper_edge_mask(graph),
        )
        no_bond_logits = torch.full_like(oracle_edge_logits, -10.0)
        no_bond_logits[:, 0] = 10.0
        corrupted = _chemistry_loss()(
            graph,
            node_logits,
            no_bond_logits,
            get_upper_edge_mask(graph),
        )
        for name in (
            "chem_edge_existence",
            "chem_valence",
            "chem_connectivity",
            "chem_aromatic",
        ):
            with self.subTest(loss=name):
                self.assertGreater(float(corrupted[name]), float(oracle[name]))

    def test_all_single_aromatic_perimeter_fails_kekule_degree(self):
        graph, upper_classes = _complete_benzene(masked=True)
        node_logits, oracle_edge_logits = _oracle_logits(upper_classes)
        oracle = _chemistry_loss()(
            graph,
            node_logits,
            oracle_edge_logits,
            get_upper_edge_mask(graph),
        )
        all_single_classes = torch.where(
            upper_classes != 0, torch.ones_like(upper_classes), upper_classes
        )
        all_single_logits = torch.full_like(oracle_edge_logits, -20.0)
        all_single_logits.scatter_(1, all_single_classes[:, None], 20.0)
        corrupted = _chemistry_loss()(
            graph,
            node_logits,
            all_single_logits,
            get_upper_edge_mask(graph),
        )
        self.assertGreater(
            float(corrupted["chem_aromatic"]),
            float(oracle["chem_aromatic"]) + 0.1,
        )

    def test_common_graph_weight_scales_loss_without_cancellation(self):
        graph, upper_classes = _complete_benzene(masked=True)
        generator = torch.Generator().manual_seed(119)
        node_logits = torch.randn(6, 2, generator=generator)
        edge_logits = torch.randn(
            upper_classes.numel(), 4, generator=generator
        )
        module = _chemistry_loss()
        unit = module(
            graph,
            node_logits,
            edge_logits,
            get_upper_edge_mask(graph),
            graph_weights=torch.ones(1),
        )
        quarter = module(
            graph,
            node_logits,
            edge_logits,
            get_upper_edge_mask(graph),
            graph_weights=torch.full((1,), 0.25),
        )
        for name in unit:
            with self.subTest(loss=name):
                self.assertTrue(
                    torch.allclose(quarter[name], 0.25 * unit[name], atol=1.0e-6)
                )

    def test_unsupported_dominant_logit_cannot_zero_supported_mass(self):
        graph, upper_classes = _complete_benzene(masked=True)
        valence_probabilities = torch.zeros(3, 5)
        valence_probabilities[0, 3] = 1.0
        valence_probabilities[1, 4] = 1.0
        module = JointChemistryAuxiliaryLoss(
            state_charges=torch.tensor([0.0, 1.0, 0.0]),
            state_valence_probabilities=valence_probabilities,
            state_support=torch.tensor([True, True, False]),
        )
        graph.ndata["s_t"] = one_hot(
            torch.full((6,), 3, dtype=torch.long), num_classes=4
        ).float()
        node_logits = torch.tensor([[0.0, 0.0, 1000.0]]).expand(6, -1)
        _, edge_logits = _oracle_logits(upper_classes)
        effective_node, _, _ = module._effective_endpoint_probabilities(
            graph,
            node_logits,
            edge_logits,
            get_upper_edge_mask(graph),
        )
        self.assertTrue(torch.isfinite(effective_node).all())
        self.assertTrue(
            torch.allclose(effective_node.sum(dim=-1), torch.ones(6))
        )
        self.assertTrue(torch.equal(effective_node[:, 2], torch.zeros(6)))

    def test_known_ctmc_entries_do_not_depend_on_endpoint_logits(self):
        graph, upper_classes = _complete_benzene(masked=False)
        generator = torch.Generator().manual_seed(31)
        node_logits = torch.randn(6, 2, generator=generator)
        edge_logits = torch.randn(
            upper_classes.numel(), 4, generator=generator
        )
        losses = _chemistry_loss()(
            graph,
            node_logits,
            edge_logits,
            get_upper_edge_mask(graph),
        )
        for name, value in losses.items():
            with self.subTest(loss=name):
                self.assertLess(abs(float(value)), 1.0e-7)

    def test_random_logits_have_finite_node_and_edge_gradients(self):
        graph, upper_classes = _complete_benzene(masked=True)
        generator = torch.Generator().manual_seed(73)
        node_logits = torch.randn(
            6, 2, generator=generator, requires_grad=True
        )
        edge_logits = torch.randn(
            upper_classes.numel(), 4, generator=generator, requires_grad=True
        )
        losses = _chemistry_loss()(
            graph,
            node_logits,
            edge_logits,
            get_upper_edge_mask(graph),
        )
        sum(losses.values()).backward()
        for gradient in (node_logits.grad, edge_logits.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_missing_graph_auxiliary_labels_fail_early(self):
        graph, upper_classes = _complete_benzene(masked=True)
        del graph.edata["e_connectivity_weight"]
        node_logits, edge_logits = _oracle_logits(upper_classes)
        with self.assertRaisesRegex(ValueError, "e_connectivity_weight"):
            _chemistry_loss()(
                graph,
                node_logits,
                edge_logits,
                get_upper_edge_mask(graph),
            )


class ChemistryLossIntegrationTests(unittest.TestCase):
    def test_total_loss_includes_every_enabled_chemistry_term(self):
        model = SimpleNamespace(
            canonical_feat_order=["x", "s", "e"],
            total_loss_weights={"x": 3.0, "s": 1.0, "e": 2.0},
            chemistry_loss_weights={
                "chem_edge_existence": 0.5,
                "chem_valence": 0.05,
                "chem_charge": 0.05,
                "chem_connectivity": 0.25,
                "chem_aromatic": 0.2,
            },
        )
        losses = {
            "x": torch.tensor(1.0),
            "s": torch.tensor(2.0),
            "e": torch.tensor(3.0),
            "chem_edge_existence": torch.tensor(4.0),
            "chem_valence": torch.tensor(5.0),
            "chem_charge": torch.tensor(6.0),
            "chem_connectivity": torch.tensor(7.0),
            "chem_aromatic": torch.tensor(8.0),
        }
        total = FlowMol._total_from_feature_losses(model, losses)
        expected = (
            3.0 + 2.0 + 6.0 + 2.0 + 0.25 + 0.3 + 1.75 + 1.6
        )
        self.assertAlmostEqual(float(total), expected, places=5)

    def test_missing_enabled_chemistry_term_is_not_silently_ignored(self):
        model = SimpleNamespace(
            canonical_feat_order=["x"],
            total_loss_weights={"x": 1.0},
            chemistry_loss_weights={"chem_valence": 0.1},
        )
        with self.assertRaisesRegex(KeyError, "chem_valence"):
            FlowMol._total_from_feature_losses(
                model, {"x": torch.tensor(1.0)}
            )


class EffectiveResistanceTests(unittest.TestCase):
    def test_cycle_and_tree_obey_uniform_spanning_tree_identities(self):
        cycle = torch.zeros(6, 6, dtype=torch.long)
        for src, dst in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)):
            cycle[src, dst] = cycle[dst, src] = 1
        cycle_weights = uniform_spanning_tree_edge_weights(cycle)
        observed = cycle_weights[torch.triu(cycle.bool(), diagonal=1)]
        self.assertTrue(
            torch.allclose(observed, torch.full_like(observed, 5.0 / 6.0))
        )
        self.assertAlmostEqual(
            float(torch.triu(cycle_weights, diagonal=1).sum()), 5.0, places=5
        )

        tree = torch.zeros(4, 4, dtype=torch.long)
        for src, dst in ((0, 1), (1, 2), (1, 3)):
            tree[src, dst] = tree[dst, src] = 1
        tree_weights = uniform_spanning_tree_edge_weights(tree)
        observed = tree_weights[torch.triu(tree.bool(), diagonal=1)]
        self.assertTrue(torch.allclose(observed, torch.ones_like(observed)))


class ChemistryDatasetAlignmentTests(unittest.TestCase):
    def test_auxiliary_edges_follow_complete_graph_upper_lower_packing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            torch.save(
                (
                    torch.tensor([1.0]),
                    torch.full((6,), 1.0 / 6.0),
                    torch.full((4,), 0.25),
                    torch.full((1, 6), 1.0 / 6.0),
                ),
                tmp_path / "train_data_marginal_dists.pt",
            )
            torch.save(torch.tensor([1.0]), tmp_path / "train_data_joint_state_marginal.pt")
            torch.save(
                {
                    "positions": torch.tensor(
                        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]]
                    ),
                    "atom_types": torch.ones(3, 1),
                    "atom_charges": torch.zeros(3, dtype=torch.long),
                    "atom_states": torch.zeros(3, dtype=torch.long),
                    "bond_types": torch.tensor([1, 1]),
                    "bond_idxs": torch.tensor([[0, 1], [1, 2]]),
                    "bond_connectivity_weights": torch.tensor([1.0, 1.0]),
                    "bond_aromatic_flags": torch.tensor([True, False]),
                    "node_idx_array": torch.tensor([[0, 3]]),
                    "edge_idx_array": torch.tensor([[0, 2]]),
                },
                tmp_path / "train_data_processed.pt",
            )
            dataset = MoleculeDataset(
                "train",
                dataset_config={
                    "dataset_name": "qme14s",
                    "processed_data_dir": str(tmp_path),
                    "conditioning": {"enabled": False},
                    "use_joint_atom_state": True,
                    "n_atom_states": 1,
                    "n_bond_types": 4,
                    "require_chemistry_auxiliary_labels": True,
                },
                prior_config={
                    "x": {"type": "centered-normal", "align": True, "kwargs": {}},
                    "s": {"type": "ctmc", "align": False, "kwargs": {}},
                    "e": {"type": "ctmc", "align": False, "kwargs": {}},
                },
            )
            graph = dataset[0]

        expected_connectivity = torch.tensor([1.0, 0.0, 1.0] * 2)
        expected_aromatic = torch.tensor([True, False, False] * 2)
        self.assertTrue(
            torch.equal(graph.edata["e_connectivity_weight"], expected_connectivity)
        )
        self.assertTrue(
            torch.equal(graph.edata["e_aromatic_true"], expected_aromatic)
        )


if __name__ == "__main__":
    unittest.main()
