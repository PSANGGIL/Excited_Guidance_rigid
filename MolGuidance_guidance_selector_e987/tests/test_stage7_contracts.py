import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn

from molguidance.models.loss_utils import reduce_categorical_loss
from molguidance.data_processing.priors import rigid_alignment
from molguidance.data_processing.utils import get_batch_idxs, get_upper_edge_mask
from molguidance.models.ctmc_vector_field import CTMCVectorField
from molguidance.models.flowmol import FlowMol
from molguidance.model_utils.provenance import sha256_file
from test_independent_time_bond_first import _field, _graph, _scheduler


class Stage7ContractTests(unittest.TestCase):
    def test_categorical_value_and_gradient_match_explicit_estimators(self):
        for mode, value, grad in (
            ("legacy_factorized", 4., [1., 1.]),
            ("aligned_all", 5., [.5, 1.5]),
            ("masked_weighted", 5., [.5, 1.5]),
            ("masked_unweighted", 2., [.5, .5]),
        ):
            loss = torch.tensor([1., 3.], requires_grad=True)
            actual = reduce_categorical_loss(
                loss, torch.tensor([[1.], [3.]]), torch.ones(2, dtype=torch.bool), mode
            )
            self.assertAlmostEqual(float(actual), value)
            actual.backward()
            torch.testing.assert_close(loss.grad, torch.tensor(grad))

    def test_clean_entries_cannot_reweight_other_graphs_in_new_mode(self):
        loss = torch.tensor([2., 0., 0.], requires_grad=True)
        active = torch.tensor([True, False, False])
        for weight in (torch.tensor([.3, 1., 1.]), torch.tensor([.3, 999., 1000.])):
            actual = reduce_categorical_loss(loss, weight, active)
            self.assertAlmostEqual(float(actual), .6, places=6)
        self.assertEqual(float(reduce_categorical_loss(loss, 1., ~torch.ones_like(active))), 0.)
        self.assertEqual(float(reduce_categorical_loss(loss[:0], 1., active[:0])), 0.)

    def test_rigid_alignment_translation_and_reflection_contracts(self):
        x = torch.randn(12, 3, generator=torch.Generator().manual_seed(31))
        rotation, _ = torch.linalg.qr(torch.randn(3, 3))
        y = x @ rotation + torch.tensor([10., -3., 4.])
        torch.testing.assert_close(rigid_alignment(x, y), y, atol=2e-6, rtol=1e-6)
        mirror = x * torch.tensor([-1., 1., 1.])
        torch.testing.assert_close(rigid_alignment(x, mirror), mirror, atol=2e-6, rtol=1e-6)
        self.assertGreater(float((rigid_alignment(x, mirror, allow_reflection=False) - mirror).square().sum()), .1)

    def test_full_multilayer_field_reflection_equivariance(self):
        torch.manual_seed(14)
        field = _field(independent=True, bond_first=True).eval()  # n_cp_feats=0
        graph = _graph(4)
        node_batch, edge_batch = get_batch_idxs(graph)
        upper = get_upper_edge_mask(graph)
        time = torch.tensor([.55])
        field.sample_conditional_path(
            graph, time, node_batch, edge_batch, upper,
            generator=torch.Generator().manual_seed(71), graph_time=time,
        )
        first = field(graph, time, node_batch, upper, graph_time=time)
        mirror = torch.tensor([-1., 1., 1.])
        graph.ndata["x_t"] = graph.ndata["x_t"] * mirror
        second = field(graph, time, node_batch, upper, graph_time=time)
        torch.testing.assert_close(second["x"], first["x"] * mirror, atol=2e-5, rtol=2e-5)
        for key in ("s", "e"):
            torch.testing.assert_close(second[key], first[key], atol=2e-5, rtol=2e-5)

    def test_raw_ev_sampling_matches_training_embedding_with_residual_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            norm = Path(tmp) / "train_data_property_normalization.pt"
            torch.save(dict(mean=torch.tensor(.5), std=torch.tensor(.25)), norm)
            field = CTMCVectorField(
                n_atom_types=2, n_atom_states=3, n_charges=0, n_bond_types=4,
                canonical_feat_order=["x", "s", "e"], interpolant_scheduler=_scheduler(),
                n_vec_channels=3, n_hidden_scalars=8, n_hidden_edge_feats=8,
                n_molecule_updates=2, convs_per_update=1, n_message_gvps=1,
                n_update_gvps=1, rbf_dim=6, property_embedding_dim=4,
                property_embedder=nn.Linear(1, 4), conditional_generation=True,
                properties_handle_method="concatenate_sum", condition_residual_scale=.5,
                dataset_name="qme14s", dropout=.1,
            ).eval()
            self.assertTrue(all(layer.dropout_rate == .1 for layer in field.conv_layers))
            graph = _graph(4)
            node_batch, edge_batch = get_batch_idxs(graph)
            upper, time = get_upper_edge_mask(graph), torch.tensor([.55])
            field.sample_conditional_path(graph, time, node_batch, edge_batch, upper)
            embedding = field.property_embedder(torch.tensor([[(0. - .5) / .25]]))
            train_output = field(graph, torch.cat((time[:, None], embedding), -1), node_batch, upper)
            field.properties_for_sampling = 0.
            field.multilple_values_to_one_property = None
            field.normalization_file_path = str(norm)
            sample_output = field(graph, time, node_batch, upper)
            for key in train_output:
                torch.testing.assert_close(train_output[key], sample_output[key])

    def test_normalization_is_automatic_and_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            norm = path / "train_data_property_normalization.pt"
            torch.save(dict(mean=torch.tensor(.5), std=torch.tensor(.25)), norm)
            model = FlowMol.__new__(FlowMol)
            nn.Module.__init__(model)
            model.enforce_data_manifest = True
            model.marginal_dists_file = path / "train_data_marginal_dists.pt"
            model.data_manifest = dict(sha256={norm.name: sha256_file(norm)})
            self.assertEqual(model.resolve_normalization_file(None), str(norm))
            torch.save(dict(mean=torch.tensor(.6), std=torch.tensor(.25)), norm)
            with self.assertRaisesRegex(ValueError, "normalization"):
                model.resolve_normalization_file(str(norm))


if __name__ == "__main__":
    unittest.main()
