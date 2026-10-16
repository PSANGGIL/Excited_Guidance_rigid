import unittest

import dgl
import torch
from torch.nn.functional import one_hot

from molguidance.data_processing.conditional_count import ConditionalAtomCountSampler
from molguidance.data_processing.data_module import MoleculeDataModule
from molguidance.models.classifier_free_guidance import SetEmbeddingType
from molguidance.models.ctmc_vector_field import CTMCVectorField
from molguidance.models.interpolant_scheduler import InterpolantScheduler


def _scheduler():
    features = ['x', 'a', 'c', 'h', 'e']
    return InterpolantScheduler(
        canonical_feat_order=features,
        schedule_type={feature: 'linear' for feature in features},
    )


def _vector_field():
    return CTMCVectorField(
        n_atom_types=3,
        n_charges=2,
        n_hydrogen_counts=4,
        n_bond_types=5,
        canonical_feat_order=['x', 'a', 'c', 'h', 'e'],
        interpolant_scheduler=_scheduler(),
        n_vec_channels=3,
        n_hidden_scalars=8,
        n_hidden_edge_feats=8,
        n_molecule_updates=1,
        convs_per_update=1,
        n_message_gvps=1,
        n_update_gvps=1,
        conditional_generation=False,
    )


def _training_graph():
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata['x_0'] = torch.zeros(2, 3)
    graph.ndata['x_1_true'] = torch.ones(2, 3)
    graph.ndata['a_1_true'] = one_hot(torch.tensor([0, 1]), 3).float()
    graph.ndata['c_1_true'] = one_hot(torch.tensor([0, 1]), 2).float()
    graph.ndata['h_1_true'] = one_hot(torch.tensor([2, 0]), 4).float()
    graph.edata['e_1_true'] = one_hot(torch.tensor([1, 1]), 5).float()
    return graph


class ConditionalCountTests(unittest.TestCase):
    def test_kernel_sampler_tracks_target_region(self):
        sampler = ConditionalAtomCountSampler(
            atom_counts=torch.tensor([5, 5, 10, 10]),
            properties=torch.tensor([-1.0, -0.9, 0.9, 1.0]),
        )
        generator = torch.Generator().manual_seed(7)
        low = sampler.sample(
            torch.full((64,), -1.0), 0.05, nearest_k=2, generator=generator
        )
        generator.manual_seed(7)
        high = sampler.sample(
            torch.full((64,), 1.0), 0.05, nearest_k=2, generator=generator
        )
        self.assertTrue(torch.equal(low, torch.full((64,), 5)))
        self.assertTrue(torch.equal(high, torch.full((64,), 10)))

    def test_vector_condition_dimension_is_checked(self):
        sampler = ConditionalAtomCountSampler(
            atom_counts=[5, 6], properties=torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        )
        with self.assertRaisesRegex(ValueError, 'dimension'):
            sampler.probabilities(torch.tensor([0.0, 1.0, 2.0]), [0.1, 0.1])


class DeterministicCorruptionTests(unittest.TestCase):
    def test_cfg_dropout_can_be_disabled(self):
        graph = dgl.batch([_training_graph(), _training_graph()])
        graph.prop = torch.tensor([0.0, 1.0])
        controller = SetEmbeddingType(p_unconditional=1.0)
        controller(graph, enabled=False)
        self.assertFalse(graph._USE_UNCONDITIONAL_EMBEDDING.any())

    def test_ctmc_corruption_repeats_for_same_generator_seed(self):
        vector_field = _vector_field()
        graph_a = dgl.batch([_training_graph(), _training_graph()])
        graph_b = dgl.batch([_training_graph(), _training_graph()])
        node_batch_idx = torch.tensor([0, 0, 1, 1])
        edge_batch_idx = torch.tensor([0, 0, 1, 1])
        upper_edge_mask = torch.tensor([True, False, True, False])
        time = torch.tensor([0.35, 0.7])

        vector_field.sample_conditional_path(
            graph_a,
            time,
            node_batch_idx,
            edge_batch_idx,
            upper_edge_mask,
            generator=torch.Generator().manual_seed(123),
        )
        vector_field.sample_conditional_path(
            graph_b,
            time,
            node_batch_idx,
            edge_batch_idx,
            upper_edge_mask,
            generator=torch.Generator().manual_seed(123),
        )
        for feature in ('a', 'c', 'h'):
            self.assertTrue(
                torch.equal(graph_a.ndata[f'{feature}_t'], graph_b.ndata[f'{feature}_t'])
            )
        self.assertTrue(torch.equal(graph_a.edata['e_t'], graph_b.edata['e_t']))


class ExplicitSplitTests(unittest.TestCase):
    def test_explicit_training_split_overrides_legacy_half_flag(self):
        module = MoleculeDataModule(
            dataset_config={
                'training_split': 'train',
                'use_first_half_training_set': True,
            },
            dm_prior_config={},
            batch_size=1,
        )
        loaded = []

        def fake_load(split):
            loaded.append(split)
            return split

        module.load_dataset = fake_load
        module.setup('fit')
        self.assertEqual(module.training_split, 'train')
        self.assertEqual(module.train_dataset, 'train')
        self.assertEqual(loaded, ['val', 'test', 'train'])


if __name__ == '__main__':
    unittest.main()
