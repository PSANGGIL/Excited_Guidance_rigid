import unittest

import dgl
import torch
from torch import nn
from torch.nn.functional import one_hot

from molguidance.data_processing.utils import (
    build_edge_idxs,
    get_batch_idxs,
    get_upper_edge_mask,
)
from molguidance.model_utils.load import validate_chemistry_schema
from molguidance.models.ctmc_vector_field import CTMCVectorField
from molguidance.models.flowmol import FlowMol
from molguidance.models.interpolant_scheduler import InterpolantScheduler
from molguidance.models.model_guidance_scale import ModelGuidance
from molguidance.models.vector_field import BondFirstCoordinateUpdate


def _scheduler():
    return InterpolantScheduler(
        canonical_feat_order=['x', 's', 'e'],
        schedule_type={'x': 'linear', 's': 'linear', 'e': 'linear'},
    )


def _field(*, independent=False, bond_first=False):
    return CTMCVectorField(
        n_atom_types=2,
        n_atom_states=3,
        category_support_masks={'s': [True, True, True]},
        n_charges=0,
        n_bond_types=4,
        canonical_feat_order=['x', 's', 'e'],
        interpolant_scheduler=_scheduler(),
        n_vec_channels=3,
        n_hidden_scalars=8,
        n_hidden_edge_feats=8,
        n_molecule_updates=2,
        convs_per_update=1,
        n_message_gvps=1,
        n_update_gvps=1,
        separate_mol_updaters=True,
        rbf_dim=6,
        conditional_generation=False,
        independent_modality_time=independent,
        bond_first_geometry=bond_first,
    )


def _graph(n_atoms=3):
    edges = build_edge_idxs(n_atoms)
    graph = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
    graph.ndata['x_0'] = torch.arange(
        n_atoms * 3, dtype=torch.float32
    ).reshape(n_atoms, 3) / 10.0
    graph.ndata['x_1_true'] = graph.ndata['x_0'] + 2.0
    graph.ndata['s_1_true'] = one_hot(
        torch.arange(n_atoms) % 3, num_classes=3
    ).float()
    upper = get_upper_edge_mask(graph)
    edge_class = torch.arange(int(upper.sum())) % 4
    edge_class = torch.cat((edge_class, edge_class))
    graph.edata['e_1_true'] = one_hot(edge_class, num_classes=4).float()
    return graph


class IndependentModalityTimeTests(unittest.TestCase):
    def test_disabled_architecture_preserves_legacy_parameter_contract(self):
        torch.manual_seed(9)
        legacy = _field()
        torch.manual_seed(9)
        explicit_disabled = _field(independent=False, bond_first=False)
        self.assertEqual(
            tuple(legacy.state_dict()), tuple(explicit_disabled.state_dict())
        )
        for key in legacy.state_dict():
            self.assertTrue(torch.equal(
                legacy.state_dict()[key], explicit_disabled.state_dict()[key]
            ), key)
        self.assertEqual(len(legacy.bond_coordinate_updaters), 0)

    def test_independent_time_adds_one_input_and_bond_modules_only(self):
        legacy = _field()
        stage6 = _field(independent=True, bond_first=True)
        self.assertEqual(stage6.node_input_dim, legacy.node_input_dim + 1)
        self.assertEqual(len(stage6.bond_coordinate_updaters), 2)
        self.assertIsNone(legacy.molecule_update_index(0))
        self.assertEqual(stage6.molecule_update_index(0), 0)
        self.assertEqual(stage6.molecule_update_index(1), 1)

    def test_training_times_are_independent_and_graph_clean_branch_is_exact(self):
        model = FlowMol.__new__(FlowMol)
        nn.Module.__init__(model)
        model.independent_modality_time = True
        model.graph_clean_probability = 1.0
        first = model.resolve_modality_times(
            batch_size=32,
            device=torch.device('cpu'),
            coordinate_time=None,
            graph_time=None,
            generator=FlowMol._seeded_generator(torch.device('cpu'), 17),
        )
        second = model.resolve_modality_times(
            batch_size=32,
            device=torch.device('cpu'),
            coordinate_time=None,
            graph_time=None,
            generator=FlowMol._seeded_generator(torch.device('cpu'), 17),
        )
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[1], torch.ones(32)))
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_explicit_coordinate_time_defaults_to_diagonal_sampling_path(self):
        model = FlowMol.__new__(FlowMol)
        nn.Module.__init__(model)
        model.independent_modality_time = True
        model.graph_clean_probability = 0.5
        coordinate = torch.tensor([0.2, 0.8])
        actual_coordinate, actual_graph = model.resolve_modality_times(
            batch_size=2,
            device=torch.device('cpu'),
            coordinate_time=coordinate,
            graph_time=None,
            generator=None,
        )
        self.assertTrue(torch.equal(actual_coordinate, coordinate))
        self.assertTrue(torch.equal(actual_graph, coordinate))

    def test_schedule_columns_use_their_own_times(self):
        model = FlowMol.__new__(FlowMol)
        nn.Module.__init__(model)
        model.independent_modality_time = True
        model.interpolant_scheduler = _scheduler()
        model.canonical_feat_order = ['x', 's', 'e']
        coordinate = torch.tensor([0.2])
        graph = torch.tensor([0.8])
        actual = model.modality_schedule_values(
            'loss_weights', coordinate, graph
        )
        coordinate_weights = model.interpolant_scheduler.loss_weights(coordinate)
        graph_weights = model.interpolant_scheduler.loss_weights(graph)
        self.assertEqual(float(actual[0, 0]), float(coordinate_weights[0, 0]))
        self.assertEqual(float(actual[0, 1]), float(graph_weights[0, 1]))
        self.assertEqual(float(actual[0, 2]), float(graph_weights[0, 2]))

    def test_graph_clean_time_leaves_coordinates_noisy_but_unmasks_graph(self):
        field = _field(independent=True)
        graph = _graph()
        node_batch, edge_batch = get_batch_idxs(graph)
        upper = get_upper_edge_mask(graph)
        field.sample_conditional_path(
            graph,
            torch.zeros(1),
            node_batch,
            edge_batch,
            upper,
            generator=torch.Generator().manual_seed(4),
            graph_time=torch.ones(1),
        )
        self.assertTrue(torch.equal(graph.ndata['x_t'], graph.ndata['x_0']))
        self.assertTrue(torch.equal(
            graph.ndata['s_t'].argmax(-1),
            graph.ndata['s_1_true'].argmax(-1),
        ))
        self.assertTrue(torch.equal(
            graph.edata['e_t'][upper].argmax(-1),
            graph.edata['e_1_true'][upper].argmax(-1),
        ))

    def test_field_consumes_graph_time_and_remains_finite(self):
        torch.manual_seed(3)
        field = _field(independent=True, bond_first=True).eval()
        graph = _graph()
        node_batch, edge_batch = get_batch_idxs(graph)
        upper = get_upper_edge_mask(graph)
        field.sample_conditional_path(
            graph,
            torch.full((1,), 0.4),
            node_batch,
            edge_batch,
            upper,
            generator=torch.Generator().manual_seed(8),
            graph_time=torch.full((1,), 0.4),
        )
        low = field(
            graph, torch.full((1,), 0.4), node_batch, upper,
            graph_time=torch.full((1,), 0.1),
        )
        high = field(
            graph, torch.full((1,), 0.4), node_batch, upper,
            graph_time=torch.full((1,), 0.9),
        )
        self.assertTrue(all(torch.isfinite(value).all() for value in low.values()))
        self.assertFalse(torch.equal(low['s'], high['s']))

    def test_schema_rejects_mismatched_time_flags(self):
        config = {
            'dataset': {
                'use_joint_atom_state': True,
                'n_atom_states': 3,
                'n_bond_types': 4,
                'use_hydrogen_counts': False,
            },
            'mol_fm': {
                'n_atom_states': 3,
                'n_bond_types': 4,
                'n_hydrogen_counts': 0,
                'parameterization': 'ctmc',
                'self_conditioning_rate': 0.0,
                'modality_time': {'enabled': True},
            },
            'vector_field': {
                'self_conditioning': False,
                'independent_modality_time': False,
            },
        }
        with self.assertRaisesRegex(ValueError, 'disagree'):
            validate_chemistry_schema(config)

    def test_model_guidance_validation_disables_training_only_branches(self):
        class ValidationProbe(ModelGuidance):
            def __init__(self):
                nn.Module.__init__(self)
                self.forward_kwargs = None

            def forward(self, graph, **kwargs):
                self.forward_kwargs = kwargs
                return {'x': torch.tensor(0.0)}

        probe = ValidationProbe()
        coordinate_time = torch.tensor([0.2, 0.8])
        graph_time = torch.tensor([1.0, 0.3])
        generator = torch.Generator().manual_seed(5)
        probe._validation_forward(
            None,
            t=coordinate_time,
            graph_time=graph_time,
            corruption_generator=generator,
        )
        self.assertFalse(probe.forward_kwargs['is_training'])
        self.assertFalse(probe.forward_kwargs['condition_dropout'])
        self.assertTrue(probe.forward_kwargs['self_conditioning'])
        self.assertIs(probe.forward_kwargs['corruption_generator'], generator)
        self.assertTrue(torch.equal(probe.forward_kwargs['t'], coordinate_time))
        self.assertTrue(torch.equal(
            probe.forward_kwargs['graph_time'], graph_time
        ))


class BondFirstCoordinateTests(unittest.TestCase):
    def _active_layer(self):
        torch.manual_seed(31)
        layer = BondFirstCoordinateUpdate(5, 4, 6, coordinate_scale=0.2)
        nn.init.normal_(layer.gate[-1].weight, std=0.15)
        nn.init.normal_(layer.gate[-1].bias, std=0.05)
        return layer

    def test_pair_update_is_equivariant_translation_invariant_and_com_safe(self):
        graph = _graph(4)
        upper = get_upper_edge_mask(graph)
        layer = self._active_layer()
        node = torch.randn(4, 5)
        edge = torch.randn(graph.num_edges(), 4)
        distance = torch.randn(graph.num_edges(), 6)
        positions = torch.randn(4, 3)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = torch.tensor([2.0, -4.0, 1.5])

        reference = layer(graph, node, edge, positions, distance, upper)
        transformed = layer(
            graph,
            node,
            edge,
            positions @ rotation.T + translation,
            distance,
            upper,
        )
        self.assertTrue(torch.allclose(
            transformed, reference @ rotation.T + translation, atol=2.0e-6
        ))
        self.assertTrue(torch.allclose(
            reference.mean(0), positions.mean(0), atol=2.0e-6
        ))

    def test_pair_update_is_reflection_equivariant(self):
        graph = _graph(4)
        upper = get_upper_edge_mask(graph)
        layer = self._active_layer()
        node = torch.randn(4, 5)
        edge = torch.randn(graph.num_edges(), 4)
        distance = torch.randn(graph.num_edges(), 6)
        positions = torch.randn(4, 3)
        reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))

        reference = layer(graph, node, edge, positions, distance, upper)
        reflected = layer(
            graph,
            node,
            edge,
            positions @ reflection.T,
            distance,
            upper,
        )
        self.assertTrue(torch.allclose(
            reflected, reference @ reflection.T, atol=2.0e-6
        ))

    def test_zero_initialization_is_identity_then_learns(self):
        torch.manual_seed(47)
        graph = _graph(3)
        upper = get_upper_edge_mask(graph)
        layer = BondFirstCoordinateUpdate(5, 4, 6, coordinate_scale=0.2)
        node = torch.randn(3, 5)
        edge = torch.randn(graph.num_edges(), 4)
        distance = torch.randn(graph.num_edges(), 6)
        positions = torch.randn(3, 3)

        initial = layer(graph, node, edge, positions, distance, upper)
        self.assertTrue(torch.equal(initial, positions))

        target = positions.clone()
        target[0] += torch.tensor([0.4, -0.2, 0.1])
        target[1] -= torch.tensor([0.4, -0.2, 0.1])
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.5)
        loss = (initial - target).square().sum()
        loss.backward()
        final_layer = layer.gate[-1]
        self.assertGreater(float(final_layer.weight.grad.abs().sum()), 0.0)
        optimizer.step()

        updated = layer(graph, node, edge, positions, distance, upper)
        self.assertFalse(torch.equal(updated, positions))
        self.assertTrue(torch.isfinite(updated).all())
        self.assertTrue(torch.allclose(
            updated.mean(0), positions.mean(0), atol=2.0e-6
        ))

    def test_refreshed_edge_state_controls_geometry_and_receives_gradient(self):
        graph = _graph(3)
        upper = get_upper_edge_mask(graph)
        layer = self._active_layer()
        node = torch.randn(3, 5)
        edge = torch.randn(graph.num_edges(), 4, requires_grad=True)
        distance = torch.randn(graph.num_edges(), 6)
        positions = torch.randn(3, 3)
        first = layer(graph, node, edge, positions, distance, upper)
        changed_edge = edge.detach().clone()
        changed_edge[0] += 3.0
        second = layer(graph, node, changed_edge, positions, distance, upper)
        self.assertFalse(torch.allclose(first, second))
        first.square().sum().backward()
        self.assertIsNotNone(edge.grad)
        self.assertGreater(float(edge.grad.abs().sum()), 0.0)


if __name__ == '__main__':
    unittest.main()
