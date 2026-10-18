import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dgl
import torch
from rdkit import Chem
from torch.nn.functional import one_hot

from molguidance.analysis.molecule_builder import dataset_mol_to_rdmol
from molguidance.data_processing.dataset import MoleculeDataset
from molguidance.data_processing.priors import edge_prior
from molguidance.models.classifier_free_guidance import CFGVectorField
from molguidance.models.ctmc_vector_field import CTMCVectorField
from molguidance.models.model_guidance_scale import ScaleAwareMGVectorField
from molguidance.models.interpolant_scheduler import InterpolantScheduler
from molguidance.model_utils.load import validate_chemistry_schema
from molguidance.utils.guidance_sampler import _assert_sampling_compatible


def _joint_scheduler():
    features = ['x', 's', 'e']
    return InterpolantScheduler(
        canonical_feat_order=features,
        schedule_type={feature: 'linear' for feature in features},
    )


class JointAtomStateKekuleTests(unittest.TestCase):
    def test_edge_prior_uses_configured_bond_vocabulary(self):
        upper = torch.tensor([True, True, False, False])
        config = {'type': 'ctmc', 'kwargs': {}}

        self.assertEqual(
            edge_prior(upper, config, n_bond_types=4).shape, (4, 5)
        )
        self.assertEqual(edge_prior(upper, config).shape, (4, 6))

    def test_joint_vector_field_masks_held_out_state_with_finite_logits(self):
        vector_field = CTMCVectorField(
            n_atom_types=2,
            n_atom_states=3,
            category_support_masks={'s': [True, False, True]},
            n_charges=0,
            n_bond_types=4,
            canonical_feat_order=['x', 's', 'e'],
            interpolant_scheduler=_joint_scheduler(),
            n_vec_channels=3,
            n_hidden_scalars=8,
            n_hidden_edge_feats=8,
            n_molecule_updates=1,
            convs_per_update=1,
            n_message_gvps=1,
            n_update_gvps=1,
            conditional_generation=False,
        )

        raw_logits = vector_field.split_node_output(torch.zeros(4, 3))['s']
        self.assertTrue(torch.equal(raw_logits, torch.zeros(4, 3)))
        logits = vector_field.split_node_output(
            torch.zeros(4, 3), enforce_support=True
        )['s']
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(logits[:, [0, 2]], torch.zeros(4, 2)))
        self.assertTrue(
            torch.equal(logits[:, 1], torch.full((4,), -1.0e4))
        )
        self.assertEqual(
            vector_field.scalar_embedding_uncond[0].in_features, 3 + 1 + 1
        )
        self.assertEqual(vector_field.edge_embedding[0].in_features, 4 + 1)

    def test_scale_aware_mg_uses_joint_node_contract(self):
        vector_field = ScaleAwareMGVectorField(
            n_atom_types=2,
            n_atom_states=3,
            category_support_masks={'s': [True, False, True]},
            n_charges=0,
            n_bond_types=4,
            canonical_feat_order=['x', 's', 'e'],
            interpolant_scheduler=_joint_scheduler(),
            n_vec_channels=3,
            n_hidden_scalars=8,
            n_hidden_edge_feats=8,
            n_molecule_updates=1,
            convs_per_update=1,
            n_message_gvps=1,
            n_update_gvps=1,
            conditional_generation=False,
        )

        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        graph.ndata['x_t'] = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        graph.ndata['s_t'] = one_hot(
            torch.tensor([3, 0]), num_classes=4
        ).float()
        graph.edata['e_t'] = one_hot(
            torch.tensor([4, 4]), num_classes=5
        ).float()
        output = vector_field(
            graph,
            t=torch.tensor([0.5]),
            node_batch_idx=torch.zeros(2, dtype=torch.long),
            upper_edge_mask=torch.tensor([True, False]),
            guidance_scale=1.0,
            apply_softmax=True,
        )

        self.assertEqual(set(output), {'x', 's', 'e'})
        self.assertEqual(output['s'].shape, (2, 3))
        self.assertEqual(output['e'].shape, (1, 4))
        self.assertTrue(torch.isfinite(output['x']).all())
        self.assertTrue(torch.isfinite(output['s']).all())
        self.assertTrue(torch.all(output['s'][:, 1] == 0.0))

    def test_auto_guidance_rejects_different_support_masks(self):
        vocabulary = [
            {'symbol': 'C', 'formal_charge': 0, 'hydrogens': 0},
            {'symbol': 'N', 'formal_charge': 0, 'hydrogens': 0},
        ]

        def make_model(support):
            vector_field = SimpleNamespace(
                get_category_support_mask=lambda feat: (
                    support if feat == 's' else None
                )
            )
            return SimpleNamespace(
                canonical_feat_order=['x', 's', 'e'],
                atom_state_vocab=vocabulary,
                vector_field=vector_field,
            )

        good_model = make_model(torch.tensor([True, True]))
        bad_model = make_model(torch.tensor([True, False]))
        with self.assertRaisesRegex(ValueError, 'different support masks'):
            _assert_sampling_compatible(good_model, bad_model)

    def test_cfg_final_step_cannot_emit_unsupported_state(self):
        n_items = 256
        support = torch.tensor([True, False, True])
        # Make the unsupported class overwhelmingly likely before projection.
        logits = torch.tensor([[0.0, 1000.0, 0.0]]).expand(n_items, -1)
        probabilities = torch.softmax(logits, dim=-1)
        masked_state = torch.full((n_items,), 3, dtype=torch.long)

        for where in ('probabilities', 'rate_matrix'):
            for guidance_format in ('linear', 'log'):
                with self.subTest(where=where, guidance_format=guidance_format):
                    state, endpoint = (
                        CFGVectorField.campbell_step_with_rate_matrix_cfg(
                            p_1_given_t_uncond=probabilities,
                            p_1_given_t_cond=probabilities,
                            xt=masked_state,
                            stochasticity=1.0,
                            alpha_t=torch.tensor(0.5),
                            alpha_t_prime=torch.tensor(1.0),
                            dt=torch.tensor(1.0),
                            guide_weight=1.0,
                            mask_index=3,
                            n_classes=4,
                            uncond_val=logits,
                            cond_val=logits,
                            last_step=True,
                            guidance_format=guidance_format,
                            where_to_apply_guide=where,
                            temperature=1.0,
                            class_support_mask=support,
                        )
                    )
                    self.assertFalse(torch.any(state.argmax(-1) == 1))
                    self.assertFalse(torch.any(endpoint.argmax(-1) == 1))

    def test_config_rejects_dataset_model_vocabulary_mismatch(self):
        config = {
            'dataset': {
                'use_joint_atom_state': True,
                'n_atom_states': 37,
                'n_bond_types': 4,
                'use_hydrogen_counts': False,
            },
            'mol_fm': {
                'n_atom_states': 36,
                'n_bond_types': 4,
                'n_hydrogen_counts': 0,
            },
        }
        with self.assertRaisesRegex(ValueError, 'n_atom_states.*disagree'):
            validate_chemistry_schema(config)

        config['mol_fm']['n_atom_states'] = 37
        config['mol_fm']['n_bond_types'] = 5
        with self.assertRaisesRegex(ValueError, 'n_bond_types.*disagree'):
            validate_chemistry_schema(config)

        config['mol_fm']['n_bond_types'] = 4
        validate_chemistry_schema(config)

    def test_joint_dataset_emits_joint_state_and_four_class_kekule_edge(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            torch.save(
                (
                    torch.tensor([0.5, 0.5]),
                    torch.full((6,), 1.0 / 6.0),
                    torch.full((4,), 0.25),
                    torch.full((2, 6), 1.0 / 6.0),
                ),
                tmp_path / 'train_data_marginal_dists.pt',
            )
            torch.save(
                {
                    'positions': torch.tensor(
                        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
                    ),
                    'atom_types': one_hot(torch.tensor([0, 0]), 2).float(),
                    # Joint states own the charge vocabulary. The loader must
                    # not apply the legacy [-2, 3] one-hot conversion first.
                    'atom_charges': torch.full((2,), 7, dtype=torch.long),
                    'atom_states': torch.tensor([0, 0]),
                    'bond_types': torch.tensor([1]),
                    'bond_idxs': torch.tensor([[0, 1]]),
                    'node_idx_array': torch.tensor([[0, 2]]),
                    'edge_idx_array': torch.tensor([[0, 1]]),
                },
                tmp_path / 'train_data_processed.pt',
            )

            dataset = MoleculeDataset(
                'train',
                dataset_config={
                    'dataset_name': 'qme14s',
                    'processed_data_dir': str(tmp_path),
                    'conditioning': {'enabled': False},
                    'use_joint_atom_state': True,
                    'n_atom_states': 2,
                    'n_bond_types': 4,
                },
                prior_config={
                    'x': {
                        'type': 'centered-normal',
                        'align': True,
                        'kwargs': {},
                    },
                    's': {'type': 'ctmc', 'align': False, 'kwargs': {}},
                    'e': {'type': 'ctmc', 'align': False, 'kwargs': {}},
                },
            )
            graph = dataset[0]

        self.assertEqual(
            set(graph.ndata), {'x_1_true', 's_1_true', 'x_0', 's_0'}
        )
        self.assertEqual(graph.ndata['s_1_true'].shape, (2, 2))
        self.assertEqual(graph.ndata['s_0'].shape, (2, 3))
        self.assertEqual(graph.edata['e_1_true'].shape, (2, 4))
        self.assertEqual(graph.edata['e_0'].shape, (2, 5))

    def test_kekule_benzene_decodes_to_rdkit_aromatic_cycle(self):
        n_atoms = 6
        upper_edges = torch.triu_indices(n_atoms, n_atoms, offset=1)
        lower_edges = upper_edges.flip(0)
        edges = torch.cat([upper_edges, lower_edges], dim=1)
        graph = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)

        angles = torch.arange(n_atoms) * (2.0 * math.pi / n_atoms)
        graph.ndata['x_1_true'] = torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.zeros(n_atoms)],
            dim=1,
        )
        graph.ndata['s_1_true'] = torch.ones(n_atoms, 1)

        ring_bonds = {
            (0, 1): 1,
            (1, 2): 2,
            (2, 3): 1,
            (3, 4): 2,
            (4, 5): 1,
            (0, 5): 2,
        }
        upper_labels = torch.tensor(
            [ring_bonds.get((int(i), int(j)), 0) for i, j in upper_edges.T]
        )
        edge_labels = torch.cat([upper_labels, upper_labels])
        graph.edata['e_1_true'] = one_hot(
            edge_labels, num_classes=4
        ).float()

        molecule = dataset_mol_to_rdmol(
            graph,
            atom_type_map=['C'],
            atom_state_vocab=[
                {'symbol': 'C', 'formal_charge': 0, 'hydrogens': 1}
            ],
        )
        Chem.SanitizeMol(molecule)

        self.assertEqual(Chem.MolToSmiles(molecule), 'c1ccccc1')
        self.assertEqual(
            sum(bond.GetIsAromatic() for bond in molecule.GetBonds()), 6
        )


if __name__ == '__main__':
    unittest.main()
