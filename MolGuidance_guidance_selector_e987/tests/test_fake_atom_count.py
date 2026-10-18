import json
import tempfile
import unittest
from pathlib import Path

import dgl
import torch
from torch import nn
from torch.nn.functional import one_hot

from molguidance.analysis.molecule_builder import SampledMolecule
from molguidance.data_processing.dataset import MoleculeDataset
from molguidance.data_processing.utils import get_upper_edge_mask
from molguidance.models.chemistry_auxiliary import JointChemistryAuxiliaryLoss
from molguidance.models.flowmol import FlowMol
from molguidance.model_utils.load import validate_chemistry_schema


def _prior_config():
    return {
        'x': {'type': 'centered-normal', 'align': True, 'kwargs': {}},
        's': {'type': 'ctmc', 'align': False, 'kwargs': {}},
        'e': {'type': 'ctmc', 'align': False, 'kwargs': {}},
    }


def _write_tiny_dataset(path: Path, split: str = 'val') -> None:
    torch.save(
        (
            torch.tensor([0.5, 0.5]),
            torch.full((6,), 1.0 / 6.0),
            torch.full((4,), 0.25),
            torch.full((2, 6), 1.0 / 6.0),
        ),
        path / 'train_data_marginal_dists.pt',
    )
    torch.save(
        {
            'positions': torch.tensor(
                [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]]
            ),
            'atom_types': one_hot(torch.tensor([0, 0, 1]), 2).float(),
            'atom_charges': torch.zeros(3, dtype=torch.long),
            'atom_states': torch.tensor([0, 0, 1]),
            'bond_types': torch.tensor([1, 1]),
            'bond_idxs': torch.tensor([[0, 1], [1, 2]]),
            'bond_connectivity_weights': torch.tensor([1.0, 1.0]),
            'bond_aromatic_flags': torch.tensor([False, False]),
            'node_idx_array': torch.tensor([[0, 3]]),
            'edge_idx_array': torch.tensor([[0, 2]]),
        },
        path / f'{split}_data_processed.pt',
    )


def _complete_graph(state_classes, edge_classes):
    n_atoms = len(state_classes)
    upper = torch.triu_indices(n_atoms, n_atoms, offset=1)
    edges = torch.cat((upper, upper.flip(0)), dim=1)
    graph = dgl.graph((edges[0], edges[1]), num_nodes=n_atoms)
    graph.ndata['x_1'] = torch.arange(
        n_atoms, dtype=torch.float32
    )[:, None].expand(-1, 3).clone()
    graph.ndata['s_1'] = one_hot(
        torch.tensor(state_classes), num_classes=max(state_classes) + 1
    ).float()
    all_edges = torch.tensor(edge_classes + edge_classes)
    graph.edata['e_1'] = one_hot(all_edges, num_classes=4).float()
    graph.edata['ue_mask'] = get_upper_edge_mask(graph)
    return graph


class FakeAtomDatasetTests(unittest.TestCase):
    def test_validation_augmentation_is_deterministic_and_fake_edges_are_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            _write_tiny_dataset(path)
            dataset = MoleculeDataset(
                'val',
                dataset_config={
                    'dataset_name': 'qme14s',
                    'processed_data_dir': str(path),
                    'conditioning': {'enabled': False},
                    'use_joint_atom_state': True,
                    'n_atom_states': 3,
                    'n_bond_types': 4,
                    'require_chemistry_auxiliary_labels': True,
                    'fake_atom_fraction': 1.0,
                    'fake_atom_std': 1.0,
                    'fake_atom_validation_seed': 0,
                },
                prior_config=_prior_config(),
            )
            first = dataset[0]
            second = dataset[0]

        self.assertEqual(first.num_nodes(), 5)
        self.assertTrue(
            torch.equal(first.ndata['x_1_true'], second.ndata['x_1_true'])
        )
        self.assertTrue(
            torch.equal(first.ndata['s_1_true'], second.ndata['s_1_true'])
        )
        self.assertTrue(torch.allclose(
            first.ndata['x_1_true'].mean(0), torch.zeros(3), atol=1.0e-6
        ))
        self.assertEqual(first.ndata['is_real_1_true'].tolist(), [True] * 3 + [False] * 2)

        state = first.ndata['s_1_true'].argmax(-1)
        src, dst = first.edges()
        fake_incident = (state[src] == 2) | (state[dst] == 2)
        self.assertTrue(
            torch.equal(
                first.edata['e_1_true'][fake_incident].argmax(-1),
                torch.zeros(int(fake_incident.sum()), dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.equal(
                first.edata['e_connectivity_weight'][fake_incident],
                torch.zeros(int(fake_incident.sum())),
            )
        )
        self.assertFalse(first.edata['e_aromatic_true'][fake_incident].any())

    def test_fake_augmentation_requires_joint_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            _write_tiny_dataset(path)
            with self.assertRaisesRegex(ValueError, 'joint atom-state'):
                MoleculeDataset(
                    'val',
                    dataset_config={
                        'dataset_name': 'qme14s',
                        'processed_data_dir': str(path),
                        'conditioning': {'enabled': False},
                        'use_joint_atom_state': False,
                        'n_atom_states': 0,
                        'n_bond_types': 4,
                        'fake_atom_fraction': 0.3,
                    },
                    prior_config=_prior_config(),
                )


class FakeAtomCodecTests(unittest.TestCase):
    def test_metadata_appends_one_supported_zero_valence_fake_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            metadata_path = path / 'joint_atom_state_metadata.json'
            metadata_path.write_text(json.dumps({
                'states': [
                    {
                        'symbol': 'C', 'formal_charge': 0, 'hydrogens': 3,
                        'train_heavy_valence_counts': {'1': 3},
                    },
                    {
                        'symbol': 'O', 'formal_charge': 0, 'hydrogens': 1,
                        'train_heavy_valence_counts': {'1': 2},
                    },
                ],
                'train_supported': [True, True],
            }))
            model = FlowMol.__new__(FlowMol)
            nn.Module.__init__(model)
            model.use_joint_atom_state = True
            model.use_fake_atoms = True
            model.n_atom_states = 3
            model.atom_type_map = ['C', 'O']
            model.joint_atom_state_metadata_file = str(metadata_path)
            model.marginal_dists_file = str(path / 'train_data_marginal_dists.pt')
            model.fake_atom_state_index = None
            model.configure_joint_atom_state()

        self.assertEqual(model.fake_atom_state_index, 2)
        self.assertEqual(model.atom_state_vocab[-1]['symbol'], '__FAKE__')
        self.assertTrue(model.atom_state_vocab[-1]['is_fake'])
        self.assertEqual(
            model.atom_state_vocab[-1]['train_heavy_valence_counts'], {'0': 1}
        )
        self.assertEqual(model.category_support_masks['s'], [True, True, True])

    def test_static_real_marginal_is_expanded_without_rewriting_db(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            marginal_path = path / 'train_data_marginal_dists.pt'
            torch.save(
                (
                    torch.tensor([1.0]),
                    torch.full((6,), 1.0 / 6.0),
                    torch.full((4,), 0.25),
                    torch.full((1, 6), 1.0 / 6.0),
                ),
                marginal_path,
            )
            torch.save(
                torch.tensor([0.75, 0.25]),
                path / 'train_data_joint_state_marginal.pt',
            )
            model = FlowMol.__new__(FlowMol)
            nn.Module.__init__(model)
            model.marginal_dists_file = str(marginal_path)
            model.use_joint_atom_state = True
            model.use_fake_atoms = True
            model.fake_atom_fraction = 0.3
            model.n_atom_states = 3
            model.prior_config = _prior_config()
            model.parameterization = 'ctmc'
            model.categorical_feats = ['s', 'e']
            model.configure_prior()

        self.assertEqual(model.p_s.shape, (3,))
        self.assertAlmostEqual(float(model.p_s.sum()), 1.0, places=6)
        self.assertGreater(float(model.p_s[-1]), 0.0)


class FakeAtomDecoderTests(unittest.TestCase):
    def test_decoder_removes_fake_nodes_edges_and_relabels_real_bond(self):
        # Upper pairs are (0,1), (0,2), (1,2). Only old 0--2 should remain.
        graph = _complete_graph([0, 2, 1], [1, 2, 1])
        molecule = SampledMolecule(
            graph,
            atom_type_map=['C', 'O'],
            atom_state_vocab=[
                {'symbol': 'C', 'formal_charge': 0, 'hydrogens': 2},
                {'symbol': 'O', 'formal_charge': 0, 'hydrogens': 0},
                {
                    'symbol': '__FAKE__', 'formal_charge': 0, 'hydrogens': 0,
                    'is_fake': True,
                },
            ],
        )
        self.assertEqual(molecule.num_slots, 3)
        self.assertEqual(molecule.num_atoms, 2)
        self.assertEqual(molecule.g.num_nodes(), 2)
        self.assertEqual(molecule.atom_types, ['C', 'O'])
        self.assertEqual(molecule.bond_src_idxs.tolist(), [0])
        self.assertEqual(molecule.bond_dst_idxs.tolist(), [1])
        self.assertEqual(molecule.bond_types.tolist(), [2])
        self.assertIsNotNone(molecule.rdkit_mol)
        self.assertEqual(molecule.rdkit_mol.GetNumAtoms(), 2)

    def test_trajectory_tolerates_a_changing_number_of_real_nodes(self):
        graph = _complete_graph([0, 2, 1], [0, 1, 0])
        state_frames = one_hot(
            torch.tensor([[0, 0, 1], [0, 2, 1]]), num_classes=3
        ).float()
        edge_frames = graph.edata['e_1'].unsqueeze(0).expand(2, -1, -1).clone()
        trajectory = {
            'x': torch.stack((graph.ndata['x_1'], graph.ndata['x_1'] + 0.1)),
            's': state_frames,
            'e': edge_frames,
        }
        molecule = SampledMolecule(
            graph,
            atom_type_map=['C', 'O'],
            atom_state_vocab=[
                {'symbol': 'C', 'formal_charge': 0, 'hydrogens': 2},
                {'symbol': 'O', 'formal_charge': 0, 'hydrogens': 0},
                {
                    'symbol': '__FAKE__', 'formal_charge': 0, 'hydrogens': 0,
                    'is_fake': True,
                },
            ],
            traj_frames=trajectory,
            build_ep_traj=False,
        )
        self.assertEqual(
            [frame.GetNumAtoms() for frame in molecule.traj_mols], [3, 2]
        )

    def test_all_fake_endpoint_is_not_a_molecule(self):
        graph = dgl.graph(([], []), num_nodes=1)
        graph.ndata['x_1'] = torch.zeros(1, 3)
        graph.ndata['s_1'] = torch.ones(1, 1)
        graph.edata['e_1'] = torch.empty(0, 4)
        graph.edata['ue_mask'] = torch.empty(0, dtype=torch.bool)
        molecule = SampledMolecule(
            graph,
            atom_type_map=['C'],
            atom_state_vocab=[{
                'symbol': '__FAKE__', 'formal_charge': 0, 'hydrogens': 0,
                'is_fake': True,
            }],
        )
        self.assertEqual(molecule.num_slots, 1)
        self.assertEqual(molecule.num_atoms, 0)
        self.assertIsNone(molecule.rdkit_mol)
        self.assertEqual(molecule.valencies.numel(), 0)


class FakeAtomChemistryTests(unittest.TestCase):
    def test_oracle_real_bond_plus_fake_node_has_near_zero_losses(self):
        graph = _complete_graph([0, 0, 1], [1, 0, 0])
        graph.ndata['s_1_true'] = graph.ndata['s_1'].clone()
        graph.ndata['is_real_1_true'] = torch.tensor([True, True, False])
        graph.ndata['s_t'] = one_hot(torch.full((3,), 2), 3).float()
        graph.edata['e_1_true'] = graph.edata['e_1'].clone()
        graph.edata['e_t'] = one_hot(torch.full((6,), 4), 5).float()
        graph.edata['e_connectivity_weight'] = torch.tensor(
            [1.0, 0.0, 0.0] * 2
        )
        graph.edata['e_aromatic_true'] = torch.zeros(6, dtype=torch.bool)

        valence = torch.zeros(2, 2)
        valence[0, 1] = 1.0
        valence[1, 0] = 1.0
        loss_module = JointChemistryAuxiliaryLoss(
            state_charges=torch.zeros(2),
            state_valence_probabilities=valence,
            state_support=torch.ones(2, dtype=torch.bool),
            state_is_fake=torch.tensor([False, True]),
        )
        node_logits = torch.full((3, 2), -20.0)
        node_logits.scatter_(1, torch.tensor([[0], [0], [1]]), 20.0)
        edge_logits = torch.full((3, 4), -20.0)
        edge_logits.scatter_(1, torch.tensor([[1], [0], [0]]), 20.0)
        losses = loss_module(
            graph, node_logits, edge_logits, get_upper_edge_mask(graph)
        )
        for name, value in losses.items():
            with self.subTest(loss=name):
                self.assertTrue(torch.isfinite(value))
                self.assertLess(float(value), 1.0e-6)


class FakeAtomSamplingTests(unittest.TestCase):
    def test_slot_capacity_matches_inclusive_integer_uniform_contract(self):
        model = FlowMol.__new__(FlowMol)
        nn.Module.__init__(model)
        model.use_fake_atoms = True
        model.fake_atom_fraction = 0.3
        model.fake_atom_capacity_mode = 'stochastic'
        counts = torch.full((1024,), 10)
        generator = torch.Generator().manual_seed(11)
        stochastic = model.add_fake_atom_capacity(counts, generator=generator)
        self.assertEqual(int(stochastic.min()), 10)
        self.assertEqual(int(stochastic.max()), 13)
        self.assertTrue(torch.equal(
            model.add_fake_atom_capacity(counts, mode='maximum'),
            torch.full_like(counts, 13),
        ))
        self.assertTrue(torch.equal(
            model.add_fake_atom_capacity(counts, mode='none'), counts
        ))

        fixed = model.resolve_atom_counts(
            2, 'cpu', number_of_atoms=[8, 12]
        )
        self.assertTrue(torch.equal(fixed, torch.tensor([8, 12])))
        self.assertEqual(model.last_atom_count_diagnostics, {'mode': 'fixed'})

    def test_schema_rejects_fake_atoms_without_joint_ctmc(self):
        config = {
            'dataset': {
                'use_joint_atom_state': False,
                'n_atom_states': 0,
                'n_bond_types': 4,
                'use_hydrogen_counts': False,
            },
            'mol_fm': {
                'n_atom_states': 0,
                'n_bond_types': 4,
                'n_hydrogen_counts': 0,
                'parameterization': 'ctmc',
                'fake_atom_fraction': 0.3,
            },
            'vector_field': {'self_conditioning': False},
        }
        with self.assertRaisesRegex(ValueError, 'joint atom-state'):
            validate_chemistry_schema(config)


if __name__ == '__main__':
    unittest.main()
