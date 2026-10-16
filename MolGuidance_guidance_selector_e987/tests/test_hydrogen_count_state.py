import dgl
import torch
from rdkit import Chem
from torch.nn.functional import one_hot

from molguidance.analysis.molecule_builder import build_molecule
from molguidance.models.ctmc_vector_field import CTMCVectorField
from molguidance.models.interpolant_scheduler import InterpolantScheduler


def _scheduler():
    features = ['x', 'a', 'c', 'h', 'e']
    return InterpolantScheduler(
        canonical_feat_order=features,
        schedule_type={feature: 'linear' for feature in features},
    )


def test_hydrogen_count_uses_its_own_ctmc_state():
    vector_field = CTMCVectorField(
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
    graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
    graph.ndata['x_0'] = torch.zeros(2, 3)
    graph.ndata['x_1_true'] = torch.ones(2, 3)
    graph.ndata['a_1_true'] = one_hot(torch.tensor([0, 1]), 3).float()
    graph.ndata['c_1_true'] = one_hot(torch.tensor([0, 1]), 2).float()
    graph.ndata['h_1_true'] = one_hot(torch.tensor([2, 0]), 4).float()
    graph.edata['e_1_true'] = one_hot(torch.tensor([1, 1]), 5).float()

    vector_field.sample_conditional_path(
        graph,
        t=torch.tensor([0.5]),
        node_batch_idx=torch.zeros(2, dtype=torch.long),
        edge_batch_idx=torch.zeros(2, dtype=torch.long),
        upper_edge_mask=torch.tensor([True, False]),
    )

    assert graph.ndata['h_t'].shape == (2, 5)
    assert vector_field.mask_idxs['h'] == 4
    assert vector_field.node_output_head[-1].out_features == 3 + 2 + 4
    assert vector_field.scalar_embedding_uncond[0].in_features == 3 + 2 + 4 + 1 + 3


def test_decoder_preserves_predicted_attached_hydrogens():
    molecule = build_molecule(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
        atom_types=['C', 'C'],
        atom_charges=torch.tensor([0, 0]),
        bond_src_idxs=torch.tensor([0]),
        bond_dst_idxs=torch.tensor([1]),
        bond_types=torch.tensor([2]),
        atom_hydrogens=torch.tensor([2, 2]),
    )
    Chem.SanitizeMol(molecule)

    assert [atom.GetNumExplicitHs() for atom in molecule.GetAtoms()] == [2, 2]
    assert all(atom.GetNoImplicit() for atom in molecule.GetAtoms())
    assert Chem.MolToSmiles(molecule) == 'C=C'


def test_decoder_rejects_residual_hydrogen_mask():
    molecule = build_molecule(
        positions=torch.zeros(1, 3),
        atom_types=['C'],
        atom_charges=torch.tensor([0]),
        bond_src_idxs=torch.empty(0, dtype=torch.long),
        bond_dst_idxs=torch.empty(0, dtype=torch.long),
        bond_types=torch.empty(0, dtype=torch.long),
        atom_hydrogens=torch.tensor([-1]),
    )
    assert molecule is None
