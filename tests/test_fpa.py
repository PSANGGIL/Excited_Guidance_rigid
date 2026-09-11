import unittest
import torch

from polyring_diffusion.fpa_data import _components, collate_fpa
from polyring_diffusion.fpa_model import FPAEntityDenoiser


class FPATests(unittest.TestCase):
    def record(self, cells=4):
        atoms = 8
        fusion = torch.zeros(cells, cells, dtype=torch.long)
        # A-B-C with D branching from B: the representation is not a chain.
        for a, b in ((0, 1), (1, 2), (1, 3)):
            if a < cells and b < cells: fusion[a, b] = fusion[b, a] = 1
        ports = torch.zeros(cells, cells, 2, dtype=torch.long)
        membership = torch.zeros(cells, atoms, dtype=torch.bool)
        for c in range(cells): membership[c, c:c + 3] = True
        return {'atom_types': torch.arange(atoms) % 4, 'atom_charges': torch.zeros(atoms, dtype=torch.long),
            'atom_positions': torch.randn(atoms, 3), 'atom_bond_types': torch.zeros(atoms, atoms, dtype=torch.long),
            'cell_types': torch.arange(cells) % 3, 'cell_sizes': torch.full((cells,), 6),
            'cell_centers': torch.randn(cells, 3), 'cell_normals': torch.tensor([[0., 0., 1.]]).repeat(cells, 1),
            'cell_tangents': torch.tensor([[1., 0., 0.]]).repeat(cells, 1), 'cell_radii': torch.ones(cells),
            'cell_membership': membership, 'fusion_types': fusion, 'fusion_ports': ports,
            'property': torch.randn(5), 'source_row': torch.tensor(0)}

    def test_branching_fpa_component(self):
        self.assertEqual(len(_components([(0, 1), (1, 2), (1, 3)], {0, 1, 2, 3})), 1)

    def test_new_model_forward(self):
        batch = collate_fpa([self.record(4), self.record(2)])
        model = FPAEntityDenoiser(4, 3, [.5, .6, .7, .8], hidden_dim=24, num_layers=2,
            time_dim=8, condition_dim=5, max_cell_ports=8, max_atoms=12, max_cells=6)
        out = model(batch['atom_types'], model.charge_index(batch['atom_charges']), batch['atom_positions'],
            batch['atom_bond_types'], batch['atom_mask'], batch['cell_types'], batch['cell_sizes'],
            batch['cell_centers'], batch['cell_normals'], batch['cell_tangents'], batch['cell_radii'],
            batch['fusion_types'], batch['cell_membership'].long(), batch['cell_mask'], torch.zeros(2, dtype=torch.long), batch['property'])
        self.assertEqual(out['fusion_logits'].shape, (2, 4, 4, 2))
        self.assertEqual(out['membership_logits'].shape, (2, 4, 8, 2))
        self.assertTrue(torch.isfinite(out['atom_coordinate_noise']).all())
        (out['fusion_logits'].sum() + out['membership_logits'].sum()).backward()
        self.assertIsNotNone(model.fusion_head[-1].weight.grad)


if __name__ == '__main__': unittest.main()
