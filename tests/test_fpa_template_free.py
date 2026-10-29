import math
import unittest
import torch

from polyring_diffusion.fpa_model_template_free import TemplateFreeFPAEntityDenoiser
from polyring_diffusion.fpa_projection_template_free import _complete_valence, _feasible_bond_kind, project_template_free_fpa
from sample_fpa_template_free import cfg_predict, decode_exact, reconstruct, scalar_s1_t1_condition
from train_fpa_template_free import aromatic_ring_objective, global_connectivity_loss


class TemplateFreeFPATests(unittest.TestCase):
    def model(self):
        return TemplateFreeFPAEntityDenoiser(3, [.5, .6, .7], hidden_dim=24,
            num_layers=2, time_dim=8, condition_dim=4, min_ring_size=3,
            max_ring_size=10, max_atoms=12, max_cells=4).eval()

    def test_model_has_no_template_identity_and_cfg_null_is_exact(self):
        model = self.model(); b, na, nc = 2, 6, 2
        am = torch.ones(b, na, dtype=torch.bool); cm = torch.ones(b, nc, dtype=torch.bool)
        args = (torch.zeros(b, na, dtype=torch.long), torch.zeros(b, na, dtype=torch.long),
            torch.ones(b, na, dtype=torch.long), torch.zeros(b, na, dtype=torch.long), torch.randn(b, na, 3),
            torch.zeros(b, na, na, dtype=torch.long), am,
            torch.full((b, nc), 3), torch.ones(b, nc, dtype=torch.long),
            torch.randn(b, nc, 3), torch.eye(3).reshape(1, 1, 3, 3).repeat(b, nc, 1, 1),
            torch.zeros(b, nc), torch.zeros(b, nc, nc, dtype=torch.long),
            torch.ones(b, nc, na, dtype=torch.long), cm, torch.zeros(b, dtype=torch.long))
        condition = torch.randn(b, 4)
        guided = cfg_predict(model, args, condition, 0.)
        null = model(*args, condition, torch.ones(b, dtype=torch.bool))
        self.assertTrue(all(torch.equal(guided[key], null[key]) for key in guided))
        self.assertFalse(hasattr(model, "cell_type"))

    def test_exact_decoder_preserves_rdkit_identity(self):
        from rdkit import Chem
        source = Chem.MolFromSmiles("Cn1c(=O)n(C)c2ncn(C)c2c1=O")
        source = Chem.AddHs(source)
        heavy = Chem.RemoveHs(source)
        atom_map = tuple(sorted({a.GetSymbol() for a in heavy.GetAtoms()}))
        atom_types = torch.tensor([atom_map.index(a.GetSymbol()) for a in heavy.GetAtoms()])
        bonds = torch.zeros(len(atom_types), len(atom_types), dtype=torch.long)
        for bond in heavy.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            kind = 4 if bond.GetIsAromatic() else int(round(bond.GetBondTypeAsDouble()))
            bonds[i, j] = bonds[j, i] = kind
        encoded = {"atom_types": atom_types,
            "atom_charges": torch.tensor([a.GetFormalCharge() for a in heavy.GetAtoms()]),
            "atom_hydrogens": torch.tensor([a.GetTotalNumHs() for a in heavy.GetAtoms()]),
            "atom_radicals": torch.tensor([a.GetNumRadicalElectrons() for a in heavy.GetAtoms()]),
            "atom_chirality": torch.zeros(len(atom_types), dtype=torch.long),
            "atom_bond_types": bonds, "atom_positions": torch.zeros(len(atom_types), 3)}
        decoded = decode_exact(encoded, atom_map)
        self.assertEqual(Chem.MolToSmiles(Chem.RemoveHs(source), isomericSmiles=False),
            Chem.MolToSmiles(Chem.RemoveHs(decoded), isomericSmiles=False))

    def test_single_heteroaromatic_ring_projection(self):
        n = 6; angle = torch.arange(n)*2*math.pi/n
        sample = {
            "atom_types": torch.tensor([[0, 0, 1, 0, 0, 0]]),
            "charges": torch.full((1, n), 2), "hydrogens": torch.ones(1, n, dtype=torch.long), "radicals": torch.zeros(1, n, dtype=torch.long),
            "positions": torch.stack((angle.cos(), angle.sin(), torch.ones(n)*.2), -1)[None],
            "bonds": torch.zeros(1, n, n, dtype=torch.long), "cell_sizes": torch.tensor([[6]]),
            "cell_aromatic": torch.ones(1, 1, dtype=torch.long), "cell_centers": torch.zeros(1, 1, 3),
            "cell_frames": torch.eye(3).reshape(1, 1, 3, 3), "cell_radii": torch.ones(1, 1),
            "cell_relations": torch.zeros(1, 1, 1, dtype=torch.long),
            "membership": torch.zeros(1, 1, n, dtype=torch.long),
            "atom_mask": torch.ones(1, n, dtype=torch.bool), "cell_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        member = torch.zeros(1, 1, n, 2); member[..., 1] = 9
        bond = torch.zeros(1, n, n, 5); bond[..., 0] = 9
        out = project_template_free_fpa(sample, {"membership_logits": member, "bond_logits": bond}, ["C", "N", "O"])
        self.assertEqual(int(out["membership"].sum()), 6)
        self.assertEqual(int((out["bonds"] == 4).sum()), 12)
        self.assertEqual(int(out["atom_types"][0, 2]), 1)
        self.assertTrue(bool(out["projection_valid"][0]))
        self.assertEqual(out["hydrogens"][0].tolist(), [1, 1, 0, 1, 1, 1])

    def test_aromatic_valence_cost_supports_fused_junction(self):
        probability = torch.tensor([0., 0., 0., 0., 1.])
        self.assertEqual(_feasible_bond_kind(probability, 4, 4, preferred=4), 4)
        self.assertEqual(_feasible_bond_kind(probability, 2, 2, preferred=4,
            left_aromatic=True, right_aromatic=True), 4)
        self.assertEqual(_feasible_bond_kind(probability, 1, 1, preferred=4), 0)

    def test_fused_aromatic_rdkit_reconstruction(self):
        from rdkit import Chem
        reference = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        n = reference.GetNumAtoms()
        bonds = torch.zeros(n, n, dtype=torch.long)
        for bond in reference.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bonds[i, j] = bonds[j, i] = 4
        hydrogens = torch.tensor([atom.GetTotalNumHs() for atom in reference.GetAtoms()])
        mol = reconstruct(torch.zeros(n, dtype=torch.long), torch.full((n,), 2), hydrogens,
            torch.zeros(n, dtype=torch.long), bonds, torch.zeros(n, 3), ["C"])
        heavy = Chem.RemoveHs(mol)
        self.assertEqual(heavy.GetNumAtoms(), 10)
        self.assertTrue(all(atom.GetIsAromatic() for atom in heavy.GetAtoms()))
        self.assertEqual(sum(atom.GetNumRadicalElectrons() for atom in heavy.GetAtoms()), 0)

    def test_rdkit_reconstruction_realizes_cip_label(self):
        from rdkit import Chem
        atom_map = ["C", "F", "Cl", "Br", "I"]
        atom_types = torch.tensor([0, 1, 2, 3, 4])
        bonds = torch.zeros(5, 5, dtype=torch.long)
        bonds[0, 1:] = 1; bonds[1:, 0] = 1
        chirality = torch.tensor([1, 0, 0, 0, 0])
        mol = reconstruct(atom_types, torch.full((5,), 2), torch.zeros(5, dtype=torch.long),
            torch.zeros(5, dtype=torch.long), bonds, torch.tensor([[0.,0.,0.],[1.,0.,0.],
            [0.,1.,0.],[0.,0.,1.],[-1.,-1.,-1.]]), atom_map, chirality=chirality)
        heavy = Chem.RemoveHs(mol); Chem.AssignStereochemistry(heavy, cleanIt=True, force=True)
        self.assertEqual(heavy.GetAtomWithIdx(0).GetProp("_CIPCode"), "R")

    def test_aromatic_objective_requires_the_whole_ring_and_has_gradients(self):
        n = 6
        membership = torch.ones(1, 1, n, dtype=torch.long)
        bonds = torch.zeros(1, n, n, dtype=torch.long)
        for i in range(n):
            bonds[0, i, (i+1) % n] = bonds[0, (i+1) % n, i] = 4
        atom_upper = torch.triu(torch.ones(1, n, n, dtype=torch.bool), 1)
        batch = {"cell_aromatic": torch.ones(1, 1, dtype=torch.long),
            "cell_membership": membership, "atom_bond_types": bonds}
        cell_logits = torch.tensor([[[-2., 2.]]], requires_grad=True)
        good = torch.full((1, n, n, 5), -4.); good[..., 0] = 4.
        for i in range(n):
            good[0, i, (i+1) % n, 0] = good[0, (i+1) % n, i, 0] = -4.
            good[0, i, (i+1) % n, 4] = good[0, (i+1) % n, i, 4] = 4.
        good.requires_grad_()
        aromatic_logits = torch.stack((-good[..., 4], good[..., 4]), -1)
        terms = aromatic_ring_objective({"cell_aromatic_logits": cell_logits, "bond_logits": good, "bond_aromatic_logits": aromatic_logits},
            batch, atom_upper, torch.ones(1, 1, dtype=torch.bool), 9.)
        self.assertEqual(float(terms[2]), 1.)
        self.assertEqual(float(terms[3]), 1.)
        (terms[0]+terms[1]).backward()
        self.assertIsNotNone(cell_logits.grad); self.assertIsNotNone(good.grad)
        broken = good.detach().clone(); broken[0, 0, 1] = torch.tensor([4., -4., -4., -4., -4.])
        broken_aromatic_logits = aromatic_logits.detach().clone(); broken_aromatic_logits[0, 0, 1] = torch.tensor([4., -4.])
        bad = aromatic_ring_objective({"cell_aromatic_logits": cell_logits.detach(), "bond_logits": broken, "bond_aromatic_logits": broken_aromatic_logits},
            batch, atom_upper, torch.ones(1, 1, dtype=torch.bool), 9.)
        self.assertGreater(float(bad[1]), float(terms[1]))

    def test_global_connectivity_prefers_a_spanning_graph(self):
        mask = torch.ones(1, 4, dtype=torch.bool)
        disconnected = torch.zeros(1, 4, 4)
        disconnected[0, 0, 1] = disconnected[0, 1, 0] = .9
        disconnected[0, 2, 3] = disconnected[0, 3, 2] = .9
        connected = disconnected.clone()
        connected[0, 1, 2] = connected[0, 2, 1] = .9
        self.assertLess(float(global_connectivity_loss(connected, mask)),
            float(global_connectivity_loss(disconnected, mask)))

    def test_fused_overlap_is_exact_and_edges_not_double_counted(self):
        n, c = 10, 2
        sample = {"atom_types": torch.zeros(1, n, dtype=torch.long), "charges": torch.full((1, n), 2),
            "hydrogens": torch.zeros(1, n, dtype=torch.long), "radicals": torch.zeros(1, n, dtype=torch.long), "positions": torch.randn(1, n, 3),
            "bonds": torch.zeros(1, n, n, dtype=torch.long), "cell_sizes": torch.full((1, c), 6),
            "cell_aromatic": torch.ones(1, c, dtype=torch.long), "cell_centers": torch.zeros(1, c, 3),
            "cell_frames": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, c, 1, 1),
            "cell_radii": torch.ones(1, c), "cell_relations": torch.tensor([[[0, 3], [3, 0]]]),
            "membership": torch.zeros(1, c, n, dtype=torch.long), "atom_mask": torch.ones(1, n, dtype=torch.bool),
            "cell_mask": torch.ones(1, c, dtype=torch.bool)}
        member = torch.zeros(1, c, n, 2)
        member[0, 0, :, 1] = torch.tensor([10, 9, 8, 7, 6, 5, 0, 0, 0, 0.])
        member[0, 1, :, 1] = torch.tensor([10, 9, 8, 0, 0, 0, 7, 6, 5, 4.])
        bond = torch.zeros(1, n, n, 5); bond[..., 0] = 9
        out = project_template_free_fpa(sample, {"membership_logits": member, "bond_logits": bond}, ["C"])
        shared = out["membership"][0, 0].bool() & out["membership"][0, 1].bool()
        self.assertEqual(int(shared.sum()), 2)
        self.assertEqual(out["membership"][0].sum(1).tolist(), [6, 6])
        ids = torch.where(shared)[0]
        self.assertEqual(int(out["bonds"][0, ids[0], ids[1]]), 4)

    def test_coordinate_heads_ignore_padding_under_rigid_transforms(self):
        torch.manual_seed(17)
        model = self.model(); b, na, nc = 1, 4, 2
        am = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool); cm = torch.tensor([[1, 0]], dtype=torch.bool)
        atom_type = torch.tensor([[0, 1, model.atom_pad_token, model.atom_pad_token]])
        zero_a = torch.zeros(b, na, dtype=torch.long); atom_pos = torch.randn(b, na, 3)*am[..., None]
        bond = torch.zeros(b, na, na, dtype=torch.long); size = torch.tensor([[0, model.size_pad_token]])
        aromatic = torch.zeros(b, nc, dtype=torch.long); cell_pos = torch.randn(b, nc, 3)*cm[..., None]
        frame = torch.eye(3).reshape(1, 1, 3, 3).repeat(b, nc, 1, 1)
        radius = torch.zeros(b, nc); relation = torch.zeros(b, nc, nc, dtype=torch.long)
        membership = torch.zeros(b, nc, na, dtype=torch.long); t = torch.ones(b, dtype=torch.long); condition = torch.randn(b, 4)
        args = [atom_type, zero_a, zero_a, zero_a, atom_pos, bond, am, size, aromatic, cell_pos,
            frame, radius, relation, membership, cm, t, condition]
        base = model(*args); shift = torch.tensor([2., -3., 1.]); shifted = list(args)
        shifted[4] = atom_pos+shift*am[..., None]; shifted[9] = cell_pos+shift*cm[..., None]
        moved = model(*shifted)
        self.assertLess(float((moved["atom_eps"]-base["atom_eps"]).abs().max()), 1e-5)
        self.assertLess(float((moved["cell_eps"]-base["cell_eps"]).abs().max()), 1e-5)
        rotation, _ = torch.linalg.qr(torch.randn(3, 3)); rotation[:, 0] *= torch.linalg.det(rotation)
        rotated = list(args); rotated[4] = atom_pos@rotation.T; rotated[9] = cell_pos@rotation.T
        rotated[10] = frame@rotation.T
        turned = model(*rotated)
        self.assertLess(float((turned["atom_eps"]-base["atom_eps"]@rotation.T).abs().max()), 1e-2)
        self.assertLess(float((turned["cell_eps"]-base["cell_eps"]@rotation.T).abs().max()), 1e-2)
        self.assertLess(float((turned["rotation_score"]-base["rotation_score"]@rotation.T).abs().max()), 3e-2)
        reflected = list(args); reflected[4] = atom_pos.clone(); reflected[4][..., 0] *= -1
        mirrored = model(*reflected)
        self.assertTrue(torch.isfinite(mirrored["chirality_logits"]).all())

    def test_scalar_condition_accepts_physical_targets(self):
        stats = {"mode":"s1_t1", "s1_energy_mean":3., "s1_energy_std":.5,
            "t1_energy_mean":2., "t1_energy_std":.25}
        value = scalar_s1_t1_condition(stats, 3.5, 2.5, 2, "cpu")
        self.assertEqual(tuple(value.shape), (2, 2))
        self.assertTrue(torch.allclose(value[0], torch.tensor([1., 2.]), atol=1e-5))

    def test_two_scalar_conditions_expand_to_ten_rbf_features_each(self):
        model = TemplateFreeFPAEntityDenoiser(3, [.5, .6, .7], hidden_dim=24,
            num_layers=1, time_dim=8, condition_dim=2, condition_rbf_kernels=10,
            condition_film=True,
            min_ring_size=3, max_ring_size=10, max_atoms=12, max_cells=4).eval()
        condition = torch.tensor([[0., 1.], [-1., 2.]])
        encoded = model.encode_condition(condition)
        self.assertEqual(tuple(encoded.shape), (2, 20))
        self.assertTrue(torch.all((encoded >= 0) & (encoded <= 1)))
        dropped = model.encode_condition(condition, torch.tensor([True, False]))
        self.assertTrue(torch.equal(dropped[0], torch.zeros(20)))
        self.assertTrue(torch.equal(dropped[1], encoded[1]))
        atom_count, cell_count = model.count_logits(condition)
        self.assertEqual(tuple(atom_count.shape), (2, 13))
        self.assertEqual(tuple(cell_count.shape), (2, 5))
        self.assertTrue(all(block.condition_film is not None for block in model.blocks))

    def test_radical_projection_requires_confidence_and_prefers_hydrogen_ties(self):
        hydrogen = torch.tensor([.5, .5, 0., 0., 0.])
        radical = torch.tensor([.5, .5, 0.])
        self.assertEqual(_complete_valence(1, hydrogen, radical, False, .8)[:2], (1, 0))
        self.assertEqual(_complete_valence(1, hydrogen, radical, False, .5)[:2], (1, 0))
        hydrogen = torch.tensor([.99, .01, 0., 0., 0.])
        radical = torch.tensor([.01, .99, 0.])
        self.assertEqual(_complete_valence(1, hydrogen, radical, False, .8)[:2], (0, 1))


if __name__ == "__main__": unittest.main()
