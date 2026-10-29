import itertools
import unittest

import numpy as np
from rdkit import Chem

from molguidance.analysis.joint_decode import joint_decode


def encode(smiles):
    aromatic = Chem.MolFromSmiles(smiles)
    mol = Chem.Mol(aromatic)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    n = mol.GetNumAtoms()
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    metadata, states = [], []
    for atom in mol.GetAtoms():
        valence = int(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds()))
        state = dict(symbol=atom.GetSymbol(), formal_charge=atom.GetFormalCharge(),
                     hydrogens=atom.GetTotalNumHs(), train_heavy_valence_counts={str(valence): 1})
        if state not in metadata:
            metadata.append(state)
        states.append(metadata.index(state))
    bonds, required = [], []
    for e, (i, j) in enumerate(pairs):
        bond = mol.GetBondBetweenAtoms(i, j)
        bonds.append(0 if bond is None else int(bond.GetBondTypeAsDouble()))
        ab = aromatic.GetBondBetweenAtoms(i, j)
        if ab is not None and ab.GetIsAromatic():
            required.append(e)
    return np.eye(len(metadata))[states], np.eye(4)[bonds], np.arange(n*3).reshape(n, 3) * .17, metadata, required


class JointDecodeTests(unittest.TestCase):
    def test_oracle_identity_heterocycles_fused_charge_and_small_rings(self):
        for smiles in ("c1ccccc1", "c1cc[nH]c1", "c1ccoc1", "c1ccsc1",
                       "c1ccncc1", "c1ccc2ccccc2c1", "c1ccc2c(c1)ccc1ccccc12",
                       "C1CC1", "C1CCC1", "C1CCCCCCCC1", "C1CC2(C1)CCC2",
                       "c1cc[nH+]cc1", "[O-][N+](=O)c1ccccc1", "CP(=O)(O)O", "CS(=O)(=O)C"):
            p, q, xyz, metadata, aromatic = encode(smiles)
            charge = sum(a.GetFormalCharge() for a in Chem.MolFromSmiles(smiles).GetAtoms())
            result = joint_decode(p, q, xyz, metadata, net_charge=charge, aromatic_edges=aromatic)
            self.assertEqual(result.status, "optimal", (smiles, result))
            self.assertEqual(Chem.MolToSmiles(result.molecule), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))
            np.testing.assert_array_equal(result.molecule.GetConformer().GetPositions(), xyz)

    def test_global_alternating_assignment_not_independent_ring_choices(self):
        for smiles in ("c1ccccc1", "c1cc[nH]c1", "c1ccc2ccccc2c1"):
            p, q, xyz, metadata, aromatic = encode(smiles)
            # Independent argmax chooses ALL single bonds: invalid. Matching
            # must couple shared atoms globally and allow odd heterocycles.
            q[aromatic] = [0., .51, .49, 0.]
            result = joint_decode(p, q, xyz, metadata, aromatic_edges=aromatic, time_limit=10.)
            self.assertIsNotNone(result.molecule, result)
            self.assertGreater(result.iterations, 0)
            self.assertEqual(Chem.MolToSmiles(result.molecule), Chem.MolToSmiles(Chem.MolFromSmiles(smiles)))

    def test_connectivity_and_fake_root(self):
        metadata = [
            dict(symbol="C", formal_charge=0, hydrogens=2, train_heavy_valence_counts={"2": 1}),
            dict(symbol="__FAKE__", formal_charge=0, hydrogens=0, is_fake=True),
        ]
        p = np.array([[0., 1.], [1., 0.], [1., 0.], [1., 0.]])
        q = np.tile([.8, .1, .09, .01], (6, 1))
        xyz = np.arange(12).reshape(4, 3)
        result = joint_decode(p, q, xyz, metadata, required_atoms=3)
        self.assertIsNotNone(result.molecule, result)
        self.assertEqual(Chem.MolToSmiles(result.molecule), "C1CC1")
        np.testing.assert_array_equal(result.molecule.GetConformer().GetPositions(), xyz[1:])

    def test_infeasible_zero_support_never_silently_repairs(self):
        p, q, xyz, metadata, aromatic = encode("c1ccccc1")
        q[aromatic] = [0., 1., 0., 0.]
        result = joint_decode(p, q, xyz, metadata, aromatic_edges=aromatic)
        self.assertEqual(result.status, "infeasible")
        self.assertIsNone(result.molecule)

    def test_numerical_and_shape_contracts(self):
        p, q, xyz, metadata, _ = encode("CC")
        for bad in (p * float("nan"), -p, p[:, :0]):
            with self.assertRaises(ValueError):
                joint_decode(bad, q, xyz, metadata)
        with self.assertRaises(ValueError):
            joint_decode(p, q, xyz, metadata, required_atoms=2.5)

    def test_milp_score_matches_exhaustive_small_problem(self):
        metadata = [
            dict(symbol="C", formal_charge=0, hydrogens=2, train_heavy_valence_counts={"2": 1}),
            dict(symbol="C", formal_charge=0, hydrogens=3, train_heavy_valence_counts={"1": 1}),
        ]
        p = np.array([[.7, .3], [.6, .4], [.8, .2]])
        q = np.tile([.6, .3, .08, .02], (3, 1))
        result = joint_decode(p, q, np.zeros((3, 3)), metadata)
        self.assertIsNotNone(result.molecule)
        best = -np.inf
        for states in itertools.product(range(2), repeat=3):
            for bonds in itertools.product(range(4), repeat=3):
                vals = [bonds[0]+bonds[1], bonds[0]+bonds[2], bonds[1]+bonds[2]]
                if vals != [2-s for s in states] or sum(v > 0 for v in bonds) < 2:
                    continue
                score = sum(np.log(p[i, s]) for i, s in enumerate(states))
                score += sum(np.log(q[e, b]) for e, b in enumerate(bonds))
                best = max(best, score)
        actual = sum(np.log(p[i, s]) for i, s in enumerate(result.states))
        actual += sum(np.log(q[e, b]) for e, b in enumerate(result.bonds))
        self.assertAlmostEqual(actual, best)


if __name__ == "__main__":
    unittest.main()
