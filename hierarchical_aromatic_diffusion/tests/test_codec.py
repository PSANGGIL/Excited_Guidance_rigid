import unittest
from rdkit import Chem
from hierarchical_aromatic_diffusion.codec import RingSystemCodec
class CodecTest(unittest.TestCase):
 def setUp(self): self.c=RingSystemCodec()
 def test_roundtrip_fused_spiro_and_substitution(self):
  for s in ('c1ccc2ccccc2c1','c1ccc2c(c1)ccc1ccccc12','C1CCC2(CC1)CCCC2','n1ccccc1C(=O)N','c1ccc2c(c1)ccc1cc3ccccc3cc12'):
   m=Chem.MolFromSmiles(s); self.assertTrue(self.c.roundtrip_equal(m),s); self.assertTrue(self.c.validate_junctions(self.c.encode(m))); self.assertTrue(self.c.validate_oracle_assembly(self.c.encode(m)))
 def test_permutation_invariant_identity(self):
  m=Chem.MolFromSmiles('Cc1ncccc1C#N'); perm=list(reversed(range(m.GetNumAtoms()))); q=Chem.RenumberAtoms(m,perm)
  self.assertEqual(self.c.encode(m).canonical_smiles,self.c.encode(q).canonical_smiles)
 def test_unknown_ring_is_lossless(self):
  m=Chem.MolFromSmiles('C1CCCCCCCCCC1'); spec=self.c.encode(m); self.assertTrue(spec.rings[0].unknown_size); self.assertTrue(self.c.roundtrip_equal(m))
if __name__=='__main__':unittest.main()
