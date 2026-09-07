import unittest, torch
from rdkit import Chem
from prepare_rigid_dataset import extract, FUSED, SINGLE
class Tests(unittest.TestCase):
 def xyz(self,s):
  n=Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms(); return torch.tensor([[float(i),float(i%3),float((i*i)%5)] for i in range(n)])
 def test_benzene(self):
  r=extract('c1ccccc1',self.xyz('c1ccccc1')); self.assertEqual(len(r['signatures']),1); self.assertEqual(r['ring_atoms'].numel(),6)
 def test_connections(self):
  a=extract('c1ccc2ccccc2c1',self.xyz('c1ccc2ccccc2c1')); b=extract('c1ccccc1-c2ccccc2',self.xyz('c1ccccc1-c2ccccc2'))
  self.assertIn(FUSED,a['connections'][:,2].tolist()); self.assertIn(SINGLE,b['connections'][:,2].tolist())
if __name__=='__main__': unittest.main()
