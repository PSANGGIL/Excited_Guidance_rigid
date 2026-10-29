import unittest,torch
from hierarchical_aromatic_diffusion.assembly import assemble_scaffold
class AssemblyTest(unittest.TestCase):
 def test_fused_edge(self):
  p=torch.zeros(2,2,6,6); p[0,1,0,0]=p[1,0,0,0]=.9; p[0,1,1,1]=p[1,0,1,1]=.8
  a=assemble_scaffold([6,6],[1,1],p); self.assertTrue(a.valid); self.assertEqual(len(a.atom_slots),10); self.assertEqual(len(a.perimeter_edges),11)
 def test_spiro(self):
  p=torch.zeros(2,2,6,6);p[0,1,0,0]=p[1,0,0,0]=.9
  a=assemble_scaffold([6,5],[1,0],p);self.assertTrue(a.valid);self.assertEqual(len(a.atom_slots),10)
 def test_no_same_ring_collapse(self):
  p=torch.zeros(3,3,4,4);p[0,1,0,0]=p[1,0,0,0]=.9;p[1,2,0,0]=p[2,1,0,0]=.9;p[0,2,1,0]=p[2,0,0,1]=.9
  a=assemble_scaffold([4,4,4],[1,1,1],p); self.assertEqual(len(set(a.slot_to_atom[0,:4].tolist())),4)
if __name__=='__main__':unittest.main()
