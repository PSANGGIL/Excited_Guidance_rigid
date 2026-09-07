import unittest, torch
from polyring_diffusion.real_data import collate_rigid
class RealDataTests(unittest.TestCase):
 def test_collate(self):
  def item(n): return {'ring_types':torch.arange(n),'centers':torch.randn(n,3),'normals':torch.randn(n,3),'radii':torch.ones(n),'connection_types':torch.zeros(n,n,dtype=torch.long),'ports':torch.zeros(n,n,dtype=torch.long),'orientation':torch.zeros(n,n,dtype=torch.long),'property':torch.tensor([1.]),'raw_property':torch.tensor([2.]),'source_row':torch.tensor(3)}
  b=collate_rigid([item(2),item(4)]); self.assertEqual(tuple(b['ring_types'].shape),(2,4)); self.assertEqual(b['node_mask'].sum(),6)
