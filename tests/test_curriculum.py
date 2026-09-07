import unittest,torch
from polyring_diffusion.real_data import ValidityCurriculumSampler
class Dummy:
 def __len__(self):return 10
 shards=[{'length':10,'ring_offsets':torch.arange(11),'connection_offsets':torch.zeros(11,dtype=torch.long),'connections':torch.empty((0,6),dtype=torch.long)}]
class CurriculumTests(unittest.TestCase):
 def test_expands_to_full_data(self):
  s=ValidityCurriculumSampler(Dummy(),fractions=(.4,1.),milestones=(0,2),seed=1);self.assertEqual(len(set(iter(s))),4);s.set_epoch(2);self.assertEqual(len(set(iter(s))),10)
