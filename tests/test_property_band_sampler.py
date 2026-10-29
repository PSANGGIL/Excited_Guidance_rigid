import unittest,torch
from polyring_diffusion.real_data import PropertyBandSampler
class Dummy:
 def __len__(self):return 10
 shards=[{'raw_properties':torch.tensor([.1,.2,.3,.4,1.,1.2,1.4,1.6,2.1,3.])}]
class PropertyBandTests(unittest.TestCase):
 def test_band_ratio_and_reproducibility(self):
  s=PropertyBandSampler(Dummy(),1.,2.,.6,7);a=list(s);self.assertEqual(a,list(s));self.assertEqual(len(a),10);self.assertEqual(sum(4<=i<=7 for i in a),6);s.set_epoch(1);self.assertNotEqual(a,list(s))
