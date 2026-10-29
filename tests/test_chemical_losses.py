import unittest,torch
from polyring_diffusion.chemical_losses import ChemicalConstraintLoss
class ChemicalLossTests(unittest.TestCase):
 def setUp(self):
  benz=[[6,0,1,4]]*6;pyr=[[7,0,1,4]]+[[6,0,1,4]]*5;self.loss=ChemicalConstraintLoss([benz,pyr],8)
 def test_finite_and_differentiable(self):
  c=torch.randn(2,3,3,4,requires_grad=True);p=torch.randn(2,3,3,8,requires_grad=True);types=torch.tensor([[0,1,0],[1,0,0]]);mask=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool)
  out=self.loss(c,p,types,mask);total=out['valence']+out['kekule']+out['port_validity'];total.backward()
  self.assertTrue(torch.isfinite(total));self.assertIsNotNone(c.grad);self.assertIsNotNone(p.grad)
 def test_invalid_ports_are_penalized(self):
  c=torch.zeros(1,2,2,4);p=torch.full((1,2,2,8),-10.);p[...,7]=10.;types=torch.zeros(1,2,dtype=torch.long);mask=torch.ones(1,2,dtype=torch.bool)
  self.assertGreater(float(self.loss(c,p,types,mask)['port_validity']),.9)
