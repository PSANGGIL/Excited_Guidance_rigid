import unittest
import torch
from polyring_diffusion.model import RingDenoiser
from sample_rigid import guided

class ConditionGuidanceTests(unittest.TestCase):
    def _inputs(self,batch=2,nodes=3):
        ring=torch.zeros((batch,nodes),dtype=torch.long)
        centers=torch.randn(batch,nodes,3)
        normals=torch.nn.functional.normalize(torch.randn(batch,nodes,3),dim=-1)
        connection=torch.zeros((batch,nodes,nodes),dtype=torch.long)
        mask=torch.ones((batch,nodes),dtype=torch.bool)
        timesteps=torch.zeros(batch,dtype=torch.long)
        return ring,centers,normals,connection,mask,timesteps

    def test_cfg_unconditional_uses_training_null_path(self):
        torch.manual_seed(7)
        model=RingDenoiser(hidden_dim=16,num_layers=1,time_dim=8,condition_dim=1)
        model.eval(); args=self._inputs(); condition=torch.ones((2,1))
        actual=guided(model,args,condition,0.0)
        dropped=torch.ones(2,dtype=torch.bool)
        expected=model(*args,torch.zeros_like(condition),dropped)
        for key in ('center_noise','normal_noise','ring_logits','connection_logits','port_logits'):
            self.assertTrue(torch.allclose(actual[key],expected[key]))

    def test_property_head_shape_and_gradient(self):
        torch.manual_seed(8)
        model=RingDenoiser(hidden_dim=16,num_layers=1,time_dim=8,condition_dim=1,property_head=True)
        output=model(*self._inputs(),torch.tensor([[0.3],[0.7]]))
        self.assertEqual(tuple(output['property_pred'].shape),(2,1))
        output['property_pred'].sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.property_head.parameters()))

    def test_condition_contribution_is_reported(self):
        model=RingDenoiser(hidden_dim=16,num_layers=1,time_dim=8,condition_dim=1)
        tracker={};guided(model,self._inputs(),torch.ones((2,1)),2.0,tracker)
        self.assertIn('ring_logits',tracker)
        self.assertEqual(tracker['ring_logits']['calls'],1)

if __name__=='__main__':unittest.main()
