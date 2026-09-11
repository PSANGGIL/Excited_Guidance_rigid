import unittest
import torch
from polyring_diffusion.fpa_model_v2 import FPAEntityDenoiserV2
from polyring_diffusion.so3_diffusion import MolecularDiffusion
from sample_fpa_v2 import cfg_predict

class RemainingFixTests(unittest.TestCase):
 def test_cfg_zero_is_training_null_path(self):
  b,a,c=2,4,1;m=FPAEntityDenoiserV2(3,2,[.5,.6,.7],hidden_dim=16,num_layers=1,time_dim=8,condition_dim=4,max_cell_ports=6,max_atoms=5,max_cells=2).eval();condition=torch.randn(b,4);args=(torch.zeros(b,a,dtype=torch.long),torch.zeros(b,a,dtype=torch.long),torch.randn(b,a,3),torch.zeros(b,a,a,dtype=torch.long),torch.ones(b,a,dtype=torch.bool),torch.zeros(b,c,dtype=torch.long),torch.full((b,c),6),torch.randn(b,c,3),torch.eye(3).reshape(1,1,3,3).repeat(b,c,1,1),torch.zeros(b,c),torch.zeros(b,c,c,dtype=torch.long),torch.zeros(b,c,a,dtype=torch.long),torch.ones(b,c,dtype=torch.bool),torch.zeros(b,dtype=torch.long));guided=cfg_predict(m,args,condition,0.);null=m(*args,condition,torch.ones(b,dtype=torch.bool));self.assertTrue(torch.equal(guided['atom_type_logits'],null['atom_type_logits']))
 def test_forward_symmetric_noise_is_seen_by_model(self):
  d=MolecularDiffusion(timesteps=8,so3_grid=64,so3_terms=32);clean=torch.randint(0,5,(4,6,6));clean=torch.triu(clean,1);clean=clean+clean.transpose(1,2);mask=torch.ones_like(clean,dtype=torch.bool);q=d.q_symmetric_tokens(clean,torch.full((4,),4),5,mask);self.assertTrue(torch.equal(q,q.transpose(1,2)))
if __name__=='__main__':unittest.main()
