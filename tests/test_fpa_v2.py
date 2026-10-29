import unittest
import torch
from polyring_diffusion.fpa_diffusion_v2 import FPADiffusion,masked_center,so3_exp
from polyring_diffusion.fpa_model_v2 import FPAEntityDenoiserV2

class FPAV2Tests(unittest.TestCase):
 def inputs(self):
  torch.manual_seed(4);b,a,c=2,5,3;m=FPAEntityDenoiserV2(4,5,[.5,.6,.7,.8],hidden_dim=24,num_layers=2,time_dim=8,condition_dim=4,max_cell_ports=8,max_atoms=8,max_cells=4);am=torch.ones(b,a,dtype=torch.bool);cm=torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool);at=torch.randint(0,4,(b,a));q=torch.zeros(b,a,dtype=torch.long);x=masked_center(torch.randn(b,a,3),am);bond=torch.zeros(b,a,a,dtype=torch.long);ct=torch.randint(0,5,(b,c));size=torch.full((b,c),6);p=torch.randn(b,c,3);frame=so3_exp(torch.randn(b,c,3));radius=torch.ones(b,c);fusion=torch.zeros(b,c,c,dtype=torch.long);member=torch.randint(0,2,(b,c,a));t=torch.tensor([2,7]);cond=torch.randn(b,4);return m,[at,q,x,bond,am,ct,size,p,frame,radius,fusion,member,cm,t,cond]
 def test_zero_center_and_so3(self):
  d=FPADiffusion(timesteps=10);mask=torch.tensor([[1,1,1,0]],dtype=torch.bool);x=torch.randn(1,4,3);noisy,eps=d.q_coordinates(masked_center(x,mask),torch.tensor([5]),mask);self.assertTrue(torch.allclose((eps*mask[...,None]).sum(1),torch.zeros(1,3),atol=1e-6));r,w=d.q_frames(torch.eye(3).reshape(1,1,3,3),torch.tensor([5]),torch.ones(1,1,dtype=torch.bool));self.assertTrue(torch.allclose(r.transpose(-1,-2)@r,torch.eye(3).reshape(1,1,3,3),atol=1e-5));self.assertTrue(torch.all(torch.linalg.det(r)>0))
 def test_equivariance_symmetry_and_com(self):
  m,args=self.inputs();m.eval();o=m(*args);q,_=torch.linalg.qr(torch.randn(3,3));q[:,0]*=torch.linalg.det(q);rot=list(args);rot[2]=args[2]@q;rot[7]=args[7]@q;rot[8]=args[8]@q;o2=m(*rot);self.assertLess(float((o2['atom_eps']-o['atom_eps']@q).abs().max()),2e-4);self.assertTrue(torch.allclose(o['bond_logits'],o['bond_logits'].transpose(1,2),atol=1e-6));self.assertTrue(torch.allclose(o['fusion_logits'],o['fusion_logits'].transpose(1,2),atol=1e-6));self.assertTrue(torch.allclose(o['atom_eps'].sum(1),torch.zeros_like(o['atom_eps'].sum(1)),atol=1e-5))
 def test_absorbing_reverse_reveals_at_zero(self):
  d=FPADiffusion(timesteps=10);x=torch.full((64,),3);valid=torch.ones_like(x,dtype=torch.bool);logits=torch.zeros(64,3);logits[:,1]=20;out=d.p_tokens(x,logits,torch.zeros(64,dtype=torch.long),3,valid);self.assertTrue(torch.equal(out,torch.ones_like(out)))
if __name__=='__main__':unittest.main()
