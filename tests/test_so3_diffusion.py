import unittest
import torch
from polyring_diffusion.so3_diffusion import MolecularDiffusion

class SO3DiffusionTests(unittest.TestCase):
 def setUp(self):self.d=MolecularDiffusion(timesteps=12,so3_grid=128,so3_terms=64)
 def test_symmetric_categorical_corruption(self):
  clean=torch.zeros(8,7,7,dtype=torch.long);valid=torch.ones_like(clean,dtype=torch.bool);q=self.d.q_symmetric_tokens(clean,torch.full((8,),6),5,valid);self.assertTrue(torch.equal(q,q.transpose(1,2)));self.assertEqual(int(torch.diagonal(q,dim1=1,dim2=2).count_nonzero()),0)
 def test_heat_kernel_tables_and_rotations(self):
  self.assertTrue(torch.isfinite(self.d.so3_cdf).all());self.assertTrue(torch.isfinite(self.d.so3_score).all());self.assertTrue((self.d.so3_cdf[:,1:]>=self.d.so3_cdf[:,:-1]).all());b,n=16,3;frame=torch.eye(3).reshape(1,1,3,3).repeat(b,n,1,1);mask=torch.ones(b,n,dtype=torch.bool);t=torch.randint(0,self.d.timesteps,(b,));noisy,score=self.d.q_frames(frame,t,mask);reverse=self.d.p_frames(noisy,score,t,mask);eye=torch.eye(3);self.assertTrue(torch.allclose(noisy.transpose(-1,-2)@noisy,eye,atol=2e-6));self.assertTrue(torch.allclose(reverse.transpose(-1,-2)@reverse,eye,atol=2e-6))
 def test_initial_prior_is_haar(self):
  rotations=self.d.initial_frames(12000,1,'cpu').squeeze(1);self.assertLess(float(rotations.mean(0).norm()),.05)
if __name__=='__main__':unittest.main()
