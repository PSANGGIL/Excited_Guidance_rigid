"""Absorbing categorical and VP Cartesian diffusion shared by all hierarchy stages."""
import torch
from torch import nn
class HierarchicalDiffusion(nn.Module):
 def __init__(self,timesteps=500,beta_start=1e-4,beta_end=.02):
  super().__init__();beta=torch.linspace(beta_start,beta_end,timesteps);alpha=1-beta;abar=alpha.cumprod(0);self.timesteps=timesteps
  for k,v in {'beta':beta,'alpha':alpha,'abar':abar,'abar_prev':torch.cat((torch.ones(1),abar[:-1]))}.items():self.register_buffer(k,v)
 def _x(self,v,t,nd):return v[t].reshape((len(t),)+(1,)*(nd-1))
 def q_tokens(self,x,t,mask_token,valid):
  keep=torch.rand_like(x.float())<self._x(self.abar,t,x.ndim);return torch.where(valid&~keep,torch.full_like(x,mask_token),x)
 def q_coords(self,x,t,mask):
  w=mask[...,None].to(x.dtype);center=(x*w).sum(1,keepdim=True)/w.sum(1,keepdim=True).clamp_min(1);clean=(x-center)*w;eps=torch.randn_like(x)*w;eps=(eps-(eps*w).sum(1,keepdim=True)/w.sum(1,keepdim=True).clamp_min(1))*w;a=self._x(self.abar,t,x.ndim);return a.sqrt()*clean+(1-a).sqrt()*eps,eps
 @torch.no_grad()
 def p_tokens(self,x,logits,t,mask_token,valid):
  sample=torch.distributions.Categorical(logits=logits).sample();a=self._x(self.abar,t,x.ndim);prev=self._x(self.abar_prev,t,x.ndim);reveal=(prev-a)/(1-a).clamp_min(1e-8);choose=(torch.rand_like(x.float())<reveal)&x.eq(mask_token)&valid;return torch.where(choose,sample,x)
