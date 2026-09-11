"""Isotropic heat-kernel diffusion on SO(3).

The density is represented with the truncated character expansion used by
IGSO(3) diffusion models.  Scores are with respect to Haar measure and live in
the right-trivialized tangent space, matching row-vector molecular frames.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor
from .fpa_diffusion_v2 import FPADiffusion, extract, so3_exp


def _haar_rotations(shape, device, dtype):
    q=torch.randn(shape+(4,),device=device,dtype=dtype);q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8);w,x,y,z=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(shape+(3,3))


class MolecularDiffusion(FPADiffusion):
    def __init__(self,*args,rotation_sigma_max=3.0,so3_grid=512,so3_terms=128,**kwargs):
        super().__init__(*args,rotation_sigma_max=rotation_sigma_max,**kwargs);self.so3_grid=int(so3_grid)
        sigma=torch.linspace(.02,rotation_sigma_max,self.timesteps);self.rot_sigma.copy_(sigma);angle=torch.linspace(1e-4,math.pi,self.so3_grid);cdf=[];score=[]
        ell=torch.arange(so3_terms,dtype=torch.float64)[:,None];a=angle.double()[None]
        character=torch.sin((ell+.5)*a)/torch.sin(a/2);laplace=ell*(ell+1)
        for s in sigma.double():
            if float(s)<.18:
                # The wrapped-normal local limit avoids cancellation in the
                # character series while converging to the same heat kernel.
                marginal=angle.double().square()*torch.exp(-angle.double().square()/(2*s*s));sc=-angle.double()/(s*s)
            else:
                f=((2*ell+1)*torch.exp(-laplace*s*s/2)*character).sum(0).clamp_min(1e-30);marginal=(1-torch.cos(angle.double()))*f
                logf=f.log();sc=torch.gradient(logf,spacing=(angle.double(),))[0]
            c=marginal.cumsum(0);cdf.append((c/c[-1]).float());score.append(sc.float())
        self.register_buffer('so3_angles',angle);self.register_buffer('so3_cdf',torch.stack(cdf));self.register_buffer('so3_score',torch.stack(score))

    def q_symmetric_tokens(self,clean:Tensor,t:Tensor,mask_token:int,valid:Tensor)->Tensor:
        upper=torch.triu(valid,1);noisy=self.q_tokens(clean,t,mask_token,upper);upper_values=torch.triu(noisy,1)
        return torch.where(valid,upper_values+upper_values.transpose(1,2),torch.zeros_like(clean))

    def q_frames(self,clean:Tensor,t:Tensor,mask:Tensor):
        b,n=mask.shape;axes=torch.randn((b,n,3),device=clean.device,dtype=clean.dtype);axes=axes/axes.norm(dim=-1,keepdim=True).clamp_min(1e-8);u=torch.rand((b,n),device=clean.device)
        angles=[];scores=[]
        for row in range(b):
            idx=torch.searchsorted(self.so3_cdf[t[row]],u[row]).clamp_max(self.so3_grid-1);angles.append(self.so3_angles[idx]);scores.append(self.so3_score[t[row],idx])
        angle=torch.stack(angles)*mask;radial_score=torch.stack(scores)*mask;rotvec=axes*angle.unsqueeze(-1);target=axes*radial_score.unsqueeze(-1)
        return clean@so3_exp(rotvec),target

    def denoise_frames(self,noisy:Tensor,score:Tensor,t:Tensor,mask:Tensor)->Tensor:
        variance=extract(self.rot_sigma.square(),t,score.ndim);out=noisy@so3_exp(variance*score)
        eye=torch.eye(3,device=out.device,dtype=out.dtype);return torch.where(mask[...,None,None],out,eye)

    @torch.no_grad()
    def initial_frames(self,batch:int,cells:int,device,dtype=torch.float32)->Tensor:
        return _haar_rotations((batch,cells),device,dtype)

    @torch.no_grad()
    def p_frames(self,noisy:Tensor,score:Tensor,t:Tensor,mask:Tensor):
        sigma2=self.rot_sigma.square();previous=torch.cat((torch.zeros(1,device=t.device),sigma2[:-1]));delta=extract(sigma2,t,score.ndim)-extract(previous,t,score.ndim);noise=torch.randn_like(score);noise=noise*(t>0).to(noise.dtype).reshape(len(t),1,1)
        out=noisy@so3_exp(delta*score+delta.sqrt()*noise);eye=torch.eye(3,device=out.device,dtype=out.dtype)
        return torch.where(mask[...,None,None],out,eye)
