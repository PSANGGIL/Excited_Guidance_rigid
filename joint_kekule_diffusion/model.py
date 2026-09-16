"""Joint atom/Kekule-bond/Cartesian DDPM for QCDGE.

Bond vocabulary is NONE/SINGLE/DOUBLE/TRIPLE. Aromatic targets marginalize over
valid non-adjacent Kekule placements, and every undirected bond shares one noise
realization. Padding never enters messages or losses.
"""
from __future__ import annotations
import math
from functools import lru_cache
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import nn

CHARGE_MASK,HYDROGEN_MASK,RADICAL_MASK,BOND_MASK=5,5,3,4

class ConditionEmbedding(nn.Module):
 def __init__(self,h=256,k=24):
  super().__init__();c=torch.linspace(-5,5,k);self.register_buffer('centers',c);self.register_buffer('iw',1/(c[1]-c[0]));self.net=nn.Sequential(nn.Linear(2*k+10,h),nn.SiLU(),nn.Linear(h,h))
 def forward(self,y):
  if y.ndim!=2 or y.shape[1]!=2:raise ValueError('normalized S1/T1 must be [B,2]')
  y=y.clamp(-12,12);rbf=torch.exp(-.5*((y[...,None]-self.centers)*self.iw).square());raw=torch.cat((y,y.square(),y.sin(),y.cos(),y.tanh()),-1);return self.net(torch.cat((rbf.flatten(1),raw),-1))

class TimeEmbedding(nn.Module):
 def __init__(self,h):
  super().__init__()
  if h%2:raise ValueError('hidden dimension must be even')
  self.h=h;self.net=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h))
 def forward(self,t):
  q=self.h//2;f=torch.exp(torch.arange(q,device=t.device)*(-math.log(10000)/max(q-1,1)));z=t.float()[:,None]*f;return self.net(torch.cat((z.sin(),z.cos()),-1))

class JointLayer(nn.Module):
 def __init__(self,h):
  super().__init__();self.msg=nn.Sequential(nn.Linear(3*h+1,2*h),nn.SiLU(),nn.Linear(2*h,h));self.edge=nn.Sequential(nn.Linear(3*h+1,2*h),nn.SiLU(),nn.Linear(2*h,h));self.n1=nn.LayerNorm(h);self.n2=nn.LayerNorm(h);self.e1=nn.LayerNorm(h);self.e2=nn.LayerNorm(h);self.nff=nn.Sequential(nn.Linear(h,4*h),nn.SiLU(),nn.Linear(4*h,h));self.eff=nn.Sequential(nn.Linear(h,2*h),nn.SiLU(),nn.Linear(2*h,h))
 def forward(self,x,e,p,mask):
  n=x.shape[1];pair=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=x.device)[None];d=(p[:,:,None]-p[:,None]).norm(dim=-1,keepdim=True);a=x[:,:,None].expand(-1,n,n,-1);b=x[:,None].expand(-1,n,n,-1);z=torch.cat((a,b,e,d),-1);x=self.n1(x+(self.msg(z)*pair[...,None]).sum(2)/pair.sum(2).clamp_min(1)[...,None]);x=self.n2(x+self.nff(x))*mask[...,None];e=self.e1(e+self.edge(z));e=self.e2(e+self.eff(e))*pair[...,None];return x,(e+e.transpose(1,2))/2

class JointKekuleDDPM(nn.Module):
 """Joint x0 categorical predictor and equivariant coordinate-epsilon head."""
 def __init__(self,hidden=256,max_atoms=32,atom_types=4,layers=8,condition_dropout=.1):
  super().__init__();self.max_atoms=max_atoms;self.atom_types=atom_types;self.condition_dropout=condition_dropout;self.cond=ConditionEmbedding(hidden);self.null=nn.Parameter(torch.zeros(hidden));self.time=TimeEmbedding(hidden);self.ae=nn.Embedding(atom_types+1,hidden);self.qe=nn.Embedding(6,hidden);self.he=nn.Embedding(6,hidden);self.re=nn.Embedding(4,hidden);self.be=nn.Embedding(5,hidden);self.layers=nn.ModuleList(JointLayer(hidden) for _ in range(layers));self.count=nn.Linear(hidden,max_atoms+1);self.atom=nn.Linear(hidden,atom_types);self.charge=nn.Linear(hidden,5);self.hydrogen=nn.Linear(hidden,5);self.radical=nn.Linear(hidden,3);self.bond=nn.Linear(hidden,4);self.aromatic=nn.Linear(hidden,1);self.coord=nn.Sequential(nn.Linear(hidden+1,hidden),nn.SiLU(),nn.Linear(hidden,1,bias=False))
 def context(self,y,t,unconditional=False):
  c=self.cond(y)
  if unconditional:c=self.null[None].expand_as(c)
  elif self.training and self.condition_dropout:c=torch.where((torch.rand(len(y),device=y.device)<self.condition_dropout)[:,None],self.null[None],c)
  return c+self.time(t)
 def forward(self,y,t,s,unconditional=False):
  m,p=s['atom_mask'],s['positions'];_,n=m.shape
  if n>self.max_atoms:raise ValueError('atom dimension exceeds max_atoms')
  c=self.context(y,t,unconditional);x=(c[:,None]+self.ae(s['atom_type'])+self.qe(s['charge'])+self.he(s['hydrogen'])+self.re(s['radical']))*m[...,None];pair=m[:,:,None]&m[:,None,:];e=(self.be(s['bond'])+self.be(s['bond']).transpose(1,2))/2*pair[...,None]
  for layer in self.layers:x,e=layer(x,e,p,m)
  pooled=(x*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1);delta=p[:,:,None]-p[:,None];dist=delta.norm(dim=-1,keepdim=True);off=pair&~torch.eye(n,dtype=torch.bool,device=p.device)[None];w=self.coord(torch.cat((e,dist),-1))*off[...,None];eps=(w*delta).sum(2)/off.sum(2).clamp_min(1)[...,None];eps=eps-(eps*m[...,None]).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)[...,None]
  return {'count_logits':self.count(pooled+c),'atom_logits':self.atom(x),'charge_logits':self.charge(x),'hydrogen_logits':self.hydrogen(x),'radical_logits':self.radical(x),'bond_logits':self.bond(e),'aromatic_edge_logits':self.aromatic(e).squeeze(-1),'coordinate_eps':eps*m[...,None]}

class JointDiffusion(nn.Module):
 def __init__(self,timesteps=500,beta_start=1e-4,beta_end=.02):
  super().__init__();beta=torch.linspace(beta_start,beta_end,timesteps);alpha=1-beta;self.timesteps=timesteps;self.register_buffer('beta',beta);self.register_buffer('alpha',alpha);self.register_buffer('abar',alpha.cumprod(0))
 def at(self,v,t,nd):return v[t].reshape((len(t),)+(1,)*(nd-1))
 def q_token(self,x,t,token,valid):
  keep=torch.rand_like(x,dtype=torch.float32)<self.at(self.abar,t,x.ndim);return torch.where(valid&~keep,torch.full_like(x,token),x)
 def q_bond_token(self,x,t,token,valid):
  """Absorb each undirected edge once, preserving graph symmetry."""
  n=x.shape[-1];upper=valid&torch.triu(torch.ones(n,n,dtype=torch.bool,device=x.device),1)[None]
  keep=torch.rand_like(x,dtype=torch.float32)<self.at(self.abar,t,x.ndim);masked=upper&~keep;masked=masked|masked.transpose(1,2)
  return torch.where(masked,torch.full_like(x,token),x)
 def q_coords(self,x,t,mask):
  w=mask[...,None].to(x.dtype);den=w.sum(1,keepdim=True).clamp_min(1);x=(x-(x*w).sum(1,keepdim=True)/den)*w;eps=torch.randn_like(x)*w;eps=(eps-(eps*w).sum(1,keepdim=True)/den)*w;a=self.at(self.abar,t,x.ndim);return a.sqrt()*x+(1-a).sqrt()*eps,eps

def _ce(logits,target,mask,weight=None):return F.cross_entropy(logits[mask],target[mask],weight=weight) if mask.any() else logits.sum()*0

@lru_cache(maxsize=None)
def _valid_cycle_patterns(length:int,doubles:int):
 rows=[]
 for bits in range(1<<length):
  row=tuple((bits>>i)&1 for i in range(length))
  if sum(row)==doubles and all(not(row[i] and row[(i+1)%length]) for i in range(length)):rows.append(row)
 return tuple(rows)

def aromatic_cycle_loss(bond_logits,aromatic_logits,cycle_edges,double_count,ring_mask,cycle_order,cycle_size,aromatic_edges,pair_mask):
 """Marginalize over every non-adjacent cyclic Kekule placement."""
 lp=F.log_softmax(bond_logits,-1);edges=aromatic_edges.bool();zero=bond_logits.sum()*0
 allowed=-torch.logsumexp(lp[...,1:3],-1)[edges].mean() if edges.any() else zero
 n=bond_logits.shape[1];valid_pairs=pair_mask&torch.triu(torch.ones(n,n,dtype=torch.bool,device=bond_logits.device),1)[None]
 if valid_pairs.any():
  labels=edges[valid_pairs].to(aromatic_logits.dtype);pos=labels.sum();neg=labels.numel()-pos;pos_weight=(neg/pos.clamp_min(1)).clamp(1,50)
  aux=F.binary_cross_entropy_with_logits(aromatic_logits[valid_pairs],labels,pos_weight=pos_weight)
 else:aux=zero
 losses=[]
 for length in range(3,cycle_order.shape[-1]+1):
  size_rows=(cycle_size==length)&ring_mask
  if not size_rows.any():continue
  for doubles in torch.unique(double_count[size_rows].long()).tolist():
   select=size_rows&(double_count.long()==int(doubles));br=select.nonzero(as_tuple=False);patterns=_valid_cycle_patterns(length,int(doubles))
   if not patterns:continue
   order=cycle_order[br[:,0],br[:,1],:length];u=order;v=torch.roll(order,-1,1);edge_lp=lp[br[:,0,None],u,v];pat=torch.tensor(patterns,dtype=torch.long,device=bond_logits.device)+1
   classes=pat[None,:,:,None].expand(len(br),-1,-1,1);scores=edge_lp[:,None].expand(-1,len(patterns),-1,-1).gather(-1,classes).squeeze(-1).sum(-1);losses.append((-torch.logsumexp(scores,-1)/length).mean())
 assignment=torch.stack(losses).mean() if losses else zero
 return allowed+aux+assignment

def global_chemistry_losses(outputs,batch):
 """Differentiable whole-graph valence, connectivity, charge and radical checks."""
 m=batch['atom_mask'];n=m.shape[1];pair=m[:,:,None]&m[:,None,:]&~torch.eye(n,dtype=torch.bool,device=m.device)[None];bp=F.softmax(outputs['bond_logits'],-1);orders=bp.new_tensor([0.,1.,2.,3.]);expected_bond=(bp*orders).sum(-1)*pair
 hp=F.softmax(outputs['hydrogen_logits'],-1);expected_h=(hp*hp.new_tensor([0.,1.,2.,3.,4.])).sum(-1);target_valence=batch['atom_hydrogen'].float()+(batch['atom_bonds'].float()*pair).sum(-1);pred_valence=expected_h+expected_bond.sum(-1);valence=F.smooth_l1_loss(pred_valence[m],target_valence[m])
 cp=F.softmax(outputs['charge_logits'],-1);expected_charge=(cp*cp.new_tensor([-2.,-1.,0.,1.,2.])).sum(-1)*m;charge=F.smooth_l1_loss(expected_charge.sum(-1),(batch['atom_charge'].float()-2).mul(m).sum(-1))
 rp=F.softmax(outputs['radical_logits'],-1);expected_radical=(rp*rp.new_tensor([0.,1.,2.])).sum(-1)*m;radical=F.smooth_l1_loss(expected_radical.sum(-1),batch['atom_radical'].float().sum(-1))
 presence=(1-bp[...,0])*pair;conn=[]
 for count in torch.unique(m.sum(-1)).tolist():
  rows=m.sum(-1)==count;k=int(count)
  if k<=1:continue
  p=presence[rows,:k,:k];lap=torch.diag_embed(p.sum(-1))-p;sign,logdet=torch.linalg.slogdet(lap[:,:k-1,:k-1]+torch.eye(k-1,device=p.device,dtype=p.dtype)*1e-6);conn.append((F.softplus(-logdet/k)+(sign<=0).to(logdet.dtype)*10).mean())
 connectivity=torch.stack(conn).mean() if conn else outputs['bond_logits'].sum()*0
 return {'valence':.2*valence,'connectivity':.05*connectivity,'charge_balance':.05*charge,'radical_count':.05*radical}

def joint_training_loss(model,diff,b):
 """All modalities share t, masks, noisy graph and coordinate realization."""
 m=b['atom_mask'];B,n=m.shape;t=torch.randint(diff.timesteps,(B,),device=m.device);pair=m[:,:,None]&m[:,None,:];diag=torch.eye(n,dtype=torch.bool,device=m.device)[None];upper=pair&torch.triu(torch.ones(n,n,dtype=torch.bool,device=m.device),1)[None];pos,noise=diff.q_coords(b['atom_positions'],t,m)
 s={'atom_mask':m,'atom_type':diff.q_token(b['atom_type'],t,model.atom_types,m),'charge':diff.q_token(b['atom_charge'],t,CHARGE_MASK,m),'hydrogen':diff.q_token(b['atom_hydrogen'],t,HYDROGEN_MASK,m),'radical':diff.q_token(b['atom_radical'],t,RADICAL_MASK,m),'bond':diff.q_bond_token(b['atom_bonds'],t,BOND_MASK,pair&~diag),'positions':pos};o=model(b['condition'],t,s);arom=b['aromatic_edge_mask'];weights=o['bond_logits'].new_tensor([.15,1,1.5,2])
 terms={'count':F.cross_entropy(o['count_logits'],m.sum(1)),'atom':_ce(o['atom_logits'],b['atom_type'],m),'charge':_ce(o['charge_logits'],b['atom_charge'],m),'hydrogen':_ce(o['hydrogen_logits'],b['atom_hydrogen'],m),'radical':_ce(o['radical_logits'],b['atom_radical'],m),'bond':_ce(o['bond_logits'],b['atom_bonds'],upper&~arom,weights),'aromatic_cycle':aromatic_cycle_loss(o['bond_logits'],o['aromatic_edge_logits'],b['aromatic_cycle_edges'],b['aromatic_cycle_double_count'],b['ring_mask'],b['aromatic_cycle_order'],b['aromatic_cycle_size'],arom,upper),'coordinate':((o['coordinate_eps']-noise).square()*m[...,None]).sum()/m.sum().clamp_min(1)};terms.update(global_chemistry_losses(o,b));return sum(terms.values()),terms

@dataclass(frozen=True)
class DecodeResult:
 molecule:object|None;smiles:str|None;valid:bool;connected:bool;aromatic_rings:int;error:str|None=None

def decode_kekule_graph(atom_type:Sequence[int],charge:Sequence[int],hydrogen:Sequence[int],radical:Sequence[int],bonds, symbols=('C','N','O','F')):
 """Strict decode: no bond repair, singleton escape, or aromatic thresholding."""
 try:
  from rdkit import Chem
  rw=Chem.RWMol()
  for a,q,h,r in zip(atom_type,charge,hydrogen,radical):
   x=Chem.Atom(symbols[int(a)]);x.SetFormalCharge(int(q)-2);x.SetNumExplicitHs(int(h));x.SetNumRadicalElectrons(int(r));x.SetNoImplicit(True);rw.AddAtom(x)
  bt={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE}
  for i in range(len(atom_type)):
   for j in range(i+1,len(atom_type)):
    if int(bonds[i,j]):rw.AddBond(i,j,bt[int(bonds[i,j])])
  mol=rw.GetMol();Chem.SanitizeMol(mol);connected=len(Chem.GetMolFrags(mol))==1;rings=sum(all(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in ring) for ring in Chem.GetSymmSSSR(mol));return DecodeResult(mol,Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False),True,connected,int(rings))
 except Exception as e:return DecodeResult(None,None,False,False,0,f'{type(e).__name__}: {e}')

def cfg_combine(cond:Mapping[str,torch.Tensor],uncond:Mapping[str,torch.Tensor],scale:float):
 if not math.isfinite(scale) or scale<0:raise ValueError('CFG scale must be finite and non-negative')
 return {k:uncond[k]+scale*(cond[k]-uncond[k]) for k in cond}

__all__=['JointKekuleDDPM','JointDiffusion','joint_training_loss','aromatic_cycle_loss','global_chemistry_losses','decode_kekule_graph','DecodeResult','cfg_combine']
