"""Joint atom/Kekule-bond/Cartesian DDPM for QCDGE.

Bond vocabulary is NONE/SINGLE/DOUBLE/TRIPLE. Aromatic targets marginalize over
valid non-adjacent Kekule placements, and every undirected bond shares one noise
realization. Padding never enters messages or losses.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch
import torch.nn.functional as F
from torch import nn

class ConditionEmbedding(nn.Module):
 def __init__(self,h=256,s1_rbf_bins=24,t1_rbf_bins=24,rbf_min=-5.,rbf_max=5.,rbf_width_scale=1.):
  super().__init__()
  if s1_rbf_bins<2 or t1_rbf_bins<2:raise ValueError('S1/T1 RBF bins must each be at least 2')
  if not math.isfinite(rbf_min) or not math.isfinite(rbf_max) or rbf_max<=rbf_min:raise ValueError('RBF range must be finite and increasing')
  if not math.isfinite(rbf_width_scale) or rbf_width_scale<=0:raise ValueError('RBF width scale must be positive and finite')
  s1=torch.linspace(rbf_min,rbf_max,s1_rbf_bins);t1=torch.linspace(rbf_min,rbf_max,t1_rbf_bins)
  self.register_buffer('s1_centers',s1);self.register_buffer('t1_centers',t1)
  self.register_buffer('s1_inv_width',torch.tensor(1./(float(s1[1]-s1[0])*rbf_width_scale)))
  self.register_buffer('t1_inv_width',torch.tensor(1./(float(t1[1]-t1[0])*rbf_width_scale)))
  self.rbf_min=float(rbf_min);self.rbf_max=float(rbf_max)
  self.net=nn.Sequential(nn.Linear(s1_rbf_bins+t1_rbf_bins,h),nn.SiLU(),nn.Linear(h,h))
 def forward(self,y):
  if y.ndim!=2 or y.shape[1]!=2:raise ValueError('normalized S1/T1 must be [B,2]')
  y=y.clamp(self.rbf_min,self.rbf_max)
  s1=torch.exp(-.5*((y[:,0,None]-self.s1_centers)*self.s1_inv_width).square())
  t1=torch.exp(-.5*((y[:,1,None]-self.t1_centers)*self.t1_inv_width).square())
  return self.net(torch.cat((s1,t1),-1))

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
  n=x.shape[1];pair=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=x.device)[None];d=(p[:,:,None]-p[:,None]).norm(dim=-1,keepdim=True).clamp_max(50.);a=x[:,:,None].expand(-1,n,n,-1);b=x[:,None].expand(-1,n,n,-1);z=torch.cat((a,b,e,d),-1);x=self.n1(x+(self.msg(z)*pair[...,None]).sum(2)/pair.sum(2).clamp_min(1)[...,None]);x=self.n2(x+self.nff(x))*mask[...,None];e=self.e1(e+self.edge(z));e=self.e2(e+self.eff(e))*pair[...,None];return x,(e+e.transpose(1,2))/2

class JointKekuleDDPM(nn.Module):
 """Joint x0 categorical predictor and equivariant coordinate-epsilon head."""
 def __init__(self,hidden=256,max_atoms=32,atom_types=4,layers=8,condition_dropout=.1,
              s1_rbf_bins=24,t1_rbf_bins=24,condition_rbf_min=-5.,condition_rbf_max=5.,condition_rbf_width_scale=1.):
  super().__init__();self.max_atoms=max_atoms;self.atom_types=atom_types;self.condition_dropout=condition_dropout;self.cond=ConditionEmbedding(hidden,s1_rbf_bins,t1_rbf_bins,condition_rbf_min,condition_rbf_max,condition_rbf_width_scale);self.null=nn.Parameter(torch.zeros(hidden));self.time=TimeEmbedding(hidden);self.ae=nn.Embedding(atom_types,hidden);self.qe=nn.Embedding(5,hidden);self.he=nn.Embedding(5,hidden);self.re=nn.Embedding(3,hidden);self.ee=nn.Embedding(2,hidden);self.be=nn.Embedding(3,hidden);self.layers=nn.ModuleList(JointLayer(hidden) for _ in range(layers));self.count=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,max_atoms));self.atom=nn.Linear(hidden,atom_types);self.charge=nn.Linear(hidden,5);self.hydrogen=nn.Linear(hidden,5);self.radical=nn.Linear(hidden,3);self.edge_exists=nn.Linear(hidden,2);self.bond_type=nn.Linear(hidden,3);self.aromatic=nn.Linear(hidden,1);self.coord=nn.Sequential(nn.Linear(hidden+1,hidden),nn.SiLU(),nn.Linear(hidden,1,bias=False))
 def condition_context(self,y,unconditional=False):
  c=self.cond(y)
  if unconditional:c=self.null[None].expand_as(c)
  elif self.training and self.condition_dropout:c=torch.where((torch.rand(len(y),device=y.device)<self.condition_dropout)[:,None],self.null[None],c)
  return c
 def context(self,y,t,unconditional=False):return self.condition_context(y,unconditional)+self.time(t)
 def forward(self,y,t,s,unconditional=False):
  m,p=s['atom_mask'],s['positions'];_,n=m.shape
  if n>self.max_atoms:raise ValueError('atom dimension exceeds max_atoms')
  base=self.condition_context(y,unconditional);c=base+self.time(t);x=(c[:,None]+self.ae(s['atom_type'])+self.qe(s['charge'])+self.he(s['hydrogen'])+self.re(s['radical']))*m[...,None];pair=m[:,:,None]&m[:,None,:];gate=s['edge_exists'].to(p.dtype)[...,None];e=self.ee(s['edge_exists'])+gate*self.be(s['bond_type']);e=(e+e.transpose(1,2))/2*pair[...,None]
  for layer in self.layers:x,e=layer(x,e,p,m)
  pooled=(x*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1);delta=p[:,:,None]-p[:,None];dist=delta.norm(dim=-1,keepdim=True).clamp_max(50.);off=pair&~torch.eye(n,dtype=torch.bool,device=p.device)[None];w=self.coord(torch.cat((e,dist),-1))*off[...,None];eps=(w*delta).sum(2)/off.sum(2).clamp_min(1)[...,None];eps=eps-(eps*m[...,None]).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)[...,None]
  edge_logits=self.edge_exists(e);type_logits=self.bond_type(e);elp=F.log_softmax(edge_logits,-1);tlp=F.log_softmax(type_logits,-1);bond_logits=torch.cat((elp[...,0:1],elp[...,1:2]+tlp),-1)
  return {'count_logits':self.count(base),'atom_logits':self.atom(x),'charge_logits':self.charge(x),'hydrogen_logits':self.hydrogen(x),'radical_logits':self.radical(x),'edge_logits':edge_logits,'bond_type_logits':type_logits,'bond_logits':bond_logits,'aromatic_edge_logits':self.aromatic(e).squeeze(-1),'coordinate_eps':eps*m[...,None]}

class JointDiffusion(nn.Module):
 def __init__(self,timesteps=500,beta_start=1e-4,beta_end=.05,marginals=None):
  super().__init__();beta=torch.linspace(beta_start,beta_end,timesteps);alpha=1-beta;self.timesteps=timesteps;self.register_buffer("beta",beta);self.register_buffer("alpha",alpha);self.register_buffer("abar",alpha.cumprod(0));marginals=marginals or {}
  defaults={"atom":torch.ones(1),"charge":torch.ones(5),"hydrogen":torch.ones(5),"radical":torch.ones(3),"edge":torch.ones(2),"bond_type":torch.ones(3)}
  for name,default in defaults.items():
   value=torch.as_tensor(marginals.get(name,default),dtype=torch.float32);value=(value+1e-8)/(value+1e-8).sum();self.register_buffer(name+"_marginal",value)
 def at(self,v,t,nd):return v[t].reshape((len(t),)+(1,)*(nd-1))
 def marginal(self,name,classes,device):
  value=getattr(self,name+"_marginal");return value.to(device) if value.numel()==classes else torch.full((classes,),1/classes,device=device)
 def q_token(self,x,t,classes,valid,symmetric=False,name="atom"):
  """Marginal-preserving categorical q(x_t|x_0)."""
  keep=torch.rand_like(x,dtype=torch.float32)<self.at(self.abar,t,x.ndim);prob=self.marginal(name,classes,x.device);random=torch.multinomial(prob,x.numel(),replacement=True).reshape_as(x)
  if symmetric:
   n=x.shape[-1];up=torch.triu(torch.ones(n,n,dtype=torch.bool,device=x.device),1)[None];random=torch.where(up,random,random.transpose(1,2));keep=torch.where(up,keep,keep.transpose(1,2))
  return torch.where(valid&~keep,random,x)
 def posterior_logits(self,x_t,x0_logits,t,classes,name="atom"):
  """Exact generalized D3PM posterior under a stationary empirical marginal."""
  p0=F.softmax(x0_logits.float(),-1);t=torch.as_tensor(t,device=x_t.device,dtype=torch.long);t=t.expand(x_t.shape[0]) if t.ndim==0 else t;ab_t=self.abar[t];ab_prev=torch.where(t==0,torch.ones_like(ab_t),self.abar[(t-1).clamp_min(0)]);alpha=ab_t/ab_prev
  shape=(len(t),)+(1,)*x_t.ndim;ab_t=ab_t.reshape(shape);ab_prev=ab_prev.reshape(shape);alpha=alpha.reshape(shape);one=F.one_hot(x_t.long(),classes).to(p0.dtype);m=self.marginal(name,classes,x_t.device);m_xt=m[x_t.long()][...,None]
  denom=(ab_t*one+(1-ab_t)*m_xt).clamp_min(1e-20);weighted=p0/denom;prior=ab_prev*weighted+(1-ab_prev)*m*weighted.sum(-1,keepdim=True);like=alpha*one+(1-alpha)*m_xt;posterior=(prior*like).clamp_min(1e-20);return posterior.log()-posterior.sum(-1,keepdim=True).log()
 def categorical_vlb(self,x0,x_t,x0_logits,t,classes,mask,name="atom"):
  true_logits=F.one_hot(x0.long(),classes).float().clamp_min(1e-20).log();qlog=self.posterior_logits(x_t,true_logits,t,classes,name);plog=self.posterior_logits(x_t,x0_logits,t,classes,name);q=qlog.exp();kl=(q*(qlog-plog)).sum(-1);nll=F.cross_entropy(x0_logits.reshape(-1,classes),x0.reshape(-1),reduction="none").reshape_as(x0);at0=t.reshape((len(t),)+(1,)*(x0.ndim-1))==0;loss=torch.where(at0,nll,kl);return loss[mask].mean() if mask.any() else x0_logits.sum()*0
 def q_coords(self,x,t,mask):
  w=mask[...,None].to(x.dtype);den=w.sum(1,keepdim=True).clamp_min(1);x=(x-(x*w).sum(1,keepdim=True)/den)*w;eps=torch.randn_like(x)*w;eps=(eps-(eps*w).sum(1,keepdim=True)/den)*w;a=self.at(self.abar,t,x.ndim);return a.sqrt()*x+(1-a).sqrt()*eps,eps

def _ce(logits,target,mask,weight=None):return F.cross_entropy(logits[mask],target[mask],weight=weight) if mask.any() else logits.sum()*0

def _cycle_log_partition(edge_lp,doubles):
 """Exact O(length*doubles) cyclic independent-set partition function."""
 rows,length,_=edge_lp.shape;neg=edge_lp.new_full((rows,doubles+1,2),-1e30);answers=[]
 for first in (0,1):
  if first>doubles:continue
  dp=neg.clone();dp[:,first,first]=edge_lp[:,0,first+1]
  for i in range(1,length):
   nxt=neg.clone();nxt[:,:,0]=torch.logaddexp(dp[:,:,0],dp[:,:,1])+edge_lp[:,i,1,None]
   if doubles:nxt[:,1:,1]=dp[:,:-1,0]+edge_lp[:,i,2,None]
   dp=nxt
  answers.append(dp[:,doubles,0] if first else torch.logsumexp(dp[:,doubles],-1))
 return torch.logsumexp(torch.stack(answers,-1),-1) if answers else edge_lp.new_full((rows,),-1e30)

def aromatic_cycle_loss(bond_logits,aromatic_logits,cycle_edges,double_count,ring_mask,cycle_order,cycle_size,kekule_edges,aromatic_edges=None,pair_mask=None):
 """Marginalize exactly over cyclic Kekule placements without exponential enumeration."""
 if pair_mask is None:pair_mask=aromatic_edges;aromatic_edges=kekule_edges
 lp=F.log_softmax(bond_logits.float(),-1);edges=aromatic_edges.bool();eligible=kekule_edges.bool();zero=bond_logits.sum()*0
 allowed=-torch.logsumexp(lp[...,1:3],-1)[eligible].mean() if eligible.any() else zero
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
   select=size_rows&(double_count.long()==int(doubles));br=select.nonzero(as_tuple=False)
   if doubles<0 or doubles>length//2:continue
   order=cycle_order[br[:,0],br[:,1],:length];u=order;v=torch.roll(order,-1,1);edge_lp=lp[br[:,0,None],u,v];partition=_cycle_log_partition(edge_lp,int(doubles));finite=torch.isfinite(partition)
   if finite.any():losses.append((-partition[finite]/length).mean())
 assignment=torch.stack(losses).mean() if losses else zero
 return allowed+aux+assignment

def aromatic_component_loss(outputs,batch):
 """Couple all fused aromatic cycles through shared-atom double-bond incidence."""
 edges=batch["aromatic_edge_mask"].bool()&(batch["atom_bonds"]>=1)&(batch["atom_bonds"]<=2);zero=outputs["bond_logits"].sum()*0
 if not edges.any():return zero
 bp=F.softmax(outputs["bond_logits"].float(),-1);pred_double=(bp[...,2]*edges).sum(-1);target_double=((batch["atom_bonds"]==2)&edges).sum(-1).float();atoms=edges.any(-1);incidence=F.smooth_l1_loss(pred_double[atoms],target_double[atoms]);pred_total=(bp[...,2]*edges).sum((1,2))/2;target_total=((batch["atom_bonds"]==2)&edges).sum((1,2)).float()/2;total=F.smooth_l1_loss(pred_total,target_total)
 return .5*incidence+.2*total

def global_chemistry_losses(outputs,batch):
 """Whole-graph valence, connectivity, degree, edge-count with size-normalized edge calibration."""
 m=batch["atom_mask"];n=m.shape[1];pair=m[:,:,None]&m[:,None,:]&~torch.eye(n,dtype=torch.bool,device=m.device)[None];bp=F.softmax(outputs["bond_logits"].float(),-1);orders=bp.new_tensor([0.,1.,2.,3.]);expected_bond=(bp*orders).sum(-1)*pair;presence=(1-bp[...,0])*pair;target_presence=(batch["atom_bonds"]>0)*pair
 hp=F.softmax(outputs["hydrogen_logits"].float(),-1);expected_h=(hp*hp.new_tensor([0.,1.,2.,3.,4.])).sum(-1);target_valence=batch["atom_hydrogen"].float()+(batch["atom_bonds"].float()*pair).sum(-1);pred_valence=expected_h+expected_bond.sum(-1);valence=F.smooth_l1_loss(pred_valence[m],target_valence[m])
 degree=F.smooth_l1_loss(presence.sum(-1)[m],target_presence.sum(-1).float()[m]);pred_edges=presence.sum((1,2))/2;target_edges=target_presence.sum((1,2)).float()/2;nodes=m.sum(-1).float().clamp_min(1);edge_count=F.smooth_l1_loss(pred_edges/nodes,target_edges/nodes)
 cp=F.softmax(outputs["charge_logits"].float(),-1);expected_charge=(cp*cp.new_tensor([-2.,-1.,0.,1.,2.])).sum(-1)*m;charge=F.smooth_l1_loss(expected_charge.sum(-1),(batch["atom_charge"].float()-2).mul(m).sum(-1));rp=F.softmax(outputs["radical_logits"].float(),-1);expected_radical=(rp*rp.new_tensor([0.,1.,2.])).sum(-1)*m;radical=F.smooth_l1_loss(expected_radical.sum(-1),batch["atom_radical"].float().sum(-1))
 conn=[]
 for count in torch.unique(m.sum(-1)).tolist():
  rows=m.sum(-1)==count;k=int(count)
  if k<=1:continue
  p=presence[rows,:k,:k];lap=torch.diag_embed(p.sum(-1))-p;sign,logdet=torch.linalg.slogdet(lap[:,:k-1,:k-1]+torch.eye(k-1,device=p.device,dtype=p.dtype)*1e-6);conn.append((F.softplus(-logdet/k)+(sign<=0).to(logdet.dtype)*10).mean())
 connectivity=torch.stack(conn).mean() if conn else outputs["bond_logits"].sum()*0
 return {"valence":.2*valence,"degree":.1*degree,"edge_count":.1*edge_count,"connectivity":.2*connectivity,"charge_balance":.05*charge,"radical_count":.05*radical}

def geometry_consistency_losses(outputs,batch,noisy_positions,t,diff):
 """Tie predicted clean geometry to graph distances and suppress clashes."""
 m=batch["atom_mask"];n=m.shape[1];pair=m[:,:,None]&m[:,None,:];upper=pair&torch.triu(torch.ones(n,n,dtype=torch.bool,device=m.device),1)[None];a=diff.at(diff.abar,t,noisy_positions.ndim);active=(diff.abar[t]>.01)[:,None,None];x0=(noisy_positions-(1-a).sqrt()*outputs["coordinate_eps"])/a.sqrt().clamp_min(.1);d=torch.cdist(x0,x0);target_d=torch.cdist(batch["atom_positions"],batch["atom_positions"]);bonded=upper&(batch["atom_bonds"]>0)&active;nonbonded=upper&~(batch["atom_bonds"]>0)&active
 bond_length=F.smooth_l1_loss(d[bonded],target_d[bonded]) if bonded.any() else d.sum()*0;clash=F.relu(.75-d[nonbonded]).square().mean() if nonbonded.any() else d.sum()*0
 return {"bond_length":.1*bond_length,"clash":.1*clash}
def joint_training_loss(model,diff,b):
 """Empirical-marginal D3PM with hybrid x0-CE/VLB and joint Cartesian DDPM."""
 m=b["atom_mask"];B,n=m.shape;t=torch.randint(diff.timesteps,(B,),device=m.device);pair=m[:,:,None]&m[:,None,:];diag=torch.eye(n,dtype=torch.bool,device=m.device)[None];upper=pair&torch.triu(torch.ones(n,n,dtype=torch.bool,device=m.device),1)[None];pos,noise=diff.q_coords(b["atom_positions"],t,m);edge=(b["atom_bonds"]>0).long();bond_type=(b["atom_bonds"]-1).clamp_min(0);arom=b["aromatic_edge_mask"].bool();kekule_arom=arom&(b["atom_bonds"]>=1)&(b["atom_bonds"]<=2);regular_bond=upper&edge.bool()&~kekule_arom
 xt_atom=diff.q_token(b["atom_type"],t,model.atom_types,m,name="atom");xt_charge=diff.q_token(b["atom_charge"],t,5,m,name="charge");xt_h=diff.q_token(b["atom_hydrogen"],t,5,m,name="hydrogen");xt_rad=diff.q_token(b["atom_radical"],t,3,m,name="radical");xt_edge=diff.q_token(edge,t,2,pair&~diag,True,"edge");xt_type=diff.q_token(bond_type,t,3,pair&~diag,True,"bond_type")
 s={"atom_mask":m,"atom_type":xt_atom,"charge":xt_charge,"hydrogen":xt_h,"radical":xt_rad,"edge_exists":xt_edge,"bond_type":xt_type,"positions":pos};o=model(b["condition"],t,s)
 terms={"count":F.cross_entropy(o["count_logits"],m.sum(1)-1),"atom":_ce(o["atom_logits"],b["atom_type"],m),"charge":_ce(o["charge_logits"],b["atom_charge"],m),"hydrogen":_ce(o["hydrogen_logits"],b["atom_hydrogen"],m),"radical":_ce(o["radical_logits"],b["atom_radical"],m),"edge_exists":_ce(o["edge_logits"],edge,upper),"bond_type":_ce(o["bond_type_logits"],bond_type,regular_bond),"aromatic_cycle":aromatic_cycle_loss(o["bond_logits"],o["aromatic_edge_logits"],b["aromatic_cycle_edges"],b["aromatic_cycle_double_count"],b["ring_mask"],b["aromatic_cycle_order"],b["aromatic_cycle_size"],kekule_arom,arom,upper),"aromatic_component":aromatic_component_loss(o,b),"coordinate":((o["coordinate_eps"]-noise).square()*m[...,None]).sum()/m.sum().clamp_min(1)}
 terms.update({"vlb_atom":.1*diff.categorical_vlb(b["atom_type"],xt_atom,o["atom_logits"],t,model.atom_types,m,"atom"),"vlb_charge":.1*diff.categorical_vlb(b["atom_charge"],xt_charge,o["charge_logits"],t,5,m,"charge"),"vlb_hydrogen":.1*diff.categorical_vlb(b["atom_hydrogen"],xt_h,o["hydrogen_logits"],t,5,m,"hydrogen"),"vlb_radical":.1*diff.categorical_vlb(b["atom_radical"],xt_rad,o["radical_logits"],t,3,m,"radical"),"vlb_edge":.1*diff.categorical_vlb(edge,xt_edge,o["edge_logits"],t,2,upper,"edge"),"vlb_bond_type":.1*diff.categorical_vlb(bond_type,xt_type,o["bond_type_logits"],t,3,regular_bond,"bond_type")});terms.update(global_chemistry_losses(o,b));terms.update(geometry_consistency_losses(o,b,pos,t,diff));return sum(terms.values()),terms

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
