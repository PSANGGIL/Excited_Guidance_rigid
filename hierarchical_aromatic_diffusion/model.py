"""Timestep-aware, condition-shared hierarchical molecular denoisers."""
import math,torch
from torch import nn
class TimeEmbedding(nn.Module):
 def __init__(self,h):super().__init__();self.h=h;self.net=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,h))
 def forward(self,t):
  half=self.h//2;f=torch.exp(torch.arange(half,device=t.device)*(-math.log(10000)/max(half-1,1)));z=t.float()[:,None]*f[None];return self.net(torch.cat((z.sin(),z.cos()),-1))
class ConditionEncoder(nn.Module):
 def __init__(self,h=256,k=16):
  super().__init__();c=torch.linspace(-4,4,k);self.register_buffer('c',c);self.register_buffer('iw',(c[1]-c[0]).reciprocal());self.net=nn.Sequential(nn.Linear(2*k,h),nn.SiLU(),nn.Linear(h,h))
 def forward(self,y):
  if y.ndim!=2 or y.shape[-1]!=2:raise ValueError('normalized S1/T1 must be [B,2]')
  return self.net(torch.exp(-.5*((y[...,None]-self.c)*self.iw).square()).flatten(1))
class PairHead(nn.Module):
 def __init__(self,h,out):super().__init__();self.net=nn.Sequential(nn.Linear(2*h+1,h),nn.SiLU(),nn.Linear(h,out))
 def forward(self,x,p):
  n=x.shape[1];d=(p[:,:,None]-p[:,None]).norm(dim=-1,keepdim=True);z=torch.cat((x[:,:,None].expand(-1,n,n,-1),x[:,None].expand(-1,n,n,-1),d),-1);o=self.net(z);return(o+o.transpose(1,2))/2
class RingTopologyDenoiser(nn.Module):
 def __init__(self,h,max_rings,max_size,layers=6):
  super().__init__();self.r=max_rings;self.s=max_size;self.query=nn.Parameter(torch.randn(max_rings,h)/h**.5);self.size_e=nn.Embedding(max_size-1,h);self.arom_e=nn.Embedding(3,h);self.rel_e=nn.Embedding(14,h);enc=nn.TransformerEncoderLayer(h,8,4*h,batch_first=True,norm_first=True);self.net=nn.TransformerEncoder(enc,layers);self.count=nn.Linear(h,max_rings+1);self.size=nn.Linear(h,max_size-2);self.arom=nn.Linear(h,2);self.eps=nn.Linear(h,3);self.rel=PairHead(h,13);self.slot=nn.Embedding(max_size,h);self.junction=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,2))
 def forward(self,c,te,state):
  size,arom,rel,pos,mask=(state[k] for k in ('size','aromatic','relation','positions','mask'));x=self.query[None]+c[:,None]+te[:,None]+self.size_e(size)+self.arom_e(arom);x=x+(self.rel_e(rel).sum(2)/mask.sum(1)[:,None,None].clamp_min(1));x=self.net(x,src_key_padding_mask=~mask)*mask[...,None];slot=x[:,:,None]+self.slot.weight[None,None];l=slot[:,:,None,:,None,:].expand(-1,-1,self.r,-1,self.s,-1);q=slot[:,None,:,None,:,:].expand(-1,self.r,-1,self.s,-1,-1);j=self.junction(torch.cat((l,q),-1));j=(j+j.permute(0,2,1,4,3,5))/2
  return {'ring_count_logits':self.count(c),'ring_size_logits':self.size(x),'ring_aromatic_logits':self.arom(x),'ring_eps':self.eps(x)*mask[...,None],'relation_logits':self.rel(x,pos),'junction_slot_logits':j,'ring_hidden':x}
class ScaffoldDenoiser(nn.Module):
 def __init__(self,h,max_slots,atom_types,layers=4):
  super().__init__();self.slot=nn.Embedding(max_slots,h);self.ae=nn.Embedding(atom_types+1,h);self.ce=nn.Embedding(6,h);self.he=nn.Embedding(6,h);self.re=nn.Embedding(4,h);enc=nn.TransformerEncoderLayer(h,8,4*h,batch_first=True,norm_first=True);self.net=nn.TransformerEncoder(enc,layers);self.atom=nn.Linear(h,atom_types);self.charge=nn.Linear(h,5);self.hydrogen=nn.Linear(h,5);self.radical=nn.Linear(h,3);self.bond=PairHead(h,4)
 def forward(self,c,te,ring_h,state):
  mask,pos=state['mask'],state['positions'];b,r,s=mask.shape;ids=torch.arange(s,device=mask.device);x=ring_h[:,:,None]+c[:,None,None]+te[:,None,None]+self.slot(ids)[None,None]+self.ae(state['atom'])+self.ce(state['charge'])+self.he(state['hydrogen'])+self.re(state['radical']);x=self.net(x.reshape(b,r*s,-1),src_key_padding_mask=~mask.reshape(b,r*s)).reshape(b,r,s,-1)*mask[...,None];flat=x.reshape(b,r*s,-1)
  return {'slot_atom_logits':self.atom(x),'slot_charge_logits':self.charge(x),'slot_hydrogen_logits':self.hydrogen(x),'slot_radical_logits':self.radical(x),'slot_bond_logits':self.bond(flat,pos.reshape(b,r*s,3)),'slot_hidden':x}
class SubstituentDenoiser(nn.Module):
 def __init__(self,h,max_atoms,atom_types,layers=4):
  super().__init__();self.n=max_atoms;self.query=nn.Parameter(torch.randn(max_atoms,h)/h**.5);self.ae=nn.Embedding(atom_types+1,h);enc=nn.TransformerEncoderLayer(h,8,4*h,batch_first=True,norm_first=True);self.net=nn.TransformerEncoder(enc,layers);self.count=nn.Linear(h,max_atoms+1);self.atom=nn.Linear(h,atom_types);self.attach=nn.Sequential(nn.Linear(2*h,h),nn.SiLU(),nn.Linear(h,4))
 def forward(self,c,te,scaffold_h,state):
  x=self.query[None]+c[:,None]+te[:,None]+self.ae(state['atom']);x=self.net(x,src_key_padding_mask=~state['mask'])*state['mask'][...,None];sh=scaffold_h.reshape(len(c),-1,scaffold_h.shape[-1]);pair=torch.cat((x[:,:,None].expand(-1,-1,sh.shape[1],-1),sh[:,None].expand(-1,self.n,-1,-1)),-1)
  return {'substituent_count_logits':self.count(c+sh.mean(1)),'substituent_atom_logits':self.atom(x),'attachment_logits':self.attach(pair),'substituent_hidden':x}
class CoordinateScore(nn.Module):
 def __init__(self,h,layers=4):super().__init__();self.layers=nn.ModuleList(nn.Sequential(nn.Linear(2*h+1,h),nn.SiLU(),nn.Linear(h,1,bias=False))for _ in range(layers))
 def forward(self,h,pos,mask):
  n=h.shape[1];pair=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=h.device)[None];delta=pos[:,:,None]-pos[:,None];d=delta.norm(dim=-1,keepdim=True);score=0
  for net in self.layers:score=score+net(torch.cat((h[:,:,None].expand(-1,n,n,-1),h[:,None].expand(-1,n,n,-1),d),-1))*delta*pair[...,None]
  score=score.sum(2)/pair.sum(2).clamp_min(1)[...,None];center=(score*mask[...,None]).sum(1,keepdim=True)/mask.sum(1,keepdim=True).clamp_min(1)[...,None];return(score-center)*mask[...,None]
class HierarchicalAromaticModel(nn.Module):
 def __init__(self,hidden=256,max_rings=12,max_ring_size=16,max_substituent_atoms=32,atom_types=4):
  super().__init__();self.condition=ConditionEncoder(hidden);self.time=TimeEmbedding(hidden);self.topology=RingTopologyDenoiser(hidden,max_rings,max_ring_size);self.scaffold=ScaffoldDenoiser(hidden,max_ring_size,atom_types);self.substituent=SubstituentDenoiser(hidden,max_substituent_atoms,atom_types);self.coord=CoordinateScore(hidden)
 def forward(self,y,t,topology_state,scaffold_state,substituent_state,coordinate_state=None):
  c,te=self.condition(y),self.time(t);top=self.topology(c,te,topology_state);sc=self.scaffold(c,te,top['ring_hidden'],scaffold_state);sub=self.substituent(c,te,sc['slot_hidden'],substituent_state);out={**top,**sc,**sub}
  if coordinate_state is not None:
   slot_h=sc['slot_hidden'].reshape(len(y),-1,sc['slot_hidden'].shape[-1]);index=coordinate_state['slot_to_scaffold'].reshape(len(y),-1);count=coordinate_state['scaffold_count'];scaffold_h=slot_h.new_zeros(len(y),count,slot_h.shape[-1]);den=slot_h.new_zeros(len(y),count,1)
   for row in range(len(y)):
    valid=index[row].ge(0); scaffold_h[row].index_add_(0,index[row,valid],slot_h[row,valid]);den[row].index_add_(0,index[row,valid],torch.ones(valid.sum(),1,device=slot_h.device))
   scaffold_h=scaffold_h/den.clamp_min(1);h=torch.cat((scaffold_h,sub['substituent_hidden']),1);out['coordinate_eps']=self.coord(h,coordinate_state['positions'],coordinate_state['mask'])
  return out
