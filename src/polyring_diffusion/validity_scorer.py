"""Learned global validity reward for completed ring-connection trees."""
from __future__ import annotations
import math
import torch
from torch import nn

FEATURE_NAMES=(
 "condition","edge_score_sum","edge_score_mean","edge_score_min","edge_score_max",
 "single_fraction","fused_fraction","spiro_fraction","hetero_fraction","aromatic_fraction",
 "charged_fraction","boron_fraction","nitrogen_fraction","oxygen_fraction","sulfur_fraction",
 "mean_ring_size","min_ring_size","max_ring_size","used_port_fraction","fused_port_fraction",
 "mean_left_valence_capacity","mean_right_valence_capacity","ring_count","edge_count",
)
_VALENCE={5:3.,6:4.,7:3.,8:2.,9:1.,15:5.,16:6.,17:1.,35:1.}
def _bond_order(x):return {1:1.,2:2.,3:3.,4:1.5}.get(int(x),0.)
def _capacity(sig,p):
 z,q=int(sig[p][0]),int(sig[p][1]);internal=_bond_order(sig[p-1][3])+_bond_order(sig[p][3]);return max(0.,_VALENCE.get(z,4.)+max(q,0)-internal)
def feature_vector(record):
 rings=record['rings'];edges=record['connections'];atoms=[a for r in rings for a in r['signature']];n=max(len(atoms),1);scores=[float(e.get('score',0.)) for e in edges] or [0.];sizes=[len(r['signature']) for r in rings] or [0]
 kinds=[e['kind'] for e in edges];den=max(len(edges),1);used=set();fused_used=set();lc=[];rc=[]
 for e in edges:
  i,j,lp,rp=e['left'],e['right'],e['left_port'],e['right_port'];used.update(((i,lp),(j,rp)));lc.append(_capacity(rings[i]['signature'],lp));rc.append(_capacity(rings[j]['signature'],rp))
  if e['kind']=='FUSED':fused_used.update(((i,lp),(i,(lp+1)%len(rings[i]['signature'])),(j,rp),(j,(rp+1)%len(rings[j]['signature']))))
 ports=max(sum(sizes),1);zs=[int(a[0]) for a in atoms]
 return [float(record.get('normalized_condition',0.)),sum(scores),sum(scores)/len(scores),min(scores),max(scores),kinds.count('SINGLE')/den,kinds.count('FUSED')/den,kinds.count('SPIRO')/den,sum(z!=6 for z in zs)/n,sum(int(a[3])==4 for a in atoms)/n,sum(int(a[1])!=0 for a in atoms)/n,zs.count(5)/n,zs.count(7)/n,zs.count(8)/n,zs.count(16)/n,sum(sizes)/len(sizes),min(sizes),max(sizes),len(used)/ports,len(fused_used)/ports,sum(lc)/max(len(lc),1),sum(rc)/max(len(rc),1),len(rings),len(edges)]

class PropertyScorer(nn.Module):
 def __init__(self,input_dim=len(FEATURE_NAMES),hidden=128):
  super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def load_property_scorer(path,device='cpu'):
 state=torch.load(path,map_location=device,weights_only=False);model=PropertyScorer(len(state['feature_names']),state['hidden']).to(device);model.load_state_dict(state['model']);model.eval();return model,state['mean'].to(device),state['std'].to(device),state

class ValidityScorer(nn.Module):
 def __init__(self,input_dim=len(FEATURE_NAMES),hidden=64):
  super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.SiLU(),nn.Dropout(.1),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
 def forward(self,x):return self.net(x).squeeze(-1)

def load_validity_scorer(path,device='cpu'):
 state=torch.load(path,map_location=device,weights_only=False);model=ValidityScorer(len(state['feature_names']),state['hidden']).to(device);model.load_state_dict(state['model']);model.eval();return model,state['mean'].to(device),state['std'].to(device),state
