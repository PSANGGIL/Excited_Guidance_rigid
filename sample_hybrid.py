#!/usr/bin/env python3
"""Conditional joint sampling of rigid rings and unconstrained free heavy atoms."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from polyring_diffusion.diffusion import LinearNoiseSchedule,center_coordinates,normalize_vectors
from polyring_diffusion.hybrid_model import HybridRingAtomDenoiser
from polyring_diffusion.chem import element_record
from sample_rigid import project,reveal


BOND_ORDER={1:1.0,2:2.0,3:3.0,4:1.5}
MAX_VALENCE={1:1.0,5:3.0,6:4.0,7:3.0,8:2.0,9:1.0,14:4.0,15:5.0,16:6.0,17:1.0,35:1.0,53:1.0}

def project_attachments(atom_type_ids,atom_bonds,attachment_logits,attachment_port_logits,rings,ring_edges,atom_map):
 """Greedily select positive-score attachments within atom and port valence budgets."""
 nr,na=attachment_logits.shape[:2];offsets=[];signatures=[];size=0
 for ring in rings:offsets.append(size);signatures.append(ring["signature"]);size+=len(ring["signature"])
 parent=list(range(size))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[b]=a
 for edge in ring_edges:
  i,j,lp,rp=edge["left"],edge["right"],edge["left_port"],edge["right_port"];kind=edge["kind"]
  if kind=="SPIRO":union(offsets[i]+lp,offsets[j]+rp)
  elif kind=="FUSED":
   li,ln=offsets[i]+lp,offsets[i]+(lp+1)%len(signatures[i]);ri,rn=offsets[j]+rp,offsets[j]+(rp+1)%len(signatures[j]);key=lambda r,p:(signatures[r][p][0],signatures[r][p][1])
   if key(i,lp)==key(j,(rp+1)%len(signatures[j])) and key(i,(lp+1)%len(signatures[i]))==key(j,rp):union(li,rn);union(ln,ri)
   else:union(li,ri);union(ln,rn)
 atomic_numbers={"H":1,"B":5,"C":6,"N":7,"O":8,"F":9,"Si":14,"P":15,"S":16,"Cl":17,"Br":35,"I":53};atom_budget=[]
 for a in range(na):
  z=atomic_numbers.get(atom_map[int(atom_type_ids[a])],6);budget=MAX_VALENCE.get(z,4.0)-sum(BOND_ORDER.get(int(atom_bonds[a,j]),0.0) for j in range(na));atom_budget.append(max(0.0,budget))
 root_budget={}
 for r,sig in enumerate(signatures):
  for port,spec in enumerate(sig):
   budget=max(0.0,MAX_VALENCE.get(int(spec[0]),4.0)-BOND_ORDER.get(int(sig[(port-1)%len(sig)][3]),1.0)-BOND_ORDER.get(int(spec[3]),1.0));root=find(offsets[r]+port);root_budget[root]=min(root_budget.get(root,budget),budget)
 used_roots=set()
 for edge in ring_edges:
  if edge["kind"]=="SINGLE":
   for r,port in ((edge["left"],edge["left_port"]),(edge["right"],edge["right_port"])):
    root=find(offsets[r]+port);root_budget[root]=max(0.0,root_budget[root]-1.0);used_roots.add(root)
 logp=torch.log_softmax(attachment_logits.float(),-1).cpu();portp=torch.log_softmax(attachment_port_logits.float(),-1).cpu();candidates=[]
 for r in range(nr):
  for a in range(na):
   for kind in range(1,attachment_logits.shape[-1]):
    port=int(portp[r,a,:len(signatures[r])].argmax());score=float(logp[r,a,kind]-logp[r,a,0]+portp[r,a,port]);candidates.append((score,r,a,port,kind))
 selected=[];ranked=sorted(candidates,reverse=True)
 def available(candidate):
  score,r,a,port,kind=candidate;order=BOND_ORDER[kind];root=find(offsets[r]+port);return root not in used_roots and atom_budget[a]>=order and root_budget[root]>=order
 def add(candidate):
  score,r,a,port,kind=candidate;order=BOND_ORDER[kind];root=find(offsets[r]+port);selected.append({"ring":r,"atom":a,"ring_port":port,"bond_type":kind,"score":score});atom_budget[a]-=order;root_budget[root]-=order;used_roots.add(root)
 for candidate in ranked:
  if candidate[0]>0 and available(candidate):add(candidate)
 atom_parent=list(range(na))
 def atom_find(x):
  while atom_parent[x]!=x:atom_parent[x]=atom_parent[atom_parent[x]];x=atom_parent[x]
  return x
 for i in range(na):
  for j in range(i+1,na):
   if int(atom_bonds[i,j]):
    a,b=atom_find(i),atom_find(j);atom_parent[b]=a
 attached_components={atom_find(x["atom"]) for x in selected}
 for component in {atom_find(i) for i in range(na)}-attached_components:
  choice=next((x for x in ranked if atom_find(x[2])==component and available(x)),None)
  if choice is not None:add(choice)
 return selected

def guided(model,args,condition,scale):
 c=model(*args,condition=condition)
 if scale==1:return c
 drop=torch.ones(condition.shape[0],dtype=torch.bool,device=condition.device);u=model(*args,condition=torch.zeros_like(condition),condition_drop_mask=drop)
 skip={'hidden','atom_hidden','property_pred'};return {k:(u[k]+scale*(c[k]-u[k]) if k not in skip else c[k]) for k in c}
@torch.no_grad()
def sample(checkpoint,target,n_samples,num_rings,guidance,seed,output,max_free_atoms=None,energy_min=None,energy_max=None,target_min=None,target_max=None,ring_size_bias_strength=0.0,zero_free_atom_bias=0.0):
 torch.manual_seed(seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');state=torch.load(checkpoint,map_location=device,weights_only=False);cfg=state['config'];features=state['template_features'];mc=dict(cfg['model']);mc.update(num_atom_types=len(state['atom_map']),num_templates=features.shape[0],template_features=features.tolist(),atom_radii=[element_record(x).vdw_radius/3.0 for x in state['atom_map']]);model=HybridRingAtomDenoiser(**mc).to(device);model.load_state_dict(state['model']);model.eval();schedule=LinearNoiseSchedule(**cfg['diffusion']).to(device);eps=1e-6
 templates=sorted(state['manifest']['templates'],key=lambda x:x['id']);size_adjust={3:.98,4:1.88,5:.54,6:-.31,7:-1.33,8:-2.96,9:-5.07,10:-6.0};ring_size_bias=torch.tensor([ring_size_bias_strength*size_adjust.get(len(x['signature']),-6.0) for x in templates],device=device)
 if state.get('condition_stats'):
  stats=state['condition_stats']['singlet'];energies=torch.full((n_samples,),float(energy_min if energy_min is not None else stats['energy_mean']),device=device) if energy_max is None else torch.empty(n_samples,device=device).uniform_(energy_min,energy_max);osc=torch.full((n_samples,),target,device=device) if target_min is None else torch.empty(n_samples,device=device).uniform_(target_min,target_max);norm=(torch.log10(osc+eps)-stats['log_os_mean'])/stats['log_os_std'];condition=torch.zeros((n_samples,mc['condition_dim']),device=device);condition[:,0]=(energies-stats['energy_mean'])/stats['energy_std'];condition[:,20]=norm;condition[:,40]=1
 else:
  energies=torch.full((n_samples,),float('nan'),device=device);osc=torch.full((n_samples,),target,device=device);norm=(math.log10(target+eps)-state['property_mean'])/state['property_std'];condition=torch.full((n_samples,1),norm,device=device)
 count_logits=model.count_head(condition).clone();count_logits[:,0]+=zero_free_atom_bias;counts=torch.distributions.Categorical(logits=count_logits).sample();cap=model.max_free_atoms if max_free_atoms is None else min(max_free_atoms,model.max_free_atoms);counts=counts.clamp_max(cap);na=max(1,int(counts.max()));am=torch.arange(na,device=device)[None]<counts[:,None];rm=torch.ones((n_samples,num_rings),dtype=torch.bool,device=device);rt=torch.full((n_samples,num_rings),model.ring_model.ring_mask_token,dtype=torch.long,device=device);rpair=rm[:,:,None]&rm[:,None,:];rpair&=~torch.eye(num_rings,dtype=torch.bool,device=device)[None];rconn=torch.zeros((n_samples,num_rings,num_rings),dtype=torch.long,device=device);rconn[rpair]=model.ring_model.connection_mask_token;centers=center_coordinates(torch.randn(n_samples,num_rings,3,device=device),rm);normals=normalize_vectors(torch.randn(n_samples,num_rings,3,device=device),rm);tangents=torch.randn_like(normals);tangents=normalize_vectors(tangents-(tangents*normals).sum(-1,keepdim=True)*normals,rm);at=torch.where(am,torch.full_like(am,model.atom_mask_token,dtype=torch.long),torch.full_like(am,model.atom_pad_token,dtype=torch.long));aq=torch.where(am,torch.full_like(am,model.charge_mask_token,dtype=torch.long),torch.zeros_like(am,dtype=torch.long));apair=am[:,:,None]&am[:,None,:];apair&=~torch.eye(na,dtype=torch.bool,device=device)[None];ab=torch.zeros((n_samples,na,na),dtype=torch.long,device=device);ab[apair]=5;attach=torch.full((n_samples,num_rings,na),5,dtype=torch.long,device=device);attach=torch.where(rm[:,:,None]&am[:,None,:],attach,torch.zeros_like(attach));apos=center_coordinates(torch.randn(n_samples,na,3,device=device),am)
 for step in reversed(range(schedule.timesteps)):
  t=torch.full((n_samples,),step,dtype=torch.long,device=device);p=guided(model,((rt,centers,normals,tangents,rconn,rm),at,aq,apos,ab,attach,am,t),condition,guidance);centers=center_coordinates(schedule.p_step(centers,p['center_noise'],t),rm);normals=normalize_vectors(schedule.p_step(normals,p['normal_noise'],t),rm);tangents=schedule.p_step(tangents,p['tangent_noise'],t);tangents=normalize_vectors(tangents-(tangents*normals).sum(-1,keepdim=True)*normals,rm);apos=center_coordinates(schedule.p_step(apos,p['atom_coordinate_noise'],t),am);rt=torch.where(reveal(rt==model.ring_model.ring_mask_token,step),(torch.distributions.Categorical(logits=p['ring_logits']+ring_size_bias).sample() if num_rings else rt),rt);at=torch.where(reveal((at==model.atom_mask_token)&am,step),torch.distributions.Categorical(logits=p['atom_type_logits']).sample(),at);aq=torch.where(reveal((aq==model.charge_mask_token)&am,step),torch.distributions.Categorical(logits=p['atom_charge_logits']).sample(),aq);ru=rpair&torch.triu(torch.ones(num_rings,num_rings,dtype=torch.bool,device=device),1)[None];rconn=torch.where(reveal((rconn==model.ring_model.connection_mask_token)&ru,step),(torch.distributions.Categorical(logits=p['connection_logits']).sample() if num_rings else rconn),rconn);rconn=torch.where(ru,rconn,rconn.transpose(1,2));au=apair&torch.triu(torch.ones(na,na,dtype=torch.bool,device=device),1)[None];ab=torch.where(reveal((ab==5)&au,step),torch.distributions.Categorical(logits=p['atom_bond_logits']).sample(),ab);ab=torch.where(au,ab,ab.transpose(1,2));cm=rm[:,:,None]&am[:,None,:];attach=torch.where(reveal((attach==5)&cm,step),(torch.distributions.Categorical(logits=p['attachment_logits']).sample() if num_rings else attach),attach)
 p=guided(model,((rt,centers,normals,tangents,rconn,rm),at,aq,apos,ab,attach,am,torch.zeros(n_samples,dtype=torch.long,device=device)),condition,guidance);templates=sorted(state['manifest']['templates'],key=lambda x:x['id']);records=[]
 for b in range(n_samples):
  nr=int(counts[b]);ids=rt[b].tolist();rings=[{'template_id':x,'signature':templates[x]['signature'],'center':centers[b,i].tolist(),'normal':normals[b,i].tolist(),'tangent':tangents[b,i].tolist()} for i,x in enumerate(ids)];edges,connected=project(ids,p['connection_logits'][b],p['port_logits'][b],templates) if num_rings else ([],True);atom_edges=[]
  for i in range(nr):
   for j in range(i+1,nr):
    kind=int(p['atom_bond_logits'][b,i,j].argmax());
    if kind:atom_edges.append({'left':i,'right':j,'bond_type':kind})
  final_atom_bonds=torch.zeros((nr,nr),dtype=torch.long,device=at.device)
  for edge in atom_edges:final_atom_bonds[edge['left'],edge['right']]=final_atom_bonds[edge['right'],edge['left']]=edge['bond_type']
  attachments=project_attachments(at[b,:nr],final_atom_bonds,p['attachment_logits'][b,:,:nr],p['attachment_port_logits'][b,:,:nr],rings,edges,state['atom_map'])
  atoms=[{'element':state['atom_map'][int(at[b,i])],'formal_charge':int(aq[b,i])+model.charge_min,'position':apos[b,i].tolist()} for i in range(nr)];records.append({'target_energy':float(energies[b]),'target_f_osc':float(osc[b]),'normalized_condition':float(norm[b]) if norm.ndim else float(norm),'connected_rings':connected,'rings':rings,'ring_connections':edges,'free_atoms':atoms,'free_atom_connections':atom_edges,'ring_atom_attachments':attachments})
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(json.dumps(x)+'\n' for x in records));output.with_suffix('.summary.json').write_text(json.dumps({'samples':len(records),'free_atom_count_histogram':{str(i):int((counts==i).sum()) for i in counts.unique().tolist()},'mean_free_atoms':float(counts.float().mean())},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--target',type=float,default=.5);p.add_argument('--target-min',type=float);p.add_argument('--target-max',type=float);p.add_argument('--energy-min',type=float);p.add_argument('--energy-max',type=float);p.add_argument('--ring-size-bias-strength',type=float,default=0.0);p.add_argument('--zero-free-atom-bias',type=float,default=0.0);p.add_argument('--n-samples',type=int,default=100);p.add_argument('--num-rings',type=int,default=6);p.add_argument('--guidance',type=float,default=1.5);p.add_argument('--seed',type=int,default=123);p.add_argument('--max-free-atoms',type=int);p.add_argument('--output',type=Path,default=Path('samples/hybrid.jsonl'));a=p.parse_args();sample(a.checkpoint,a.target,a.n_samples,a.num_rings,a.guidance,a.seed,a.output,a.max_free_atoms,a.energy_min,a.energy_max,a.target_min,a.target_max,a.ring_size_bias_strength,a.zero_free_atom_bias)
if __name__=='__main__':main()
