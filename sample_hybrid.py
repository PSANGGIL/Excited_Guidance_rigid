#!/usr/bin/env python3
"""Conditional joint sampling of rigid rings and unconstrained free heavy atoms."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import torch
from polyring_diffusion.diffusion import LinearNoiseSchedule,center_coordinates,normalize_vectors
from polyring_diffusion.hybrid_model import HybridRingAtomDenoiser
from sample_rigid import project,reveal

def guided(model,args,condition,scale):
 c=model(*args,condition=condition)
 if scale==1:return c
 drop=torch.ones(condition.shape[0],dtype=torch.bool,device=condition.device);u=model(*args,condition=torch.zeros_like(condition),condition_drop_mask=drop)
 skip={'hidden','atom_hidden','property_pred'};return {k:(u[k]+scale*(c[k]-u[k]) if k not in skip else c[k]) for k in c}
@torch.no_grad()
def sample(checkpoint,target,n_samples,num_rings,guidance,seed,output,max_free_atoms=None):
 torch.manual_seed(seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');state=torch.load(checkpoint,map_location=device,weights_only=False);cfg=state['config'];features=state['template_features'];mc=dict(cfg['model']);mc.update(num_atom_types=len(state['atom_map']),num_templates=features.shape[0],template_features=features.tolist());model=HybridRingAtomDenoiser(**mc).to(device);model.load_state_dict(state['model']);model.eval();schedule=LinearNoiseSchedule(**cfg['diffusion']).to(device);eps=1e-6;norm=(math.log10(target+eps)-state['property_mean'])/state['property_std'];condition=torch.full((n_samples,1),norm,device=device)
 count_logits=model.count_head(condition);counts=torch.distributions.Categorical(logits=count_logits).sample();cap=model.max_free_atoms if max_free_atoms is None else min(max_free_atoms,model.max_free_atoms);counts=counts.clamp_max(cap);na=max(1,int(counts.max()));am=torch.arange(na,device=device)[None]<counts[:,None];rm=torch.ones((n_samples,num_rings),dtype=torch.bool,device=device);rt=torch.full((n_samples,num_rings),model.ring_model.ring_mask_token,dtype=torch.long,device=device);rpair=rm[:,:,None]&rm[:,None,:];rpair&=~torch.eye(num_rings,dtype=torch.bool,device=device)[None];rconn=torch.zeros((n_samples,num_rings,num_rings),dtype=torch.long,device=device);rconn[rpair]=model.ring_model.connection_mask_token;centers=center_coordinates(torch.randn(n_samples,num_rings,3,device=device),rm);normals=normalize_vectors(torch.randn(n_samples,num_rings,3,device=device),rm);at=torch.where(am,torch.full_like(am,model.atom_mask_token,dtype=torch.long),torch.full_like(am,model.atom_pad_token,dtype=torch.long));aq=torch.where(am,torch.full_like(am,model.charge_mask_token,dtype=torch.long),torch.zeros_like(am,dtype=torch.long));apair=am[:,:,None]&am[:,None,:];apair&=~torch.eye(na,dtype=torch.bool,device=device)[None];ab=torch.zeros((n_samples,na,na),dtype=torch.long,device=device);ab[apair]=5;attach=torch.full((n_samples,num_rings,na),5,dtype=torch.long,device=device);attach=torch.where(rm[:,:,None]&am[:,None,:],attach,torch.zeros_like(attach));apos=center_coordinates(torch.randn(n_samples,na,3,device=device),am)
 for step in reversed(range(schedule.timesteps)):
  t=torch.full((n_samples,),step,dtype=torch.long,device=device);p=guided(model,((rt,centers,normals,rconn,rm),at,aq,apos,ab,attach,am,t),condition,guidance);centers=center_coordinates(schedule.p_step(centers,p['center_noise'],t),rm);normals=normalize_vectors(schedule.p_step(normals,p['normal_noise'],t),rm);apos=center_coordinates(schedule.p_step(apos,p['atom_coordinate_noise'],t),am);rt=torch.where(reveal(rt==model.ring_model.ring_mask_token,step),torch.distributions.Categorical(logits=p['ring_logits']).sample(),rt);at=torch.where(reveal((at==model.atom_mask_token)&am,step),torch.distributions.Categorical(logits=p['atom_type_logits']).sample(),at);aq=torch.where(reveal((aq==model.charge_mask_token)&am,step),torch.distributions.Categorical(logits=p['atom_charge_logits']).sample(),aq);ru=rpair&torch.triu(torch.ones(num_rings,num_rings,dtype=torch.bool,device=device),1)[None];rconn=torch.where(reveal((rconn==model.ring_model.connection_mask_token)&ru,step),torch.distributions.Categorical(logits=p['connection_logits']).sample(),rconn);rconn=torch.where(ru,rconn,rconn.transpose(1,2));au=apair&torch.triu(torch.ones(na,na,dtype=torch.bool,device=device),1)[None];ab=torch.where(reveal((ab==5)&au,step),torch.distributions.Categorical(logits=p['atom_bond_logits']).sample(),ab);ab=torch.where(au,ab,ab.transpose(1,2));cm=rm[:,:,None]&am[:,None,:];attach=torch.where(reveal((attach==5)&cm,step),torch.distributions.Categorical(logits=p['attachment_logits']).sample(),attach)
 p=guided(model,((rt,centers,normals,rconn,rm),at,aq,apos,ab,attach,am,torch.zeros(n_samples,dtype=torch.long,device=device)),condition,guidance);templates=sorted(state['manifest']['templates'],key=lambda x:x['id']);records=[]
 for b in range(n_samples):
  nr=int(counts[b]);ids=rt[b].tolist();rings=[{'template_id':x,'signature':templates[x]['signature'],'center':centers[b,i].tolist(),'normal':normals[b,i].tolist()} for i,x in enumerate(ids)];edges,connected=project(ids,p['connection_logits'][b],p['port_logits'][b],templates);atom_edges=[]
  for i in range(nr):
   for j in range(i+1,nr):
    kind=int(p['atom_bond_logits'][b,i,j].argmax());
    if kind:atom_edges.append({'left':i,'right':j,'bond_type':kind})
  attachments=[]
  for r in range(num_rings):
   for i in range(nr):
    kind=int(attach[b,r,i]);
    if kind:attachments.append({'ring':r,'atom':i,'ring_port':int(p['attachment_port_logits'][b,r,i].argmax()),'bond_type':kind})
  atoms=[{'element':state['atom_map'][int(at[b,i])],'formal_charge':int(aq[b,i])+model.charge_min,'position':apos[b,i].tolist()} for i in range(nr)];records.append({'target_f_osc':target,'normalized_condition':norm,'connected_rings':connected,'rings':rings,'ring_connections':edges,'free_atoms':atoms,'free_atom_connections':atom_edges,'ring_atom_attachments':attachments})
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(json.dumps(x)+'\n' for x in records));output.with_suffix('.summary.json').write_text(json.dumps({'samples':len(records),'free_atom_count_histogram':{str(i):int((counts==i).sum()) for i in counts.unique().tolist()},'mean_free_atoms':float(counts.float().mean())},indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--target',type=float,default=.5);p.add_argument('--n-samples',type=int,default=100);p.add_argument('--num-rings',type=int,default=6);p.add_argument('--guidance',type=float,default=1.5);p.add_argument('--seed',type=int,default=123);p.add_argument('--max-free-atoms',type=int);p.add_argument('--output',type=Path,default=Path('samples/hybrid.jsonl'));a=p.parse_args();sample(a.checkpoint,a.target,a.n_samples,a.num_rings,a.guidance,a.seed,a.output,a.max_free_atoms)
if __name__=='__main__':main()
