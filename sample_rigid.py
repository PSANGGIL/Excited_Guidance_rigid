#!/usr/bin/env python3
"""Conditional reverse diffusion for data-derived rigid ring graphs."""
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import torch,torch.nn.functional as F
from polyring_diffusion.diffusion import LinearNoiseSchedule,center_coordinates,normalize_vectors
from polyring_diffusion.model import RingDenoiser

def reveal(masked,step): return masked & (torch.rand(masked.shape,device=masked.device)<(1.0 if step==0 else 1.0/(step+1)))
def guided(model,args,condition,scale):
    cond=model(*args,condition)
    if scale==1.0:return cond
    uncond=model(*args,torch.zeros_like(condition))
    return {k:(uncond[k]+scale*(cond[k]-uncond[k]) if k!='hidden' else cond[k]) for k in cond}
def compatible(sig1,p1,sig2,p2,kind):
    a=(sig1[p1][0],sig1[p1][1]); b=(sig2[p2][0],sig2[p2][1])
    if kind==3:return a==b
    if kind==2:
        an=(sig1[(p1+1)%len(sig1)][0],sig1[(p1+1)%len(sig1)][1]); bn=(sig2[(p2+1)%len(sig2)][0],sig2[(p2+1)%len(sig2)][1])
        return (a==bn and an==b) or (a==b and an==bn)
    return True

def project(type_ids,conn_logits,port_logits,templates,max_extra=0,noise_scale=0.0,rng=None):
    n=len(type_ids); candidates=[]; cl=F.log_softmax(conn_logits,-1); pl=F.log_softmax(port_logits,-1)
    sigs=[templates[i]['signature'] for i in type_ids]
    for i in range(n):
      for j in range(i+1,n):
       for kind in (1,2,3):
        for lp in range(len(sigs[i])):
         for rp in range(len(sigs[j])):
          if compatible(sigs[i],lp,sigs[j],rp,kind):
           score=float(cl[i,j,kind]+pl[i,j,lp]+pl[j,i,rp]);rank=score
           if noise_scale and rng is not None:rank+=noise_scale*(-math.log(-math.log(max(rng.random(),1e-12))))
           candidates.append((rank,score,i,j,kind,lp,rp))
    candidates.sort(reverse=True); parent=list(range(n)); used_atoms=set(); pairs=set(); selected=[]
    def find(x):
      while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
      return x
    def consumed(c):
      _,_,i,j,k,lp,rp=c
      keys={(i,lp),(j,rp)}
      if k==2: keys.update(((i,(lp+1)%len(sigs[i])),(j,(rp+1)%len(sigs[j]))))
      return keys
    def valid(c):
      _,_,i,j,_,_,_=c
      return (i,j) not in pairs and not bool(consumed(c)&used_atoms)
    def add(c):
      _,score,i,j,k,lp,rp=c; selected.append({'left':i,'right':j,'kind':['NONE','SINGLE','FUSED','SPIRO'][k],'left_port':lp,'right_port':rp,'score':score}); pairs.add((i,j)); used_atoms.update(consumed(c))
    for c in candidates:
      _,_,i,j,*_=c
      if find(i)!=find(j) and valid(c): add(c); parent[find(j)]=find(i)
    connected=len({find(i) for i in range(n)})==1
    extra=0
    if connected:
      for c in candidates:
       if extra>=max_extra:break
       if valid(c):add(c);extra+=1
    return selected,connected

@torch.no_grad()
def sample(checkpoint,targets,n_samples,num_rings,guidance,seed,output,connection_attempts=1,connection_noise=0.35):
    torch.manual_seed(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); state=torch.load(checkpoint,map_location=device,weights_only=False); cfg=state['config']; features=state['template_features']
    mc=dict(cfg['model']); mc.update(num_templates=features.shape[0],template_features=features.tolist()); model=RingDenoiser(**mc).to(device);model.load_state_dict(state['model']);model.eval(); schedule=LinearNoiseSchedule(**cfg['diffusion']).to(device)
    templates=sorted(state['manifest']['templates'],key=lambda x:x['id']); records=[]
    transform=state['manifest']['splits'][0].get('target_transform','log10') if False else 'log10'; epsilon=1e-6
    for target in targets:
      torch.manual_seed(seed)
      normalized=(math.log10(target+epsilon)-float(state['property_mean']))/float(state['property_std']); condition=torch.full((n_samples,1),normalized,device=device)
      mask=torch.ones((n_samples,num_rings),dtype=torch.bool,device=device); rt=torch.full((n_samples,num_rings),model.ring_mask_token,dtype=torch.long,device=device)
      pair=mask[:,:,None]&mask[:,None,:]; pair&=~torch.eye(num_rings,dtype=torch.bool,device=device)[None]; ct=torch.zeros((n_samples,num_rings,num_rings),dtype=torch.long,device=device);ct[pair]=model.connection_mask_token
      centers=center_coordinates(torch.randn(n_samples,num_rings,3,device=device),mask);normals=normalize_vectors(torch.randn(n_samples,num_rings,3,device=device),mask)
      for step in reversed(range(schedule.timesteps)):
        t=torch.full((n_samples,),step,dtype=torch.long,device=device); pred=guided(model,(rt,centers,normals,ct,mask,t),condition,guidance)
        centers=center_coordinates(schedule.p_step(centers,pred['center_noise'],t),mask);normals=normalize_vectors(schedule.p_step(normals,pred['normal_noise'],t),mask)
        rs=torch.distributions.Categorical(logits=pred['ring_logits']).sample();rt=torch.where(reveal(rt==model.ring_mask_token,step),rs,rt)
        upper=pair&torch.triu(torch.ones(num_rings,num_rings,dtype=torch.bool,device=device),1)[None];cs=torch.distributions.Categorical(logits=pred['connection_logits']).sample();rv=reveal((ct==model.connection_mask_token)&upper,step);ct=torch.where(rv,cs,ct);ct=torch.where(upper,ct,ct.transpose(1,2));ct[:,torch.arange(num_rings),torch.arange(num_rings)]=0
      pred=guided(model,(rt,centers,normals,ct,mask,torch.zeros(n_samples,dtype=torch.long,device=device)),condition,guidance)
      for b in range(n_samples):
        ids=rt[b].tolist();rings=[{'template_id':tid,'signature':templates[tid]['signature'],'center':centers[b,i].tolist(),'normal':normals[b,i].tolist()} for i,tid in enumerate(ids)]
        rng=random.Random(seed+int(round(target*1000003))+b*10007);best=None;tested=0;valid_count=0
        if connection_attempts>1:
          from export_rigid_sdf import decode
        for attempt in range(max(1,connection_attempts)):
          noise=0.0 if attempt==0 else connection_noise
          edges,connected=project(ids,pred['connection_logits'][b],pred['port_logits'][b],templates,noise_scale=noise,rng=rng)
          rec={'target_f_osc':target,'normalized_condition':normalized,'connected':connected,'rings':rings,'connections':edges}
          score=sum(e['score'] for e in edges);is_valid=False
          if connection_attempts>1 and connected:
           tested+=1
           try:decode(rec,seed,embed=False);is_valid=True;valid_count+=1
           except Exception:pass
          if best is None or (is_valid and (not best[0] or score>best[1])):best=(is_valid,score,rec,attempt)
        rec=best[2];rec['connection_search']={'attempts_requested':max(1,connection_attempts),'candidates_tested':tested,'valid_candidates':valid_count,'selected_attempt':best[3],'selected_valid':best[0]}
        records.append(rec)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w') as f:
      for r in records:f.write(json.dumps(r)+'\n')
    summary={}
    for target in targets:
      subset=[r for r in records if r['target_f_osc']==target]; tf={};cf={}
      for r in subset:
       for ring in r['rings']:tf[str(ring['template_id'])]=tf.get(str(ring['template_id']),0)+1
       for e in r['connections']:cf[e['kind']]=cf.get(e['kind'],0)+1
      summary[str(target)]={'samples':len(subset),'connected':sum(r['connected'] for r in subset),'template_counts':tf,'connection_counts':cf}
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/rigid_100k/best.pt');p.add_argument('--targets',type=float,nargs='+',default=[0.1,0.8,3.0]);p.add_argument('--n-samples',type=int,default=32);p.add_argument('--num-rings',type=int,default=13);p.add_argument('--guidance',type=float,default=1.5);p.add_argument('--seed',type=int,default=123);p.add_argument('--connection-attempts',type=int,default=1);p.add_argument('--connection-noise',type=float,default=.35);p.add_argument('--output',type=Path,default=Path('samples/rigid_100k.jsonl'));a=p.parse_args();sample(a.checkpoint,a.targets,a.n_samples,a.num_rings,a.guidance,a.seed,a.output,a.connection_attempts,a.connection_noise)
if __name__=='__main__':main()
