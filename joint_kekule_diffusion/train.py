#!/usr/bin/env python3
"""Production trainer for the consistency-corrected joint Kekule DDPM."""
from __future__ import annotations
import argparse,json,os,random,time
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import QCDGEHierarchyDataset,collate_hierarchy
from .hp1kg_data import HP1KGDataset,collate_hp1kg
from .model import JointDiffusion,JointKekuleDDPM,joint_training_loss,decode_kekule_graph
from .sample import sample_batch

TRAIN_KEYS={'condition','ring_mask','atom_mask','atom_type','atom_charge','atom_hydrogen','atom_radical','atom_positions','atom_bonds','aromatic_cycle_edges','aromatic_cycle_double_count','aromatic_cycle_order','aromatic_cycle_size','aromatic_edge_mask'}

def collate_joint(rows):
 out=collate_hierarchy(rows,max_rings=12,max_slots=16,max_atoms=32);B,R,A=len(rows),12,32
 cycles=torch.zeros(B,R,A,A,dtype=torch.bool);doubles=torch.zeros(B,R);order=torch.full((B,R,A),-1,dtype=torch.long);sizes=torch.zeros(B,R,dtype=torch.long);aromatic_edges=torch.zeros(B,A,A,dtype=torch.bool)
 for b,row in enumerate(rows):
  n=len(row['atom_types']);aromatic_edges[b,:n,:n]=row['atom_bond_aromatic']
  for r,ring in enumerate(row['ordered_rings']):
   if not bool(row['cell_aromatic'][r]) or any(int(row['kekule_bonds'][ring[i],ring[(i+1)%len(ring)]]) not in (1,2) for i in range(len(ring))):continue
   sizes[b,r]=len(ring);order[b,r,:len(ring)]=torch.tensor(ring)
   for i,u in enumerate(ring):
    v=ring[(i+1)%len(ring)];cycles[b,r,u,v]=cycles[b,r,v,u]=1
   doubles[b,r]=(((row['kekule_bonds']==2)&cycles[b,r,:n,:n]).sum()/2)
 out.update(aromatic_cycle_edges=cycles,aromatic_cycle_double_count=doubles,aromatic_cycle_order=order,aromatic_cycle_size=sizes,aromatic_edge_mask=aromatic_edges)
 return {k:v for k,v in out.items() if k in TRAIN_KEYS}

def compute_marginals(dataset):
 counts={"atom":torch.zeros(len(dataset.atom_map)),"charge":torch.zeros(5),"hydrogen":torch.zeros(5),"radical":torch.zeros(3),"edge":torch.zeros(2),"bond_type":torch.zeros(3)}
 for i in range(len(dataset)):
  x=dataset[i];atom=x["atom_types"].long();charge=(x["atom_charges"].long()+2).clamp(0,4);hydrogen=x["atom_hydrogens"].long().clamp(0,4);radical=x["atom_radicals"].long().clamp(0,2);bonds=x["kekule_bonds"].long();n=len(atom);upper=torch.triu(torch.ones(n,n,dtype=torch.bool),1);present=bonds[upper]>0
  counts["atom"]+=torch.bincount(atom,minlength=len(dataset.atom_map));counts["charge"]+=torch.bincount(charge,minlength=5);counts["hydrogen"]+=torch.bincount(hydrogen,minlength=5);counts["radical"]+=torch.bincount(radical,minlength=3);counts["edge"]+=torch.tensor([int((~present).sum()),int(present.sum())]);counts["bond_type"]+=torch.bincount((bonds[upper][present]-1).clamp_min(0),minlength=3)
 return {k:(v+1e-6).div((v+1e-6).sum()).tolist() for k,v in counts.items()}

def device_batch(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

@torch.no_grad()
def validate(model,diff,loader,device,max_batches=8,seed=1701):
 model.eval();tot={};count=0;probe=None;cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()]
 with torch.random.fork_rng(devices=cuda_devices):
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  for batch in loader:
   batch=device_batch(batch,device);probe=batch if probe is None else probe;loss,terms=joint_training_loss(model,diff,batch)
   if not torch.isfinite(loss):model.train();return {"loss":float("nan")}
   for k,v in terms.items():tot[k]=tot.get(k,0.)+float(v)
   count+=1
   if count>=max_batches:break
  torch.manual_seed(seed+1);torch.cuda.manual_seed_all(seed+1);correct=float(joint_training_loss(model,diff,probe)[0]);torch.manual_seed(seed+1);torch.cuda.manual_seed_all(seed+1);shuffled={**probe,"condition":probe["condition"].roll(1,0)};shuffle=float(joint_training_loss(model,diff,shuffled)[0])
 model.train();tot={k:v/max(count,1) for k,v in tot.items()};tot["loss"]=sum(tot.values());tot["condition_shuffle_delta"]=shuffle-correct;return tot

@torch.no_grad()
def rollout_validate(model,diff,loader,device,atom_map,samples=4,observed_counts=None,coordinate_clip=25.):
 from rdkit import Chem
 model.eval();batch=device_batch(next(iter(loader)),device);k=min(samples,len(batch["condition"]));limits=[{"B":3,"C":4,"N":4,"O":2,"F":1,"Si":4,"P":5,"S":6,"Cl":1,"As":5,"Se":6,"Br":1}.get(x,4) for x in atom_map];out=sample_batch(model,diff,batch["condition"][:k],k,0,1.,limits,atom_map,observed_counts,coordinate_clip);valid=connected=aromatic=violations=0;edges=[];bond_lengths=[];clashes=[]
 for i in range(k):
  n=int(out["atom_count"][i]);b=out["bond"][i,:n,:n];p=out["positions"][i,:n];r=decode_kekule_graph(out["atom_type"][i,:n].tolist(),out["charge"][i,:n].tolist(),out["hydrogen"][i,:n].tolist(),out["radical"][i,:n].tolist(),b,atom_map);valid+=int(r.valid);connected+=int(r.connected);aromatic+=int(r.aromatic_rings>0);edges.append(float((b>0).sum()/2));lim=torch.tensor(limits)[out["atom_type"][i,:n]];violations+=int(bool((b.sum(-1)+out["hydrogen"][i,:n]>lim).any()));d=torch.cdist(p,p);upper=torch.triu(torch.ones(n,n,dtype=torch.bool),1);bond=(b>0)&upper;nonbond=(b==0)&upper
  if bond.any():bond_lengths.append(float(d[bond].mean()))
  if nonbond.any():clashes.append(float((d[nonbond]<.75).float().mean()))
 model.train();return {"samples":k,"valid_rate":valid/k,"connected_rate":connected/k,"aromatic_rate":aromatic/k,"valence_violation_rate":violations/k,"mean_edges":sum(edges)/k,"mean_bond_length":sum(bond_lengths)/max(len(bond_lengths),1),"clash_fraction":sum(clashes)/max(len(clashes),1)}

def safe_rollout_validate(model,*args,**kwargs):
 try:return rollout_validate(model,*args,**kwargs)
 except Exception as exc:model.train();return {"failed":True,"error":f"{type(exc).__name__}: {exc}"}

def atomic_save(payload,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='joint_kekule_diffusion/data_qcdge_ab_aromatic');p.add_argument('--run-dir',required=True);p.add_argument('--epochs',type=int,default=10000);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--workers',type=int,default=4);p.add_argument('--hidden',type=int,default=256);p.add_argument('--layers',type=int,default=8);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--grad-clip',type=float,default=1.0);p.add_argument('--save-every',type=int,default=10);p.add_argument('--resume');p.add_argument('--seed',type=int,default=61)
 p.add_argument('--s1-rbf-bins',type=int,default=24,help='RBF features for the single normalized S1 scalar')
 p.add_argument('--t1-rbf-bins',type=int,default=24,help='RBF features for the single normalized T1 scalar')
 p.add_argument('--condition-rbf-min',type=float,default=-5.,help='lower RBF center in normalized condition units')
 p.add_argument('--condition-rbf-max',type=float,default=5.,help='upper RBF center in normalized condition units')
 p.add_argument('--condition-rbf-width-scale',type=float,default=1.,help='Gaussian width as a multiple of center spacing')
 p.add_argument('--condition-dropout',type=float,default=.1,help='CFG unconditional training probability')
 p.add_argument('--dataset-kind',choices=('auto','qcdge','hp1kg'),default='auto')
 p.add_argument('--max-atoms',type=int,default=0,help='0 infers the dataset maximum')
 p.add_argument('--rollout-every',type=int,default=10,help='run full reverse-diffusion validation every N epochs')
 p.add_argument('--rollout-samples',type=int,default=4,help='molecules generated by each rollout validation')
 p.add_argument('--coordinate-clip',type=float,default=25.,help='maximum centered coordinate radius during reverse sampling');a=p.parse_args()
 if a.epochs<=0 or a.batch_size<=0 or a.workers<0 or a.hidden%2 or a.s1_rbf_bins<2 or a.t1_rbf_bins<2 or not 0<=a.condition_dropout<1 or a.rollout_every<=0 or a.rollout_samples<=0 or not np.isfinite(a.coordinate_clip) or a.coordinate_clip<=0:raise ValueError('invalid training dimensions or condition configuration')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=True
 device=torch.device('cuda');run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
 manifest=json.loads((Path(a.root)/'manifest.json').read_text());kind=a.dataset_kind
 if kind=='auto':kind='hp1kg' if manifest.get('dataset_type')=='hp1kg_nevpt2' else 'qcdge'
 dataset_cls=HP1KGDataset if kind=='hp1kg' else QCDGEHierarchyDataset
 train=dataset_cls(a.root,'train');val=dataset_cls(a.root,'val',condition_stats=train.condition_stats,atom_map=train.atom_map)
 count_at=train.heavy_count_at if hasattr(train,'heavy_count_at') else train.base.heavy_count_at
 observed_max=max(count_at(i) for i in range(len(train)));max_atoms=a.max_atoms or (manifest.get('heavy_atom_max',observed_max) if kind=='hp1kg' else 32)
 if max_atoms<observed_max:raise ValueError(f'max_atoms={max_atoms} below observed train maximum {observed_max}')
 collate=partial(collate_hp1kg,max_atoms=max_atoms) if kind=='hp1kg' else collate_joint
 loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,collate_fn=collate,drop_last=False)
 vloader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=max(1,a.workers//2),pin_memory=True,persistent_workers=a.workers>0,collate_fn=collate)
 marginals=compute_marginals(train);model=JointKekuleDDPM(a.hidden,max_atoms,len(train.atom_map),a.layers,a.condition_dropout,a.s1_rbf_bins,a.t1_rbf_bins,a.condition_rbf_min,a.condition_rbf_max,a.condition_rbf_width_scale).to(device);diff=JointDiffusion(marginals=marginals).to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);start=0;step=0
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False);old=ck.get('metadata',{});expected='marginal_factorized_chem_d3pm_v4'
  if old.get('model_kind')!=expected or old.get('diffusion_marginals')!=marginals:raise ValueError('resume checkpoint is incompatible with v4 chemistry-stable diffusion')
  model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);start=ck['epoch']+1;step=ck['step']
  if 'rng' in ck:
   random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda'])
 meta={'schema_version':6,'model_kind':'marginal_factorized_chem_d3pm_v4','diffusion_marginals':marginals,'args':vars(a),'dataset_kind':kind,'max_atoms':max_atoms,'condition_stats':train.condition_stats,'atom_map':train.atom_map,'train_size':len(train),'val_size':len(val),'pid':os.getpid(),'fixed_atom_count':sorted({count_at(i) for i in range(len(train))})};(run/'metadata.json').write_text(json.dumps(meta,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
 model.train();started=time.time()
 for epoch in range(start,a.epochs):
  sums={};seen=0
  for batch in loader:
   batch=device_batch(batch,device);opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms=joint_training_loss(model,diff,batch)
   if not torch.isfinite(loss):
    atomic_save({'epoch':epoch,'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'batch':{k:v.cpu() for k,v in batch.items()}},run/'nan_checkpoint.pt');raise FloatingPointError(f'non-finite loss epoch={epoch} step={step}')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip)
   if not torch.isfinite(norm):raise FloatingPointError(f'non-finite grad epoch={epoch} step={step}')
   opt.step();step+=1;seen+=1
   for k,v in terms.items():sums[k]=sums.get(k,0.)+float(v.detach())
   if step%20==0:print(json.dumps({'event':'step','epoch':epoch,'step':step,'loss':float(loss),'grad_norm':float(norm),'gpu_mem_gb':torch.cuda.max_memory_allocated()/2**30,'elapsed_s':time.time()-started}),flush=True)
  sched.step();train_metrics={k:v/max(seen,1) for k,v in sums.items()};val_metrics=validate(model,diff,vloader,device);rollout=None
  if (epoch+1)%a.rollout_every==0:rollout=safe_rollout_validate(model,diff,vloader,device,train.atom_map,a.rollout_samples,meta["fixed_atom_count"],a.coordinate_clip)
  metrics={"event":"epoch","epoch":epoch,"step":step,"train":train_metrics,"val":val_metrics,"rollout":rollout,"lr":sched.get_last_lr()[0],"elapsed_s":time.time()-started};print(json.dumps(metrics),flush=True)
  payload={'epoch':epoch,'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metadata':meta,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
  atomic_save(payload,run/'checkpoints'/'latest.pt')
  if (epoch+1)%a.save_every==0:atomic_save(payload,run/'checkpoints'/f'epoch_{epoch+1:05d}.pt')

if __name__=='__main__':main()
