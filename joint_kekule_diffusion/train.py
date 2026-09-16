#!/usr/bin/env python3
"""Production trainer for the consistency-corrected joint Kekule DDPM."""
from __future__ import annotations
import argparse,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import QCDGEHierarchyDataset,collate_hierarchy
from .model import JointDiffusion,JointKekuleDDPM,joint_training_loss

TRAIN_KEYS={'condition','ring_mask','atom_mask','atom_type','atom_charge','atom_hydrogen','atom_radical','atom_positions','atom_bonds','aromatic_cycle_edges','aromatic_cycle_double_count','aromatic_cycle_order','aromatic_cycle_size','aromatic_edge_mask'}

def collate_joint(rows):
 out=collate_hierarchy(rows,max_rings=12,max_slots=16,max_atoms=32);B,R,A=len(rows),12,32
 cycles=torch.zeros(B,R,A,A,dtype=torch.bool);doubles=torch.zeros(B,R);order=torch.full((B,R,A),-1,dtype=torch.long);sizes=torch.zeros(B,R,dtype=torch.long);aromatic_edges=torch.zeros(B,A,A,dtype=torch.bool)
 for b,row in enumerate(rows):
  n=len(row['atom_types']);aromatic_edges[b,:n,:n]=row['atom_bond_aromatic']
  for r,ring in enumerate(row['ordered_rings']):
   if not bool(row['cell_aromatic'][r]):continue
   sizes[b,r]=len(ring);order[b,r,:len(ring)]=torch.tensor(ring)
   for i,u in enumerate(ring):
    v=ring[(i+1)%len(ring)];cycles[b,r,u,v]=cycles[b,r,v,u]=1
   doubles[b,r]=(((row['kekule_bonds']==2)&cycles[b,r,:n,:n]).sum()/2)
 out.update(aromatic_cycle_edges=cycles,aromatic_cycle_double_count=doubles,aromatic_cycle_order=order,aromatic_cycle_size=sizes,aromatic_edge_mask=aromatic_edges)
 return {k:v for k,v in out.items() if k in TRAIN_KEYS}

def device_batch(batch,device):return {k:v.to(device,non_blocking=True) for k,v in batch.items()}

@torch.no_grad()
def validate(model,diff,loader,device,max_batches=8,seed=1701):
 model.eval();tot={};count=0;cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()]
 with torch.random.fork_rng(devices=cuda_devices):
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
  for batch in loader:
   loss,terms=joint_training_loss(model,diff,device_batch(batch,device))
   if not torch.isfinite(loss):model.train();return {'loss':float('nan')}
   for k,v in terms.items():tot[k]=tot.get(k,0.)+float(v)
   count+=1
   if count>=max_batches:break
 model.train();tot={k:v/max(count,1) for k,v in tot.items()};tot['loss']=sum(tot.values());return tot

def atomic_save(payload,path):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,tmp);os.replace(tmp,path)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='joint_kekule_diffusion/data_qcdge_ab_aromatic');p.add_argument('--run-dir',required=True);p.add_argument('--epochs',type=int,default=10000);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--workers',type=int,default=4);p.add_argument('--hidden',type=int,default=256);p.add_argument('--layers',type=int,default=8);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--grad-clip',type=float,default=1.0);p.add_argument('--save-every',type=int,default=10);p.add_argument('--resume');p.add_argument('--seed',type=int,default=61);a=p.parse_args()
 if a.epochs<=0 or a.batch_size<=0 or a.workers<0 or a.hidden%2:raise ValueError('invalid training dimensions')
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);torch.backends.cuda.matmul.allow_tf32=True
 device=torch.device('cuda');run=Path(a.run_dir);run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
 train=QCDGEHierarchyDataset(a.root,'train');val=QCDGEHierarchyDataset(a.root,'val',condition_stats=train.condition_stats,atom_map=train.atom_map)
 loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,collate_fn=collate_joint,drop_last=False)
 vloader=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=max(1,a.workers//2),pin_memory=True,persistent_workers=a.workers>0,collate_fn=collate_joint)
 model=JointKekuleDDPM(a.hidden,32,len(train.atom_map),a.layers,.1).to(device);diff=JointDiffusion().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs);start=0;step=0
 if a.resume:
  ck=torch.load(a.resume,map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);start=ck['epoch']+1;step=ck['step']
  if 'rng' in ck:
   random.setstate(ck['rng']['python']);np.random.set_state(ck['rng']['numpy']);torch.set_rng_state(ck['rng']['torch']);torch.cuda.set_rng_state_all(ck['rng']['cuda'])
 meta={'schema_version':2,'args':vars(a),'condition_stats':train.condition_stats,'atom_map':train.atom_map,'train_size':len(train),'val_size':len(val),'pid':os.getpid(),'fixed_atom_count':sorted({train.base.heavy_count_at(i) for i in range(len(train))})};(run/'metadata.json').write_text(json.dumps(meta,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
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
  sched.step();train_metrics={k:v/max(seen,1) for k,v in sums.items()};metrics={'event':'epoch','epoch':epoch,'step':step,'train':train_metrics,'val':validate(model,diff,vloader,device),'lr':sched.get_last_lr()[0],'elapsed_s':time.time()-started};print(json.dumps(metrics),flush=True)
  payload={'epoch':epoch,'step':step,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'metadata':meta,'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}}
  atomic_save(payload,run/'checkpoints'/'latest.pt')
  if (epoch+1)%a.save_every==0:atomic_save(payload,run/'checkpoints'/f'epoch_{epoch+1:05d}.pt')

if __name__=='__main__':main()
