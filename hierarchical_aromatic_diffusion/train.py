#!/usr/bin/env python3
"""Joint masked-diffusion trainer; supports a cheap structural smoke run."""
import argparse,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from hierarchical_aromatic_diffusion.data import QCDGEHierarchyDataset,collate_hierarchy
from hierarchical_aromatic_diffusion.diffusion import HierarchicalDiffusion
from hierarchical_aromatic_diffusion.model import HierarchicalAromaticModel

def ce(logits,target,mask):return F.cross_entropy(logits[mask],target[mask]) if mask.any() else logits.sum()*0
def step(batch,model,diff,device):
 x={k:v.to(device) for k,v in batch.items()};b=len(x['condition']);t=torch.randint(diff.timesteps,(b,),device=device);rm=x['ring_mask'];sm=x['slot_mask'];subm=x['sub_mask'];rp,reps=diff.q_coords(x['ring_positions'],t,rm)
 relmask=rm[:,:,None]&rm[:,None,:];top={'size':diff.q_tokens((x['ring_size']-3).clamp(0,13),t,14,rm),'aromatic':diff.q_tokens(x['ring_aromatic'],t,2,rm),'relation':diff.q_tokens(x['ring_relation'],t,13,relmask),'positions':rp,'mask':rm}
 sc={'atom':diff.q_tokens(x['slot_atom'],t,4,sm),'charge':diff.q_tokens(x['slot_charge'],t,5,sm),'hydrogen':diff.q_tokens(x['slot_hydrogen'],t,5,sm),'radical':diff.q_tokens(x['slot_radical'],t,3,sm),'positions':x['slot_positions'],'mask':sm};sub={'atom':diff.q_tokens(x['sub_atom'],t,4,subm),'mask':subm}
 pos=torch.cat((x['scaffold_positions'],x['sub_positions']),1);cm=torch.cat((x['scaffold_mask'],subm),1);noisy,eps=diff.q_coords(pos,t,cm);coord={'positions':noisy,'mask':cm,'slot_to_scaffold':x['slot_to_scaffold'],'scaffold_count':x['scaffold_positions'].shape[1]};o=model(x['condition'],t,top,sc,sub,coord)
 upper=relmask&torch.triu(torch.ones(relmask.shape[1:],dtype=torch.bool,device=device),1);cross=sm[:,:,None,:,None]&sm[:,None,:,None,:];attach=subm[:,:,None]&sm.reshape(b,1,-1)
 terms={'ring_count':F.cross_entropy(o['ring_count_logits'],rm.sum(1)),'ring_size':ce(o['ring_size_logits'],(x['ring_size']-3).clamp(0,13),rm),'ring_aromatic':ce(o['ring_aromatic_logits'],x['ring_aromatic'],rm),'relation':ce(o['relation_logits'],x['ring_relation'],upper),'junction':ce(o['junction_slot_logits'],x['junction'],cross),'slot_atom':ce(o['slot_atom_logits'],x['slot_atom'],sm),'slot_charge':ce(o['slot_charge_logits'],x['slot_charge'],sm),'slot_hydrogen':ce(o['slot_hydrogen_logits'],x['slot_hydrogen'],sm),'slot_radical':ce(o['slot_radical_logits'],x['slot_radical'],sm),'sub_count':F.cross_entropy(o['substituent_count_logits'],subm.sum(1)),'sub_atom':ce(o['substituent_atom_logits'],x['sub_atom'],subm),'attachment':ce(o['attachment_logits'],x['attachment'],attach),'ring_eps':((o['ring_eps']-reps).square()*rm[...,None]).sum()/rm.sum().clamp_min(1),'coordinate_eps':((o['coordinate_eps']-eps).square()*cm[...,None]).sum()/cm.sum().clamp_min(1)}
 return sum(terms.values()),{k:float(v.detach()) for k,v in terms.items()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='data_qcdge_a10_full');p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--steps',type=int,default=1);p.add_argument('--hidden',type=int,default=256);a=p.parse_args();device=torch.device(a.device);d=QCDGEHierarchyDataset(a.root,'train');loader=DataLoader(d,batch_size=a.batch_size,shuffle=True,collate_fn=collate_hierarchy);m=HierarchicalAromaticModel(a.hidden).to(device);diff=HierarchicalDiffusion().to(device);opt=torch.optim.AdamW(m.parameters(),1e-4)
 for i,b in zip(range(a.steps),loader):loss,terms=step(b,m,diff,device);assert torch.isfinite(loss);opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm);opt.step();print({'step':i,'loss':float(loss),'grad_norm':float(norm),'terms':terms},flush=True)
if __name__=='__main__':main()
