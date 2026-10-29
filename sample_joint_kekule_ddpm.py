#!/usr/bin/env python3
"""Reverse sample JointKekuleDDPM with strict graph and 3D evaluation."""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
from rdkit.Chem import Draw
from .model import JointKekuleDDPM,JointDiffusion,decode_kekule_graph,cfg_combine

def reveal(x,logits,t,diff,token,valid,symmetric=False):
 if not torch.isfinite(logits).all():raise FloatingPointError(f'non-finite categorical logits at t={t}')
 sample=torch.distributions.Categorical(logits=logits.float()).sample()
 if symmetric:
  n=x.shape[1];up=torch.triu(torch.ones(n,n,dtype=torch.bool,device=x.device),1)[None];sample=torch.where(up,sample,sample.transpose(1,2));sample=sample.masked_fill(torch.eye(n,dtype=torch.bool,device=x.device)[None],0)
 prev=torch.ones_like(diff.abar[t]) if t==0 else diff.abar[t-1];prob=(prev-diff.abar[t])/(1-diff.abar[t]).clamp_min(1e-8);choose=(torch.rand_like(x,dtype=torch.float32)<prob)&x.eq(token)&valid
 if symmetric:choose=choose|choose.transpose(1,2)
 return torch.where(choose,sample,x)

def reverse_coords(x,eps,t,diff,mask):
 if not torch.isfinite(x).all() or not torch.isfinite(eps).all():raise FloatingPointError(f'non-finite coordinate state at t={t}')
 beta,alpha,abar=diff.beta[t],diff.alpha[t],diff.abar[t];mean=(x-beta/(1-abar).sqrt()*eps)/alpha.sqrt()
 if t>0:
  prev=diff.abar[t-1];var=beta*(1-prev)/(1-abar);mean=mean+var.clamp_min(1e-12).sqrt()*torch.randn_like(x)
 w=mask[...,None];mean=mean*w;mean=(mean-(mean*w).sum(1,keepdim=True)/w.sum(1,keepdim=True).clamp_min(1))*w
 if not torch.isfinite(mean).all():raise FloatingPointError(f'non-finite reverse coordinate at t={t}')
 return mean

@torch.inference_mode()
def sample_batch(model,diff,y,batch,atoms,cfg):
 if batch<=0 or not 1<=atoms<=model.max_atoms:raise ValueError(f'atoms must be in [1,{model.max_atoms}] and batch must be positive')
 if not math.isfinite(cfg) or cfg<0:raise ValueError('CFG scale must be finite and non-negative')
 dev=y.device;m=torch.zeros(batch,model.max_atoms,dtype=torch.bool,device=dev);m[:,:atoms]=1;pair=m[:,:,None]&m[:,None,:];diag=torch.eye(model.max_atoms,dtype=torch.bool,device=dev)[None];state={'atom_mask':m,'atom_type':torch.full_like(m,model.atom_types,dtype=torch.long),'charge':torch.full_like(m,5,dtype=torch.long),'hydrogen':torch.full_like(m,5,dtype=torch.long),'radical':torch.full_like(m,3,dtype=torch.long),'bond':torch.full((batch,model.max_atoms,model.max_atoms),4,dtype=torch.long,device=dev),'positions':torch.randn(batch,model.max_atoms,3,device=dev)*m[...,None]};state['bond']=state['bond'].masked_fill(diag,0)
 for ti in reversed(range(diff.timesteps)):
  t=torch.full((batch,),ti,dtype=torch.long,device=dev);cond=model(y,t,state)
  if cfg!=1:out=cfg_combine(cond,model(y,t,state,unconditional=True),cfg)
  else:out=cond
  state['atom_type']=reveal(state['atom_type'],out['atom_logits'],ti,diff,model.atom_types,m);state['charge']=reveal(state['charge'],out['charge_logits'],ti,diff,5,m);state['hydrogen']=reveal(state['hydrogen'],out['hydrogen_logits'],ti,diff,5,m);state['radical']=reveal(state['radical'],out['radical_logits'],ti,diff,3,m);state['bond']=reveal(state['bond'],out['bond_logits'],ti,diff,4,pair&~diag,True);state['positions']=reverse_coords(state['positions'],out['coordinate_eps'],ti,diff,m)
 if not torch.equal(state['bond'],state['bond'].transpose(1,2)):raise RuntimeError('reverse sampler produced asymmetric bonds')
 return {k:v[:,:atoms].cpu() if v.ndim<3 else v[:,:atoms,:atoms].cpu() if k=='bond' else v[:,:atoms].cpu() for k,v in state.items() if k!='atom_mask'}

def load_db(root):
 manifest=json.loads((Path(root)/'manifest.json').read_text());known=set()
 for split in manifest['splits']:
  for rel in split['shards']:
   shard=torch.load(Path(root)/rel,map_location='cpu',weights_only=False)
   for raw in shard['smiles']:
    s=raw.decode() if isinstance(raw,bytes) else str(raw);mol=Chem.MolFromSmiles(s)
    if mol is not None:known.add(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False))
 return known

def molecule_with_coordinates(mol,positions):
 out=Chem.Mol(mol);out.RemoveAllConformers();conf=Chem.Conformer(out.GetNumAtoms());conf.Set3D(True)
 for i,(x,y,z) in enumerate(positions.tolist()):conf.SetAtomPosition(i,(float(x),float(y),float(z)))
 out.AddConformer(conf,assignId=True);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--metadata',required=True);p.add_argument('--root',default='joint_kekule_diffusion/data_qcdge_ab_aromatic');p.add_argument('--output',required=True);p.add_argument('--num',type=int,default=1000);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--atoms',type=int,default=10);p.add_argument('--allow-count-extrapolation',action='store_true');p.add_argument('--s1',type=float,default=5.2696352);p.add_argument('--t1',type=float,default=3.4800408);p.add_argument('--cfg',type=float,default=1.0);a=p.parse_args()
 if a.num<=0 or a.batch_size<=0 or not 1<=a.atoms<=32:raise ValueError('num/batch-size must be positive and atoms must be in [1,32]')
 if not all(math.isfinite(x) for x in (a.s1,a.t1,a.cfg)) or a.cfg<0:raise ValueError('conditions and CFG must be finite; CFG must be non-negative')
 RDLogger.DisableLog('rdApp.*');out=Path(a.output);out.mkdir(parents=True,exist_ok=True);meta=json.loads(Path(a.metadata).read_text());fixed=meta.get('fixed_atom_count',[10])
 if fixed and a.atoms not in fixed and not a.allow_count_extrapolation:raise ValueError(f'atoms={a.atoms} is outside trained counts {fixed}; pass --allow-count-extrapolation explicitly')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);args=meta['args'];required=('s1_rbf_bins','t1_rbf_bins','condition_rbf_min','condition_rbf_max','condition_rbf_width_scale')
 if not all(k in args for k in required):raise ValueError('checkpoint metadata predates configurable scalar RBF conditioning and is incompatible with this sampler')
 model=JointKekuleDDPM(args['hidden'],32,len(meta['atom_map']),args['layers'],args.get('condition_dropout',.1),args['s1_rbf_bins'],args['t1_rbf_bins'],args['condition_rbf_min'],args['condition_rbf_max'],args['condition_rbf_width_scale']);model.load_state_dict(ck['model']);model.cuda().eval();diff=JointDiffusion().cuda();st=meta['condition_stats'];yn=torch.tensor([[(a.s1-st['s1_energy_mean'])/st['s1_energy_std'],(a.t1-st['t1_energy_mean'])/st['t1_energy_std']]],device='cuda');known=load_db(a.root);records=[];unique_mols=[];unique_smiles=set();start=time.time()
 for begin in range(0,a.num,a.batch_size):
  bs=min(a.batch_size,a.num-begin);sample=sample_batch(model,diff,yn.expand(bs,-1),bs,a.atoms,a.cfg)
  for j in range(bs):
   r=decode_kekule_graph(sample['atom_type'][j].tolist(),sample['charge'][j].tolist(),sample['hydrogen'][j].tolist(),sample['radical'][j].tolist(),sample['bond'][j],meta['atom_map']);novel=bool(r.smiles and r.smiles not in known);unique=bool(novel and r.connected and r.smiles not in unique_smiles)
   records.append({'index':begin+j,'smiles':r.smiles or '','valid':int(r.valid),'connected':int(r.connected),'aromatic_rings':r.aromatic_rings,'novel_vs_db':int(novel),'unique_novel_connected':int(unique),'error':r.error or ''})
   if unique:unique_smiles.add(r.smiles);unique_mols.append(molecule_with_coordinates(r.molecule,sample['positions'][j]))
  with (out/'molecules.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
  summary={'requested':a.num,'completed':len(records),'valid':sum(x['valid'] for x in records),'connected':sum(x['connected'] for x in records),'aromatic':sum(x['aromatic_rings']>0 for x in records),'novel_vs_db':sum(x['novel_vs_db'] for x in records),'unique_novel_connected':len(unique_mols),'checkpoint_epoch':ck['epoch'],'s1':a.s1,'t1':a.t1,'cfg':a.cfg,'atoms':a.atoms,'elapsed_s':time.time()-start};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
 writer=Chem.SDWriter(str(out/'unique_novel_connected.sdf'))
 for i,mol in enumerate(unique_mols):mol.SetProp('_Name',f'generated_{i:06d}');mol.SetProp('SMILES',Chem.MolToSmiles(mol));writer.write(mol)
 writer.close()
 for page in range(0,len(unique_mols),100):Draw.MolsToGridImage(unique_mols[page:page+100],molsPerRow=10,subImgSize=(250,200),legends=[Chem.MolToSmiles(x) for x in unique_mols[page:page+100]]).save(str(out/f'novel_valid_{page//100:03d}.png'))

if __name__=='__main__':main()
