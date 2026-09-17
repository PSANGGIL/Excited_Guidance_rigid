#!/usr/bin/env python3
"""Reverse sample JointKekuleDDPM with strict graph and 3D evaluation."""
from __future__ import annotations
import argparse,csv,json,math,time
from pathlib import Path
import torch
import networkx as nx
from rdkit import Chem,RDLogger
from rdkit.Chem import Draw
from .model import JointKekuleDDPM,JointDiffusion,decode_kekule_graph,cfg_combine

def posterior_step(x,x0_logits,t,diff,classes,valid,symmetric=False,name="atom"):
 if not torch.isfinite(x0_logits).all():raise FloatingPointError(f"non-finite categorical logits at t={t}")
 logits=diff.posterior_logits(x,x0_logits,t,classes,name);sample=torch.distributions.Categorical(logits=logits).sample()
 if symmetric:
  n=x.shape[1];up=torch.triu(torch.ones(n,n,dtype=torch.bool,device=x.device),1)[None];sample=torch.where(up,sample,sample.transpose(1,2));sample=sample.masked_fill(torch.eye(n,dtype=torch.bool,device=x.device)[None],0)
 return torch.where(valid,sample,x)

def reverse_coords(x,eps,t,diff,mask):
 if not torch.isfinite(x).all() or not torch.isfinite(eps).all():raise FloatingPointError(f'non-finite coordinate state at t={t}')
 beta,alpha,abar=diff.beta[t],diff.alpha[t],diff.abar[t];mean=(x-beta/(1-abar).sqrt()*eps)/alpha.sqrt()
 if t>0:
  prev=diff.abar[t-1];var=beta*(1-prev)/(1-abar);mean=mean+var.clamp_min(1e-12).sqrt()*torch.randn_like(x)
 w=mask[...,None];mean=mean*w;mean=(mean-(mean*w).sum(1,keepdim=True)/w.sum(1,keepdim=True).clamp_min(1))*w
 if not torch.isfinite(mean).all():raise FloatingPointError(f'non-finite reverse coordinate at t={t}')
 return mean
def chemistry_projection(state,outputs,mask,valence_limits,force_connectivity=False,project_aromatic=False):
 """Greedy valence-safe graph projection; final mode builds a spanning forest and one global Kekule matching per aromatic component."""
 edge_prob=outputs["edge_logits"].float().softmax(-1)[...,1].detach().cpu();type_prob=outputs["bond_type_logits"].float().softmax(-1).detach().cpu();arom_prob=outputs["aromatic_edge_logits"].float().sigmoid().detach().cpu();edge=state["edge_exists"].detach().cpu();kind=state["bond_type"].detach().cpu();atoms=state["atom_type"].detach().cpu();hydrogen=state["hydrogen"].detach().cpu();limits=torch.as_tensor(valence_limits,dtype=torch.long)
 for b in range(len(mask)):
  n=int(mask[b].sum());capacity=limits[atoms[b,:n]].clamp_min(0).tolist();allpairs=[]
  for i in range(n):
   for j in range(i+1,n):
    order=int(kind[b,i,j])+1;score=float(edge_prob[b,i,j]*type_prob[b,i,j,order-1]);allpairs.append((score,i,j,order,bool(edge[b,i,j])))
  selected=[];used=[0]*n;parent=list(range(n))
  def find(x):
   while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
   return x
  def add(item):
   score,i,j,order,present=item
   if used[i]+order>capacity[i] or used[j]+order>capacity[j]:return False
   selected.append(item);used[i]+=order;used[j]+=order;ri,rj=find(i),find(j)
   if ri!=rj:parent[ri]=rj
   return True
  ranked=sorted(allpairs,reverse=True)
  if force_connectivity:
   for item in ranked:
    if find(item[1])!=find(item[2]):add((item[0],item[1],item[2],1,item[4]))
  for item in ranked:
   if item[4] and item not in selected:add(item)
  edge[b,:n,:n]=0;kind[b,:n,:n]=0
  for score,i,j,order,present in selected:edge[b,i,j]=edge[b,j,i]=1;kind[b,i,j]=kind[b,j,i]=order-1
  if project_aromatic:
   g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from((i,j) for _,i,j,_,_ in selected if arom_prob[b,i,j]>.5)
   for comp in nx.connected_components(g):
    sub=g.subgraph(comp)
    if sub.number_of_edges()<sub.number_of_nodes():continue
    for i,j in sub.edges:kind[b,i,j]=kind[b,j,i]=0
    room=[capacity[i]-sum(int(kind[b,i,j])+1 for j in range(n) if edge[b,i,j]) for i in range(n)]
    mg=nx.Graph();mg.add_nodes_from(i for i in comp if room[i]>0);mg.add_weighted_edges_from((i,j,float(type_prob[b,i,j,1])) for i,j in sub.edges if room[i]>0 and room[j]>0)
    for i,j in nx.max_weight_matching(mg,maxcardinality=True):kind[b,i,j]=kind[b,j,i]=1
  hydrogen[b,:n]=torch.minimum(hydrogen[b,:n],limits[atoms[b,:n]]-torch.tensor([sum(int(kind[b,i,j])+1 for j in range(n) if edge[b,i,j]) for i in range(n)]))
 state["edge_exists"]=edge.to(mask.device);state["bond_type"]=kind.to(mask.device);state["hydrogen"]=hydrogen.to(mask.device);return state


@torch.inference_mode()
def sample_batch(model,diff,y,batch,atoms,cfg,valence_limits=None):
 if batch<=0 or not 0<=atoms<=model.max_atoms:raise ValueError(f"atoms must be 0 or in [1,{model.max_atoms}]")
 if not math.isfinite(cfg) or cfg<0:raise ValueError("CFG scale must be finite and non-negative")
 dev=y.device;tmax=torch.full((batch,),diff.timesteps-1,dtype=torch.long,device=dev);cc=model.count(model.condition_context(y));cu=model.count(model.condition_context(y,unconditional=True));count_logits=cu+cfg*(cc-cu) if cfg!=1 else cc
 counts=torch.full((batch,),atoms,dtype=torch.long,device=dev) if atoms else torch.distributions.Categorical(logits=count_logits).sample()+1
 idx=torch.arange(model.max_atoms,device=dev);m=idx[None]<counts[:,None];pair=m[:,:,None]&m[:,None,:];diag=torch.eye(model.max_atoms,dtype=torch.bool,device=dev)[None];upper=pair&torch.triu(torch.ones(model.max_atoms,model.max_atoms,dtype=torch.bool,device=dev),1)[None]
 def symmetric_marginal(name,classes):
  prob=diff.marginal(name,classes,dev);z=torch.multinomial(prob,batch*model.max_atoms*model.max_atoms,replacement=True).reshape(batch,model.max_atoms,model.max_atoms);return torch.where(torch.triu(torch.ones_like(z,dtype=torch.bool),1),z,z.transpose(1,2)).masked_fill(diag,0)
 def node_marginal(name,classes):return torch.multinomial(diff.marginal(name,classes,dev),batch*model.max_atoms,replacement=True).reshape(batch,model.max_atoms)
 state={"atom_mask":m,"atom_type":node_marginal("atom",model.atom_types),"charge":node_marginal("charge",5),"hydrogen":node_marginal("hydrogen",5),"radical":node_marginal("radical",3),"edge_exists":symmetric_marginal("edge",2),"bond_type":symmetric_marginal("bond_type",3),"positions":torch.randn(batch,model.max_atoms,3,device=dev)*m[...,None]};state["positions"]-=((state["positions"]*m[...,None]).sum(1,keepdim=True)/m.sum(1,keepdim=True).clamp_min(1)[...,None])*m[...,None];valence_limits=valence_limits or [4]*model.atom_types
 for ti in reversed(range(diff.timesteps)):
  t=torch.full((batch,),ti,dtype=torch.long,device=dev);cond=model(y,t,state);out=cfg_combine(cond,model(y,t,state,unconditional=True),cfg) if cfg!=1 else cond
  state["atom_type"]=posterior_step(state["atom_type"],out["atom_logits"],ti,diff,model.atom_types,m,False,"atom");state["charge"]=posterior_step(state["charge"],out["charge_logits"],ti,diff,5,m,False,"charge");state["hydrogen"]=posterior_step(state["hydrogen"],out["hydrogen_logits"],ti,diff,5,m,False,"hydrogen");state["radical"]=posterior_step(state["radical"],out["radical_logits"],ti,diff,3,m,False,"radical");state["edge_exists"]=posterior_step(state["edge_exists"],out["edge_logits"],ti,diff,2,pair&~diag,True,"edge");state["bond_type"]=posterior_step(state["bond_type"],out["bond_type_logits"],ti,diff,3,pair&~diag,True,"bond_type");state["positions"]=reverse_coords(state["positions"],out["coordinate_eps"],ti,diff,m)
  if ti in (9,4,0):state=chemistry_projection(state,out,m,valence_limits,force_connectivity=ti==0,project_aromatic=ti==0)
 bonds=state["edge_exists"]*(state["bond_type"]+1);bonds=bonds.masked_fill(~pair|diag,0)
 if not torch.equal(bonds,bonds.transpose(1,2)):raise RuntimeError("reverse sampler produced asymmetric bonds")
 return {"atom_count":counts.cpu(),"atom_type":state["atom_type"].cpu(),"charge":state["charge"].cpu(),"hydrogen":state["hydrogen"].cpu(),"radical":state["radical"].cpu(),"bond":bonds.cpu(),"positions":state["positions"].cpu()}

def load_db(root):
 root=Path(root);manifest=json.loads((root/"manifest.json").read_text());known=set()
 if manifest.get("dataset_type")=="hp1kg_nevpt2":
  for split in ("train","val","test"):
   for row in torch.load(root/f"{split}.pt",map_location="cpu",weights_only=False):known.add(row["smiles"])
  return known
 for split in manifest["splits"]:
  for rel in split["shards"]:
   shard=torch.load(root/rel,map_location="cpu",weights_only=False)
   for raw in shard["smiles"]:
    text=raw.decode() if isinstance(raw,bytes) else str(raw);mol=Chem.MolFromSmiles(text)
    if mol is not None:known.add(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False))
 return known

def molecule_with_coordinates(mol,positions):
 out=Chem.Mol(mol);out.RemoveAllConformers();conf=Chem.Conformer(out.GetNumAtoms());conf.Set3D(True)
 for i,(x,y,z) in enumerate(positions.tolist()):conf.SetAtomPosition(i,(float(x),float(y),float(z)))
 out.AddConformer(conf,assignId=True);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--metadata',required=True);p.add_argument('--root',default='joint_kekule_diffusion/data_qcdge_ab_aromatic');p.add_argument('--output',required=True);p.add_argument('--num',type=int,default=1000);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--atoms',type=int,default=0,help='0 samples molecular size from the conditional count head');p.add_argument('--allow-count-extrapolation',action='store_true');p.add_argument('--s1',type=float,default=5.2696352);p.add_argument('--t1',type=float,default=3.4800408);p.add_argument('--cfg',type=float,default=1.0);a=p.parse_args()
 if a.num<=0 or a.batch_size<=0 or a.atoms<0:raise ValueError('num and batch-size must be positive; atoms must be non-negative')
 if not all(math.isfinite(x) for x in (a.s1,a.t1,a.cfg)) or a.cfg<0:raise ValueError('conditions and CFG must be finite; CFG must be non-negative')
 RDLogger.DisableLog('rdApp.*');out=Path(a.output);out.mkdir(parents=True,exist_ok=True);meta=json.loads(Path(a.metadata).read_text());fixed=meta.get('fixed_atom_count',[10]);max_atoms=int(meta.get('max_atoms',32))
 if a.atoms>max_atoms:raise ValueError(f'atoms={a.atoms} exceeds checkpoint max_atoms={max_atoms}')
 if a.atoms and fixed and a.atoms not in fixed and not a.allow_count_extrapolation:raise ValueError(f'atoms={a.atoms} is outside trained counts {fixed}; pass --allow-count-extrapolation explicitly')
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);args=meta['args'];required=('s1_rbf_bins','t1_rbf_bins','condition_rbf_min','condition_rbf_max','condition_rbf_width_scale')
 if meta.get('model_kind')!='marginal_factorized_chem_d3pm_v3':raise ValueError('checkpoint is not a marginal factorized chemistry-D3PM model and cannot be loaded')
 if not all(k in args for k in required):raise ValueError('checkpoint metadata predates configurable scalar RBF conditioning and is incompatible with this sampler')
 model=JointKekuleDDPM(args['hidden'],max_atoms,len(meta['atom_map']),args['layers'],args.get('condition_dropout',.1),args['s1_rbf_bins'],args['t1_rbf_bins'],args['condition_rbf_min'],args['condition_rbf_max'],args['condition_rbf_width_scale']);model.load_state_dict(ck['model']);model.cuda().eval();diff=JointDiffusion(marginals=meta['diffusion_marginals']).cuda();st=meta['condition_stats'];yn=torch.tensor([[(a.s1-st['s1_energy_mean'])/st['s1_energy_std'],(a.t1-st['t1_energy_mean'])/st['t1_energy_std']]],device='cuda');known=load_db(a.root);records=[];unique_mols=[];unique_smiles=set();start=time.time()
 for begin in range(0,a.num,a.batch_size):
  bs=min(a.batch_size,a.num-begin);sample=sample_batch(model,diff,yn.expand(bs,-1),bs,a.atoms,a.cfg,[{'B':3,'C':4,'N':4,'O':2,'F':1,'Si':4,'P':5,'S':6,'Cl':1,'As':5,'Se':6,'Br':1}.get(x,4) for x in meta['atom_map']])
  for j in range(bs):
   count=int(sample['atom_count'][j]);r=decode_kekule_graph(sample['atom_type'][j,:count].tolist(),sample['charge'][j,:count].tolist(),sample['hydrogen'][j,:count].tolist(),sample['radical'][j,:count].tolist(),sample['bond'][j,:count,:count],meta['atom_map']);novel=bool(r.smiles and r.smiles not in known);unique=bool(novel and r.connected and r.smiles not in unique_smiles)
   records.append({'index':begin+j,'atom_count':count,'smiles':r.smiles or '','valid':int(r.valid),'connected':int(r.connected),'aromatic_rings':r.aromatic_rings,'novel_vs_db':int(novel),'unique_novel_connected':int(unique),'error':r.error or ''})
   if unique:unique_smiles.add(r.smiles);unique_mols.append(molecule_with_coordinates(r.molecule,sample['positions'][j,:count]))
  with (out/'molecules.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
  summary={'requested':a.num,'completed':len(records),'valid':sum(x['valid'] for x in records),'connected':sum(x['connected'] for x in records),'aromatic':sum(x['aromatic_rings']>0 for x in records),'novel_vs_db':sum(x['novel_vs_db'] for x in records),'unique_novel_connected':len(unique_mols),'checkpoint_epoch':ck['epoch'],'s1':a.s1,'t1':a.t1,'cfg':a.cfg,'atoms':a.atoms,'atom_count_histogram':{str(k):sum(r['atom_count']==k for r in records) for k in sorted(set(r['atom_count'] for r in records))},'elapsed_s':time.time()-start};(out/'summary.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
 writer=Chem.SDWriter(str(out/'unique_novel_connected.sdf'))
 for i,mol in enumerate(unique_mols):mol.SetProp('_Name',f'generated_{i:06d}');mol.SetProp('SMILES',Chem.MolToSmiles(mol));writer.write(mol)
 writer.close()
 for page in range(0,len(unique_mols),100):Draw.MolsToGridImage(unique_mols[page:page+100],molsPerRow=10,subImgSize=(250,200),legends=[Chem.MolToSmiles(x) for x in unique_mols[page:page+100]]).save(str(out/f'novel_valid_{page//100:03d}.png'))

if __name__=='__main__':main()
