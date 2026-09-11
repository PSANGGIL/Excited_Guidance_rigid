#!/usr/bin/env python3
"""Reverse sampler and heavy-atom RDKit reconstruction for FPA diffusion v2."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch,yaml
from polyring_diffusion.chem import element_record
from polyring_diffusion.fpa_diffusion_v2 import masked_center
from polyring_diffusion.so3_diffusion import MolecularDiffusion
from polyring_diffusion.fpa_model_v2 import FPAEntityDenoiserV2
from polyring_diffusion.fpa_projection import project_fpa_batch

def symmetric_step(diff,noisy,logits,t,token,mask):
 out=diff.p_tokens(noisy,logits,t,token,mask);upper=torch.triu(out,1);return upper+upper.transpose(1,2)
def cfg_predict(model,args,condition,guidance):
 b=len(condition);conditional=model(*args,condition);unconditional=model(*args,condition,torch.ones(b,dtype=torch.bool,device=condition.device));return {k:unconditional[k]+guidance*(v-unconditional[k]) for k,v in conditional.items()}

def reconstruct(atom_types,charges,bonds,positions,atom_map,charge_min=-2):
 from rdkit import Chem
 rw=Chem.RWMol()
 for idx,q in zip(atom_types.tolist(),charges.tolist()):
  atom=Chem.Atom(atom_map[idx]);atom.SetFormalCharge(int(q)+charge_min);rw.AddAtom(atom)
 kinds={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
 for i in range(len(atom_types)):
  for j in range(i+1,len(atom_types)):
   k=int(bonds[i,j]);
   if k in kinds:
    if k==4:rw.GetAtomWithIdx(i).SetIsAromatic(True);rw.GetAtomWithIdx(j).SetIsAromatic(True)
    rw.AddBond(i,j,kinds[k])
 mol=rw.GetMol();conf=Chem.Conformer(len(atom_types))
 for i,p in enumerate(positions.tolist()):conf.SetAtomPosition(i,p)
 mol.AddConformer(conf);Chem.SanitizeMol(mol);return Chem.AddHs(mol,addCoords=True)
@torch.no_grad()
def sample(model,diff,condition,atom_count,cell_count,guidance=2.,atom_map=None):
 dev=condition.device;b=len(condition);na=max(atom_count);nc=max(1,max(cell_count));am=torch.arange(na,device=dev)[None]<torch.tensor(atom_count,device=dev)[:,None];cm=torch.arange(nc,device=dev)[None]<torch.tensor(cell_count,device=dev)[:,None]
 at=torch.where(am,model.atom_mask_token,model.atom_pad_token);charge=torch.full((b,na),model.charge_mask_token,device=dev,dtype=torch.long);bond=torch.full((b,na,na),5,device=dev,dtype=torch.long);ct=torch.where(cm,model.cell_mask_token,model.cell_pad_token);size=torch.full((b,nc),model.size_mask_token,device=dev,dtype=torch.long);fusion=torch.full((b,nc,nc),2,device=dev,dtype=torch.long);member=torch.full((b,nc,na),2,device=dev,dtype=torch.long)
 apos=masked_center(torch.randn(b,na,3,device=dev),am);cpos=torch.randn(b,nc,3,device=dev)*cm.unsqueeze(-1);radius=torch.randn(b,nc,device=dev)*cm;frame=diff.initial_frames(b,nc,dev)
 for step in reversed(range(diff.timesteps)):
  t=torch.full((b,),step,device=dev,dtype=torch.long);args=(at,charge,apos,bond,am,ct,size,cpos,frame,radius,fusion,member,cm,t)
  p=cfg_predict(model,args,condition,guidance)
  apos=masked_center(diff.p_coordinates(apos,p['atom_eps'],t,am),am);cpos=diff.p_coordinates(cpos,p['cell_eps'],t,cm);radius=diff.p_coordinates(radius.unsqueeze(-1),p['radius_eps'].unsqueeze(-1),t,cm).squeeze(-1);frame=diff.p_frames(frame,p['rotation_score'],t,cm)
  at=diff.p_tokens(at,p['atom_type_logits'],t,model.atom_mask_token,am);charge=diff.p_tokens(charge,p['charge_logits'],t,model.charge_mask_token,am);ct=diff.p_tokens(ct,p['cell_type_logits'],t,model.cell_mask_token,cm);size=diff.p_tokens(size,p['cell_size_logits'],t,model.size_mask_token,cm);bond=symmetric_step(diff,bond,p['bond_logits'],t,5,am[:,:,None]&am[:,None]);fusion=symmetric_step(diff,fusion,p['fusion_logits'],t,2,cm[:,:,None]&cm[:,None]);member=diff.p_tokens(member,p['membership_logits'],t,2,cm[:,:,None]&am[:,None])
 out={'atom_types':at,'charges':charge,'positions':apos,'bonds':bond,'cell_types':ct,'cell_sizes':size,'cell_centers':cpos,'cell_frames':frame,'cell_radii':radius.exp().clamp(.1,10.),'fusion':fusion,'membership':member,'atom_mask':am,'cell_mask':cm}
 return project_fpa_batch(out,p,atom_map) if atom_map is not None else out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--conditions',required=True,help='torch file [N, condition_dim]');ap.add_argument('--output',required=True);ap.add_argument('--guidance',type=float,default=2.);a=ap.parse_args();state=torch.load(a.checkpoint,map_location='cuda',weights_only=False);cfg=state['config'];atom_map=state['atom_map'];mc=dict(cfg['model']);mc.update(num_atom_types=len(atom_map),num_cell_types=len(state['manifest']['templates']),atom_radii=[element_record(x).vdw_radius/3 for x in atom_map],max_cell_ports=max(len(x['signature']) for x in state['manifest']['templates']),max_atoms=state['model']['atom_count.2.weight'].shape[0]-1,max_cells=state['model']['cell_count.2.weight'].shape[0]-1);model=FPAEntityDenoiserV2(**mc).cuda();model.load_state_dict(state['model']);model.eval();diff=MolecularDiffusion(**cfg['diffusion']).cuda();condition=torch.load(a.conditions,map_location='cuda').float();ac=torch.distributions.Categorical(logits=model.atom_count(condition)).sample().clamp_min(1).tolist();cc=torch.distributions.Categorical(logits=model.cell_count(condition)).sample().tolist();out=sample(model,diff,condition,ac,cc,a.guidance,atom_map);mols=[]
 for i in range(len(condition)):
  try:mols.append(reconstruct(out['atom_types'][i,out['atom_mask'][i]],out['charges'][i,out['atom_mask'][i]],out['bonds'][i][out['atom_mask'][i]][:,out['atom_mask'][i]],out['positions'][i,out['atom_mask'][i]],atom_map))
  except Exception:mols.append(None)
 torch.save({'samples':{k:v.cpu() for k,v in out.items()},'molecules':mols},a.output);print(json.dumps({'requested':len(condition),'rdkit_valid':sum(x is not None for x in mols),'output':a.output}))
if __name__=='__main__':main()
