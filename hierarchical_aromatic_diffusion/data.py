"""QCDGE hierarchy tensors in the coordinate atom index and Kekule convention."""
from __future__ import annotations
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from polyring_diffusion.fpa_data_template_free import TemplateFreeFPAEntityDataset

def _cycle_order(atoms,bonds):
 atoms=tuple(sorted(map(int,atoms)));start=atoms[0]
 def dfs(path):
  if len(path)==len(atoms):return path if bonds[path[-1],start]>0 else None
  for nxt in atoms:
   if nxt not in path and bonds[path[-1],nxt]>0:
    hit=dfs(path+(nxt,))
    if hit:return hit
 out=dfs((start,))
 if out is None:raise ValueError(f'no simple perimeter for {atoms}')
 rev=(out[0],)+tuple(reversed(out[1:]));return min(out,rev)

def kekule_matrix(x,atom_map):
 n=len(x['atom_types']);rw=Chem.RWMol()
 for kind,charge,h,rad,arom in zip(x['atom_types'],x['atom_charges'],x['atom_hydrogens'],x['atom_radicals'],x['atom_aromatic']):
  a=Chem.Atom(atom_map[int(kind)]);a.SetFormalCharge(int(charge));a.SetNumExplicitHs(int(h));a.SetNumRadicalElectrons(int(rad));a.SetNoImplicit(True);a.SetIsAromatic(bool(arom));rw.AddAtom(a)
 for i in range(n):
  for j in range(i+1,n):
   k=int(x['atom_bond_types'][i,j])
   if not k:continue
   bt=(Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE,Chem.BondType.AROMATIC)[k-1];rw.AddBond(i,j,bt)
   if bool(x['atom_bond_aromatic'][i,j]):rw.GetBondBetweenAtoms(i,j).SetIsAromatic(True)
 mol=rw.GetMol();Chem.SanitizeMol(mol);Chem.Kekulize(mol,clearAromaticFlags=True);out=torch.zeros(n,n,dtype=torch.long)
 for b in mol.GetBonds():
  k=int(round(b.GetBondTypeAsDouble()));i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx();out[i,j]=out[j,i]=k
 return out

class QCDGEHierarchyDataset(Dataset):
 def __init__(self,root,split,condition_stats=None,atom_map=None):
  self.base=TemplateFreeFPAEntityDataset(root,split,condition_mode='s1_t1',sidecar_dir='hydrogen_counts_v4_aromatic',aromatic_only=True,frame_augmentation=False,condition_stats=condition_stats,atom_map=atom_map);self.condition_stats=self.base.condition_stats;self.atom_map=self.base.atom_map
 def __len__(self):return len(self.base)
 def __getitem__(self,i):
  x=self.base[i];kek=kekule_matrix(x,self.atom_map);rings=[_cycle_order(torch.where(m)[0].tolist(),kek) for m in x['cell_membership']];scaffold=sorted(set().union(*(set(r) for r in rings)));ss=set(scaffold);sub=[a for a in range(len(x['atom_types'])) if a not in ss]
  return {**x,'kekule_bonds':kek,'ordered_rings':rings,'scaffold_atoms':torch.tensor(scaffold),'substituent_atoms':torch.tensor(sub)}

def collate_hierarchy(batch,max_rings=12,max_slots=16,max_atoms=32):
 B=len(batch);R,S,A=max_rings,max_slots,max_atoms
 z=lambda *shape,dtype=torch.long:torch.zeros(*shape,dtype=dtype)
 o={'condition':torch.stack([x['property'] for x in batch]),'ring_mask':z(B,R,dtype=torch.bool),'ring_size':z(B,R),'ring_aromatic':z(B,R),'ring_relation':z(B,R,R),'ring_positions':z(B,R,3,dtype=torch.float32),'slot_mask':z(B,R,S,dtype=torch.bool),'slot_atom':z(B,R,S),'slot_charge':z(B,R,S),'slot_hydrogen':z(B,R,S),'slot_radical':z(B,R,S),'slot_positions':z(B,R,S,3,dtype=torch.float32),'slot_to_atom':torch.full((B,R,S),-1,dtype=torch.long),'junction':z(B,R,R,S,S),'atom_mask':z(B,A,dtype=torch.bool),'atom_type':z(B,A),'atom_charge':z(B,A),'atom_hydrogen':z(B,A),'atom_radical':z(B,A),'atom_aromatic':z(B,A),'atom_positions':z(B,A,3,dtype=torch.float32),'atom_bonds':z(B,A,A),'scaffold_mask':z(B,A,dtype=torch.bool),'substituent_mask':z(B,A,dtype=torch.bool),'perimeter_mask':z(B,A,A,dtype=torch.bool),'attachment_mask':z(B,A,A,dtype=torch.bool),'substituent_bond_mask':z(B,A,A,dtype=torch.bool)}
 for b,x in enumerate(batch):
  n=len(x['atom_types']);nr=len(x['ordered_rings']);assert n<=A and nr<=R;o['atom_mask'][b,:n]=1;o['atom_type'][b,:n]=x['atom_types'];o['atom_charge'][b,:n]=(x['atom_charges']+2).clamp(0,4);o['atom_hydrogen'][b,:n]=x['atom_hydrogens'].clamp(0,4);o['atom_radical'][b,:n]=x['atom_radicals'].clamp(0,2);o['atom_aromatic'][b,:n]=x['atom_aromatic'];o['atom_positions'][b,:n]=x['atom_positions'];o['atom_bonds'][b,:n,:n]=x['kekule_bonds'];o['ring_mask'][b,:nr]=1;o['ring_size'][b,:nr]=x['cell_sizes'];o['ring_aromatic'][b,:nr]=x['cell_aromatic'];o['ring_relation'][b,:nr,:nr]=x['cell_relations'];o['ring_positions'][b,:nr]=x['cell_centers'];sc=set(x['scaffold_atoms'].tolist());sub=set(x['substituent_atoms'].tolist());o['scaffold_mask'][b,list(sc)]=1;o['substituent_mask'][b,list(sub)]=1
  lookup={}
  for ri,ring in enumerate(x['ordered_rings']):
   assert len(ring)<=S
   for si,a in enumerate(ring):
    o['slot_mask'][b,ri,si]=1;o['slot_to_atom'][b,ri,si]=a;o['slot_atom'][b,ri,si]=x['atom_types'][a];o['slot_charge'][b,ri,si]=(x['atom_charges'][a]+2).clamp(0,4);o['slot_hydrogen'][b,ri,si]=x['atom_hydrogens'][a].clamp(0,4);o['slot_radical'][b,ri,si]=x['atom_radicals'][a].clamp(0,2);o['slot_positions'][b,ri,si]=x['atom_positions'][a];lookup.setdefault(a,[]).append((ri,si))
    u,v=ring[si],ring[(si+1)%len(ring)];o['perimeter_mask'][b,u,v]=o['perimeter_mask'][b,v,u]=1
  for slots in lookup.values():
   for ra,sa in slots:
    for rb,sb in slots:o['junction'][b,ra,rb,sa,sb]=1
  for i in sub:
   for j in sub:
    if i!=j:o['substituent_bond_mask'][b,i,j]=1
   for j in sc:
    o['attachment_mask'][b,i,j]=o['attachment_mask'][b,j,i]=1
 return o
