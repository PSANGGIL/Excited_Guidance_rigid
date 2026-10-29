#!/usr/bin/env python3
"""Convert QCDGE A_10 HDF5 into lossless sharded hybrid ring/atom data."""
from __future__ import annotations
import argparse,hashlib,json,math,os,random
from collections import Counter
from pathlib import Path
import h5py,numpy as np,torch
from rdkit import Chem,RDLogger
from rdkit.Chem import rdDetermineBonds
from prepare_rigid_dataset import canonical_ring,connections,cycle_order,ring_pose,template_hash,offsets,cat
RDLogger.DisableLog('rdApp.*')
ATOM_MAP=(5,6,7,8,9,14,15,16,17,35,53)

def state_arrays(group,max_states=20):
 raw=group['excited_state']['Info_of_AllExcitedStates'][0];states=json.loads(raw.decode() if isinstance(raw,bytes) else raw).values();bucket={'singlet':[],'triplet':[]}
 for x in states:
  typ=str(x.get('state_type','')).lower()
  if typ not in bucket:continue
  energy=float(str(x['excitation_e_eV']).split()[0]);osc=float(x.get('oscillator_trength',x.get('oscillator_strength',0.0)));bucket[typ].append((int(x['state']),energy,osc))
 out={}
 for typ in ('singlet','triplet'):
  rows=sorted(bucket[typ])[:max_states];e=torch.zeros(max_states);f=torch.zeros(max_states);m=torch.zeros(max_states,dtype=torch.bool)
  for i,(_,energy,osc) in enumerate(rows):e[i]=energy;f[i]=osc;m[i]=True
  out[typ+'_energies']=e;out[typ+'_oscillator_strengths']=f;out[typ+'_mask']=m
 if not out['singlet_mask'][0] or not out['triplet_mask'][0]:raise ValueError('missing S1 or T1')
 return out

def ring_record(mol,xyz):
 sigs=[];rings=[];centers=[];normals=[];radii=[]
 for br in mol.GetRingInfo().BondRings():
  sig,ids=canonical_ring(mol,cycle_order(mol,br));c,n,r=ring_pose(torch.from_numpy(xyz),ids);sigs.append(sig);rings.append(ids);centers.append(c);normals.append(n);radii.append(r)
 conns,shared=connections(mol,rings);ro=[0]
 for ring in rings:ro.append(ro[-1]+len(ring))
 return {'signatures':sigs,'hashes':torch.tensor([template_hash(x) for x in sigs],dtype=torch.int64),'centers':torch.stack(centers) if centers else torch.empty((0,3)),'normals':torch.stack(normals) if normals else torch.empty((0,3)),'radii':torch.tensor(radii),'ring_atoms':torch.tensor([a for r in rings for a in r],dtype=torch.int32),'ring_atom_offsets':torch.tensor(ro,dtype=torch.int32),'connections':conns,'shared':shared}

def molecule(group):
 smiles=str(group.attrs.get('SMILES_RDKIT_CAN') or group.attrs.get('SMILES_RDKIT') or group.attrs.get('SMILES_PYBEL'));z=np.asarray(group['ground_state']['labels'][()],dtype=np.int64).reshape(-1);xyz=np.asarray(group['ground_state']['coords'][()],dtype=np.float32)
 lines=[str(len(z)),'QCDGE']+[f"{Chem.GetPeriodicTable().GetElementSymbol(int(zi))} {x:.10f} {y:.10f} {zz:.10f}" for zi,(x,y,zz) in zip(z,xyz)];mol=Chem.MolFromXYZBlock('\n'.join(lines)+'\n')
 if mol is None:raise ValueError('XYZ parse failed')
 rdDetermineBonds.DetermineBonds(mol,charge=0,covFactor=1.3,allowChargedFragments=True,embedChiral=True,useHueckel=False);Chem.SanitizeMol(mol);mol=Chem.RemoveAllHs(mol);Chem.SanitizeMol(mol)
 if any(a.GetAtomicNum()==1 for a in mol.GetAtoms()):raise ValueError('hydrogen removal failed')
 conf=mol.GetConformer();xyz=np.asarray([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())],dtype=np.float32)
 unknown=sorted({a.GetAtomicNum() for a in mol.GetAtoms()}-set(ATOM_MAP))
 if unknown:raise ValueError('unsupported elements '+str(unknown))
 ring=ring_record(mol,xyz);types=torch.zeros((mol.GetNumAtoms(),len(ATOM_MAP)),dtype=torch.bool);types[torch.arange(mol.GetNumAtoms()),torch.tensor([ATOM_MAP.index(a.GetAtomicNum()) for a in mol.GetAtoms()])]=True;bonds=[];bond_types=[]
 for b in mol.GetBonds():bonds.append((b.GetBeginAtomIdx(),b.GetEndAtomIdx()));bond_types.append(4 if b.GetIsAromatic() else {1.0:1,2.0:2,3.0:3}[b.GetBondTypeAsDouble()])
 states=state_arrays(group);s1=float(states['singlet_oscillator_strengths'][0]);return {'smiles':smiles,'positions':torch.from_numpy(xyz),'atom_types':types,'atom_charges':torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()],dtype=torch.int32),'bond_indices':torch.tensor(bonds,dtype=torch.int32).reshape(-1,2),'bond_types':torch.tensor(bond_types,dtype=torch.int32),'ring':ring,'states':states,'property':math.log10(s1+1e-6),'raw_property':s1}

def write_shard(path,ids,source):
 records=[];errors=[];stats=Counter();templates={}
 with h5py.File(source,'r') as f:
  for cid in ids:
   try:r=molecule(f[cid]);records.append((cid,r));stats['molecules']+=1;stats['atoms']+=len(r['positions']);stats['rings']+=len(r['ring']['signatures']);ring_atoms=set(map(int,r['ring']['ring_atoms']));stats['free_heavy_atoms']+=sum(int(r['atom_types'][i].to(torch.uint8).argmax())!=0 and i not in ring_atoms for i in range(len(r['positions'])));[templates.__setitem__(str(int(h)),sig) for h,sig in zip(r['ring']['hashes'],r['ring']['signatures'])]
   except Exception as e:errors.append({'id':cid,'error':str(e)});stats['errors']+=1
 rr=[x[1]['ring'] for x in records];payload={'schema_version':2,'split':path.parent.name,'molecule_ids':[x[0] for x in records],'smiles':[x[1]['smiles'] for x in records],'source_rows':torch.tensor([int(x[0]) for x in records]),'properties':torch.tensor([x[1]['property'] for x in records]),'raw_properties':torch.tensor([x[1]['raw_property'] for x in records]),'atom_positions':torch.cat([x[1]['positions'] for x in records]),'atom_types':torch.cat([x[1]['atom_types'] for x in records]),'atom_charges':torch.cat([x[1]['atom_charges'] for x in records]),'atom_offsets':offsets(len(x[1]['positions']) for x in records),'bond_indices':torch.cat([x[1]['bond_indices'] for x in records]),'bond_types':torch.cat([x[1]['bond_types'] for x in records]),'bond_offsets':offsets(len(x[1]['bond_types']) for x in records),'ring_template_hashes':cat(rr,'hashes',(0,),torch.int64),'ring_centers':cat(rr,'centers',(0,3),torch.float32),'ring_normals':cat(rr,'normals',(0,3),torch.float32),'ring_radii':cat(rr,'radii',(0,),torch.float32),'ring_offsets':offsets(len(x['signatures']) for x in rr),'ring_atom_indices':cat(rr,'ring_atoms',(0,),torch.int32),'ring_atom_offsets':offsets(x['ring_atoms'].numel() for x in rr),'per_ring_atom_offsets':[x['ring_atom_offsets'] for x in rr],'connections':cat(rr,'connections',(0,6),torch.int32),'connection_offsets':offsets(x['connections'].shape[0] for x in rr),'connection_shared_atoms':[x['shared'] for x in rr],'singlet_energies':torch.stack([x[1]['states']['singlet_energies'] for x in records]),'singlet_oscillator_strengths':torch.stack([x[1]['states']['singlet_oscillator_strengths'] for x in records]),'singlet_mask':torch.stack([x[1]['states']['singlet_mask'] for x in records]),'triplet_energies':torch.stack([x[1]['states']['triplet_energies'] for x in records]),'triplet_oscillator_strengths':torch.stack([x[1]['states']['triplet_oscillator_strengths'] for x in records]),'triplet_mask':torch.stack([x[1]['states']['triplet_mask'] for x in records]),'errors':errors,'stats':dict(stats),'templates':templates,'metadata':{'atom_map':['B','C','N','O','F','Si','P','S','Cl','Br','I'],'target_transform':'log10_s1_os','state_columns':'singlet/triplet energy, oscillator_strength, mask','max_states_per_multiplicity':20,'geometry_source':'QCDGE ground_state'}}
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path);return dict(stats),templates

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--shard-size',type=int,default=2000);p.add_argument('--seed',type=int,default=47);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 with h5py.File(a.input,'r') as f:ids=list(f.keys())
 random.Random(a.seed).shuffle(ids)
 if a.limit:ids=ids[:a.limit]
 n=len(ids);split_ids={'train':ids[:int(.8*n)],'val':ids[int(.8*n):int(.9*n)],'test':ids[int(.9*n):]};all_templates={};splits=[]
 for split,items in split_ids.items():
  shards=[];total=Counter()
  for lo in range(0,len(items),a.shard_size):
   path=a.output/split/f'shard_{lo:06d}_{min(lo+a.shard_size,len(items)):06d}.pt';stats,templates=write_shard(path,items[lo:lo+a.shard_size],a.input);total.update(stats);all_templates.update(templates);shards.append(str(path.relative_to(a.output)));print(json.dumps({'split':split,'done':min(lo+a.shard_size,len(items)),'total':len(items),'stats':stats}),flush=True)
  splits.append({'name':split,'source_molecules':len(items),'shards':shards,'stats':dict(total)})
 templates=[{'id':i,'hash':int(h),'signature':json.loads(sig)} for i,(h,sig) in enumerate(sorted(all_templates.items(),key=lambda x:x[1]))];manifest={'schema_version':2,'source_file':str(a.input.resolve()),'selected_molecules':n,'condition_schema':{'singlet':['energy_eV','oscillator_strength','mask'],'triplet':['energy_eV','oscillator_strength','mask'],'max_states':20},'splits':splits,'templates':templates};(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'complete':True,'output':str(a.output),'selected':n,'templates':len(templates)}),flush=True)
if __name__=='__main__':main()
