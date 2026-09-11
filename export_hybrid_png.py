#!/usr/bin/env python3
"""Decode connected hybrid ring/free-atom graphs and render RDKit 2-D PNGs."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw,rdDepictor
from export_rigid_sdf import DSU

def connected(record):
 n=len(record['free_atoms']);has_rings=bool(record['rings']);offset=1 if has_rings else 0;adj=[set() for _ in range(n+offset)]
 if not adj:return True
 for e in record['free_atom_connections']:adj[e['left']+offset].add(e['right']+offset);adj[e['right']+offset].add(e['left']+offset)
 for e in record['ring_atom_attachments']:adj[0].add(e['atom']+offset);adj[e['atom']+offset].add(0)
 seen={0};todo=[0]
 while todo:
  u=todo.pop()
  for v in adj[u]-seen:seen.add(v);todo.append(v)
 return record['connected_rings'] and len(seen)==len(adj)

def decode(record):
 rings=record['rings'];offsets=[];raw=[]
 for ring in rings:offsets.append(len(raw));raw.extend(ring['signature'])
 d=DSU(len(raw));singles=[]
 def pair(ri,p):n=len(rings[ri]['signature']);return offsets[ri]+p,offsets[ri]+((p+1)%n)
 for e in record['ring_connections']:
  i,j,lp,rp=e['left'],e['right'],e['left_port'],e['right_port'];kind=e['kind']
  if kind=='SPIRO':d.union(offsets[i]+lp,offsets[j]+rp)
  elif kind=='FUSED':
   la,lb=pair(i,lp);ra,rb=pair(j,rp);key=lambda x:(raw[x][0],raw[x][1])
   if key(la)==key(rb) and key(lb)==key(ra):d.union(la,rb);d.union(lb,ra)
   elif key(la)==key(ra) and key(lb)==key(rb):d.union(la,ra);d.union(lb,rb)
   else:raise ValueError('incompatible fused atoms')
  elif kind=='SINGLE':singles.append((offsets[i]+lp,offsets[j]+rp,1))
 roots={};rw=Chem.RWMol()
 for idx,sig in enumerate(raw):
  root=d.find(idx)
  if root not in roots:
   atom=Chem.Atom(int(sig[0]));atom.SetFormalCharge(int(sig[1]));roots[root]=rw.AddAtom(atom)
 def add_idx(a,b,label):
  if a==b:raise ValueError('self bond')
  bt={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}[int(label)]
  old=rw.GetBondBetweenAtoms(a,b)
  if old is None:
   rw.AddBond(a,b,bt)
   if int(label)==4:rw.GetAtomWithIdx(a).SetIsAromatic(True);rw.GetAtomWithIdx(b).SetIsAromatic(True)
  elif old.GetBondType()!=bt:raise ValueError('conflicting duplicate bond')
 def ring_idx(raw_idx):return roots[d.find(raw_idx)]
 for ri,ring in enumerate(rings):
  sig=ring['signature'];n=len(sig)
  for p in range(n):add_idx(ring_idx(offsets[ri]+p),ring_idx(offsets[ri]+((p+1)%n)),sig[p][3])
 for a,b,k in singles:add_idx(ring_idx(a),ring_idx(b),k)
 free=[]
 for spec in record['free_atoms']:
  atom=Chem.Atom(str(spec['element']));atom.SetFormalCharge(int(spec['formal_charge']));free.append(rw.AddAtom(atom))
 for e in record['free_atom_connections']:add_idx(free[e['left']],free[e['right']],e['bond_type'])
 for e in record['ring_atom_attachments']:add_idx(ring_idx(offsets[e['ring']]+e['ring_port']),free[e['atom']],e['bond_type'])
 mol=rw.GetMol();Chem.SanitizeMol(mol);return mol,Chem.MolToSmiles(mol)

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--db-root',type=Path);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);rows=[];grid=[];legends=[];db_smiles=set();novel_smiles=set()
 if a.db_root:
  import torch
  for shard in a.db_root.rglob('shard_*.pt'):
   raw=torch.load(shard,map_location='cpu',weights_only=False)
   for value in raw.get('smiles',[]):
    mol=Chem.MolFromSmiles(str(value))
    if mol is not None:db_smiles.add(Chem.MolToSmiles(Chem.RemoveHs(mol)))
 for i,line in enumerate(a.input.read_text().splitlines()):
  rec=json.loads(line)
  if not rec['free_atoms'] or not connected(rec):continue
  name=f'hybrid_{i:04d}';status='ok';smiles='';error=''
  try:
   mol,smiles=decode(rec)
   if smiles in db_smiles:status='in_database'
   elif smiles in novel_smiles:status='generated_duplicate'
   else:
    novel_smiles.add(smiles);rdDepictor.Compute2DCoords(mol,canonOrient=True,clearConfs=True);legend="{} | E={:.3f} | OS={:.3f} | rings={} | free atoms={}".format(name,rec.get("target_energy",0.0),rec.get("target_f_osc",0.0),len(rec["rings"]),len(rec["free_atoms"]));Draw.MolToFile(mol,str(a.output_dir/f"{name}.png"),size=(1100,800),legend=legend)
    if len(grid)<36:grid.append(mol);legends.append(name)
  except Exception as exc:status='rdkit_failed';error=str(exc)
  rows.append({'generation_index':i,'name':name,'status':status,'canonical_smiles':smiles,'error':error})
 if grid:Draw.MolsToGridImage(grid,molsPerRow=4,subImgSize=(500,400),legends=legends,useSVG=False).save(a.output_dir/'contact_sheet.png')
 with (a.output_dir/'manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=('generation_index','name','status','canonical_smiles','error'));w.writeheader();w.writerows(rows)
 summary={'database_size':len(db_smiles),'connected_candidates':len(rows),'novel_unique_rendered':sum(r['status']=='ok' for r in rows),'in_database':sum(r['status']=='in_database' for r in rows),'generated_duplicates':sum(r['status']=='generated_duplicate' for r in rows),'rdkit_failed':sum(r['status']=='rdkit_failed' for r in rows)};(a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
