#!/usr/bin/env python3
"""Decode generated rigid ring graphs and export sanitized 3-D SDF files."""
import argparse,csv,json,math
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem

class DSU:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a,b=self.find(a),self.find(b)
  if a!=b:self.p[b]=a

def decode(record,seed,embed=True):
 rings=record['rings']; offsets=[]; raw=[]
 for ring in rings:
  offsets.append(len(raw));raw.extend(ring['signature'])
 d=DSU(len(raw)); singles=[]
 def pair(ri,p):
  n=len(rings[ri]['signature']);return offsets[ri]+p,offsets[ri]+((p+1)%n)
 for e in record['connections']:
  i,j,lp,rp=e['left'],e['right'],e['left_port'],e['right_port']; kind=e['kind']
  if kind=='SPIRO':d.union(offsets[i]+lp,offsets[j]+rp)
  elif kind=='FUSED':
   la,lb=pair(i,lp);ra,rb=pair(j,rp); A=lambda x:(raw[x][0],raw[x][1])
   if A(la)==A(rb) and A(lb)==A(ra):d.union(la,rb);d.union(lb,ra)
   elif A(la)==A(ra) and A(lb)==A(rb):d.union(la,ra);d.union(lb,rb)
   else:raise ValueError('incompatible fused atoms')
  elif kind=='SINGLE':singles.append((offsets[i]+lp,offsets[j]+rp,1))
 roots={}; rw=Chem.RWMol()
 for idx,sig in enumerate(raw):
  root=d.find(idx)
  if root not in roots:
   atom=Chem.Atom(int(sig[0]));atom.SetFormalCharge(int(sig[1]));roots[root]=rw.AddAtom(atom)
 def add(a,b,label):
  a,b=roots[d.find(a)],roots[d.find(b)]
  if a==b:raise ValueError('self bond')
  existing=rw.GetBondBetweenAtoms(a,b)
  bt={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}[int(label)]
  if existing is None:
   rw.AddBond(a,b,bt)
   if label==4:
    rw.GetAtomWithIdx(a).SetIsAromatic(True);rw.GetAtomWithIdx(b).SetIsAromatic(True)
 for ri,ring in enumerate(rings):
  sig=ring['signature'];n=len(sig)
  for p in range(n):add(offsets[ri]+p,offsets[ri]+((p+1)%n),sig[p][3])
 for a,b,k in singles:add(a,b,k)
 mol=rw.GetMol();Chem.SanitizeMol(mol)
 if not embed:return mol,Chem.MolToSmiles(mol)
 mol=Chem.AddHs(mol)
 params=AllChem.ETKDGv3();params.randomSeed=int(seed);params.useRandomCoords=True;params.maxIterations=1000
 if AllChem.EmbedMolecule(mol,params)!=0:raise ValueError('ETKDG embedding failed')
 if AllChem.MMFFHasAllMoleculeParams(mol):AllChem.MMFFOptimizeMolecule(mol,maxIters=1000)
 else:AllChem.UFFOptimizeMolecule(mol,maxIters=1000)
 Chem.SanitizeMol(mol);return mol,Chem.MolToSmiles(Chem.RemoveHs(mol))

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('samples/rigid_100k.jsonl'));p.add_argument('--output-dir',type=Path,default=Path('samples/rigid_100k_xtb'));p.add_argument('--per-target',type=int,default=3);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 used={};rows=[]
 for index,line in enumerate(a.input.read_text().splitlines()):
  rec=json.loads(line);target=str(rec['target_f_osc']);used.setdefault(target,0)
  if used[target]>=a.per_target:continue
  name=f"target_{target.replace('.','p')}_{used[target]:02d}";status='ok';smiles='';error=''
  try:
   mol,smiles=decode(rec,20260907+index);w=Chem.SDWriter(str(a.output_dir/f'{name}.sdf'));mol.SetProp('target_f_osc',target);mol.SetProp('source_generation_index',str(index));w.write(mol);w.close();used[target]+=1
  except Exception as exc:status='decode_failed';error=str(exc)
  rows.append({'name':name,'generation_index':index,'target_f_osc':target,'canonical_smiles':smiles,'status':status,'error':error})
  if all(used.get(t,0)>=a.per_target for t in ('0.1','0.8','3.0')):break
 with (a.output_dir/'decode_manifest.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps({'exported':sum(used.values()),'per_target':used,'attempts':len(rows),'failures':sum(r['status']!='ok' for r in rows)},indent=2))
if __name__=='__main__':main()
