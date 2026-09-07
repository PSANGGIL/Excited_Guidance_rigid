#!/usr/bin/env python3
"""Check exact uniqueness and 100k-database novelty of generated SDFs."""
import argparse,json
from pathlib import Path
import torch
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
def identity(mol):
 mol=Chem.RemoveHs(mol);Chem.SanitizeMol(mol)
 return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True),Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False),Chem.MolToInchiKey(mol)
def main():
 p=argparse.ArgumentParser();p.add_argument('--sdf-dir',type=Path,required=True);p.add_argument('--db-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();generated=[]
 for sdf in sorted(a.sdf_dir.glob('*.sdf')):
  mol=next((m for m in Chem.SDMolSupplier(str(sdf),removeHs=False) if m is not None),None)
  if mol:
   iso,noniso,ik=identity(mol);generated.append({'name':sdf.stem,'canonical_smiles':iso,'nonisomeric_smiles':noniso,'inchikey':ik,'matching_splits':[]})
 targets={g['canonical_smiles'] for g in generated};non_targets={g['nonisomeric_smiles'] for g in generated}
 for split in ('train','val','test'):
  hits=set();non_hits=set()
  for shard in sorted((a.db_root/split).glob('shard_*.pt')):
   d=torch.load(shard,map_location='cpu',weights_only=False)
   for smi in d['smiles']:
    mol=Chem.MolFromSmiles(smi)
    if mol is None:continue
    iso=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    if iso in targets:hits.add(iso)
    noniso=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=False)
    if noniso in non_targets:non_hits.add(noniso)
  for g in generated:
   if g['canonical_smiles'] in hits:g['matching_splits'].append(split)
   g.setdefault('nonisomeric_matching_splits',[])
   if g['nonisomeric_smiles'] in non_hits:g['nonisomeric_matching_splits'].append(split)
 for g in generated:g['novel_exact']=not g['matching_splits']
 report={'generated_count':len(generated),'unique_canonical_smiles':len({g['canonical_smiles'] for g in generated}),'unique_inchikeys':len({g['inchikey'] for g in generated}),'internal_duplicate_count':len(generated)-len({g['inchikey'] for g in generated}),'exact_novel_count':sum(g['novel_exact'] for g in generated),'nonisomeric_novel_count':sum(not g['nonisomeric_matching_splits'] for g in generated),'database_molecules_checked':100000,'molecules':generated}
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
