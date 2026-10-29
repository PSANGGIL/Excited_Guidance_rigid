#!/usr/bin/env python3
"""Verify shard invariants and optional row-level provenance against dataset_train.csv."""
import argparse,csv,json,math
from pathlib import Path
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('data_ring_rigid'));p.add_argument('--csv',type=Path);a=p.parse_args();m=json.loads((a.root/'manifest.json').read_text()); expected={}; counts={}
 for split in m['splits']:
  n=atoms=rings=connections=0
  for rel in split['shards']:
   d=torch.load(a.root/rel,map_location='cpu',weights_only=False); k=len(d['smiles']); n+=k
   assert len(d['atom_offsets'])==len(d['ring_offsets'])==len(d['connection_offsets'])==k+1
   assert int(d['atom_offsets'][-1])==len(d['atom_positions'])==len(d['atom_types'])==len(d['atom_charges'])
   assert int(d['ring_offsets'][-1])==len(d['ring_template_hashes'])==len(d['ring_centers'])==len(d['ring_normals'])==len(d['ring_radii'])
   assert int(d['connection_offsets'][-1])==len(d['connections'])
   for key in ('atom_positions','ring_centers','ring_normals','ring_radii','properties','raw_properties'): assert torch.isfinite(d[key]).all(),f'{rel}: {key} nonfinite'
   atoms+=len(d['atom_positions']);rings+=len(d['ring_centers']);connections+=len(d['connections'])
   for row,smi,prop in zip(d['source_rows'],d['smiles'],d['raw_properties'].tolist()):
    if int(row) in expected: raise AssertionError(f'duplicate source row {row}')
    expected[int(row)]=(smi,float(prop),split['name'])
  assert n==split['source_molecules'];counts[split['name']]={'molecules':n,'atoms':atoms,'rings':rings,'connections':connections}
 assert len(expected)==m['total_source_molecules']==100000
 provenance={'checked':0,'missing':0,'smiles_mismatch':0,'property_mismatch':0}
 if a.csv:
  with a.csv.open(newline='',encoding='utf-8-sig') as f:
   for source_id,row in enumerate(csv.DictReader(f)):
    if source_id not in expected:continue
    smi,prop,_=expected.pop(source_id);provenance['checked']+=1
    provenance['smiles_mismatch']+=row['molecule']!=smi
    provenance['property_mismatch']+=not math.isclose(float(row['f_osc']),prop,rel_tol=1e-6,abs_tol=1e-7)
  provenance['missing']=len(expected)
  assert all(provenance[k]==0 for k in ('missing','smiles_mismatch','property_mismatch')),provenance
 report={'valid':True,'schema_version':m['schema_version'],'templates':len(m['templates']),'counts':counts,'csv_provenance':provenance}
 (a.root/'verification_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
