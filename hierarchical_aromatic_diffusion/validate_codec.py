#!/usr/bin/env python3
import argparse,json,torch
from pathlib import Path
from rdkit import Chem,RDLogger
from hierarchical_aromatic_diffusion.codec import RingSystemCodec

def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',default='../data_qcdge_a10_full'); p.add_argument('--aromatic-only',action='store_true'); p.add_argument('--limit',type=int); p.add_argument('--output',default='validation.json'); a=p.parse_args(); RDLogger.DisableLog('rdApp.*')
 root=Path(a.root); manifest=json.loads((root/'manifest.json').read_text()); codec=RingSystemCodec(); total=selected=parsed=exact=junction_ok=oracle_ok=0; failures=[]
 for split in manifest['splits']:
  for rel in split['shards']:
   shard=torch.load(root/rel,map_location='cpu',weights_only=False)
   for row,s in enumerate(shard['smiles']):
    total+=1; mol=Chem.MolFromSmiles(s.decode() if isinstance(s,bytes) else str(s))
    if mol is None: continue
    parsed+=1
    if a.aromatic_only and not any(x.GetIsAromatic() for x in mol.GetAtoms()): continue
    selected+=1
    try:
     spec=codec.encode(mol); ok=codec.roundtrip_equal(mol); jok=codec.validate_junctions(spec); ook=codec.validate_oracle_assembly(spec); exact+=ok; junction_ok+=jok; oracle_ok+=ook
     if not ok or not jok: failures.append({'split':split['name'],'shard':rel,'row':row,'smiles':str(s),'exact':ok,'junction':jok,'oracle':ook})
    except Exception as e: failures.append({'split':split['name'],'shard':rel,'row':row,'smiles':str(s),'error':repr(e)})
    if a.limit and selected>=a.limit: break
   if a.limit and selected>=a.limit: break
  if a.limit and selected>=a.limit: break
 report={'seen':total,'parsed':parsed,'selected':selected,'roundtrip_exact':exact,'junction_valid':junction_ok,'oracle_scaffold_exact':oracle_ok,'success_rate':exact/max(selected,1),'failures':failures[:100]}; Path(a.output).write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='failures'},indent=2))
if __name__=='__main__': main()
