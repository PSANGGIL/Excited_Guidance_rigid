#!/usr/bin/env python3
import argparse,csv,json,re
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw,rdDepictor
from run_xtb_stda import states

def main():
 p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,default=Path('samples'));p.add_argument('--output-dir',type=Path,default=Path('samples/xtb_molecule_os_top3'));a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);results=[]
 for log in sorted(a.samples.glob('rigid_1x_100ep_seed*_xtb/*/stda.log')):
  if 'sTDA done.' not in log.read_text(errors='replace'):continue
  parsed=states(log)
  if not parsed:continue
  top=sorted(parsed,key=lambda x:x['f_osc'],reverse=True)[:3];m=re.search(r'seed(\d+)_xtb',log.parents[1].name);seed=m.group(1) if m else 'unknown';name=log.parent.name
  mol=Chem.MolFromMolFile(str(log.parent/'xtbopt.sdf'),removeHs=False)
  if mol is None:continue
  mol=Chem.RemoveHs(mol);rdDepictor.Compute2DCoords(mol,canonOrient=True,clearConfs=True)
  legend=[f'seed {seed} | {name}']+[f"OS#{i}: S{x['state']}  E={x['energy_eV']:.3f} eV  lambda={x['wavelength_nm']:.1f} nm  f={x['f_osc']:.4f}" for i,x in enumerate(top,1)]
  out=a.output_dir/f'seed{seed}';out.mkdir(parents=True,exist_ok=True);png=out/f'{name}.png';Draw.MolToFile(mol,str(png),size=(1200,900),legend='\n'.join(legend));results.append({'seed':seed,'name':name,'png':str(png),'top3':top})
 with (a.output_dir/'os_top3.csv').open('w',newline='') as f:
  fields=['seed','name','rank','state','energy_eV','wavelength_nm','f_osc','png'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for item in results:
   for rank,row in enumerate(item['top3'],1):w.writerow({'seed':item['seed'],'name':item['name'],'rank':rank,**row,'png':item['png']})
 (a.output_dir/'os_top3.json').write_text(json.dumps(results,indent=2));print(json.dumps({'rendered_molecules':len(results),'states_per_molecule':3,'output':str(a.output_dir)},indent=2))
if __name__=='__main__':main()
