#!/usr/bin/env python3
"""Render the top completed xTB/sTDA molecules by oscillator strength."""
import argparse,csv,json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor

from run_xtb_stda import states


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=Path,default=Path('samples'))
    p.add_argument('--output-dir',type=Path,default=Path('samples/xtb_top3_snapshot'))
    p.add_argument('--top',type=int,default=3)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    completed=[]
    for log in sorted(a.samples.glob('rigid_1x_100ep_seed*_xtb/*/stda.log')):
        text=log.read_text(errors='replace')
        if 'sTDA done.' not in text:continue
        parsed=states(log)
        if not parsed:continue
        strongest=max(parsed,key=lambda x:x['f_osc'])
        completed.append({'name':log.parent.name,'work_dir':str(log.parent),'strongest':strongest,'states_1_10':parsed})
    ranked=sorted(completed,key=lambda x:x['strongest']['f_osc'],reverse=True)[:a.top]
    mols=[];legends=[];state_rows=[]
    for rank,item in enumerate(ranked,1):
        sdf=Path(item['work_dir'])/'xtbopt.sdf';mol=Chem.MolFromMolFile(str(sdf),removeHs=False)
        if mol is None:continue
        mol=Chem.RemoveHs(mol)
        rdDepictor.Compute2DCoords(mol,canonOrient=True,clearConfs=True)
        mols.append(mol);s=item['strongest'];legends.append(f"#{rank} {item['name']} | S{s['state']} | E={s['energy_eV']:.3f} eV | lambda={s['wavelength_nm']:.1f} nm | f={s['f_osc']:.4f}")
        Draw.MolToFile(mol,str(a.output_dir/f"top{rank}_{item['name']}.png"),size=(1200,900),legend=legends[-1])
        for row in item['states_1_10']:state_rows.append({'rank':rank,'name':item['name'],**row})
    if mols:
        Draw.MolsToGridImage(mols,molsPerRow=min(3,len(mols)),subImgSize=(600,450),legends=legends).save(a.output_dir/'top3_grid.png')
    with (a.output_dir/'top3_states_1_10.csv').open('w',newline='') as f:
        fields=['rank','name','state','energy_eV','wavelength_nm','f_osc'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(state_rows)
    summary={'completed_molecules_scanned':len(completed),'ranking':'maximum f_osc among states 1-10','top':ranked}
    (a.output_dir/'top3_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'completed_molecules_scanned':len(completed),'top':[{'rank':i+1,'name':x['name'],**x['strongest']} for i,x in enumerate(ranked)]},indent=2))


if __name__=='__main__':main()
