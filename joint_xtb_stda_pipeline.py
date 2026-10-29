#!/usr/bin/env python3
"""Prepare generated molecules and run restartable xTB/sTDA with 4x1 CPU."""
import argparse,csv,json,os,re,shutil,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem,rdMolDescriptors

ROW=re.compile(r'^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\b')

def states(path):
 found=False;out=[]
 if not path.exists():return out
 for line in path.read_text(errors='replace').splitlines():
  if ' '.join(line.split()).startswith('state eV nm fL'):found=True;continue
  if not found:continue
  m=ROW.match(line)
  if not m:
   if out:break
   continue
  state=int(m.group(1))
  if state>10:break
  out.append({'state':state,'energy_eV':float(m.group(2)),'wavelength_nm':float(m.group(3)),'f_osc':float(m.group(4))})
 return out

def prepare(source,out,seed):
 out.mkdir(parents=True,exist_ok=True);manifest=out/'manifest.csv'
 if manifest.exists() and any(out.glob('generated_*.sdf')):
  with manifest.open(newline='') as f:return list(csv.DictReader(f))
 rows=[]
 for index,mol in enumerate(Chem.SDMolSupplier(str(source),removeHs=False,sanitize=True)):
  name=f'generated_{index:06d}';row={'name':name,'source_index':index,'status':'','smiles':'','formal_charge':'','uhf':'','heavy_atoms':'','rings':'','aromatic_rings':'','elements':'','error':''}
  try:
   if mol is None:raise ValueError('RDKit read failed')
   base=Chem.MolFromSmiles(Chem.MolToSmiles(Chem.RemoveHs(mol),canonical=True,isomericSmiles=False))
   if base is None:raise ValueError('SMILES round-trip failed')
   smiles=Chem.MolToSmiles(base,canonical=True,isomericSmiles=False);charge=sum(a.GetFormalCharge() for a in base.GetAtoms());uhf=sum(a.GetNumRadicalElectrons() for a in base.GetAtoms())
   mol3d=Chem.AddHs(base);params=AllChem.ETKDGv3();params.randomSeed=seed+index;params.useRandomCoords=True;params.maxIterations=1000
   if AllChem.EmbedMolecule(mol3d,params)!=0:raise ValueError('ETKDGv3 failed')
   if AllChem.MMFFHasAllMoleculeParams(mol3d):AllChem.MMFFOptimizeMolecule(mol3d,maxIters=1000)
   elif AllChem.UFFHasAllMoleculeParams(mol3d):AllChem.UFFOptimizeMolecule(mol3d,maxIters=1000)
   mol3d.SetProp('_Name',name);mol3d.SetProp('SMILES',smiles);mol3d.SetIntProp('formal_charge',charge);mol3d.SetIntProp('uhf',uhf)
   writer=Chem.SDWriter(str(out/f'{name}.sdf'));writer.write(mol3d);writer.close()
   row.update(status='ok',smiles=smiles,formal_charge=charge,uhf=uhf,heavy_atoms=base.GetNumHeavyAtoms(),rings=rdMolDescriptors.CalcNumRings(base),aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(base),elements=','.join(sorted({a.GetSymbol() for a in base.GetAtoms()})))
  except Exception as exc:row.update(status='failed',error=str(exc))
  rows.append(row)
  if (index+1)%50==0:print(json.dumps({'event':'prepare','completed':index+1}),flush=True)
 if not rows:raise RuntimeError('no molecules in source SDF')
 with manifest.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 (out/'summary.json').write_text(json.dumps({'total':len(rows),'exported':sum(r['status']=='ok' for r in rows),'failed':sum(r['status']!='ok' for r in rows)},indent=2))
 return rows

def call(cmd,work,env,log):
 with (work/log).open('w') as f:return subprocess.run(cmd,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200).returncode

def run_one(sdf,root,env):
 name=sdf.stem;work=root/name;work.mkdir(parents=True,exist_ok=True);cache=work/'result.json'
 if cache.exists():
  try:
   old=json.loads(cache.read_text())
   if old.get('status')=='ok':return old
  except Exception:pass
 shutil.copy2(sdf,work/sdf.name);result={'name':name,'status':'','xtb_returncode':None,'xtb4stda_returncode':None,'singlet_returncode':None,'triplet_returncode':None}
 def done(status,**extra):
  result.update(status=status,**extra);cache.write_text(json.dumps(result,indent=2));return result
 try:
  mol=Chem.MolFromMolFile(str(work/sdf.name),removeHs=False,sanitize=True)
  if mol is None:return done('input_invalid')
  charge=sum(a.GetFormalCharge() for a in mol.GetAtoms());uhf=sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms());result.update(formal_charge=charge,uhf=uhf)
  result['xtb_returncode']=call(['xtb',sdf.name,'--opt','loose','--cycles','500','--chrg',str(charge),'--uhf',str(uhf)],work,env,'xtb_opt.log')
  if result['xtb_returncode'] or not (work/'xtbopt.sdf').exists():return done('xtb_failed')
  result['xtb4stda_returncode']=call(['xtb4stda','xtbopt.sdf','-chrg',str(charge),'-uhf',str(uhf)],work,env,'xtb4stda.log')
  if result['xtb4stda_returncode'] or not (work/'wfn.xtb').exists():return done('xtb4stda_failed')
  result['singlet_returncode']=call(['stda','-xtb','-e','10.0'],work,env,'stda_singlet.log');singlets=states(work/'stda_singlet.log')
  if (work/'tda.dat').exists():shutil.copy2(work/'tda.dat',work/'tda_singlet.dat')
  result['triplet_returncode']=call(['stda','-xtb','-e','10.0','-t'],work,env,'stda_triplet.log');triplets=states(work/'stda_triplet.log')
  if (work/'tda.dat').exists():shutil.copy2(work/'tda.dat',work/'tda_triplet.dat')
  result.update(singlet_states_1_10=singlets,triplet_states_1_10=triplets)
  if not singlets or not triplets:return done('stda_parse_failed')
  strongest=max(singlets,key=lambda x:x['f_osc'])
  return done('ok',s1_energy_eV=singlets[0]['energy_eV'],s1_f_osc=singlets[0]['f_osc'],t1_energy_eV=triplets[0]['energy_eV'],strongest_singlet_state=strongest['state'],strongest_singlet_energy_eV=strongest['energy_eV'],strongest_singlet_f_osc=strongest['f_osc'])
 except subprocess.TimeoutExpired:return done('timeout')
 except Exception as exc:return done('error',error=str(exc))

def aggregate(work,metadata,target_s1,target_t1):
 meta={r['name']:r for r in metadata};results=[]
 for path in sorted(work.glob('generated_*/result.json')):
  try:results.append(json.loads(path.read_text()))
  except Exception:pass
 rows=[]
 for r in results:
  m=meta.get(r['name'],{});row={k:m.get(k,'') for k in ('name','source_index','smiles','heavy_atoms','rings','aromatic_rings','elements','formal_charge','uhf')};row.update({k:r.get(k,'') for k in ('status','s1_energy_eV','s1_f_osc','t1_energy_eV','strongest_singlet_state','strongest_singlet_energy_eV','strongest_singlet_f_osc')})
  if r.get('status')=='ok':row.update(s1_error_eV=r['s1_energy_eV']-target_s1,t1_error_eV=r['t1_energy_eV']-target_t1,joint_abs_error_eV=abs(r['s1_energy_eV']-target_s1)+abs(r['t1_energy_eV']-target_t1))
  else:row.update(s1_error_eV='',t1_error_eV='',joint_abs_error_eV='')
  rows.append(row)
 if rows:
  with (work/'results.csv').open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 (work/'results.json').write_text(json.dumps(results,indent=2));counts={}
 for r in results:counts[r['status']]=counts.get(r['status'],0)+1
 summary={'completed':len(results),'status_counts':counts,'target_s1_eV':target_s1,'target_t1_eV':target_t1};(work/'summary.json').write_text(json.dumps(summary,indent=2));return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--input-sdf',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--xtb-env',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=1);p.add_argument('--target-s1',type=float,required=True);p.add_argument('--target-t1',type=float,required=True);p.add_argument('--seed',type=int,default=20260918);a=p.parse_args()
 if a.workers!=4 or a.threads!=1:raise ValueError('required setting is 4 workers x 1 thread')
 a.output_dir.mkdir(parents=True,exist_ok=True);inputs=a.output_dir/'inputs';work=a.output_dir/'work';work.mkdir(exist_ok=True);metadata=prepare(a.input_sdf,inputs,a.seed);files=sorted(inputs.glob('generated_*.sdf'))
 env=os.environ.copy();env['PATH']=str(a.xtb_env/'bin')+':/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe:'+env['PATH'];env['LD_LIBRARY_PATH']=str(a.xtb_env/'lib')+':'+env.get('LD_LIBRARY_PATH','');env['XTB4STDAHOME']='/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home';env['OMP_STACKSIZE']='2G'
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):env[key]='1'
 start=time.time()
 with ThreadPoolExecutor(max_workers=4) as pool:
  futures={pool.submit(run_one,sdf,work,env):sdf for sdf in files}
  with (work/'progress.jsonl').open('a') as progress:
   for completed,future in enumerate(as_completed(futures),1):
    result=future.result();progress.write(json.dumps(result)+'\n');progress.flush();print(json.dumps({'event':'result','completed':completed,'total':len(files),'name':result['name'],'status':result['status'],'s1':result.get('s1_energy_eV'),'t1':result.get('t1_energy_eV'),'elapsed_s':time.time()-start}),flush=True)
 print(json.dumps({'event':'complete',**aggregate(work,metadata,a.target_s1,a.target_t1)},indent=2),flush=True)

if __name__=='__main__':main()
