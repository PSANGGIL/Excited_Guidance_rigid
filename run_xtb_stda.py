#!/usr/bin/env python3
"""Run xTB geometry optimization and sTDA-xTB for exported rigid samples."""
import argparse,csv,json,os,re,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROW=re.compile(r'^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\b')
def states(path):
 found=False;out=[]
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
def run_one(sdf,out,env):
 name=sdf.stem;work=out/name;work.mkdir(parents=True,exist_ok=True);local=work/sdf.name;shutil.copy2(sdf,local)
 def call(cmd,log):
  with (work/log).open('w') as f:return subprocess.run(cmd,cwd=work,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7200).returncode
 result={'name':name,'input_sdf':str(sdf),'status':'','xtb_returncode':None,'xtb4stda_returncode':None,'stda_returncode':None}
 try:
  result['xtb_returncode']=call(['xtb',sdf.name,'--opt','loose','--cycles','500'],'xtb_opt.log')
  optimized=work/'xtbopt.sdf'
  if result['xtb_returncode']!=0 or not optimized.exists():result['status']='xtb_failed';return result
  result['xtb4stda_returncode']=call(['xtb4stda','xtbopt.sdf'],'xtb4stda.log')
  result['stda_returncode']=call(['stda','-xtb','-e','10.0'],'stda.log')
  parsed=states(work/'stda.log') if (work/'stda.log').exists() else []
  if not parsed:result['status']='stda_parse_failed';return result
  top=max(parsed,key=lambda x:x['f_osc']);result.update(top);result['states_1_10']=parsed;result['status']='ok';return result
 except subprocess.TimeoutExpired:result['status']='timeout';return result
 except Exception as e:result['status']='error';result['error']=str(e);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=3);p.add_argument('--xtb-env',type=Path,required=True);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env['PATH']=str(a.xtb_env/'bin')+':/home/jovyan/cpu-only-1-datavol-1/software/xtb4stda_home/exe:'+env['PATH'];env['LD_LIBRARY_PATH']=str(a.xtb_env/'lib')+':'+env.get('LD_LIBRARY_PATH','');env['OMP_NUM_THREADS']='4';env['MKL_NUM_THREADS']='4';env['OMP_STACKSIZE']='2G'
 files=sorted(a.input_dir.glob('*.sdf'));results=[]
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(run_one,f,a.output_dir,env):f for f in files}
  for future in as_completed(futures):
   r=future.result();results.append(r);print(json.dumps({k:r.get(k) for k in ('name','status','f_osc','wavelength_nm')}),flush=True)
 results.sort(key=lambda x:x['name']);(a.output_dir/'results.json').write_text(json.dumps(results,indent=2))
 fields=['name','status','state','energy_eV','wavelength_nm','f_osc','xtb_returncode','xtb4stda_returncode','stda_returncode']
 with (a.output_dir/'results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k,'') for k in fields} for r in results)
 print(json.dumps({'total':len(results),'ok':sum(r['status']=='ok' for r in results),'failed':sum(r['status']!='ok' for r in results)},indent=2))
if __name__=='__main__':main()
