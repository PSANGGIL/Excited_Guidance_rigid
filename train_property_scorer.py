#!/usr/bin/env python3
"""Train a forward ring-graph surrogate for log10 oscillator strength."""
import argparse,json,math
from pathlib import Path
import torch
from polyring_diffusion.real_data import RigidRingDataset
from polyring_diffusion.validity_scorer import FEATURE_NAMES,PropertyScorer,feature_vector
KIND=['NONE','SINGLE','FUSED','SPIRO']
def record(ds,x):
 templates=sorted(ds.manifest['templates'],key=lambda z:z['id']);rings=[{'signature':templates[int(t)]['signature']} for t in x['ring_types']];edges=[];n=len(rings)
 for i in range(n):
  for j in range(i+1,n):
   k=int(x['connection_types'][i,j])
   if k:edges.append({'left':i,'right':j,'kind':KIND[k],'left_port':int(x['ports'][i,j]),'right_port':int(x['ports'][j,i]),'score':0.})
 return {'normalized_condition':0.,'rings':rings,'connections':edges}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',default='data_ring_rigid');p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=43);a=p.parse_args();torch.manual_seed(a.seed);ds=RigidRingDataset(a.root,'train');xs=[];ys=[]
 for i in range(len(ds)):
  x=ds[i];xs.append(feature_vector(record(ds,x)));ys.append(math.log10(max(float(x['raw_property']),0.)+1e-6))
 x=torch.tensor(xs);y=torch.tensor(ys);order=torch.randperm(len(y));cut=int(.9*len(y));tr,va=order[:cut],order[cut:];mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-5);x=(x-mean)/std;model=PropertyScorer();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);best=None
 for epoch in range(a.epochs):
  model.train();perm=tr[torch.randperm(len(tr))]
  for ix in perm.split(4096):loss=torch.nn.functional.smooth_l1_loss(model(x[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
  model.eval()
  with torch.no_grad():pred=model(x[va]);vl=float(torch.nn.functional.mse_loss(pred,y[va]));mae=float((pred-y[va]).abs().mean());corr=float(torch.corrcoef(torch.stack((pred,y[va])))[0,1])
  if best is None or vl<best[0]:best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch,mae,corr)
 a.output.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':best[1],'mean':mean,'std':std,'feature_names':FEATURE_NAMES,'hidden':128,'epoch':best[2],'val_mse':best[0],'val_mae_log10':best[3],'val_correlation':best[4]},a.output);print(json.dumps({'rows':len(y),'best_epoch':best[2],'val_mse':best[0],'val_mae_log10':best[3],'val_correlation':best[4]},indent=2))
if __name__=='__main__':main()
