#!/usr/bin/env python3
"""Train a molecule-level RDKit validity reward model from candidate JSONL."""
import argparse,json,random
from pathlib import Path
import torch
from polyring_diffusion.validity_scorer import FEATURE_NAMES,ValidityScorer

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=80);p.add_argument('--seed',type=int,default=43);a=p.parse_args();random.seed(a.seed);torch.manual_seed(a.seed)
 rows=[json.loads(x) for x in a.input.read_text().splitlines()];x=torch.tensor([r['features'] for r in rows],dtype=torch.float32);y=torch.tensor([r['valid'] for r in rows],dtype=torch.float32);order=torch.randperm(len(y));cut=int(.8*len(y));tr,va=order[:cut],order[cut:];mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(1e-5);x=(x-mean)/std
 model=ValidityScorer();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);pos=(y[tr]==0).sum()/max(int((y[tr]==1).sum()),1);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pos)
 best=None
 for epoch in range(a.epochs):
  model.train();logit=model(x[tr]);loss=lossfn(logit,y[tr]);opt.zero_grad();loss.backward();opt.step();model.eval()
  with torch.no_grad():
   vl=float(lossfn(model(x[va]),y[va]));prob=model(x[va]).sigmoid();pred=prob>=.5;acc=float((pred==y[va].bool()).float().mean());tp=int((pred&y[va].bool()).sum());fp=int((pred&~y[va].bool()).sum());fn=int((~pred&y[va].bool()).sum());precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
  if best is None or vl<best[0]:best=(vl,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch,acc,precision,recall)
 model.load_state_dict(best[1]);a.output.parent.mkdir(parents=True,exist_ok=True);torch.save({'model':model.state_dict(),'mean':mean,'std':std,'feature_names':FEATURE_NAMES,'hidden':64,'epoch':best[2],'val_loss':best[0],'val_accuracy':best[3],'val_precision':best[4],'val_recall':best[5],'rows':len(rows),'positive_rate':float(y.mean())},a.output);print(json.dumps({'rows':len(rows),'positive_rate':float(y.mean()),'best_epoch':best[2],'val_loss':best[0],'val_accuracy':best[3],'val_precision':best[4],'val_recall':best[5]},indent=2))
if __name__=='__main__':main()
