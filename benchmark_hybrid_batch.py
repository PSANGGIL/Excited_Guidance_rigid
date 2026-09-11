#!/usr/bin/env python3
import argparse,json,torch,yaml
from torch.utils.data import DataLoader
import train_hybrid
from polyring_diffusion.diffusion import LinearNoiseSchedule
from polyring_diffusion.hybrid_data import HybridRingAtomDataset,collate_hybrid
from polyring_diffusion.hybrid_model import HybridRingAtomDenoiser

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,required=True);p.add_argument('--config',required=True);a=p.parse_args();cfg=yaml.safe_load(open(a.config));device=torch.device(cfg.get('device','cuda'));ds=HybridRingAtomDataset(cfg['data']['root'],'train');mc=dict(cfg['model']);mc.update(num_atom_types=ds.num_atom_types,max_free_atoms=mc.pop('max_free_atoms'),num_templates=len(ds.template_hash_to_id),template_features=ds.template_features.tolist());model=HybridRingAtomDenoiser(**mc).to(device);train_hybrid.model=model;raw=next(iter(DataLoader(ds,batch_size=a.batch_size,shuffle=True,collate_fn=collate_hybrid,num_workers=0)));schedule=LinearNoiseSchedule(**cfg['diffusion']).to(device);torch.cuda.reset_peak_memory_stats()
 with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['training'].get('bf16',True)):
  pack=train_hybrid.batch_forward(raw,model,schedule,device,True);terms=train_hybrid.losses(pack,torch.tensor(cfg['training']['connection_class_weights'],device=device),None);loss=sum(cfg['training']['loss_weights'].get(k,0)*v for k,v in terms.items())
 loss.backward();torch.cuda.synchronize();print(json.dumps({'batch_size':a.batch_size,'loss':float(loss.detach()),'allocated_peak_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_peak_gib':torch.cuda.max_memory_reserved()/2**30,'terms':{k:float(v.detach()) for k,v in terms.items()}}))
if __name__=='__main__':main()
