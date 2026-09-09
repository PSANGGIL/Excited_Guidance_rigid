#!/usr/bin/env python3
"""Train conditional PolyRing Diffusion on the sharded rigid-ring database."""
from __future__ import annotations
import argparse,json,math,random,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
from torch.utils.data import DataLoader
from polyring_diffusion.chemical_losses import ChemicalConstraintLoss
from polyring_diffusion.diffusion import LinearNoiseSchedule
from polyring_diffusion.graph import ConnectionType
from polyring_diffusion.model import RingDenoiser
from polyring_diffusion.real_data import PropertyBandSampler,RigidRingDataset,ValidityCurriculumSampler,collate_rigid

def masked_mean(x,mask):
 selected=x[mask];return selected.mean() if selected.numel() else x.new_zeros(())
def make_batch(batch,model,schedule,device,training=True):
 batch={k:v.to(device,non_blocking=True) for k,v in batch.items()};mask=batch['node_mask'];pair=mask[:,:,None]&mask[:,None,:];pair&=~torch.eye(mask.shape[1],dtype=torch.bool,device=device)[None];upper=pair&torch.triu(torch.ones(mask.shape[1],mask.shape[1],dtype=torch.bool,device=device),1)[None]
 types=torch.where(mask,batch['ring_types'],torch.full_like(batch['ring_types'],model.ring_pad_token));t=torch.randint(schedule.timesteps,(mask.shape[0],),device=device);xc,cn=schedule.q_sample(batch['centers'],t);xn,nn=schedule.q_sample(batch['normals'],t);rt=schedule.corrupt_tokens(types,t,model.ring_mask_token,mask);ct=schedule.corrupt_tokens(batch['connection_types'],t,model.connection_mask_token,pair);drop=(torch.rand(mask.shape[0],device=device)<model.condition_dropout) if training and model.condition_dim else None;pred=model(rt,xc,xn,ct,mask,t,batch['property'],drop);return batch,mask,pair,upper,cn,nn,pred
def loss_terms(args,class_weights,chemical=None):
 batch,mask,pair,upper,cn,nn,pred=args;terms={}
 if 'property_pred' in pred:terms['property']=F.smooth_l1_loss(pred['property_pred'],batch['property'])
 terms['center']=masked_mean((pred['center_noise']-cn).square().sum(-1),mask);terms['normal']=masked_mean((pred['normal_noise']-nn).square().sum(-1),mask);terms['ring']=F.cross_entropy(pred['ring_logits'][mask],batch['ring_types'][mask]);terms['connection']=F.cross_entropy(pred['connection_logits'][upper],batch['connection_types'][upper],weight=class_weights);material=pair&batch['connection_types'].ne(int(ConnectionType.NONE));terms['port']=F.cross_entropy(pred['port_logits'][material],batch['ports'][material]) if material.any() else terms['center'].new_zeros(())
 if chemical is not None:
  aux=chemical(pred['connection_logits'],pred['port_logits'],batch['ring_types'],mask)
  for key in ('valence','kekule','port_validity'):terms[key]=aux[key]
 return terms
def total_loss(terms,weights):return sum(float(weights.get(k,0))*v for k,v in terms.items())
def make_loader(ds,batch_size,workers,shuffle=False,sampler=None):return DataLoader(ds,batch_size=batch_size,shuffle=shuffle if sampler is None else False,sampler=sampler,num_workers=workers,pin_memory=True,collate_fn=collate_rigid,persistent_workers=workers>0)
def save(path,model,opt,scheduler,epoch,step,cfg,train_ds,metrics):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,'config':cfg,'property_mean':train_ds.property_mean,'property_std':train_ds.property_std,'template_features':train_ds.template_features,'manifest':train_ds.manifest,'metrics':metrics},tmp);tmp.replace(path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='scripts/real.yaml');ap.add_argument('--resume');ap.add_argument('--seed-model');a=ap.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());seed=int(cfg['seed']);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if not torch.cuda.is_available():raise RuntimeError('CUDA is required for real training')
 device=torch.device(cfg.get('device','cuda'));torch.cuda.set_device(device);print('loading train dataset...',flush=True);train_ds=RigidRingDataset(cfg['data']['root'],'train');print('loading val dataset...',flush=True);val_ds=RigidRingDataset(cfg['data']['root'],'val',train_ds.property_mean,train_ds.property_std)
 model_cfg=dict(cfg['model']);model_cfg.update(num_templates=len(train_ds.template_hash_to_id),template_features=train_ds.template_features.tolist());model=RingDenoiser(**model_cfg).to(device);schedule=LinearNoiseSchedule(**cfg['diffusion']).to(device)
 signatures=[x['signature'] for x in sorted(train_ds.manifest['templates'],key=lambda x:x['id'])];chemical=ChemicalConstraintLoss(signatures,model.max_ports).to(device) if cfg['training'].get('chemical_constraints',{}).get('enabled',False) else None
 opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=int(cfg['training']['epochs']));start_epoch=step=0
 if a.seed_model:
  state=torch.load(a.seed_model,map_location=device,weights_only=False);incompat=model.load_state_dict(state['model'],strict=False);print('seeded model weights from',a.seed_model,'missing',list(incompat.missing_keys),'unexpected',list(incompat.unexpected_keys),flush=True)
 if a.resume:
  state=torch.load(a.resume,map_location=device,weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);start_epoch=state['epoch']+1;step=state['step'];print('resumed',a.resume,flush=True)
 tc=cfg['training'].get('curriculum',{});curriculum=ValidityCurriculumSampler(train_ds,tc.get('fractions',[.4,.7,1.]),tc.get('milestones',[0,10,20]),seed) if tc.get('enabled',False) else None
 pb=cfg['training'].get('property_band',{});band_sampler=PropertyBandSampler(train_ds,pb.get('low',1.),pb.get('high',2.),pb.get('fraction',.6),seed) if pb.get('enabled',False) else None
 if curriculum is not None and band_sampler is not None:raise ValueError('curriculum and property_band cannot both be enabled')
 sampler=band_sampler or curriculum
 bs=int(cfg['training']['batch_size']);workers=int(cfg['training'].get('num_workers',0));train_loader=make_loader(train_ds,bs,workers,shuffle=sampler is None,sampler=sampler);val_loader=make_loader(val_ds,bs,workers);weights=cfg['training']['loss_weights'];class_weights=torch.tensor(cfg['training']['connection_class_weights'],device=device)
 run=Path(cfg['training']['output_dir']);run.mkdir(parents=True,exist_ok=True);(run/'config.resolved.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False));(run/'dataset_summary.json').write_text(json.dumps({'train':len(train_ds),'val':len(val_ds),'templates':model.num_templates,'max_rings':train_ds.max_rings,'property_mean':train_ds.property_mean,'property_std':train_ds.property_std},indent=2));best=math.inf;patience=0;metrics_file=(run/'metrics.jsonl').open('a',buffering=1);use_amp=bool(cfg['training'].get('bf16',True));print({'device':torch.cuda.get_device_name(device),'batch_size':bs,'chemical_constraints':chemical is not None,'curriculum':curriculum is not None,'parameters':sum(p.numel() for p in model.parameters())},flush=True)
 for epoch in range(start_epoch,int(cfg['training']['epochs'])):
  if sampler:sampler.set_epoch(epoch)
  model.train();sums={'loss':0.};seen=0;started=time.time();torch.cuda.reset_peak_memory_stats(device)
  for raw in train_loader:
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=use_amp):args=make_batch(raw,model,schedule,device,True);terms=loss_terms(args,class_weights,chemical);loss=total_loss(terms,weights)
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg['training']['grad_clip']));opt.step();step+=1;seen+=1;sums['loss']+=float(loss.detach())
   for k,v in terms.items():sums[k]=sums.get(k,0.)+float(v.detach())
  model.eval();val_sums={'loss':0.};vb=0;torch.manual_seed(seed+100000)
  with torch.no_grad():
   for raw in val_loader:
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=use_amp):terms=loss_terms(make_batch(raw,model,schedule,device,False),class_weights,chemical);value=total_loss(terms,weights)
    val_sums['loss']+=float(value);vb+=1
    for k,v in terms.items():val_sums[k]=val_sums.get(k,0.)+float(v)
    if vb>=int(cfg['training'].get('validation_batches',0) or len(val_loader)):break
  scheduler.step();val=val_sums['loss']/max(vb,1);metrics={'epoch':epoch,'step':step,'train_loss':sums['loss']/seen,'val_loss':val,'seconds':time.time()-started,'lr':scheduler.get_last_lr()[0],'peak_gpu_gib':torch.cuda.max_memory_allocated(device)/2**30,'curriculum_fraction':curriculum.fraction() if curriculum else 1.0,'property_band_fraction':band_sampler.band_fraction if band_sampler else 0.0};metrics.update({f'train_{k}':v/seen for k,v in sums.items() if k!='loss'});metrics.update({f'val_{k}':v/max(vb,1) for k,v in val_sums.items() if k!='loss'});print(json.dumps(metrics),flush=True);metrics_file.write(json.dumps(metrics)+'\n');save(run/'last.pt',model,opt,scheduler,epoch,step,cfg,train_ds,metrics)
  if val<best:best=val;patience=0;save(run/'best.pt',model,opt,scheduler,epoch,step,cfg,train_ds,metrics)
  else:patience+=1
  if patience>=int(cfg['training'].get('early_stopping_patience',20)):print('early stopping',flush=True);break
 metrics_file.close()
if __name__=='__main__':main()
