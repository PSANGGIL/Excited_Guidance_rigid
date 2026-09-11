#!/usr/bin/env python3
"""Train joint rigid-ring and free-heavy-atom conditional diffusion."""
from __future__ import annotations
import argparse,json,math,random,time
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F,yaml
from torch.utils.data import DataLoader
from polyring_diffusion.chemical_losses import ChemicalConstraintLoss
from polyring_diffusion.diffusion import LinearNoiseSchedule
from polyring_diffusion.graph import ConnectionType
from polyring_diffusion.hybrid_data import HybridRingAtomDataset,collate_hybrid
from polyring_diffusion.hybrid_model import HybridRingAtomDenoiser
from polyring_diffusion.chem import element_record

def mean(x,m):
 x=x[m];return x.mean() if x.numel() else m.new_zeros((),dtype=torch.float32)
def batch_forward(raw,model,schedule,device,training,condition_override=None,force_unconditional=False):
 b={k:v.to(device,non_blocking=True) for k,v in raw.items()};rm=b['node_mask'];am=b['free_atom_mask'];rp=rm[:,:,None]&rm[:,None,:];ap=am[:,:,None]&am[:,None,:];cross=rm[:,:,None]&am[:,None,:];ru=rp&torch.triu(torch.ones(rm.shape[1],rm.shape[1],dtype=torch.bool,device=device),1)[None];au=ap&torch.triu(torch.ones(am.shape[1],am.shape[1],dtype=torch.bool,device=device),1)[None];t=torch.randint(schedule.timesteps,(rm.shape[0],),device=device);rc,center_noise=schedule.q_sample(b['centers'],t);rnorm,normal_noise=schedule.q_sample(b['normals'],t);rtan,tangent_noise=schedule.q_sample(b['ring_tangents'],t);ac,atom_noise=schedule.q_sample(b['free_positions'],t);rt=schedule.corrupt_tokens(torch.where(rm,b['ring_types'],model.ring_model.ring_pad_token),t,model.ring_model.ring_mask_token,rm);rct=schedule.corrupt_tokens(b['connection_types'],t,model.ring_model.connection_mask_token,rp);at=schedule.corrupt_tokens(torch.where(am,b['free_atom_types'],model.atom_pad_token),t,model.atom_mask_token,am);qi=model.charge_index(b['free_atom_charges']);aq=schedule.corrupt_tokens(qi,t,model.charge_mask_token,am);ab=schedule.corrupt_tokens(b['free_bond_types'],t,5,ap);attach=schedule.corrupt_tokens(b['attachment_types'],t,5,cross);condition=b['property'] if condition_override is None else condition_override.to(device);drop=torch.ones(rm.shape[0],dtype=torch.bool,device=device) if force_unconditional else (torch.rand(rm.shape[0],device=device)<model.ring_model.condition_dropout if training else None);pred=model((rt,rc,rnorm,rtan,rct,rm),at,aq,ac,ab,attach,am,t,condition,drop);return b,pred,rm,am,ru,au,cross,center_noise,normal_noise,tangent_noise,atom_noise
def losses(pack,class_weights,chemical):
 b,p,rm,am,ru,au,cross,center_noise,normal_noise,tangent_noise,atom_noise=pack;o={};o['center']=mean((p['center_noise']-center_noise).square().sum(-1),rm);o['normal']=mean((p['normal_noise']-normal_noise).square().sum(-1),rm);o['tangent']=mean((p['tangent_noise']-tangent_noise).square().sum(-1),rm);o['ring']=F.cross_entropy(p['ring_logits'][rm],b['ring_types'][rm]);o['connection']=F.cross_entropy(p['connection_logits'][ru],b['connection_types'][ru],weight=class_weights);material=(rm[:,:,None]&rm[:,None,:])&b['connection_types'].gt(0)&b['connection_types'].lt(model.ring_model.connection_mask_token-1);o['port']=F.cross_entropy(p['port_logits'][material],b['ports'][material]) if material.any() else o['ring'].new_zeros(());complex_pair=(rm[:,:,None]&rm[:,None,:])&b['connection_types'].eq(model.ring_model.connection_mask_token-1);o['junction_port']=F.binary_cross_entropy_with_logits(p['junction_port_logits'][complex_pair],b['junction_ports'][complex_pair].to(p['junction_port_logits'].dtype)) if complex_pair.any() else o['ring'].new_zeros(());o['atom_coordinate']=mean((p['atom_coordinate_noise']-atom_noise).square().sum(-1),am);o['atom_type']=F.cross_entropy(p['atom_type_logits'][am],b['free_atom_types'][am]) if am.any() else o['ring'].new_zeros(());o['atom_charge']=F.cross_entropy(p['atom_charge_logits'][am],model.charge_index(b['free_atom_charges'])[am]) if am.any() else o['ring'].new_zeros(());o['atom_bond']=F.cross_entropy(p['atom_bond_logits'][au],b['free_bond_types'][au]) if au.any() else o['ring'].new_zeros(());o['attachment']=F.cross_entropy(p['attachment_logits'][cross],b['attachment_types'][cross]) if cross.any() else o['ring'].new_zeros(());attached=cross&b['attachment_types'].ne(0);o['attachment_port']=F.cross_entropy(p['attachment_port_logits'][attached],b['attachment_ports'][attached]) if attached.any() else o['ring'].new_zeros(());o['atom_count']=F.cross_entropy(p['free_atom_count_logits'],b['free_atom_count'].clamp_max(model.max_free_atoms));
 if 'property_pred' in p:o['property']=F.smooth_l1_loss(p['property_pred'],b['property'])
 if chemical:
  aux=chemical(p['connection_logits'],p['port_logits'],b['ring_types'],rm);o.update({k:aux[k] for k in ('valence','kekule','port_validity')})
 return o

DIAGNOSTIC_TERMS=('ring','connection','junction_port','atom_type','atom_bond','attachment','attachment_port','atom_count')
DIAGNOSTIC_HEADS=('ring_logits','connection_logits','junction_port_logits','atom_type_logits','atom_bond_logits','attachment_logits','attachment_port_logits','free_atom_count_logits')

def shifted_condition(condition,mode):
 shifted=condition.clone();rolled=condition.roll(1,0)
 if mode=='full':return rolled
 index={'s1_energy':0,'s1_os':20}[mode]
 shifted[:,index]=rolled[:,index]
 return shifted

def condition_diagnostics(raw,model,schedule,device,class_weights,chemical,weights):
 """Measure condition use with identical diffusion noise and counterfactual conditions."""
 cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
 def restore():
  torch.set_rng_state(cpu_rng)
  if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng,device)
 restore();base_pack=batch_forward(raw,model,schedule,device,False);base_terms=losses(base_pack,class_weights,chemical);base_pred=base_pack[1];result={}
 for mode in ('full','s1_energy','s1_os'):
  restore();variant_pack=batch_forward(raw,model,schedule,device,False,shifted_condition(raw['property'],mode));variant_terms=losses(variant_pack,class_weights,chemical);weighted=0.0
  for key in DIAGNOSTIC_TERMS:
   if key not in base_terms or key not in variant_terms:continue
   gain=float((variant_terms[key]-base_terms[key]).detach());result[f'condition_gain_{mode}_{key}']=gain;weighted+=weights.get(key,0)*gain
  result[f'condition_gain_{mode}_weighted']=weighted
  del variant_pack,variant_terms
 restore();unconditional=batch_forward(raw,model,schedule,device,False,force_unconditional=True)[1]
 for key in DIAGNOSTIC_HEADS:
  if key in base_pred and key in unconditional:result[f'cfg_response_{key}_rms']=float((base_pred[key].float()-unconditional[key].float()).square().mean().sqrt())
 restore()
 return result

def save(path,model,opt,scheduler,epoch,step,cfg,ds,metrics):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'step':step,'config':cfg,'property_mean':ds.property_mean,'property_std':ds.property_std,'template_features':ds.template_features,'manifest':ds.manifest,'atom_map':ds.atom_map,'condition_stats':ds.condition_stats,'metrics':metrics},tmp);tmp.replace(path)
def main():
 global model
 ap=argparse.ArgumentParser();ap.add_argument('--config',default='scripts/hybrid_finetune.yaml');ap.add_argument('--seed-model');ap.add_argument('--resume');a=ap.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=torch.device(cfg.get('device','cuda'));train=HybridRingAtomDataset(cfg['data']['root'],'train');val=HybridRingAtomDataset(cfg['data']['root'],'val',train.property_mean,train.property_std,condition_stats=train.condition_stats);mc=dict(cfg['model']);mc.update(num_atom_types=train.num_atom_types,max_free_atoms=mc.pop('max_free_atoms',32),num_templates=len(train.template_hash_to_id),template_features=train.template_features.tolist(),atom_radii=[element_record(x).vdw_radius/3.0 for x in train.atom_map]);model=HybridRingAtomDenoiser(**mc).to(device);schedule=LinearNoiseSchedule(**cfg['diffusion']).to(device);signatures=[x['signature'] for x in sorted(train.manifest['templates'],key=lambda x:x['id'])];chemical=ChemicalConstraintLoss(signatures,model.ring_model.max_ports).to(device);opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['training']['learning_rate']),weight_decay=float(cfg['training']['weight_decay']));sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['training']['epochs']);start=step=0
 if a.seed_model:
  if model.ring_model.full_frame_orientation or model.entity_size_aware:raise ValueError('rigid-frame/size-aware model changes the input representation and must be trained from scratch; --seed-model is not allowed')
  state=torch.load(a.seed_model,map_location=device,weights_only=False);source=state['model'];target=model.state_dict();mapped={k:v for k,v in source.items() if k in target and target[k].shape==v.shape}
  if not mapped:mapped={'ring_model.'+k:v for k,v in source.items() if 'ring_model.'+k in target and target['ring_model.'+k].shape==v.shape}
  inc=model.load_state_dict(mapped,strict=False);print('seeded compatible parameters',len(mapped),'new parameters',len(inc.missing_keys),flush=True)
 if a.resume:
  state=torch.load(a.resume,map_location=device,weights_only=False);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sch.load_state_dict(state['scheduler']);sch.T_max=cfg['training']['epochs'];start=state['epoch']+1;step=state['step']
 loader=lambda ds,shuffle:DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=shuffle,num_workers=cfg['training'].get('num_workers',0),collate_fn=collate_hybrid,pin_memory=True);tl=loader(train,True);vl=loader(val,False);weights=cfg['training']['loss_weights'];cw=torch.tensor(cfg['training']['connection_class_weights'],device=device);run=Path(cfg['training']['output_dir']);run.mkdir(parents=True,exist_ok=True);(run/'config.resolved.yaml').write_text(yaml.safe_dump(cfg));log=(run/'metrics.jsonl').open('a',buffering=1);best=min((json.loads(x)['val_loss'] for x in (run/'metrics.jsonl').read_text().splitlines() if x.strip()),default=math.inf)
 for epoch in range(start,cfg['training']['epochs']):
  model.train();sums={};nb=0;started=time.time()
  for raw in tl:
   with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['training'].get('bf16',True)):term=losses(batch_forward(raw,model,schedule,device,True),cw,chemical);loss=sum(weights.get(k,0)*v for k,v in term.items())
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['training']['grad_clip']);opt.step();step+=1;nb+=1;sums['loss']=sums.get('loss',0)+float(loss.detach());[sums.__setitem__(k,sums.get(k,0)+float(v.detach())) for k,v in term.items()]
  model.eval();vs={};vn=0;diag={}
  with torch.no_grad():
   for raw in vl:
    if vn==0 and epoch%cfg['training'].get('condition_diagnostics_every',10)==0:
     with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['training'].get('bf16',True)):diag=condition_diagnostics(raw,model,schedule,device,cw,chemical,weights)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=cfg['training'].get('bf16',True)):term=losses(batch_forward(raw,model,schedule,device,False),cw,chemical);loss=sum(weights.get(k,0)*v for k,v in term.items())
    vs['loss']=vs.get('loss',0)+float(loss);[vs.__setitem__(k,vs.get(k,0)+float(v)) for k,v in term.items()];vn+=1
    if vn>=cfg['training'].get('validation_batches',10):break
  sch.step();m={'epoch':epoch,'step':step,'train_loss':sums['loss']/nb,'val_loss':vs['loss']/vn,'seconds':time.time()-started,**{'train_'+k:v/nb for k,v in sums.items() if k!='loss'},**{'val_'+k:v/vn for k,v in vs.items() if k!='loss'},**diag};print(json.dumps(m),flush=True);log.write(json.dumps(m)+'\n');save(run/'last.pt',model,opt,sch,epoch,step,cfg,train,m)
  if m['val_loss']<best:best=m['val_loss'];save(run/'best.pt',model,opt,sch,epoch,step,cfg,train,m)
if __name__=='__main__':main()
