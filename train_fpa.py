#!/usr/bin/env python3
"""Train the from-scratch heavy-atom + fused-polyaromatic-cell model."""
from __future__ import annotations
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from polyring_diffusion.chem import element_record
from polyring_diffusion.diffusion import LinearNoiseSchedule
from polyring_diffusion.fpa_data import FPAEntityDataset, collate_fpa
from polyring_diffusion.fpa_model import FPAEntityDenoiser


def masked_mean(value, mask):
    selected = value[mask]
    return selected.mean() if selected.numel() else value.new_zeros(())


def forward_batch(raw, model, schedule, device, training=True):
    b = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
    am, cm = b['atom_mask'], b['cell_mask']
    ap = am[:, :, None] & am[:, None, :]
    cp = cm[:, :, None] & cm[:, None, :]
    cross = cm[:, :, None] & am[:, None, :]
    t = torch.randint(schedule.timesteps, (am.shape[0],), device=device)
    apos, an = schedule.q_sample(b['atom_positions'], t)
    cpos, cn = schedule.q_sample(b['cell_centers'], t)
    norm, nn = schedule.q_sample(b['cell_normals'], t)
    tangent, tn = schedule.q_sample(b['cell_tangents'], t)
    at = schedule.corrupt_tokens(torch.where(am, b['atom_types'], model.atom_pad_token), t, model.atom_mask_token, am)
    aq = schedule.corrupt_tokens(model.charge_index(b['atom_charges']), t, model.charge_mask_token, am)
    ct = schedule.corrupt_tokens(torch.where(cm, b['cell_types'], model.cell_pad_token), t, model.cell_mask_token, cm)
    ab = schedule.corrupt_tokens(b['atom_bond_types'], t, 5, ap)
    ft = schedule.corrupt_tokens(b['fusion_types'], t, 2, cp)
    membership = schedule.corrupt_tokens(b['cell_membership'].long(), t, 2, cross)
    drop = torch.rand(am.shape[0], device=device) < model.condition_dropout if training else None
    pred = model(at, aq, apos, ab, am, ct, b['cell_sizes'], cpos, norm, tangent,
                 b['cell_radii'], ft, membership, cm, t, b['property'], drop)
    return b, pred, an, cn, nn, tn


def loss_terms(pack, model):
    b, p, an, cn, nn, tn = pack
    am, cm = b['atom_mask'], b['cell_mask']
    au = am[:, :, None] & am[:, None, :] & torch.triu(torch.ones(am.shape[1], am.shape[1], dtype=torch.bool, device=am.device), 1)[None]
    cu = cm[:, :, None] & cm[:, None, :] & torch.triu(torch.ones(cm.shape[1], cm.shape[1], dtype=torch.bool, device=cm.device), 1)[None]
    cross = cm[:, :, None] & am[:, None, :]
    fused = cu & b['fusion_types'].bool()
    return {
        'atom_coordinate': masked_mean((p['atom_coordinate_noise'] - an).square().sum(-1), am),
        'cell_center': masked_mean((p['cell_center_noise'] - cn).square().sum(-1), cm),
        'cell_normal': masked_mean((p['cell_normal_noise'] - nn).square().sum(-1), cm),
        'cell_tangent': masked_mean((p['cell_tangent_noise'] - tn).square().sum(-1), cm),
        'atom_type': F.cross_entropy(p['atom_type_logits'][am], b['atom_types'][am]),
        'atom_charge': F.cross_entropy(p['atom_charge_logits'][am], model.charge_index(b['atom_charges'])[am]),
        'atom_bond': F.cross_entropy(p['atom_bond_logits'][au], b['atom_bond_types'][au]) if au.any() else p['atom_type_logits'].new_zeros(()),
        'cell_type': F.cross_entropy(p['cell_type_logits'][cm], b['cell_types'][cm]) if cm.any() else p['atom_type_logits'].new_zeros(()),
        'cell_size': F.cross_entropy(p['cell_size_logits'][cm], b['cell_sizes'][cm]) if cm.any() else p['atom_type_logits'].new_zeros(()),
        'fusion': F.cross_entropy(p['fusion_logits'][cu], b['fusion_types'][cu]) if cu.any() else p['atom_type_logits'].new_zeros(()),
        'left_port': F.cross_entropy(p['left_port_logits'][fused], b['fusion_ports'][..., 0][fused]) if fused.any() else p['atom_type_logits'].new_zeros(()),
        'right_port': F.cross_entropy(p['right_port_logits'][fused], b['fusion_ports'][..., 1][fused]) if fused.any() else p['atom_type_logits'].new_zeros(()),
        'membership': F.cross_entropy(p['membership_logits'][cross], b['cell_membership'][cross].long()) if cross.any() else p['atom_type_logits'].new_zeros(()),
        'atom_count': F.cross_entropy(p['atom_count_logits'], am.sum(1).clamp_max(p['atom_count_logits'].shape[-1] - 1)),
        'cell_count': F.cross_entropy(p['cell_count_logits'], cm.sum(1).clamp_max(p['cell_count_logits'].shape[-1] - 1)),
    }


def save(path, model, optimizer, scheduler, epoch, step, config, dataset, metrics):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'epoch': epoch, 'step': step, 'config': config, 'atom_map': dataset.atom_map,
                'condition_stats': dataset.condition_stats, 'manifest': dataset.manifest, 'metrics': metrics}, tmp)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', default='scripts/qcdge_a10_fpa.yaml'); parser.add_argument('--resume')
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    device = torch.device(cfg.get('device', 'cuda'))
    train = FPAEntityDataset(cfg['data']['root'], 'train')
    val = FPAEntityDataset(cfg['data']['root'], 'val', train.property_mean, train.property_std, condition_stats=train.condition_stats)
    mc = dict(cfg['model']); mc.update(num_atom_types=train.num_atom_types, num_cell_types=len(train.template_hash_to_id),
        max_cell_ports=train.max_cell_ports, max_atoms=max(train.max_heavy_atoms, mc.pop('max_atoms', 0)),
        max_cells=max(train.max_fpa_cells, mc.pop('max_cells', 0)), atom_radii=[element_record(x).vdw_radius / 3.0 for x in train.atom_map])
    model = FPAEntityDenoiser(**mc).to(device)
    schedule = LinearNoiseSchedule(**cfg['diffusion']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['training']['learning_rate']), weight_decay=float(cfg['training']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])
    start = step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start, step = state['epoch'] + 1, state['step']
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=shuffle,
        num_workers=cfg['training'].get('num_workers', 0), collate_fn=collate_fpa, pin_memory=True)
    train_loader, val_loader = loader(train, True), loader(val, False)
    weights = cfg['training']['loss_weights']; run = Path(cfg['training']['output_dir']); run.mkdir(parents=True, exist_ok=True)
    (run / 'config.resolved.yaml').write_text(yaml.safe_dump(cfg)); log = (run / 'metrics.jsonl').open('a', buffering=1); best = math.inf
    for epoch in range(start, cfg['training']['epochs']):
        model.train(); sums = {}; batches = 0; started = time.time()
        for raw in train_loader:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda' and cfg['training'].get('bf16', True)):
                terms = loss_terms(forward_batch(raw, model, schedule, device), model); loss = sum(weights.get(k, 0) * v for k, v in terms.items())
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip']); optimizer.step()
            batches += 1; step += 1; sums['loss'] = sums.get('loss', 0) + float(loss.detach())
            for key, value in terms.items(): sums[key] = sums.get(key, 0) + float(value.detach())
        model.eval(); val_sum = 0.; val_batches = 0
        with torch.no_grad():
            for raw in val_loader:
                terms = loss_terms(forward_batch(raw, model, schedule, device, False), model)
                val_sum += float(sum(weights.get(k, 0) * v for k, v in terms.items())); val_batches += 1
                if val_batches >= cfg['training'].get('validation_batches', 10): break
        scheduler.step(); metrics = {'epoch': epoch, 'step': step, 'train_loss': sums['loss'] / batches,
            'val_loss': val_sum / val_batches, 'seconds': time.time() - started,
            **{f'train_{k}': v / batches for k, v in sums.items() if k != 'loss'}}
        print(json.dumps(metrics), flush=True); log.write(json.dumps(metrics) + '\n'); save(run / 'last.pt', model, optimizer, scheduler, epoch, step, cfg, train, metrics)
        if metrics['val_loss'] < best: best = metrics['val_loss']; save(run / 'best.pt', model, optimizer, scheduler, epoch, step, cfg, train, metrics)


if __name__ == '__main__': main()
