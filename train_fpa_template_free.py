#!/usr/bin/env python3
"""Train template-free heavy-atom/ring-cell diffusion from scratch."""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from polyring_diffusion.chem import element_record
from polyring_diffusion.fpa_data_template_free import TemplateFreeFPAEntityDataset, collate_template_free_fpa
from polyring_diffusion.fpa_diffusion_v2 import extract, masked_center
from polyring_diffusion.fpa_model_template_free import TemplateFreeFPAEntityDenoiser
from polyring_diffusion.so3_diffusion import MolecularDiffusion


def mean_masked(value, mask):
    selected = value[mask]
    return selected.mean() if selected.numel() else value.sum()*0


def cross_entropy(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def q_scalar(clean, t, diffusion, mask):
    eps = torch.randn_like(clean)*mask.unsqueeze(-1)
    abar = extract(diffusion.abar, t, clean.ndim)
    return (abar.sqrt()*clean+(1-abar).sqrt()*eps)*mask.unsqueeze(-1), eps


def forward_batch(raw, model, diffusion, device, training=True):
    batch = {k: v.to(device, non_blocking=True) for k, v in raw.items()}
    atom_mask, cell_mask = batch["atom_mask"], batch["cell_mask"]
    atom_pair = atom_mask[:, :, None] & atom_mask[:, None, :]
    cell_pair = cell_mask[:, :, None] & cell_mask[:, None, :]
    cross = cell_mask[:, :, None] & atom_mask[:, None, :]
    t = torch.randint(diffusion.timesteps, (len(atom_mask),), device=device)
    atom_pos, atom_eps = diffusion.q_coordinates(batch["atom_positions"], t, atom_mask)
    cell_pos, cell_eps = q_scalar(batch["cell_centers"], t, diffusion, cell_mask)
    log_radius, radius_eps = q_scalar(batch["cell_radii"].clamp_min(.1).log().unsqueeze(-1), t, diffusion, cell_mask)
    frame, rotation_score = diffusion.q_frames(batch["cell_frames"], t, cell_mask)
    atom_type = diffusion.q_tokens(torch.where(atom_mask, batch["atom_types"], model.atom_pad_token), t, model.atom_mask_token, atom_mask)
    charge = diffusion.q_tokens(model.charge_index(batch["atom_charges"]), t, model.charge_mask_token, atom_mask)
    hydrogen = diffusion.q_tokens(batch["atom_hydrogens"].clamp(0, 4), t, model.hydrogen_mask_token, atom_mask)
    radical = diffusion.q_tokens(batch["atom_radicals"].clamp(0, 2), t, model.radical_mask_token, atom_mask)
    size_clean = model.encode_size(batch["cell_sizes"])
    size_clean = torch.where(cell_mask, size_clean, model.size_pad_token)
    size = diffusion.q_tokens(size_clean, t, model.size_mask_token, cell_mask)
    aromatic = diffusion.q_tokens(batch["cell_aromatic"], t, model.aromatic_mask_token, cell_mask)
    bond = diffusion.q_symmetric_tokens(batch["atom_bond_types"], t, 5, atom_pair)
    relation = diffusion.q_symmetric_tokens(batch["cell_relations"], t, model.relation_mask_token, cell_pair)
    member = diffusion.q_tokens(batch["cell_membership"].long(), t, 2, cross)
    drop = torch.rand(len(atom_mask), device=device) < model.condition_dropout if training else None
    prediction = model(atom_type, charge, hydrogen, radical, atom_pos, bond, atom_mask,
        size, aromatic, cell_pos, frame, log_radius.squeeze(-1), relation, member,
        cell_mask, t, batch["property"], drop)
    targets = atom_eps, cell_eps, radius_eps.squeeze(-1), rotation_score
    noisy = atom_pos, cell_pos, log_radius.squeeze(-1), frame
    return batch, prediction, targets, t, noisy


def loss_terms(pack, model, diffusion, covalent_radii, max_valence, positive_valence_bonus=None):
    batch, pred, targets, t, noisy = pack
    atom_eps, cell_eps, radius_eps, rotation_score = targets
    atom_pos, cell_pos, log_radius, frame = noisy
    am, cm = batch["atom_mask"], batch["cell_mask"]
    eye_a = torch.eye(am.shape[1], dtype=torch.bool, device=am.device)[None]
    eye_c = torch.eye(cm.shape[1], dtype=torch.bool, device=cm.device)[None]
    atom_upper = am[:, :, None] & am[:, None, :] & ~eye_a & torch.triu(torch.ones_like(eye_a[0]), 1)[None]
    cell_upper = cm[:, :, None] & cm[:, None, :] & ~eye_c & torch.triu(torch.ones_like(eye_c[0]), 1)[None]
    cross = cm[:, :, None] & am[:, None, :]
    atom_pair = am[:, :, None] & am[:, None, :] & ~eye_a
    x0 = masked_center(diffusion.predict_x0(atom_pos, pred["atom_eps"], t), am)
    c0 = diffusion.predict_x0(cell_pos, pred["cell_eps"], t)
    r0 = diffusion.predict_x0(log_radius, pred["radius_eps"], t).exp().clamp(.1, 10.)
    frame0 = diffusion.denoise_frames(frame, pred["rotation_score"], t, cm)
    member_prob = pred["membership_logits"].softmax(-1)[..., 1]*cross
    denominator = member_prob.sum(-1).clamp_min(1)
    derived_center = (member_prob[..., None]*x0[:, None]).sum(2)/denominator[..., None]
    rel = x0[:, None]-c0[:, :, None]
    normal = frame0[..., 2]
    height = (rel*normal[:, :, None]).sum(-1)
    radial = (rel-height[..., None]*normal[:, :, None]).norm(dim=-1)
    geometry = (derived_center-c0).square().sum(-1)
    geometry += (member_prob*height.square()).sum(-1)/denominator
    geometry += (member_prob*(radial-r0[:, :, None]).square()).sum(-1)/denominator

    bond_prob = pred["bond_logits"].softmax(-1)
    orders = bond_prob.new_tensor([0., 1., 2., 3., 1.])
    expected_bond = (bond_prob*orders).sum(-1)*atom_pair
    expected_h = (pred["hydrogen_logits"].softmax(-1)*torch.arange(5, device=am.device)).sum(-1)
    expected_radical = (pred["radical_logits"].softmax(-1)*torch.arange(3, device=am.device)).sum(-1)
    valence = expected_bond.sum(-1)+expected_h+expected_radical
    allowed = max_valence[batch["atom_types"]]
    if positive_valence_bonus is not None:
        allowed = allowed+positive_valence_bonus[batch["atom_types"]]*batch["atom_charges"].gt(0)
    valence_loss = mean_masked(F.relu(valence-allowed).square(), am)
    distance = (x0[:, :, None]-x0[:, None]).norm(dim=-1)
    pair_radius = covalent_radii[batch["atom_types"]][:, :, None]+covalent_radii[batch["atom_types"]][:, None]
    length_factor = bond_prob.new_tensor([1., 1., .90, .82, .95])
    target_length = (bond_prob*length_factor).sum(-1)*pair_radius
    bond_length = mean_masked((distance-target_length).square(), atom_upper & batch["atom_bond_types"].gt(0))
    connectivity = mean_masked(F.relu(1-((1-bond_prob[..., 0])*atom_pair).sum(-1)), am)

    overlap = torch.einsum("bca,bda->bcd", member_prob, member_prob)
    target_overlap = torch.where(batch["cell_relations"].ge(3), batch["cell_relations"].float()-1, torch.where(batch["cell_relations"].eq(2), 1., 0.))
    overlap_loss = mean_masked((overlap-target_overlap).square(), cell_upper)
    cardinality = mean_masked((member_prob.sum(-1)-batch["cell_sizes"].float()).square(), cm)
    co_member = torch.einsum("bca,bcj->baj", batch["cell_membership"].float(), batch["cell_membership"].float()).gt(0)
    aromatic_edges = atom_upper & co_member & batch["atom_bond_types"].eq(4)
    aromatic_loss = mean_masked(-pred["bond_logits"].log_softmax(-1)[..., 4], aromatic_edges)
    zero = pred["atom_eps"].sum()*0
    return {
        "atom_eps": mean_masked((pred["atom_eps"]-atom_eps).square().sum(-1), am),
        "cell_eps": mean_masked((pred["cell_eps"]-cell_eps).square().sum(-1), cm),
        "radius_eps": mean_masked((pred["radius_eps"]-radius_eps).square(), cm),
        "rotation": mean_masked(((pred["rotation_score"]-rotation_score)*extract(diffusion.rot_sigma, t, rotation_score.ndim)).square().sum(-1), cm),
        "atom_type": cross_entropy(pred["atom_type_logits"][am], batch["atom_types"][am]),
        "charge": cross_entropy(pred["charge_logits"][am], model.charge_index(batch["atom_charges"])[am]),
        "hydrogen": cross_entropy(pred["hydrogen_logits"][am & batch["atom_hydrogen_mask"]], batch["atom_hydrogens"][am & batch["atom_hydrogen_mask"]].clamp(0, 4)) if (am & batch["atom_hydrogen_mask"]).any() else zero,
        "radical": cross_entropy(pred["radical_logits"][am & batch["atom_hydrogen_mask"]], batch["atom_radicals"][am & batch["atom_hydrogen_mask"]].clamp(0, 2)) if (am & batch["atom_hydrogen_mask"]).any() else zero,
        "bond": cross_entropy(pred["bond_logits"][atom_upper], batch["atom_bond_types"][atom_upper]) if atom_upper.any() else zero,
        "cell_size": cross_entropy(pred["cell_size_logits"][cm], model.encode_size(batch["cell_sizes"])[cm]) if cm.any() else zero,
        "cell_aromatic": cross_entropy(pred["cell_aromatic_logits"][cm], batch["cell_aromatic"][cm]) if cm.any() else zero,
        "relation": cross_entropy(pred["relation_logits"][cell_upper], batch["cell_relations"][cell_upper]) if cell_upper.any() else zero,
        "membership": cross_entropy(pred["membership_logits"][cross], batch["cell_membership"][cross].long()) if cross.any() else zero,
        "atom_count": cross_entropy(pred["atom_count_logits"], am.sum(1)),
        "cell_count": cross_entropy(pred["cell_count_logits"], cm.sum(1)),
        "cell_geometry": mean_masked(geometry, cm), "cell_cardinality": cardinality,
        "relation_overlap": overlap_loss, "aromatic": aromatic_loss,
        "valence": valence_loss, "bond_length": bond_length, "connectivity": connectivity,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/qcdge_a10_fpa_template_free.yaml")
    parser.add_argument("--resume")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda"))
    train = TemplateFreeFPAEntityDataset(cfg["data"]["root"], "train")
    if not train.has_hydrogen_counts:
        raise RuntimeError("hydrogen sidecars missing; run augment_qcdge_hydrogen_counts.py first")
    val = TemplateFreeFPAEntityDataset(cfg["data"]["root"], "val", train.property_mean, train.property_std, condition_stats=train.condition_stats)
    model_cfg = dict(cfg["model"])
    model_cfg.update(num_atom_types=train.num_atom_types,
        atom_radii=[element_record(x).vdw_radius/3 for x in train.atom_map],
        max_atoms=train.max_heavy_atoms, max_cells=train.max_fpa_cells)
    model = TemplateFreeFPAEntityDenoiser(**model_cfg).to(device)
    diffusion = MolecularDiffusion(**cfg["diffusion"]).to(device)
    covalent = torch.tensor([element_record(x).covalent_radius for x in train.atom_map], device=device)
    valence = torch.tensor([element_record(x).default_valence for x in train.atom_map], device=device, dtype=torch.float32)
    positive_bonus = torch.tensor([1. if x in ("N", "O") else 0. for x in train.atom_map], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    start = step = 0; best = math.inf
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("representation") != "template_free_fpa_v3": raise ValueError("incompatible checkpoint representation")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start, step, best = state["epoch"]+1, state["step"], float(state.get("best_val", math.inf))
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=shuffle,
        num_workers=cfg["training"].get("num_workers", 0), collate_fn=collate_template_free_fpa, pin_memory=True)
    train_loader, val_loader = loader(train, True), loader(val, False)
    weights = cfg["training"]["loss_weights"]
    run = Path(cfg["training"]["output_dir"]); run.mkdir(parents=True, exist_ok=True)
    (run/"config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    log = (run/"metrics.jsonl").open("a", buffering=1)
    for epoch in range(start, cfg["training"]["epochs"]):
        model.train(); total = batches = 0; tic = time.time()
        for raw in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and cfg["training"].get("bf16", True)):
                terms = loss_terms(forward_batch(raw, model, diffusion, device), model, diffusion, covalent, valence, positive_bonus)
                loss = sum(weights.get(k, 0)*v for k, v in terms.items())
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            if not torch.isfinite(norm): raise FloatingPointError(f"non-finite gradient at epoch={epoch} step={step}")
            optimizer.step(); total += float(loss.detach()); batches += 1; step += 1
        model.eval(); val_total = val_batches = condition_gain = 0
        with torch.no_grad():
            for raw in val_loader:
                cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                base = loss_terms(forward_batch(raw, model, diffusion, device, False), model, diffusion, covalent, valence, positive_bonus)
                base_total = sum(weights.get(k, 0)*v for k, v in base.items()); val_total += float(base_total)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, device)
                shifted = dict(raw); shifted["property"] = raw["property"].roll(1, 0)
                alt = loss_terms(forward_batch(shifted, model, diffusion, device, False), model, diffusion, covalent, valence, positive_bonus)
                condition_gain += float(sum(weights.get(k, 0)*v for k, v in alt.items())-base_total)
                val_batches += 1
                if val_batches >= cfg["training"].get("validation_batches", 10): break
        scheduler.step()
        metrics = {"epoch": epoch, "step": step, "train_loss": total/batches,
            "val_loss": val_total/val_batches, "condition_shuffle_gain": condition_gain/val_batches,
            "seconds": time.time()-tic}
        if metrics["val_loss"] < best: best = metrics["val_loss"]
        state = {"representation": "template_free_fpa_v3", "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "step": step, "best_val": best, "config": cfg,
            "atom_map": train.atom_map, "condition_stats": train.condition_stats,
            "max_atoms": train.max_heavy_atoms, "max_cells": train.max_fpa_cells,
            "metrics": metrics}
        torch.save(state, run/"last.pt")
        if metrics["val_loss"] == best: torch.save(state, run/"best.pt")
        print(json.dumps(metrics), flush=True); log.write(json.dumps(metrics)+"\n")


if __name__ == "__main__": main()
