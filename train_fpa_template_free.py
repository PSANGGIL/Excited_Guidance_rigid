#!/usr/bin/env python3
"""Train template-free heavy-atom/ring-cell diffusion from scratch."""
from __future__ import annotations

import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from polyring_diffusion.chem import element_record
from polyring_diffusion.fpa_data_template_free import TemplateFreeFPAEntityDataset, collate_template_free_fpa
from polyring_diffusion.fpa_diffusion_v2 import extract, masked_center
from polyring_diffusion.fpa_model_template_free import TemplateFreeFPAEntityDenoiser, template_free_representation_name
from polyring_diffusion.so3_diffusion import MolecularDiffusion


def mean_masked(value, mask):
    selected = value[mask]
    return selected.mean() if selected.numel() else value.sum()*0


def cross_entropy(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def focal_cross_entropy(logits, target, class_weight=None, gamma=2.):
    log_probability = logits.log_softmax(-1)
    selected = log_probability.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    modulation = (1-selected.exp()).pow(float(gamma))
    weight = class_weight[target] if class_weight is not None else 1.
    return -weight*modulation*selected


def aromatic_ring_objective(prediction, batch, atom_upper, cell_mask, positive_weight=1.):
    """Couple cell aromaticity to a complete aromatic boundary in the atom graph."""
    target = batch["cell_aromatic"].float()
    logits = prediction["cell_aromatic_logits"]
    cell_logit = logits[..., 1]-logits[..., 0]
    class_weight = torch.where(target.gt(.5), target.new_tensor(float(positive_weight)), target.new_tensor(1.))
    cell_bce = mean_masked(F.binary_cross_entropy_with_logits(
        cell_logit, target, reduction="none")*class_weight, cell_mask)
    membership = batch["cell_membership"].bool()
    boundary = (membership[:, :, :, None] & membership[:, :, None, :]
        & atom_upper[:, None] & batch["atom_bond_types"][:, None].gt(0))
    boundary_count = boundary.sum((-1, -2))
    aromatic_probability = prediction["bond_aromatic_logits"].softmax(-1)[..., 1].clamp_min(1e-7)
    cycle_log_probability = (boundary*aromatic_probability[:, None].log()).sum((-1, -2)) / boundary_count.clamp_min(1)
    cycle_probability = cycle_log_probability.exp().clamp(1e-6, 1-1e-6)
    cycle_logit = cycle_log_probability-torch.log1p(-cycle_probability)
    cycle_mask = cell_mask & boundary_count.ge(3)
    cycle_bce = mean_masked(F.binary_cross_entropy_with_logits(
        cycle_logit, target, reduction="none")*class_weight, cycle_mask)
    cell_probability = cell_logit.sigmoid()
    consistency = mean_masked((cell_probability-cycle_probability).square(), cycle_mask)
    positive = cycle_mask & target.bool()
    cell_recall = mean_masked(cell_probability.ge(.5).float(), positive).detach()
    cycle_recall = mean_masked(cycle_probability.ge(.5).float(), positive).detach()
    return cell_bce, cycle_bce+consistency, cell_recall, cycle_recall


def global_connectivity_loss(edge_probability, atom_mask, target_edge_count=None):
    """Negative log weighted spanning-tree mass (matrix-tree theorem)."""
    losses = []
    for row, (probability, mask) in enumerate(zip(edge_probability.float(), atom_mask)):
        n = int(mask.sum())
        if n <= 1: continue
        adjacency = probability[:n, :n].clamp(1e-6, 1-1e-6)
        adjacency = (adjacency+adjacency.T)/2
        adjacency.fill_diagonal_(0)
        laplacian = torch.diag(adjacency.sum(-1))-adjacency
        sign, logabsdet = torch.linalg.slogdet(laplacian[:-1, :-1]+torch.eye(n-1, device=adjacency.device)*1e-6)
        normalization = (n-2)*math.log(n) if n > 1 else 0.
        tree_loss = torch.where(sign > 0, (normalization-logabsdet)/(n-1), logabsdet.new_tensor(20.))
        expected_edges = adjacency.triu(1).sum()
        target_edges = float(n-1) if target_edge_count is None else target_edge_count[row].float()
        density_loss = (expected_edges-target_edges).square()/n
        losses.append(tree_loss+density_loss)
    return torch.stack(losses).mean() if losses else edge_probability.sum()*0


@torch.no_grad()
def checkpoint_generation_metrics(model, diffusion, dataset, atom_map, config, device):
    from sample_fpa_template_free import reconstruct, sample as generate
    count = min(int(config.get("samples", 16)), len(dataset))
    condition = torch.stack([dataset[i]["property"] for i in range(count)]).to(device)
    atom_logits, cell_logits = model.count_logits(condition)
    atom_count = atom_logits.argmax(-1).clamp_min(1).tolist()
    cell_count = cell_logits.argmax(-1).tolist()
    generated = generate(model, diffusion, condition, atom_count, cell_count,
        float(config.get("guidance", 1.)), atom_map, closed_shell=bool(config.get("closed_shell", False)))
    valid = connected = aromatic = 0
    projection_valid = int(generated["projection_valid"].sum())
    projected_aromatic = int(generated.get("projection_aromatic_cells", torch.zeros(1)).sum())
    complete_aromatic = int(generated.get("projection_complete_aromatic_cells", torch.zeros(1)).sum())
    from rdkit import Chem
    for row in range(count):
        mask = generated["atom_mask"][row]
        try:
            mol = reconstruct(generated["atom_types"][row, mask], generated["charges"][row, mask],
                generated["hydrogens"][row, mask], generated["radicals"][row, mask],
                generated["bonds"][row][mask][:, mask], generated["positions"][row, mask], atom_map,
                charge_min=model.charge_min, chirality=generated["chirality"][row, mask], require_connected=False,
                bond_aromatic=generated["bond_aromatic"][row][mask][:, mask])
            valid += 1; connected += int(len(Chem.GetMolFrags(mol)) == 1)
            aromatic += int(any(all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)
                for ring in mol.GetRingInfo().AtomRings()))
        except Exception:
            pass
    return {"checkpoint_rdkit_valid_rate": valid/max(count, 1),
        "checkpoint_connected_rate": connected/max(count, 1),
        "checkpoint_projection_valid_rate": projection_valid/max(count, 1),
        "checkpoint_aromatic_molecule_rate": aromatic/max(valid, 1),
        "checkpoint_aromatic_cycle_completion": complete_aromatic/max(projected_aromatic, 1),
        "checkpoint_eval_samples": count}


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
    chirality = diffusion.q_tokens(batch["atom_chirality"].clamp(0, 2), t, model.chirality_mask_token, atom_mask)
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
        cell_mask, t, batch["property"], drop, chirality=chirality)
    targets = atom_eps, cell_eps, radius_eps.squeeze(-1), rotation_score
    noisy = atom_pos, cell_pos, log_radius.squeeze(-1), frame
    return batch, prediction, targets, t, noisy


def loss_terms(pack, model, diffusion, covalent_radii, max_valence,
               positive_valence_bonus=None, chemistry=None, charge_class_weights=None):
    chemistry = chemistry or {}
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
    # Kekule-compatible aromatic valence: each aromatic edge contributes one,
    # plus one participation unit per atom with at least one aromatic neighbor.
    orders = bond_prob.new_tensor([0., 1., 2., 3., 1.])
    expected_bond = (bond_prob*orders).sum(-1)*atom_pair
    bond_aromatic_prob = pred["bond_aromatic_logits"].softmax(-1)[..., 1]
    aromatic_participation = 1-(1-bond_aromatic_prob*atom_pair).prod(-1)
    atom_probability = pred["atom_type_logits"].softmax(-1)
    charge_probability = pred["charge_logits"].softmax(-1)
    expected_h = (pred["hydrogen_logits"].softmax(-1)*torch.arange(5, device=am.device)).sum(-1)
    radical_probability = pred["radical_logits"].softmax(-1)
    expected_radical = (radical_probability*torch.arange(3, device=am.device)).sum(-1)
    valence = expected_bond.sum(-1)+aromatic_participation+expected_h+expected_radical
    target_orders = orders[batch["atom_bond_types"]]
    target_aromatic_participation = (batch["atom_bond_aromatic"]*atom_pair).any(-1).float()
    target_valence = target_orders.sum(-1)+target_aromatic_participation+batch["atom_hydrogens"].clamp(0, 4).float()+batch["atom_radicals"].clamp(0, 2).float()
    chemistry_mask = am & batch["atom_hydrogen_mask"]
    # Couple the independently predicted bond, H and radical heads. The old
    # overflow-only loss did not penalize under-valent structures. Use the DB
    # valence target so genuine radical states remain learnable.
    valence_completion = mean_masked((valence-target_valence).square(), chemistry_mask)
    charge_values = torch.arange(model.num_charges, device=am.device)+model.charge_min
    expected_charge = (charge_probability*charge_values).sum(-1)*am
    atom_count_scale = am.sum(-1).clamp_min(1).float().sqrt()
    charge_balance = (((expected_charge.sum(-1)-(batch["atom_charges"]*am).sum(-1))/atom_count_scale).square()).mean()
    radical_count = (((expected_radical.mul(am).sum(-1)-batch["atom_radicals"].mul(am).sum(-1))/atom_count_scale).square()).mean()
    allowed = (atom_probability*max_valence).sum(-1)
    if positive_valence_bonus is not None:
        type_bonus = (atom_probability*positive_valence_bonus).sum(-1)
        charge_values = torch.arange(model.num_charges, device=am.device)+model.charge_min
        positive_probability = charge_probability[:, :, charge_values.gt(0)].sum(-1)
        allowed = allowed+type_bonus*positive_probability
    valence_loss = mean_masked(F.relu(valence-allowed).square(), am)
    valence_violation = mean_masked(valence.gt(allowed+.25).float(), am).detach()
    radical_target = batch["atom_radicals"].gt(0).float()
    radical_nonzero_logit = torch.logsumexp(pred["radical_logits"][..., 1:].float(), -1)-pred["radical_logits"][..., 0].float()
    radical_prior = mean_masked(F.binary_cross_entropy_with_logits(
        radical_nonzero_logit, radical_target, reduction="none"), am & batch["atom_hydrogen_mask"])
    distance = (x0[:, :, None]-x0[:, None]).norm(dim=-1)
    pair_radius = covalent_radii[batch["atom_types"]][:, :, None]+covalent_radii[batch["atom_types"]][:, None]
    length_factor = bond_prob.new_tensor([1., 1., .90, .82, .95])
    target_length = (bond_prob*length_factor).sum(-1)*pair_radius
    zero = pred["atom_eps"].sum()*0
    bond_weights = bond_prob.new_tensor(chemistry.get("bond_class_weights", [.05, 1., 1.5, 2., 1.25]))
    bond_target = batch["atom_bond_types"]
    if atom_upper.any():
        bond_per_edge = focal_cross_entropy(pred["bond_logits"][atom_upper],
            bond_target[atom_upper], bond_weights, chemistry.get("bond_focal_gamma", 2.))
        timestep_weight = 1+float(chemistry.get("discrete_timestep_boost", 1.))*(1-diffusion.abar[t])
        selected_timestep_weight = timestep_weight[:, None, None].expand_as(atom_upper)[atom_upper]
        bond_loss = (bond_per_edge*selected_timestep_weight).mean()
    else:
        bond_loss = zero
    target_aromatic_degree = (batch["atom_bond_aromatic"]*atom_pair).sum(-1).float()
    bond_length = mean_masked((distance-target_length).square(), atom_upper & batch["atom_bond_types"].gt(0))
    aromatic_degree = mean_masked(((bond_aromatic_prob*atom_pair).sum(-1)-target_aromatic_degree).square(), am)
    connectivity = mean_masked(F.relu(1-((1-bond_prob[..., 0])*atom_pair).sum(-1)), am)
    target_edge_count = (batch["atom_bond_types"].gt(0)*atom_pair).triu(1).sum((1, 2))
    global_connectivity = global_connectivity_loss((1-bond_prob[..., 0])*atom_pair, am, target_edge_count)

    overlap = torch.einsum("bca,bda->bcd", member_prob, member_prob)
    target_overlap = torch.where(batch["cell_relations"].ge(3), batch["cell_relations"].float()-1, torch.where(batch["cell_relations"].eq(2), 1., 0.))
    overlap_loss = mean_masked((overlap-target_overlap).square(), cell_upper)
    cardinality = mean_masked((member_prob.sum(-1)-batch["cell_sizes"].float()).square(), cm)
    co_member = torch.einsum("bca,bcj->baj", batch["cell_membership"].float(), batch["cell_membership"].float()).gt(0)
    aromatic_edges = atom_upper & co_member & batch["atom_bond_aromatic"]
    aromatic_loss = mean_masked(-pred["bond_aromatic_logits"].log_softmax(-1)[..., 1], aromatic_edges)
    bond_aromatic_loss = cross_entropy(pred["bond_aromatic_logits"][atom_upper], batch["atom_bond_aromatic"][atom_upper].long()) if atom_upper.any() else zero
    cell_aromatic_loss, aromatic_consistency, aromatic_cell_recall, aromatic_cycle_recall = aromatic_ring_objective(
        pred, batch, atom_upper, cm, chemistry.get("aromatic_positive_weight", 9.0))
    zero = pred["atom_eps"].sum()*0
    return {
        "atom_eps": mean_masked((pred["atom_eps"]-atom_eps).square().sum(-1), am),
        "cell_eps": mean_masked((pred["cell_eps"]-cell_eps).square().sum(-1), cm),
        "radius_eps": mean_masked((pred["radius_eps"]-radius_eps).square(), cm),
        "rotation": mean_masked(((pred["rotation_score"]-rotation_score)*extract(diffusion.rot_sigma, t, rotation_score.ndim)).square().sum(-1), cm),
        "atom_type": cross_entropy(pred["atom_type_logits"][am], batch["atom_types"][am]),
        "charge": F.cross_entropy(pred["charge_logits"][am], model.charge_index(batch["atom_charges"])[am],
            weight=charge_class_weights),
        "hydrogen": cross_entropy(pred["hydrogen_logits"][am & batch["atom_hydrogen_mask"]], batch["atom_hydrogens"][am & batch["atom_hydrogen_mask"]].clamp(0, 4)) if (am & batch["atom_hydrogen_mask"]).any() else zero,
        "radical": cross_entropy(pred["radical_logits"][am & batch["atom_hydrogen_mask"]], batch["atom_radicals"][am & batch["atom_hydrogen_mask"]].clamp(0, 2)) if (am & batch["atom_hydrogen_mask"]).any() else zero,
        "chirality": cross_entropy(pred["chirality_logits"][am & batch["atom_hydrogen_mask"]], batch["atom_chirality"][am & batch["atom_hydrogen_mask"]].clamp(0, 2)) if (am & batch["atom_hydrogen_mask"]).any() else zero,
        "bond": bond_loss,
        "cell_size": cross_entropy(pred["cell_size_logits"][cm], model.encode_size(batch["cell_sizes"])[cm]) if cm.any() else zero,
        "cell_aromatic": cell_aromatic_loss,
        "relation": cross_entropy(pred["relation_logits"][cell_upper], batch["cell_relations"][cell_upper]) if cell_upper.any() else zero,
        "membership": cross_entropy(pred["membership_logits"][cross], batch["cell_membership"][cross].long()) if cross.any() else zero,
        "atom_count": cross_entropy(pred["atom_count_logits"], am.sum(1)),
        "cell_count": cross_entropy(pred["cell_count_logits"], cm.sum(1)),
        "cell_geometry": mean_masked(geometry, cm), "cell_cardinality": cardinality,
        "relation_overlap": overlap_loss, "aromatic": aromatic_loss,
        "bond_aromatic": bond_aromatic_loss,
        "aromatic_consistency": aromatic_consistency,
        "aromatic_cell_recall": aromatic_cell_recall,
        "aromatic_cycle_recall": aromatic_cycle_recall,
        "aromatic_degree": aromatic_degree, "radical_prior": radical_prior,
        "valence": valence_loss, "valence_completion": valence_completion,
        "charge_balance": charge_balance, "radical_count": radical_count,
        "bond_length": bond_length, "connectivity": connectivity,
        "global_connectivity": global_connectivity,
        "valence_violation_rate": valence_violation,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/qcdge_a10_fpa_template_free.yaml")
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint", help="load compatible model weights but reset optimizer, scheduler, epoch and best metric")
    args = parser.parse_args()
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda"))
    condition_mode = cfg["data"].get("condition_mode", "full_spectrum")
    train = TemplateFreeFPAEntityDataset(cfg["data"]["root"], "train", condition_mode=condition_mode,
        sidecar_dir=cfg["data"].get("sidecar_dir", "hydrogen_counts"),
        aromatic_only=cfg["data"].get("aromatic_only", False),
        frame_augmentation=cfg["data"].get("frame_augmentation", True))
    if not train.has_hydrogen_counts:
        raise RuntimeError("hydrogen sidecars missing; run augment_qcdge_hydrogen_counts.py first")
    if cfg["model"].get("chirality_enabled", False) and not train.has_chirality_labels:
        raise RuntimeError("chirality sidecars missing; rerun augment_qcdge_hydrogen_counts_v2.py")
    val = TemplateFreeFPAEntityDataset(cfg["data"]["root"], "val", train.property_mean, train.property_std,
        condition_stats=train.condition_stats, condition_mode=condition_mode,
        atom_map=train.atom_map,
        sidecar_dir=cfg["data"].get("sidecar_dir", "hydrogen_counts"),
        aromatic_only=cfg["data"].get("aromatic_only", False), frame_augmentation=False)
    if int(cfg["model"]["condition_dim"]) != train.condition_dim:
        raise ValueError(f"model condition_dim={cfg['model']['condition_dim']} does not match dataset condition_dim={train.condition_dim}")
    representation = template_free_representation_name(condition_mode, cfg["model"])
    model_cfg = dict(cfg["model"])
    model_cfg.update(num_atom_types=train.num_atom_types,
        atom_radii=[element_record(x).vdw_radius/3 for x in train.atom_map],
        max_atoms=train.max_heavy_atoms, max_cells=train.max_fpa_cells)
    model = TemplateFreeFPAEntityDenoiser(**model_cfg).to(device)
    diffusion = MolecularDiffusion(**cfg["diffusion"]).to(device)
    covalent = torch.tensor([element_record(x).covalent_radius for x in train.atom_map], device=device)
    valence = torch.tensor([element_record(x).default_valence for x in train.atom_map], device=device, dtype=torch.float32)
    positive_bonus = torch.tensor([1. if x in ("N", "O") else 0. for x in train.atom_map], device=device)
    charge_counts = torch.zeros(model.num_charges, dtype=torch.float64)
    for shard in train.shards:
        charge_counts += torch.bincount(model.charge_index(shard["atom_charges"].long()), minlength=model.num_charges)
    observed = charge_counts.gt(0)
    charge_class_weights = torch.zeros_like(charge_counts)
    charge_class_weights[observed] = (charge_counts[observed].max()/charge_counts[observed]).sqrt().clamp_max(10)
    charge_class_weights = charge_class_weights.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    start = step = 0; best = math.inf
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state.get("representation") != representation: raise ValueError("incompatible checkpoint representation")
        if tuple(state.get("atom_map", ())) != tuple(train.atom_map):
            raise ValueError(f"incompatible atom vocabulary: checkpoint={state.get('atom_map')} dataset={train.atom_map}")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        start, step, best = state["epoch"]+1, state["step"], float(state.get("best_val", math.inf))
    elif args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if state.get("representation") != representation: raise ValueError("incompatible checkpoint representation")
        if tuple(state.get("atom_map", ())) != tuple(train.atom_map):
            raise ValueError(f"incompatible atom vocabulary: checkpoint={state.get('atom_map')} dataset={train.atom_map}")
        model.load_state_dict(state["model"])
    sampler = None
    if cfg["training"].get("balance_heavy_atom_count", True):
        counts = torch.tensor([train.heavy_count_at(i) for i in range(len(train))])
        if counts.unique().numel() > 1:
            frequency = torch.bincount(counts)
            sampler = WeightedRandomSampler((1/frequency[counts]).double(), len(counts), replacement=True)
    common = dict(batch_size=cfg["training"]["batch_size"], num_workers=cfg["training"].get("num_workers", 0),
        collate_fn=collate_template_free_fpa, pin_memory=True)
    train_loader = DataLoader(train, sampler=sampler, shuffle=sampler is None, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    weights = cfg["training"]["loss_weights"]
    chemistry = cfg["training"].get("chemistry", {})
    run = Path(cfg["training"]["output_dir"]); run.mkdir(parents=True, exist_ok=True)
    (run/"config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    log = (run/"metrics.jsonl").open("a", buffering=1)
    for epoch in range(start, cfg["training"]["epochs"]):
        model.train(); total = batches = 0; tic = time.time()
        for raw in train_loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda" and cfg["training"].get("bf16", True)):
                terms = loss_terms(forward_batch(raw, model, diffusion, device), model, diffusion, covalent,
                    valence, positive_bonus, chemistry, charge_class_weights)
                loss = sum(weights.get(k, 0)*v for k, v in terms.items())
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at epoch={epoch} step={step}")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            if not torch.isfinite(norm): raise FloatingPointError(f"non-finite gradient at epoch={epoch} step={step}")
            optimizer.step(); total += float(loss.detach()); batches += 1; step += 1
        model.eval(); val_total = val_batches = condition_gain = 0
        chemistry_totals = {key: 0. for key in ("valence_violation_rate", "valence_completion",
            "charge_balance", "radical_count", "aromatic_degree", "radical_prior",
            "aromatic_cell_recall", "aromatic_cycle_recall")}
        with torch.no_grad():
            for raw in val_loader:
                cpu_rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
                base = loss_terms(forward_batch(raw, model, diffusion, device, False), model, diffusion, covalent,
                    valence, positive_bonus, chemistry, charge_class_weights)
                base_total = sum(weights.get(k, 0)*v for k, v in base.items()); val_total += float(base_total)
                for key in chemistry_totals: chemistry_totals[key] += float(base[key])
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None: torch.cuda.set_rng_state(cuda_rng, device)
                shifted = dict(raw); shifted["property"] = raw["property"].roll(1, 0)
                alt = loss_terms(forward_batch(shifted, model, diffusion, device, False), model, diffusion, covalent,
                    valence, positive_bonus, chemistry, charge_class_weights)
                condition_gain += float(sum(weights.get(k, 0)*v for k, v in alt.items())-base_total)
                val_batches += 1
                if val_batches >= cfg["training"].get("validation_batches", 10): break
        scheduler.step()
        metrics = {"epoch": epoch, "step": step, "train_loss": total/batches,
            "val_loss": val_total/val_batches, "condition_shuffle_gain": condition_gain/val_batches,
            "condition_fidelity_proxy": condition_gain/val_batches,
            "seconds": time.time()-tic}
        metrics.update({"val_"+key: value/val_batches for key, value in chemistry_totals.items()})
        if metrics["val_loss"] < best: best = metrics["val_loss"]
        state = {"representation": representation, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "step": step, "best_val": best, "config": cfg,
            "atom_map": train.atom_map, "condition_stats": train.condition_stats,
            "max_atoms": train.max_heavy_atoms, "max_cells": train.max_fpa_cells,
            "metrics": metrics}
        torch.save(state, run/"last.pt")
        if metrics["val_loss"] == best: torch.save(state, run/"best.pt")
        evaluation = cfg["training"].get("checkpoint_evaluation", {})
        if evaluation.get("enabled", False) and (epoch+1) % int(evaluation.get("interval", 100)) == 0:
            metrics.update(checkpoint_generation_metrics(model, diffusion, val, train.atom_map, evaluation, device))
            state["metrics"] = metrics; torch.save(state, run/"last.pt")
            if metrics["val_loss"] == best: torch.save(state, run/"best.pt")
        print(json.dumps(metrics), flush=True); log.write(json.dumps(metrics)+"\n")


if __name__ == "__main__": main()
