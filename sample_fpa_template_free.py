#!/usr/bin/env python3
"""CFG sampler and RDKit reconstruction for template-free FPA v3."""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
import torch
import yaml

from polyring_diffusion.chem import element_record
from polyring_diffusion.fpa_diffusion_v2 import masked_center
from polyring_diffusion.fpa_model_template_free import TemplateFreeFPAEntityDenoiser, template_free_representation_name
from polyring_diffusion.fpa_projection_template_free import project_template_free_fpa
from polyring_diffusion.so3_diffusion import MolecularDiffusion


def scalar_s1_t1_condition(stats, s1_energy, t1_energy, count, device):
    if stats.get("mode") != "s1_t1":
        raise ValueError("checkpoint does not use scalar S1/T1 conditioning")
    values = torch.tensor([(s1_energy-stats["s1_energy_mean"])/stats["s1_energy_std"],
        (t1_energy-stats["t1_energy_mean"])/stats["t1_energy_std"]], device=device)
    return values.unsqueeze(0).repeat(count, 1)


def symmetric_step(diffusion, noisy, logits, t, token, mask):
    out = diffusion.p_tokens(noisy, logits, t, token, mask)
    upper = torch.triu(out, 1)
    return upper+upper.transpose(1, 2)


def _sanitize_prediction(prediction, tracker=None, step=None, source=None):
    clean = {}
    for key, value in prediction.items():
        finite = torch.isfinite(value).reshape(len(value), -1).all(1)
        if tracker is not None and not bool(finite.all()):
            tracker["valid"] &= finite
            count = int((~finite).sum())
            tracker["counts"][key] = tracker["counts"].get(key, 0)+count
            tracker["first"].setdefault(key, {"step": int(step), "source": source, "samples": count})
        clean[key] = torch.nan_to_num(value, nan=0., posinf=30., neginf=-30.)
    return clean


def cfg_predict(model, args, condition, guidance, tracker=None, step=None, chirality=None):
    if guidance == 0:
        drop = torch.ones(len(condition), dtype=torch.bool, device=condition.device)
        raw = model(*args, condition, drop, chirality=chirality)
        return _sanitize_prediction(raw, tracker, step, "unconditional")
    conditional = model(*args, condition, chirality=chirality)
    conditional = _sanitize_prediction(conditional, tracker, step, "conditional")
    if guidance == 1:
        return conditional
    drop = torch.ones(len(condition), dtype=torch.bool, device=condition.device)
    unconditional = model(*args, condition, drop, chirality=chirality)
    unconditional = _sanitize_prediction(unconditional, tracker, step, "unconditional")
    guided = {key: unconditional[key]+guidance*(value-unconditional[key]) for key, value in conditional.items()}
    return _sanitize_prediction(guided, tracker, step, "guided")


def _clip_vectors(value, maximum):
    norm = value.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return value*(float(maximum)/norm).clamp_max(1.)


def _track_state(value, tracker, key, step):
    finite = torch.isfinite(value).reshape(len(value), -1).all(1)
    if not bool(finite.all()):
        tracker["valid"] &= finite
        count = int((~finite).sum())
        tracker["counts"][key] = tracker["counts"].get(key, 0)+count
        tracker["first"].setdefault(key, {"step": int(step), "source": "reverse_state", "samples": count})
    return torch.nan_to_num(value)


def reconstruct(atom_types, charges, hydrogens, radicals, bonds, positions, atom_map, charge_min=-2,
                closed_shell=False, chirality=None, require_connected=True, atom_aromatic=None, bond_aromatic=None):
    from rdkit import Chem
    rw = Chem.RWMol()
    chirality = torch.zeros_like(atom_types) if chirality is None else chirality
    for atom_type, charge, hydrogen, radical, chiral in zip(atom_types.tolist(), charges.tolist(), hydrogens.tolist(), radicals.tolist(), chirality.tolist()):
        atom = Chem.Atom(atom_map[atom_type])
        atom.SetFormalCharge(int(charge)+charge_min)
        atom.SetNumExplicitHs(int(hydrogen)); atom.SetNumRadicalElectrons(int(radical)); atom.SetNoImplicit(True)
        if chiral: atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        rw.AddAtom(atom)
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    if atom_aromatic is not None:
        for i, aromatic in enumerate(atom_aromatic.tolist()):
            rw.GetAtomWithIdx(i).SetIsAromatic(bool(aromatic))
    for i in range(len(atom_types)):
        for j in range(i+1, len(atom_types)):
            kind = int(bonds[i, j])
            if kind in kinds:
                aromatic = kind == 4 if bond_aromatic is None else bool(bond_aromatic[i, j])
                if aromatic:
                    rw.GetAtomWithIdx(i).SetIsAromatic(True); rw.GetAtomWithIdx(j).SetIsAromatic(True)
                rw.AddBond(i, j, kinds[kind])
                if aromatic:
                    rw.GetBondBetweenAtoms(i, j).SetIsAromatic(True)
    mol = rw.GetMol(); conformer = Chem.Conformer(len(atom_types))
    for i, point in enumerate(positions.tolist()): conformer.SetAtomPosition(i, point)
    mol.AddConformer(conformer)
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for index, desired in enumerate(chirality.tolist()):
        if not desired: continue
        atom = mol.GetAtomWithIdx(index)
        target = "R" if desired == 1 else "S"
        if not atom.HasProp("_CIPCode"):
            raise ValueError(f"undefined predicted chirality at atom {index}")
        if atom.GetProp("_CIPCode") != target:
            tag = atom.GetChiralTag()
            atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW else Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for index, desired in enumerate(chirality.tolist()):
        if desired and (not mol.GetAtomWithIdx(index).HasProp("_CIPCode") or
                mol.GetAtomWithIdx(index).GetProp("_CIPCode") != ("R" if desired == 1 else "S")):
            raise ValueError(f"failed to realize predicted chirality at atom {index}")
    if require_connected and len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("disconnected molecule")
    radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if closed_shell and radical_electrons:
        raise ValueError(f"open-shell molecule with {radical_electrons} radical electrons")
    return Chem.AddHs(mol, explicitOnly=False, addCoords=True)


def decode_exact(encoded, atom_map, charge_min=-2):
    """Decode a clean DB representation without projection or chemistry repair."""
    return reconstruct(encoded["atom_types"], encoded["atom_charges"]-charge_min,
        encoded["atom_hydrogens"], encoded["atom_radicals"], encoded["atom_bond_types"],
        encoded["atom_positions"], atom_map, charge_min=charge_min,
        chirality=encoded.get("atom_chirality"), require_connected=False,
        atom_aromatic=encoded.get("atom_aromatic"), bond_aromatic=encoded.get("atom_bond_aromatic"))


@torch.no_grad()
def sample(model, diffusion, condition, atom_count, cell_count, guidance=2., atom_map=None,
           progress_every=0, score_clip=20., coordinate_clip=10.,
           radius_min=.5, radius_max=3., closed_shell=False, radical_threshold=.8):
    device = condition.device; batch = len(condition)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    na, nc = max(atom_count), max(1, max(cell_count))
    atom_mask = torch.arange(na, device=device)[None] < torch.tensor(atom_count, device=device)[:, None]
    cell_mask = torch.arange(nc, device=device)[None] < torch.tensor(cell_count, device=device)[:, None]
    atom_type = torch.where(atom_mask, model.atom_mask_token, model.atom_pad_token)
    charge = torch.full((batch, na), model.charge_mask_token, device=device, dtype=torch.long)
    hydrogen = torch.full((batch, na), model.hydrogen_mask_token, device=device, dtype=torch.long)
    radical = torch.full((batch, na), model.radical_mask_token, device=device, dtype=torch.long)
    chirality = torch.full((batch, na), model.chirality_mask_token, device=device, dtype=torch.long)
    bond = torch.full((batch, na, na), 5, device=device, dtype=torch.long)
    size = torch.where(cell_mask, model.size_mask_token, model.size_pad_token)
    aromatic = torch.full((batch, nc), model.aromatic_mask_token, device=device, dtype=torch.long)
    relation = torch.full((batch, nc, nc), model.relation_mask_token, device=device, dtype=torch.long)
    membership = torch.full((batch, nc, na), 2, device=device, dtype=torch.long)
    atom_pos = masked_center(torch.randn(batch, na, 3, device=device), atom_mask)
    cell_pos = torch.randn(batch, nc, 3, device=device)*cell_mask.unsqueeze(-1)
    log_radius = torch.randn(batch, nc, device=device)*cell_mask
    frame = diffusion.initial_frames(batch, nc, device)
    prediction = None
    tracker = {"valid": torch.ones(batch, dtype=torch.bool, device=device), "counts": {}, "first": {}}
    reverse_started = time.perf_counter(); prediction_seconds = 0.; update_seconds = 0.
    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((batch,), step, device=device, dtype=torch.long)
        args = (atom_type, charge, hydrogen, radical, atom_pos, bond, atom_mask, size, aromatic,
            cell_pos, frame, log_radius, relation, membership, cell_mask, t)
        synchronize(); phase_started = time.perf_counter()
        prediction = cfg_predict(model, args, condition, guidance, tracker, step, chirality)
        synchronize(); prediction_seconds += time.perf_counter()-phase_started; phase_started = time.perf_counter()
        atom_pos = masked_center(diffusion.p_coordinates(atom_pos, prediction["atom_eps"], t, atom_mask, score_clip, coordinate_clip), atom_mask)
        cell_pos = diffusion.p_coordinates(cell_pos, prediction["cell_eps"], t, cell_mask, score_clip, coordinate_clip)
        radius_clip = max(abs(math.log(radius_min)), abs(math.log(radius_max)))
        log_radius = diffusion.p_coordinates(log_radius.unsqueeze(-1), prediction["radius_eps"].unsqueeze(-1),
            t, cell_mask, score_clip, radius_clip).squeeze(-1)
        log_radius = _track_state(log_radius, tracker, "log_radius", step).clamp(math.log(radius_min), math.log(radius_max))
        frame = diffusion.p_frames(frame, _clip_vectors(prediction["rotation_score"], score_clip), t, cell_mask)
        atom_pos = _track_state(atom_pos, tracker, "atom_position", step).clamp(-coordinate_clip, coordinate_clip)
        cell_pos = _track_state(cell_pos, tracker, "cell_position", step).clamp(-coordinate_clip, coordinate_clip)
        frame = _track_state(frame, tracker, "cell_frame", step)
        atom_type = diffusion.p_tokens(atom_type, prediction["atom_type_logits"], t, model.atom_mask_token, atom_mask)
        charge = diffusion.p_tokens(charge, prediction["charge_logits"], t, model.charge_mask_token, atom_mask)
        hydrogen = diffusion.p_tokens(hydrogen, prediction["hydrogen_logits"], t, model.hydrogen_mask_token, atom_mask)
        radical = diffusion.p_tokens(radical, prediction["radical_logits"], t, model.radical_mask_token, atom_mask)
        chirality = diffusion.p_tokens(chirality, prediction["chirality_logits"], t, model.chirality_mask_token, atom_mask)
        size = diffusion.p_tokens(size, prediction["cell_size_logits"], t, model.size_mask_token, cell_mask)
        aromatic = diffusion.p_tokens(aromatic, prediction["cell_aromatic_logits"], t, model.aromatic_mask_token, cell_mask)
        bond = symmetric_step(diffusion, bond, prediction["bond_logits"], t, 5, atom_mask[:, :, None]&atom_mask[:, None, :])
        relation = symmetric_step(diffusion, relation, prediction["relation_logits"], t, model.relation_mask_token, cell_mask[:, :, None]&cell_mask[:, None, :])
        membership = diffusion.p_tokens(membership, prediction["membership_logits"], t, 2, cell_mask[:, :, None]&atom_mask[:, None, :])
        synchronize(); update_seconds += time.perf_counter()-phase_started
        completed = diffusion.timesteps-step
        if progress_every and (completed == 1 or completed % progress_every == 0 or step == 0):
            print(json.dumps({"stage":"reverse", "completed":completed, "total":diffusion.timesteps, "elapsed":time.perf_counter()-reverse_started, "prediction_seconds":prediction_seconds, "update_seconds":update_seconds}), flush=True)
    bond_aromatic = prediction["bond_aromatic_logits"].argmax(-1)
    bond_aromatic = torch.triu(bond_aromatic, 1); bond_aromatic = bond_aromatic+bond_aromatic.transpose(1, 2)
    out = {"atom_types": atom_type, "charges": charge, "hydrogens": hydrogen, "radicals": radical,
        "chirality": chirality, "positions": atom_pos, "bonds": bond, "bond_aromatic": bond_aromatic.bool(),
        "cell_sizes": model.decode_size(size),
        "cell_aromatic": aromatic, "cell_centers": cell_pos, "cell_frames": frame,
        "cell_radii": log_radius.exp().clamp(.1, 10.), "cell_relations": relation,
        "membership": membership, "atom_mask": atom_mask, "cell_mask": cell_mask}
    synchronize(); projection_started = time.perf_counter()
    if atom_map is not None:
        cpu_out = {key: value.cpu() for key, value in out.items()}
        cpu_prediction = {key: prediction[key].cpu() for key in ("membership_logits", "bond_logits",
            "bond_aromatic_logits", "hydrogen_logits", "radical_logits")}
        projected = project_template_free_fpa(cpu_out, cpu_prediction, atom_map, closed_shell=closed_shell,
            radical_threshold=radical_threshold, charge_min=model.charge_min)
    else:
        projected = {key: value.cpu() for key, value in out.items()}
    projected["numerical_valid"] = tracker["valid"].cpu()
    if "projection_valid" in projected:
        projected["projection_valid"] &= projected["numerical_valid"]
        projected["projection_reason"] |= (~projected["numerical_valid"]).long()*64
    projected["numerical_diagnostics"] = {
        "nonfinite_counts": tracker["counts"], "first_nonfinite": tracker["first"],
        "score_clip": score_clip, "coordinate_clip": coordinate_clip,
        "radius_range": [radius_min, radius_max]}
    print(json.dumps({"stage":"projection", "seconds":time.perf_counter()-projection_started}), flush=True)
    return projected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--conditions")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--s1-energy", type=float); parser.add_argument("--t1-energy", type=float)
    parser.add_argument("--output", required=True); parser.add_argument("--guidance", type=float, default=2.)
    parser.add_argument("--cell-count-min", type=int)
    parser.add_argument("--cell-count-max", type=int)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--score-clip", type=float, default=20.)
    parser.add_argument("--coordinate-clip", type=float, default=10.)
    parser.add_argument("--radius-min", type=float, default=.5)
    parser.add_argument("--radius-max", type=float, default=3.)
    parser.add_argument("--closed-shell", action="store_true",
        help="force zero radical electrons during projection and reject open-shell reconstructions")
    parser.add_argument("--radical-threshold", type=float,
        help="minimum per-atom nonzero-radical probability; defaults to checkpoint chemistry setting")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    cfg, atom_map = state["config"], state["atom_map"]
    expected_representation = template_free_representation_name(
        cfg["data"].get("condition_mode", "full_spectrum"), cfg["model"])
    if state.get("representation") != expected_representation:
        raise ValueError(f"incompatible checkpoint representation: expected {expected_representation}")
    model_cfg = dict(cfg["model"]); model_cfg.update(num_atom_types=len(atom_map),
        atom_radii=[element_record(x).vdw_radius/3 for x in atom_map],
        max_atoms=state["max_atoms"], max_cells=state["max_cells"])
    model = TemplateFreeFPAEntityDenoiser(**model_cfg).cuda(); model.load_state_dict(state["model"]); model.eval()
    diffusion = MolecularDiffusion(**cfg["diffusion"]).cuda()
    if args.conditions:
        condition = torch.load(args.conditions, map_location="cuda").float()
    else:
        if args.num_samples is None or args.num_samples < 1 or any(x is None for x in (args.s1_energy, args.t1_energy)):
            raise ValueError("provide --conditions or positive --num-samples with --s1-energy and --t1-energy")
        condition = scalar_s1_t1_condition(state["condition_stats"], args.s1_energy,
            args.t1_energy, args.num_samples, "cuda")
    if condition.ndim != 2 or condition.shape[1] != model.condition_dim:
        raise ValueError(f"conditions must be standardized [N,{model.condition_dim}] tensors")
    atom_count_logits, cell_count_logits = model.count_logits(condition)
    atom_count = torch.distributions.Categorical(logits=atom_count_logits).sample().clamp_min(1).tolist()
    cell_count = torch.distributions.Categorical(logits=cell_count_logits).sample().tolist()
    if args.cell_count_min is not None:
        maximum = args.cell_count_max if args.cell_count_max is not None else args.cell_count_min
        if not 0 <= args.cell_count_min <= maximum <= model.max_cells:
            raise ValueError("cell count range is outside checkpoint capacity")
        cell_count = [args.cell_count_min + i % (maximum-args.cell_count_min+1) for i in range(len(condition))]
    if not 0 < args.radius_min < args.radius_max:
        raise ValueError("radius bounds must satisfy 0 < min < max")
    radical_threshold = args.radical_threshold
    if radical_threshold is None:
        radical_threshold = float(cfg.get("training", {}).get("chemistry", {}).get("radical_projection_threshold", .8))
    if not 0 <= radical_threshold <= 1:
        raise ValueError("radical threshold must be between 0 and 1")
    out = sample(model, diffusion, condition, atom_count, cell_count, args.guidance, atom_map,
        args.progress_every, args.score_clip, args.coordinate_clip, args.radius_min, args.radius_max,
        args.closed_shell, radical_threshold)
    molecules = []
    for row in range(len(condition)):
        mask = out["atom_mask"][row]
        if not bool(out["projection_valid"][row]): molecules.append(None); continue
        try:
            molecules.append(reconstruct(out["atom_types"][row, mask], out["charges"][row, mask],
                out["hydrogens"][row, mask], out["radicals"][row, mask], out["bonds"][row][mask][:, mask],
                out["positions"][row, mask], atom_map, closed_shell=args.closed_shell,
                charge_min=model.charge_min, chirality=out["chirality"][row, mask],
                bond_aromatic=out["bond_aromatic"][row][mask][:, mask]))
        except Exception: molecules.append(None)
    diagnostics = out.pop("numerical_diagnostics")
    torch.save({"samples": {k: v.cpu() for k, v in out.items()}, "molecules": molecules,
        "numerical_diagnostics": diagnostics}, args.output)
    print(json.dumps({"requested": len(condition), "projection_valid": int(out["projection_valid"].sum()),
        "rdkit_valid": sum(x is not None for x in molecules), "numerical_invalid": int((~out["numerical_valid"]).sum()),
        "nonfinite": diagnostics["nonfinite_counts"], "output": str(Path(args.output).resolve())}))


if __name__ == "__main__": main()
