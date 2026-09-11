#!/usr/bin/env python3
"""CFG sampler and RDKit reconstruction for template-free FPA v3."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch
import yaml

from polyring_diffusion.chem import element_record
from polyring_diffusion.fpa_diffusion_v2 import masked_center
from polyring_diffusion.fpa_model_template_free import TemplateFreeFPAEntityDenoiser
from polyring_diffusion.fpa_projection_template_free import project_template_free_fpa
from polyring_diffusion.so3_diffusion import MolecularDiffusion


def symmetric_step(diffusion, noisy, logits, t, token, mask):
    out = diffusion.p_tokens(noisy, logits, t, token, mask)
    upper = torch.triu(out, 1)
    return upper+upper.transpose(1, 2)


def cfg_predict(model, args, condition, guidance):
    conditional = model(*args, condition)
    drop = torch.ones(len(condition), dtype=torch.bool, device=condition.device)
    unconditional = model(*args, condition, drop)
    return {key: unconditional[key]+guidance*(value-unconditional[key]) for key, value in conditional.items()}


def reconstruct(atom_types, charges, hydrogens, radicals, bonds, positions, atom_map, charge_min=-2):
    from rdkit import Chem
    rw = Chem.RWMol()
    for atom_type, charge, hydrogen, radical in zip(atom_types.tolist(), charges.tolist(), hydrogens.tolist(), radicals.tolist()):
        atom = Chem.Atom(atom_map[atom_type])
        atom.SetFormalCharge(int(charge)+charge_min)
        atom.SetNumExplicitHs(int(hydrogen)); atom.SetNumRadicalElectrons(int(radical)); atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    kinds = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for i in range(len(atom_types)):
        for j in range(i+1, len(atom_types)):
            kind = int(bonds[i, j])
            if kind in kinds:
                if kind == 4:
                    rw.GetAtomWithIdx(i).SetIsAromatic(True); rw.GetAtomWithIdx(j).SetIsAromatic(True)
                rw.AddBond(i, j, kinds[kind])
    mol = rw.GetMol(); conformer = Chem.Conformer(len(atom_types))
    for i, point in enumerate(positions.tolist()): conformer.SetAtomPosition(i, point)
    mol.AddConformer(conformer)
    Chem.SanitizeMol(mol)
    return Chem.AddHs(mol, explicitOnly=False, addCoords=True)


@torch.no_grad()
def sample(model, diffusion, condition, atom_count, cell_count, guidance=2., atom_map=None):
    device = condition.device; batch = len(condition)
    na, nc = max(atom_count), max(1, max(cell_count))
    atom_mask = torch.arange(na, device=device)[None] < torch.tensor(atom_count, device=device)[:, None]
    cell_mask = torch.arange(nc, device=device)[None] < torch.tensor(cell_count, device=device)[:, None]
    atom_type = torch.where(atom_mask, model.atom_mask_token, model.atom_pad_token)
    charge = torch.full((batch, na), model.charge_mask_token, device=device, dtype=torch.long)
    hydrogen = torch.full((batch, na), model.hydrogen_mask_token, device=device, dtype=torch.long)
    radical = torch.full((batch, na), model.radical_mask_token, device=device, dtype=torch.long)
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
    for step in reversed(range(diffusion.timesteps)):
        t = torch.full((batch,), step, device=device, dtype=torch.long)
        args = (atom_type, charge, hydrogen, radical, atom_pos, bond, atom_mask, size, aromatic,
            cell_pos, frame, log_radius, relation, membership, cell_mask, t)
        prediction = cfg_predict(model, args, condition, guidance)
        atom_pos = masked_center(diffusion.p_coordinates(atom_pos, prediction["atom_eps"], t, atom_mask), atom_mask)
        cell_pos = diffusion.p_coordinates(cell_pos, prediction["cell_eps"], t, cell_mask)
        log_radius = diffusion.p_coordinates(log_radius.unsqueeze(-1), prediction["radius_eps"].unsqueeze(-1), t, cell_mask).squeeze(-1)
        frame = diffusion.p_frames(frame, prediction["rotation_score"], t, cell_mask)
        atom_type = diffusion.p_tokens(atom_type, prediction["atom_type_logits"], t, model.atom_mask_token, atom_mask)
        charge = diffusion.p_tokens(charge, prediction["charge_logits"], t, model.charge_mask_token, atom_mask)
        hydrogen = diffusion.p_tokens(hydrogen, prediction["hydrogen_logits"], t, model.hydrogen_mask_token, atom_mask)
        radical = diffusion.p_tokens(radical, prediction["radical_logits"], t, model.radical_mask_token, atom_mask)
        size = diffusion.p_tokens(size, prediction["cell_size_logits"], t, model.size_mask_token, cell_mask)
        aromatic = diffusion.p_tokens(aromatic, prediction["cell_aromatic_logits"], t, model.aromatic_mask_token, cell_mask)
        bond = symmetric_step(diffusion, bond, prediction["bond_logits"], t, 5, atom_mask[:, :, None]&atom_mask[:, None, :])
        relation = symmetric_step(diffusion, relation, prediction["relation_logits"], t, model.relation_mask_token, cell_mask[:, :, None]&cell_mask[:, None, :])
        membership = diffusion.p_tokens(membership, prediction["membership_logits"], t, 2, cell_mask[:, :, None]&atom_mask[:, None, :])
    out = {"atom_types": atom_type, "charges": charge, "hydrogens": hydrogen, "radicals": radical,
        "positions": atom_pos, "bonds": bond, "cell_sizes": model.decode_size(size),
        "cell_aromatic": aromatic, "cell_centers": cell_pos, "cell_frames": frame,
        "cell_radii": log_radius.exp().clamp(.1, 10.), "cell_relations": relation,
        "membership": membership, "atom_mask": atom_mask, "cell_mask": cell_mask}
    return project_template_free_fpa(out, prediction, atom_map) if atom_map is not None else out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--conditions", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--guidance", type=float, default=2.)
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if state.get("representation") != "template_free_fpa_v3": raise ValueError("incompatible checkpoint representation")
    cfg, atom_map = state["config"], state["atom_map"]
    model_cfg = dict(cfg["model"]); model_cfg.update(num_atom_types=len(atom_map),
        atom_radii=[element_record(x).vdw_radius/3 for x in atom_map],
        max_atoms=state["max_atoms"], max_cells=state["max_cells"])
    model = TemplateFreeFPAEntityDenoiser(**model_cfg).cuda(); model.load_state_dict(state["model"]); model.eval()
    diffusion = MolecularDiffusion(**cfg["diffusion"]).cuda()
    condition = torch.load(args.conditions, map_location="cuda").float()
    if condition.ndim != 2 or condition.shape[1] != model.condition_dim:
        raise ValueError(f"conditions must be standardized [N,{model.condition_dim}] tensors")
    atom_count = torch.distributions.Categorical(logits=model.atom_count(condition)).sample().clamp_min(1).tolist()
    cell_count = torch.distributions.Categorical(logits=model.cell_count(condition)).sample().tolist()
    out = sample(model, diffusion, condition, atom_count, cell_count, args.guidance, atom_map)
    molecules = []
    for row in range(len(condition)):
        mask = out["atom_mask"][row]
        if not bool(out["projection_valid"][row]): molecules.append(None); continue
        try:
            molecules.append(reconstruct(out["atom_types"][row, mask], out["charges"][row, mask],
                out["hydrogens"][row, mask], out["radicals"][row, mask], out["bonds"][row][mask][:, mask],
                out["positions"][row, mask], atom_map))
        except Exception: molecules.append(None)
    torch.save({"samples": {k: v.cpu() for k, v in out.items()}, "molecules": molecules}, args.output)
    print(json.dumps({"requested": len(condition), "projection_valid": int(out["projection_valid"].sum()),
        "rdkit_valid": sum(x is not None for x in molecules), "output": str(Path(args.output).resolve())}))


if __name__ == "__main__": main()
