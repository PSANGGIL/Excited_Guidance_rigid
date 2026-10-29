#!/usr/bin/env python3
"""Strict reverse sampler for the v5 heavy-atom joint-state DDPM."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors

from .chemistry_v5 import AtomStateCodec, TopologyPrior, decode_heavy_graph, project_heavy_graph
from .model import JointDiffusion, cfg_combine
from .model_v5 import HeavyAtomJointDDPM
from .sample import posterior_step, reverse_coords


@torch.inference_mode()
def sample_batch_v5(
    model,
    diffusion,
    condition,
    batch_size,
    atoms,
    cfg,
    topology_prior,
    observed_counts=None,
    coordinate_clip=25.,
    rng=None,
):
    if batch_size <= 0 or not 0 <= atoms <= model.max_atoms:
        raise ValueError(f"atoms must be 0 or in [1,{model.max_atoms}]")
    if not math.isfinite(cfg) or cfg < 0:
        raise ValueError("CFG scale must be finite and non-negative")
    rng = rng or random.Random()
    device = condition.device
    conditional_count = model.count_head(model.condition_context(condition))
    unconditional_count = model.count_head(model.condition_context(condition, unconditional=True))
    count_logits = unconditional_count + cfg * (conditional_count - unconditional_count) if cfg != 1 else conditional_count
    if not atoms and observed_counts:
        support = torch.zeros(model.max_atoms, dtype=torch.bool, device=device)
        support[torch.as_tensor(observed_counts, device=device) - 1] = True
        count_logits = count_logits.masked_fill(~support[None], -torch.inf)
    counts = torch.full((batch_size,), atoms, dtype=torch.long, device=device) if atoms else torch.distributions.Categorical(logits=count_logits).sample() + 1
    index = torch.arange(model.max_atoms, device=device)
    mask = index[None] < counts[:, None]
    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(model.max_atoms, dtype=torch.bool, device=device)[None]

    def node_marginal(classes):
        return torch.multinomial(diffusion.marginal("atom", classes, device), batch_size * model.max_atoms, replacement=True).reshape(batch_size, model.max_atoms)

    def symmetric_marginal(name, classes):
        probability = diffusion.marginal(name, classes, device)
        value = torch.multinomial(probability, batch_size * model.max_atoms * model.max_atoms, replacement=True).reshape(batch_size, model.max_atoms, model.max_atoms)
        value = torch.where(torch.triu(torch.ones_like(value, dtype=torch.bool), 1), value, value.transpose(1, 2))
        return value.masked_fill(diagonal, 0)

    state = {
        "atom_mask": mask,
        "atom_state": node_marginal(model.state_types),
        "edge_exists": symmetric_marginal("edge", 2),
        "bond_type": symmetric_marginal("bond_type", 3),
        "positions": torch.randn(batch_size, model.max_atoms, 3, device=device) * mask[..., None],
    }
    denominator = mask.sum(1, keepdim=True).clamp_min(1)[..., None]
    state["positions"] -= ((state["positions"] * mask[..., None]).sum(1, keepdim=True) / denominator) * mask[..., None]
    final_outputs = None
    for time_index in reversed(range(diffusion.timesteps)):
        timestep = torch.full((batch_size,), time_index, dtype=torch.long, device=device)
        conditional = model(condition, timestep, state)
        outputs = cfg_combine(conditional, model(condition, timestep, state, unconditional=True), cfg) if cfg != 1 else conditional
        state["atom_state"] = posterior_step(state["atom_state"], outputs["state_logits"], time_index, diffusion, model.state_types, mask, False, "atom")
        state["edge_exists"] = posterior_step(state["edge_exists"], outputs["edge_logits"], time_index, diffusion, 2, pair & ~diagonal, True, "edge")
        state["bond_type"] = posterior_step(state["bond_type"], outputs["bond_type_logits"], time_index, diffusion, 3, pair & ~diagonal, True, "bond_type")
        state["positions"] = reverse_coords(state["positions"], outputs["coordinate_eps"], time_index, diffusion, mask, coordinate_clip)
        final_outputs = outputs

    projected_states = torch.zeros(batch_size, model.max_atoms, dtype=torch.long)
    projected_bonds = torch.zeros(batch_size, model.max_atoms, model.max_atoms, dtype=torch.long)
    projected_hydrogens = torch.zeros(batch_size, model.max_atoms, dtype=torch.long)
    valid = torch.zeros(batch_size, dtype=torch.bool)
    reasons = []
    target_cycle_rank = torch.zeros(batch_size, dtype=torch.long)
    realized_cycle_rank = torch.zeros(batch_size, dtype=torch.long)
    for batch_index, count in enumerate(counts.cpu().tolist()):
        result = project_heavy_graph(
            final_outputs["state_logits"][batch_index, :count].cpu(),
            final_outputs["edge_logits"][batch_index, :count, :count].cpu(),
            final_outputs["bond_type_logits"][batch_index, :count, :count].cpu(),
            final_outputs["aromatic_edge_logits"][batch_index, :count, :count].cpu(),
            model.codec,
            topology_prior,
            rng,
        )
        projected_states[batch_index, :count] = result.state_ids
        projected_bonds[batch_index, :count, :count] = result.bonds
        projected_hydrogens[batch_index, :count] = result.hydrogens
        valid[batch_index] = result.valid
        reasons.append(result.reason)
        target_cycle_rank[batch_index] = result.target_cycle_rank
        realized_cycle_rank[batch_index] = result.realized_cycle_rank
    return {
        "atom_count": counts.cpu(),
        "atom_state": projected_states,
        "hydrogen": projected_hydrogens,
        "bond": projected_bonds,
        "positions": state["positions"].cpu(),
        "projection_valid": valid,
        "projection_reason": reasons,
        "target_cycle_rank": target_cycle_rank,
        "realized_cycle_rank": realized_cycle_rank,
    }


def molecule_with_coordinates(mol, positions):
    output = Chem.Mol(mol)
    output.RemoveAllConformers()
    conformer = Chem.Conformer(output.GetNumAtoms())
    conformer.Set3D(True)
    for index, (x, y, z) in enumerate(positions.tolist()):
        conformer.SetAtomPosition(index, (float(x), float(y), float(z)))
    output.AddConformer(conformer, assignId=True)
    return output


def load_known_hp1kg(root):
    known = set()
    for split in ("train", "val", "test"):
        for row in torch.load(Path(root) / f"{split}.pt", map_location="cpu", weights_only=False):
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is not None:
                known.add(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False))
    return known


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--root", default="joint_kekule_diffusion/data_hp1kg_nevpt2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=0, help="0 samples from the conditional count head")
    parser.add_argument("--s1", type=float, default=3.640037775)
    parser.add_argument("--t1", type=float, default=3.111467600)
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--coordinate-clip", type=float, default=25.)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.num <= 0 or args.batch_size <= 0 or args.atoms < 0:
        raise ValueError("num and batch-size must be positive; atoms must be non-negative")
    RDLogger.DisableLog("rdApp.*")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(Path(args.metadata).read_text())
    if metadata.get("model_kind") != "heavy_atom_joint_state_d3pm_v5":
        raise ValueError("checkpoint metadata is not v5 heavy-atom joint-state diffusion")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = metadata["args"]
    codec = AtomStateCodec.from_metadata(metadata["atom_states"])
    prior = TopologyPrior.from_metadata(metadata["topology_prior"])
    max_atoms = int(metadata["max_atoms"])
    model = HeavyAtomJointDDPM(
        codec, model_args["hidden"], max_atoms, model_args["layers"], model_args["condition_dropout"],
        model_args["s1_rbf_bins"], model_args["t1_rbf_bins"], model_args["condition_rbf_min"],
        model_args["condition_rbf_max"], model_args["condition_rbf_width_scale"],
    )
    model.load_state_dict(checkpoint["model"])
    model.cuda().eval()
    diffusion = JointDiffusion(marginals=metadata["diffusion_marginals"]).cuda()
    stats = metadata["condition_stats"]
    normalized = torch.tensor([[
        (args.s1 - stats["s1_energy_mean"]) / stats["s1_energy_std"],
        (args.t1 - stats["t1_energy_mean"]) / stats["t1_energy_std"],
    ]], device="cuda")
    known = load_known_hp1kg(args.root)
    records = []
    unique_smiles = set()
    unique_molecules = []
    started = time.time()
    observed_counts = metadata["fixed_atom_count"]
    for begin in range(0, args.num, args.batch_size):
        size = min(args.batch_size, args.num - begin)
        sampled = sample_batch_v5(model, diffusion, normalized.expand(size, -1), size, args.atoms, args.cfg, prior, observed_counts, args.coordinate_clip, rng)
        for row_index in range(size):
            index = begin + row_index
            count = int(sampled["atom_count"][row_index])
            reason = sampled["projection_reason"][row_index]
            decoded = None
            if bool(sampled["projection_valid"][row_index]):
                decoded = decode_heavy_graph(
                    sampled["atom_state"][row_index, :count].tolist(),
                    sampled["bond"][row_index, :count, :count], codec,
                )
                if not decoded.valid:
                    reason = decoded.error or "strict decode failed"
            valid = bool(decoded and decoded.valid)
            smiles = decoded.smiles if valid else ""
            novel = bool(smiles and smiles not in known)
            unique = bool(novel and smiles not in unique_smiles)
            molecule = decoded.molecule if valid else None
            rings = int(rdMolDescriptors.CalcNumRings(molecule)) if molecule is not None else 0
            record = {
                "index": index,
                "atom_count": count,
                "smiles": smiles,
                "valid": int(valid),
                "connected": int(valid and decoded.connected),
                "radical_electrons": int(sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())) if molecule is not None else -1,
                "net_charge": int(sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())) if molecule is not None else 999,
                "hydrogens": int(sum(decoded.hydrogens)) if valid else -1,
                "rings": rings,
                "aromatic_rings": decoded.aromatic_rings if valid else 0,
                "target_cycle_rank": int(sampled["target_cycle_rank"][row_index]),
                "realized_cycle_rank": int(sampled["realized_cycle_rank"][row_index]),
                "novel_vs_db": int(novel),
                "unique_novel": int(unique),
                "error": "" if valid else reason,
            }
            records.append(record)
            if unique:
                unique_smiles.add(smiles)
                unique_molecules.append(molecule_with_coordinates(molecule, sampled["positions"][row_index, :count]))
        with (output_dir / "molecules.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps({"event": "generation", "completed": len(records), "valid": sum(row["valid"] for row in records)}), flush=True)

    error_counts = collections.Counter(row["error"] for row in records if row["error"])
    summary = {
        "requested": args.num,
        "completed": len(records),
        "valid": sum(row["valid"] for row in records),
        "connected": sum(row["connected"] for row in records),
        "closed_shell": sum(row["valid"] and row["radical_electrons"] == 0 for row in records),
        "neutral": sum(row["valid"] and row["net_charge"] == 0 for row in records),
        "aromatic": sum(row["aromatic_rings"] > 0 for row in records),
        "novel_vs_db": sum(row["novel_vs_db"] for row in records),
        "unique_novel": len(unique_molecules),
        "checkpoint_epoch": checkpoint["epoch"],
        "s1": args.s1,
        "t1": args.t1,
        "cfg": args.cfg,
        "atoms": args.atoms,
        "failure_reasons": dict(error_counts),
        "elapsed_s": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    writer = Chem.SDWriter(str(output_dir / "unique_novel_connected.sdf"))
    for index, molecule in enumerate(unique_molecules):
        molecule.SetProp("_Name", f"generated_{index:06d}")
        molecule.SetProp("SMILES", Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
        writer.write(molecule)
    writer.close()

    # PNG is always a true RDKit 2D depiction, independent of generated 3D.
    for page_start in range(0, len(unique_molecules), 100):
        page = []
        legends = []
        for molecule in unique_molecules[page_start:page_start + 100]:
            depicted = Chem.Mol(molecule)
            depicted.RemoveAllConformers()
            rdDepictor.Compute2DCoords(depicted, canonOrient=True, clearConfs=True)
            page.append(depicted)
            legends.append(Chem.MolToSmiles(depicted, canonical=True, isomericSmiles=False))
        Draw.MolsToGridImage(page, molsPerRow=10, subImgSize=(300, 240), legends=legends).save(str(output_dir / f"novel_valid_{page_start // 100:03d}.png"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()


__all__ = ["sample_batch_v5", "main"]
