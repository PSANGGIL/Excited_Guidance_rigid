#!/usr/bin/env python3
"""Reverse sampler for v6 joint aromatic/Kekule diffusion."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import random
import time
from pathlib import Path

import networkx as nx
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdDepictor, rdMolDescriptors

from .chemistry_v6 import (
    AtomStateCodec,
    ChemistryPriorV6,
    assess_structure,
    decode_heavy_graph,
    project_heavy_graph_v6,
)
from .model import JointDiffusion, cfg_combine
from .model_v6 import HeavyAtomJointDDPMV6
from .sample import posterior_step, reverse_coords


def synchronize_joint_edge_state(state, outputs, pair, diagonal):
    """Enforce support relations after every categorical posterior step.

    Aromatic membership implies a present non-triple edge.  Applying this at
    every reverse step makes aromatic topology and Kekule order a coupled
    reverse state rather than a final-only aromaticity threshold.
    """
    valid = pair & ~diagonal
    edge = state["edge_exists"].masked_fill(~valid, 0)
    aromatic = (state["aromatic_edge"] * edge).masked_fill(~valid, 0)
    bond_type = state["bond_type"].masked_fill(~valid, 0)
    aromatic_triple = aromatic.bool() & (bond_type == 2)
    if aromatic_triple.any():
        replacement = outputs["bond_type_logits"][..., :2].argmax(-1)
        bond_type = torch.where(aromatic_triple, replacement, bond_type)
    state["edge_exists"] = edge
    state["aromatic_edge"] = aromatic
    state["bond_type"] = bond_type
    return state


def project_aromatic_cycles_in_reverse(state, outputs, model, counts):
    """Resolve aromatic cycle membership and a global Kekule matching in x_t.

    This is deliberately applied only at sparse late reverse steps: it gives
    the denoiser a cycle-consistent state without putting a Python graph
    traversal on all 500 steps.  A biconnected cyclic component becomes
    aromatic only when all atom roles allow it and the component-level mean
    aromatic posterior is at least 0.5.
    """
    edge = state["edge_exists"].detach().cpu()
    aromatic = state["aromatic_edge"].detach().cpu()
    bond_type = state["bond_type"].detach().cpu()
    atom_state = state["atom_state"].detach().cpu()
    aromatic_probability = outputs["aromatic_edge_logits"].float().softmax(-1)[..., 1].detach().cpu()
    type_probability = outputs["bond_type_logits"].float().softmax(-1).detach().cpu()
    roles = model.state_role_id.detach().cpu()
    for batch_index, count in enumerate(counts.detach().cpu().tolist()):
        graph = nx.Graph()
        graph.add_nodes_from(range(count))
        graph.add_edges_from(
            (i, j) for i in range(count) for j in range(i + 1, count)
            if int(edge[batch_index, i, j])
        )
        chosen = set()
        for component in nx.biconnected_components(graph):
            subgraph = graph.subgraph(component)
            if subgraph.number_of_edges() < subgraph.number_of_nodes():
                continue
            component_edges = list(subgraph.edges())
            if not all(int(roles[int(atom_state[batch_index, node])]) > 0 for node in component):
                continue
            mean_probability = sum(float(aromatic_probability[batch_index, i, j]) for i, j in component_edges) / len(component_edges)
            if mean_probability >= .5:
                chosen.update(tuple(sorted((i, j))) for i, j in component_edges)
        aromatic[batch_index, :count, :count] = False
        matching_graph = nx.Graph()
        for i, j in chosen:
            aromatic[batch_index, i, j] = aromatic[batch_index, j, i] = True
            bond_type[batch_index, i, j] = bond_type[batch_index, j, i] = 0
            if int(roles[int(atom_state[batch_index, i])]) == 2 or int(roles[int(atom_state[batch_index, j])]) == 2:
                continue
            single = max(float(type_probability[batch_index, i, j, 0]), 1e-12)
            double = max(float(type_probability[batch_index, i, j, 1]), 1e-12)
            matching_graph.add_edge(i, j, weight=math.log(double) - math.log(single))
        for i, j in nx.max_weight_matching(matching_graph, maxcardinality=True):
            bond_type[batch_index, i, j] = bond_type[batch_index, j, i] = 1
    state["aromatic_edge"] = aromatic.to(state["aromatic_edge"].device)
    state["bond_type"] = bond_type.to(state["bond_type"].device)
    return state


def _condition_logits_on_sample(logits, sampled_state, strength=4.):
    """Make the t=0 categorical sample participate in strict projection."""
    one_hot = torch.nn.functional.one_hot(sampled_state.long(), logits.shape[-1]).to(logits.dtype)
    return logits + float(strength) * (2 * one_hot - 1)


@torch.inference_mode()
def sample_batch_v6(
    model,
    diffusion,
    condition,
    batch_size,
    atoms,
    cfg,
    chemistry_prior,
    observed_counts=None,
    coordinate_clip=25.,
    rng=None,
    enforce_support=True,
    apply_topology_prior=True,
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
        probability = diffusion.marginal("atom", classes, device)
        return torch.multinomial(probability, batch_size * model.max_atoms, replacement=True).reshape(batch_size, model.max_atoms)

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
        "aromatic_edge": symmetric_marginal("aromatic_edge", 2),
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
        state["aromatic_edge"] = posterior_step(state["aromatic_edge"], outputs["aromatic_edge_logits"], time_index, diffusion, 2, pair & ~diagonal, True, "aromatic_edge")
        state = synchronize_joint_edge_state(state, outputs, pair, diagonal)
        if time_index == 0 or (time_index <= 100 and time_index % 20 == 0):
            state = project_aromatic_cycles_in_reverse(state, outputs, model, counts)
        state["positions"] = reverse_coords(state["positions"], outputs["coordinate_eps"], time_index, diffusion, mask, coordinate_clip)
        final_outputs = outputs

    projected_states = torch.zeros(batch_size, model.max_atoms, dtype=torch.long)
    projected_bonds = torch.zeros(batch_size, model.max_atoms, model.max_atoms, dtype=torch.long)
    projected_aromatic = torch.zeros(batch_size, model.max_atoms, model.max_atoms, dtype=torch.bool)
    projected_hydrogens = torch.zeros(batch_size, model.max_atoms, dtype=torch.long)
    valid = torch.zeros(batch_size, dtype=torch.bool)
    reasons = []
    warnings = []
    target_cycle_rank = torch.zeros(batch_size, dtype=torch.long)
    realized_cycle_rank = torch.zeros(batch_size, dtype=torch.long)
    topology_acceptance = torch.zeros(batch_size, dtype=torch.float32)
    for batch_index, count in enumerate(counts.cpu().tolist()):
        # Projection repairs chemistry but must not discard the categorical x0
        # sampled by the reverse chain.  A finite bias keeps alternatives
        # available when neutrality/valence makes the sampled state infeasible.
        state_logits = _condition_logits_on_sample(
            final_outputs["state_logits"][batch_index, :count],
            state["atom_state"][batch_index, :count],
        ).cpu()
        edge_logits = _condition_logits_on_sample(
            final_outputs["edge_logits"][batch_index, :count, :count],
            state["edge_exists"][batch_index, :count, :count],
        ).cpu()
        bond_type_logits = _condition_logits_on_sample(
            final_outputs["bond_type_logits"][batch_index, :count, :count],
            state["bond_type"][batch_index, :count, :count],
        ).cpu()
        aromatic_logits = _condition_logits_on_sample(
            final_outputs["aromatic_edge_logits"][batch_index, :count, :count],
            state["aromatic_edge"][batch_index, :count, :count],
        ).cpu()
        result = project_heavy_graph_v6(
            state_logits, edge_logits, bond_type_logits, aromatic_logits,
            model.codec, chemistry_prior, rng,
            enforce_support=enforce_support,
            apply_topology_prior=apply_topology_prior,
        )
        projected_states[batch_index, :count] = result.state_ids
        projected_bonds[batch_index, :count, :count] = result.bonds
        projected_aromatic[batch_index, :count, :count] = result.aromatic_edges
        projected_hydrogens[batch_index, :count] = result.hydrogens
        valid[batch_index] = result.valid
        reasons.append(result.reason)
        warnings.append(result.warnings)
        target_cycle_rank[batch_index] = result.target_cycle_rank
        realized_cycle_rank[batch_index] = result.realized_cycle_rank
        topology_acceptance[batch_index] = result.topology_acceptance
    return {
        "atom_count": counts.cpu(),
        "atom_state": projected_states,
        "hydrogen": projected_hydrogens,
        "bond": projected_bonds,
        "aromatic_edge": projected_aromatic,
        "positions": state["positions"].cpu(),
        "projection_valid": valid,
        "projection_reason": reasons,
        "projection_warnings": warnings,
        "target_cycle_rank": target_cycle_rank,
        "realized_cycle_rank": realized_cycle_rank,
        "topology_acceptance": topology_acceptance,
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
    parser.add_argument("--atoms", type=int, default=0)
    parser.add_argument("--s1", type=float, default=3.640037775)
    parser.add_argument("--t1", type=float, default=3.111467600)
    parser.add_argument("--cfg", type=float, default=1.)
    parser.add_argument("--coordinate-clip", type=float, default=25.)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--evaluate-3d", action="store_true", help="run ETKDG+MMFF/UFF for every valid molecule")
    parser.add_argument("--no-support-filter", action="store_true")
    parser.add_argument("--no-topology-prior", action="store_true")
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
    if metadata.get("model_kind") != "heavy_atom_joint_aromatic_d3pm_v6":
        raise ValueError("checkpoint metadata is not v6 joint aromatic diffusion")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_args = metadata["args"]
    codec = AtomStateCodec.from_metadata(metadata["atom_states"])
    prior = ChemistryPriorV6.from_metadata(metadata["chemistry_prior"])
    max_atoms = int(metadata["max_atoms"])
    model = HeavyAtomJointDDPMV6(
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
        sampled = sample_batch_v6(
            model, diffusion, normalized.expand(size, -1), size, args.atoms, args.cfg,
            prior, observed_counts, args.coordinate_clip, rng,
            not args.no_support_filter, not args.no_topology_prior,
        )
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
            assessment = assess_structure(
                molecule, sampled["projection_warnings"][row_index], args.evaluate_3d, args.seed + index,
            ) if molecule is not None else None
            record = {
                "index": index,
                "atom_count": count,
                "smiles": smiles,
                "valid": int(valid),
                "chemically_clean": int(bool(assessment and assessment.chemically_clean)),
                "etkdg_success": int(bool(assessment and assessment.etkdg_success)) if args.evaluate_3d else -1,
                "forcefield_converged": int(bool(assessment and assessment.forcefield_converged)) if args.evaluate_3d else -1,
                "forcefield_final_energy": assessment.forcefield_final_energy if assessment else None,
                "etkdg_relaxation_energy_per_heavy_atom": assessment.strain_energy_per_heavy_atom if assessment else None,
                "steric_clashes": assessment.steric_clashes if assessment else None,
                "structure_warnings": ";".join(assessment.warnings) if assessment else "",
                "rings": int(rdMolDescriptors.CalcNumRings(molecule)) if molecule is not None else 0,
                "aromatic_rings": decoded.aromatic_rings if valid else 0,
                "target_cycle_rank": int(sampled["target_cycle_rank"][row_index]),
                "realized_cycle_rank": int(sampled["realized_cycle_rank"][row_index]),
                "topology_acceptance": float(sampled["topology_acceptance"][row_index]),
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
        print(json.dumps({"event": "generation", "completed": len(records), "valid": sum(row["valid"] for row in records), "clean": sum(row["chemically_clean"] for row in records)}), flush=True)

    error_counts = collections.Counter(row["error"] for row in records if row["error"])
    warning_counts = collections.Counter(warning for row in records for warning in row["structure_warnings"].split(";") if warning)
    valid_count = sum(row["valid"] for row in records)
    clean_count = sum(row["chemically_clean"] for row in records)
    evaluated = [row for row in records if row["etkdg_success"] >= 0]
    energies = [row["etkdg_relaxation_energy_per_heavy_atom"] for row in evaluated if row["etkdg_relaxation_energy_per_heavy_atom"] is not None]
    summary = {
        "requested": args.num,
        "completed": len(records),
        "valid": valid_count,
        "valid_rate": valid_count / max(1, len(records)),
        "chemically_clean_valid": clean_count,
        "chemically_clean_valid_rate": clean_count / max(1, len(records)),
        "etkdg_success_rate": sum(row["etkdg_success"] == 1 for row in evaluated) / max(1, len(evaluated)) if evaluated else None,
        "mean_etkdg_relaxation_energy_per_heavy_atom": sum(energies) / len(energies) if energies else None,
        "aromatic": sum(row["aromatic_rings"] > 0 for row in records),
        "novel_vs_db": sum(row["novel_vs_db"] for row in records),
        "unique_novel": len(unique_molecules),
        "checkpoint_epoch": checkpoint["epoch"],
        "s1": args.s1,
        "t1": args.t1,
        "cfg": args.cfg,
        "atoms": args.atoms,
        "failure_reasons": dict(error_counts),
        "structure_warning_counts": dict(warning_counts),
        "elapsed_s": time.time() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    writer = Chem.SDWriter(str(output_dir / "unique_novel_connected.sdf"))
    for index, molecule in enumerate(unique_molecules):
        molecule.SetProp("_Name", f"generated_{index:06d}")
        molecule.SetProp("SMILES", Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
        writer.write(molecule)
    writer.close()
    for page_start in range(0, len(unique_molecules), 100):
        page, legends = [], []
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


__all__ = ["sample_batch_v6", "synchronize_joint_edge_state", "project_aromatic_cycles_in_reverse", "main"]
