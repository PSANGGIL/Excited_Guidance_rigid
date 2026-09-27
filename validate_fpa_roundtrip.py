#!/usr/bin/env python3
"""Validate clean DB encode -> constraint decode -> RDKit round trips."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import torch

from polyring_diffusion.fpa_data_template_free import TemplateFreeFPAEntityDataset, collate_template_free_fpa
from polyring_diffusion.fpa_projection_template_free import project_template_free_fpa
from sample_fpa_template_free import decode_exact, reconstruct


def oracle_logits(labels, classes):
    logits = torch.full((*labels.shape, classes), -12.)
    logits.scatter_(-1, labels.long().unsqueeze(-1), 12.)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_qcdge_a10_full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sidecar-dir", default="hydrogen_counts")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--require-hydrogens", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    dataset = TemplateFreeFPAEntityDataset(args.data, args.split, sidecar_dir=args.sidecar_dir, frame_augmentation=False)
    if args.require_hydrogens and not dataset.has_hydrogen_counts:
        raise RuntimeError("hydrogen sidecars missing; run augment_qcdge_hydrogen_counts.py")
    stats = {"total": 0, "exact_decode_valid": 0, "exact_canonical": 0,
             "membership_exact": 0, "bond_exact": 0, "projection_valid": 0,
             "projected_rdkit_valid": 0, "projected_canonical": 0,
             "aromatic_total": 0, "aromatic_exact_canonical": 0,
             "aromatic_projected_canonical": 0}
    failures = []
    for index in range(min(args.limit, len(dataset))):
        encoded = dataset[index]
        batch = collate_template_free_fpa([encoded])
        na, nc = int(batch["atom_mask"].sum()), int(batch["cell_mask"].sum())
        prediction = {
            "membership_logits": oracle_logits(batch["cell_membership"], 2),
            "bond_logits": oracle_logits(batch["atom_bond_types"], 5),
            "bond_aromatic_logits": oracle_logits(batch["atom_bond_aromatic"].long(), 2),
        }
        sample = {
            "atom_types": batch["atom_types"].clone(), "charges": batch["atom_charges"].clone()+2,
            "hydrogens": batch["atom_hydrogens"].clone(), "radicals": batch["atom_radicals"].clone(), "positions": batch["atom_positions"].clone(),
            "bonds": batch["atom_bond_types"].clone(), "bond_aromatic": batch["atom_bond_aromatic"].clone(),
            "cell_sizes": batch["cell_sizes"].clone(),
            "cell_aromatic": batch["cell_aromatic"].clone(), "cell_centers": batch["cell_centers"].clone(),
            "cell_frames": batch["cell_frames"].clone(), "cell_radii": batch["cell_radii"].clone(),
            "cell_relations": batch["cell_relations"].clone(),
            "membership": batch["cell_membership"].long().clone(),
            "atom_mask": batch["atom_mask"].clone(), "cell_mask": batch["cell_mask"].clone(),
        }
        original_membership, original_bonds = sample["membership"].clone(), sample["bonds"].clone()
        output = project_template_free_fpa(sample, prediction, dataset.atom_map)
        membership_ok = torch.equal(output["membership"][0, :nc, :na], original_membership[0, :nc, :na])
        bond_ok = torch.equal(output["bonds"][0, :na, :na], original_bonds[0, :na, :na])
        projection_ok = bool(output["projection_valid"][0])
        from rdkit import Chem
        reference = Chem.MolFromSmiles(encoded["source_smiles"])
        reference_smiles = Chem.MolToSmiles(Chem.RemoveHs(reference), canonical=True, isomericSmiles=False) if reference is not None else None
        try:
            exact_mol = decode_exact(encoded, dataset.atom_map)
            exact_smiles = Chem.MolToSmiles(Chem.RemoveHs(exact_mol), canonical=True, isomericSmiles=False)
        except Exception:
            exact_mol = None; exact_smiles = None
        try:
            projected_mol = reconstruct(output["atom_types"][0, :na], output["charges"][0, :na],
                output["hydrogens"][0, :na], output["radicals"][0, :na], output["bonds"][0, :na, :na],
                output["positions"][0, :na], dataset.atom_map,
                bond_aromatic=output["bond_aromatic"][0, :na, :na])
            projected_smiles = Chem.MolToSmiles(Chem.RemoveHs(projected_mol), canonical=True, isomericSmiles=False)
        except Exception:
            projected_mol = None; projected_smiles = None
        exact_ok = reference_smiles is not None and exact_smiles == reference_smiles
        projected_ok = reference_smiles is not None and projected_smiles == reference_smiles
        aromatic = reference is not None and any(atom.GetIsAromatic() for atom in reference.GetAtoms())
        stats["total"] += 1; stats["exact_decode_valid"] += int(exact_mol is not None); stats["exact_canonical"] += int(exact_ok)
        stats["membership_exact"] += int(membership_ok); stats["bond_exact"] += int(bond_ok)
        stats["projection_valid"] += int(projection_ok); stats["projected_rdkit_valid"] += int(projected_mol is not None)
        stats["projected_canonical"] += int(projected_ok); stats["aromatic_total"] += int(aromatic)
        stats["aromatic_exact_canonical"] += int(aromatic and exact_ok); stats["aromatic_projected_canonical"] += int(aromatic and projected_ok)
        if not all((membership_ok, bond_ok, projection_ok, exact_ok, projected_ok)) and len(failures) < 20:
            failures.append({"index": index, "source_row": int(encoded["source_row"]),
                "membership": membership_ok, "bond": bond_ok, "projection": projection_ok,
                "exact_canonical": exact_ok, "projected_canonical": projected_ok,
                "reason": int(output["projection_reason"][0]), "cells": nc})
        if args.progress_every and (index+1) % args.progress_every == 0:
            print(json.dumps({"progress": index+1, "total": min(args.limit, len(dataset)), "exact": stats["exact_canonical"], "projected": stats["projected_canonical"]}), flush=True)
    report = {"stats": stats, "hydrogen_sidecars": dataset.has_hydrogen_counts,
        "excluded_mapping_failures": dataset.exclude_mapping_failures, "failures": failures}
    rendered = json.dumps(report, indent=2)
    if args.output: Path(args.output).write_text(rendered+"\n")
    print(rendered)


if __name__ == "__main__": main()
