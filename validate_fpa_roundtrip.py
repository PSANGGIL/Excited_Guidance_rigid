#!/usr/bin/env python3
"""Validate clean DB encode -> constraint decode -> RDKit round trips."""
from __future__ import annotations

import argparse, json
import torch

from polyring_diffusion.fpa_data_template_free import TemplateFreeFPAEntityDataset, collate_template_free_fpa
from polyring_diffusion.fpa_projection_template_free import project_template_free_fpa
from sample_fpa_template_free import reconstruct


def oracle_logits(labels, classes):
    logits = torch.full((*labels.shape, classes), -12.)
    logits.scatter_(-1, labels.long().unsqueeze(-1), 12.)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data_qcdge_a10_full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--require-hydrogens", action="store_true")
    args = parser.parse_args()
    dataset = TemplateFreeFPAEntityDataset(args.data, args.split)
    if args.require_hydrogens and not dataset.has_hydrogen_counts:
        raise RuntimeError("hydrogen sidecars missing; run augment_qcdge_hydrogen_counts.py")
    stats = {"total": 0, "membership_exact": 0, "bond_exact": 0,
             "projection_valid": 0, "rdkit_valid": 0}
    failures = []
    for index in range(min(args.limit, len(dataset))):
        batch = collate_template_free_fpa([dataset[index]])
        na, nc = int(batch["atom_mask"].sum()), int(batch["cell_mask"].sum())
        prediction = {
            "membership_logits": oracle_logits(batch["cell_membership"], 2),
            "bond_logits": oracle_logits(batch["atom_bond_types"], 5),
        }
        sample = {
            "atom_types": batch["atom_types"].clone(), "charges": batch["atom_charges"].clone()+2,
            "hydrogens": batch["atom_hydrogens"].clone(), "radicals": batch["atom_radicals"].clone(), "positions": batch["atom_positions"].clone(),
            "bonds": batch["atom_bond_types"].clone(), "cell_sizes": batch["cell_sizes"].clone(),
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
        try:
            reconstruct(output["atom_types"][0, :na], output["charges"][0, :na],
                output["hydrogens"][0, :na], output["radicals"][0, :na], output["bonds"][0, :na, :na],
                output["positions"][0, :na], dataset.atom_map)
            rdkit_ok = True
        except Exception:
            rdkit_ok = False
        stats["total"] += 1; stats["membership_exact"] += int(membership_ok)
        stats["bond_exact"] += int(bond_ok); stats["projection_valid"] += int(projection_ok)
        stats["rdkit_valid"] += int(rdkit_ok)
        if not all((membership_ok, bond_ok, projection_ok, rdkit_ok)) and len(failures) < 20:
            failures.append({"index": index, "membership": membership_ok, "bond": bond_ok,
                "projection": projection_ok, "rdkit": rdkit_ok,
                "reason": int(output["projection_reason"][0]), "cells": nc})
    print(json.dumps({"stats": stats, "hydrogen_sidecars": dataset.has_hydrogen_counts,
        "failures": failures}, indent=2))


if __name__ == "__main__": main()
