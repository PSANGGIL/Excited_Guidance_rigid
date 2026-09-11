#!/usr/bin/env python3
"""Check molecular identity after DB encode and template-free decode."""
from __future__ import annotations

import argparse, bisect, json
from pathlib import Path
import torch
from rdkit import Chem, RDLogger

from polyring_diffusion.fpa_data_template_free import TemplateFreeFPAEntityDataset, collate_template_free_fpa
from polyring_diffusion.fpa_projection_template_free import project_template_free_fpa
from sample_fpa_template_free import reconstruct

RDLogger.DisableLog("rdApp.*")


def oracle(labels, classes):
    logits = torch.full((*labels.shape, classes), -12.)
    return logits.scatter_(-1, labels.long().unsqueeze(-1), 12.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_qcdge_a10_full"))
    parser.add_argument("--split", default="train"); parser.add_argument("--limit", type=int, default=2000)
    args = parser.parse_args(); dataset = TemplateFreeFPAEntityDataset(args.data, args.split)
    info = next(x for x in dataset.manifest["splits"] if x["name"] == args.split)
    raw_cache = {}; stats = {"total": 0, "reference_parseable": 0, "canonical_exact": 0,
        "membership_exact": 0, "bond_exact": 0, "projection_valid": 0, "rdkit_valid": 0}
    failures = []
    for index in range(min(args.limit, len(dataset))):
        shard_index = bisect.bisect_right(dataset.ends, index)
        base = 0 if shard_index == 0 else dataset.ends[shard_index-1]
        local_index = index-base
        if shard_index not in raw_cache:
            raw_cache[shard_index] = torch.load(args.data/info["shards"][shard_index], map_location="cpu", weights_only=False)
        raw = raw_cache[shard_index]
        item = dataset[index]; batch = collate_template_free_fpa([item])
        na, nc = int(batch["atom_mask"].sum()), int(batch["cell_mask"].sum())
        sample = {"atom_types": batch["atom_types"].clone(), "charges": batch["atom_charges"].clone()+2,
            "hydrogens": batch["atom_hydrogens"].clone(), "radicals": batch["atom_radicals"].clone(),
            "positions": batch["atom_positions"].clone(), "bonds": batch["atom_bond_types"].clone(),
            "cell_sizes": batch["cell_sizes"].clone(), "cell_aromatic": batch["cell_aromatic"].clone(),
            "cell_centers": batch["cell_centers"].clone(), "cell_frames": batch["cell_frames"].clone(),
            "cell_radii": batch["cell_radii"].clone(), "cell_relations": batch["cell_relations"].clone(),
            "membership": batch["cell_membership"].long().clone(),
            "atom_mask": batch["atom_mask"].clone(), "cell_mask": batch["cell_mask"].clone()}
        original_membership, original_bonds = sample["membership"].clone(), sample["bonds"].clone()
        output = project_template_free_fpa(sample, {"membership_logits": oracle(batch["cell_membership"], 2),
            "bond_logits": oracle(batch["atom_bond_types"], 5)}, dataset.atom_map)
        membership_ok = torch.equal(output["membership"][0, :nc, :na], original_membership[0, :nc, :na])
        bond_ok = torch.equal(output["bonds"][0, :na, :na], original_bonds[0, :na, :na])
        projection_ok = bool(output["projection_valid"][0]); rdkit_ok = canonical_ok = False
        reference = Chem.MolFromSmiles(raw["smiles"][local_index])
        try:
            decoded = reconstruct(output["atom_types"][0, :na], output["charges"][0, :na],
                output["hydrogens"][0, :na], output["radicals"][0, :na],
                output["bonds"][0, :na, :na], output["positions"][0, :na], dataset.atom_map)
            rdkit_ok = True
            if reference is not None:
                ref_smiles = Chem.MolToSmiles(Chem.RemoveHs(reference), canonical=True, isomericSmiles=True)
                out_smiles = Chem.MolToSmiles(Chem.RemoveHs(decoded), canonical=True, isomericSmiles=True)
                canonical_ok = ref_smiles == out_smiles
        except Exception:
            pass
        stats["total"] += 1; stats["reference_parseable"] += int(reference is not None)
        stats["canonical_exact"] += int(canonical_ok); stats["membership_exact"] += int(membership_ok)
        stats["bond_exact"] += int(bond_ok); stats["projection_valid"] += int(projection_ok); stats["rdkit_valid"] += int(rdkit_ok)
        expected_identity = reference is None or canonical_ok
        if not all((membership_ok, bond_ok, projection_ok, rdkit_ok, expected_identity)) and len(failures) < 20:
            failures.append({"index": index, "source_row": int(item["source_row"]),
                "reference_parseable": reference is not None, "canonical_exact": canonical_ok,
                "membership_exact": membership_ok, "bond_exact": bond_ok,
                "projection_valid": projection_ok, "rdkit_valid": rdkit_ok})
    print(json.dumps({"stats": stats, "failures": failures}, indent=2))


if __name__ == "__main__": main()
