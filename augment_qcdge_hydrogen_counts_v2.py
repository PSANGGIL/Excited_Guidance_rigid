#!/usr/bin/env python3
"""Map SMILES hydrogen states onto QCDGE coordinate atom indices via graph isomorphism."""
from __future__ import annotations

import argparse, json
from pathlib import Path
import h5py
import networkx as nx
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds

RDLogger.DisableLog("rdApp.*")


def xyz_molecule(group):
    atomic_numbers = np.asarray(group["ground_state"]["labels"][()], dtype=np.int64).reshape(-1)
    xyz = np.asarray(group["ground_state"]["coords"][()], dtype=np.float32)
    lines = [str(len(atomic_numbers)), "QCDGE"] + [
        f"{Chem.GetPeriodicTable().GetElementSymbol(int(z))} {x:.10f} {y:.10f} {zz:.10f}"
        for z, (x, y, zz) in zip(atomic_numbers, xyz)
    ]
    mol = Chem.MolFromXYZBlock("\n".join(lines)+"\n")
    if mol is None: raise ValueError("XYZ parse failed")
    rdDetermineBonds.DetermineBonds(mol, charge=0, covFactor=1.3,
        allowChargedFragments=True, embedChiral=True, useHueckel=False)
    Chem.SanitizeMol(mol)
    return Chem.RemoveAllHs(mol)


def graph(mol):
    out = nx.Graph()
    for atom in mol.GetAtoms():
        out.add_node(atom.GetIdx(), z=atom.GetAtomicNum(), charge=atom.GetFormalCharge(), aromatic=int(atom.GetIsAromatic()))
    for bond in mol.GetBonds():
        kind = 4 if bond.GetIsAromatic() else int(round(bond.GetBondTypeAsDouble()))
        out.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), kind=kind)
    return out


def mapping_score(mapping, coordinate_mol, smiles_mol):
    score = 0
    for source, target in mapping.items():
        a, b = coordinate_mol.GetAtomWithIdx(source), smiles_mol.GetAtomWithIdx(target)
        score += 4*int(a.GetFormalCharge() != b.GetFormalCharge())+int(a.GetIsAromatic() != b.GetIsAromatic())
        heavy = sum(1 if bond.GetIsAromatic() else int(round(bond.GetBondTypeAsDouble())) for bond in a.GetBonds())
        hydrogens = int(b.GetTotalNumHs(includeNeighbors=True))
        maximum = 4 if a.GetAtomicNum() == 6 or (a.GetAtomicNum() == 7 and a.GetFormalCharge() > 0) else 3 if a.GetAtomicNum() in (5, 7, 15) else 2 if a.GetAtomicNum() in (8, 16) else 1
        score += 20*max(0, heavy+hydrogens-maximum)
    for bond in coordinate_mol.GetBonds():
        left, right = mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
        other = smiles_mol.GetBondBetweenAtoms(left, right)
        a = 4 if bond.GetIsAromatic() else int(round(bond.GetBondTypeAsDouble()))
        b = 4 if other.GetIsAromatic() else int(round(other.GetBondTypeAsDouble()))
        score += int(a != b)
    return score


def mapped_atom_states(group):
    coordinate_mol = xyz_molecule(group)
    smiles_mol = None
    for key in ("SMILES_RDKIT_CAN", "SMILES_RDKIT", "SMILES_PYBEL"):
        value = group.attrs.get(key)
        if value is None: continue
        candidate = Chem.MolFromSmiles(value.decode() if isinstance(value, bytes) else str(value))
        if candidate is not None:
            smiles_mol = Chem.RemoveHs(candidate); break
    if smiles_mol is None: raise ValueError("no parseable SMILES")
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        graph(coordinate_mol), graph(smiles_mol),
        node_match=lambda a, b: a["z"] == b["z"],
    )
    mappings = list(matcher.isomorphisms_iter())
    if not mappings: raise ValueError("SMILES/coordinate heavy graphs are not isomorphic")
    mapping = min(mappings, key=lambda item: mapping_score(item, coordinate_mol, smiles_mol))
    hydrogens = torch.tensor([smiles_mol.GetAtomWithIdx(mapping[index]).GetTotalNumHs(includeNeighbors=True) for index in range(coordinate_mol.GetNumAtoms())], dtype=torch.uint8)
    radicals = torch.tensor([smiles_mol.GetAtomWithIdx(mapping[index]).GetNumRadicalElectrons() for index in range(coordinate_mol.GetNumAtoms())], dtype=torch.uint8)
    charges = torch.tensor([smiles_mol.GetAtomWithIdx(mapping[index]).GetFormalCharge() for index in range(coordinate_mol.GetNumAtoms())], dtype=torch.int8)
    bond_map = {}
    for bond in coordinate_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); other = smiles_mol.GetBondBetweenAtoms(mapping[i], mapping[j])
        bond_map[tuple(sorted((i, j)))] = 4 if other.GetIsAromatic() else int(round(other.GetBondTypeAsDouble()))
    return hydrogens, radicals, charges, bond_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data_qcdge_a10_full"))
    parser.add_argument("--source", type=Path); parser.add_argument("--split")
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args(); manifest = json.loads((args.data/"manifest.json").read_text())
    source = args.source or Path(manifest["source_file"])
    with h5py.File(source, "r") as h5:
        for split in manifest["splits"]:
            if args.split and split["name"] != args.split: continue
            paths = split["shards"][:args.max_shards] if args.max_shards else split["shards"]
            for rel in paths:
                raw = torch.load(args.data/rel, map_location="cpu", weights_only=False)
                chunks = []; radical_chunks = []; charge_chunks = []; bond_chunks = []; masks = []; errors = []
                for row, molecule_id in enumerate(raw["molecule_ids"]):
                    a, b = map(int, raw["atom_offsets"][row:row+2]); ba, bb = map(int, raw["bond_offsets"][row:row+2])
                    try:
                        values, radical_values, charges, bond_map = mapped_atom_states(h5[str(molecule_id)])
                        mapped_bonds = torch.tensor([bond_map[tuple(sorted(map(int, edge)))] for edge in raw["bond_indices"][ba:bb]], dtype=torch.uint8)
                        chunks.append(values); radical_chunks.append(radical_values); charge_chunks.append(charges); bond_chunks.append(mapped_bonds); masks.append(torch.ones_like(values, dtype=torch.bool))
                    except Exception as exc:
                        chunks.append(torch.zeros(b-a, dtype=torch.uint8)); radical_chunks.append(torch.zeros(b-a, dtype=torch.uint8)); charge_chunks.append(raw["atom_charges"][a:b].to(torch.int8)); bond_chunks.append(raw["bond_types"][ba:bb].to(torch.uint8)); masks.append(torch.zeros(b-a, dtype=torch.bool))
                        errors.append({"row": row, "molecule_id": str(molecule_id), "error": str(exc)})
                counts = torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.uint8)
                radicals = torch.cat(radical_chunks) if radical_chunks else torch.empty(0, dtype=torch.uint8)
                charges = torch.cat(charge_chunks) if charge_chunks else torch.empty(0, dtype=torch.int8)
                bonds = torch.cat(bond_chunks) if bond_chunks else torch.empty(0, dtype=torch.uint8)
                valid = torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool)
                if len(counts) != int(raw["atom_offsets"][-1]): raise RuntimeError(f"atom alignment mismatch: {rel}")
                if len(bonds) != int(raw["bond_offsets"][-1]): raise RuntimeError(f"bond alignment mismatch: {rel}")
                target = args.data/"hydrogen_counts"/rel; target.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"atom_hydrogens": counts, "atom_radicals": radicals, "atom_formal_charges": charges, "bond_types": bonds, "atom_hydrogen_mask": valid, "atom_offsets": raw["atom_offsets"], "bond_offsets": raw["bond_offsets"], "errors": errors}, target)
                print(json.dumps({"shard": rel, "atoms": len(counts), "errors": len(errors), "output": str(target)}), flush=True)


if __name__ == "__main__": main()
