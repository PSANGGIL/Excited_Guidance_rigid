#!/usr/bin/env python3
"""Build a lossless, deduplicated aromatic-only QCDGE A+B database.

The output uses the shard schema consumed by QCDGEHierarchyDataset and writes
the chemistry sidecar in the same pass, so atom indices cannot drift between
coordinates, bonds, hydrogens, charges, radicals, and aromatic flags.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import h5py
import torch
from rdkit import Chem, RDLogger

from augment_qcdge_hydrogen_counts_v2 import mapped_atom_states
from convert_qcdge_a10 import ATOM_MAP, molecule, offsets, cat, template_hash

RDLogger.DisableLog("rdApp.*")


def smiles_of(group):
    for key in ("SMILES_RDKIT_CAN", "SMILES_RDKIT", "SMILES_PYBEL"):
        value = group.attrs.get(key)
        if value is not None:
            return value.decode() if isinstance(value, bytes) else str(value)
    raise ValueError("missing SMILES")


def select(inputs, seed):
    selected, rejected, seen = [], Counter(), {}
    for source, path in inputs.items():
        with h5py.File(path, "r") as h5:
            for cid, group in h5.items():
                try:
                    mol = Chem.MolFromSmiles(smiles_of(group))
                    if mol is None:
                        rejected["smiles_parse"] += 1; continue
                    if len(Chem.GetMolFrags(mol)) != 1:
                        rejected["multifragment"] += 1; continue
                    if not any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
                        rejected["non_aromatic"] += 1; continue
                    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                    if canonical in seen:
                        rejected["duplicate_canonical_smiles"] += 1; continue
                    seen[canonical] = (source, cid)
                    selected.append((source, cid, canonical))
                except Exception:
                    rejected["selection_error"] += 1
        print(json.dumps({"event":"selected_source","source":source,
            "selected_so_far":len(selected),"rejected":dict(rejected)}), flush=True)
    random.Random(seed).shuffle(selected)
    return selected, rejected


def encode_record(source, cid, group):
    rec = molecule(group)
    hydrogens, radicals, charges, chirality, aromatic_atoms, bond_map = mapped_atom_states(group)
    if not bool(aromatic_atoms.any()):
        raise ValueError("mapped graph has no aromatic atom")
    if len(hydrogens) != len(rec["positions"]):
        raise ValueError("atom alignment mismatch")
    mapped_bonds = torch.tensor([
        bond_map[tuple(sorted(map(int, edge)))] for edge in rec["bond_indices"].tolist()
    ], dtype=torch.uint8)
    rw = Chem.RWMol()
    for atom_index, (q, h, radical, aromatic) in enumerate(zip(charges, hydrogens, radicals, aromatic_atoms)):
        kind = int(rec["atom_types"][atom_index].to(torch.uint8).argmax())
        atom = Chem.Atom(int(ATOM_MAP[kind])); atom.SetFormalCharge(int(q))
        atom.SetNumExplicitHs(int(h)); atom.SetNumRadicalElectrons(int(radical))
        atom.SetNoImplicit(True); atom.SetIsAromatic(bool(aromatic)); rw.AddAtom(atom)
    bond_kinds = {1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,4:Chem.BondType.AROMATIC}
    for edge, kind in zip(rec["bond_indices"].tolist(), mapped_bonds.tolist()):
        u, v = map(int, edge); rw.AddBond(u, v, bond_kinds[int(kind)])
        if int(kind) == 4: rw.GetBondBetweenAtoms(u, v).SetIsAromatic(True)
    codec_mol = rw.GetMol(); Chem.SanitizeMol(codec_mol); Chem.Kekulize(codec_mol, clearAromaticFlags=True); Chem.SanitizeMol(codec_mol)
    decoded = Chem.MolToSmiles(codec_mol, canonical=True, isomericSmiles=False)
    reference = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_of(group)), canonical=True, isomericSmiles=False)
    if decoded != reference: raise ValueError(f"strict codec identity mismatch: {reference} != {decoded}")
    sidecar = {
        "atom_hydrogens": hydrogens,
        "atom_radicals": radicals,
        "atom_formal_charges": charges,
        "atom_chirality": chirality,
        "atom_aromatic": aromatic_atoms,
        "bond_types": mapped_bonds,
        "bond_aromatic": mapped_bonds.eq(4),
    }
    rec["atom_charges"] = charges.to(torch.int32)
    rec["bond_types"] = mapped_bonds.to(torch.int32)
    return rec, sidecar


def write_shard(root, rel, rows, handles):
    records, sides, errors, stats, templates = [], [], [], Counter(), {}
    for source, cid, canonical in rows:
        try:
            rec, side = encode_record(source, cid, handles[source][cid])
            records.append((source, cid, canonical, rec)); sides.append(side)
            stats["molecules"] += 1; stats["atoms"] += len(rec["positions"])
            stats["rings"] += len(rec["ring"]["signatures"])
            stats[f"source_{source}"] += 1
            for h, sig in zip(rec["ring"]["hashes"], rec["ring"]["signatures"]):
                templates[str(int(h))] = sig
        except Exception as exc:
            errors.append({"source":source,"molecule_id":cid,"error":str(exc)})
            stats["errors"] += 1
    if not records:
        return dict(stats), templates, errors
    rr = [x[3]["ring"] for x in records]
    atom_offsets = offsets(len(x[3]["positions"]) for x in records)
    bond_offsets = offsets(len(x[3]["bond_types"]) for x in records)
    payload = {
        "schema_version": 2, "split": rel.parts[0],
        "molecule_ids": [f"{x[0]}:{x[1]}" for x in records],
        "source_datasets": [x[0] for x in records],
        "source_molecule_ids": [x[1] for x in records],
        "canonical_smiles": [x[2] for x in records],
        "smiles": [x[3]["smiles"] for x in records],
        "source_rows": torch.tensor([int(x[1]) for x in records], dtype=torch.int64),
        "properties": torch.tensor([x[3]["property"] for x in records]),
        "raw_properties": torch.tensor([x[3]["raw_property"] for x in records]),
        "atom_positions": torch.cat([x[3]["positions"] for x in records]),
        "atom_types": torch.cat([x[3]["atom_types"] for x in records]),
        "atom_charges": torch.cat([x[3]["atom_charges"] for x in records]),
        "atom_offsets": atom_offsets,
        "bond_indices": torch.cat([x[3]["bond_indices"] for x in records]),
        "bond_types": torch.cat([x[3]["bond_types"] for x in records]),
        "bond_offsets": bond_offsets,
        "ring_template_hashes": cat(rr,"hashes",(0,),torch.int64),
        "ring_centers": cat(rr,"centers",(0,3),torch.float32),
        "ring_normals": cat(rr,"normals",(0,3),torch.float32),
        "ring_radii": cat(rr,"radii",(0,),torch.float32),
        "ring_offsets": offsets(len(x["signatures"]) for x in rr),
        "ring_atom_indices": cat(rr,"ring_atoms",(0,),torch.int32),
        "ring_atom_offsets": offsets(x["ring_atoms"].numel() for x in rr),
        "per_ring_atom_offsets": [x["ring_atom_offsets"] for x in rr],
        "connections": cat(rr,"connections",(0,6),torch.int32),
        "connection_offsets": offsets(x["connections"].shape[0] for x in rr),
        "connection_shared_atoms": [x["shared"] for x in rr],
        "singlet_energies": torch.stack([x[3]["states"]["singlet_energies"] for x in records]),
        "singlet_oscillator_strengths": torch.stack([x[3]["states"]["singlet_oscillator_strengths"] for x in records]),
        "singlet_mask": torch.stack([x[3]["states"]["singlet_mask"] for x in records]),
        "triplet_energies": torch.stack([x[3]["states"]["triplet_energies"] for x in records]),
        "triplet_oscillator_strengths": torch.stack([x[3]["states"]["triplet_oscillator_strengths"] for x in records]),
        "triplet_mask": torch.stack([x[3]["states"]["triplet_mask"] for x in records]),
        "errors": errors, "stats": dict(stats), "templates": templates,
        "metadata": {"atom_map":["B","C","N","O","F","Si","P","S","Cl","Br","I"],
            "selection":"connected and >=1 RDKit aromatic atom", "geometry_source":"QCDGE ground_state"},
    }
    target = root/rel; target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp"); torch.save(payload,tmp); os.replace(tmp,target)
    side = {
        "atom_hydrogens": torch.cat([x["atom_hydrogens"] for x in sides]),
        "atom_radicals": torch.cat([x["atom_radicals"] for x in sides]),
        "atom_formal_charges": torch.cat([x["atom_formal_charges"] for x in sides]),
        "atom_chirality": torch.cat([x["atom_chirality"] for x in sides]),
        "atom_aromatic": torch.cat([x["atom_aromatic"] for x in sides]),
        "bond_types": torch.cat([x["bond_types"] for x in sides]),
        "bond_aromatic": torch.cat([x["bond_aromatic"] for x in sides]),
        "atom_hydrogen_mask": torch.ones(int(atom_offsets[-1]),dtype=torch.bool),
        "atom_offsets": atom_offsets, "bond_offsets": bond_offsets, "errors": [],
    }
    side_target=root/"hydrogen_counts_v4_aromatic"/rel
    side_target.parent.mkdir(parents=True,exist_ok=True)
    side_tmp=side_target.with_suffix(".tmp");torch.save(side,side_tmp);os.replace(side_tmp,side_target)
    return dict(stats), templates, errors


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--a",type=Path,default=Path("joint_kekule_diffusion/raw_qcdge/A_10.hdf5"))
    p.add_argument("--b",type=Path,default=Path("joint_kekule_diffusion/raw_qcdge/B_10.hdf5"))
    p.add_argument("--output",type=Path,default=Path("joint_kekule_diffusion/data_qcdge_ab_aromatic"))
    p.add_argument("--shard-size",type=int,default=2000);p.add_argument("--seed",type=int,default=47)
    p.add_argument("--limit",type=int)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    inputs={"A":args.a.resolve(),"B":args.b.resolve()}
    selected,rejected=select(inputs,args.seed)
    if args.limit:selected=selected[:args.limit]
    n=len(selected);cuts=(int(.8*n),int(.9*n)); split_rows={"train":selected[:cuts[0]],"val":selected[cuts[0]:cuts[1]],"test":selected[cuts[1]:]}
    handles={k:h5py.File(v,"r") for k,v in inputs.items()};all_templates={};splits=[];all_errors=[]
    try:
        for split,rows in split_rows.items():
            shards=[];total=Counter()
            for lo in range(0,len(rows),args.shard_size):
                rel=Path(split)/f"shard_{lo:06d}_{min(lo+args.shard_size,len(rows)):06d}.pt"
                stats,templates,errors=write_shard(args.output,rel,rows[lo:lo+args.shard_size],handles)
                total.update(stats);all_templates.update(templates);all_errors.extend(errors);shards.append(str(rel))
                print(json.dumps({"event":"shard","split":split,"done":min(lo+args.shard_size,len(rows)),"total":len(rows),"stats":stats}),flush=True)
            splits.append({"name":split,"source_molecules":len(rows),"shards":shards,"stats":dict(total)})
    finally:
        for h in handles.values():h.close()
    templates=[{"id":i,"hash":int(h),"signature":json.loads(sig)} for i,(h,sig) in enumerate(sorted(all_templates.items(),key=lambda x:x[1]))]
    manifest={"schema_version":2,"source_files":{k:str(v) for k,v in inputs.items()},"selection":"connected, deduplicated canonical graph, at least one RDKit aromatic atom, exact coordinate/SMILES mapping","selected_candidates":n,"rejected_selection":dict(rejected),"conversion_errors":all_errors,"condition_schema":{"singlet":["energy_eV","oscillator_strength","mask"],"triplet":["energy_eV","oscillator_strength","mask"],"max_states":20},"splits":splits,"templates":templates}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"event":"complete","output":str(args.output.resolve()),"candidates":n,"written":sum(x["stats"].get("molecules",0) for x in splits),"errors":len(all_errors),"templates":len(templates)}),flush=True)


if __name__=="__main__":main()
