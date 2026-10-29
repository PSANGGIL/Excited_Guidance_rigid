#!/usr/bin/env python3
"""Add lossless per-atom aromatic flags to existing chemistry sidecars."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import networkx as nx
import torch
from rdkit import Chem,RDLogger
RDLogger.DisableLog("rdApp.*")

def bounds(offsets,row): return map(int,offsets[row:row+2])
def kind(bond): return 4 if bond.GetIsAromatic() else int(round(bond.GetBondTypeAsDouble()))
def map_aromatic(raw, sidecar, row):
    a,b=bounds(raw["atom_offsets"],row); ba,bb=bounds(raw["bond_offsets"],row)
    reference=Chem.RemoveHs(Chem.MolFromSmiles(raw["smiles"][row]))
    if reference is None or reference.GetNumAtoms()!=b-a: raise ValueError("reference atom count mismatch")
    atom_map=raw["metadata"]["atom_map"]; types=raw["atom_types"][a:b].to(torch.uint8).argmax(-1)
    left=nx.Graph(); right=nx.Graph()
    for i,t in enumerate(types.tolist()): left.add_node(i,z=Chem.GetPeriodicTable().GetAtomicNumber(atom_map[t]))
    for edge,k in zip(raw["bond_indices"][ba:bb].tolist(),sidecar["bond_types"][ba:bb].tolist()): left.add_edge(*map(int,edge),kind=int(k))
    for atom in reference.GetAtoms(): right.add_node(atom.GetIdx(),z=atom.GetAtomicNum())
    for bond in reference.GetBonds(): right.add_edge(bond.GetBeginAtomIdx(),bond.GetEndAtomIdx(),kind=kind(bond))
    matcher=nx.algorithms.isomorphism.GraphMatcher(left,right,node_match=lambda x,y:x["z"]==y["z"],edge_match=lambda x,y:x["kind"]==y["kind"])
    mapping=next(matcher.isomorphisms_iter(),None)
    if mapping is None: raise ValueError("sidecar/reference graph mismatch")
    atom_aromatic = torch.tensor([reference.GetAtomWithIdx(mapping[i]).GetIsAromatic() for i in range(b-a)],dtype=torch.bool)
    bond_types, bond_aromatic = [], []
    for edge in raw["bond_indices"][ba:bb].tolist():
        u, v = map(int, edge); bond = reference.GetBondBetweenAtoms(mapping[u], mapping[v])
        bond_types.append(4 if bond.GetBondType() == Chem.BondType.AROMATIC else int(round(bond.GetBondTypeAsDouble())))
        bond_aromatic.append(bond.GetIsAromatic())
    return atom_aromatic, torch.tensor(bond_types,dtype=torch.uint8), torch.tensor(bond_aromatic,dtype=torch.bool)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,default=Path("data_qcdge_a10_full")); p.add_argument("--source-dir",default="hydrogen_counts"); p.add_argument("--output-dir",default="hydrogen_counts_v4_aromatic"); args=p.parse_args()
    manifest=json.loads((args.data/"manifest.json").read_text()); total=failed=0
    for split in manifest["splits"]:
        for rel in split["shards"]:
            raw=torch.load(args.data/rel,map_location="cpu",weights_only=False); old=torch.load(args.data/args.source_dir/rel,map_location="cpu",weights_only=False)
            chunks=[]; bond_type_chunks=[]; bond_aromatic_chunks=[]; errors=list(old.get("errors",[])); mask=old["atom_hydrogen_mask"].clone()
            for row in range(len(raw["smiles"])):
                a,b=bounds(raw["atom_offsets"],row)
                ba,bb=bounds(raw["bond_offsets"],row)
                if not bool(mask[a:b].all()):
                    chunks.append(torch.zeros(b-a,dtype=torch.bool)); bond_type_chunks.append(old["bond_types"][ba:bb]); bond_aromatic_chunks.append(old["bond_types"][ba:bb].eq(4)); continue
                try:
                    aa, bt, bar = map_aromatic(raw,old,row); chunks.append(aa); bond_type_chunks.append(bt); bond_aromatic_chunks.append(bar)
                except Exception as exc:
                    chunks.append(torch.zeros(b-a,dtype=torch.bool)); bond_type_chunks.append(old["bond_types"][ba:bb]); bond_aromatic_chunks.append(old["bond_types"][ba:bb].eq(4)); mask[a:b]=False; failed+=1
                    errors.append({"row":row,"molecule_id":str(raw["molecule_ids"][row]),"error":"atom_aromatic: "+str(exc)})
                total+=1
            payload=dict(old); payload["atom_aromatic"]=torch.cat(chunks); payload["bond_types"]=torch.cat(bond_type_chunks); payload["bond_aromatic"]=torch.cat(bond_aromatic_chunks); payload["atom_hydrogen_mask"]=mask; payload["errors"]=errors
            target=args.data/args.output_dir/rel; target.parent.mkdir(parents=True,exist_ok=True); torch.save(payload,target)
            print(json.dumps({"shard":rel,"eligible":total,"new_failures":failed,"output":str(target)}),flush=True)
    print(json.dumps({"total_eligible":total,"new_failures":failed}),flush=True)
if __name__=="__main__": main()
