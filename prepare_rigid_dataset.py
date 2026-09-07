#!/usr/bin/env python3
"""Build a sharded, lossless atom+rigid-ring database from MolGuidance PT data."""
from __future__ import annotations
import argparse, hashlib, json, os
from collections import Counter
from pathlib import Path
import torch
from rdkit import Chem

SINGLE, FUSED, SPIRO, COMPLEX = 1, 2, 3, 4

def bond_label(b):
    if b.GetIsAromatic(): return 4
    return {1.0:1, 2.0:2, 3.0:3}.get(b.GetBondTypeAsDouble(), 0)

def cycle_order(mol, bond_ids):
    adj = {}
    for bi in bond_ids:
        b=mol.GetBondWithIdx(int(bi)); a,c=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        adj.setdefault(a,[]).append(c); adj.setdefault(c,[]).append(a)
    if not adj or any(len(v)!=2 for v in adj.values()): raise ValueError("non-simple ring")
    start=min(adj); out=[start]; prev=-1; cur=start
    while True:
        choices=[x for x in adj[cur] if x!=prev]
        nxt=min(choices) if len(out)==1 else choices[0]
        if nxt==start: break
        if nxt in out: raise ValueError("repeated ring atom")
        out.append(nxt); prev,cur=cur,nxt
    if len(out)!=len(adj): raise ValueError("incomplete ring traversal")
    return out

def canonical_ring(mol, order):
    candidates=[]; n=len(order)
    for direction in (1,-1):
        for offset in range(n):
            ids=[order[(offset+direction*i)%n] for i in range(n)]; sig=[]
            for i,a in enumerate(ids):
                atom=mol.GetAtomWithIdx(a); bond=mol.GetBondBetweenAtoms(a,ids[(i+1)%n])
                sig.append((atom.GetAtomicNum(),atom.GetFormalCharge(),int(atom.GetIsAromatic()),bond_label(bond)))
            candidates.append((tuple(sig),ids))
    sig,ids=min(candidates,key=lambda x:x[0])
    return json.dumps(sig,separators=(",",":")),ids

def template_hash(sig):
    return int.from_bytes(hashlib.sha256(sig.encode()).digest()[:8],"big") & ((1<<63)-1)

def ring_pose(xyz, ids):
    pts=xyz[torch.tensor(ids)]; center=pts.mean(0); centered=pts-center
    normal=torch.linalg.svd(centered,full_matrices=False).Vh[-1]
    normal=normal/normal.norm().clamp_min(1e-8); pivot=int(torch.argmax(normal.abs()))
    if normal[pivot]<0: normal=-normal
    return center.float(),normal.float(),float(torch.linalg.vector_norm(centered,dim=-1).mean())

def edge_port(order, shared):
    for i,a in enumerate(order):
        b=order[(i+1)%len(order)]
        if a in shared and b in shared: return i,(a,b)
    raise ValueError("non-adjacent fused atoms")

def connections(mol,rings):
    out=[]; shared_out=[]; sets=list(map(set,rings))
    for i in range(len(rings)):
        for j in range(i+1,len(rings)):
            shared=sets[i]&sets[j]
            if len(shared)==1:
                a=next(iter(shared)); out.append([i,j,SPIRO,rings[i].index(a),rings[j].index(a),0]); shared_out.append([a])
            elif len(shared)==2 and mol.GetBondBetweenAtoms(*sorted(shared)) is not None:
                lp,lpair=edge_port(rings[i],shared); rp,rpair=edge_port(rings[j],shared)
                out.append([i,j,FUSED,lp,rp,int(lpair==rpair)]); shared_out.append(sorted(shared))
            elif len(shared)>=2:
                out.append([i,j,COMPLEX,-1,-1,0]); shared_out.append(sorted(shared))
            else:
                direct=[(a,b) for a in rings[i] for b in rings[j] if mol.GetBondBetweenAtoms(a,b)]
                if len(direct)==1:
                    a,b=direct[0]; out.append([i,j,SINGLE,rings[i].index(a),rings[j].index(b),0]); shared_out.append([])
                elif len(direct)>1:
                    out.append([i,j,COMPLEX,-1,-1,0]); shared_out.append([])
    return torch.tensor(out,dtype=torch.int32).reshape(-1,6),shared_out

def extract(smiles,xyz):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: raise ValueError("invalid SMILES")
    if mol.GetNumAtoms()!=len(xyz):
        mol=Chem.AddHs(mol)
        if mol.GetNumAtoms()!=len(xyz): raise ValueError(f"atom mismatch {mol.GetNumAtoms()} != {len(xyz)}")
    sigs=[]; rings=[]; centers=[]; normals=[]; radii=[]
    for br in mol.GetRingInfo().BondRings():
        sig,ids=canonical_ring(mol,cycle_order(mol,br)); c,n,r=ring_pose(xyz,ids)
        sigs.append(sig); rings.append(ids); centers.append(c); normals.append(n); radii.append(r)
    conns,shared=connections(mol,rings); ro=[0]
    for ring in rings: ro.append(ro[-1]+len(ring))
    return {"signatures":sigs,"hashes":torch.tensor([template_hash(s) for s in sigs],dtype=torch.int64),
      "centers":torch.stack(centers) if centers else torch.empty((0,3)),"normals":torch.stack(normals) if normals else torch.empty((0,3)),
      "radii":torch.tensor(radii),"ring_atoms":torch.tensor([a for r in rings for a in r],dtype=torch.int32),
      "ring_atom_offsets":torch.tensor(ro,dtype=torch.int32),"connections":conns,"shared":shared}

def offsets(lengths):
    out=[0]
    for n in lengths: out.append(out[-1]+int(n))
    return torch.tensor(out,dtype=torch.int64)

def cat(records,key,shape,dtype):
    vals=[r[key] for r in records if r[key].numel()]
    return torch.cat(vals) if vals else torch.empty(shape,dtype=dtype)

def convert_split(src_path,out_dir,shard_size,templates):
    src=torch.load(src_path,map_location="cpu",weights_only=False); split=str(src.get("split_name",src_path.name.split("_")[0]))
    dest=out_dir/split; dest.mkdir(parents=True,exist_ok=True); shards=[]; total=Counter()
    for lo in range(0,len(src["smiles"]),shard_size):
        hi=min(lo+shard_size,len(src["smiles"])); path=dest/f"shard_{lo:06d}_{hi:06d}.pt"; shards.append(str(path.relative_to(out_dir)))
        if path.exists():
            old=torch.load(path,map_location="cpu",weights_only=False); total.update(old["stats"]); templates.update({int(k):v for k,v in old["templates"].items()}); print("reuse",path,flush=True); continue
        records=[]; keep=[]; errors=[]; stats=Counter(); local_templates={}
        for idx in range(lo,hi):
            a,b=map(int,src["node_idx_array"][idx])
            try:
                rec=extract(src["smiles"][idx],src["positions"][a:b]); records.append(rec); keep.append(idx); stats["molecules"]+=1
                stats["rings"]+=len(rec["signatures"]); stats["ringless"]+=not rec["signatures"]
                stats["complex_connections"]+=int((rec["connections"][:,2]==COMPLEX).sum()) if rec["connections"].numel() else 0
                for sig,key in zip(rec["signatures"],rec["hashes"].tolist()):
                    if key in templates and templates[key]!=sig: raise RuntimeError("template hash collision")
                    templates[key]=sig; local_templates[str(key)]=sig; stats[f"template:{key}"]+=1
            except Exception as exc:
                stats["errors"]+=1; errors.append({"index":idx,"source_row":int(src["source_rows"][idx]),"error":str(exc)})
        choose=torch.tensor(keep,dtype=torch.long); ar=src["node_idx_array"][choose].tolist(); er=src["edge_idx_array"][choose].tolist()
        pos=[]; at=[]; ac=[]; bi=[]; bt=[]
        for (a,b),(ea,eb) in zip(ar,er):
            pos.append(src["positions"][a:b]); at.append(src["atom_types"][a:b]); ac.append(src["atom_charges"][a:b])
            local=src["bond_idxs"][ea:eb].to(torch.int64).clone()
            if local.numel() and int(local.max())>=b-a: local-=a
            bi.append(local.to(torch.int32)); bt.append(src["bond_types"][ea:eb])
        payload={"schema_version":1,"split":split,"source_indices":choose,"source_rows":[int(src["source_rows"][i]) for i in keep],
          "molecule_ids":[src["molecule_ids"][i] for i in keep],"smiles":[src["smiles"][i] for i in keep],
          "properties":src["properties"][choose],"raw_properties":src["raw_properties"][choose],
          "atom_positions":torch.cat(pos) if pos else torch.empty((0,3)),"atom_types":torch.cat(at) if at else torch.empty((0,src["atom_types"].shape[1]),dtype=torch.bool),
          "atom_charges":torch.cat(ac) if ac else torch.empty(0,dtype=torch.int32),"atom_offsets":offsets(b-a for a,b in ar),
          "bond_indices":torch.cat(bi) if bi else torch.empty((0,2),dtype=torch.int32),"bond_types":torch.cat(bt) if bt else torch.empty(0,dtype=torch.int32),"bond_offsets":offsets(eb-ea for ea,eb in er),
          "ring_template_hashes":cat(records,"hashes",(0,),torch.int64),"ring_centers":cat(records,"centers",(0,3),torch.float32),"ring_normals":cat(records,"normals",(0,3),torch.float32),"ring_radii":cat(records,"radii",(0,),torch.float32),
          "ring_offsets":offsets(len(r["signatures"]) for r in records),"ring_atom_indices":cat(records,"ring_atoms",(0,),torch.int32),"ring_atom_offsets":offsets(r["ring_atoms"].numel() for r in records),"per_ring_atom_offsets":[r["ring_atom_offsets"] for r in records],
          "connections":cat(records,"connections",(0,6),torch.int32),"connection_offsets":offsets(r["connections"].shape[0] for r in records),"connection_shared_atoms":[r["shared"] for r in records],
          "errors":errors,"stats":dict(stats),"templates":local_templates,"metadata":{k:src.get(k) for k in ("atom_map","property_names","raw_property_column","target_transform","target_epsilon","geometry_source","add_hydrogens","bond_representation")}}
        tmp=path.with_suffix(".tmp"); torch.save(payload,tmp); os.replace(tmp,path); total.update(stats); print(f"wrote {path} {len(keep)}/{hi-lo}",flush=True)
    return {"name":split,"source_molecules":len(src["smiles"]),"shards":shards,"stats":dict(total)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,default=Path("data_ring_rigid")); p.add_argument("--shard-size",type=int,default=2000); p.add_argument("--splits",nargs="+",default=["train","val","test"]); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    templates={}; splits=[convert_split(a.source_dir/f"{s}_data_processed.pt",a.output_dir,a.shard_size,templates) for s in a.splits]
    records=[{"id":i,"hash":k,"signature":json.loads(v)} for i,(k,v) in enumerate(sorted(templates.items(),key=lambda x:x[1]))]
    manifest={"schema_version":1,"source_dir":str(a.source_dir.resolve()),"total_source_molecules":sum(x["source_molecules"] for x in splits),"connection_types":{"NONE":0,"SINGLE":1,"FUSED":2,"SPIRO":3,"COMPLEX":4},"splits":splits,"templates":records}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print("templates",len(records),flush=True)
if __name__=="__main__": main()
