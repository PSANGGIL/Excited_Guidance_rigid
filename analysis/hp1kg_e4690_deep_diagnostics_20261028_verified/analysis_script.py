#!/usr/bin/env python3
"""Endpoint diagnostics only; no new sampling, optimization, or model edits."""
import argparse
from collections import Counter, deque
import csv
from decimal import Decimal
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT/"runs/generation_stage8_hp1kg_best_e4690_s1_minus_2t1_0_n1000_cfg1_20261027"
PREV=ROOT/"analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027"
OUT=ROOT/"analysis/hp1kg_e4690_deep_diagnostics_20261028"


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module); return module


A=load(ROOT/"analysis/analyze_hp1kg_e3911_n10000_20261023.py","audit_helpers")
A.ROOT,A.GEN,A.XTB=ROOT,GEN,GEN/"xtb_stda_diffusion_coords_24x1"


def graph_metrics(mol):
    from rdkit import Chem
    import numpy as np
    n=mol.GetNumAtoms(); adj=[set() for _ in range(n)]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); adj[i].add(j); adj[j].add(i)
    tri=sum(len(adj[i]&adj[j]) for i in range(n) for j in adj[i] if i<j)//3
    c4=sum(len(adj[i]&adj[j])*(len(adj[i]&adj[j])-1)//2 for i in range(n) for j in range(i+1,n))//2
    edge_cycles=Counter()
    for i in range(n):
        for j in adj[i]:
            if j<=i: continue
            seen={i}; queue=deque([(i,0)]); found=None
            while queue and found is None:
                current,d=queue.popleft()
                for nb in adj[current]:
                    if (current==i and nb==j) or (current==j and nb==i): continue
                    if nb==j: found=d+2; break
                    if nb not in seen: seen.add(nb); queue.append((nb,d+1))
            edge_cycles[found or 0]+=1
    rings=list(mol.GetRingInfo().AtomRings()); ringatoms=set().union(*map(set,rings)) if rings else set()
    systems=[]
    for ring in rings:
        group=set(ring)
        for s in [s for s in systems if s&group]: group|=s; systems.remove(s)
        systems.append(group)
    comp=[]
    for ring in rings:
        bonds=[mol.GetBondBetweenAtoms(ring[k],ring[(k+1)%len(ring)]) for k in range(len(ring))]
        comp.append(dict(size=len(ring),hetero_atoms=sum(mol.GetAtomWithIdx(i).GetAtomicNum()!=6 for i in ring),
                         aromatic=all(b.GetIsAromatic() for b in bonds),
                         double_bonds=sum(b.GetBondType()==Chem.BondType.DOUBLE for b in bonds),
                         triple_bonds=sum(b.GetBondType()==Chem.BondType.TRIPLE for b in bonds)))
    return dict(exact_triangles=tri,exact_4cycles=c4,shortest_cycle_per_edge=dict(edge_cycles),
        long_cycle_edge_count=sum(v for k,v in edge_cycles.items() if k>=7),
        macro_cycle_edge_count=sum(v for k,v in edge_cycles.items() if k>=9),
        ring_atom_fraction=len(ringatoms)/n,nonring_atoms=n-len(ringatoms),largest_ring_system=max(map(len,systems),default=0),
        atoms_in_two_plus_rings=sum(sum(i in r for r in rings)>=2 for i in range(n)),
        ring_composition=comp)


def property_fail(atom):
    try: atom.UpdatePropertyCache(strict=True); return False
    except (ValueError,RuntimeError): return True


def sanitize(mol):
    from rdkit import Chem
    result=Chem.Mol(mol)
    try:
        Chem.SanitizeMol(result)
        radicals=sum(a.GetNumRadicalElectrons() for a in result.GetAtoms())
        charge=sum(a.GetFormalCharge() for a in result.GetAtoms())
        connected=len(Chem.GetMolFrags(result))==1
        return dict(valid=True,radicals=radicals,net_charge=charge,connected=connected,
                    eligible=connected and charge==0 and radicals==0),result
    except (ValueError,RuntimeError) as exc:
        return dict(valid=False,eligible=False,error=str(exc)),None


def endpoint_diagnostic(saved,vocabulary,original,joint):
    import numpy as np
    from rdkit import Chem
    states=saved["nodes"]["s_1"].argmax(-1).numpy()
    probs=saved["nodes"]["s_endpoint_probs"].numpy()
    src,dst=saved["edge_index"].numpy(); upper=src<dst
    ix=np.where(upper)[0]; ix=ix[np.argsort(src[ix]*len(states)+dst[ix])]
    src,dst=src[ix],dst[ix]
    bonds=saved["edges"]["e_1"].argmax(-1).numpy()[ix]
    ep=saved["edges"]["e_endpoint_probs"].numpy()[ix]
    assert max(states)<=len(vocabulary) and max(bonds)<=3
    real=[i for i,x in enumerate(states) if x<len(vocabulary)]; mapping={old:new for new,old in enumerate(real)}
    rw=Chem.RWMol(); heavy=np.zeros(len(states),dtype=int)
    for i in real:
        s=vocabulary[states[i]]; atom=Chem.Atom(s["symbol"])
        atom.SetFormalCharge(s["formal_charge"]); atom.SetNumExplicitHs(s["hydrogens"]); atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    types=[None,Chem.BondType.SINGLE,Chem.BondType.DOUBLE,Chem.BondType.TRIPLE]
    ignored=0
    for i,j,b in zip(src,dst,bonds):
        if not b: continue
        if i not in mapping or j not in mapping: ignored+=1; continue
        rw.AddBond(mapping[i],mapping[j],types[b]); heavy[i]+=b; heavy[j]+=b
    mol=rw.GetMol(); result=dict(generation_index=saved["generation_index"],raw_valid=original["raw_valid"],
        requested=original["requested_heavy_atoms"],actual=len(real),slots=len(states),ignored_bonds_to_fake=ignored)
    assert len(real)==original["sampled_real_atoms"]
    noH=Chem.Mol(mol)
    for atom in noH.GetAtoms(): atom.SetNumExplicitHs(0)
    changed=Chem.Mol(mol); atoms=[]; alternatives_possible=True; h_edits=0; delta_h=0
    over=heavy_over=0
    for i in real:
        a=mol.GetAtomWithIdx(mapping[i]); s=vocabulary[states[i]]; h=s["hydrogens"]
        fullbad=property_fail(a); heavybad=property_fail(noH.GetAtomWithIdx(mapping[i]))
        over+=fullbad; heavy_over+=heavybad
        supported={int(k) for k,n in s["train_heavy_valence_counts"].items() if n>0}
        options=[(idx,t) for idx,t in enumerate(vocabulary) if t["symbol"]==s["symbol"]
                 and t["formal_charge"]==s["formal_charge"] and t["train_heavy_valence_counts"].get(str(heavy[i]),0)>0]
        if options:
            idx,t=min(options,key=lambda item:(abs(item[1]["hydrogens"]-h),-float(probs[i,item[0]]),item[0]))
            changed.GetAtomWithIdx(mapping[i]).SetNumExplicitHs(t["hydrogens"])
            h_edits+=t["hydrogens"]!=h; delta_h+=t["hydrogens"]-h
        else: alternatives_possible=False
        atoms.append(dict(generation_index=saved["generation_index"],slot_index=i,atom_index=mapping[i],
            state_index=int(states[i]),element=s["symbol"],formal_charge=s["formal_charge"],hydrogens=h,
            heavy_bond_order=int(heavy[i]),total_valence=int(heavy[i]+h),heavy_degree=a.GetDegree(),
            original_valence_error=fullbad,error_with_zero_H=heavybad,
            heavy_valence_seen_for_selected_state=int(heavy[i]) in supported,
            H_only_train_supported_option=bool(options),raw_molecule_valid=original["raw_valid"]))
    classification,clean=sanitize(mol)
    assert classification["valid"]==original["raw_valid"]
    if clean is not None:
        assert Chem.MolToSmiles(clean,isomericSmiles=False)==original["smiles"]
        result.update(graph_metrics(clean)); result.update(radicals=classification["radicals"])
    cf,_=sanitize(changed) if alternatives_possible else (dict(valid=False,eligible=False),None)
    result.update(overvalent_atoms=over,heavy_only_overvalent_atoms=heavy_over,
        all_overflows_H_removable=over>0 and heavy_over==0,
        H_only_train_supported_assignment_exists=alternatives_possible,H_only_changed_atoms=h_edits,H_only_total_H_delta=delta_h,
        H_only_counterfactual_valid=cf["valid"],H_only_counterfactual_eligible=cf["eligible"],
        H_only_counterfactual_radicals=cf.get("radicals"),
        atom_probability_singleton_fraction=float(((probs>0).sum(-1)==1).mean()),
        bond_probability_singleton_fraction=float(((ep>0).sum(-1)==1).mean()),
        atom_probability_max_ge_0p999999_fraction=float((probs.max(-1)>=.999999).mean()),
        bond_probability_max_ge_0p999999_fraction=float((ep.max(-1)>=.999999).mean()),
        node_max_probability_mean=float(probs.max(-1).mean()),bond_max_probability_mean=float(ep.max(-1).mean()),
        atom_argmax_diff_from_raw=int(np.sum(probs.argmax(-1)!=states)),bond_argmax_diff_from_raw=int(np.sum(ep.argmax(-1)!=bonds)))
    # Necessary feasibility bounds for the EXISTING positive-support MILP.
    allowed=[(i,int(v)) for i,s in enumerate(vocabulary) for v,n in s["train_heavy_valence_counts"].items() if n>0]
    allowed.append((len(vocabulary),0))
    choices=[[(idx,v) for idx,v in allowed if probs[i,idx]>0] for i in range(len(states))]
    fake_possible=[any(idx==len(vocabulary) for idx,v in opts) for opts in choices]
    real_possible=[any(idx<len(vocabulary) for idx,v in opts) for opts in choices]
    lower=sum(not f for f in fake_possible); upper_count=sum(real_possible)
    min_val=np.zeros(len(states)); max_val=np.zeros(len(states))
    for i,j,p in zip(src,dst,ep):
        choices_b=np.flatnonzero(p>0)
        min_val[i]+=choices_b.min(); min_val[j]+=choices_b.min()
        max_val[i]+=choices_b.max(); max_val[j]+=choices_b.max()
    local_impossible=sum(not opts or min_val[i]>max(v for k,v in opts) or max_val[i]<min(v for k,v in opts) for i,opts in enumerate(choices))
    bad_count=not lower<=original["requested_heavy_atoms"]<=upper_count
    result.update(projection_count_lower_bound=lower,projection_count_upper_bound=upper_count,
        projection_count_bound_impossible=bad_count,projection_local_valence_bound_impossible_atoms=local_impossible,
        joint_status=joint["decoder"]["status"],joint_valid=joint["raw_valid"],joint_atom_edits=joint["decoder"].get("atom_state_edits"),
        joint_bond_edits=joint["decoder"].get("bond_edits"),joint_identical_raw_graph=bool(clean is not None and joint["raw_valid"] and original["smiles"]==joint["smiles"]))
    if bad_count or local_impossible: assert not joint["raw_valid"]
    return result,atoms


def run(output):
    import numpy as np
    import torch
    from rdkit import Chem,RDLogger
    torch.set_num_threads(1); RDLogger.DisableLog("rdApp.error"); RDLogger.DisableLog("rdApp.warning")
    manifest=json.loads((GEN/"manifest.json").read_text()); data_dir=Path(manifest["data_manifest"]["directory"])
    vocab=json.loads((data_dir/"joint_atom_state_metadata.json").read_text())["states"]
    raw=json.loads((GEN/"raw_analysis/records.json").read_text()); raw={r["generation_index"]:r for r in raw}
    joint=json.loads((GEN/"joint_analysis/records.json").read_text()); joint={r["generation_index"]:r for r in joint}
    previous=list(csv.DictReader((PREV/"all_1000_structure_energy.csv").open())); prev={int(r["generation_index"]):r for r in previous}
    endpoints=sorted((GEN/"raw_endpoints").glob("batch_*.pt"))
    srcs=[GEN/"manifest.json",GEN/"raw_analysis/records.json",GEN/"joint_analysis/records.json",PREV/"summary.json",
          PREV/"all_1000_structure_energy.csv",PREV/"db_reference_and_xtb_s1_minus_2t1.csv",
          data_dir/"joint_atom_state_metadata.json",data_dir/"train_data_processed.pt",Path(__file__).resolve(),*endpoints]
    hashes={str(p):A.sha256(p) for p in srcs}
    status=dict(phase="endpoint_audit",started_utc=A.stamp(),pid=os.getpid(),completed=0)
    A.json_file(output/"status.json",status)
    rows=[]; atoms=[]
    for path in endpoints:
        for item in torch.load(path,map_location="cpu",weights_only=False):
            r,a=endpoint_diagnostic(item,vocab,raw[item["generation_index"]],joint[item["generation_index"]])
            old=prev[r["generation_index"]]
            r["xtb_success"]=old["calculation_status"]=="ok"
            r["target_hit"]=r["xtb_success"] and abs(Decimal(old["q_exact"]))<=Decimal(".1")
            for key in ("s1","t1","q","s1_f","heavy_rmsd_A","nearest_train_tanimoto"):
                r[key]=float(old[key]) if old.get(key) else None
            r["wbo_warning"]=old.get("wbo_change_warning")=="True"
            r["smiles"]=old.get("smiles")
            rows.append(r); atoms.extend(a)
        status.update(completed=len(rows),updated_utc=A.stamp()); A.json_file(output/"status.json",status)
    assert len(rows)==1000 and len({r["generation_index"] for r in rows})==1000
    invalid=[r for r in rows if not r["raw_valid"]]; valid=[r for r in rows if r["raw_valid"]]
    success=[r for r in rows if r["xtb_success"]]; near=[r for r in success if r["target_hit"]]
    countmod=load(GEN/"source_snapshot/molguidance/data_processing/conditional_count.py","frozen_count")
    sampler=countmod.ConditionalAtomCountSampler.from_processed_file(data_dir/"train_data_processed.pt")
    probs,diagnostics=sampler.probabilities([0.],bandwidth=.15,nearest_k=128); weights=probs[0].numpy().astype(float)
    assert abs(float(diagnostics.effective_sample_size[0])-17.04051399230957)<1e-4
    refs=json.loads((GEN/"db_graph_reference.json").read_text())["records"]
    train=sorted([r for r in refs if r["split"]=="train"],key=lambda r:r["split_index"])
    original=torch.load(ROOT/"joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz/train.pt",map_location="cpu",weights_only=False)
    dbcalc={r["database_id"]:r for r in csv.DictReader((PREV/"db_reference_and_xtb_s1_minus_2t1.csv").open())}
    assert len(train)==len(weights)==len(original)
    for r,o,w,n in zip(train,original,weights,sampler.atom_counts):
        mol=Chem.MolFromSmiles(r["smiles"]); r.update(graph_metrics(mol))
        r.update(id=o["id"],q_reference=float(Decimal(str(o["s1"]))-2*Decimal(str(o["t1"]))),
            q_xtb=float(dbcalc[o["id"]]["calculated_q"]),kernel_weight=float(w),s1_xtb=float(dbcalc[o["id"]]["s1"]),
            t1_xtb=float(dbcalc[o["id"]]["t1"]),s1_reference=o["s1"],t1_reference=o["t1"])
        assert int(n)==r["heavy_atoms"]
    def weighted(key,w): return float(np.dot(w,[r[key] for r in train])/sum(w))
    def graph_summary(part):
        if not part:return dict(n=0)
        return dict(n=len(part),exact_triangle_molecules=sum(r["exact_triangles"]>0 for r in part),
            exact_4cycle_molecules=sum(r["exact_4cycles"]>0 for r in part),
            unavoidable_7plus_edge_molecules=sum(r["long_cycle_edge_count"]>0 for r in part),
            unavoidable_9plus_edge_molecules=sum(r["macro_cycle_edge_count"]>0 for r in part),
            ring_atom_fraction=A.stats([r["ring_atom_fraction"] for r in part]),nonring_atoms=A.stats([r["nonring_atoms"] for r in part]),
            all_atoms_in_rings=sum(r["nonring_atoms"]==0 for r in part),largest_ring_system=A.stats([r["largest_ring_system"] for r in part]),
            small_ring_total=sum(x["size"]<=4 for r in part for x in r["ring_composition"]),
            small_ring_allcarbon=sum(x["size"]<=4 and x["hetero_atoms"]==0 for r in part for x in r["ring_composition"]),
            seven_plus_total=sum(x["size"]>=7 for r in part for x in r["ring_composition"]),
            seven_plus_allcarbon=sum(x["size"]>=7 and x["hetero_atoms"]==0 for r in part for x in r["ring_composition"]),
            ring_5_6_hetero_fraction=sum(x["size"] in (5,6) and x["hetero_atoms"]>0 for r in part for x in r["ring_composition"])/max(1,sum(x["size"] in (5,6) for r in part for x in r["ring_composition"])))
    forced=[r for r in rows if r["projection_count_bound_impossible"] or r["projection_local_valence_bound_impossible_atoms"]]
    result=dict(created_utc=A.stamp(),source_sha256=hashes,
        valence=dict(invalid_molecules=len(invalid),invalid_with_any_heavy_only_overflow=sum(r["heavy_only_overvalent_atoms"]>0 for r in invalid),
            invalid_all_overflows_H_removable=sum(r["all_overflows_H_removable"] for r in invalid),
            invalid_H_only_train_assignment_exists=sum(r["H_only_train_supported_assignment_exists"] for r in invalid),
            invalid_H_only_counterfactual_valid=sum(r["H_only_counterfactual_valid"] for r in invalid),
            invalid_H_only_counterfactual_eligible=sum(r["H_only_counterfactual_eligible"] for r in invalid),
            valence_error_atoms=sum(a["original_valence_error"] for a in atoms),
            heavy_only_error_atoms=sum(a["error_with_zero_H"] for a in atoms),
            overvalence_atom_patterns=dict(Counter(f"{a['element']} charge={a['formal_charge']} H={a['hydrogens']} heavyBO={a['heavy_bond_order']}" for a in atoms if a["original_valence_error"]).most_common()),
            valid_radical_molecules=sum(r.get("radicals",0)>0 for r in valid),
            valid_radicals_H_only_counterfactual_eligible=sum(r.get("radicals",0)>0 and r["H_only_counterfactual_eligible"] for r in valid),
            note="Counterfactual H-only reassignment: fixed heavy atoms, formal charges and all heavy bonds; uses train-supported valence/H states. Not new generated/calculated successes and not proof that H alone caused the error."),
        projection=dict(status_counts=dict(Counter(r["joint_status"] for r in rows)),
            raw_invalid_rescued=sum(not r["raw_valid"] and r["joint_valid"] for r in rows),
            raw_valid_to_joint_invalid=sum(r["raw_valid"] and not r["joint_valid"] for r in rows),
            identical_raw_graph=sum(r["joint_identical_raw_graph"] for r in rows),
            successes_with_atom_edits=sum(r["joint_valid"] and r["joint_atom_edits"]>0 for r in rows),
            successes_with_bond_edits=sum(r["joint_valid"] and r["joint_bond_edits"]>0 for r in rows),
            count_bound_impossible=sum(r["projection_count_bound_impossible"] for r in rows),
            local_valence_bound_impossible=sum(r["projection_local_valence_bound_impossible_atoms"]>0 for r in rows),
            either_bound_impossible=len(forced),
            node_prob_near_one_molecule_average=A.stats([r["atom_probability_max_ge_0p999999_fraction"] for r in rows]),
            edge_prob_singleton_molecule_average=A.stats([r["bond_probability_singleton_fraction"] for r in rows]),
            node_argmax_matches_raw_molecules=sum(r["atom_argmax_diff_from_raw"]==0 for r in rows),
            edge_argmax_matches_raw_molecules=sum(r["bond_argmax_diff_from_raw"]==0 for r in rows),
            molecules_with_ignored_fake_bonds=sum(r["ignored_bonds_to_fake"]>0 for r in rows),
            note="Existing projection only. Positive-probability support and fixed requested count impose necessary feasibility bounds. No MILP rerun and no projection XTB."),
        graph_groups={k:graph_summary(v) for k,v in dict(train=train,raw_valid=valid,xtb_success=success,target_hits=near).items()},
        conditional_reference=dict(ess=float(diagnostics.effective_sample_size[0]),bandwidth=.15,nearest_k=128,
            metrics={key:dict(unweighted=weighted(key,np.ones(len(train))),kernel_weighted=weighted(key,weights)) for key in
                ("heavy_atoms","ring_count","ring_atom_fraction","nonring_atoms","q_reference","q_xtb","s1_reference","t1_reference","s1_xtb","t1_xtb")},
            kernel_weighted_XTB_hit_rate=float(sum(r["kernel_weight"] for r in train if abs(Decimal(dbcalc[r["id"]]["calculated_q_text"]))<=Decimal(".1"))),
            top10_weight_mass=float(np.sort(weights)[-10:].sum()),
            top_references=[{k:r[k] for k in ("id","split_index","kernel_weight","heavy_atoms","ring_count","q_reference","q_xtb","s1_xtb","t1_xtb")} for r in sorted(train,key=lambda r:-r["kernel_weight"])[:20]],
            note="This is the existing count-sampler reference weighting. It is not a CFG=0 control and is not an expected novel-molecule energy distribution."))
    # Convert numpy scalar counters to native numbers, not strings, in the audit.
    result=json.loads(json.dumps(result,default=lambda value:value.item()))
    A.json_file(output/"summary.json",result)
    A.write_csv(output/"endpoint_molecule_diagnostics_1000.csv",rows)
    A.write_csv(output/"endpoint_atom_valence_diagnostics.csv",atoms)
    A.write_csv(output/"train_conditional_reference_1264.csv",train)
    A.write_csv(output/"ring_composition.csv",[dict(source=source,generation_index=r.get("generation_index"),db_id=r.get("id"),**c)
                 for source,part in (("train",train),("generated_valid",valid)) for r in part for c in r["ring_composition"]])
    for p,d in hashes.items(): assert A.sha256(Path(p))==d
    status.update(phase="complete",finished_utc=A.stamp(),exit_code=0); A.json_file(output/"status.json",status)
    print(json.dumps({k:result[k] for k in ("valence","projection","conditional_reference")}),flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--worker",action="store_true"); p.add_argument("--output",type=Path,default=OUT)
    args=p.parse_args(); output=args.output.resolve()
    if not args.worker:
        output.mkdir(parents=True,exist_ok=False); shutil.copyfile(__file__,output/"analysis_script.py")
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONUNBUFFERED="1")
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker","--output",str(output)]
        with (output/"worker.log").open("xb") as f:
            worker=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
        A.json_file(output/"launch.json",dict(pid=worker.pid,detached=True,command=command)); print(worker.pid); return
    try: run(output)
    except BaseException:
        error=traceback.format_exc(); A.json_file(output/"status.json",dict(phase="failed",error=error)); print(error,flush=True); raise


if __name__=="__main__": main()
