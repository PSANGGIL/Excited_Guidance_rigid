#!/usr/bin/env python3
"""Read-only audit of e4690 S1-2T1 generation; saves separate report artifacts."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "runs/generation_stage8_hp1kg_best_e4690_s1_minus_2t1_0_n1000_cfg1_20261027"
XTB = GEN / "xtb_stda_diffusion_coords_24x1"
DB = ROOT / "joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz"
DB_CALC = ROOT / "runs/hp1kg_nevpt2_db_xtb_stda_all_20260919/results_with_fiplul_retry.csv"
HELPER = ROOT / "analysis/analyze_hp1kg_e3911_n10000_20261023.py"
OUT = ROOT / "analysis/hp1kg_e4690_s1_minus_2t1_n1000_report_20261027"


def load_helper():
    spec = importlib.util.spec_from_file_location("saved_geometry_audit", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.ROOT, helper.GEN, helper.XTB, helper.DB = ROOT, GEN, XTB, DB
    return helper


def wilson(k, n):
    if not n:
        return None
    z, p = 1.95996398454, k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * ((p * (1-p) / n + z*z / (4*n*n)) ** .5) / den
    return [center-half, center+half]


def serial(row):
    return {k: v for k, v in row.items() if k != "mol"}


def run(output):
    import numpy as np
    import torch
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from scipy.spatial.distance import jensenshannon
    from scipy.stats import spearmanr
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    audit = load_helper()
    sys.path.insert(0, str(ROOT / "MolGuidance_guidance_selector_e987/tools"))
    from analyze_structure_xtb_detail import atom_signature, bond_signature
    from run_stage8_generation_xtb import render_spaced
    files = [GEN/"manifest.json", GEN/"raw_analysis/records.json", GEN/"db_graph_reference.json",
             GEN/"atom_count_manifest.json", XTB/"results.csv", XTB/"all_1000_outcomes.csv",
             XTB/"rdkit_2d_novel_unique_spaced/molecules.csv", DB_CALC, HELPER,
             Path(__file__).resolve(), *[DB/f"{s}.pt" for s in ("train", "val", "test")]]
    hashes = {str(p): audit.sha256(p) for p in files}
    state = dict(phase="graph_analysis", started_utc=audit.stamp(), pid=os.getpid())
    audit.json_file(output/"status.json", state)
    raw = json.loads((GEN/"raw_analysis/records.json").read_text())
    calculations = {int(r["generation_index"]): r for r in csv.DictReader((XTB/"results.csv").open())}
    outcomes = {int(r["generation_index"]): r for r in csv.DictReader((XTB/"all_1000_outcomes.csv").open())}
    assert len(raw) == len(outcomes) == 1000
    assert {r["generation_index"] for r in raw} == set(range(1000))
    assert all(r["condition_name"] == "s1_minus_2t1" and r["target_value_eV"] == 0 for r in raw)
    db_ref = json.loads((GEN/"db_graph_reference.json").read_text())["records"]
    db_map = {(r["split"], r["split_index"]): r for r in db_ref}
    database = []
    for split in ("train", "val", "test"):
        sources = torch.load(DB/f"{split}.pt", map_location="cpu", weights_only=False)
        for i, source in enumerate(sources):
            row = audit.add_graph_features(db_map[(split, i)])
            assert Chem.MolToSmiles(Chem.MolFromSmiles(source["smiles"]), isomericSmiles=False) == row["smiles"]
            s, t = Decimal(str(source["s1"])), Decimal(str(source["t1"]))
            row.update(id=source["id"], s1=float(s), t1=float(t), q=float(s-2*t), q_exact=s-2*t,
                       est=float(s-t), energy_method="NEVPT2")
            database.append(row)
    train = [r for r in database if r["split"] == "train"]
    assert len(database) == 1581 and len(train) == 1264
    reference = {r["smiles"] for r in database}
    atom_support = {atom_signature(a) for r in train for a in r["mol"].GetAtoms()}
    bond_support = {bond_signature(b) for r in train for b in r["mol"].GetBonds()}
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [fpgen.GetFingerprint(r["mol"]) for r in train]
    valid, calculated = [], []
    for original in raw:
        if not original["raw_valid"]:
            continue
        row = audit.add_graph_features(original)
        assert row["db_member"] == (row["smiles"] in reference)
        row["unsupported_atoms"] = sum(atom_signature(a) not in atom_support for a in row["mol"].GetAtoms())
        row["unsupported_bonds"] = sum(bond_signature(b) not in bond_support for b in row["mol"].GetBonds())
        similarities = DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(row["mol"]), fps)
        row["nearest_train_tanimoto"] = max(similarities)
        row["nearest_train_id"] = train[int(np.argmax(similarities))]["id"]
        result = calculations.get(row["generation_index"])
        if result and result["status"] == "ok":
            s, t = Decimal(result["s1_energy_eV"]), Decimal(result["t1_energy_eV"])
            q = s - 2*t
            assert q == Decimal(result["calculated_condition_eV"]) == Decimal(result["calculated_s1_minus_2t1_eV"])
            assert abs(q) == Decimal(result["abs_condition_error_eV"])
            assert s-t == Decimal(result["calculated_est_eV"]) and row["eligible"]
            assert result["condition_name"] == "s1_minus_2t1"
            row.update(s1=float(s), t1=float(t), q=float(q), q_exact=q, abs_q=float(abs(q)),
                       est=float(s-t), s1_text=str(s), t1_text=str(t), s1_f=float(result["s1_f_osc"]),
                       energy_method="XTB/sTDA")
            calculated.append(row)
        valid.append(row)
    assert len(valid) == 574 and len(calculated) == 452
    unique, seen = [], set()
    for row in sorted(calculated, key=lambda r:r["generation_index"]):
        identity = row["reparsed_canonical_smiles"] or row["smiles"]
        if not row["db_member"] and identity not in seen:
            unique.append(row)
            seen.add(identity)
    published = {int(r["generation_index"]) for r in csv.DictReader((XTB/"rdkit_2d_novel_unique_spaced/molecules.csv").open())}
    assert published == {r["generation_index"] for r in unique}
    near = [r for r in unique if abs(r["q_exact"]) <= Decimal(".1")]
    assert len(near) == 34
    state.update(phase="geometry_audit", completed=0, total=len(calculated))
    audit.json_file(output/"status.json", state)
    indexed = {r["generation_index"]:r for r in calculated}
    audits = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(audit.geometry_audit, r) for r in calculated]):
            result = future.result()
            audits.append(result)
            indexed[result["generation_index"]].update(result)
            if len(audits) % 50 == 0:
                state.update(completed=len(audits), updated_utc=audit.stamp())
                audit.json_file(output/"status.json", state)
    warning_free = [r for r in unique if not r["geometry_audit_error"] and r["optimization_converged"] and not r["wbo_change_warning"]]
    clean56 = [r for r in warning_free if r["aromatic"] and r["only56"] and not r["has_atomic_charge"]]
    conservative = [r for r in unique if not r["geometry_audit_error"] and r["optimization_converged"]
                    and not r["wbo_change_warning"] and r["topology_screen"] and r["geometry_screen"]]
    groups = dict(train_db=train, full_db=database, train_db_q_abs_le_0p1=[r for r in train if abs(r["q_exact"]) <= Decimal(".1")],
                  train_db_q_abs_le_0p5=[r for r in train if abs(r["q_exact"]) <= Decimal(".5")],
                  raw_valid=valid, calculation_success=calculated, novel_unique_success=unique,
                  target_hits=near, wbo_warning_free=warning_free,
                  target_hits_wbo_warning_free=[r for r in warning_free if abs(r["q_exact"]) <= Decimal(".1")],
                  conservative_screen=conservative,
                  target_hits_conservative_screen=[r for r in conservative if abs(r["q_exact"]) <= Decimal(".1")],
                  aromatic_only56_uncharged_wbo_clear=clean56,
                  target_hits_aromatic_only56_uncharged_wbo_clear=[r for r in clean56 if abs(r["q_exact"]) <= Decimal(".1")])
    for flag in ("aromatic", "only56", "small", "seven_plus", "nine_plus", "has_atomic_charge"):
        groups["success_"+flag] = [r for r in unique if r[flag]]
        groups["success_not_"+flag] = [r for r in unique if not r[flag]]

    def describe(rows):
        result = audit.describe(rows)
        e = [r for r in rows if "q_exact" in r]
        if e:
            result["energy"] = dict(s1=audit.stats([r["s1"] for r in e]), t1=audit.stats([r["t1"] for r in e]),
                q=audit.stats([r["q"] for r in e]), abs_q=audit.stats([abs(r["q"]) for r in e]),
                est=audit.stats([r["est"] for r in e]), negative=sum(r["q_exact"]<0 for r in e),
                exact_zero=sum(r["q_exact"]==0 for r in e),
                hits={s:sum(abs(r["q_exact"])<=Decimal(s) for r in e) for s in ("0.01","0.05","0.1","0.25","0.5","1.0")})
        return result

    # Separate baseline: same excitation method, different initial-geometry workflow.
    db_calc = []
    db_ids = {str(r["id"]):r for r in database}
    for original in csv.DictReader(DB_CALC.open()):
        if not original["status"].startswith("ok"):
            continue
        row = db_ids[original["database_id"]]
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(original["smiles"]), isomericSmiles=False)
        assert canonical == row["smiles"]
        assert Decimal(original["reference_s1_eV"]) == Decimal(str(row["s1"]))
        assert Decimal(original["reference_t1_eV"]) == Decimal(str(row["t1"]))
        s, t = Decimal(original["calculated_s1_eV"]), Decimal(original["calculated_t1_eV"])
        db_calc.append(dict(database_id=row["id"], split=row["split"], heavy_atoms=row["heavy_atoms"],
                            reference_q=row["q"], reference_q_text=str(row["q_exact"]), calculated_q=float(s-2*t),
                            calculated_q_text=str(s-2*t), q_error=float(s-2*t-row["q_exact"]),
                            s1=float(s), t1=float(t), original_status=original["status"]))
    assert len(db_calc) == len(database)
    def calibration(rows):
        ref=np.array([r["reference_q"] for r in rows]); pred=np.array([r["calculated_q"] for r in rows])
        delta=pred-ref
        return dict(n=len(rows), reference=audit.stats(ref), calculated=audit.stats(pred),
                    bias=float(delta.mean()), mae=float(np.abs(delta).mean()), rmse=float(np.sqrt(np.mean(delta**2))),
                    pearson=float(np.corrcoef(ref,pred)[0,1]) if len(rows)>1 else None,
                    xtb_hits_0p1=sum(abs(Decimal(r["calculated_q_text"]))<=Decimal(".1") for r in rows))
    baseline = dict(full_db=calibration(db_calc), train=calibration([r for r in db_calc if r["split"]=="train"]),
                    reference_near_zero=calibration([r for r in db_calc if abs(Decimal(r["reference_q_text"]))<=Decimal(".1")]),
                    reference_near_zero_train=calibration([r for r in db_calc if r["split"]=="train" and abs(Decimal(r["reference_q_text"]))<=Decimal(".1")]),
                    calibration_applied=False,
                    note="Historical ETKDG/MMFF-or-UFF/XTB initial conformers; current generation uses diffusion coordinates. Not a matched geometry control.")

    # Exact actual-heavy-atom-count reweighting separates a size-composition effect.
    generated_counts=Counter(r["heavy_atoms"] for r in unique); training_counts=Counter(r["heavy_atoms"] for r in train)
    weight=np.array([generated_counts[r["heavy_atoms"]]/training_counts[r["heavy_atoms"]] for r in train], dtype=float)
    matched_n=sum(n for c,n in generated_counts.items() if c in training_counts)
    assert abs(weight.sum()-matched_n)<1e-8
    def ring_vector(rows,weights=None):
        counts=np.zeros(7)
        for row,w in zip(rows,weights if weights is not None else np.ones(len(rows))):
            for size in row["ring_sizes"]:
                counts[min(max(size,3),9)-3]+=w
        return counts/counts.sum()
    train_rings=ring_vector(train); gen_rings=ring_vector(unique); matched_rings=ring_vector(train,weight)
    matched=dict(matched_generated=matched_n, unmatched_generated=len(unique)-matched_n,
        flags={key:float(np.dot(weight,[bool(r[key]) for r in train])/weight.sum())
               for key in ("aromatic","only56","small","seven_plus","nine_plus","has_atomic_charge")},
        mean_rings=float(np.dot(weight,[len(r["ring_sizes"]) for r in train])/weight.sum()),
        ring_size_categories=["3","4","5","6","7","8","9+"],
        train_ring_fractions=train_rings.tolist(), generated_ring_fractions=gen_rings.tolist(),
        count_matched_train_ring_fractions=matched_rings.tolist(),
        ring_size_jsd_bits=float(jensenshannon(train_rings,gen_rings,base=2)**2),
        count_matched_ring_size_jsd_bits=float(jensenshannon(matched_rings,gen_rings,base=2)**2))
    strata=[]
    for lo,hi in ((1,15),(16,25),(26,35),(36,45),(46,100)):
        rows=[r for r in raw if lo<=r["requested_heavy_atoms"]<=hi]
        success=[indexed[r["generation_index"]] for r in rows if r["generation_index"] in indexed]
        strata.append(dict(requested_atom_range=[lo,hi], n=len(rows), valid=sum(r["raw_valid"] for r in rows),
                           eligible=sum(r["eligible"] for r in rows), calculated=len(success),
                           target_hits=sum(abs(r["q_exact"])<=Decimal(".1") for r in success),
                           mean_q=float(np.mean([r["q"] for r in success])) if success else None,
                           wbo_warnings=sum(r.get("wbo_change_warning",False) for r in success)))
    feature_correlations={}
    for name,values in (("heavy_atoms",[r["heavy_atoms"] for r in unique]),("ring_count",[len(r["ring_sizes"]) for r in unique]),
                        ("nearest_train_tanimoto",[r["nearest_train_tanimoto"] for r in unique])):
        c=spearmanr(values,[r["abs_q"] for r in unique])
        feature_correlations[name]=dict(spearman_rho_abs_error=float(c.statistic), pvalue=float(c.pvalue))
    invalid=Counter(re.sub(r"atom # \d+", "atom # INDEX",r["error"] or "unknown") for r in raw if not r["raw_valid"])
    summary=dict(created_utc=audit.stamp(), generation=str(GEN), calculation=str(XTB), condition="S1 - 2*T1",
        target_eV=0, checkpoint_epoch_zero_based=4690, source_sha256=hashes,
        groups={k:describe(v) for k,v in groups.items()}, count_matched_train=matched,
        db_recalculation_comparison=baseline, requested_size_strata=strata,
        condition_atom_count_sampler=json.loads((GEN/"atom_count_manifest.json").read_text()),
        count_control=dict(requested=audit.stats([r["requested_heavy_atoms"] for r in raw]),
            actual=audit.stats([r["sampled_real_atoms"] for r in raw]),
            exact=sum(r["sampled_real_atoms"]==r["requested_heavy_atoms"] for r in raw),
            actual_minus_requested=audit.stats([r["sampled_real_atoms"]-r["requested_heavy_atoms"] for r in raw])),
        invalid_error_groups=dict(invalid.most_common()), outcome_counts=dict(Counter(r["status"] for r in outcomes.values())),
        feature_correlations=feature_correlations,
        uncertainty=dict(hit_rate_of_success=len(near)/len(unique),wilson95_success=wilson(len(near),len(unique)),
            hit_rate_of_attempts=len(near)/len(raw),wilson95_attempts=wilson(len(near),len(raw)),
            note="Single-run binomial descriptive interval, not checkpoint/seed reproducibility uncertainty."),
        verification=dict(audit_errors=sum(bool(r["geometry_audit_error"]) for r in audits),
            converged=sum(r.get("optimization_converged",False) for r in audits),
            correct_spin_flags=sum(r.get("singlet_triplet_flags_verified",False) for r in audits),
            log_energy_matches=sum(r.get("energy_log_match",False) for r in audits),
            identical_first10_singlet_triplet=sum(r.get("identical_first10_energies",False) for r in audits),
            smiles_roundtrip_warnings=sum(r["smiles_roundtrip_warning"] for r in valid)),
        definitions=dict(target_hits="abs(Decimal(S1)-2*Decimal(T1)) <= 0.1 eV; energies as printed in logs",
            topology_screen="No RDKit perceived ring <=4 or >=9; 7 and 8 member rings remain allowed",
            geometry_screen="Original diffusion coordinates: bond/covalent-radii-sum between .65 and 1.4, nonbond ratio >=.65, aromatic-plane RMS <=.15 A",
            wbo_warning="Original heavy bond WBO<.3, new heavy contact WBO>.5, or old H bond WBO<.3 and new H bond WBO>.5",
            conservative_screen="Connected neutral closed-shell, original topology+geometry screens, optimization converged, no WBO warning",
            only56="All RDKit perceived rings 5 or 6 member; diagnostic restriction, not a chemical feasibility rule",
            novelty="Canonical nonstereo graph absent from this HP1KG train/val/test DB, not from all chemical databases",
            std="Population standard deviation (ddof=0)"))
    audit.json_file(output/"summary.json",summary)
    enriched={r["generation_index"]:r for r in valid}
    ledger=[]
    for row in raw:
        r=serial(enriched.get(row["generation_index"],row)); r["calculation_status"]=outcomes[row["generation_index"]]["status"]
        ledger.append(r)
    audit.write_csv(output/"all_1000_structure_energy.csv",ledger)
    audit.write_csv(output/"geometry_audit_452.csv",sorted(audits,key=lambda r:r["generation_index"]))
    audit.write_csv(output/"db_reference_and_xtb_s1_minus_2t1.csv",db_calc)
    for name in ("target_hits","target_hits_wbo_warning_free","target_hits_conservative_screen",
                 "target_hits_aromatic_only56_uncharged_wbo_clear"):
        rows=sorted(groups[name],key=lambda r:abs(r["q_exact"]))
        audit.write_csv(output/f"{name}.csv",[serial(r) for r in rows])
        if rows:
            depiction=[dict(generation_index=r["generation_index"],canonical_smiles=r["smiles"],
                            s1_energy_eV=r["s1_text"],t1_energy_eV=r["t1_text"]) for r in rows]
            for start in range(0,len(depiction),20):
                render_spaced(depiction[start:start+20],output/f"{name}_{start//20:02d}.png",
                    "HP1KG e4690 | "+name.replace("_"," "),"s1_minus_2t1",0.)
    make_plots(output,summary,unique,train,db_calc)
    for p,old in hashes.items():
        assert audit.sha256(Path(p))==old, "Analysis input changed: "+p
    state.update(phase="complete",finished_utc=audit.stamp(),exit_code=0,completed=len(audits))
    audit.json_file(output/"status.json",state)
    print(json.dumps(dict(output=str(output),verification=summary["verification"],
                         hit_groups={k:len(v) for k,v in groups.items() if k.startswith("target_hits")})),flush=True)


def make_plots(output,summary,rows,train,db_calc):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    q=np.array([r["q"] for r in rows]); s=np.array([r["s1"] for r in rows]); t=np.array([r["t1"] for r in rows])
    g=summary["groups"]
    with PdfPages(output/"analysis_figures.pdf") as pdf:
        fig,axs=plt.subplots(2,2,figsize=(14,10),layout="constrained")
        labels=["Attempts","RDKit valid","XTB/sTDA success","|S1-2T1| <= 0.1","Hit + WBO clear","Hit + conservative screen"]
        n=[1000,574,len(rows),g["target_hits"]["n"],g["target_hits_wbo_warning_free"]["n"],g["target_hits_conservative_screen"]["n"]]
        ax=axs[0,0]; ax.barh(labels,n,color=["#9ca3af","#527ca6","#527ca6","#cd8647","#4e9379","#4e9379"]); ax.invert_yaxis()
        for i,v in enumerate(n): ax.text(v+8,i,str(v),va="center")
        ax.set_xlim(0,1130); ax.set_xlabel("Molecules / 1,000 attempts"); ax.set_title("Success and target-hit funnel")
        ax=axs[0,1]; bins=np.arange(-5.25,1.51,.25); ax.hist(q,bins=bins,color="#527ca6",edgecolor="white")
        ax.axvspan(-.1,.1,color="#cd8647",alpha=.4,label="Target window +/-0.1 eV")
        ax.axvline(0,color="black",lw=1); ax.set_xlabel("S1 - 2*T1 (eV)"); ax.set_ylabel("Molecules"); ax.legend(fontsize=9)
        ax.set_title(f"452 XTB/sTDA results | mean {q.mean():.3f}, std {q.std():.3f}")
        ax=axs[1,0]; flags=["small","seven_plus","nine_plus","only56"]; names=["3/4-member","7+ member","9+ member","Only 5/6"]
        x=np.arange(4); w=.25
        for offset,key,label,color in [(-w,"train_db","Train DB","#9ca3af"),(0,"calculation_success","Generated","#527ca6"),(w,"target_hits","Target hits","#cd8647")]:
            ax.bar(x+offset,[100*g[key]["flags"][f]/max(1,g[key]["n"]) for f in flags],w,label=label,color=color)
        ax.set_xticks(x,names); ax.set_ylabel("Molecules (%)"); ax.set_ylim(0,105); ax.legend(fontsize=9)
        ax.set_title("Original graph topology (overlapping categories)")
        ax=axs[1,1]; warning=np.array([bool(r.get("wbo_change_warning")) for r in rows]); ax.scatter(t[~warning],s[~warning],s=16,alpha=.55,label="WBO clear",color="#527ca6")
        ax.scatter(t[warning],s[warning],s=28,marker="x",label="WBO warning",color="#cd8647")
        lim=max(float(t.max()),float(s.max())/2)+.2; line=np.linspace(0,lim,100); ax.plot(line,2*line,"k--",label="S1 = 2*T1")
        ax.set_xlim(0,float(t.max())+.2); ax.set_ylim(0,float(s.max())+.2); ax.set_xlabel("T1 (eV)"); ax.set_ylabel("S1 (eV)"); ax.legend(fontsize=9)
        ax.set_title("Target line and optimization-change warnings")
        fig.suptitle("HP1KG e4690 | target S1 - 2*T1 = 0 eV | CFG 1 | 1,000 attempts",fontsize=16)
        fig.savefig(output/"analysis_overview.png",dpi=180); pdf.savefig(fig); plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout="constrained")
        for ax,v,name in zip(axs,[s,t,q],["S1","T1","S1 - 2*T1"]):
            bins=np.arange(np.floor(v.min()/.25)*.25,np.ceil(v.max()/.25)*.25+.26,.25)
            ax.hist(v,bins=bins,color="#527ca6",edgecolor="white"); ax.set_xlabel(name+" (eV)"); ax.set_ylabel("Molecules")
            ax.set_title(f"{name}: {v.mean():.3f} +/- {v.std():.3f} eV")
        fig.suptitle("XTB/sTDA energy histograms | 452 successful novel unique graphs | bin width 0.25 eV")
        fig.savefig(output/"energy_histograms.png",dpi=180); pdf.savefig(fig); plt.close(fig)
        fig,axs=plt.subplots(1,2,figsize=(13,5),layout="constrained")
        x=np.arange(7); m=summary["count_matched_train"]
        for d,key,label,color in [(-.25,"train_ring_fractions","Train DB","#9ca3af"),(0,"count_matched_train_ring_fractions","Count-matched train","#4e9379"),(.25,"generated_ring_fractions","Generated success","#527ca6")]:
            axs[0].bar(x+d,100*np.array(m[key]),.25,label=label,color=color)
        axs[0].set_xticks(x,m["ring_size_categories"]); axs[0].set_xlabel("RDKit perceived ring size"); axs[0].set_ylabel("Rings (%)"); axs[0].legend(fontsize=9)
        bins=np.arange(-8,1.51,.5)
        axs[1].hist([r["q"] for r in train],bins=bins,density=True,histtype="step",lw=2,label="Train: NEVPT2")
        axs[1].hist([r["calculated_q"] for r in db_calc if r["split"]=="train"],bins=bins,density=True,histtype="step",lw=2,label="Train DB: historical XTB/sTDA")
        axs[1].hist(q,bins=bins,density=True,histtype="step",lw=2,label="Generated: XTB/sTDA")
        axs[1].axvline(0,color="black",ls="--"); axs[1].set_xlabel("S1 - 2*T1 (eV)"); axs[1].set_ylabel("Density"); axs[1].legend(fontsize=8)
        fig.suptitle("DB comparison | size reweighting for topology; historical quantum baseline is not a matched geometry control")
        fig.savefig(output/"database_comparison.png",dpi=180); pdf.savefig(fig); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT); parser.add_argument("--worker",action="store_true")
    args=parser.parse_args(); output=args.output.resolve(); helper=load_helper()
    if not args.worker:
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(__file__,output/"analysis_script.py")
        shutil.copyfile(HELPER,output/"geometry_audit_helper.py")
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--output",str(output),"--worker"]
        with (output/"worker.log").open("xb") as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                     close_fds=True,start_new_session=True)
        helper.json_file(output/"launch.json",dict(pid=process.pid,detached=True,command=command,started_utc=helper.stamp()))
        print(json.dumps(dict(pid=process.pid,output=str(output),detached=True))); return
    try:
        run(output)
    except BaseException:
        error=traceback.format_exc(); helper.json_file(output/"status.json",dict(phase="failed",error=error,finished_utc=helper.stamp()))
        print(error,flush=True); raise


if __name__=="__main__":
    main()
