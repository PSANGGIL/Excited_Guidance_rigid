#!/usr/bin/env python3
"""Audit saved Stage8 graphs and XTB artifacts; never rerun generation or chemistry.

All energy hit windows use Decimal and the pipeline's first-success-per-graph
representatives. Geometry/WBO screens are warnings, not identity/stability proofs.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from functools import lru_cache
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "runs/generation_stage8_hp1kg_best_e3911_est0_n10000_cfg1_20261023"
XTB = GEN / "xtb_stda_diffusion_coords_24x1"
DB = ROOT / "joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz"
OUT = ROOT / "analysis/hp1kg_e3911_n10000_structure_energy_20261023"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def json_file(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def stats(values):
    import numpy as np
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite statistic input")
    return dict(n=len(x), mean=float(x.mean()), std_ddof0=float(x.std()),
                min=float(x.min()), q05=float(np.quantile(x, .05)), q25=float(np.quantile(x, .25)),
                median=float(np.median(x)), q75=float(np.quantile(x, .75)),
                q95=float(np.quantile(x, .95)), max=float(x.max()))


@lru_cache(maxsize=16)
def endpoint_batch(start):
    import torch
    return torch.load(GEN / "raw_endpoints" / f"batch_{start:05d}.pt", map_location="cpu", weights_only=False)


@lru_cache(maxsize=1)
def endpoint_metadata():
    manifest = json.loads((GEN / "manifest.json").read_text())
    path = Path(manifest["data_manifest"]["directory"]) / "joint_atom_state_metadata.json"
    if sha256(path) != manifest["data_manifest"]["sha256"][path.name]:
        raise ValueError("Atom vocabulary changed since generation")
    return manifest["batch_size"], json.loads(path.read_text())["states"]


def original_endpoint_molecule(index):
    """Recover the exact sampled graph when SMILES reparsing loses information."""
    from rdkit import Chem
    batch_size, vocabulary = endpoint_metadata()
    saved = next(r for r in endpoint_batch(index // batch_size * batch_size) if r["generation_index"] == index)
    states = saved["nodes"]["s_1"].argmax(-1).tolist()
    if any(s > len(vocabulary) for s in states):
        raise ValueError("Unresolved state mask in valid original graph")
    keep = [i for i, s in enumerate(states) if s < len(vocabulary)]
    indices = {old: new for new, old in enumerate(keep)}
    rw = Chem.RWMol()
    for old in keep:
        state = vocabulary[states[old]]
        atom = Chem.Atom(state["symbol"])
        atom.SetFormalCharge(int(state["formal_charge"]))
        atom.SetNumExplicitHs(int(state["hydrogens"]))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
    src, dst = saved["edge_index"]
    bonds = saved["edges"]["e_1"].argmax(-1)
    for a, b, bond in zip(src.tolist(), dst.tolist(), bonds.tolist()):
        if a >= b or a not in indices or b not in indices or bond == 0:
            continue
        if bond >= len(types):
            raise ValueError("Unresolved bond mask in valid original graph")
        rw.AddBond(indices[a], indices[b], types[bond])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def add_graph_features(row):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    result = dict(row)
    mol = Chem.MolFromSmiles(row["smiles"])
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else None
    roundtrip_warning = canonical != row["smiles"]
    result.update(smiles_roundtrip_warning=roundtrip_warning, reparsed_canonical_smiles=canonical,
                  used_saved_endpoint_graph=False)
    if roundtrip_warning:
        if "generation_index" not in row:
            raise ValueError("Database SMILES round-trip discrepancy")
        mol = original_endpoint_molecule(row["generation_index"])
        if Chem.MolToSmiles(mol, isomericSmiles=False) != row["smiles"]:
            raise ValueError("Original saved endpoint differs from recorded graph")
        result["used_saved_endpoint_graph"] = True
    result["mol"] = mol
    sizes = row["ring_sizes"]
    scaffold_error = ""
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False) if scaffold.GetNumAtoms() else ""
    except (ValueError, RuntimeError) as exc:
        scaffold_smiles, scaffold_error = "", str(exc)
    result.update(ringless=not sizes, only56=bool(sizes) and all(size in (5, 6) for size in sizes),
                  small=any(size <= 4 for size in sizes), seven_plus=any(size >= 7 for size in sizes),
                  nine_plus=any(size >= 9 for size in sizes),
                  has_atomic_charge=any(a.GetFormalCharge() for a in mol.GetAtoms()),
                  opposite_formal_charges=any(a.GetFormalCharge() > 0 for a in mol.GetAtoms())
                      and any(a.GetFormalCharge() < 0 for a in mol.GetAtoms()),
                  carbon_cumulene=any(a.GetAtomicNum() == 6 and sum(
                      b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds()) >= 2 for a in mol.GetAtoms()),
                  scaffold=scaffold_smiles, scaffold_error=scaffold_error,
                  ring_basis_reparse_changed=sorted(sizes) != sorted(len(r) for r in mol.GetRingInfo().AtomRings()))
    return result


FLAGS = ("aromatic", "ringless", "only56", "small", "seven_plus", "nine_plus",
         "has_atomic_charge", "opposite_formal_charges", "carbon_cumulene",
         "topology_screen", "geometry_screen", "ring_basis_reparse_changed", "smiles_roundtrip_warning")


def describe(rows):
    rings = Counter(size for row in rows for size in row["ring_sizes"])
    elements = Counter(element for row in rows for element in row["elements"])
    result = dict(n=len(rows), flags={key: sum(bool(row.get(key)) for row in rows) for key in FLAGS},
                  bridgehead_molecules=sum(row.get("bridgeheads", 0) > 0 for row in rows),
                  heavy_atoms=stats([row["heavy_atoms"] for row in rows]),
                  ring_count=stats([len(row["ring_sizes"]) for row in rows]),
                  ring_sizes=dict(sorted(rings.items())),
                  fraction_5_6_rings=(rings[5] + rings[6]) / max(1, sum(rings.values())),
                  element_molecule_counts=dict(elements), unique_graphs=len({r["smiles"] for r in rows}),
                  unique_nonempty_scaffolds=len({r["scaffold"] for r in rows if r["scaffold"]}))
    energy = [r for r in rows if "est_exact" in r]
    if energy:
        result.update(energy_n=len(energy), s1=stats([r["s1"] for r in energy]),
                      t1=stats([r["t1"] for r in energy]), est=stats([r["est"] for r in energy]),
                      abs_est=stats([abs(r["est"]) for r in energy]),
                      negative=sum(r["est_exact"] < 0 for r in energy),
                      exact_zero=sum(r["est_exact"] == 0 for r in energy),
                      inclusive_hits={s: sum(abs(r["est_exact"]) <= Decimal(s) for r in energy)
                                      for s in ("0.01", "0.05", "0.1", "0.25", "0.5")})
    oscillator = [r for r in rows if "s1_f" in r]
    if oscillator:
        result.update(s1_f=stats([r["s1_f"] for r in oscillator]),
                      f_zero=sum(r["s1_f"] == 0 for r in oscillator),
                      f_lt_0p001=sum(r["s1_f"] < .001 for r in oscillator),
                      f_ge_0p01=sum(r["s1_f"] >= .01 for r in oscillator),
                      s1_below_1=sum(r["s1"] < 1 for r in oscillator),
                      s1_between_1p5_4=sum(1.5 <= r["s1"] <= 4 for r in oscillator))
    if rows and "unsupported_atoms" in rows[0]:
        result["local_support"] = dict(
            all_atoms_and_bonds_supported=sum(r["unsupported_atoms"] == r["unsupported_bonds"] == 0 for r in rows),
            all_atoms_supported=sum(r["unsupported_atoms"] == 0 for r in rows),
            all_bonds_supported=sum(r["unsupported_bonds"] == 0 for r in rows),
            nearest_train_tanimoto=stats([r["nearest_train_tanimoto"] for r in rows]))
    audit_rows = [r for r in rows if "geometry_audit_error" in r]
    if audit_rows:
        good = [r for r in audit_rows if not r["geometry_audit_error"]]
        result["optimized_geometry"] = dict(
            audited=len(good), audit_errors=len(audit_rows) - len(good),
            not_audited=len(rows) - len(audit_rows),
            converged=sum(r["optimization_converged"] for r in good),
            correct_singlet_triplet_flags=sum(r["singlet_triplet_flags_verified"] for r in good),
            energy_log_match=sum(r["energy_log_match"] for r in good),
            heavy_rmsd_A=stats([r["heavy_rmsd_A"] for r in good]),
            heavy_rmsd_above_1A=sum(r["heavy_rmsd_A"] > 1 for r in good),
            weak_old_heavy_bonds=sum(r["weak_heavy_bonds"] > 0 for r in good),
            new_heavy_contacts=sum(r["new_heavy_contacts"] > 0 for r in good),
            hydrogen_transfer=sum(r["hydrogen_transfer"] > 0 for r in good),
            any_wbo_change_warning=sum(r["wbo_change_warning"] for r in good),
            postopt_bond_stretch_warning=sum(r["postopt_max_bond_ratio"] > 1.4 for r in good),
            postopt_aromatic_plane_warning=sum(r["postopt_aromatic_plane_rms_A"] > .15 for r in good),
            postopt_identity_warning_free=sum(r["optimization_converged"] and not r["wbo_change_warning"] for r in good))
    return result


def parse_states(text):
    found, result = False, []
    pattern = re.compile(r"^\s*(\d+)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\b")
    for line in text.splitlines():
        if " ".join(line.split()).startswith("state eV nm fL"):
            found = True
            continue
        if not found:
            continue
        match = pattern.match(line)
        if not match:
            if result:
                break
            continue
        if int(match.group(1)) > 10:
            break
        result.append((int(match.group(1)), Decimal(match.group(2))))
    return result


def geometry_audit(row):
    import numpy as np
    from rdkit import Chem
    index = row["generation_index"]
    result = dict(generation_index=index, geometry_audit_error="")
    try:
        work = XTB / "work" / f"generated_{index:06d}"
        inp = Chem.MolFromMolFile(str(XTB / "inputs" / f"generated_{index:06d}.sdf"), removeHs=False, sanitize=True)
        opt = Chem.MolFromMolFile(str(work / "xtbopt.sdf"), removeHs=False, sanitize=False)
        if inp is None or opt is None:
            raise ValueError("Cannot read input/optimized SDF")
        if [a.GetAtomicNum() for a in inp.GetAtoms()] != [a.GetAtomicNum() for a in opt.GetAtoms()]:
            raise ValueError("Atom count/element order changed")
        heavy = [a.GetIdx() for a in inp.GetAtoms() if a.GetAtomicNum() > 1]
        heavy_set = set(heavy)
        x, y = inp.GetConformer().GetPositions()[heavy], opt.GetConformer().GetPositions()[heavy]
        xc, yc = x - x.mean(0), y - y.mean(0)
        u, _, vt = np.linalg.svd(xc.T @ yc)
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        displacement = xc @ (u @ correction @ vt) - yc
        edges = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in inp.GetBonds()
                 if b.GetBeginAtomIdx() in heavy_set and b.GetEndAtomIdx() in heavy_set}
        wbo = {}
        for line in (work / "wbo").read_text().splitlines():
            a, b, value = line.split()
            wbo[tuple(sorted((int(a) - 1, int(b) - 1)))] = float(value)
        weak = [p for p in edges if wbo.get(p, 0.0) < .3]
        new = [p for p, v in wbo.items() if v > .5 and p not in edges and set(p) <= heavy_set]
        transfers = []
        for atom in inp.GetAtoms():
            if atom.GetAtomicNum() != 1 or atom.GetDegree() != 1:
                continue
            pair = tuple(sorted((atom.GetIdx(), atom.GetNeighbors()[0].GetIdx())))
            if wbo.get(pair, 0.0) < .3 and any(atom.GetIdx() in p and p != pair and v > .5 for p, v in wbo.items()):
                transfers.append(atom.GetIdx())
        pos = opt.GetConformer().GetPositions()
        periodic = Chem.GetPeriodicTable()
        ratios = [np.linalg.norm(pos[a] - pos[b]) / (
            periodic.GetRcovalent(inp.GetAtomWithIdx(a).GetAtomicNum()) +
            periodic.GetRcovalent(inp.GetAtomWithIdx(b).GetAtomicNum())) for a, b in edges]
        planes = []
        for ring in inp.GetRingInfo().AtomRings():
            if all(inp.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                centered = pos[list(ring)] - pos[list(ring)].mean(0)
                _, _, normals = np.linalg.svd(centered, full_matrices=False)
                planes.append(float(np.sqrt(np.mean((centered @ normals[-1]) ** 2))))
        opt_log = (work / "xtb_opt.log").read_text(errors="replace")
        convergence = re.search(r"GEOMETRY OPTIMIZATION CONVERGED AFTER\s+(\d+)\s+ITERATIONS", opt_log)
        singlet = (work / "stda_singlet.log").read_text(errors="replace")
        triplet = (work / "stda_triplet.log").read_text(errors="replace")
        ss, ts = parse_states(singlet), parse_states(triplet)
        energy_match = bool(ss and ts and ss[0][1] == Decimal(row["s1_text"]) and ts[0][1] == Decimal(row["t1_text"]))
        result.update(
            optimization_converged=bool(convergence),
            optimization_iterations=int(convergence.group(1)) if convergence else None,
            singlet_triplet_flags_verified=bool(re.search(r"triplet\s*:\s*F", singlet) and re.search(r"triplet\s*:\s*T", triplet)),
            energy_log_match=energy_match, singlet_states=len(ss), triplet_states=len(ts),
            identical_first10_energies=bool(len(ss) == len(ts) == 10 and [v for _, v in ss] == [v for _, v in ts]),
            heavy_rmsd_A=float(np.sqrt(np.mean(np.sum(displacement ** 2, axis=1)))),
            max_heavy_displacement_A=float(np.linalg.norm(displacement, axis=1).max()),
            weak_heavy_bonds=len(weak), new_heavy_contacts=len(new), hydrogen_transfer=len(transfers),
            weak_heavy_pairs=weak, new_heavy_pairs=new, hydrogen_transfer_indices=transfers,
            wbo_change_warning=bool(weak or new or transfers),
            postopt_max_bond_ratio=float(max(ratios, default=0)),
            postopt_aromatic_plane_rms_A=max(planes, default=0.0))
    except Exception as exc:
        result["geometry_audit_error"] = f"{type(exc).__name__}: {exc}"
    return result


def run(output):
    import numpy as np
    import torch
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    sys.path.insert(0, str(ROOT / "MolGuidance_guidance_selector_e987/tools"))
    from analyze_structure_xtb_detail import atom_signature, bond_signature
    sources = [GEN / "manifest.json", GEN / "raw_analysis/records.json", GEN / "db_graph_reference.json",
               XTB / "results.csv", XTB / "rdkit_2d_novel_unique_spaced/molecules.csv",
               *[DB / f"{s}.pt" for s in ("train", "val", "test")]]
    hashes = {str(p): sha256(p) for p in sources}
    state = dict(phase="graph_analysis", pid=os.getpid(), started_utc=stamp())
    json_file(output / "status.json", state)
    raw = json.loads((GEN / "raw_analysis/records.json").read_text())
    if len(raw) != 10000 or len({r["generation_index"] for r in raw}) != 10000:
        raise ValueError("Invalid full-generation ledger")
    with (XTB / "results.csv").open() as handle:
        calculations = {int(r["generation_index"]): r for r in csv.DictReader(handle)}
    db_ref = json.loads((GEN / "db_graph_reference.json").read_text())["records"]
    db_map = {(r["split"], r["split_index"]): r for r in db_ref}
    database, train = [], []
    for split in ("train", "val", "test"):
        for i, source in enumerate(torch.load(DB / f"{split}.pt", map_location="cpu", weights_only=False)):
            row = add_graph_features(db_map[(split, i)])
            if Chem.MolToSmiles(Chem.MolFromSmiles(source["smiles"]), isomericSmiles=False) != row["smiles"]:
                raise ValueError("Database index mapping mismatch")
            gap = Decimal(str(source["s1"])) - Decimal(str(source["t1"]))
            row.update(id=source["id"], s1=source["s1"], t1=source["t1"], est=float(gap), est_exact=gap)
            database.append(row)
            if split == "train":
                train.append(row)
    reference = {r["smiles"] for r in database}
    atom_support = {atom_signature(a) for r in train for a in r["mol"].GetAtoms()}
    bond_support = {bond_signature(b) for r in train for b in r["mol"].GetBonds()}
    fingerprinter = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [fingerprinter.GetFingerprint(r["mol"]) for r in train]
    valid, calculated = [], []
    for original in raw:
        if not original["raw_valid"]:
            continue
        row = add_graph_features(original)
        if row["db_member"] != (row["smiles"] in reference):
            raise ValueError("Novelty mismatch")
        mol = row["mol"]
        row["unsupported_atoms"] = sum(atom_signature(a) not in atom_support for a in mol.GetAtoms())
        row["unsupported_bonds"] = sum(bond_signature(b) not in bond_support for b in mol.GetBonds())
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprinter.GetFingerprint(mol), fingerprints)
        row["nearest_train_tanimoto"] = max(similarities)
        row["nearest_train_id"] = train[int(np.argmax(similarities))]["id"]
        result = calculations.get(row["generation_index"])
        if result and result["status"] == "ok":
            s1, t1 = Decimal(result["s1_energy_eV"]), Decimal(result["t1_energy_eV"])
            if s1 - t1 != Decimal(result["calculated_est_eV"]) or not row["eligible"]:
                raise ValueError("Calculation target/eligibility mismatch")
            row.update(s1=float(s1), t1=float(t1), est=float(s1-t1), est_exact=s1-t1,
                       s1_text=str(s1), t1_text=str(t1), s1_f=float(result["s1_f_osc"]))
            calculated.append(row)
        valid.append(row)
    unique, seen = [], set()
    for row in sorted(calculated, key=lambda r: r["generation_index"]):
        identity = row["reparsed_canonical_smiles"] or row["smiles"]
        if not row["db_member"] and identity not in seen:
            unique.append(row)
            seen.add(identity)
    with (XTB / "rdkit_2d_novel_unique_spaced/molecules.csv").open() as handle:
        published = {int(r["generation_index"]) for r in csv.DictReader(handle)}
    if {r["generation_index"] for r in unique} != published:
        raise ValueError("Unique representative selection changed")
    near = [r for r in unique if abs(r["est_exact"]) <= Decimal(".1")]
    groups = dict(train_db=train, full_db=database,
                  train_db_near_zero=[r for r in train if abs(r["est_exact"]) <= Decimal(".1")],
                  raw_valid=valid, raw_eligible=[r for r in valid if r["eligible"]],
                  calculation_success=calculated, novel_unique_success=unique,
                  near_zero_novel_unique=near,
                  exact_zero_novel_unique=[r for r in unique if r["est_exact"] == 0],
                  negative_novel_unique=[r for r in unique if r["est_exact"] < 0])
    preliminary = {k: describe(v) for k, v in groups.items()}
    json_file(output / "preliminary_statistics.json", preliminary)
    state.update(phase="geometry_audit", graph_records=len(valid), calculations=len(calculated), completed=0)
    json_file(output / "status.json", state)
    by_index = {r["generation_index"]: r for r in calculated}
    audits = []
    with ThreadPoolExecutor(max_workers=4) as pool, (output / "geometry_progress.jsonl").open("x") as handle:
        pending = [pool.submit(geometry_audit, row) for row in calculated]
        for future in as_completed(pending):
            audit = future.result()
            audits.append(audit)
            by_index[audit["generation_index"]].update(audit)
            handle.write(json.dumps(audit, allow_nan=False) + "\n")
            if len(audits) % 200 == 0:
                handle.flush()
                state.update(completed=len(audits), updated_utc=stamp())
                json_file(output / "status.json", state)
                print(json.dumps(state), flush=True)
    warnings_free = [r for r in near if not r["geometry_audit_error"] and r["optimization_converged"] and not r["wbo_change_warning"]]
    screened = [r for r in warnings_free if r["aromatic"] and r["only56"] and not r["has_atomic_charge"]]
    groups.update(near_zero_no_wbo_warning=warnings_free, near_zero_aromatic_only56_uncharged_no_wbo_warning=screened,
                  near_zero_aromatic_only56_uncharged_no_wbo_warning_f_ge_0p01=[r for r in screened if r["s1_f"] >= .01])
    for flag in ("aromatic", "ringless", "only56", "small", "seven_plus", "has_atomic_charge"):
        groups["unique_"+flag] = [r for r in unique if r[flag]]
        groups["unique_not_"+flag] = [r for r in unique if not r[flag]]
    for element in ("Se", "P", "S", "B", "Br"):
        groups["unique_contains_"+element] = [r for r in unique if element in r["elements"]]
    groups["near_zero_no_Se_no_P"] = [r for r in near if not (set(r["elements"]) & {"Se", "P"})]
    strata = []
    for low, high in ((1, 15), (16, 25), (26, 35), (36, 100)):
        part = [r for r in raw if low <= r["requested_heavy_atoms"] <= high]
        strata.append(dict(requested_heavy_atom_range=[low, high], n=len(part),
                           raw_valid=sum(r["raw_valid"] for r in part), eligible=sum(r["eligible"] for r in part),
                           calculated=sum(r["generation_index"] in by_index for r in part),
                           target_hits=sum(r["generation_index"] in by_index and abs(by_index[r["generation_index"]]["est_exact"]) <= Decimal(".1") for r in part)))
    near_any = {r["reparsed_canonical_smiles"] or r["smiles"] for r in calculated
                if not r["db_member"] and abs(r["est_exact"]) <= Decimal(".1")}
    failure_details = []
    for index, result in calculations.items():
        if result["status"] == "ok":
            continue
        item = dict(generation_index=index, status=result["status"], error=result.get("error"),
                    smiles=raw[index]["smiles"])
        if result["status"] == "stda_parse_failed":
            work = XTB / "work" / f"generated_{index:06d}"
            item["no_CSF_within_10eV_singlet_and_triplet"] = all(
                "no CSF, increase energy threshold" in (work / f"stda_{state_name}.log").read_text(errors="replace")
                for state_name in ("singlet", "triplet"))
        failure_details.append(item)
    summary = dict(created_utc=stamp(), generation=str(GEN), calculation=str(XTB),
                   source_sha256=hashes, groups={k: describe(v) for k, v in groups.items()},
                   requested_size_strata=strata, failure_details=failure_details,
                   invalid_error_groups=dict(Counter(re.sub(r"atom # \d+", "atom # INDEX", r["error"] or "unknown") for r in raw if not r["raw_valid"])),
                   count_control=dict(requested=stats([r["requested_heavy_atoms"] for r in raw]),
                       sampled=stats([r["sampled_real_atoms"] for r in raw]),
                       exact=sum(r["sampled_real_atoms"] == r["requested_heavy_atoms"] for r in raw)),
                   novelty_selection=dict(primary="First successful generation per novel canonical nonstereo graph, matching published PNG/CSV",
                       near_zero_unique_first_success=len(near), near_zero_unique_any_repeated_geometry=len(near_any)),
                   spectral_audit=dict(identical_full_first10_singlet_triplet_energies=sum(a.get("identical_first10_energies", False) for a in audits),
                       audit_errors=sum(bool(a["geometry_audit_error"]) for a in audits)),
                   smiles_roundtrip_warnings=[{k: r.get(k) for k in (
                       "generation_index", "smiles", "reparsed_canonical_smiles", "eligible", "used_saved_endpoint_graph")}
                       for r in valid if r["smiles_roundtrip_warning"]],
                   caveats=["No generation, XTB, or training was modified or rerun.",
                       "Raw graph molecules are analyzed; optional MAP-projected molecules are excluded.",
                       "DB uses NEVPT2, generated energies use uncalibrated XTB/sTDA; hit rates are method-specific.",
                       "RDKit ring sizes depend on the perceived cycle basis, not all possible cycles.",
                       "WBO thresholds 0.3 on original bonds and 0.5 on new contacts are heuristic, not an identity certificate.",
                       "Formal +/- charges include resonance representations such as nitro; they do not all establish zwitterions.",
                       "Local environments absent from training are not necessarily chemically impossible.",
                       "No Hessian, excited-state optimization, or experimental stability check was performed.",
                       "0.000 eV gaps and zero oscillator strengths refer to printed precision.",
                       "Subset screens are diagnostic selection, not a new model success metric."])
    for path, old in hashes.items():
        if sha256(Path(path)) != old:
            raise ValueError(f"Source changed during analysis: {path}")
    # Preserve all attempts, including invalid/skipped/failing calculations.
    enriched = {r["generation_index"]: r for r in valid}
    ledger = []
    for row in raw:
        record = {k: v for k, v in enriched.get(row["generation_index"], row).items() if k != "mol"}
        result = calculations.get(row["generation_index"])
        record["calculation_status"] = result["status"] if result else "not_attempted"
        ledger.append(record)
    write_csv(output / "all_10000_structure_energy.csv", ledger)
    write_csv(output / "geometry_audit_all_successes.csv", sorted(audits, key=lambda r: r["generation_index"]))
    for name, rows in (("near_zero_novel_unique", near), ("screened_aromatic_near_zero", screened)):
        write_csv(output / f"{name}.csv", [{k: v for k, v in r.items() if k != "mol"} for r in rows])
    json_file(output / "summary.json", summary)
    state.update(phase="complete", completed=len(audits), finished_utc=stamp(), exit_code=0)
    json_file(output / "status.json", state)
    print(json.dumps(dict(output=str(output), phase="complete", audits=len(audits), near_zero_unique=len(near),
                          near_zero_no_wbo_warning=len(warnings_free), screened_aromatic_near_zero=len(screened))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not args.worker:
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(__file__, output / "analysis_script.py")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--output", str(output), "--worker"]
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, close_fds=True, start_new_session=True)
        json_file(output / "launch.json", dict(pid=process.pid, detached=True, command=command, started_utc=stamp()))
        print(json.dumps(dict(pid=process.pid, output=str(output), detached=True)))
        return
    try:
        run(output)
    except BaseException:
        error = traceback.format_exc()
        json_file(output / "status.json", dict(phase="failed", error=error, finished_utc=stamp()))
        print(error, flush=True)
        raise


if __name__ == "__main__":
    main()
