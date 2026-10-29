"""Lossless STGG -> Stage8 heavy-atom/H-state adapter; originals are read-only.

S1 is the first singlet transition exported by the upstream sTDA parser:
1239.84193 / wavelength_nm. No T1/EST or oscillator-strength proxy is invented.
Coordinates are the existing RDKit conformers, NOT HP1KG ab-initio geometries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ.get("STGG_WORKSPACE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from molguidance_hp1kg_baseline.convert_hp1kg_joint_kekule import (
    index_ranges, marginal_distributions, rebuild_molecule, save_split, sha256,
    uniform_spanning_tree_edge_weights,
)

SPLITS = ("train", "val", "test")
_SOURCE = None
_FRAMES = {}
_RAW = None
_MAP = None


def read_atom_map(source):
    value = json.loads((Path(source) / "atom_map.json").read_text())
    if isinstance(value, dict):
        value = value["atom_map"]
    if not isinstance(value, list) or "C" not in value:
        raise ValueError("unsupported source atom_map schema")
    return [s for s in value if s != "H"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def init_worker(source, raw_csv, raw_rows):
    global _SOURCE, _RAW, _MAP
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    _SOURCE = Path(source)
    _RAW = pd.read_csv(raw_csv, nrows=raw_rows, usecols=["wavelength", "f_osc", "molecule"])
    _MAP = read_atom_map(_SOURCE)


def identity(mol):
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def convert_one(split, i):
    if split not in _FRAMES:
        _FRAMES[split] = torch.load(_SOURCE / f"{split}_data_processed.pt", map_location="cpu", weights_only=False)
    data = _FRAMES[split]
    row = int(data["source_rows"][i])
    source_smiles = data["smiles"][i]
    raw = _RAW.iloc[row]
    wavelength = float(raw.wavelength)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("missing/nonpositive wavelength")
    prop = 1239.84193 / wavelength
    expected = Chem.MolFromSmiles(source_smiles)
    original = Chem.MolFromSmiles(str(raw.molecule))
    if expected is None or original is None or identity(expected) != identity(original):
        raise ValueError("raw-CSV/processed-SMILES identity mismatch")
    ns, ne = map(int, data["node_idx_array"][i])
    es, ee = map(int, data["edge_idx_array"][i])
    xyz = data["positions"][ns:ne].numpy().copy()
    elements = data["atom_types"][ns:ne].float().argmax(-1).tolist()
    symbols = [data["atom_map"][k] for k in elements]
    charges = data["atom_charges"][ns:ne].numpy().astype(np.int32)
    edges = data["bond_idxs"][es:ee].numpy().astype(np.int32)
    orders = data["bond_types"][es:ee].numpy().astype(np.int32)
    if not np.isfinite(xyz).all() or xyz.shape != (ne - ns, 3):
        raise ValueError("invalid source coordinates")
    if len(edges) and (edges.min() < 0 or edges.max() >= len(symbols)):
        raise ValueError("bond index outside molecule")
    if not np.isin(orders, [1, 2, 3]).all():
        raise ValueError("source bond is not Kekule single/double/triple")
    if len({tuple(sorted(e)) for e in edges.tolist()}) != len(edges):
        raise ValueError("duplicate source edges")
    heavy = [j for j, s in enumerate(symbols) if s != "H"]
    if not heavy:
        raise ValueError("no heavy atoms")
    inverse = {j: k for k, j in enumerate(heavy)}
    h = np.zeros(len(heavy), dtype=np.int64)
    h_degree = Counter()
    bond = np.zeros((len(heavy), len(heavy)), dtype=np.int64)
    for (a, b), order in zip(edges, orders):
        if a == b:
            raise ValueError("self bond")
        if symbols[a] == "H" or symbols[b] == "H":
            if symbols[a] == symbols[b] == "H":
                raise ValueError("H-H component unsupported by attached-H codec")
            hi, atom = (a, b) if symbols[a] == "H" else (b, a)
            if order != 1 or charges[hi] != 0:
                raise ValueError("charged/multiple-bond hydrogen unsupported")
            h[inverse[atom]] += 1
            h_degree[int(hi)] += 1
        else:
            bond[inverse[a], inverse[b]] = bond[inverse[b], inverse[a]] = order
    if any(h_degree[j] != 1 for j, s in enumerate(symbols) if s == "H"):
        raise ValueError("hydrogen is not attached to exactly one heavy atom")
    type_idx = np.array([_MAP.index(symbols[j]) for j in heavy], dtype=np.int64)
    heavy_charges = charges[heavy]
    if heavy_charges.min() < -2 or heavy_charges.max() > 3:
        raise ValueError("charge outside existing model range")
    keys = list(zip(type_idx.tolist(), heavy_charges.tolist(), h.tolist()))
    local_metadata = {"states": [dict(symbol=_MAP[a], formal_charge=c, hydrogens=nh) for a, c, nh in keys]}
    restored = rebuild_molecule(torch.arange(len(keys)), torch.from_numpy(bond), local_metadata)
    if identity(restored) != identity(expected):
        raise ValueError("heavy+H-count codec identity mismatch")
    if any(a.GetNumRadicalElectrons() for a in restored.GetAtoms()):
        raise ValueError("radical not represented by current joint state")
    if len(Chem.GetMolFrags(restored)) != 1:
        raise ValueError("multifragment excluded consistently with current model")
    ij = np.stack(np.where(np.triu(bond, 1) > 0), axis=1).astype(np.int32)
    edge_types = bond[ij[:, 0], ij[:, 1]].astype(np.int32)
    weights = uniform_spanning_tree_edge_weights(torch.from_numpy(bond)).numpy()[ij[:, 0], ij[:, 1]]
    aromatic = np.array([restored.GetBondBetweenAtoms(int(a), int(b)).GetIsAromatic() for a, b in ij], dtype=bool)
    ring_sizes = [len(r) for r in restored.GetRingInfo().AtomRings()]
    return dict(split=split, source_split_index=i, source_row=row, smiles=identity(restored),
                positions=xyz[heavy].copy(), source_heavy_indices=heavy,
                atom_types=type_idx, atom_charges=heavy_charges, atom_hydrogens=h,
                keys=keys, heavy_valences=bond.sum(1).tolist(), bond_idxs=ij,
                bond_types=edge_types, bond_connectivity_weights=weights, bond_aromatic_flags=aromatic,
                s1=prop, f_osc=float(raw.f_osc), aromatic=bool(aromatic.any()), ring_sizes=ring_sizes)


def convert_chunk(task):
    split, indices = task
    records, rejected = [], []
    for i in indices:
        try:
            records.append(convert_one(split, i))
        except Exception as exc:
            rejected.append(dict(split=split, source_split_index=i, error=str(exc)))
    return records, rejected


def prepare(source, raw_csv, output, workers=8, limit=None):
    torch.set_num_threads(1)
    source, raw_csv, output = Path(source).resolve(), Path(raw_csv).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    split_sizes, raw_rows, tasks = {}, 0, []
    for split in SPLITS:
        data = torch.load(source / f"{split}_data_processed.pt", map_location="cpu", weights_only=False)
        split_sizes[split] = min(len(data["node_idx_array"]), limit or len(data["node_idx_array"]))
        raw_rows = max(raw_rows, max(map(int, data["source_rows"][:split_sizes[split]])) + 1)
        tasks += [(split, list(range(start, min(start + 200, split_sizes[split]))))
                  for start in range(0, split_sizes[split], 200)]
        del data
    records = {split: [] for split in SPLITS}
    rejected, completed = [], 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(str(source), str(raw_csv), raw_rows)) as pool:
        for accepted, errors in pool.map(convert_chunk, tasks):
            for item in accepted:
                records[item["split"]].append(item)
            rejected.extend(errors)
            completed += len(accepted) + len(errors)
            write_json(output / "status.json", dict(phase="converting", completed=completed,
                       total=sum(split_sizes.values()), rejected=len(rejected), seconds=time.monotonic() - started))
            if completed % 2000 == 0 or completed == sum(split_sizes.values()):
                print("CONVERT", completed, "rejected", len(rejected), flush=True)
    write_json(output / "rejected.json", rejected)
    if len(rejected) > .05 * sum(split_sizes.values()):
        raise ValueError("more than 5% rejected: inspect adapter/source rather than silently training")
    # Preserve training frequencies, but never leak a canonical training graph into held-out evaluation.
    seen, split_overlap_removed = set(), {}
    for split in SPLITS:
        keep = [r for r in records[split] if r["smiles"] not in seen]
        split_overlap_removed[split] = len(records[split]) - len(keep)
        records[split] = keep
        seen.update(r["smiles"] for r in keep)
        if not keep:
            raise ValueError(f"empty {split} split")
    atom_map = read_atom_map(source)
    state_sets = {s: {key for r in rows for key in r["keys"]} for s, rows in records.items()}
    union = sorted(set.union(*state_sets.values()))
    vocab_index = {state: i for i, state in enumerate(union)}
    valences = defaultdict(Counter)
    for r in records["train"]:
        for key, valence in zip(r["keys"], r["heavy_valences"]):
            valences[key][int(valence)] += 1
    metadata = dict(version=2, definition="(element, formal_charge, attached_hydrogen_count)",
                    states=[dict(atom_type_index=a, symbol=atom_map[a], formal_charge=c, hydrogens=h,
                                 train_heavy_valence_counts={str(v): n for v, n in sorted(valences[(a,c,h)].items())})
                            for a, c, h in union],
                    train_supported=[s in state_sets["train"] for s in union],
                    split_state_counts={s: len(v) for s, v in state_sets.items()})
    write_json(output / "joint_atom_state_metadata.json", metadata)
    write_json(output / "atom_map.json", atom_map)
    report = dict(source=str(source), raw_csv=str(raw_csv), property="s1", unit="eV",
                  energy_conversion="1239.84193 / wavelength_nm; upstream first singlet transition",
                  chemistry_representation="joint_atom_state+kekule_bond", atom_map=atom_map,
                  atom_state_count=len(union), train_supported_atom_state_count=sum(metadata["train_supported"]),
                  source_rows_requested=split_sizes, rejected=len(rejected),
                  rejected_reasons=dict(Counter(r["error"] for r in rejected)),
                  cross_split_identity_overlap_removed=split_overlap_removed,
                  splits={}, source_sha256={"raw_csv": sha256(raw_csv), "adapter": sha256(Path(__file__))},
                  geometry_source="existing STGG RDKit_ETKDGv3/force-field conformers; not ab-initio XYZ",
                  heavy_atom_coordinate_delta_max=0.0, source_atom_order_preserved=True,
                  comparison_limitations=["Different dataset, elements, molecule sizes, geometry source and S1 condition: NOT an isolated causal aromaticity ablation.",
                                          "First 100,000 previously preprocessed entries, not all 2.9 million entries.",
                                          "Codec covers declared split states; train_support and all empirical priors are train-only."])
    stats_rows = []
    for split, rows in records.items():
        def concat(key, dtype=None):
            result = torch.from_numpy(np.concatenate([r[key] for r in rows], axis=0))
            return result if dtype is None else result.to(dtype)
        states = torch.tensor([vocab_index[key] for r in rows for key in r["keys"]], dtype=torch.int64)
        node_types = concat("atom_types", torch.int64)
        data = dict(smiles=[r["smiles"] for r in rows], positions=concat("positions", torch.float32),
                    atom_types=torch.nn.functional.one_hot(node_types, len(atom_map)).bool(),
                    atom_charges=concat("atom_charges", torch.int32), atom_hydrogens=concat("atom_hydrogens", torch.int64),
                    atom_states=states, bond_idxs=concat("bond_idxs", torch.int32), bond_types=concat("bond_types", torch.int32),
                    bond_connectivity_weights=concat("bond_connectivity_weights", torch.float32),
                    bond_aromatic_flags=concat("bond_aromatic_flags", torch.bool),
                    node_idx_array=index_ranges([len(r["positions"]) for r in rows]),
                    edge_idx_array=index_ranges([len(r["bond_types"]) for r in rows]),
                    properties=torch.tensor([r["s1"] for r in rows], dtype=torch.float32), property_names="s1",
                    source_rows=[r["source_row"] for r in rows], source_split_indices=[r["source_split_index"] for r in rows],
                    source_heavy_indices=[r["source_heavy_indices"] for r in rows])
        save_split(output, split, data)
        if split == "train":
            torch.save(marginal_distributions(data, len(atom_map)), output / "train_data_marginal_dists.pt")
            state_counts = torch.bincount(states, minlength=len(union)).float()
            torch.save(state_counts / state_counts.sum(), output / "train_data_joint_state_marginal.pt")
        e = np.array([r["s1"] for r in rows])
        near = [r for r in rows if abs(r["s1"] - 4.0) <= .1]
        report["splits"][split] = dict(molecules=len(rows), oracle_identity_matches=len(rows), oracle_identity_fraction=1.,
            property_mean=float(e.mean()), property_std=float(e.std(ddof=1)), property_min=float(e.min()), property_max=float(e.max()),
            aromatic_molecules=sum(r["aromatic"] for r in rows),
            s1_3p9_to_4p1=dict(molecules=len(near), aromatic_molecules=sum(r["aromatic"] for r in near)),
            atom_count_min=min(len(r["positions"]) for r in rows), atom_count_max=max(len(r["positions"]) for r in rows))
        report["source_sha256"][split] = sha256(source / f"{split}_data_processed.pt")
        stats_rows += [dict(split=split, source_row=r["source_row"], smiles=r["smiles"], s1_eV=r["s1"],
                           aromatic=r["aromatic"], heavy_atoms=len(r["positions"]), ring_sizes=json.dumps(r["ring_sizes"])) for r in rows]
        del data
    with (output / "db_structure_properties.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)
    report["oracle_identity_total"] = report["oracle_identity_expected"] = len(stats_rows)
    report["complete"] = True
    report["conversion_seconds"] = time.monotonic() - started
    write_json(output / "conversion_report.json", report)
    write_json(output / "status.json", dict(phase="complete", accepted=len(stats_rows), rejected=len(rejected)))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.raw_csv, args.output, args.workers, args.limit_per_split), indent=2))
