"""Audit a completed panel without regenerating coordinates or changing models."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rdkit import Chem, RDLogger
from generate_stage8_panel import write_json


def topology(rows):
    sizes, counts = Counter(), Counter()
    valid, arom, small, medium, large = 0, 0, 0, 0, 0
    for row in rows:
        if not row["raw_valid"]:
            continue
        # Reparse one canonical non-isomeric SMILES order for every panel.
        # This avoids comparing arbitrary atom-order-dependent SSSR choices.
        mol = Chem.MolFromSmiles(row["smiles"])
        rings = [len(r) for r in mol.GetRingInfo().AtomRings()]
        valid += 1
        arom += any(a.GetIsAromatic() for a in mol.GetAtoms())
        small += any(s <= 4 for s in rings)
        medium += any(s >= 7 for s in rings)
        large += any(s >= 9 for s in rings)
        sizes.update(rings)
        counts.update([len(rings)])
    return dict(valid=valid, aromatic=arom, ring_sizes=dict(sorted(sizes.items())),
                ring_counts=dict(sorted(counts.items())), small_ring_molecules=small,
                seven_plus_molecules=medium, nine_plus_molecules=large,
                five_six_fraction=(sizes[5] + sizes[6]) / max(1, sum(sizes.values())))


def jsd(left, right):
    keys = sorted(set(left) | set(right), key=int)
    p = np.array([left.get(k, 0) for k in keys], dtype=float)
    q = np.array([right.get(k, 0) for k in keys], dtype=float)
    if not p.sum() or not q.sum():
        return None
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(.5 * np.sum(p[p > 0] * np.log2(p[p > 0] / m[p > 0])) +
                 .5 * np.sum(q[q > 0] * np.log2(q[q > 0] / m[q > 0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "audit.json").exists():
        raise FileExistsError(run / "audit.json")
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.error")
    summary = json.loads((run / "summary.json").read_text())
    n = summary["manifest"]["requested"]
    raw = json.loads((run / "raw_analysis/records.json").read_text())
    joint = json.loads((run / "joint_analysis/records.json").read_text())
    previous = json.loads((run / "previous_same_metrics.json").read_text())["records"]
    db = json.loads((run / "db_coordinate_reference.json").read_text())["records"]
    assert len(raw) == len(joint) == len(previous) == n
    assert [r["generation_index"] for r in raw] == [r["generation_index"] for r in joint] == list(range(n))
    metadata_file = Path(summary["manifest"]["data_manifest"]["directory"]) / "joint_atom_state_metadata.json"
    metadata = json.loads(metadata_file.read_text())["states"]
    # This database has real states only; the dataset appends one fake state.
    assert not any(s.get("is_fake", False) for s in metadata)
    fake = len(metadata)
    sdf_mols = {m.GetIntProp("generation_index"): m for m in
                Chem.SDMolSupplier(str(run / "raw_generated.sdf"), sanitize=False, removeHs=False) if m is not None}
    endpoint_rows = []
    max_coordinate_error = 0.
    for path in sorted((run / "raw_endpoints").glob("*.pt")):
        for item in torch.load(path, map_location="cpu", weights_only=False):
            index = item["generation_index"]
            for data in (item["nodes"], item["edges"]):
                assert all(torch.isfinite(v).all() for v in data.values()), (path, index)
            states = item["nodes"]["s_1"].argmax(-1)
            edge = item["edges"]["e_1"].argmax(-1)
            assert states.max() <= fake and edge.max() <= 3, (index, "residual mask")
            src, dst = item["edge_index"]
            real = states != fake
            upper = (src < dst) & real[src] & real[dst]
            v = torch.zeros(len(states), dtype=torch.long)
            v.index_add_(0, src[upper], edge[upper])
            v.index_add_(0, dst[upper], edge[upper])
            unsupported = high = low = 0
            for s, val in zip(states[real].tolist(), v[real].tolist()):
                allowed = [int(k) for k, count in metadata[s].get("train_heavy_valence_counts", {}).items() if count > 0]
                unsupported += val not in allowed
                high += bool(allowed) and val > max(allowed)
                low += bool(allowed) and val < min(allowed)
            error = None
            if index in sdf_mols:
                original = item["nodes"]["x_1"][real].numpy()
                written = sdf_mols[index].GetConformer().GetPositions()
                assert original.shape == written.shape
                error = float(np.max(np.abs(original - written)))
                assert error <= 5.1e-5, (index, "coordinates changed beyond SDF precision", error)
                max_coordinate_error = max(max_coordinate_error, error)
            endpoint_rows.append(dict(generation_index=index, unsupported_heavy_valence_atoms=unsupported,
                                      above_train_max_atoms=high, below_train_min_atoms=low,
                                      sdf_coordinate_rounding_error=error))
    assert sorted(r["generation_index"] for r in endpoint_rows) == list(range(n))
    endpoint_rows.sort(key=lambda r: r["generation_index"])
    topo = {name: topology(rows) for name, rows in
            (("raw", raw), ("joint", joint), ("previous", previous), ("all_db", db),
             ("train", [r for r in db if r["split"] == "train"]),
             ("train_est_near_zero", [r for r in db if r["split"] == "train" and abs(r["target_est_eV"]) <= .5]))}
    for name in ("raw", "joint", "previous"):
        topo[name]["ring_size_jsd_bits_vs_train"] = jsd(topo[name]["ring_sizes"], topo["train"]["ring_sizes"])
    failure_elements = Counter()
    for row in raw:
        match = re.search(r"atom # \d+ (\w+),", row.get("error") or "")
        if match:
            failure_elements[match.group(1)] += 1
    subsets = {}
    for name, predicate in (("all", lambda r: True), ("invalid", lambda r: not r["raw_valid"]),
                            ("radical_valid", lambda r: r["raw_valid"] and r.get("radicals", 0) > 0)):
        selected = [e for r, e in zip(raw, endpoint_rows) if predicate(r)]
        subsets[name] = dict(n=len(selected),
                             molecules_with_unsupported_valence=sum(e["unsupported_heavy_valence_atoms"] > 0 for e in selected),
                             molecules_above_train_valence_max=sum(e["above_train_max_atoms"] > 0 for e in selected),
                             molecules_below_train_valence_min=sum(e["below_train_min_atoms"] > 0 for e in selected))
    paired = {key: dict(raw_pass_joint_pass=sum(a[key] and b[key] for a, b in zip(raw, joint)),
                        raw_fail_joint_pass=sum(not a[key] and b[key] for a, b in zip(raw, joint)),
                        raw_pass_joint_unavailable_or_fail=sum(a[key] and not b[key] for a, b in zip(raw, joint)))
              for key in ("raw_valid", "eligible", "geometry_eligible")}
    result = dict(endpoint_count=n, all_endpoint_tensors_finite=True, residual_masks=0,
                  original_diffusion_coordinates_verified=len(sdf_mols),
                  max_sdf_coordinate_rounding_error_angstrom=max_coordinate_error,
                  canonical_topology=topo, first_reported_invalid_valence_element=dict(failure_elements),
                  train_valence_support_subsets=subsets, paired_raw_joint=paired,
                  endpoint_diagnostics=endpoint_rows,
                  caveats=["Train support violation is not automatically chemical impossibility.",
                           "First reported valence error is not necessarily the only invalid atom.",
                           "Canonical SMILES order standardizes SSSR but SSSR is not a unique physical ring decomposition.",
                           "Joint solver failures include time-budget outcomes, not proven infeasibility."])
    write_json(run / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "endpoint_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
