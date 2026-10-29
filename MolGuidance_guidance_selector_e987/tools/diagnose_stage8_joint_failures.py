"""Read-only endpoint support diagnostics; count-relaxed controls are NOT production outputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rdkit import Chem, RDLogger
from generate_stage8_panel import write_json
from molguidance.analysis.joint_decode import joint_decode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if (run / "joint_support_diagnosis.json").exists():
        raise FileExistsError(run / "joint_support_diagnosis.json")
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.error")
    manifest = json.loads((run / "manifest.json").read_text())
    metadata = json.loads((Path(manifest["data_manifest"]["directory"]) / "joint_atom_state_metadata.json").read_text())["states"]
    metadata = metadata + [dict(is_fake=True)]
    supported_real = torch.tensor([bool(s.get("train_heavy_valence_counts", {})) for s in metadata])
    joint = json.loads((run / "joint_analysis/records.json").read_text())
    failures = [r["generation_index"] for r in joint if not r["raw_valid"]]
    selected = set(np.random.default_rng(17041).choice(failures, min(64, len(failures)), replace=False).tolist())
    rows, controls = [], []
    totals = Counter()
    for path in sorted((run / "raw_endpoints").glob("*.pt")):
        for item in torch.load(path, weights_only=False, map_location="cpu"):
            index = item["generation_index"]
            p = item["nodes"]["s_endpoint_probs"]
            src, dst = item["edge_index"]
            upper = torch.where(src < dst)[0]
            upper = upper[torch.argsort(src[upper] * len(p) + dst[upper])]
            q = item["edges"]["e_endpoint_probs"][upper]
            mandatory = p[:, -1] == 0
            forced_edge = q[:, 0] == 0
            mandatory[src[upper][forced_edge]] = True
            mandatory[dst[upper][forced_edge]] = True
            minimum = int(mandatory.sum())
            maximum = int((p[:, supported_real] > 0).any(-1).sum())
            requested = item["requested_heavy_atoms"]
            conflict = not minimum <= requested <= maximum
            totals.update(node_rows=len(p), edge_rows=len(q),
                          node_singleton=int(((p > 0).sum(-1) == 1).sum()),
                          edge_singleton=int(((q > 0).sum(-1) == 1).sum()),
                          count_support_conflicts=int(conflict))
            row = dict(generation_index=index, minimum_real_from_nonzero_support=minimum,
                       maximum_real_from_nonzero_support=maximum, requested=requested,
                       count_support_conflict=conflict,
                       original_joint_status=joint[index]["decoder"]["status"])
            rows.append(row)
            if index in selected:
                tick = time.monotonic()
                result = joint_decode(p, q, item["nodes"]["x_1"], metadata,
                                      required_atoms=None, time_limit=2., max_rounds=8)
                controls.append(dict(**row, relaxed_count_status=result.status,
                                     relaxed_count_valid=result.molecule is not None,
                                     retained_count=len(result.retained_indices) if result.molecule is not None else None,
                                     smiles=Chem.MolToSmiles(result.molecule, isomericSmiles=False) if result.molecule is not None else None,
                                     reason=result.reason, seconds=time.monotonic() - tick))
    controls.sort(key=lambda r: r["generation_index"])
    report = dict(all_endpoint_support_counts=dict(totals),
                  singleton_edge_fraction=totals["edge_singleton"] / totals["edge_rows"],
                  singleton_node_fraction=totals["node_singleton"] / totals["node_rows"],
                  uniformly_selected_failure_controls=64, seed=17041,
                  count_relaxed_successes=sum(r["relaxed_count_valid"] for r in controls),
                  count_relaxed_status_counts=dict(Counter(r["relaxed_count_status"] for r in controls)),
                  controls=controls, support_records=rows,
                  note="No new diffusion sampling, no model modification, no probability smoothing. Only required_atoms=None for diagnostic controls; these are not the production results.")
    write_json(run / "joint_support_diagnosis.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("controls", "support_records")}, indent=2))


if __name__ == "__main__":
    main()
