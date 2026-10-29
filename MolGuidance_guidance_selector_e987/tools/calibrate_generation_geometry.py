"""Read-only DB oracle reconstruction and the same geometry screen as generation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rdkit import Chem
from generate_stage8_panel import aggregate, detailed_record, write_json
from molguidance.analysis.molecule_builder import build_molecule
from molguidance.model_utils.provenance import data_manifest, verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    before = data_manifest(args.data)
    metadata = json.loads((args.data / "joint_atom_state_metadata.json").read_text())["states"]
    rows = []
    for split in ("train", "val", "test"):
        data = torch.load(args.data / f"{split}_data_processed.pt", weights_only=False, map_location="cpu")
        for index, smi in enumerate(data["smiles"]):
            ns, ne = map(int, data["node_idx_array"][index])
            es, ee = map(int, data["edge_idx_array"][index])
            states = [metadata[int(s)] for s in data["atom_states"][ns:ne]]
            bonds = data["bond_idxs"][es:ee]
            mol = build_molecule(
                data["positions"][ns:ne], [s["symbol"] for s in states],
                [s["formal_charge"] for s in states], bonds[:, 0], bonds[:, 1],
                data["bond_types"][es:ee].tolist(), [s["hydrogens"] for s in states])
            row = detailed_record(mol)
            reference = Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=False)
            assert row["raw_valid"] and row["smiles"] == reference, (split, index, row)
            row.update(split=split, split_index=index, target_est_eV=float(data["properties"][index]))
            rows.append(row)
    verify_manifest(before, data_manifest(args.data))
    result = dict(all_db=aggregate(rows),
                  train=aggregate([r for r in rows if r["split"] == "train"]),
                  train_abs_est_le_0p5=aggregate([r for r in rows if r["split"] == "train" and abs(r["target_est_eV"]) <= .5]),
                  identity_verified=len(rows), records=rows, data_manifest=before,
                  geometry="Official optimized DB XYZ in stored atom order; no reembedding or optimization.")
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "data_manifest")}, indent=2))


if __name__ == "__main__":
    main()
