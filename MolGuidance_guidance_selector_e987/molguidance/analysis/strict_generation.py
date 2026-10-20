"""Unrepaired graph/coordinate screens. These are NOT physical stability tests."""
import numpy as np
from rdkit import Chem


def strict_molecule_record(molecule):
    record = dict(raw_valid=False, connected=False, neutral_closed_shell=False,
                  aromatic=False, topology_screen=False, geometry_screen=False,
                  eligible=False, geometry_evaluable=False, geometry_eligible=False,
                  smiles=None, error=None, heavy_atoms=0, bonds=0)
    if molecule is None:
        record["error"] = "decoder_none"
        return record
    mol = Chem.Mol(molecule)
    record["heavy_atoms"] = mol.GetNumHeavyAtoms()
    record["bonds"] = mol.GetNumBonds()
    try:
        Chem.SanitizeMol(mol)  # whole molecule; never select the largest fragment
        if not mol.GetNumAtoms():
            record["error"] = "empty_molecule"
            return record
        record["raw_valid"] = True
        record["connected"] = len(Chem.GetMolFrags(mol)) == 1
        record["net_charge"] = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        record["radicals"] = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        record["neutral_closed_shell"] = record["net_charge"] == 0 and record["radicals"] == 0
        record["aromatic"] = any(a.GetIsAromatic() for a in mol.GetAtoms())
        sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        record["ring_sizes"] = sizes
        record["small_ring_warning"] = any(size <= 4 for size in sizes)
        record["large_ring_warning"] = any(size >= 9 for size in sizes)
        record["topology_screen"] = not (record["small_ring_warning"] or record["large_ring_warning"])
        record["smiles"] = Chem.MolToSmiles(mol, isomericSmiles=False)
        record["eligible"] = record["connected"] and record["neutral_closed_shell"]
        if mol.GetNumConformers():
            xyz = mol.GetConformer().GetPositions()
            if np.isfinite(xyz).all():
                radius = np.array([Chem.GetPeriodicTable().GetRcovalent(a.GetAtomicNum()) for a in mol.GetAtoms()])
                distance = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1)
                ratio = distance / np.maximum(radius[:, None] + radius[None, :], 1e-8)
                adjacency = Chem.GetAdjacencyMatrix(mol).astype(bool)
                upper = np.triu(np.ones_like(adjacency), 1).astype(bool)
                bonded = ratio[adjacency & upper]
                nonbonded = ratio[~adjacency & upper]
                # A one-heavy-atom collapse must not pass all geometry tests
                # merely because every tested bond/nonbond array is empty.
                record["geometry_evaluable"] = bool(bonded.size > 0)
                plane_error = []
                for ring in mol.GetRingInfo().AtomRings():
                    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                        centered = xyz[list(ring)] - xyz[list(ring)].mean(0)
                        _, _, vt = np.linalg.svd(centered, full_matrices=False)
                        plane_error.append(float(np.sqrt(np.mean((centered @ vt[-1]) ** 2))))
                record["max_aromatic_plane_rms_angstrom"] = max(plane_error, default=0.)
                record["geometry_screen"] = bool(
                    record["geometry_evaluable"]
                    and np.all((bonded >= .65) & (bonded <= 1.4))
                    and np.all(nonbonded >= .65)
                    and record["max_aromatic_plane_rms_angstrom"] <= .15
                )
        record["geometry_eligible"] = record["eligible"] and record["geometry_screen"]
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
    return record


def summarize_records(records):
    total = len(records)
    keys = ("raw_valid", "connected", "neutral_closed_shell", "eligible",
            "geometry_evaluable", "geometry_screen", "geometry_eligible", "aromatic", "topology_screen")
    result = {"requested": total}
    for key in keys:
        result[f"{key}_count"] = sum(bool(row[key]) for row in records)
        result[f"{key}_rate"] = result[f"{key}_count"] / max(total, 1)
    result["aromatic_eligible_rate"] = sum(row["eligible"] and row["aromatic"] for row in records) / max(total, 1)
    result["geometry_aromatic_eligible_rate"] = sum(
        row["geometry_eligible"] and row["aromatic"] for row in records
    ) / max(total, 1)
    result["topology_eligible_rate"] = sum(
        row["eligible"] and row["topology_screen"] for row in records
    ) / max(total, 1)
    result["novel_unique_rate"] = sum(row.get("novel_unique", False) for row in records) / max(total, 1)
    result["novel_unique_connected_rate"] = sum(
        row.get("novel_unique", False) and row["connected"] for row in records
    ) / max(total, 1)
    result["novel_unique_eligible_rate"] = sum(
        row.get("novel_unique", False) and row["eligible"] for row in records
    ) / max(total, 1)
    return result
