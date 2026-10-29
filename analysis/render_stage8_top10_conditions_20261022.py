#!/usr/bin/env python3
"""Render target-ranked, DB-novel unique molecules with explicit label spacing."""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(source, top_n):
    generation_manifest = json.loads((source / "manifest.json").read_text())
    properties = [key for key in ("est", "s1")
                  if generation_manifest.get(f"target_{key}_eV") is not None]
    if len(properties) != 1:
        raise ValueError("Expected one unambiguous generation condition")
    condition = properties[0]
    target = Decimal(str(generation_manifest[f"target_{condition}_eV"]))
    if not target.is_finite():
        raise ValueError("Nonfinite target")
    calculation = source / "xtb_stda_diffusion_coords_12x1_20261022"
    if json.loads((calculation / "status.json").read_text())["phase"] != "complete":
        raise ValueError("Calculations are not complete")
    protocol = json.loads((calculation / "protocol.json").read_text())
    if protocol["checkpoint_sha256"] != generation_manifest["checkpoint_sha256"]:
        raise ValueError("Calculation/generation checkpoint mismatch")
    if Decimal(str(protocol[f"target_{condition}_eV"])) != target:
        raise ValueError("Calculation/generation condition mismatch")
    raw = json.loads((source / "raw_analysis/records.json").read_text())
    raw_by_index = {int(row["generation_index"]): row for row in raw}
    if len(raw) != len(raw_by_index):
        raise ValueError("Duplicate raw indices")
    with (calculation / "results.csv").open(newline="") as handle:
        results = list(csv.DictReader(handle))
    candidates, seen_indices = [], set()
    excluded_db, succeeded = 0, 0
    for row in results:
        index = int(row["generation_index"])
        if index in seen_indices:
            raise ValueError("Duplicate calculation indices")
        seen_indices.add(index)
        if row["status"] != "ok":
            continue
        succeeded += 1
        original = raw_by_index[index]
        if not original["eligible"]:
            raise ValueError("Calculation succeeded for an ineligible raw graph")
        if original["db_member"] is True:
            excluded_db += 1
            continue
        if original["db_member"] is not False:
            raise ValueError("Missing or ambiguous DB-membership flag")
        if row["condition_name"] != condition:
            raise ValueError("Wrong result condition")
        if Decimal(row[f"target_{condition}_eV"]) != target:
            raise ValueError("Wrong per-record target")
        for key in ("xtb_returncode", "xtb4stda_returncode", "singlet_returncode", "triplet_returncode"):
            if int(row[key]) != 0:
                raise ValueError("Nonzero return code in a success row")
        s1, t1 = Decimal(row["s1_energy_eV"]), Decimal(row["t1_energy_eV"])
        if not s1.is_finite() or not t1.is_finite():
            raise ValueError("Nonfinite result energy")
        est = s1 - t1
        error = abs((est if condition == "est" else s1) - target)
        if Decimal(row["abs_condition_error_eV"]) != error:
            raise ValueError("Recorded condition error disagrees with reported energies")
        mol = Chem.MolFromSmiles(row["smiles"])
        reference = Chem.MolFromSmiles(original["smiles"])
        if mol is None or reference is None:
            raise ValueError("Unparseable successful graph")
        identity = Chem.MolToSmiles(mol, isomericSmiles=False)
        if identity != Chem.MolToSmiles(reference, isomericSmiles=False):
            raise ValueError("Calculation/raw graph identity mismatch")
        if len(Chem.GetMolFrags(mol)) != 1 or sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
            raise ValueError("Graph must be connected and net-neutral")
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            raise ValueError("Open-shell graph in successful subset")
        candidates.append(dict(index=index, mol=mol, identity=identity, s1=s1, t1=t1,
                               est=est, error=error, result=row, original=original))
    candidates.sort(key=lambda row: (row["error"], row["index"]))
    unique, seen = [], set()
    for row in candidates:
        if row["identity"] not in seen:
            seen.add(row["identity"])
            unique.append(row)
    if len(unique) < top_n:
        raise ValueError(f"Only {len(unique)} eligible novel unique molecules, requested {top_n}")
    return generation_manifest, calculation, condition, target, unique[:top_n], dict(
        calculation_successes=succeeded, excluded_db_members=excluded_db,
        novel_success_records=len(candidates), novel_unique_successes=len(unique),
        duplicate_success_records=len(candidates) - len(unique))


def render(rows, png, title, condition, target):
    columns = 5
    cell_width, cell_height, structure_height = 680, 620, 410
    header, footer = 130, 90
    nrows = (len(rows) + columns - 1) // columns
    width, height = columns * cell_width, header + nrows * cell_height + footer
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height, cell_width, structure_height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.padding = .10
    options.minFontSize = 14
    options.maxFontSize = 32
    options.bondLineWidth = 2.2
    drawer.ClearDrawing()
    for position, row in enumerate(rows):
        mol = Chem.Mol(row["mol"])
        rdDepictor.Compute2DCoords(mol)
        if Chem.MolToSmiles(mol, isomericSmiles=False) != row["identity"]:
            raise ValueError("2D depiction changed graph identity")
        drawer.SetOffset((position % columns) * cell_width,
                         header + (position // columns) * cell_height)
        drawer.DrawMolecule(mol)
    drawer.SetOffset(0, 0)
    drawer.SetColour((.10, .13, .18))
    drawer.SetFontSize(36)
    drawer.DrawString(f"{title} | Target {condition.upper()} = {target:.3f} eV | Top {len(rows)}",
                      Point2D(width / 2, 42), 0, True)
    drawer.SetFontSize(24)
    drawer.DrawString("Ranked by absolute XTB/sTDA target error; DB members and duplicate graphs excluded",
                      Point2D(width / 2, 91), 0, True)
    line_centers = (455, 507, 559)
    for position, row in enumerate(rows):
        center_x = (position % columns) * cell_width + cell_width / 2
        top_y = header + (position // columns) * cell_height
        labels = (
            f"Rank {position + 1:02d} | Molecule #{row['index']:06d}",
            f"S1 = {row['s1']:.3f} eV | T1 = {row['t1']:.3f} eV",
            f"EST = {row['est']:+.3f} eV | Error = {row['error']:.3f} eV",
        )
        for line_number, (label, y_offset) in enumerate(zip(labels, line_centers)):
            drawer.SetColour((.08, .12, .18) if line_number == 0 else (.16, .19, .24))
            drawer.SetFontSize(29 if line_number == 0 else 27)
            drawer.DrawString(label, Point2D(center_x, top_y + y_offset), 0, True)
    drawer.SetFontSize(22)
    drawer.SetColour((.30, .33, .38))
    drawer.DrawString("Original generated graphs in RDKit 2D; energies after XTB optimization, uncalibrated.",
                      Point2D(width / 2, height - 59), 0, True)
    drawer.DrawString("Top ranking is target proximity, not a certificate of physical stability or unchanged optimized connectivity.",
                      Point2D(width / 2, height - 25), 0, True)
    drawer.FinishDrawing()
    with png.open("xb") as handle:
        handle.write(drawer.GetDrawingText())
    return dict(width_px=width, height_px=height, columns=columns, rows=nrows,
                cell_size_px=[cell_width, cell_height], structure_height_px=structure_height,
                label_line_centers_px=list(line_centers), label_line_separation_px=52,
                id_font_px=29, energy_font_px=27)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    if args.n < 1:
        parser.error("n must be positive")
    source, output = args.generation.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    sources = [source / "manifest.json", source / "raw_analysis/records.json",
               source / "xtb_stda_diffusion_coords_12x1_20261022/results.csv",
               source / "xtb_stda_diffusion_coords_12x1_20261022/protocol.json"]
    hashes = {str(p): digest(p) for p in sources}
    manifest, calculation, condition, target, selected, counts = select(source, args.n)
    output.mkdir(parents=True, exist_ok=False)
    png = output / f"top{args.n}_{condition}_target_match_spaced.png"
    layout = render(selected, png, args.title, condition, target)
    exported = []
    for rank, row in enumerate(selected, 1):
        exported.append(dict(rank=rank, generation_index=row["index"], canonical_smiles=row["identity"],
            condition_name=condition, target_eV=str(target), s1_energy_eV=str(row["s1"]),
            t1_energy_eV=str(row["t1"]), calculated_est_eV=str(row["est"]),
            absolute_target_error_eV=str(row["error"]), s1_f_osc=row["result"]["s1_f_osc"],
            db_member=False, connected=True, net_charge=0, radicals=0,
            aromatic=row["original"].get("aromatic"),
            topology_screen=row["original"].get("topology_screen"),
            geometry_screen=row["original"].get("geometry_screen"),
            ring_sizes=json.dumps(row["original"].get("ring_sizes"))))
    csv_path = output / "selected_molecules.csv"
    with csv_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    with (output / "selected_molecules.smi").open("x") as handle:
        for row in exported:
            handle.write(f"{row['canonical_smiles']}\trank{row['rank']:02d}_generated_{row['generation_index']:06d}\n")
    if any(digest(Path(path)) != sha for path, sha in hashes.items()):
        raise ValueError("Source artifact changed while rendering")
    summary = dict(generation=str(source), calculation=str(calculation), condition_name=condition,
        target_eV=str(target), checkpoint_epoch=manifest.get("checkpoint_epoch_zero_based", manifest.get("checkpoint_epoch")),
        checkpoint_sha256=manifest["checkpoint_sha256"], source_sha256=hashes,
        selection="calculation ok, raw eligible, DB nonmember; unique canonical non-stereo identity; abs target error ascending, then generation index",
        novelty_scope=manifest.get("novelty_scope", "used HP1KG train+validation+test corpus"),
        topology_or_geometry_filter=False, counts=counts, selected=len(selected),
        generation_indices=[row["index"] for row in selected],
        min_absolute_error_eV=str(selected[0]["error"]), max_absolute_error_eV=str(selected[-1]["error"]),
        png=str(png), csv=str(csv_path), layout=layout, rdkit_version=rdBase.rdkitVersion,
        notes="RDKit 2D of original generated graphs; uncalibrated XTB/sTDA energies; no new quantum calculation or 3D re-embedding")
    with (output / "summary.json").open("x") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    shutil.copy2(Path(__file__).resolve(), output / "render_script.py")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
