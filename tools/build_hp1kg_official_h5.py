#!/usr/bin/env python3
"""Build a lossless, validated HDF5 copy of the official HP1KG dataset.

The container keeps both a convenient packed numerical representation and the
exact bytes of every downloaded XYZ file.  Coordinates are the optimized S0
geometries distributed by PSDI; the excited-state properties are vertical
SA-CASSCF and SC-NEVPT2 values evaluated at those geometries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SCHEMA_NAME = "hp1kg_official_3d"
SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
ENERGY_STATES = ("S1", "S2", "T1", "T2")
OSCILLATOR_STATES = ("S1", "S2")

# Index in this tuple is the atomic number.  HP1KG only uses a small subset,
# but accepting the full periodic table makes the XYZ parser unambiguous.
ELEMENTS = (
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
    "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
ATOMIC_NUMBER = {symbol: number for number, symbol in enumerate(ELEMENTS) if symbol}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "/home/jovyan/cpu-only-1-datavol-1/psg/generation/"
            "data/hp1kg-21m81/BenchmarkSet1500.csv"
        ),
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=Path("data_hp1kg_official_psdi"),
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("joint_kekule_diffusion/data_hp1kg_nevpt2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data_hp1kg_official_psdi/"
            "hp1kg_benchmarkset1500_official_3d.h5"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_repository_md5(value: str) -> str:
    if not value.startswith("md5:") or len(value) != 36:
        raise ValueError(f"invalid repository checksum: {value!r}")
    return value[4:].lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1581:
        raise ValueError(f"expected 1581 CSV rows, found {len(rows)}")
    ids = [row["ID"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"duplicate CSV IDs: {duplicates[:10]}")
    return rows


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("manifest contains duplicate IDs")
    return by_id


def read_split_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for split in SPLITS:
        records = torch.load(path / f"{split}.pt", map_location="cpu", weights_only=False)
        for record in records:
            molecule_id = str(record["id"])
            if molecule_id in result:
                raise ValueError(f"ID {molecule_id} occurs in multiple splits")
            result[molecule_id] = split
    return result


def parse_formula(formula: str) -> Counter[str]:
    # Official formulas are whitespace-separated (for example C10 H8 S6).
    counts: Counter[str] = Counter()
    for token in formula.split():
        boundary = 1
        if len(token) >= 2 and token[1].islower():
            boundary = 2
        symbol = token[:boundary]
        count_text = token[boundary:]
        if symbol not in ATOMIC_NUMBER:
            raise ValueError(f"unsupported formula element {symbol!r} in {formula!r}")
        counts[symbol] += int(count_text) if count_text else 1
    if not counts:
        raise ValueError(f"empty formula: {formula!r}")
    return counts


def parse_xyz(raw: bytes, source: Path) -> tuple[str, list[str], np.ndarray]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: XYZ is not UTF-8") from exc
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError(f"{source}: truncated XYZ")
    try:
        declared_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{source}: invalid atom count {lines[0]!r}") from exc
    atom_lines = lines[2:]
    if len(atom_lines) != declared_count:
        raise ValueError(
            f"{source}: declared {declared_count} atoms but found {len(atom_lines)}"
        )
    symbols: list[str] = []
    coordinates = np.empty((declared_count, 3), dtype=np.float64)
    for index, line in enumerate(atom_lines):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"{source}:{index + 3}: expected symbol and xyz")
        symbol = fields[0]
        if symbol not in ATOMIC_NUMBER:
            raise ValueError(f"{source}:{index + 3}: unknown element {symbol!r}")
        symbols.append(symbol)
        try:
            coordinates[index] = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise ValueError(f"{source}:{index + 3}: invalid coordinate") from exc
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{source}: non-finite coordinate")
    return lines[1], symbols, coordinates


def as_float(row: dict[str, str], column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value):
        raise ValueError(f"{row['ID']}: non-finite value in {column}")
    return value


def create_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)


def create_numeric_dataset(
    group: h5py.Group, name: str, values: np.ndarray, *, compression: bool = True
) -> h5py.Dataset:
    kwargs: dict[str, Any] = {}
    if compression and values.size:
        kwargs.update(compression="gzip", compression_opts=4, shuffle=True)
    return group.create_dataset(name, data=values, **kwargs)


def build(args: argparse.Namespace) -> dict[str, Any]:
    csv_path = args.csv.resolve()
    download_root = args.download_root.resolve()
    split_root = args.split_root.resolve()
    manifest_path = download_root / "manifest.csv"
    output = args.output.resolve()
    report_path = output.with_suffix(".validation.json")

    for required in (csv_path, manifest_path, split_root):
        if not required.exists():
            raise FileNotFoundError(required)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    rows = read_csv(csv_path)
    manifest = read_manifest(manifest_path)
    split_by_id = read_split_map(split_root)
    csv_ids = {row["ID"] for row in rows}
    if csv_ids != set(manifest):
        raise ValueError("CSV and download manifest ID sets differ")
    if csv_ids != set(split_by_id):
        raise ValueError("CSV and train/val/test split ID sets differ")

    molecule_strings: dict[str, list[str]] = {
        key: []
        for key in (
            "id", "smiles", "inchi", "formula", "ccdc_url", "doi", "split",
            "record_id", "source_url", "xyz_comment", "xyz_md5",
        )
    }
    atom_ptr = [0]
    xyz_byte_ptr = [0]
    atomic_numbers: list[np.ndarray] = []
    coordinates: list[np.ndarray] = []
    xyz_bytes: list[np.ndarray] = []
    n_atoms: list[int] = []
    n_heavy_atoms: list[int] = []
    n_hydrogens: list[int] = []
    casscf_energy: list[list[float]] = []
    nevpt2_energy: list[list[float]] = []
    casscf_oscillator: list[list[float]] = []
    nevpt2_oscillator: list[list[float]] = []

    for row in rows:
        molecule_id = row["ID"]
        item = manifest[molecule_id]
        xyz_path = download_root / item["xyz_file"]
        raw = xyz_path.read_bytes()
        observed_md5 = hashlib.md5(raw).hexdigest()
        expected_md5 = decode_repository_md5(item["repository_checksum"])
        if observed_md5 != expected_md5:
            raise ValueError(
                f"{molecule_id}: XYZ MD5 mismatch {observed_md5} != {expected_md5}"
            )
        comment, symbols, xyz = parse_xyz(raw, xyz_path)
        formula_counts = parse_formula(row["formula"])
        xyz_counts = Counter(symbols)
        if xyz_counts != formula_counts:
            raise ValueError(
                f"{molecule_id}: XYZ composition {xyz_counts} != formula {formula_counts}"
            )

        numbers = np.fromiter((ATOMIC_NUMBER[s] for s in symbols), dtype=np.uint8)
        encoded = np.frombuffer(raw, dtype=np.uint8).copy()
        atom_total = len(symbols)
        hydrogen_total = int(np.count_nonzero(numbers == 1))
        heavy_total = atom_total - hydrogen_total

        for key, value in (
            ("id", molecule_id),
            ("smiles", row["SMILES"]),
            ("inchi", row["InChI"]),
            ("formula", row["formula"]),
            ("ccdc_url", row["ccdc_url"]),
            ("doi", row["doi"]),
            ("split", split_by_id[molecule_id]),
            ("record_id", item["record_id"]),
            ("source_url", item["source_url"]),
            ("xyz_comment", comment),
            ("xyz_md5", observed_md5),
        ):
            molecule_strings[key].append(value)
        atomic_numbers.append(numbers)
        coordinates.append(xyz)
        xyz_bytes.append(encoded)
        n_atoms.append(atom_total)
        n_heavy_atoms.append(heavy_total)
        n_hydrogens.append(hydrogen_total)
        atom_ptr.append(atom_ptr[-1] + atom_total)
        xyz_byte_ptr.append(xyz_byte_ptr[-1] + len(encoded))

        casscf_energy.append(
            [as_float(row, f"SA_CASSCF_E({state})") for state in ENERGY_STATES]
        )
        nevpt2_energy.append(
            [as_float(row, f"NEVPT2_E({state})") for state in ENERGY_STATES]
        )
        casscf_oscillator.append(
            [as_float(row, f"SA_CASSCF_f({state})") for state in OSCILLATOR_STATES]
        )
        nevpt2_oscillator.append(
            [as_float(row, f"NEVPT2_f({state})") for state in OSCILLATOR_STATES]
        )

    atom_ptr_array = np.asarray(atom_ptr, dtype=np.int64)
    byte_ptr_array = np.asarray(xyz_byte_ptr, dtype=np.int64)
    numbers_array = np.concatenate(atomic_numbers)
    coordinates_array = np.concatenate(coordinates, axis=0)
    bytes_array = np.concatenate(xyz_bytes)
    casscf_energy_array = np.asarray(casscf_energy, dtype=np.float64)
    nevpt2_energy_array = np.asarray(nevpt2_energy, dtype=np.float64)
    casscf_oscillator_array = np.asarray(casscf_oscillator, dtype=np.float64)
    nevpt2_oscillator_array = np.asarray(nevpt2_oscillator, dtype=np.float64)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w", track_order=True) as handle:
        handle.attrs.update(
            {
                "schema_name": SCHEMA_NAME,
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "n_molecules": len(rows),
                "n_atoms": int(numbers_array.size),
                "n_heavy_atoms": int(np.count_nonzero(numbers_array != 1)),
                "n_hydrogens": int(np.count_nonzero(numbers_array == 1)),
                "coordinate_unit": "angstrom",
                "explicit_hydrogens": True,
                "geometry_state": "S0",
                "geometry_method": "B3LYP",
                "geometry_basis": "6-31G*",
                "property_type": "vertical excitation at the stored S0 geometry",
                "energy_unit": "eV",
                "oscillator_strength_unit": "dimensionless",
                "source_collection": "PSDI BenchmarkSet1500 / HP1KG-21M81",
                "source_record_url": (
                    "https://data-collections.psdi.ac.uk/records/hp1kg-21m81"
                ),
                "source_csv_sha256": digest_file(csv_path),
                "source_manifest_sha256": digest_file(manifest_path),
                "molecule_order": "BenchmarkSet1500.csv row order",
            }
        )

        molecules = handle.create_group("molecules", track_order=True)
        for name, values in molecule_strings.items():
            create_string_dataset(molecules, name, values)
        create_numeric_dataset(molecules, "n_atoms", np.asarray(n_atoms, dtype=np.int32))
        create_numeric_dataset(
            molecules, "n_heavy_atoms", np.asarray(n_heavy_atoms, dtype=np.int32)
        )
        create_numeric_dataset(
            molecules, "n_hydrogens", np.asarray(n_hydrogens, dtype=np.int32)
        )

        atoms = handle.create_group("atoms", track_order=True)
        create_numeric_dataset(atoms, "ptr", atom_ptr_array)
        create_numeric_dataset(atoms, "atomic_numbers", numbers_array)
        coordinate_dataset = create_numeric_dataset(
            atoms, "coordinates", coordinates_array
        )
        coordinate_dataset.attrs["unit"] = "angstrom"
        coordinate_dataset.attrs["geometry_state"] = "S0"

        source_xyz = handle.create_group("source_xyz", track_order=True)
        create_numeric_dataset(source_xyz, "byte_ptr", byte_ptr_array)
        create_numeric_dataset(source_xyz, "bytes", bytes_array)
        source_xyz.attrs["encoding"] = "UTF-8"
        source_xyz.attrs["description"] = "Exact downloaded XYZ file bytes"

        properties = handle.create_group("properties", track_order=True)
        casscf = properties.create_group("sa_casscf", track_order=True)
        nevpt2 = properties.create_group("nevpt2", track_order=True)
        for group, energy, oscillator in (
            (casscf, casscf_energy_array, casscf_oscillator_array),
            (nevpt2, nevpt2_energy_array, nevpt2_oscillator_array),
        ):
            ds_energy = create_numeric_dataset(group, "energy_ev", energy)
            ds_energy.attrs["states"] = np.asarray(
                ENERGY_STATES, dtype=h5py.string_dtype("utf-8")
            )
            ds_oscillator = create_numeric_dataset(
                group, "oscillator_strength", oscillator
            )
            ds_oscillator.attrs["states"] = np.asarray(
                OSCILLATOR_STATES, dtype=h5py.string_dtype("utf-8")
            )
        est = nevpt2_energy_array[:, 0] - nevpt2_energy_array[:, 2]
        est_dataset = create_numeric_dataset(properties, "nevpt2_est_ev", est)
        est_dataset.attrs["definition"] = "NEVPT2 E(S1) - E(T1)"

    os.replace(temporary, output)

    # Full post-write verification, including exact recovery of all XYZ bytes.
    with h5py.File(output, "r") as handle:
        if handle.attrs["schema_name"] != SCHEMA_NAME:
            raise RuntimeError("post-write schema mismatch")
        if int(handle.attrs["n_molecules"]) != len(rows):
            raise RuntimeError("post-write molecule count mismatch")
        read_ptr = handle["atoms/ptr"][:]
        read_byte_ptr = handle["source_xyz/byte_ptr"][:]
        if not np.array_equal(read_ptr, atom_ptr_array):
            raise RuntimeError("post-write atom pointer mismatch")
        if not np.array_equal(read_byte_ptr, byte_ptr_array):
            raise RuntimeError("post-write XYZ byte pointer mismatch")
        if not np.isfinite(handle["atoms/coordinates"][:]).all():
            raise RuntimeError("post-write non-finite coordinate")
        for index, row in enumerate(rows):
            begin, end = read_byte_ptr[index : index + 2]
            recovered = handle["source_xyz/bytes"][begin:end].tobytes()
            expected = (download_root / manifest[row["ID"]]["xyz_file"]).read_bytes()
            if recovered != expected:
                raise RuntimeError(f"post-write XYZ byte mismatch for {row['ID']}")

    report = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "output_sha256": digest_file(output),
        "molecules": len(rows),
        "split_counts": dict(Counter(split_by_id.values())),
        "atoms": int(numbers_array.size),
        "heavy_atoms": int(np.count_nonzero(numbers_array != 1)),
        "hydrogens": int(np.count_nonzero(numbers_array == 1)),
        "elements": {
            ELEMENTS[int(number)]: int(count)
            for number, count in sorted(Counter(numbers_array.tolist()).items())
        },
        "coordinate_dtype": str(coordinates_array.dtype),
        "coordinate_unit": "angstrom",
        "all_coordinates_finite": bool(np.isfinite(coordinates_array).all()),
        "all_properties_finite": bool(
            np.isfinite(casscf_energy_array).all()
            and np.isfinite(nevpt2_energy_array).all()
            and np.isfinite(casscf_oscillator_array).all()
            and np.isfinite(nevpt2_oscillator_array).all()
        ),
        "xyz_md5_match_count": len(rows),
        "xyz_exact_byte_roundtrip_count": len(rows),
        "geometry": "S0 B3LYP/6-31G* optimized",
        "properties": "vertical SA-CASSCF and SC-NEVPT2 at stored S0 geometry",
        "nevpt2_ranges_ev": {
            state: [
                float(nevpt2_energy_array[:, index].min()),
                float(nevpt2_energy_array[:, index].max()),
            ]
            for index, state in enumerate(ENERGY_STATES)
        },
        "nevpt2_est_range_ev": [float(est.min()), float(est.max())],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    args = parse_args()
    report = build(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
