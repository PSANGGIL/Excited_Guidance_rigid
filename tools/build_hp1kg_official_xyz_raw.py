#!/usr/bin/env python3
"""Replace synthetic HP1KG coordinates with mapped official S0 coordinates.

The molecular graph, atom/charge/attached-H/radical states, split membership,
and excited-state targets are retained byte-for-byte at the Python value
level.  Only ``atom_positions`` is replaced.  Official XYZ atom order is
matched to the existing heavy-atom graph by graph isomorphism; mapping quality
and inferred-connectivity exceptions are recorded in the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator

import h5py
import networkx as nx
import numpy as np
import torch
from openbabel import openbabel


SPLITS = ("train", "val", "test")
GEOMETRY_SOURCE = "PSDI_official_S0_B3LYP_6-31Gstar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path(
            "data_hp1kg_official_psdi/"
            "hp1kg_benchmarkset1500_official_3d.h5"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("joint_kekule_diffusion/data_hp1kg_nevpt2"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "joint_kekule_diffusion/data_hp1kg_nevpt2_official_xyz"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def target_graph(record: dict[str, Any], atom_map: list[str]) -> nx.Graph:
    graph = nx.Graph()
    periodic_table = openbabel.OBElementTable() if hasattr(openbabel, "OBElementTable") else None
    for index, atom_type in enumerate(torch.as_tensor(record["atom_types"])):
        symbol = atom_map[int(atom_type)]
        if periodic_table is not None:
            atomic_number = int(periodic_table.GetAtomicNum(symbol))
        else:
            # OpenBabel 3 removed OBElementTable from some builds.
            atomic_number = int(openbabel.GetAtomicNum(symbol))
        graph.add_node(
            index,
            atomic_number=atomic_number,
            hydrogens=int(torch.as_tensor(record["atom_hydrogens"])[index]),
        )

    adjacency = torch.as_tensor(record["kekule_bonds"]) > 0
    adjacency |= torch.as_tensor(record["aromatic_edges"], dtype=torch.bool)
    for left, right in torch.triu(adjacency, diagonal=1).nonzero(as_tuple=False):
        graph.add_edge(int(left), int(right))
    return graph


def xyz_graph(
    raw_xyz: bytes, atomic_numbers: np.ndarray
) -> tuple[nx.Graph, dict[int, int]]:
    conversion = openbabel.OBConversion()
    if not conversion.SetInFormat("xyz"):
        raise RuntimeError("OpenBabel XYZ reader is unavailable")
    molecule = openbabel.OBMol()
    if not conversion.ReadString(molecule, raw_xyz.decode("utf-8")):
        raise ValueError("OpenBabel failed to parse an already validated XYZ")
    if molecule.NumAtoms() != len(atomic_numbers):
        raise ValueError(
            f"OpenBabel atom count {molecule.NumAtoms()} != HDF5 count "
            f"{len(atomic_numbers)}"
        )

    graph = nx.Graph()
    attached_hydrogens: dict[int, int] = {}
    for index, atomic_number in enumerate(atomic_numbers):
        if int(atomic_number) != 1:
            attached_hydrogens[index] = 0
            graph.add_node(
                index,
                atomic_number=int(atomic_number),
                hydrogens=0,
            )
    for bond in openbabel.OBMolBondIter(molecule):
        left = bond.GetBeginAtomIdx() - 1
        right = bond.GetEndAtomIdx() - 1
        left_is_h = int(atomic_numbers[left]) == 1
        right_is_h = int(atomic_numbers[right]) == 1
        if not left_is_h and not right_is_h:
            graph.add_edge(left, right)
        elif not left_is_h:
            attached_hydrogens[left] += 1
        elif not right_is_h:
            attached_hydrogens[right] += 1
    nx.set_node_attributes(graph, attached_hydrogens, "hydrogens")
    return graph, attached_hydrogens


def first_mapping(iterator: Iterator[dict[int, int]], size: int) -> dict[int, int] | None:
    for mapping in iterator:
        if len(mapping) == size:
            return mapping
    return None


def find_mapping(
    target: nx.Graph, xyz: nx.Graph
) -> tuple[dict[int, int], str]:
    """Return target-node -> full-XYZ-index mapping and its quality class."""

    if target.number_of_nodes() != xyz.number_of_nodes():
        raise ValueError(
            f"heavy atom count differs: graph={target.number_of_nodes()} "
            f"XYZ={xyz.number_of_nodes()}"
        )
    size = target.number_of_nodes()

    def match_atomic_number(left: dict[str, int], right: dict[str, int]) -> bool:
        return left["atomic_number"] == right["atomic_number"]

    def match_atomic_number_and_h(left: dict[str, int], right: dict[str, int]) -> bool:
        return (
            left["atomic_number"] == right["atomic_number"]
            and left["hydrogens"] == right["hydrogens"]
        )

    # Prefer mappings that also preserve explicit-H attachment.  The fallback
    # deliberately ignores this attribute because bond perception in an XYZ
    # file is distance-based, while the trusted H-count state comes from the
    # source SMILES graph.
    for suffix, node_match in (
        ("hcount", match_atomic_number_and_h),
        ("element", match_atomic_number),
    ):
        mapping = first_mapping(
            nx.isomorphism.GraphMatcher(
                target, xyz, node_match=node_match
            ).isomorphisms_iter(),
            size,
        )
        if mapping is not None:
            return mapping, f"exact_{suffix}"

        # Same node set but one graph may contain a small number of spurious
        # distance-perceived bonds.  Monomorphism retains every trusted edge
        # on the graph with fewer edges without pretending the graphs match.
        xyz_to_target = first_mapping(
            nx.isomorphism.GraphMatcher(
                xyz, target, node_match=node_match
            ).subgraph_monomorphisms_iter(),
            size,
        )
        if xyz_to_target is not None:
            return (
                {target_index: xyz_index for xyz_index, target_index in xyz_to_target.items()},
                f"xyz_superset_{suffix}",
            )

        target_to_xyz = first_mapping(
            nx.isomorphism.GraphMatcher(
                target, xyz, node_match=node_match
            ).subgraph_monomorphisms_iter(),
            size,
        )
        if target_to_xyz is not None:
            return target_to_xyz, f"target_superset_{suffix}"

    raise ValueError("no element-preserving heavy-atom graph mapping exists")


def verify_record_unchanged(
    before: dict[str, Any], after: dict[str, Any]
) -> None:
    if set(after) - set(before) != {
        "geometry_source_original",
        "geometry_hdf5_record_index",
        "geometry_mapping_mode",
        "geometry_hcount_mismatch_atoms",
    }:
        raise RuntimeError("unexpected output record metadata keys")
    for key, before_value in before.items():
        if key in ("atom_positions", "geometry_source"):
            continue
        after_value = after[key]
        if torch.is_tensor(before_value):
            if not torch.equal(before_value, after_value):
                raise RuntimeError(f"record field {key} changed")
        elif before_value != after_value:
            raise RuntimeError(f"record field {key} changed")


def main() -> None:
    args = parse_args()
    h5_path = args.h5.resolve()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output {output}")
    staging = output.with_name(output.name + ".part")
    if staging.exists():
        raise FileExistsError(f"stale staging directory exists: {staging}")
    staging.mkdir(parents=True)

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    atom_map = list(source_manifest["atom_map"])
    h5_hash = sha256(h5_path)
    mode_counts: Counter[str] = Counter()
    mismatch_molecules = 0
    mismatch_atoms = 0
    edge_delta_counts: Counter[str] = Counter()
    output_hashes: dict[str, str] = {}
    coordinate_extrema: list[float] = []

    openbabel.obErrorLog.SetOutputLevel(0)
    with h5py.File(h5_path, "r") as handle:
        if handle.attrs["schema_name"] != "hp1kg_official_3d":
            raise ValueError("unsupported HDF5 schema")
        ids = decode_strings(handle["molecules/id"][:])
        h5_index = {molecule_id: index for index, molecule_id in enumerate(ids)}
        if len(ids) != len(h5_index):
            raise ValueError("HDF5 contains duplicate IDs")
        atom_ptr = handle["atoms/ptr"][:]
        atomic_numbers = handle["atoms/atomic_numbers"][:]
        coordinates = handle["atoms/coordinates"][:]
        byte_ptr = handle["source_xyz/byte_ptr"][:]
        raw_bytes = handle["source_xyz/bytes"][:]

        seen_ids: set[str] = set()
        for split in SPLITS:
            records = torch.load(
                source / f"{split}.pt", map_location="cpu", weights_only=False
            )
            converted: list[dict[str, Any]] = []
            for record in records:
                molecule_id = str(record["id"])
                if molecule_id in seen_ids:
                    raise ValueError(f"duplicate source ID {molecule_id}")
                seen_ids.add(molecule_id)
                if molecule_id not in h5_index:
                    raise KeyError(f"{molecule_id} absent from HDF5")
                record_index = h5_index[molecule_id]
                atom_begin, atom_end = atom_ptr[record_index : record_index + 2]
                byte_begin, byte_end = byte_ptr[record_index : record_index + 2]
                molecule_numbers = atomic_numbers[atom_begin:atom_end]
                molecule_coordinates = coordinates[atom_begin:atom_end]
                molecule_xyz = raw_bytes[byte_begin:byte_end].tobytes()

                trusted_graph = target_graph(record, atom_map)
                perceived_graph, perceived_h = xyz_graph(
                    molecule_xyz, molecule_numbers
                )
                mapping, mode = find_mapping(trusted_graph, perceived_graph)
                mapped_positions = np.stack(
                    [molecule_coordinates[mapping[index]] for index in range(len(mapping))]
                )
                if not np.isfinite(mapped_positions).all():
                    raise ValueError(f"{molecule_id}: mapped coordinate is non-finite")
                expected_numbers = np.asarray(
                    [trusted_graph.nodes[index]["atomic_number"] for index in range(len(mapping))]
                )
                mapped_numbers = np.asarray(
                    [molecule_numbers[mapping[index]] for index in range(len(mapping))]
                )
                if not np.array_equal(expected_numbers, mapped_numbers):
                    raise RuntimeError(f"{molecule_id}: mapped element order mismatch")

                h_mismatch = sum(
                    trusted_graph.nodes[index]["hydrogens"]
                    != perceived_h[mapping[index]]
                    for index in range(len(mapping))
                )
                mismatch_molecules += int(h_mismatch > 0)
                mismatch_atoms += h_mismatch
                mode_counts[mode] += 1
                edge_delta = perceived_graph.number_of_edges() - trusted_graph.number_of_edges()
                edge_delta_counts[str(edge_delta)] += 1
                coordinate_extrema.append(float(np.abs(mapped_positions).max()))

                new_record = dict(record)
                new_record["geometry_source_original"] = str(
                    record.get("geometry_source", "unknown")
                )
                new_record["geometry_source"] = GEOMETRY_SOURCE
                new_record["geometry_hdf5_record_index"] = int(record_index)
                new_record["geometry_mapping_mode"] = mode
                new_record["geometry_hcount_mismatch_atoms"] = int(h_mismatch)
                new_record["atom_positions"] = torch.from_numpy(
                    mapped_positions.astype(np.float32)
                )
                verify_record_unchanged(record, new_record)
                converted.append(new_record)

            split_path = staging / f"{split}.pt"
            torch.save(converted, split_path)
            reloaded = torch.load(split_path, map_location="cpu", weights_only=False)
            if len(reloaded) != len(records):
                raise RuntimeError(f"{split}: post-write record count mismatch")
            for before, after in zip(records, reloaded):
                verify_record_unchanged(before, after)
                if after["atom_positions"].shape != (
                    len(after["atom_types"]), 3
                ):
                    raise RuntimeError(f"{after['id']}: invalid mapped coordinate shape")
                if not torch.isfinite(after["atom_positions"]).all():
                    raise RuntimeError(f"{after['id']}: non-finite mapped coordinate")
            output_hashes[split] = sha256(split_path)

    if seen_ids != set(h5_index):
        missing = sorted(set(h5_index) - seen_ids)
        extra = sorted(seen_ids - set(h5_index))
        raise ValueError(f"source/HDF5 ID mismatch: missing={missing[:5]}, extra={extra[:5]}")

    output_manifest = dict(source_manifest)
    output_manifest.update(
        {
            "schema_version": 2,
            "dataset_type": "hp1kg_nevpt2_official_xyz",
            "coordinate_source": GEOMETRY_SOURCE,
            "coordinate_unit": "angstrom",
            "coordinate_storage_dtype": "float32",
            "coordinate_hdf5_source": str(h5_path),
            "coordinate_hdf5_sha256": h5_hash,
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": sha256(source_manifest_path),
            "source_split_sha256": {
                split: sha256(source / f"{split}.pt") for split in SPLITS
            },
            "output_split_sha256": output_hashes,
            "mapping_mode_counts": dict(sorted(mode_counts.items())),
            "inferred_xyz_edge_delta_counts": dict(sorted(edge_delta_counts.items())),
            "hcount_mapping_mismatch_molecules": mismatch_molecules,
            "hcount_mapping_mismatch_atoms": mismatch_atoms,
            "mapping_policy": (
                "element+attached-H graph isomorphism preferred; element-only heavy-atom "
                "isomorphism/monomorphism fallback for distance-perception differences"
            ),
            "hydrogen_policy": (
                "explicit H coordinates are retained losslessly in HDF5; this model view "
                "uses mapped heavy-atom coordinates and trusted SMILES attached-H states"
            ),
            "max_abs_coordinate_angstrom": max(coordinate_extrema),
            "geometry_sources": {GEOMETRY_SOURCE: len(seen_ids)},
            "errors": [],
        }
    )
    (staging / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n"
    )
    os.replace(staging, output)
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
