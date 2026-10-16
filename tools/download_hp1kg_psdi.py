#!/usr/bin/env python3
"""Download and verify the official BenchmarkSet1500 geometries from PSDI.

The aggregated HP1KG record contains only a CSV.  The optimized XYZ and ORCA
output files live in 1,581 individual records in the BenchmarkSet1500
community.  This utility downloads only the XYZ files by default, verifies
their repository checksums and XYZ syntax, and can additionally download one
ORCA output for a coordinate/provenance cross-check.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


COMMUNITY_API = (
    "https://data-collections.psdi.ac.uk/api/communities/"
    "benchmarkset1500/records?size=2000&page=1"
)
USER_AGENT = "hp1kg-official-geometry-downloader/1.0"
ELEMENT_RE = re.compile(r"^[A-Z][a-z]?$")
FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*)")
MD5_RE = re.compile(r"^md5:([0-9a-fA-F]{32})$")


@dataclass(frozen=True)
class RemoteFile:
    molecule_id: str
    record_id: str
    key: str
    size: int
    checksum: str

    @property
    def url(self) -> str:
        quoted_key = urllib.parse.quote(self.key, safe="")
        return (
            "https://data-collections.psdi.ac.uk/api/records/"
            f"{self.record_id}/files/{quoted_key}/content"
        )


@dataclass(frozen=True)
class XyzData:
    atom_count: int
    symbols: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    comment: str

    @property
    def hydrogen_count(self) -> int:
        return sum(symbol == "H" for symbol in self.symbols)

    @property
    def heavy_atom_count(self) -> int:
        return self.atom_count - self.hydrogen_count


def request_bytes(url: str, timeout: float, retries: int) -> bytes:
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def md5sum(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - repository supplies MD5 checksums
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(checksum: str) -> str:
    match = MD5_RE.fullmatch(checksum)
    if not match:
        raise ValueError(f"unsupported checksum: {checksum!r}")
    return match.group(1).lower()


def file_matches(path: Path, remote: RemoteFile) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == remote.size
        and md5sum(path) == expected_md5(remote.checksum)
    )


def download_file(
    remote: RemoteFile,
    destination: Path,
    timeout: float,
    retries: int,
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if file_matches(destination, remote):
        return "cached"

    partial = destination.with_name(destination.name + ".part")
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        digest = hashlib.md5()  # noqa: S324 - repository supplies MD5 checksums
        byte_count = 0
        try:
            request = urllib.request.Request(
                remote.url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with partial.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        byte_count += len(chunk)

            if byte_count != remote.size:
                raise ValueError(
                    f"size mismatch for {remote.key}: {byte_count} != {remote.size}"
                )
            observed_md5 = digest.hexdigest()
            if observed_md5 != expected_md5(remote.checksum):
                raise ValueError(
                    f"MD5 mismatch for {remote.key}: {observed_md5}"
                )
            os.replace(partial, destination)
            return "downloaded"
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt == retries:
                break
            time.sleep(min(2**attempt, 16))
    raise RuntimeError(f"failed to download {remote.key}: {last_error}")


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(content)
    os.replace(partial, path)


def load_records(
    metadata_path: Path,
    supplied_records: Path | None,
    timeout: float,
    retries: int,
) -> dict[str, Any]:
    if supplied_records is not None:
        content = supplied_records.read_bytes()
        data = json.loads(content)
        atomic_write_bytes(metadata_path, content)
        return data
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())
    content = request_bytes(COMMUNITY_API, timeout=timeout, retries=retries)
    data = json.loads(content)
    atomic_write_bytes(metadata_path, content)
    return data


def one_remote_file(record: dict[str, Any], extension: str) -> RemoteFile:
    molecule_id = str(record["metadata"]["title"]).strip()
    entries = record.get("files", {}).get("entries", {})
    matches = [entry for key, entry in entries.items() if key.lower().endswith(extension)]
    if len(matches) != 1:
        raise ValueError(
            f"{molecule_id}: expected exactly one {extension} file, found {len(matches)}"
        )
    entry = matches[0]
    return RemoteFile(
        molecule_id=molecule_id,
        record_id=str(record["id"]),
        key=str(entry["key"]),
        size=int(entry["size"]),
        checksum=str(entry["checksum"]),
    )


def load_csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        molecule_id = row["ID"].strip()
        if molecule_id in result:
            raise ValueError(f"duplicate ID in CSV: {molecule_id}")
        result[molecule_id] = row
    return result


def parse_xyz(path: Path) -> XyzData:
    lines = path.read_text(errors="strict").splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: incomplete XYZ header")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{path}: invalid XYZ atom count") from exc
    if atom_count <= 0 or len(lines) < atom_count + 2:
        raise ValueError(
            f"{path}: expected {atom_count} coordinate rows, found {max(0, len(lines)-2)}"
        )

    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    for line_number, line in enumerate(lines[2 : atom_count + 2], start=3):
        fields = line.split()
        if len(fields) < 4 or not ELEMENT_RE.fullmatch(fields[0]):
            raise ValueError(f"{path}:{line_number}: malformed coordinate row")
        xyz = tuple(float(value) for value in fields[1:4])
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError(f"{path}:{line_number}: non-finite coordinate")
        symbols.append(fields[0])
        coordinates.append(xyz)  # type: ignore[arg-type]

    return XyzData(
        atom_count=atom_count,
        symbols=tuple(symbols),
        coordinates=tuple(coordinates),
        comment=lines[1].strip(),
    )


def formula_atom_count(formula: str) -> int:
    compact = formula.replace(" ", "")
    matches = list(FORMULA_RE.finditer(compact))
    if not matches or "".join(match.group(0) for match in matches) != compact:
        raise ValueError(f"cannot parse molecular formula: {formula!r}")
    return sum(int(match.group(2) or "1") for match in matches)


def parse_orca_coordinates(text: str) -> XyzData:
    lines = text.splitlines()
    header_indices = [
        index
        for index, line in enumerate(lines)
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line.upper()
    ]
    for header_index in header_indices:
        symbols: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        started = False
        for line in lines[header_index + 1 :]:
            fields = line.split()
            if len(fields) >= 4 and ELEMENT_RE.fullmatch(fields[0]):
                try:
                    xyz = tuple(float(value) for value in fields[1:4])
                except ValueError:
                    if started:
                        break
                    continue
                symbols.append(fields[0])
                coordinates.append(xyz)  # type: ignore[arg-type]
                started = True
            elif started:
                break
        if symbols:
            return XyzData(
                atom_count=len(symbols),
                symbols=tuple(symbols),
                coordinates=tuple(coordinates),
                comment="ORCA CARTESIAN COORDINATES (ANGSTROEM)",
            )
    raise ValueError("ORCA output contains no parseable Cartesian-coordinate block")


def coordinate_comparison(xyz: XyzData, orca: XyzData) -> dict[str, Any]:
    same_symbols = xyz.symbols == orca.symbols
    if xyz.atom_count != orca.atom_count or not same_symbols:
        return {
            "atom_count_match": xyz.atom_count == orca.atom_count,
            "symbol_order_match": same_symbols,
            "coordinate_rmsd_angstrom": None,
            "coordinate_max_abs_delta_angstrom": None,
        }
    squared = []
    max_abs_delta = 0.0
    for left, right in zip(xyz.coordinates, orca.coordinates):
        for left_value, right_value in zip(left, right):
            delta = left_value - right_value
            squared.append(delta * delta)
            max_abs_delta = max(max_abs_delta, abs(delta))
    return {
        "atom_count_match": True,
        "symbol_order_match": True,
        "coordinate_rmsd_angstrom": math.sqrt(sum(squared) / len(squared)),
        "coordinate_max_abs_delta_angstrom": max_abs_delta,
    }


def batched_progress(completed: int, total: int, downloaded: int, cached: int) -> None:
    if completed == total or completed % 100 == 0:
        print(
            f"XYZ progress {completed}/{total} "
            f"(downloaded={downloaded}, cached={cached})",
            flush=True,
        )


def write_manifest(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    fieldnames = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with partial.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--records-json",
        type=Path,
        help="Optional already-fetched community API JSON (useful for resuming)",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--test-out",
        metavar="CCDC_ID",
        help="Also download one ORCA .out and compare its printed coordinates",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    output: Path = args.output.resolve()
    xyz_dir = output / "xyz"
    metadata_dir = output / "metadata"
    out_test_dir = output / "out_test"
    output.mkdir(parents=True, exist_ok=True)

    records_document = load_records(
        metadata_dir / "community_records.json",
        args.records_json,
        timeout=args.timeout,
        retries=args.retries,
    )
    records = records_document["hits"]["hits"]
    declared_total = int(records_document["hits"]["total"])
    if len(records) != declared_total:
        raise ValueError(
            f"community response is incomplete: {len(records)} of {declared_total} records"
        )

    records_by_id: dict[str, dict[str, Any]] = {}
    xyz_remotes: dict[str, RemoteFile] = {}
    out_remotes: dict[str, RemoteFile] = {}
    for record in records:
        molecule_id = str(record["metadata"]["title"]).strip()
        if molecule_id in records_by_id:
            raise ValueError(f"duplicate API molecule ID: {molecule_id}")
        records_by_id[molecule_id] = record
        xyz_remotes[molecule_id] = one_remote_file(record, ".xyz")
        out_remotes[molecule_id] = one_remote_file(record, ".out")

    csv_rows = load_csv_rows(args.csv.resolve())
    api_ids = set(records_by_id)
    csv_ids = set(csv_rows)
    if api_ids != csv_ids:
        raise ValueError(
            "CSV/API ID mismatch: "
            f"CSV-only={sorted(csv_ids-api_ids)[:20]}, "
            f"API-only={sorted(api_ids-csv_ids)[:20]}"
        )

    statuses: dict[str, str] = {}
    failures: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_file,
                remote,
                xyz_dir / remote.key,
                args.timeout,
                args.retries,
            ): molecule_id
            for molecule_id, remote in xyz_remotes.items()
        }
        downloaded = 0
        cached = 0
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            molecule_id = futures[future]
            try:
                status = future.result()
                statuses[molecule_id] = status
                downloaded += status == "downloaded"
                cached += status == "cached"
            except Exception as exc:  # report every failed molecule together
                failures[molecule_id] = str(exc)
            batched_progress(completed, len(futures), downloaded, cached)

    if failures:
        atomic_write_bytes(
            output / "download_failures.json",
            json.dumps(failures, indent=2, sort_keys=True).encode(),
        )
        raise RuntimeError(f"{len(failures)} XYZ downloads failed; rerun to resume")

    manifest_rows: list[dict[str, Any]] = []
    formula_mismatches: list[str] = []
    heavy_count_mismatches: list[str] = []
    comments_without_id: list[str] = []
    for molecule_id in sorted(csv_ids):
        remote = xyz_remotes[molecule_id]
        path = xyz_dir / remote.key
        if not file_matches(path, remote):
            raise ValueError(f"post-download checksum failure: {path}")
        xyz = parse_xyz(path)
        row = csv_rows[molecule_id]
        expected_total_atoms = formula_atom_count(row["formula"])
        csv_heavy_atoms = int(float(row["#atoms"]))
        if xyz.atom_count != expected_total_atoms:
            formula_mismatches.append(molecule_id)
        if xyz.heavy_atom_count != csv_heavy_atoms:
            heavy_count_mismatches.append(molecule_id)
        if molecule_id not in xyz.comment:
            comments_without_id.append(molecule_id)
        manifest_rows.append(
            {
                "id": molecule_id,
                "record_id": remote.record_id,
                "xyz_file": str(path.relative_to(output)),
                "source_url": remote.url,
                "repository_size_bytes": remote.size,
                "repository_checksum": remote.checksum,
                "download_status": statuses[molecule_id],
                "xyz_atom_count": xyz.atom_count,
                "xyz_heavy_atom_count": xyz.heavy_atom_count,
                "xyz_hydrogen_count": xyz.hydrogen_count,
                "formula_atom_count": expected_total_atoms,
                "csv_heavy_atom_count": csv_heavy_atoms,
                "formula_count_match": xyz.atom_count == expected_total_atoms,
                "heavy_count_match": xyz.heavy_atom_count == csv_heavy_atoms,
            }
        )

    write_manifest(output / "manifest.csv", manifest_rows)

    out_validation: dict[str, Any] | None = None
    if args.test_out:
        test_id = args.test_out.strip()
        if test_id not in out_remotes:
            raise ValueError(f"unknown --test-out ID: {test_id}")
        out_remote = out_remotes[test_id]
        out_path = out_test_dir / out_remote.key
        out_status = download_file(
            out_remote, out_path, args.timeout, args.retries
        )
        xyz = parse_xyz(xyz_dir / xyz_remotes[test_id].key)
        out_text = out_path.read_text(errors="replace")
        orca_xyz = parse_orca_coordinates(out_text)
        comparison = coordinate_comparison(xyz, orca_xyz)
        custom = records_by_id[test_id]["metadata"].get("custom_fields", {})
        domain_metadata = (custom.get("dsmd") or [{}])[0]
        out_validation = {
            "id": test_id,
            "out_file": str(out_path.relative_to(output)),
            "download_status": out_status,
            "repository_size_bytes": out_remote.size,
            "repository_checksum": out_remote.checksum,
            "orca_terminated_normally": "ORCA TERMINATED NORMALLY" in out_text,
            "contains_casscf": "CASSCF" in out_text.upper(),
            "contains_nevpt2": "NEVPT2" in out_text.upper(),
            "ground_state_functional": domain_metadata.get(
                "bs1500_ground_state_functional"
            ),
            "ground_state_basis_set": domain_metadata.get(
                "bs1500_ground_state_basis_set"
            ),
            **comparison,
        }
        atomic_write_bytes(
            output / "out_test_validation.json",
            json.dumps(out_validation, indent=2, sort_keys=True).encode(),
        )

    summary = {
        "source_community_api": COMMUNITY_API,
        "source_csv": str(args.csv.resolve()),
        "api_record_count": len(records),
        "csv_record_count": len(csv_rows),
        "id_sets_match": api_ids == csv_ids,
        "xyz_downloaded_this_run": sum(value == "downloaded" for value in statuses.values()),
        "xyz_reused_from_cache": sum(value == "cached" for value in statuses.values()),
        "xyz_checksum_valid_count": len(manifest_rows),
        "xyz_parse_valid_count": len(manifest_rows),
        "formula_atom_count_mismatch_count": len(formula_mismatches),
        "formula_atom_count_mismatch_ids": formula_mismatches,
        "csv_heavy_atom_count_mismatch_count": len(heavy_count_mismatches),
        "csv_heavy_atom_count_mismatch_ids": heavy_count_mismatches,
        "xyz_comments_without_id_count": len(comments_without_id),
        "xyz_comments_without_id_examples": comments_without_id[:20],
        "total_xyz_atoms": sum(row["xyz_atom_count"] for row in manifest_rows),
        "total_xyz_hydrogens": sum(row["xyz_hydrogen_count"] for row in manifest_rows),
        "out_test": out_validation,
    }
    atomic_write_bytes(
        output / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True).encode(),
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130)
