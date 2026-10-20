"""Content hashes, independent of machine-specific absolute data paths."""
import hashlib
from pathlib import Path
import torch


DATA_FILES = (
    "train_data_processed.pt", "val_data_processed.pt", "test_data_processed.pt",
    "train_data_property_normalization.pt", "joint_atom_state_metadata.json",
    "train_data_marginal_dists.pt", "train_data_joint_state_marginal.pt",
    "train_data_n_atoms_histogram.pt", "conversion_report.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_manifest(directory):
    directory = Path(directory).resolve()
    return {
        "version": 1, "directory": str(directory),
        "sha256": {name: sha256_file(directory / name) for name in DATA_FILES},
    }


def verify_manifest(expected, actual):
    if not expected or expected.get("version") != 1:
        raise ValueError("checkpoint has no supported data manifest; do not silently resume")
    if expected.get("sha256") != actual.get("sha256"):
        changed = sorted(
            name for name in set(expected.get("sha256", {})) | set(actual.get("sha256", {}))
            if expected.get("sha256", {}).get(name) != actual.get("sha256", {}).get(name)
        )
        raise ValueError(f"checkpoint/data manifest mismatch: {changed}")


def state_dict_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
