#!/usr/bin/env python3
"""Build a split-identical HP1KG ΔE_ST dataset and matching training configs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import yaml
from rdkit import Chem


BUNDLE_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = BUNDLE_DIR.parent
SOURCE_DATA = WORKSPACE_DIR / "molguidance_hp1kg_baseline" / "data_s1"
OUTPUT_DATA = WORKSPACE_DIR / "molguidance_hp1kg_baseline" / "data_est"
RAW_DATA = WORKSPACE_DIR / "joint_kekule_diffusion" / "data_hp1kg_nevpt2"
SOURCE_CONFIGS = BUNDLE_DIR / "configs" / "local" / "guidance_fresh"
OUTPUT_CONFIGS = BUNDLE_DIR / "configs" / "local" / "guidance_est"
RUN_ROOT = WORKSPACE_DIR / "runs" / "molguidance_guidance_est_20261001"

SPLIT_SOURCES = {
    "train_data": "train",
    "train_a_data": "train",
    "train_b_data": "train",
    "val_data": "val",
    "test_data": "test",
}

CONFIG_ROLES = {
    "hp1kg_s1_cfg_fresh.yaml": ("hp1kg_est_cfg_fresh.yaml", "cfg"),
    "hp1kg_s1_mg_fresh.yaml": ("hp1kg_est_mg_fresh.yaml", "mg"),
    "hp1kg_s1_base_fresh.yaml": ("hp1kg_est_base_fresh.yaml", "base_good"),
    "hp1kg_s1_cg_classifier_fresh.yaml": (
        "hp1kg_est_cg_classifier_fresh.yaml",
        "cg_classifier",
    ),
    "hp1kg_s1_inferior_fresh.yaml": (
        "hp1kg_est_inferior_fresh.yaml",
        "ag_inferior",
    ),
}


def canonical_smiles(value: str) -> str:
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError(f"invalid SMILES while validating split alignment: {value}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def validate_and_convert_split(prefix: str, raw_split: str) -> dict[str, float | int]:
    source_path = SOURCE_DATA / f"{prefix}_processed.pt"
    raw_path = RAW_DATA / f"{raw_split}.pt"
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)

    source_properties = torch.as_tensor(source["properties"]).reshape(-1)
    if len(raw) != len(source_properties) or len(source["smiles"]) != len(raw):
        raise ValueError(
            f"split length mismatch for {prefix}: raw={len(raw)}, "
            f"properties={len(source_properties)}, smiles={len(source['smiles'])}"
        )

    raw_s1 = torch.tensor([float(row["s1"]) for row in raw], dtype=torch.float32)
    raw_t1 = torch.tensor([float(row["t1"]) for row in raw], dtype=torch.float32)
    if not torch.allclose(source_properties.float(), raw_s1, atol=1e-6, rtol=0):
        difference = (source_properties.float() - raw_s1).abs().max().item()
        raise ValueError(f"S1 alignment failed for {prefix}; max difference={difference}")

    for index, (processed_smiles, raw_row) in enumerate(zip(source["smiles"], raw)):
        if canonical_smiles(processed_smiles) != canonical_smiles(raw_row["smiles"]):
            raise ValueError(f"SMILES alignment failed for {prefix} at index {index}")

    est = raw_s1 - raw_t1
    converted = dict(source)
    converted["properties"] = est
    converted["property_names"] = "est"
    torch.save(converted, OUTPUT_DATA / f"{prefix}_processed.pt")

    normalization = {
        "mean": est.mean(),
        "std": est.std(),
        "property_names": "est",
    }
    torch.save(normalization, OUTPUT_DATA / f"{prefix}_property_normalization.pt")

    alias_prefix = prefix.removesuffix("_data")
    torch.save(
        normalization,
        OUTPUT_DATA / f"{alias_prefix}_property_normalization.pt",
    )

    return {
        "count": len(est),
        "mean_eV": est.mean().item(),
        "std_eV": est.std().item(),
        "min_eV": est.min().item(),
        "max_eV": est.max().item(),
        "negative_count": int((est < 0).sum().item()),
        "near_zero_abs_le_0p1_count": int((est.abs() <= 0.1).sum().item()),
    }


def build_data() -> dict[str, dict[str, float | int]]:
    if OUTPUT_DATA.exists() and any(OUTPUT_DATA.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {OUTPUT_DATA}")
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)

    excluded_suffixes = ("_processed.pt", "_property_normalization.pt")
    for source in SOURCE_DATA.iterdir():
        if source.is_file() and not source.name.endswith(excluded_suffixes):
            shutil.copy2(source, OUTPUT_DATA / source.name)

    split_stats = {
        prefix: validate_and_convert_split(prefix, raw_split)
        for prefix, raw_split in SPLIT_SOURCES.items()
    }
    report = {
        "schema_version": 1,
        "property": "est",
        "definition": "s1_eV - t1_eV",
        "source_processed_data": str(SOURCE_DATA.resolve()),
        "source_raw_data": str(RAW_DATA.resolve()),
        "split_alignment": "canonical SMILES and all S1 values verified",
        "retains_negative_est": True,
        "splits": split_stats,
    }
    (OUTPUT_DATA / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return split_stats


def build_configs() -> None:
    if OUTPUT_CONFIGS.exists() and any(OUTPUT_CONFIGS.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty config directory: {OUTPUT_CONFIGS}")
    OUTPUT_CONFIGS.mkdir(parents=True, exist_ok=True)

    for source_name, (output_name, role) in CONFIG_ROLES.items():
        source_path = SOURCE_CONFIGS / source_name
        config = yaml.safe_load(source_path.read_text())
        config["dataset"]["conditioning"]["property"] = "est"
        config["dataset"]["processed_data_dir"] = (
            "../molguidance_hp1kg_baseline/data_est"
        )
        config["training"]["output_dir"] = f"../runs/molguidance_guidance_est_20261001/{role}"
        if "wandb" in config:
            config["wandb"]["name"] = str(config["wandb"].get("name", "")).replace(
                "hp1kg-s1-", "hp1kg-est-"
            )
            config["wandb"]["save_dir"] = (
                f"../runs/molguidance_guidance_est_20261001/{role}"
            )
        (OUTPUT_CONFIGS / output_name).write_text(
            yaml.safe_dump(config, sort_keys=False)
        )


def main() -> None:
    for required in (SOURCE_DATA, RAW_DATA, SOURCE_CONFIGS):
        if not required.is_dir():
            raise FileNotFoundError(required)
    if RUN_ROOT.exists():
        raise RuntimeError(f"training output already exists: {RUN_ROOT}")
    stats = build_data()
    build_configs()
    print(json.dumps({"data_dir": str(OUTPUT_DATA), "config_dir": str(OUTPUT_CONFIGS), "splits": stats}, indent=2))


if __name__ == "__main__":
    main()
