"""YAML configuration loading with recursive defaults."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


DEFAULT_CONFIG = {
    "seed": 7,
    "device": "cuda",
    "data": {
        "num_samples": 20_000,
        "min_rings": 2,
        "max_rings": 8,
        "peri_probability": 0.2,
        "templates": [
            "benzene",
            "pyridine",
            "pyrazine",
            "pyrrole",
            "furan",
            "thiophene",
            "borinine",
        ],
    },
    "diffusion": {"timesteps": 200, "beta_start": 1e-4, "beta_end": 2e-2},
    "model": {"hidden_dim": 192, "num_layers": 6, "time_dim": 64, "max_ports": 8},
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 2e-4,
        "weight_decay": 1e-6,
        "grad_clip": 1.0,
        "center_weight": 1.0,
        "normal_weight": 0.25,
        "ring_type_weight": 1.0,
        "connection_weight": 1.0,
        "port_weight": 0.5,
        "checkpoint": "runs/polyring.pt",
    },
    "sampling": {
        "num_rings": 6,
        "num_samples": 16,
        "max_extra_fusions": 2,
        "output": "samples/generated.jsonl",
    },
}


def _merge(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    return _merge(config, payload)

