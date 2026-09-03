"""Train the ring-level diffusion prototype."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .data import SyntheticDataConfig, SyntheticRingDataset
from .diffusion import LinearNoiseSchedule
from .graph import ConnectionType
from .model import RingDenoiser


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    return selected.mean() if selected.numel() else values.new_zeros(())


def train(config: dict) -> Path:
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(config["device"])

    data_config = SyntheticDataConfig(
        num_samples=int(config["data"]["num_samples"]),
        min_rings=int(config["data"]["min_rings"]),
        max_rings=int(config["data"]["max_rings"]),
        peri_probability=float(config["data"]["peri_probability"]),
        templates=tuple(config["data"]["templates"]),
    )
    dataset = SyntheticRingDataset(data_config, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    model = RingDenoiser(**config["model"]).to(device)
    schedule = LinearNoiseSchedule(**config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    epochs = int(config["training"]["epochs"])
    for epoch in range(epochs):
        model.train()
        totals: dict[str, float] = {key: 0.0 for key in ("loss", "center", "ring", "conn", "port")}
        batches = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            node_mask = batch["node_mask"]
            pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
            diagonal = torch.eye(pair_mask.shape[1], dtype=torch.bool, device=device)[None]
            pair_mask &= ~diagonal
            upper_mask = pair_mask & torch.triu(
                torch.ones(pair_mask.shape[1:], dtype=torch.bool, device=device), diagonal=1
            )[None]
            timesteps = torch.randint(
                schedule.timesteps, (node_mask.shape[0],), device=device, dtype=torch.long
            )
            noisy_centers, center_noise = schedule.q_sample(batch["centers"], timesteps)
            noisy_normals, normal_noise = schedule.q_sample(batch["normals"], timesteps)
            noisy_ring_types = schedule.corrupt_tokens(
                batch["ring_types"], timesteps, model.ring_mask_token, node_mask
            )
            noisy_connections = schedule.corrupt_tokens(
                batch["connection_types"],
                timesteps,
                model.connection_mask_token,
                pair_mask,
            )
            predictions = model(
                noisy_ring_types,
                noisy_centers,
                noisy_normals,
                noisy_connections,
                node_mask,
                timesteps,
            )

            center_loss = _masked_mean(
                (predictions["center_noise"] - center_noise).square().sum(dim=-1), node_mask
            )
            normal_loss = _masked_mean(
                (predictions["normal_noise"] - normal_noise).square().sum(dim=-1), node_mask
            )
            ring_loss = F.cross_entropy(
                predictions["ring_logits"][node_mask], batch["ring_types"][node_mask]
            )
            connection_loss = F.cross_entropy(
                predictions["connection_logits"][upper_mask],
                batch["connection_types"][upper_mask],
            )
            materialized = pair_mask & batch["connection_types"].ne(int(ConnectionType.NONE))
            port_loss = F.cross_entropy(
                predictions["port_logits"][materialized], batch["ports"][materialized]
            )
            weights = config["training"]
            loss = (
                float(weights["center_weight"]) * center_loss
                + float(weights["normal_weight"]) * normal_loss
                + float(weights["ring_type_weight"]) * ring_loss
                + float(weights["connection_weight"]) * connection_loss
                + float(weights["port_weight"]) * port_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(weights["grad_clip"]))
            optimizer.step()

            totals["loss"] += float(loss.detach())
            totals["center"] += float(center_loss.detach())
            totals["ring"] += float(ring_loss.detach())
            totals["conn"] += float(connection_loss.detach())
            totals["port"] += float(port_loss.detach())
            batches += 1
        summary = " ".join(f"{key}={value / max(batches, 1):.4f}" for key, value in totals.items())
        print(f"epoch={epoch + 1:03d} {summary}")

    checkpoint = Path(config["training"]["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "template_names": model.template_names,
        },
        checkpoint,
    )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.example.yaml")
    args = parser.parse_args()
    checkpoint = train(load_config(args.config))
    print(f"saved checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

