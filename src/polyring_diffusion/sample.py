"""Sample ring graphs, project them, and deterministically decode atom graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import load_config
from .decoder import DecodeError, decode_ring_graph
from .diffusion import LinearNoiseSchedule, center_coordinates, normalize_vectors
from .graph import ConnectionType, RingConnection, RingNode
from .model import RingDenoiser
from .projector import ConnectionCandidate, RingConstraintProjector
from .templates import DEFAULT_LIBRARY


def _sample_token(logits: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).sample()


def _reveal_mask(masked: torch.Tensor, timestep: int, generator_shape: tuple[int, ...]) -> torch.Tensor:
    probability = 1.0 if timestep == 0 else 1.0 / float(timestep + 1)
    return masked & (torch.rand(generator_shape, device=masked.device) < probability)


def _connection_candidates(
    nodes: list[RingNode],
    connection_logits: torch.Tensor,
    port_logits: torch.Tensor,
) -> list[ConnectionCandidate]:
    candidates: list[ConnectionCandidate] = []
    connection_logp = F.log_softmax(connection_logits, dim=-1)
    port_logp = F.log_softmax(port_logits, dim=-1)
    for left in range(len(nodes)):
        left_template = DEFAULT_LIBRARY.get(nodes[left].template)
        for right in range(left + 1, len(nodes)):
            right_template = DEFAULT_LIBRARY.get(nodes[right].template)
            for kind in (ConnectionType.SINGLE, ConnectionType.SPIRO):
                for lp in left_template.vertex_ports:
                    for rp in right_template.vertex_ports:
                        if left_template.atoms[lp] != "C" or right_template.atoms[rp] != "C":
                            continue
                        if kind is ConnectionType.SPIRO and (
                            left_template.charges[lp] != right_template.charges[rp]
                        ):
                            continue
                        score = (
                            connection_logp[left, right, int(kind)]
                            + port_logp[left, right, lp]
                            + port_logp[right, left, rp]
                        )
                        candidates.append(
                            ConnectionCandidate(
                                float(score), RingConnection(left, right, kind, lp, rp)
                            )
                        )
            compatible = left_template.compatible_edge_ports(right_template)
            for lp, rp in compatible:
                score = (
                    connection_logp[left, right, int(ConnectionType.FUSED)]
                    + port_logp[left, right, lp]
                    + port_logp[right, left, rp]
                )
                candidates.append(
                    ConnectionCandidate(
                        float(score),
                        RingConnection(left, right, ConnectionType.FUSED, lp, rp),
                    )
                )
    return candidates


@torch.no_grad()
def sample(config: dict, checkpoint_path: str | Path | None = None) -> list[dict]:
    requested = config["device"]
    device = torch.device(requested if not requested.startswith("cuda") or torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_path or config["training"]["checkpoint"]
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trained = payload["config"]
    model = RingDenoiser(**trained["model"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    schedule = LinearNoiseSchedule(**trained["diffusion"]).to(device)

    batch = int(config["sampling"]["num_samples"])
    count = int(config["sampling"]["num_rings"])
    maximum = int(trained["data"]["max_rings"])
    if count < 1 or count > maximum:
        raise ValueError(f"num_rings must be in [1, {maximum}]")
    node_mask = torch.zeros(batch, maximum, dtype=torch.bool, device=device)
    node_mask[:, :count] = True
    ring_types = torch.full(
        (batch, maximum), model.ring_pad_token, dtype=torch.long, device=device
    )
    ring_types[node_mask] = model.ring_mask_token
    connections = torch.zeros(batch, maximum, maximum, dtype=torch.long, device=device)
    pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
    diagonal = torch.eye(maximum, dtype=torch.bool, device=device)[None]
    pair_mask &= ~diagonal
    connections[pair_mask] = model.connection_mask_token
    centers = center_coordinates(torch.randn(batch, maximum, 3, device=device), node_mask)
    normals = normalize_vectors(torch.randn(batch, maximum, 3, device=device), node_mask)

    final_predictions: dict[str, torch.Tensor] | None = None
    for step in reversed(range(schedule.timesteps)):
        timesteps = torch.full((batch,), step, dtype=torch.long, device=device)
        final_predictions = model(ring_types, centers, normals, connections, node_mask, timesteps)
        centers = schedule.p_step(centers, final_predictions["center_noise"], timesteps)
        centers = center_coordinates(centers, node_mask)
        normals = schedule.p_step(normals, final_predictions["normal_noise"], timesteps)
        normals = normalize_vectors(normals, node_mask)

        ring_reveal = _reveal_mask(ring_types.eq(model.ring_mask_token), step, ring_types.shape)
        ring_samples = _sample_token(final_predictions["ring_logits"])
        ring_types = torch.where(ring_reveal & node_mask, ring_samples, ring_types)

        upper = pair_mask & torch.triu(
            torch.ones(maximum, maximum, dtype=torch.bool, device=device), diagonal=1
        )[None]
        reveal = _reveal_mask(connections.eq(model.connection_mask_token) & upper, step, connections.shape)
        connection_samples = _sample_token(final_predictions["connection_logits"])
        connections = torch.where(reveal, connection_samples, connections)
        connections = torch.where(upper, connections, connections.transpose(1, 2))
        connections[:, torch.arange(maximum), torch.arange(maximum)] = int(ConnectionType.NONE)

    ring_types[node_mask & ring_types.eq(model.ring_mask_token)] = 0
    # The last reveal can change categorical inputs. Re-evaluate at the clean
    # endpoint so projection uses connection/port logits for the final tokens.
    final_predictions = model(
        ring_types,
        centers,
        normals,
        connections,
        node_mask,
        torch.zeros(batch, dtype=torch.long, device=device),
    )
    projector = RingConstraintProjector(
        max_extra_fusions=int(config["sampling"]["max_extra_fusions"])
    )
    outputs: list[dict] = []
    for sample_index in range(batch):
        nodes = [
            RingNode(
                model.template_names[int(ring_types[sample_index, index])],
                center=tuple(float(value) for value in centers[sample_index, index]),
                normal=tuple(float(value) for value in normals[sample_index, index]),
            )
            for index in range(count)
        ]
        candidates = _connection_candidates(
            nodes,
            final_predictions["connection_logits"][sample_index, :count, :count],
            final_predictions["port_logits"][sample_index, :count, :count],
        )
        try:
            ring_graph = projector.project(nodes, candidates, require_connected=True)
            atom_graph = decode_ring_graph(ring_graph)
            outputs.append(
                {
                    "valid": True,
                    "rings": [
                        {"template": node.template, "center": node.center, "normal": node.normal}
                        for node in ring_graph.nodes
                    ],
                    "connections": [
                        {
                            "left": edge.left,
                            "right": edge.right,
                            "kind": edge.kind.name,
                            "left_port": edge.left_port,
                            "right_port": edge.right_port,
                        }
                        for edge in ring_graph.connections
                    ],
                    "atom_graph": atom_graph.to_dict(),
                }
            )
        except DecodeError as exc:
            outputs.append({"valid": False, "error": str(exc)})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    outputs = sample(config, args.checkpoint)
    output_path = Path(config["sampling"]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in outputs:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    valid = sum(bool(record.get("valid")) for record in outputs)
    print(f"wrote {len(outputs)} samples to {output_path}; projected-valid={valid}/{len(outputs)}")


if __name__ == "__main__":
    main()
