import random
from pathlib import Path

import torch

from joint_kekule_diffusion.chemistry_v5 import (
    build_hp1kg_codec,
    build_hp1kg_topology_prior,
    decode_heavy_graph,
    hp1kg_records,
    project_heavy_graph,
)


def _oracle_projection(row, atom_map, codec, prior, seed):
    states = codec.encode_record(row, atom_map)
    count = len(states)
    bonds = row["kekule_bonds"].long()
    aromatic = row["aromatic_edges"].bool()
    state_logits = torch.full((count, len(codec)), -12.)
    state_logits[torch.arange(count), states] = 12.
    edge_logits = torch.full((count, count, 2), -8.)
    edge_logits[..., 0] = 8.
    present = bonds > 0
    edge_logits[..., 0][present] = -8.
    edge_logits[..., 1][present] = 8.
    type_logits = torch.full((count, count, 3), -8.)
    kind = (bonds - 1).clamp_min(0)
    i, j = present.nonzero(as_tuple=True)
    type_logits[i, j, kind[i, j]] = 8.
    aromatic_logits = torch.where(aromatic, torch.tensor(8.), torch.tensor(-8.))
    return project_heavy_graph(state_logits, edge_logits, type_logits, aromatic_logits, codec, prior, random.Random(seed))


def test_rare_bridged_hypervalent_and_triple_bond_oracles_are_lossless():
    root = Path("joint_kekule_diffusion/data_hp1kg_nevpt2")
    codec = build_hp1kg_codec(root)
    prior = build_hp1kg_topology_prior(root)
    rows, atom_map = hp1kg_records(root)
    # These previously exposed greedy spanning-tree, ring-system-definition,
    # and sequential single->double->triple upgrade failures.
    for index in (36, 733, 968, 1525):
        row = rows[index]
        projected = _oracle_projection(row, atom_map, codec, prior, 1000 + index)
        assert projected.valid, projected.reason
        assert torch.equal(projected.bonds, row["kekule_bonds"])
        decoded = decode_heavy_graph(projected.state_ids.tolist(), projected.bonds, codec)
        assert decoded.valid, decoded.error
        assert decoded.smiles == row["smiles"]
