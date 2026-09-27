import random
from functools import lru_cache
from pathlib import Path

import torch
from rdkit import Chem

from joint_kekule_diffusion.chemistry_v5 import hp1kg_records
from joint_kekule_diffusion.chemistry_v6 import (
    assess_structure,
    build_hp1kg_chemistry_prior,
    build_hp1kg_codec,
    collate_hp1kg_v5,
    decode_heavy_graph,
    project_heavy_graph_v6,
    support_warnings,
)
from joint_kekule_diffusion.hp1kg_data import HP1KGDataset
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v6 import HeavyAtomJointDDPMV6, joint_training_loss_v6
from joint_kekule_diffusion.sample_v6 import project_aromatic_cycles_in_reverse, synchronize_joint_edge_state


DATA_ROOT = Path("joint_kekule_diffusion/data_hp1kg_nevpt2")


@lru_cache(maxsize=1)
def chemistry():
    codec = build_hp1kg_codec(DATA_ROOT)
    return codec, build_hp1kg_chemistry_prior(DATA_ROOT, codec)


def _oracle_inputs(row, atom_map, codec):
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
    aromatic_logits = torch.full((count, count, 2), -8.)
    aromatic_logits[..., 0] = 8.
    aromatic_logits[..., 0][aromatic] = -8.
    aromatic_logits[..., 1][aromatic] = 8.
    return states, state_logits, edge_logits, type_logits, aromatic_logits


def test_aromatic_edge_is_a_reverse_diffusion_state():
    codec, _ = chemistry()
    model = HeavyAtomJointDDPMV6(codec, hidden=32, max_atoms=8, layers=2, condition_dropout=0).eval()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :6] = True
    noisy = {
        "atom_mask": mask,
        "atom_state": torch.randint(len(codec), (2, 8)),
        "edge_exists": torch.randint(2, (2, 8, 8)),
        "bond_type": torch.randint(3, (2, 8, 8)),
        "aromatic_edge": torch.randint(2, (2, 8, 8)),
        "positions": torch.randn(2, 8, 3),
    }
    output = model(torch.zeros(2, 2), torch.tensor([2, 7]), noisy)
    assert output["aromatic_edge_logits"].shape == (2, 8, 8, 2)
    assert model.aromatic_embedding.num_embeddings == 2

    pair = mask[:, :, None] & mask[:, None, :]
    diagonal = torch.eye(8, dtype=torch.bool)[None]
    state = {key: value.clone() for key, value in noisy.items()}
    state["edge_exists"].zero_()
    state["aromatic_edge"].fill_(1)
    synchronize_joint_edge_state(state, output, pair, diagonal)
    assert not state["aromatic_edge"].any()


def test_reverse_cycle_projection_builds_one_global_kekule_matching():
    codec, _ = chemistry()
    model = HeavyAtomJointDDPMV6(codec, hidden=32, max_atoms=6, layers=1, condition_dropout=0).eval()
    aromatic_carbon = next(i for i, item in enumerate(codec.states) if item.symbol == "C" and item.aromatic_role == 1)
    edge = torch.zeros(1, 6, 6, dtype=torch.long)
    for i in range(6):
        j = (i + 1) % 6
        edge[0, i, j] = edge[0, j, i] = 1
    state = {
        "atom_state": torch.full((1, 6), aromatic_carbon),
        "edge_exists": edge,
        "bond_type": torch.zeros(1, 6, 6, dtype=torch.long),
        "aromatic_edge": torch.zeros(1, 6, 6, dtype=torch.long),
    }
    aromatic_logits = torch.zeros(1, 6, 6, 2)
    aromatic_logits[..., 1] = 8
    bond_logits = torch.zeros(1, 6, 6, 3)
    bond_logits[..., 1] = 4
    output = {"aromatic_edge_logits": aromatic_logits, "bond_type_logits": bond_logits}
    project_aromatic_cycles_in_reverse(state, output, model, torch.tensor([6]))
    assert int(state["aromatic_edge"].sum() // 2) == 6
    assert int((state["bond_type"] == 1).sum() // 2) == 3


def test_v6_loss_and_gradients_are_finite_on_real_batch():
    codec, _ = chemistry()
    dataset = HP1KGDataset(DATA_ROOT, "train")
    batch = collate_hp1kg_v5([dataset[0], dataset[1]], max_atoms=69, codec=codec, atom_map=dataset.atom_map)
    model = HeavyAtomJointDDPMV6(codec, hidden=32, max_atoms=69, layers=2, condition_dropout=0)
    diffusion = JointDiffusion(
        timesteps=20,
        marginals={"atom": torch.ones(len(codec)), "edge": [1., 1.], "bond_type": [1., 1., 1.], "aromatic_edge": [1., 1.]},
    )
    loss, terms = joint_training_loss_v6(model, diffusion, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    for required in ("aromatic_edge", "vlb_aromatic_edge", "small_ring_prior", "bridgehead_prior", "cumulene_prior"):
        assert required in terms
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_all_db_records_are_inside_exact_chemistry_support():
    codec, prior = chemistry()
    rows, atom_map = hp1kg_records(DATA_ROOT)
    for row in rows:
        states = codec.encode_record(row, atom_map).tolist()
        assert not support_warnings(states, row["kekule_bonds"], row["aromatic_edges"], prior)
    restored = type(prior).from_metadata(prior.to_metadata())
    assert restored.bond_signatures == prior.bond_signatures
    assert restored.local_signatures == prior.local_signatures


def test_rare_oracle_graphs_remain_lossless_under_v6_support():
    codec, prior = chemistry()
    rows, atom_map = hp1kg_records(DATA_ROOT)
    for index in (36, 733, 968, 1525):
        row = rows[index]
        states, state_logits, edge_logits, type_logits, aromatic_logits = _oracle_inputs(row, atom_map, codec)
        projected = project_heavy_graph_v6(
            state_logits, edge_logits, type_logits, aromatic_logits, codec, prior,
            random.Random(1000 + index), enforce_support=True, apply_topology_prior=False,
        )
        assert projected.valid, projected.reason
        assert torch.equal(projected.bonds, row["kekule_bonds"])
        decoded = decode_heavy_graph(projected.state_ids.tolist(), projected.bonds, codec)
        assert decoded.valid, decoded.error
        assert decoded.smiles == row["smiles"]


def test_structure_assessment_separates_clean_and_warned_topology():
    benzene = Chem.MolFromSmiles("c1ccccc1")
    clean = assess_structure(benzene, evaluate_3d=True, seed=7)
    assert clean.etkdg_success
    assert clean.strain_energy_per_heavy_atom is not None
    assert clean.chemically_clean

    cyclopropane = Chem.MolFromSmiles("C1CC1")
    warned = assess_structure(cyclopropane, evaluate_3d=False)
    assert warned.small_rings == 1
    assert not warned.chemically_clean
    assert "three_or_four_member_ring" in warned.warnings
