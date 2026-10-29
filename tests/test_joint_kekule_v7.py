from __future__ import annotations

import inspect
import json
from functools import partial
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import DataLoader

from joint_kekule_diffusion.chemistry_v5 import decode_heavy_graph, hp1kg_records
from joint_kekule_diffusion.chemistry_v7 import (
    HP1KGV7Dataset,
    assess_generated_coordinates,
    build_train_codec_v7,
    build_v7_references,
    collate_hp1kg_v7,
    dataset_fingerprint,
    metadata_fingerprint,
    project_joint_graph_v7,
    support_warnings_v7,
    source_fingerprints_v7,
)
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v7 import (
    JOINT_EDGE_CLASSES,
    HeavyAtomJointDDPMV7,
    _adjacency_topology_surrogates,
    _expected_small_cycles,
    decode_joint_edge_state,
    encode_joint_edge_state,
    joint_training_loss_v7,
)
from joint_kekule_diffusion.sample_v7 import sample_batch_v7, validate_v7_artifacts
from joint_kekule_diffusion.train_v7 import compute_marginals_v7


ROOT = Path("joint_kekule_diffusion/data_hp1kg_nevpt2")


@pytest.fixture(scope="module")
def codec():
    return build_train_codec_v7(ROOT)


@pytest.fixture(scope="module")
def records():
    return hp1kg_records(ROOT)


def oracle_logits(row, atom_map, codec):
    state = codec.encode_record(row, atom_map)
    edge = encode_joint_edge_state(row["kekule_bonds"], row["aromatic_edges"])
    count = len(state)
    state_logits = torch.full((count, len(codec)), -20.)
    state_logits[torch.arange(count), state] = 20.
    edge_logits = torch.full((count, count, JOINT_EDGE_CLASSES), -20.)
    edge_logits.scatter_(-1, edge[..., None], 20.)
    return state, edge, state_logits, edge_logits


def test_train_only_codec_covers_heldout_without_learning_from_it(codec, records):
    all_rows, atom_map = records
    train_rows, _ = hp1kg_records(ROOT, ("train",))
    reconstructed = build_train_codec_v7(ROOT)
    assert reconstructed.to_metadata() == codec.to_metadata()
    assert len(train_rows) < len(all_rows)
    for row in all_rows:
        assert len(codec.encode_record(row, atom_map)) == len(row["atom_types"])


def test_joint_edge_state_is_lossless_including_aromatic_triple(records):
    rows, _ = records
    saw_aromatic_triple = False
    for row in rows:
        edge = encode_joint_edge_state(row["kekule_bonds"], row["aromatic_edges"])
        bonds, aromatic = decode_joint_edge_state(edge)
        assert torch.equal(bonds, torch.as_tensor(row["kekule_bonds"]).long())
        assert torch.equal(aromatic, torch.as_tensor(row["aromatic_edges"]).bool())
        saw_aromatic_triple |= bool((edge == 6).any())
    assert saw_aromatic_triple


def test_full_database_oracle_rdkit_identity_100_percent(codec, records):
    rows, atom_map = records
    failures = []
    for index, row in enumerate(rows):
        _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
        projected = project_joint_graph_v7(state_logits, edge_logits, codec)
        if not projected.valid or projected.solver_used:
            failures.append((index, row["smiles"], projected.reason))
            continue
        decoded = decode_heavy_graph(projected.state_ids.tolist(), projected.bonds, codec)
        expected = Chem.MolToSmiles(
            Chem.MolFromSmiles(row["smiles"]), canonical=True, isomericSmiles=False,
        )
        if not decoded.valid or decoded.smiles != expected:
            failures.append((index, row["smiles"], decoded.smiles))
    assert not failures
    assert len(rows) == 1581


def test_soft_four_cycle_expectation_is_probability_product():
    adjacency = torch.zeros(1, 4, 4)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 0)):
        adjacency[0, left, right] = adjacency[0, right, left] = .5
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert float(_expected_small_cycles(adjacency, mask)) == pytest.approx(.5 ** 4, abs=1e-7)


def test_topology_targets_come_from_rdkit_not_degree_heuristic():
    dataset = HP1KGV7Dataset(ROOT, "train")
    for row in dataset.rows[:64]:
        mol = Chem.MolFromSmiles(row["smiles"])
        graph = __import__("networkx").Graph()
        graph.add_nodes_from(range(mol.GetNumAtoms()))
        graph.add_edges_from((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
        max_ring_system = max(
            (len(component) for component in __import__("networkx").biconnected_components(graph)
             if graph.subgraph(component).number_of_edges() >= len(component)),
            default=0,
        )
        expected = torch.tensor([
            mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol)),
            max_ring_system,
            sum(len(ring) in (3, 4) for ring in mol.GetRingInfo().AtomRings()),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            sum(sum(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()) >= 2 for atom in mol.GetAtoms()),
        ], dtype=torch.float32)
        assert torch.equal(row["topology_target"], expected)


def test_adjacency_topology_surrogates_have_edge_gradients():
    presence = torch.full((1, 6, 6), .02, requires_grad=True)
    mask = torch.ones(1, 6, dtype=torch.bool)
    cycle, system, bridge = _adjacency_topology_surrogates(presence, mask)
    (cycle + system + bridge).sum().backward()
    assert presence.grad is not None and torch.isfinite(presence.grad).all()
    assert float(presence.grad.abs().sum()) > 0


def test_cfg_dropout_and_regular_dropout_are_distinct(codec):
    model = HeavyAtomJointDDPMV7(
        codec, hidden=32, max_atoms=6, layers=2,
        condition_dropout=0., model_dropout=.4, count_dropout=.5,
    )
    assert not hasattr(model, "topology_head")
    assert isinstance(model.node_dropout, torch.nn.Dropout)
    mask = torch.ones(1, 6, dtype=torch.bool)
    state = {
        "atom_mask": mask,
        "atom_state": torch.randint(len(codec), (1, 6)),
        "edge_state": torch.zeros(1, 6, 6, dtype=torch.long),
        "positions": torch.randn(1, 6, 3),
    }
    condition, timestep = torch.randn(1, 2), torch.tensor([5])
    model.train()
    torch.manual_seed(1)
    first = model(condition, timestep, state)
    torch.manual_seed(2)
    second = model(condition, timestep, state)
    assert not torch.allclose(first["state_logits"], second["state_logits"])
    assert not torch.allclose(first["count_logits"], second["count_logits"])
    model.eval()
    torch.manual_seed(1)
    first = model(condition, timestep, state)
    torch.manual_seed(2)
    second = model(condition, timestep, state)
    assert torch.allclose(first["state_logits"], second["state_logits"])
    assert torch.allclose(first["count_logits"], second["count_logits"])


def test_final_joint_solver_repairs_connectivity(codec, records):
    row = next(row for row in records[0] if len(row["atom_types"]) >= 4)
    _, edge, state_logits, edge_logits = oracle_logits(row, records[1], codec)
    true_pairs = torch.triu(edge > 0, 1).nonzero()
    left, right = true_pairs[0].tolist()
    edge_logits[left, right] = edge_logits[right, left] = -8.
    edge_logits[left, right, 0] = edge_logits[right, left, 0] = 8.
    result = project_joint_graph_v7(
        state_logits, edge_logits, codec, solver_time_limit=5., solver_attempts=2,
    )
    assert result.valid
    decoded = decode_heavy_graph(result.state_ids.tolist(), result.bonds, codec)
    assert decoded.valid and decoded.connected
    assert result.solver_used


def test_support_is_metric_not_projection_filter(codec, records):
    chemistry, _, _ = build_v7_references(ROOT, codec)
    rows, atom_map = records
    found = False
    for row in rows[len(hp1kg_records(ROOT, ("train",))[0]):]:
        state = codec.encode_record(row, atom_map)
        warnings = support_warnings_v7(
            state.tolist(), row["kekule_bonds"], row["aromatic_edges"], chemistry,
        )
        if warnings:
            _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
            result = project_joint_graph_v7(state_logits, edge_logits, codec)
            assert result.valid
            found = True
            break
    assert found, "held-out set unexpectedly contains no unseen train support"


def test_generated_coordinate_metric_uses_given_coordinates(codec, records):
    _, geometry, _ = build_v7_references(ROOT, codec)
    row = records[0][0]
    state = codec.encode_record(row, records[1]).tolist()
    decoded = decode_heavy_graph(state, row["kekule_bonds"], codec)
    good = assess_generated_coordinates(
        decoded.molecule, state, row["kekule_bonds"], row["aromatic_edges"],
        row["atom_positions"], codec, geometry, optimize=False,
    )
    distorted = torch.as_tensor(row["atom_positions"]).clone() * 10
    bad = assess_generated_coordinates(
        decoded.molecule, state, row["kekule_bonds"], row["aromatic_edges"],
        distorted, codec, geometry, optimize=False,
    )
    assert good.finite and bad.finite
    assert bad.bond_z_rms > good.bond_z_rms
    assert bad.bond_outliers > 0


def test_padding_permutation_and_rotation_equivariance(codec):
    torch.manual_seed(7)
    model = HeavyAtomJointDDPMV7(codec, hidden=32, max_atoms=8, layers=2).eval()
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    state = {
        "atom_mask": mask,
        "atom_state": torch.randint(len(codec), (1, 8)),
        "edge_state": torch.randint(JOINT_EDGE_CLASSES, (1, 8, 8)),
        "positions": torch.randn(1, 8, 3) * mask[..., None],
    }
    upper_edge = torch.triu(state["edge_state"], 1)
    state["edge_state"] = upper_edge + upper_edge.transpose(1, 2)
    condition, timestep = torch.randn(1, 2), torch.tensor([17])
    base = model(condition, timestep, state)

    altered = {key: value.clone() for key, value in state.items()}
    altered["atom_state"][0, 4:] = torch.randint(len(codec), (4,))
    altered["edge_state"][0, 4:, :] = torch.randint(JOINT_EDGE_CLASSES, (4, 8))
    altered["edge_state"][0, :, 4:] = torch.randint(JOINT_EDGE_CLASSES, (8, 4))
    altered["positions"][0, 4:] = torch.randn(4, 3) * 100
    padded = model(condition, timestep, altered)
    assert torch.allclose(base["state_logits"][:, :4], padded["state_logits"][:, :4], atol=1e-6)

    permutation = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7])
    permuted = {
        "atom_mask": state["atom_mask"][:, permutation],
        "atom_state": state["atom_state"][:, permutation],
        "edge_state": state["edge_state"][:, permutation][:, :, permutation],
        "positions": state["positions"][:, permutation],
    }
    permuted_output = model(condition, timestep, permuted)
    assert torch.allclose(
        base["state_logits"][:, permutation], permuted_output["state_logits"], atol=2e-5,
    )
    assert torch.allclose(
        base["joint_edge_logits"][:, permutation][:, :, permutation],
        permuted_output["joint_edge_logits"], atol=2e-5,
    )

    rotation = torch.linalg.qr(torch.randn(3, 3)).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1
    rotated = {**state, "positions": state["positions"] @ rotation}
    rotated_output = model(condition, timestep, rotated)
    assert torch.allclose(
        base["coordinate_eps"] @ rotation, rotated_output["coordinate_eps"], atol=2e-5,
    )


def test_loss_gradient_and_marginals_are_finite(codec):
    dataset = HP1KGV7Dataset(ROOT, "train")
    collate = partial(collate_hp1kg_v7, max_atoms=69, codec=codec, atom_map=dataset.atom_map)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))
    marginals = compute_marginals_v7(dataset, codec)
    assert len(marginals["joint_edge"]) == JOINT_EDGE_CLASSES
    assert sum(marginals["joint_edge"]) == pytest.approx(1., abs=1e-6)
    model = HeavyAtomJointDDPMV7(codec, hidden=32, max_atoms=69, layers=2)
    diffusion = JointDiffusion(timesteps=10, marginals=marginals)
    loss, terms = joint_training_loss_v7(model, diffusion, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_reverse_sampler_calls_projection_only_once_at_end(codec):
    model = HeavyAtomJointDDPMV7(codec, hidden=32, max_atoms=6, layers=1).eval()
    diffusion = JointDiffusion(
        timesteps=3,
        marginals={"atom": [1 / len(codec)] * len(codec), "joint_edge": [1 / 7] * 7},
    )
    import joint_kekule_diffusion.sample_v7 as module
    original = module.project_joint_graph_v7
    with mock.patch.object(module, "project_joint_graph_v7", wraps=original) as wrapped:
        output = sample_batch_v7(
            model, diffusion, torch.zeros(1, 2), 1, atoms=4,
            solver_time_limit=.5, solver_attempts=1,
        )
    assert wrapped.call_count == 1
    assert output["raw_edge_state"].shape == (1, 6, 6)
    source = inspect.getsource(sample_batch_v7)
    assert "project_aromatic_cycles_in_reverse" not in source
    assert "synchronize_joint_edge_state" not in source


def test_hash_validation_detects_metadata_checkpoint_and_db_mismatch(tmp_path):
    dataset_hash = dataset_fingerprint(ROOT)
    metadata = {"model_kind": "heavy_atom_joint_edge_d3pm_v7", "dataset_hash": dataset_hash, "source_hashes": source_fingerprints_v7()}
    metadata["metadata_hash"] = metadata_fingerprint(metadata)
    checkpoint = {"metadata_hash": metadata["metadata_hash"], "metadata": dict(metadata)}
    assert validate_v7_artifacts(metadata, checkpoint, ROOT) == metadata["metadata_hash"]
    corrupted = dict(metadata)
    corrupted["max_atoms"] = 123
    with pytest.raises(ValueError, match="metadata hash"):
        validate_v7_artifacts(corrupted, checkpoint, ROOT)
    bad_checkpoint = dict(checkpoint)
    bad_checkpoint["metadata_hash"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        validate_v7_artifacts(metadata, bad_checkpoint, ROOT)


def test_condition_surrogate_is_train_fit_and_reports_heldout_metrics(codec):
    _, _, surrogate = build_v7_references(ROOT, codec)
    metrics = surrogate.validation_metrics
    assert metrics["samples"] == len(hp1kg_records(ROOT, ("val",))[0])
    assert np.isfinite(metrics["s1_mae"]) and np.isfinite(metrics["t1_mae"])
    molecule = Chem.MolFromSmiles(records_smiles := hp1kg_records(ROOT, ("test",))[0][0]["smiles"])
    assert records_smiles
    prediction = surrogate.predict(molecule)
    assert prediction.shape == (2,) and np.isfinite(prediction).all()

