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
from joint_kekule_diffusion.chemistry_v8 import (
    HP1KGV8Dataset,
    ChemistryReferenceV8,
    assess_generated_coordinates,
    build_train_codec_v8,
    build_v8_references,
    collate_hp1kg_v8,
    dataset_fingerprint,
    metadata_fingerprint,
    runtime_versions_v8,
    project_joint_graph_v8,
    support_warnings_v8,
    ring_support_warnings_v8,
    _cyclic_bond_signatures,
    _minimum_cycle_sizes,
    _ring_edge_targets,
    _ring_topology_from_graph,
    _solve_joint_milp,
    source_fingerprints_v8,
)
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v8 import (
    JOINT_EDGE_CLASSES,
    RING_SIZE_CLASSES,
    HeavyAtomJointDDPMV8,
    decode_joint_edge_state,
    encode_joint_edge_state,
    aromatic_system_loss_v8,
    joint_training_loss_v8,
    ring_size_to_class,
)
from joint_kekule_diffusion.sample_v8 import sample_batch_v8, summarize_records_v8, validate_v8_artifacts
from joint_kekule_diffusion.train_v8 import compute_marginals_v8


ROOT = Path("joint_kekule_diffusion/data_hp1kg_nevpt2")


@pytest.fixture(scope="module")
def codec():
    return build_train_codec_v8(ROOT)


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
    reconstructed = build_train_codec_v8(ROOT)
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
        projected = project_joint_graph_v8(state_logits, edge_logits, codec)
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


def test_topology_targets_are_stored_graph_invariants():
    dataset = HP1KGV8Dataset(ROOT, "train")
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
        bonds = torch.as_tensor(row["kekule_bonds"]).long()
        aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
        topology = _ring_topology_from_graph(bonds, aromatic)
        expected = torch.tensor([
            topology.closure_count,
            topology.max_ring_system,
            sum(size in (3, 4) for size in topology.ring_sizes),
            rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
            sum(sum(b.GetBondTypeAsDouble() == 2 for b in atom.GetBonds()) >= 2 for atom in mol.GetAtoms()),
        ], dtype=torch.float32)
        assert torch.equal(row["topology_target"], expected)
        assert not (row["ring_closure_target"] & ~(bonds > 0)).any()
        assert not (aromatic & ~row["ring_closure_target"]).any()


def test_cfg_dropout_and_regular_dropout_are_distinct(codec):
    model = HeavyAtomJointDDPMV8(
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
    result = project_joint_graph_v8(
        state_logits, edge_logits, codec, solver_time_limit=5., solver_attempts=2,
    )
    assert result.valid
    decoded = decode_heavy_graph(result.state_ids.tolist(), result.bonds, codec)
    assert decoded.valid and decoded.connected
    assert result.solver_used


def test_support_is_metric_not_projection_filter(codec, records):
    chemistry, _, _ = build_v8_references(ROOT, codec)
    rows, atom_map = records
    found = False
    for row in rows[len(hp1kg_records(ROOT, ("train",))[0]):]:
        state = codec.encode_record(row, atom_map)
        warnings = support_warnings_v8(
            state.tolist(), row["kekule_bonds"], row["aromatic_edges"], chemistry,
        )
        if warnings:
            _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
            result = project_joint_graph_v8(state_logits, edge_logits, codec)
            assert result.valid
            found = True
            break
    assert found, "held-out set unexpectedly contains no unseen train support"


def test_generated_coordinate_metric_uses_given_coordinates(codec, records):
    _, geometry, _ = build_v8_references(ROOT, codec)
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
    model = HeavyAtomJointDDPMV8(codec, hidden=32, max_atoms=8, layers=2).eval()
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
    dataset = HP1KGV8Dataset(ROOT, "train")
    collate = partial(collate_hp1kg_v8, max_atoms=69, codec=codec, atom_map=dataset.atom_map)
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))
    marginals = compute_marginals_v8(dataset, codec)
    assert len(marginals["joint_edge"]) == JOINT_EDGE_CLASSES
    assert sum(marginals["joint_edge"]) == pytest.approx(1., abs=1e-6)
    model = HeavyAtomJointDDPMV8(codec, hidden=32, max_atoms=69, layers=2)
    diffusion = JointDiffusion(timesteps=10, marginals=marginals)
    loss, terms = joint_training_loss_v8(model, diffusion, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    assert float(loss) < 100
    assert max(float(value) for value in terms.values()) / float(loss) < .75
    loss.backward()
    assert model.ring_closure_head.weight.grad is not None
    assert model.ring_size_head.weight.grad is not None
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_reverse_sampler_calls_projection_only_once_at_end(codec):
    model = HeavyAtomJointDDPMV8(codec, hidden=32, max_atoms=6, layers=1).eval()
    diffusion = JointDiffusion(
        timesteps=3, beta_start=.96, beta_end=.96,
        marginals={"atom": [1 / len(codec)] * len(codec), "joint_edge": [1 / 7] * 7},
    )
    import joint_kekule_diffusion.sample_v8 as module
    original = module.project_joint_graph_v8
    with mock.patch.object(module, "project_joint_graph_v8", wraps=original) as wrapped:
        output = sample_batch_v8(
            model, diffusion, torch.zeros(1, 2), 1, atoms=4,
            solver_time_limit=.5, solver_attempts=1,
        )
    assert wrapped.call_count == 1
    assert output["raw_edge_state"].shape == (1, 6, 6)
    source = inspect.getsource(sample_batch_v8)
    assert "project_aromatic_cycles_in_reverse" not in source
    assert "synchronize_joint_edge_state" not in source


def test_hash_validation_detects_metadata_checkpoint_and_db_mismatch(tmp_path):
    dataset_hash = dataset_fingerprint(ROOT)
    metadata = {
        "model_kind": "heavy_atom_joint_edge_d3pm_v8",
        "dataset_hash": dataset_hash,
        "source_hashes": source_fingerprints_v8(),
        "runtime_versions": runtime_versions_v8(),
    }
    metadata["metadata_hash"] = metadata_fingerprint(metadata)
    checkpoint = {"metadata_hash": metadata["metadata_hash"], "metadata": dict(metadata)}
    assert validate_v8_artifacts(metadata, checkpoint, ROOT) == metadata["metadata_hash"]
    corrupted = dict(metadata)
    corrupted["max_atoms"] = 123
    with pytest.raises(ValueError, match="metadata hash"):
        validate_v8_artifacts(corrupted, checkpoint, ROOT)
    bad_checkpoint = dict(checkpoint)
    bad_checkpoint["metadata_hash"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        validate_v8_artifacts(metadata, bad_checkpoint, ROOT)


def test_condition_surrogate_is_train_fit_and_reports_heldout_metrics(codec):
    _, _, surrogate = build_v8_references(ROOT, codec)
    metrics = surrogate.validation_metrics
    assert metrics["samples"] == len(hp1kg_records(ROOT, ("val",))[0])
    assert np.isfinite(metrics["s1_mae"]) and np.isfinite(metrics["t1_mae"])
    molecule = Chem.MolFromSmiles(records_smiles := hp1kg_records(ROOT, ("test",))[0][0]["smiles"])
    assert records_smiles
    prediction = surrogate.predict(molecule)
    assert prediction.shape == (2,) and np.isfinite(prediction).all()



def test_ring_closure_targets_are_permutation_invariant(records):
    row = next(
        row for row in records[0]
        if Chem.MolFromSmiles(row["smiles"]).GetRingInfo().NumRings() >= 3
    )
    molecule = Chem.MolFromSmiles(row["smiles"])
    permutation = torch.randperm(molecule.GetNumAtoms())
    permuted = Chem.RenumberAtoms(molecule, permutation.tolist())
    closure, size, histogram = _ring_edge_targets(molecule)
    other_closure, other_size, other_histogram = _ring_edge_targets(permuted)
    inverse = torch.argsort(permutation)
    assert torch.equal(closure, other_closure[inverse][:, inverse])
    assert torch.equal(size, other_size[inverse][:, inverse])
    assert torch.equal(histogram, other_histogram)
    assert histogram.shape == (RING_SIZE_CLASSES,)


def test_train_ring_likelihood_favours_five_and_six(codec):
    chemistry, _, _ = build_v8_references(ROOT, codec)
    for count in (20, 28, 40):
        normal = chemistry.ring_size_probability(count, 5) + chemistry.ring_size_probability(count, 6)
        abnormal = chemistry.ring_size_probability(count, 3) + chemistry.ring_size_probability(count, 4)
        assert normal > abnormal
        assert chemistry.ring_size_probability(count, 6) > chemistry.ring_size_probability(count, 9)


def test_train_ring_environment_oracle_support_is_complete(codec):
    chemistry, _, _ = build_v8_references(ROOT, codec)
    rows, atom_map = hp1kg_records(ROOT, ("train",))
    for row in rows:
        states = codec.encode_record(row, atom_map)
        decoded = decode_heavy_graph(states.tolist(), row["kekule_bonds"], codec)
        molecule = decoded.molecule
        assert not ring_support_warnings_v8(
            states.tolist(), row["kekule_bonds"], row["aromatic_edges"],
            molecule, chemistry,
        )


def test_ring_metrics_are_separate_and_jsd_is_finite(codec):
    chemistry, _, _ = build_v8_references(ROOT, codec)
    record = {
        "raw_rdkit_valid": 1, "raw_joint_valid": 1, "projection_valid": 1,
        "atom_count": 20, "chemical_clean": 1, "ring_aware_clean": 1,
        "db_supported_clean": 1,
        "support_pass": 1, "ring_environment_pass": 1,
        "normal_ring_only": 1, "rings": 2, "five_six_rings": 2,
        "small_rings_3_4": 0, "large_rings_ge_9": 0,
        "ring_sizes": "5;6", "topology_reprojections": 0,
        "generated_geometry_finite": 1, "generated_steric_clashes": 0,
        "reembedded_etkdg_success": -1, "projection_error": "",
        "topology_prior_score": 1., "generated_bond_z_rms": 0.,
        "generated_relaxation_energy_per_heavy_atom": None,
        "s1_absolute_error": 0., "t1_absolute_error": 0.,
    }
    summary = summarize_records_v8([record], chemistry)
    assert summary["five_six_ring_fraction"] == 1.
    assert summary["small_ring_molecule_rate"] == 0.
    assert summary["large_ring_molecule_rate"] == 0.
    assert np.isfinite(summary["ring_size_jsd"])


def test_aromatic_system_loss_is_resonance_invariant():
    size = 6
    aromatic = torch.zeros(1, size, size, dtype=torch.bool)
    first = torch.zeros(1, size, size, dtype=torch.long)
    second = torch.zeros_like(first)
    for index in range(size):
        other = (index + 1) % size
        aromatic[0, index, other] = aromatic[0, other, index] = True
        first_order = 2 if index % 2 == 0 else 1
        second_order = 1 if index % 2 == 0 else 2
        first[0, index, other] = first[0, other, index] = first_order
        second[0, index, other] = second[0, other, index] = second_order
    outputs = {"bond_logits": torch.zeros(1, size, size, 4)}
    left = aromatic_system_loss_v8(outputs, {
        "aromatic_edge_mask": aromatic, "atom_bonds": first,
    })
    right = aromatic_system_loss_v8(outputs, {
        "aromatic_edge_mask": aromatic, "atom_bonds": second,
    })
    assert torch.equal(
        ((first == 2) & aromatic).sum(-1),
        ((second == 2) & aromatic).sum(-1),
    )
    assert float(left) == pytest.approx(float(right), abs=1e-7)
    assert "aromatic_cycle_loss" not in inspect.getsource(joint_training_loss_v8)


def test_flow_milp_has_no_mandatory_tree_gauge(codec, records):
    chemistry, geometry, _ = build_v8_references(ROOT, codec)
    rows, atom_map = records
    for row in rows[::400]:
        state, edge, state_logits, edge_logits = oracle_logits(
            row, atom_map, codec,
        )
        solved = _solve_joint_milp(
            state_logits, edge_logits, codec, 0, 5., 2,
            topology_reference=chemistry,
            topology_prior_weight=0.,
            topology_reproject_threshold=0.,
            geometry_reference=geometry,
            positions=row["atom_positions"],
        )
        assert solved[4] is not None
        assert torch.equal(torch.tensor(solved[0]), state)
        assert torch.equal(solved[1], edge)
    source = inspect.getsource(_solve_joint_milp)
    assert "flow_variables" in source
    assert "tree_pairs" not in source


def test_topology_reference_is_conditioned_and_roundtrips(codec):
    chemistry, _, _ = build_v8_references(ROOT, codec)
    restored = ChemistryReferenceV8.from_metadata(chemistry.to_metadata())
    rows, _ = hp1kg_records(ROOT, ("train",))
    molecule = Chem.MolFromSmiles(rows[0]["smiles"])
    from joint_kekule_diffusion.chemistry_v8 import _ring_topology
    topology = _ring_topology(molecule)
    count = molecule.GetNumHeavyAtoms()
    condition = chemistry.condition_by_count[count][0]
    score = chemistry.topology_score(count, topology, condition)
    assert score == pytest.approx(
        restored.topology_score(count, topology, condition), abs=1e-12,
    )
    assert sum(map(len, chemistry.condition_by_count.values())) == len(rows)


def test_sampling_rejects_implicit_ood_requests(codec):
    model = HeavyAtomJointDDPMV8(
        codec, hidden=32, max_atoms=10, layers=1,
        condition_rbf_min=-6., condition_rbf_max=6.,
    ).eval()
    diffusion = JointDiffusion(timesteps=2)
    with pytest.raises(ValueError, match="outside train support"):
        sample_batch_v8(
            model, diffusion, torch.zeros(1, 2), 1, atoms=9,
            observed_counts=[6, 7, 8],
        )
    with pytest.raises(ValueError, match="outside the RBF range"):
        sample_batch_v8(
            model, diffusion, torch.tensor([[7., 0.]]), 1, atoms=6,
            observed_counts=[6],
        )
