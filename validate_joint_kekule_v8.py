#!/usr/bin/env python3
"""Dependency-free comprehensive validation gate for v8."""
from __future__ import annotations

import inspect
import json
import math
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import networkx as nx
import numpy as np
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
    _cyclic_bond_signatures,
    _minimum_cycle_sizes,
    _ring_edge_targets_from_graph,
    _ring_topology_from_graph,
    _solve_joint_milp,
    source_fingerprints_v8,
)
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v8 import (
    JOINT_EDGE_CLASSES,
    HeavyAtomJointDDPMV8,
    _normalized_algebraic_connectivity,
    decode_joint_edge_state,
    encode_joint_edge_state,
    aromatic_system_loss_v8,
    joint_training_loss_v8,
)
from joint_kekule_diffusion.sample_v8 import (
    reverse_coords_v8,
    sample_batch_v8,
    validate_v8_artifacts,
)
from joint_kekule_diffusion.train_v8 import build_metadata, compute_marginals_v8


ROOT = Path("joint_kekule_diffusion/data_hp1kg_nevpt2")


def close(left, right, tolerance=2e-5):
    return torch.allclose(left, right, atol=tolerance, rtol=tolerance)


def oracle_logits(row, atom_map, codec):
    state = codec.encode_record(row, atom_map)
    edge = encode_joint_edge_state(row["kekule_bonds"], row["aromatic_edges"])
    count = len(state)
    state_logits = torch.full((count, len(codec)), -20.)
    state_logits[torch.arange(count), state] = 20.
    edge_logits = torch.full((count, count, JOINT_EDGE_CLASSES), -20.)
    edge_logits.scatter_(-1, edge[..., None], 20.)
    return state, edge, state_logits, edge_logits


def main():
    started = time.time()
    report = {"checks": {}, "failures": []}

    def check(name, action):
        begin = time.time()
        try:
            detail = action()
            report["checks"][name] = {"passed": True, "detail": detail, "elapsed_s": time.time() - begin}
            print(json.dumps({"check": name, "passed": True, "detail": detail}), flush=True)
        except Exception as exc:
            report["checks"][name] = {"passed": False, "error": f"{type(exc).__name__}: {exc}", "elapsed_s": time.time() - begin}
            report["failures"].append(name)
            print(json.dumps({"check": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}), flush=True)

    rows, atom_map = hp1kg_records(ROOT)
    train_rows, _ = hp1kg_records(ROOT, ("train",))
    val_rows, _ = hp1kg_records(ROOT, ("val",))
    test_rows, _ = hp1kg_records(ROOT, ("test",))
    codec = build_train_codec_v8(ROOT)
    chemistry, geometry, surrogate = build_v8_references(ROOT, codec)

    def split_codec():
        assert len(train_rows) == 1264 and len(val_rows) == 158 and len(test_rows) == 159
        for row in rows:
            codec.encode_record(row, atom_map)
        assert build_train_codec_v8(ROOT).to_metadata() == codec.to_metadata()
        return {"states": len(codec), "splits": [len(train_rows), len(val_rows), len(test_rows)]}
    check("train_only_codec_heldout_coverage", split_codec)

    def data_integrity():
        split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
        identifiers, canonical = {}, {}
        element_counts = {}
        coordinate_fallbacks = {}
        condition_ranges = {}
        for split, values in split_rows.items():
            split_ids, split_smiles = set(), set()
            s1_values, t1_values = [], []
            for row in values:
                identifier = str(row.get("id", ""))
                molecule = Chem.MolFromSmiles(str(row["smiles"]))
                assert identifier and identifier not in split_ids and molecule is not None
                normalized = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=False,
                )
                assert normalized not in split_smiles
                split_ids.add(identifier)
                split_smiles.add(normalized)
                bonds = torch.as_tensor(row["kekule_bonds"])
                aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
                positions = torch.as_tensor(row["atom_positions"]).float()
                count = len(row["atom_types"])
                assert bonds.shape == (count, count) and aromatic.shape == (count, count)
                assert positions.shape == (count, 3) and torch.isfinite(positions).all()
                assert torch.equal(bonds, bonds.T) and torch.equal(aromatic, aromatic.T)
                assert not torch.diagonal(bonds).any() and not (aromatic & ~(bonds > 0)).any()
                assert len(Chem.GetMolFrags(molecule)) == 1
                assert float(positions.mean(0).abs().max()) < 1e-4
                for atom in molecule.GetAtoms():
                    symbol = atom.GetSymbol()
                    element_counts[symbol] = element_counts.get(symbol, 0) + 1
                s1_values.append(float(row["s1"]))
                t1_values.append(float(row["t1"]))
                method = str(row.get("coordinate_method", "unknown"))
                coordinate_fallbacks[method] = coordinate_fallbacks.get(method, 0) + 1
            identifiers[split], canonical[split] = split_ids, split_smiles
            condition_ranges[split] = {
                "s1": [min(s1_values), max(s1_values)],
                "t1": [min(t1_values), max(t1_values)],
            }
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            assert not identifiers[left] & identifiers[right]
            assert not canonical[left] & canonical[right]
        assert all(math.isfinite(bound) for split in condition_ranges.values() for axis in split.values() for bound in axis)
        return {
            "rows": {key: len(value) for key, value in split_rows.items()},
            "cross_split_id_or_smiles_leakage": 0,
            "elements": element_counts,
            "coordinate_methods": coordinate_fallbacks,
            "condition_ranges": condition_ranges,
        }
    check("database_graph_coordinate_split_integrity", data_integrity)

    def edge_roundtrip():
        aromatic_triples = 0
        histogram = torch.zeros(JOINT_EDGE_CLASSES, dtype=torch.long)
        for row in rows:
            edge = encode_joint_edge_state(row["kekule_bonds"], row["aromatic_edges"])
            bonds, aromatic = decode_joint_edge_state(edge)
            assert torch.equal(bonds, torch.as_tensor(row["kekule_bonds"]).long())
            assert torch.equal(aromatic, torch.as_tensor(row["aromatic_edges"]).bool())
            histogram += torch.bincount(edge[torch.triu(torch.ones_like(edge, dtype=torch.bool), 1)], minlength=JOINT_EDGE_CLASSES)
            aromatic_triples += int((edge == 6).sum().item() // 2)
        assert aromatic_triples > 0
        return {"classes": JOINT_EDGE_CLASSES, "aromatic_triple_edges": aromatic_triples, "histogram": histogram.tolist()}
    check("joint_edge_lossless", edge_roundtrip)

    def full_oracle():
        failures = []
        for index, row in enumerate(rows):
            _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
            projected = project_joint_graph_v8(state_logits, edge_logits, codec)
            decoded = (
                decode_heavy_graph(projected.state_ids.tolist(), projected.bonds, codec)
                if projected.valid else None
            )
            expected = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]), canonical=True, isomericSmiles=False)
            if (
                projected.solver_used or decoded is None or not decoded.valid
                or decoded.smiles != expected
            ):
                failures.append((index, row.get("id"), row["smiles"], projected.reason, None if decoded is None else decoded.smiles))
        assert not failures, failures[:10]
        return {"total": len(rows), "rdkit_identity": len(rows), "rate": 1.0}
    check("full_db_oracle_identity", full_oracle)

    def flow_solver_oracle():
        stride = max(1, len(rows) // 24)
        charge_aware = [row for row in rows if "[nH+]" in row["smiles"]]
        selected_by_id = {}
        for row in charge_aware + rows[::stride][:24]:
            selected_by_id[row.get("id", row["smiles"])] = row
        selected = list(selected_by_id.values())
        exact = 0
        for row in selected:
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
            solved_state, solved_edge, _, _, decoded = solved[:5]
            assert decoded is not None
            assert torch.equal(torch.tensor(solved_state), state)
            assert torch.equal(solved_edge, edge)
            exact += 1
        source = inspect.getsource(_solve_joint_milp)
        assert "tree_pairs" not in source
        assert "flow_variables" in source
        return {
            "sampled_across_full_db": len(selected),
            "exact": exact, "charge_aware_nh_plus": len(charge_aware),
            "connectivity": "single_commodity_flow",
            "spanning_tree_gauge": False,
        }
    check("flow_milp_oracle_identity", flow_solver_oracle)


    def topology_targets():
        dataset = HP1KGV8Dataset(ROOT, "train")
        for row in dataset.rows:
            mol = Chem.MolFromSmiles(row["smiles"])
            graph = nx.Graph()
            graph.add_nodes_from(range(mol.GetNumAtoms()))
            graph.add_edges_from((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds())
            max_ring_system = max(
                (len(component) for component in nx.biconnected_components(graph)
                 if graph.subgraph(component).number_of_edges() >= len(component)),
                default=0,
            )
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            invariant = _ring_topology_from_graph(bonds, aromatic)
            bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            assert invariant.bridgeheads == bridgeheads
            expected = torch.tensor([
                invariant.closure_count,
                invariant.max_ring_system,
                sum(size in (3, 4) for size in invariant.ring_sizes),
                bridgeheads,
                sum(
                    sum(bond.GetBondTypeAsDouble() == 2 for bond in atom.GetBonds()) >= 2
                    for atom in mol.GetAtoms()
                ),
            ], dtype=torch.float32)
            assert torch.equal(row["topology_target"], expected)
            assert not (row["ring_closure_target"] & ~(bonds > 0)).any()
            assert not (aromatic & ~row["ring_closure_target"]).any()
        probe = dataset.rows[0]
        closure_edges = int(torch.triu(probe["ring_closure_target"], 1).sum())
        assert closure_edges > 0
        assert torch.equal(probe["ring_closure_target"], probe["ring_closure_target"].T)
        return {
            "rows": len(dataset), "cyclic_edges_in_probe": closure_edges,
            "bridgehead_target": "RDKit_CalcNumBridgeheadAtoms",
            "index_invariants": "cyclic_subset_bond_and_aromatic_subset_cyclic",
        }
    check("index_safe_topology_targets", topology_targets)

    def ring_permutation_invariance():
        generator = torch.Generator().manual_seed(20260925)
        tested = 0
        for row in rows:
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            states = codec.encode_record(row, atom_map)
            baseline_topology = _ring_topology_from_graph(bonds, aromatic)
            baseline_sizes = _minimum_cycle_sizes(bonds)
            baseline_environment = _cyclic_bond_signatures(states, bonds, aromatic)
            for _ in range(2):
                permutation = torch.randperm(len(states), generator=generator)
                permuted_bonds = bonds[permutation][:, permutation]
                permuted_aromatic = aromatic[permutation][:, permutation]
                permuted_states = states[permutation]
                assert _ring_topology_from_graph(permuted_bonds, permuted_aromatic) == baseline_topology
                assert _minimum_cycle_sizes(permuted_bonds) == baseline_sizes
                assert _cyclic_bond_signatures(
                    permuted_states, permuted_bonds, permuted_aromatic,
                ) == baseline_environment
                tested += 1
        return {
            "molecules": len(rows), "permutations": tested,
            "basis": "minimum_cycle_basis_size_multiset",
            "environment": "cyclic_bond_signature",
        }
    check("full_db_ring_atom_permutation_invariance", ring_permutation_invariance)

    def resonance_invariance():
        size = 6
        aromatic = torch.zeros(1, size, size, dtype=torch.bool)
        first_bonds = torch.zeros(1, size, size, dtype=torch.long)
        second_bonds = torch.zeros_like(first_bonds)
        for index in range(size):
            other = (index + 1) % size
            aromatic[0, index, other] = aromatic[0, other, index] = True
            first_order = 2 if index % 2 == 0 else 1
            second_order = 1 if index % 2 == 0 else 2
            first_bonds[0, index, other] = first_bonds[0, other, index] = first_order
            second_bonds[0, index, other] = second_bonds[0, other, index] = second_order
        outputs = {"bond_logits": torch.zeros(1, size, size, 4)}
        first = aromatic_system_loss_v8(outputs, {
            "aromatic_edge_mask": aromatic, "atom_bonds": first_bonds,
        })
        second = aromatic_system_loss_v8(outputs, {
            "aromatic_edge_mask": aromatic, "atom_bonds": second_bonds,
        })
        first_incidence = ((first_bonds == 2) & aromatic).sum(-1)
        second_incidence = ((second_bonds == 2) & aromatic).sum(-1)
        assert torch.equal(first_incidence, second_incidence)
        assert abs(float(first - second)) < 1e-7
        source = inspect.getsource(joint_training_loss_v8)
        assert "aromatic_cycle_loss" not in source
        return {
            "system_losses": [float(first), float(second)],
            "absolute_difference": abs(float(first - second)),
            "ring_local_fixed_double_count_removed": True,
        }
    check("resonance_invariant_aromatic_system_loss", resonance_invariance)

    def full_resonance_phase_audit():
        generator = torch.Generator().manual_seed(20260926)
        changed_assignments = 0
        tested = 0
        for row in rows:
            states = codec.encode_record(row, atom_map)
            bonds = torch.as_tensor(row["kekule_bonds"]).long()
            aromatic = torch.as_tensor(row["aromatic_edges"]).bool()
            decoded = decode_heavy_graph(states.tolist(), bonds, codec)
            assert decoded.valid and decoded.molecule is not None
            target_incidence = ((bonds == 2) & aromatic).sum(-1)
            for _ in range(2):
                permutation = torch.randperm(len(states), generator=generator)
                molecule = Chem.RenumberAtoms(
                    decoded.molecule, permutation.tolist(),
                )
                Chem.Kekulize(molecule, clearAromaticFlags=True)
                alternative = torch.zeros_like(bonds)
                incidence = torch.zeros_like(target_incidence)
                for bond in molecule.GetBonds():
                    new_left, new_right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    left = int(permutation[new_left])
                    right = int(permutation[new_right])
                    order = int(round(bond.GetBondTypeAsDouble()))
                    alternative[left, right] = alternative[right, left] = order
                    if bool(aromatic[left, right]) and order == 2:
                        incidence[left] += 1
                        incidence[right] += 1
                changed_assignments += int(
                    not torch.equal(alternative[aromatic], bonds[aromatic])
                )
                assert torch.equal(incidence, target_incidence)
                tested += 1
        assert changed_assignments > 0
        return {
            "molecules": len(rows), "alternative_kekule_trials": tested,
            "changed_edge_assignments": changed_assignments,
            "atom_double_incidence_failures": 0,
        }
    check("full_db_kekule_resonance_incidence_invariance", full_resonance_phase_audit)

    def solver_repair():
        selected = None
        for row in rows:
            graph = nx.Graph()
            count = len(row["atom_types"])
            graph.add_nodes_from(range(count))
            bonds = torch.as_tensor(row["kekule_bonds"])
            graph.add_edges_from((i, j) for i in range(count) for j in range(i + 1, count) if int(bonds[i, j]))
            bridges = list(nx.bridges(graph))
            if count >= 4 and bridges:
                selected = row, bridges[0]
                break
        assert selected is not None
        row, (left, right) = selected
        _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
        edge_logits[left, right] = edge_logits[right, left] = -8.
        edge_logits[left, right, 0] = edge_logits[right, left, 0] = 8.
        result = project_joint_graph_v8(
            state_logits, edge_logits, codec, solver_time_limit=8., solver_attempts=3,
            topology_reference=chemistry, topology_reproject_threshold=0., topology_seed=7,
        )
        assert result.valid and result.solver_used
        decoded = decode_heavy_graph(result.state_ids.tolist(), result.bonds, codec)
        assert decoded.valid and decoded.connected
        return {"source": row.get("id"), "solver": result.solver_status, "smiles": decoded.smiles}
    check("joint_milp_repair", solver_repair)

    def exact_no_good_and_rerank():
        carbon = next(
            index for index, state in enumerate(codec.states)
            if state.symbol == "C" and state.formal_charge == 0
            and state.valence == 4 and state.aromatic_role == 0
        )
        state_logits = torch.full((3, len(codec)), -20.)
        state_logits[:, carbon] = 20.
        edge_logits = torch.full((3, 3, JOINT_EDGE_CLASSES), -20.)
        edge_logits[..., 0] = 0.
        for left, right in ((0, 1), (1, 2)):
            edge_logits[left, right, 1] = edge_logits[right, left, 1] = 10.
        edge_logits[0, 2, 1] = edge_logits[2, 0, 1] = -.1

        class PreferCycle:
            @staticmethod
            def topology_score(count, topology, condition=None):
                return 1. if topology.closure_count else .001

        solved = _solve_joint_milp(
            state_logits, edge_logits, codec, 0, 10., 2,
            complete_edges=True, topology_reference=PreferCycle(),
            topology_prior_weight=10., topology_reproject_threshold=0.,
        )
        assert solved[4] is not None
        assert int((solved[1] > 0).triu(1).sum()) == 3
        assert "attempt=2" in solved[5] and "candidate_rerank=1" in solved[5]
        return {
            "first_map": "two-edge_tree",
            "selected_after_rerank": "three-edge_cycle_superset",
            "solver_status": solved[5],
        }
    check("milp_exact_no_good_and_topology_rerank", exact_no_good_and_rerank)

    def split_support():
        heldout_unseen = 0
        heldout_valid = 0
        for row in val_rows + test_rows:
            state = codec.encode_record(row, atom_map)
            warnings = support_warnings_v8(state.tolist(), row["kekule_bonds"], row["aromatic_edges"], chemistry)
            if warnings:
                heldout_unseen += 1
                _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
                heldout_valid += int(project_joint_graph_v8(state_logits, edge_logits, codec).valid)
        assert heldout_unseen > 0 and heldout_valid == heldout_unseen
        return {"heldout_unseen_support": heldout_unseen, "still_projection_valid": heldout_valid}
    check("support_metric_not_filter", split_support)

    def geometry_direct():
        row = rows[0]
        states = codec.encode_record(row, atom_map).tolist()
        decoded = decode_heavy_graph(states, row["kekule_bonds"], codec)
        good = assess_generated_coordinates(
            decoded.molecule, states, row["kekule_bonds"], row["aromatic_edges"],
            row["atom_positions"], codec, geometry, optimize=False,
        )
        bad = assess_generated_coordinates(
            decoded.molecule, states, row["kekule_bonds"], row["aromatic_edges"],
            torch.as_tensor(row["atom_positions"]) * 10, codec, geometry, optimize=False,
        )
        assert good.finite and bad.finite and bad.bond_z_rms > good.bond_z_rms and bad.bond_outliers > 0
        return {"reference_z_rms": good.bond_z_rms, "distorted_z_rms": bad.bond_z_rms}
    check("generated_coordinate_geometry", geometry_direct)

    def equivariance():
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
        padding_error = float((base["state_logits"][:, :4] - padded["state_logits"][:, :4]).abs().max())
        assert padding_error < 1e-6

        permutation = torch.tensor([2, 0, 3, 1, 4, 5, 6, 7])
        permuted = {
            "atom_mask": state["atom_mask"][:, permutation],
            "atom_state": state["atom_state"][:, permutation],
            "edge_state": state["edge_state"][:, permutation][:, :, permutation],
            "positions": state["positions"][:, permutation],
        }
        output = model(condition, timestep, permuted)
        permutation_error = float((base["state_logits"][:, permutation] - output["state_logits"]).abs().max())
        pair_error = float((
            base["joint_edge_logits"][:, permutation][:, :, permutation] - output["joint_edge_logits"]
        ).abs().max())
        assert permutation_error < 2e-5 and pair_error < 2e-5

        rotation = torch.linalg.qr(torch.randn(3, 3)).Q
        if torch.det(rotation) < 0:
            rotation[:, 0] *= -1
        rotated = model(condition, timestep, {**state, "positions": state["positions"] @ rotation})
        rotation_error = float((base["coordinate_eps"] @ rotation - rotated["coordinate_eps"]).abs().max())
        assert rotation_error < 2e-5
        return {"padding_error": padding_error, "permutation_error": permutation_error, "pair_error": pair_error, "rotation_error": rotation_error}
    check("padding_permutation_o3", equivariance)

    def regular_dropout():
        model = HeavyAtomJointDDPMV8(
            codec, hidden=32, max_atoms=6, layers=2,
            condition_dropout=0., model_dropout=.4, count_dropout=.5,
        )
        assert not hasattr(model, "topology_head")
        mask = torch.ones(1, 6, dtype=torch.bool)
        state = {
            "atom_mask": mask,
            "atom_state": torch.randint(len(codec), (1, 6)),
            "edge_state": torch.zeros(1, 6, 6, dtype=torch.long),
            "positions": torch.randn(1, 6, 3),
        }
        condition, timestep = torch.randn(1, 2), torch.tensor([5])
        model.train(); torch.manual_seed(1); first = model(condition, timestep, state)
        torch.manual_seed(2); second = model(condition, timestep, state)
        train_state_delta = float((first["state_logits"] - second["state_logits"]).abs().max())
        train_count_delta = float((first["count_logits"] - second["count_logits"]).abs().max())
        assert train_state_delta > 0 and train_count_delta > 0
        model.eval(); torch.manual_seed(1); first = model(condition, timestep, state)
        torch.manual_seed(2); second = model(condition, timestep, state)
        eval_delta = float((first["state_logits"] - second["state_logits"]).abs().max())
        assert eval_delta == 0
        return {
            "model_dropout": model.model_dropout, "count_dropout": model.count_dropout,
            "train_state_delta": train_state_delta, "train_count_delta": train_count_delta,
            "eval_delta": eval_delta, "detached_topology_head_removed": True,
        }
    check("cfg_separate_regular_dropout", regular_dropout)

    def topology_reference_score():
        from joint_kekule_diffusion.chemistry_v8 import RingTopologySampleV8, _ring_topology
        molecule = Chem.MolFromSmiles(train_rows[0]["smiles"])
        exact = _ring_topology(molecule)
        distorted = RingTopologySampleV8(
            exact.closure_count + 3, exact.max_ring_system + 5,
            tuple(sorted(exact.ring_sizes + (3, 4, 9))),
            exact.aromatic_ring_sizes,
            exact.bridgeheads + 2,
        )
        count = molecule.GetNumHeavyAtoms()
        stats = json.loads((ROOT / "manifest.json").read_text())["condition_stats"]
        condition = (
            (float(train_rows[0]["s1"]) - stats["s1_energy_mean"]) / stats["s1_energy_std"],
            (float(train_rows[0]["t1"]) - stats["t1_energy_mean"]) / stats["t1_energy_std"],
        )
        exact_score = chemistry.topology_score(count, exact, condition)
        distorted_score = chemistry.topology_score(count, distorted, condition)
        far_score = chemistry.topology_score(count, exact, (6., 6.))
        restored = ChemistryReferenceV8.from_metadata(chemistry.to_metadata())
        restored_score = restored.topology_score(count, exact, condition)
        normal = (
            chemistry.ring_size_probability(count, 5)
            + chemistry.ring_size_probability(count, 6)
        )
        abnormal = (
            chemistry.ring_size_probability(count, 3)
            + chemistry.ring_size_probability(count, 4)
        )
        assert 0 <= distorted_score < exact_score <= 1
        assert 0 <= far_score <= 1
        assert abs(restored_score - exact_score) < 1e-12
        assert normal > abnormal
        return {
            "exact": exact_score, "distorted": distorted_score,
            "same_topology_far_condition": far_score,
            "normal_vs_small": [normal, abnormal],
            "serialized_condition_rows": sum(
                map(len, chemistry.condition_by_count.values())
            ),
        }
    check("s1_t1_conditioned_ring_topology_prior", topology_reference_score)

    def optimization():
        dataset = HP1KGV8Dataset(ROOT, "train")
        collate = partial(collate_hp1kg_v8, max_atoms=69, codec=codec, atom_map=dataset.atom_map)
        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))
        marginals = compute_marginals_v8(dataset, codec)
        assert len(marginals["joint_edge"]) == JOINT_EDGE_CLASSES
        assert abs(sum(marginals["joint_edge"]) - 1) < 1e-6
        model = HeavyAtomJointDDPMV8(codec, hidden=32, max_atoms=69, layers=2)
        diffusion = JointDiffusion(timesteps=10, marginals=marginals)
        loss, terms = joint_training_loss_v8(model, diffusion, batch)
        assert torch.isfinite(loss) and all(torch.isfinite(value) for value in terms.values())
        assert "ring_size_distribution" not in terms
        assert "bridgehead_prior" not in terms
        assert float(loss) < 100, "initial loss terms are not normalized"
        largest_name, largest_value = max(terms.items(), key=lambda item: float(item[1]))
        assert float(largest_value / loss) < .75, "one auxiliary loss drowns denoising"
        loss.backward()
        assert model.ring_closure_head.weight.grad is not None
        assert model.ring_size_head.weight.grad is not None
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        return {
            "loss": float(loss), "terms": len(terms),
            "largest_term": [largest_name, float(largest_value)],
            "joint_marginal": marginals["joint_edge"],
        }
    check("finite_forward_backward", optimization)

    def spectral_connectivity_oracle():
        minimum_target = float("inf")
        for row in train_rows:
            adjacency = (torch.as_tensor(row["kekule_bonds"]) > 0).float()
            value = float(_normalized_algebraic_connectivity(adjacency[None]))
            minimum_target = min(minimum_target, value)
        assert minimum_target > 1e-7
        disconnected = torch.zeros(4, 4)
        disconnected[0, 1] = disconnected[1, 0] = 1
        disconnected[2, 3] = disconnected[3, 2] = 1
        disconnected_value = float(
            _normalized_algebraic_connectivity(disconnected[None]),
        )
        path = torch.zeros(4, 4)
        for left, right in ((0, 1), (1, 2), (2, 3)):
            path[left, right] = path[right, left] = 1
        dense = torch.ones(4, 4) - torch.eye(4)
        path_value = float(_normalized_algebraic_connectivity(path[None]))
        dense_value = float(_normalized_algebraic_connectivity(dense[None]))
        assert abs(disconnected_value) < 1e-6
        assert path_value > disconnected_value and dense_value > path_value
        return {
            "connected_oracle_rows": len(train_rows),
            "minimum_target_lambda2": minimum_target,
            "probe_lambda2": {
                "disconnected": disconnected_value,
                "path": path_value,
                "dense": dense_value,
            },
            "oracle_regression_loss": 0.,
        }
    check("spectral_connectivity_distinguishes_sparse_dense", spectral_connectivity_oracle)

    def diffusion_math():
        torch.manual_seed(1701)
        classes = 4
        marginal = torch.tensor([.1, .2, .3, .4])
        diffusion = JointDiffusion(
            timesteps=10, beta_start=.01, beta_end=.08,
            marginals={"atom": marginal},
        )
        logits = torch.randn(4, 3, classes)
        observed = torch.randint(classes, (4, 3))
        timesteps = torch.tensor([0, 1, 5, 9])
        implementation = diffusion.posterior_logits(
            observed, logits, timesteps, classes, "atom",
        ).exp()
        p0 = logits.softmax(-1)
        brute = torch.zeros_like(implementation)
        for batch_index, timestep in enumerate(timesteps.tolist()):
            alpha_bar = float(diffusion.abar[timestep])
            previous = 1. if timestep == 0 else float(diffusion.abar[timestep - 1])
            one_step = alpha_bar / previous
            for item in range(observed.shape[1]):
                terminal = int(observed[batch_index, item])
                for clean in range(classes):
                    denominator = alpha_bar * (clean == terminal) + (1 - alpha_bar) * float(marginal[terminal])
                    for prior_state in range(classes):
                        q_previous = previous * (prior_state == clean) + (1 - previous) * float(marginal[prior_state])
                        q_step = one_step * (prior_state == terminal) + (1 - one_step) * float(marginal[terminal])
                        brute[batch_index, item, prior_state] += p0[batch_index, item, clean] * q_previous * q_step / denominator
        brute /= brute.sum(-1, keepdim=True)
        posterior_error = float((implementation - brute).abs().max())
        assert posterior_error < 2e-6

        production = JointDiffusion(timesteps=500, beta_start=1e-4, beta_end=.05)
        mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        clean = torch.randn(1, 4, 3) * mask[..., None]
        clean -= clean.sum(1, keepdim=True) / mask.sum(1, keepdim=True)[..., None]
        clean *= mask[..., None]
        noise = torch.randn_like(clean) * mask[..., None]
        noise -= noise.sum(1, keepdim=True) / mask.sum(1, keepdim=True)[..., None]
        noise *= mask[..., None]
        noisy = production.abar[0].sqrt() * clean + (1 - production.abar[0]).sqrt() * noise
        recovered = reverse_coords_v8(noisy, noise, 0, production, mask)
        coordinate_error = float((recovered - clean).abs().max())
        assert coordinate_error < 2e-5
        assert float(production.abar[-1]) < 1e-4
        try:
            reverse_coords_v8(noisy * 1e6, torch.zeros_like(noisy), 0, production, mask, 10.)
        except FloatingPointError:
            rejected = True
        else:
            rejected = False
        assert rejected
        return {
            "d3pm_bruteforce_max_error": posterior_error,
            "coordinate_t0_oracle_max_error": coordinate_error,
            "terminal_alpha_bar": float(production.abar[-1]),
            "silent_coordinate_clipping": False,
            "runaway_state_rejected": rejected,
        }
    check("exact_d3pm_and_cartesian_reverse_math", diffusion_math)

    def sampler_alignment():
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
        source = inspect.getsource(sample_batch_v8)
        assert "project_aromatic_cycles_in_reverse" not in source
        assert "synchronize_joint_edge_state" not in source
        return {"diffusion_steps": diffusion.timesteps, "final_projection_calls": wrapped.call_count, "raw_state_retained": "raw_edge_state" in output}
    check("forward_reverse_kernel_alignment", sampler_alignment)

    def hashes():
        digest = dataset_fingerprint(ROOT)
        metadata = {
            "model_kind": "heavy_atom_joint_edge_d3pm_v8",
            "dataset_hash": digest,
            "source_hashes": source_fingerprints_v8(),
            "runtime_versions": runtime_versions_v8(),
        }
        metadata["metadata_hash"] = metadata_fingerprint(metadata)
        checkpoint = {"metadata_hash": metadata["metadata_hash"], "metadata": dict(metadata)}
        assert validate_v8_artifacts(metadata, checkpoint, ROOT) == metadata["metadata_hash"]
        corrupted = dict(metadata)
        corrupted["max_atoms"] = 123
        try:
            validate_v8_artifacts(corrupted, checkpoint, ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted metadata was accepted")
        runtime_corrupted = dict(metadata)
        runtime_corrupted["runtime_versions"] = {
            **runtime_corrupted["runtime_versions"], "rdkit": "0.invalid",
        }
        runtime_corrupted["metadata_hash"] = metadata_fingerprint(runtime_corrupted)
        runtime_checkpoint = {
            "metadata_hash": runtime_corrupted["metadata_hash"],
            "metadata": dict(runtime_corrupted),
        }
        try:
            validate_v8_artifacts(runtime_corrupted, runtime_checkpoint, ROOT)
        except ValueError as exc:
            assert "runtime" in str(exc)
        else:
            raise AssertionError("runtime version mismatch was accepted")
        return {"dataset_hash": digest, "metadata_hash": metadata["metadata_hash"], "source_files": len(metadata["source_hashes"]), "mismatch_rejected": True, "runtime_mismatch_rejected": True}
    check("artifact_hashes", hashes)

    def metadata_contract():
        training = HP1KGV8Dataset(ROOT, "train")
        validation = HP1KGV8Dataset(
            ROOT, "val", condition_stats=training.condition_stats,
            atom_map=training.atom_map,
        )
        marginals = compute_marginals_v8(training, codec)
        arguments = SimpleNamespace(
            root=str(ROOT), run_dir="unused", resume=None, epochs=10000,
            batch_size=192, precision="fp32", allow_tf32=False, workers=4,
            hidden=256, layers=8, lr=2e-4, grad_clip=5., save_every=100,
            keep_periodic_checkpoints=10, seed=61, s1_rbf_bins=24,
            t1_rbf_bins=24, condition_rbf_min=-6., condition_rbf_max=6.,
            condition_rbf_width_scale=1., condition_dropout=.1,
            model_dropout=.1, count_dropout=.2, topology_prior_weight=1.5,
            topology_reproject_threshold=.2, geometry_prior_weight=.05,
            max_atoms=0, rollout_every=10, rollout_samples=32,
            rollout_3d=False, coordinate_limit=1000., ema_decay=.999,
            early_stopping_patience=500,
        )
        manifest = json.loads((ROOT / "manifest.json").read_text())
        metadata = build_metadata(
            arguments, manifest, codec, (chemistry, geometry, surrogate),
            marginals, training, validation, 69,
        )
        assert metadata["schema_version"] == 13
        assert metadata["runtime_versions"] == runtime_versions_v8()
        assert metadata["metadata_hash"] == metadata_fingerprint(metadata)
        json.dumps(metadata, allow_nan=False)
        restored = ChemistryReferenceV8.from_metadata(
            metadata["chemistry_reference"],
        )
        assert restored.to_metadata() == metadata["chemistry_reference"]
        args = metadata["model_args"]
        rebuilt = HeavyAtomJointDDPMV8(
            codec, args["hidden"], metadata["max_atoms"], args["layers"],
            args["condition_dropout"], args["s1_rbf_bins"],
            args["t1_rbf_bins"], args["condition_rbf_min"],
            args["condition_rbf_max"], args["condition_rbf_width_scale"],
            args["model_dropout"], args["count_dropout"],
        )
        schedule = metadata["diffusion"]
        rebuilt_diffusion = JointDiffusion(
            schedule["timesteps"], schedule["beta_start"],
            schedule["beta_end"], metadata["diffusion_marginals"],
        )
        assert rebuilt.max_atoms == 69 and float(rebuilt_diffusion.abar[-1]) < 1e-4
        return {
            "schema": metadata["schema_version"],
            "metadata_hash": metadata["metadata_hash"],
            "runtime_versions": metadata["runtime_versions"],
            "json_roundtrip": True,
            "model_and_diffusion_rebuilt": True,
        }
    check("production_metadata_roundtrip_and_rebuild", metadata_contract)

    def condition_model():
        metrics = surrogate.validation_metrics
        assert metrics["samples"] == len(val_rows)
        assert all(math.isfinite(metrics[key]) for key in ("s1_mae", "t1_mae", "s1_r2", "t1_r2"))
        prediction = surrogate.predict(Chem.MolFromSmiles(test_rows[0]["smiles"]))
        assert prediction.shape == (2,) and np.isfinite(prediction).all()
        return {"validation_metrics": metrics, "test_prediction": prediction.tolist()}
    check("s1_t1_structure_surrogate", condition_model)

    report["passed"] = not report["failures"]
    report["elapsed_s"] = time.time() - started
    output = Path("validation/v8_validation_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "failures": report["failures"], "report": str(output.resolve()), "elapsed_s": report["elapsed_s"]}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
