#!/usr/bin/env python3
"""Dependency-free comprehensive validation gate for v7."""
from __future__ import annotations

import inspect
import json
import math
import time
from functools import partial
from pathlib import Path
from unittest import mock

import networkx as nx
import numpy as np
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
from joint_kekule_diffusion.model import JointDiffusion, aromatic_cycle_loss
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
    codec = build_train_codec_v7(ROOT)
    chemistry, geometry, surrogate = build_v7_references(ROOT, codec)

    def split_codec():
        assert len(train_rows) == 1264 and len(val_rows) == 158 and len(test_rows) == 159
        for row in rows:
            codec.encode_record(row, atom_map)
        assert build_train_codec_v7(ROOT).to_metadata() == codec.to_metadata()
        return {"states": len(codec), "splits": [len(train_rows), len(val_rows), len(test_rows)]}
    check("train_only_codec_heldout_coverage", split_codec)

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
            projected = project_joint_graph_v7(state_logits, edge_logits, codec)
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

    def topology_targets():
        dataset = HP1KGV7Dataset(ROOT, "train")
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
            expected = torch.tensor([
                mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol)),
                max_ring_system,
                sum(len(ring) in (3, 4) for ring in mol.GetRingInfo().AtomRings()),
                rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
                sum(sum(bond.GetBondTypeAsDouble() == 2 for bond in atom.GetBonds()) >= 2 for atom in mol.GetAtoms()),
            ], dtype=torch.float32)
            assert torch.equal(row["topology_target"], expected)
        adjacency = torch.zeros(1, 4, 4)
        for left, right in ((0, 1), (1, 2), (2, 3), (3, 0)):
            adjacency[0, left, right] = adjacency[0, right, left] = .5
        value = float(_expected_small_cycles(adjacency, torch.ones(1, 4, dtype=torch.bool)))
        assert abs(value - .5 ** 4) < 1e-7
        presence = adjacency.clone().requires_grad_(True)
        topology = _adjacency_topology_surrogates(presence, torch.ones(1, 4, dtype=torch.bool))
        sum(item.sum() for item in topology).backward()
        assert presence.grad is not None and torch.isfinite(presence.grad).all()
        return {
            "rows": len(dataset), "soft_c4": value,
            "bridgehead_target": "RDKit_CalcNumBridgeheadAtoms",
            "adjacency_gradient_l1": float(presence.grad.abs().sum()),
        }
    check("topology_surrogates", topology_targets)

    def resonance_invariance():
        size = 6
        aromatic = torch.zeros(1, size, size, dtype=torch.bool)
        order = torch.arange(size).reshape(1, 1, size)
        for i in range(size):
            j = (i + 1) % size
            aromatic[0, i, j] = aromatic[0, j, i] = True
        cycle_edges = aromatic[:, None]
        ring_mask = torch.ones(1, 1, dtype=torch.bool)
        double_count = torch.tensor([[3.]])
        cycle_size = torch.tensor([[size]])
        pair = torch.ones(1, size, size, dtype=torch.bool)
        scalar = torch.where(aromatic, torch.tensor(8.), torch.tensor(-8.))
        values = []
        for phase in (0, 1):
            logits = torch.full((1, size, size, 4), -6.)
            logits[..., 0] = 0.
            for i in range(size):
                j = (i + 1) % size
                bond_class = 2 if i % 2 == phase else 1
                logits[0, i, j, bond_class] = logits[0, j, i, bond_class] = 6.
            values.append(float(aromatic_cycle_loss(
                logits, scalar, cycle_edges, double_count, ring_mask, order,
                cycle_size, aromatic, aromatic, pair,
            )))
        assert abs(values[0] - values[1]) < 1e-7
        return {"alternating_phase_losses": values, "absolute_difference": abs(values[0] - values[1])}
    check("resonance_invariant_cycle_partition", resonance_invariance)

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
        result = project_joint_graph_v7(
            state_logits, edge_logits, codec, solver_time_limit=8., solver_attempts=3,
            topology_reference=chemistry, topology_reproject_threshold=0., topology_seed=7,
        )
        assert result.valid and result.solver_used
        decoded = decode_heavy_graph(result.state_ids.tolist(), result.bonds, codec)
        assert decoded.valid and decoded.connected
        return {"source": row.get("id"), "solver": result.solver_status, "smiles": decoded.smiles}
    check("joint_milp_repair", solver_repair)

    def split_support():
        heldout_unseen = 0
        heldout_valid = 0
        for row in val_rows + test_rows:
            state = codec.encode_record(row, atom_map)
            warnings = support_warnings_v7(state.tolist(), row["kekule_bonds"], row["aromatic_edges"], chemistry)
            if warnings:
                heldout_unseen += 1
                _, _, state_logits, edge_logits = oracle_logits(row, atom_map, codec)
                heldout_valid += int(project_joint_graph_v7(state_logits, edge_logits, codec).valid)
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
        model = HeavyAtomJointDDPMV7(
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
        molecule = Chem.MolFromSmiles(train_rows[0]["smiles"])
        from joint_kekule_diffusion.chemistry_v6 import TopologySample, _topology_counts
        exact = TopologySample(*_topology_counts(molecule))
        distorted = TopologySample(
            exact.cycle_rank + 3, exact.max_ring_system + 5,
            exact.small_rings + 3, exact.bridgeheads + 2,
            exact.cumulenes, exact.aromatic_rings,
        )
        exact_score = chemistry.topology_score(molecule.GetNumHeavyAtoms(), exact)
        distorted_score = chemistry.topology_score(molecule.GetNumHeavyAtoms(), distorted)
        assert 0 <= distorted_score < exact_score <= 1
        return {"exact": exact_score, "distorted": distorted_score}
    check("count_conditioned_topology_prior", topology_reference_score)

    def optimization():
        dataset = HP1KGV7Dataset(ROOT, "train")
        collate = partial(collate_hp1kg_v7, max_atoms=69, codec=codec, atom_map=dataset.atom_map)
        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate)))
        marginals = compute_marginals_v7(dataset, codec)
        assert len(marginals["joint_edge"]) == JOINT_EDGE_CLASSES
        assert abs(sum(marginals["joint_edge"]) - 1) < 1e-6
        model = HeavyAtomJointDDPMV7(codec, hidden=32, max_atoms=69, layers=2)
        diffusion = JointDiffusion(timesteps=10, marginals=marginals)
        loss, terms = joint_training_loss_v7(model, diffusion, batch)
        assert torch.isfinite(loss) and all(torch.isfinite(value) for value in terms.values())
        assert float(loss) < 1e4, "initial topology loss is not normalized"
        loss.backward()
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        return {"loss": float(loss), "terms": len(terms), "joint_marginal": marginals["joint_edge"]}
    check("finite_forward_backward", optimization)

    def sampler_alignment():
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
        source = inspect.getsource(sample_batch_v7)
        assert "project_aromatic_cycles_in_reverse" not in source
        assert "synchronize_joint_edge_state" not in source
        return {"diffusion_steps": diffusion.timesteps, "final_projection_calls": wrapped.call_count, "raw_state_retained": "raw_edge_state" in output}
    check("forward_reverse_kernel_alignment", sampler_alignment)

    def hashes():
        digest = dataset_fingerprint(ROOT)
        metadata = {"model_kind": "heavy_atom_joint_edge_d3pm_v7", "dataset_hash": digest, "source_hashes": source_fingerprints_v7()}
        metadata["metadata_hash"] = metadata_fingerprint(metadata)
        checkpoint = {"metadata_hash": metadata["metadata_hash"], "metadata": dict(metadata)}
        assert validate_v7_artifacts(metadata, checkpoint, ROOT) == metadata["metadata_hash"]
        corrupted = dict(metadata)
        corrupted["max_atoms"] = 123
        try:
            validate_v7_artifacts(corrupted, checkpoint, ROOT)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted metadata was accepted")
        return {"dataset_hash": digest, "metadata_hash": metadata["metadata_hash"], "source_files": len(metadata["source_hashes"]), "mismatch_rejected": True}
    check("artifact_hashes", hashes)

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
    output = Path("validation/v7_validation_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "failures": report["failures"], "report": str(output.resolve()), "elapsed_s": report["elapsed_s"]}), flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

