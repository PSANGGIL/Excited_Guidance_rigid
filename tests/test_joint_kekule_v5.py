import json
import random
from functools import partial
from pathlib import Path

import torch
from rdkit import Chem

from joint_kekule_diffusion.chemistry_v5 import (
    AtomStateCodec,
    HeavyAtomState,
    TopologyPrior,
    build_hp1kg_codec,
    collate_hp1kg_v5,
    decode_heavy_graph,
    hp1kg_records,
    materialize_hydrogens,
    project_heavy_graph,
)
from joint_kekule_diffusion.hp1kg_data import HP1KGDataset
from joint_kekule_diffusion.model import JointDiffusion
from joint_kekule_diffusion.model_v5 import HeavyAtomJointDDPM, joint_training_loss_v5


DATA_ROOT = Path("joint_kekule_diffusion/data_hp1kg_nevpt2")


def test_hp1kg_joint_state_codec_is_lossless_for_every_molecule():
    codec = build_hp1kg_codec(DATA_ROOT)
    rows, atom_map = hp1kg_records(DATA_ROOT)
    assert rows and len(codec) > 1
    for row in rows:
        state = codec.encode_record(row, atom_map)
        decoded = decode_heavy_graph(state.tolist(), row["kekule_bonds"], codec)
        assert decoded.valid, (row["id"], decoded.error)
        reference = Chem.MolToSmiles(Chem.MolFromSmiles(row["smiles"]), canonical=True, isomericSmiles=False)
        assert decoded.smiles == reference


def test_v5_model_has_no_independent_hydrogen_or_radical_head():
    codec = AtomStateCodec([
        HeavyAtomState("C", 0, 4, 0, 0, 3),
        HeavyAtomState("N", 0, 3, 0, 0, 2),
    ])
    model = HeavyAtomJointDDPM(codec, hidden=32, max_atoms=8, layers=2, condition_dropout=0).eval()
    assert not hasattr(model, "hydrogen_head")
    assert not hasattr(model, "radical_head")
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :6] = True
    noisy = {
        "atom_mask": mask,
        "atom_state": torch.randint(len(codec), (2, 8)),
        "edge_exists": torch.zeros(2, 8, 8, dtype=torch.long),
        "bond_type": torch.zeros(2, 8, 8, dtype=torch.long),
        "positions": torch.randn(2, 8, 3),
    }
    outputs = model(torch.zeros(2, 2), torch.tensor([2, 7]), noisy)
    assert outputs["state_logits"].shape == (2, 8, len(codec))
    assert "hydrogen_logits" not in outputs and "radical_logits" not in outputs and "charge_logits" not in outputs


def test_topology_projection_uses_cycle_rank_and_full_ring_prior():
    codec = AtomStateCodec([HeavyAtomState("C", 0, 4, 0, 0, 3)])
    prior = TopologyPrior({6: (1,)}, {6: (6,)}, {3: 1, 4: 2, 5: 10, 6: 100, 7: 8, 8: 3}, 8)
    assert prior.ring_log_bias(6) > prior.ring_log_bias(3)
    assert prior.ring_log_bias(6) > prior.ring_log_bias(8)
    state_logits = torch.zeros(6, 1)
    edge_logits = torch.full((6, 6, 2), -8.)
    edge_logits[..., 0] = 0
    for i, j in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)):
        edge_logits[i, j, 1] = edge_logits[j, i, 1] = 8
    bond_logits = torch.zeros(6, 6, 3)
    bond_logits[..., 0] = 4
    aromatic_logits = torch.full((6, 6), -8.)
    projected = project_heavy_graph(state_logits, edge_logits, bond_logits, aromatic_logits, codec, prior, random.Random(4))
    assert projected.valid, projected.reason
    assert projected.realized_cycle_rank == projected.target_cycle_rank == 1
    assert int((projected.bonds > 0).sum() // 2) == 6
    decoded = decode_heavy_graph(projected.state_ids.tolist(), projected.bonds, codec)
    assert decoded.valid and sum(decoded.hydrogens) == 12
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in decoded.molecule.GetAtoms())


def test_hydrogen_is_materialized_once_after_strict_decode():
    codec = AtomStateCodec([HeavyAtomState("C", 0, 4, 0, 0, 3)])
    bonds = torch.tensor([[0, 1], [1, 0]])
    decoded = decode_heavy_graph([0, 0], bonds, codec)
    assert decoded.valid and decoded.hydrogens == (3, 3)
    assert decoded.molecule.GetNumAtoms() == 2
    explicit = materialize_hydrogens(decoded.molecule, add_coordinates=False)
    assert explicit.GetNumAtoms() == 8
    try:
        materialize_hydrogens(explicit, add_coordinates=False)
    except ValueError:
        pass
    else:
        raise AssertionError("second hydrogen materialization must be rejected")


def test_v5_loss_and_gradients_are_finite_on_real_batch():
    codec = build_hp1kg_codec(DATA_ROOT)
    dataset = HP1KGDataset(DATA_ROOT, "train")
    batch = collate_hp1kg_v5([dataset[0], dataset[1]], max_atoms=69, codec=codec, atom_map=dataset.atom_map)
    model = HeavyAtomJointDDPM(codec, hidden=32, max_atoms=69, layers=2, condition_dropout=0)
    diffusion = JointDiffusion(timesteps=20, marginals={"atom": torch.ones(len(codec)), "edge": [1., 1.], "bond_type": [1., 1., 1.]})
    loss, terms = joint_training_loss_v5(model, diffusion, batch)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in terms.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
