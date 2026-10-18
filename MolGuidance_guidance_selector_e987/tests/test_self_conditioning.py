import unittest
from types import SimpleNamespace

import dgl
import torch
from torch import nn

from molguidance.models.classifier_free_guidance import (
    CFGVectorField,
    ZerosEmbedding,
)
from molguidance.models.ctmc_vector_field import CTMCVectorField
from molguidance.models.flowmol import FlowMol
from molguidance.models.interpolant_scheduler import InterpolantScheduler
from molguidance.models.self_conditioning import SelfConditioningResidualLayer
from molguidance.model_utils.load import validate_chemistry_schema
from molguidance.utils.guidance_sampler import _get_cfg_branch_predictions


def _scheduler():
    return InterpolantScheduler(
        canonical_feat_order=["x", "s", "e"],
        schedule_type={"x": "linear", "s": "linear", "e": "linear"},
    )


def _small_field(self_conditioning=True):
    return CTMCVectorField(
        n_atom_types=2,
        n_atom_states=3,
        category_support_masks={"s": [True, False, True]},
        n_charges=0,
        n_bond_types=4,
        canonical_feat_order=["x", "s", "e"],
        interpolant_scheduler=_scheduler(),
        n_vec_channels=3,
        n_hidden_scalars=8,
        n_hidden_edge_feats=8,
        n_molecule_updates=1,
        convs_per_update=1,
        n_message_gvps=1,
        n_update_gvps=1,
        conditional_generation=False,
        self_conditioning=self_conditioning,
    )


class _RecordingField(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_conditioning = True
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, graph, time, previous_endpoint=None, **kwargs):
        self.calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "previous": previous_endpoint,
                "time_id": id(time),
            }
        )
        value = self.scale * torch.ones(2, 1)
        return {"x": value.expand(2, 3), "s": value.expand(2, 2), "e": value}

    def prepare_self_conditioning_endpoint(
        self, endpoint, *, categorical_are_logits
    ):
        return {key: value.detach() for key, value in endpoint.items()}


class _MinimalCFG(CFGVectorField):
    """Only the coordinate branch is needed to test branch history routing."""

    def __init__(self):
        nn.Module.__init__(self)
        self.self_conditioning = True
        self.unconditional_embedder = ZerosEmbedding(hidden_dim=1)
        self.canonical_feat_order = ["x"]
        self.seen_unconditional = []
        self.seen_conditional = []

    def forward(
        self,
        graph,
        t,
        node_batch_idx,
        upper_edge_mask,
        previous_endpoint=None,
        **kwargs,
    ):
        if t.ndim == 2:
            self.seen_unconditional.append(previous_endpoint)
            offset = 1.0
        else:
            self.seen_conditional.append(previous_endpoint)
            offset = 2.0
        return {"x": graph.ndata["x_t"] + offset}

    def prepare_self_conditioning_endpoint(
        self, endpoint, *, categorical_are_logits
    ):
        return {"x": endpoint["x"].detach()}

    @staticmethod
    def vector_field(x_t, x_1, alpha_t, alpha_t_prime):
        return x_1 - x_t


class _HybridRecordingField(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        graph,
        time,
        node_batch_idx,
        upper_edge_mask,
        previous_endpoint=None,
        **kwargs,
    ):
        self.calls.append((time.detach().clone(), previous_endpoint))
        condition_value = time[:, 1:].mean()
        return {
            "x": torch.full(
                (graph.num_nodes(), 3), float(condition_value)
            ),
            "s": torch.zeros(graph.num_nodes(), 3),
            "e": torch.zeros(int(upper_edge_mask.sum()), 4),
        }

    def prepare_self_conditioning_endpoint(
        self, endpoint, *, categorical_are_logits
    ):
        return {key: value.detach() for key, value in endpoint.items()}


class SelfConditioningTests(unittest.TestCase):
    def test_schema_requires_architecture_and_training_rate_together(self):
        config = {
            "dataset": {
                "use_joint_atom_state": True,
                "n_atom_states": 3,
                "n_bond_types": 4,
                "use_hydrogen_counts": False,
            },
            "mol_fm": {
                "n_atom_states": 3,
                "n_bond_types": 4,
                "n_hydrogen_counts": 0,
                "parameterization": "ctmc",
                "self_conditioning_rate": 0.5,
            },
            "vector_field": {"self_conditioning": False},
        }
        with self.assertRaisesRegex(ValueError, "enabled together"):
            validate_chemistry_schema(config)
        config["vector_field"]["self_conditioning"] = True
        validate_chemistry_schema(config)
        config["mol_fm"]["parameterization"] = "endpoint"
        with self.assertRaisesRegex(ValueError, "only for CTMC"):
            validate_chemistry_schema(config)

    def test_disabled_field_preserves_stage3_parameter_contract(self):
        disabled = _small_field(self_conditioning=False)
        enabled = _small_field(self_conditioning=True)
        self.assertFalse(
            any("self_conditioner" in key for key in disabled.state_dict())
        )
        self.assertTrue(
            any("self_conditioner" in key for key in enabled.state_dict())
        )

    def test_endpoint_preparation_masks_support_normalizes_and_detaches(self):
        field = _small_field()
        endpoint = {
            "x": torch.randn(2, 3, requires_grad=True),
            "s": torch.tensor(
                [[0.0, 1000.0, 0.0], [1.0, 1000.0, -1.0]],
                requires_grad=True,
            ),
            "e": torch.randn(1, 4, requires_grad=True),
        }
        prepared = field.prepare_self_conditioning_endpoint(
            endpoint, categorical_are_logits=True
        )
        self.assertTrue(torch.equal(prepared["s"][:, 1], torch.zeros(2)))
        self.assertTrue(
            torch.allclose(prepared["s"].sum(-1), torch.ones(2))
        )
        self.assertTrue(
            torch.allclose(prepared["e"].sum(-1), torch.ones(1))
        )
        self.assertFalse(any(value.requires_grad for value in prepared.values()))

    def test_probability_feedback_recovers_from_zero_mass_and_nan(self):
        field = _small_field()
        prepared = field.prepare_self_conditioning_endpoint(
            {
                "x": torch.zeros(2, 3),
                # All finite mass is on the unsupported state.
                "s": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
                "e": torch.full((1, 4), float("nan")),
            },
            categorical_are_logits=False,
        )
        expected_state = torch.tensor([[0.5, 0.0, 0.5]]).expand(2, -1)
        self.assertTrue(torch.equal(prepared["s"], expected_state))
        self.assertTrue(torch.isfinite(prepared["e"]).all())
        self.assertTrue(
            torch.allclose(prepared["e"], torch.full((1, 4), 0.25))
        )

    def test_residual_is_rotation_and_translation_invariant(self):
        torch.manual_seed(4)
        upper = torch.triu_indices(3, 3, offset=1)
        edges = torch.cat((upper, upper.flip(0)), dim=1)
        graph = dgl.graph((edges[0], edges[1]), num_nodes=3)
        upper_mask = torch.tensor([True, True, True, False, False, False])
        layer = SelfConditioningResidualLayer(
            node_feature_names=["s"],
            node_feature_dims=[3],
            edge_probability_dim=4,
            node_hidden_dim=8,
            edge_hidden_dim=7,
            rbf_dim=6,
            rbf_dmax=8.0,
        )
        node_hidden = torch.randn(3, 8)
        edge_hidden = torch.randn(6, 7)
        current = torch.randn(3, 3)
        previous = torch.randn(3, 3)
        endpoint = {
            "x": previous,
            "s": torch.softmax(torch.randn(3, 3), -1),
            "e": torch.softmax(torch.randn(3, 4), -1),
        }
        reference = layer(
            graph, node_hidden, edge_hidden, current, upper_mask, endpoint
        )
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = torch.tensor([3.0, -2.0, 5.0])
        transformed_endpoint = dict(endpoint)
        transformed_endpoint["x"] = previous @ rotation.T + translation
        transformed = layer(
            graph,
            node_hidden,
            edge_hidden,
            current @ rotation.T + translation,
            upper_mask,
            transformed_endpoint,
        )
        for actual, expected in zip(transformed, reference):
            self.assertTrue(torch.allclose(actual, expected, atol=2.0e-6))

    def test_training_first_pass_is_no_grad_and_reuses_same_inputs(self):
        model = FlowMol.__new__(FlowMol)
        nn.Module.__init__(model)
        model.vector_field = _RecordingField()
        model.self_conditioning_rate = 1.0
        model.train()
        time = torch.tensor([0.5])
        output = model.predict_endpoint(
            SimpleNamespace(device=torch.device("cpu")),
            time,
            torch.tensor([0, 0]),
            torch.tensor([True, False]),
            self_conditioning=None,
        )
        self.assertEqual(len(model.vector_field.calls), 2)
        first, second = model.vector_field.calls
        self.assertFalse(first["grad_enabled"])
        self.assertTrue(second["grad_enabled"])
        self.assertEqual(first["time_id"], second["time_id"])
        self.assertIsNone(first["previous"])
        self.assertIsNotNone(second["previous"])
        self.assertFalse(
            any(value.requires_grad for value in second["previous"].values())
        )
        output["x"].sum().backward()
        self.assertIsNotNone(model.vector_field.scale.grad)

    def test_cfg_keeps_conditional_and_unconditional_histories_separate(self):
        field = _MinimalCFG()
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        graph.ndata["x_t"] = torch.zeros(2, 3)
        uncond_previous = {"x": torch.full((2, 3), -3.0)}
        cond_previous = {"x": torch.full((2, 3), 7.0)}
        result = field.step_with_CFG_guidance(
            graph,
            s_i=torch.tensor(0.5),
            t_i=torch.tensor(0.25),
            alpha_t_i=torch.tensor([0.25]),
            alpha_s_i=torch.tensor([0.5]),
            alpha_t_prime_i=torch.tensor([1.0]),
            node_batch_idx=torch.zeros(2, dtype=torch.long),
            edge_batch_idx=torch.zeros(2, dtype=torch.long),
            upper_edge_mask=torch.tensor([True, False]),
            guide_w={"x": 1.0},
            previous_unconditional_endpoint=uncond_previous,
            previous_conditional_endpoint=cond_previous,
            return_endpoints=True,
        )
        _, next_unconditional, next_conditional = result
        self.assertIs(field.seen_unconditional[0], uncond_previous)
        self.assertIs(field.seen_conditional[0], cond_previous)
        self.assertTrue(torch.equal(next_unconditional["x"], torch.ones(2, 3)))
        self.assertTrue(
            torch.equal(next_conditional["x"], torch.full((2, 3), 2.0))
        )


    def test_hybrid_cfg_helper_keeps_branch_histories_separate(self):
        field = _HybridRecordingField()
        model = SimpleNamespace(
            vector_field=field,
            property_embedder=nn.Identity(),
            unconditional_embedder=ZerosEmbedding(hidden_dim=1),
        )
        graph = dgl.batch(
            [dgl.graph(([0, 1], [1, 0]), num_nodes=2)]
        )
        upper_edge_mask = torch.tensor([True, False])
        unconditional_previous = {"branch": torch.tensor(-1.0)}
        conditional_previous = {"branch": torch.tensor(1.0)}
        result = _get_cfg_branch_predictions(
            model,
            graph,
            torch.tensor(0.25),
            torch.zeros(2, dtype=torch.long),
            upper_edge_mask,
            property_value=3.0,
            previous_unconditional_endpoint=unconditional_previous,
            previous_conditional_endpoint=conditional_previous,
        )
        self.assertIs(field.calls[0][1], unconditional_previous)
        self.assertIs(field.calls[1][1], conditional_previous)
        self.assertTrue(torch.equal(field.calls[0][0][:, 1], torch.zeros(1)))
        self.assertTrue(torch.equal(field.calls[1][0][:, 1], torch.full((1,), 3.0)))
        self.assertEqual(len(result), 4)
        self.assertFalse(any(value.requires_grad for value in result[2].values()))
        self.assertFalse(any(value.requires_grad for value in result[3].values()))

if __name__ == "__main__":
    unittest.main()
