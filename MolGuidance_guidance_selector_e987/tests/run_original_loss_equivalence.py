"""Dependency-free numerical parity check for the released loss reduction."""

import json

import torch
import torch.nn.functional as F

from molguidance.models.loss_utils import reduce_time_scaled_loss_equivalent


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def categorical_case(dtype: torch.dtype) -> dict:
    torch.manual_seed(1701)
    n = 19
    loss_ref = torch.randn(n, dtype=dtype, requires_grad=True)
    weight_ref = torch.rand(n, 1, dtype=dtype, requires_grad=True)
    objective_ref = (loss_ref * weight_ref).mean()
    gradients_ref = torch.autograd.grad(objective_ref, (loss_ref, weight_ref))

    loss_new = loss_ref.detach().clone().requires_grad_(True)
    weight_new = weight_ref.detach().clone().requires_grad_(True)
    objective_new = reduce_time_scaled_loss_equivalent(loss_new, weight_new)
    gradients_new = torch.autograd.grad(objective_new, (loss_new, weight_new))

    torch.testing.assert_close(objective_new, objective_ref)
    torch.testing.assert_close(gradients_new[0], gradients_ref[0])
    torch.testing.assert_close(gradients_new[1], gradients_ref[1])
    return {
        "dtype": str(dtype),
        "loss_abs_diff": abs(float(objective_new - objective_ref)),
        "loss_gradient_max_abs_diff": max_abs(gradients_new[0], gradients_ref[0]),
        "weight_gradient_max_abs_diff": max_abs(gradients_new[1], gradients_ref[1]),
    }


def coordinate_case() -> dict:
    torch.manual_seed(1702)
    n = 23
    loss_ref = torch.randn(n, 3, requires_grad=True)
    weight_ref = torch.rand(n, 1, requires_grad=True)
    objective_ref = (loss_ref * weight_ref).mean()
    gradients_ref = torch.autograd.grad(objective_ref, (loss_ref, weight_ref))

    loss_new = loss_ref.detach().clone().requires_grad_(True)
    weight_new = weight_ref.detach().clone().requires_grad_(True)
    objective_new = reduce_time_scaled_loss_equivalent(loss_new, weight_new)
    gradients_new = torch.autograd.grad(objective_new, (loss_new, weight_new))

    torch.testing.assert_close(objective_new, objective_ref)
    torch.testing.assert_close(gradients_new[0], gradients_ref[0])
    torch.testing.assert_close(gradients_new[1], gradients_ref[1])
    return {
        "loss_abs_diff": abs(float(objective_new - objective_ref)),
        "loss_gradient_max_abs_diff": max_abs(gradients_new[0], gradients_ref[0]),
        "weight_gradient_max_abs_diff": max_abs(gradients_new[1], gradients_ref[1]),
    }


def ctmc_optimizer_case(n_classes: int) -> dict:
    torch.manual_seed(1703 + n_classes)
    n = 31
    target = torch.randint(0, n_classes, (n,))
    target[::3] = -100
    weight = torch.rand(n, 1).clamp_(0.05, 1.5)
    initial_logits = torch.randn(n, n_classes)

    logits_ref = torch.nn.Parameter(initial_logits.clone())
    optimizer_ref = torch.optim.SGD([logits_ref], lr=0.031)
    ce_ref = F.cross_entropy(logits_ref, target, reduction="none", ignore_index=-100)
    objective_ref = (ce_ref * weight).mean()
    objective_ref.backward()
    gradient_ref = logits_ref.grad.detach().clone()
    optimizer_ref.step()

    logits_new = torch.nn.Parameter(initial_logits.clone())
    optimizer_new = torch.optim.SGD([logits_new], lr=0.031)
    ce_new = F.cross_entropy(logits_new, target, reduction="none", ignore_index=-100)
    objective_new = reduce_time_scaled_loss_equivalent(ce_new, weight)
    objective_new.backward()
    gradient_new = logits_new.grad.detach().clone()
    optimizer_new.step()

    torch.testing.assert_close(objective_new, objective_ref)
    torch.testing.assert_close(gradient_new, gradient_ref)
    torch.testing.assert_close(logits_new, logits_ref)
    return {
        "n_classes": n_classes,
        "loss_abs_diff": abs(float(objective_new - objective_ref)),
        "logit_gradient_max_abs_diff": max_abs(gradient_new, gradient_ref),
        "parameter_after_step_max_abs_diff": max_abs(logits_new, logits_ref),
    }


if __name__ == "__main__":
    result = {
        "categorical": [categorical_case(torch.float32), categorical_case(torch.float64)],
        "coordinate": coordinate_case(),
        "ctmc_optimizer_step": [ctmc_optimizer_case(5), ctmc_optimizer_case(12)],
    }
    print(json.dumps(result, indent=2))
