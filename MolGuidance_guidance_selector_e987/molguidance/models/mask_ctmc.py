"""Exact transition of a *frozen* mask-star generator for one time interval.

Q(mask,j)=r_j; Q(j,mask)=eta. The neural field and rates are re-evaluated at
every integration step. This is not an exact solution of the time-varying flow.
"""
import torch


def frozen_mask_transition(unmask_rates, remask_rate, xt, dt):
    if unmask_rates.ndim != 2 or xt.shape != unmask_rates.shape[:1]:
        raise ValueError("rates must be [items, real_classes] and xt [items]")
    if not torch.isfinite(unmask_rates).all() or (unmask_rates < 0).any():
        raise ValueError("off-diagonal rates must be finite and nonnegative")
    n, classes = unmask_rates.shape
    if (xt < 0).any() or (xt > classes).any():
        raise ValueError("current state lies outside real+mask vocabulary")
    eta = unmask_rates.new_tensor(remask_rate)
    delta = unmask_rates.new_tensor(dt)
    if eta.numel() != 1 or delta.numel() != 1 or not torch.isfinite(eta + delta) or eta < 0 or delta < 0:
        raise ValueError("remask rate and dt must be finite nonnegative scalars")
    rate = unmask_rates.sum(-1, keepdim=True)
    total = rate + eta
    safe = total.clamp_min(torch.finfo(unmask_rates.dtype).tiny)
    jump = -torch.expm1(-total * delta)
    distribution = unmask_rates / rate.clamp_min(torch.finfo(unmask_rates.dtype).tiny)
    masked = (xt == classes)[:, None]
    mask_prob = torch.where(masked, 1 - rate / safe * jump, eta / safe * jump)
    retained = torch.exp(-eta * delta).expand(n, 1)
    # exp(-eta dt) retains the initial real class, all returned real-class
    # mass is distributed according to mask exit probabilities.
    real_mass = torch.where(masked, 1 - mask_prob, 1 - mask_prob - retained)
    probs = torch.cat((distribution * real_mass.clamp_min(0), mask_prob), -1)
    rows = torch.where(~masked[:, 0])[0]
    probs[rows, xt[rows]] += retained[rows, 0]
    # When both rates are zero, the initial state is absorbing.
    zero = total[:, 0] == 0
    probs[zero] = torch.nn.functional.one_hot(xt[zero], classes + 1).to(probs.dtype)
    return probs / probs.sum(-1, keepdim=True)
