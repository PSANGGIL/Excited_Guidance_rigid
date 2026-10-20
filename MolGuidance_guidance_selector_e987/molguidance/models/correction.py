"""Training-only off-path denoising objective; NOT an altered CTMC likelihood.

Corruptions never become clean data or acquire new EST labels. The target is
always the original molecule/official geometry. No test records are used.
"""
import math
import torch
import torch.nn.functional as F

from molguidance.data_processing.utils import get_batch_idxs, get_upper_edge_mask
from molguidance.models.loss_utils import reduce_categorical_loss


def correction_options(config=None):
    options = dict(weight=0., probability=.25, time=.85, node_rate=.1,
                   edge_rate=.02, coordinate_std=.1)
    unknown = set(config or {}) - set(options)
    if unknown:
        raise ValueError(f"unknown correction options: {sorted(unknown)}")
    options.update(config or {})
    options = {key: float(value) for key, value in options.items()}
    if any(not math.isfinite(value) or value < 0 for value in options.values()):
        raise ValueError("correction settings must be finite and non-negative")
    if any(options[key] > 1 for key in ("probability", "node_rate", "edge_rate")):
        raise ValueError("correction probabilities must be in [0, 1]")
    if not 0 < options["time"] < 1:
        raise ValueError("correction time must be strictly between zero and one")
    if options["weight"] > 0 and options["probability"] == 0:
        raise ValueError("enabled correction requires a nonzero probability")
    return options


def corrupt_visible_state(g, field, options, generator=None):
    """Modify only *_t (symmetric edges); return actual changed/unknown masks."""
    upper = get_upper_edge_mask(g)
    masks = {}
    for feat, rate, source in (
        ("s", options["node_rate"], g.ndata),
        ("e", options["edge_rate"], g.edata),
    ):
        original = source[f"{feat}_t"]
        before = (original[upper] if feat == "e" else original).argmax(-1)
        k = field.n_cat_feats[feat]
        support = field.get_category_support_mask(feat)
        support = torch.ones(k, dtype=torch.bool, device=g.device) if support is None else support
        choices = torch.where(support)[0]
        replacement = choices[torch.randint(
            choices.numel(), before.shape, device=g.device, generator=generator
        )]
        change = torch.rand(before.shape, device=g.device, generator=generator) < rate
        after = torch.where(change, replacement, before)
        masks[feat] = (after != before) | (before == k)
        encoded = F.one_hot(after, k + 1).to(original)
        if feat == "e":
            value = torch.zeros_like(original)
            value[upper], value[~upper] = encoded, encoded
        else:
            value = encoded
        source[f"{feat}_t"] = value
    node_batch, _ = get_batch_idxs(g)
    perturbation = torch.randn(
        g.ndata["x_t"].shape, device=g.device, dtype=g.ndata["x_t"].dtype,
        generator=generator,
    ) * options["coordinate_std"]
    centers = perturbation.new_zeros(g.batch_size, 3)
    centers.index_add_(0, node_batch, perturbation)
    centers /= g.batch_num_nodes().to(centers)[:, None]
    g.ndata["x_t"] = g.ndata["x_t"] + perturbation - centers[node_batch]
    return masks


def correction_losses(model, g, generator=None, force=False):
    """Separate auxiliary loss on locally corrupted late-time input."""
    opts = model.correction_config
    zero = next(model.parameters()).sum() * 0.
    empty = {f"correction_{key}": zero for key in ("x", "s", "e")}
    if not model.training or opts["weight"] <= 0:
        return empty
    if not force and float(torch.rand((), device=g.device, generator=generator)) >= opts["probability"]:
        return empty
    node_batch, edge_batch = get_batch_idxs(g)
    upper = get_upper_edge_mask(g)
    time = torch.full((g.batch_size,), opts["time"], device=g.device)
    with g.local_scope():
        model.vector_field.sample_conditional_path(
            g, time, node_batch, edge_batch, upper, generator=generator, graph_time=time
        )
        active = corrupt_visible_state(g, model.vector_field, opts, generator)
        model_time = time
        if model.conditional_generation:
            prop = g.prop[:, None] if g.prop.ndim == 1 else g.prop
            embedding = model.property_embedder(prop.to(g.device))
            # Respect the same graph-level CFG branch selected by the main pass.
            if hasattr(g, "_USE_UNCONDITIONAL_EMBEDDING"):
                embedding = torch.where(
                    g._USE_UNCONDITIONAL_EMBEDDING.to(g.device),
                    torch.zeros_like(embedding), embedding,
                )
            model_time = torch.cat((time[:, None], embedding), -1)
        output = model.predict_endpoint(
            g, model_time, node_batch, upper, graph_time=time,
            corruption_generator=generator, self_conditioning=True,
        )
        result = {}
        real = g.ndata.get("is_real_1_true", torch.ones(g.num_nodes(), device=g.device, dtype=torch.bool))
        coord_error = (output["x"] - g.ndata["x_1_true"]).square().mean(-1)
        result["correction_x"] = (coord_error * real).sum() / real.sum().clamp_min(1)
        for feat, source in (("s", g.ndata), ("e", g.edata)):
            target = source[f"{feat}_1_true"]
            target = (target[upper] if feat == "e" else target).argmax(-1)
            support = model.vector_field.get_category_support_mask(feat)
            keep = active[feat] if support is None else active[feat] & support[target]
            # Unlike the CTMC objective, changed VISIBLE entries are supervised.
            target = target.masked_fill(~keep, -100)
            ce = F.cross_entropy(output[feat], target, reduction="none", ignore_index=-100)
            result[f"correction_{feat}"] = reduce_categorical_loss(ce, 1., keep, "masked_unweighted")
        return result
