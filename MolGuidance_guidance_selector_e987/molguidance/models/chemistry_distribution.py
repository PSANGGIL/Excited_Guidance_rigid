"""Factorized compatibility probabilities, not a global molecular solver."""
import torch
import torch.nn.functional as F

from molguidance.data_processing.utils import get_batch_idxs


def bounded_sum_distribution(probabilities, maximum):
    """Convolve independent non-negative integer variables, with overflow.

    Input [batch, variables, categories], category k has integer value k.
    Output bins 0..maximum AND an absorbing >maximum bin. No mass is discarded
    or renormalized. Padding variables must be delta(value=0).
    """
    if probabilities.ndim != 3 or maximum < 0 or probabilities.shape[-1] < 1:
        raise ValueError("invalid discrete convolution dimensions")
    n = probabilities.shape[0]
    dist = probabilities.new_zeros(n, maximum + 1)
    dist[:, 0] = 1.
    overflow = probabilities.new_zeros(n)
    for variable in probabilities.unbind(1):
        terms, tails = [], []
        for value in range(probabilities.shape[-1]):
            p = variable[:, value:value + 1]
            if value == 0:
                terms.append(dist * p)
            elif value <= maximum:
                terms.append(F.pad(dist[:, :-value], (value, 0)) * p)
                tails.append(dist[:, -value:].sum(-1) * p[:, 0])
            else:
                tails.append(dist.sum(-1) * p[:, 0])
        overflow = overflow * variable.sum(-1)
        if tails:
            overflow = overflow + torch.stack(tails).sum(0)
        dist = torch.stack(terms).sum(0) if terms else torch.zeros_like(dist)
    return torch.cat((dist, overflow[:, None]), -1)


def compatibility_probabilities(g, node, edge, upper, allowed_valence, charges, true_charge):
    """Local valence and global net-charge probabilities under head marginals."""
    node_batch, _ = get_batch_idxs(g)
    sizes = g.batch_num_nodes().to(node.device)
    max_nodes = int(sizes.max())
    offsets = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)[:-1]))
    local = torch.arange(g.num_nodes(), device=node.device) - offsets[node_batch]
    src, dst = g.edges()
    src, dst = src[upper], dst[upper]
    degree = max(max_nodes - 1, 0)
    incident = edge.new_zeros(g.num_nodes(), degree, 4)
    incident[:, :, 0] = 1.
    src_slot = local[dst] - (local[dst] > local[src]).long()
    dst_slot = local[src] - (local[src] > local[dst]).long()
    incident[src, src_slot] = edge
    incident[dst, dst_slot] = edge
    valence_dist = bounded_sum_distribution(incident, allowed_valence.shape[1] - 1)
    allowed = node @ allowed_valence.to(node)
    local_probability = (valence_dist[:, :-1] * allowed).sum(-1)

    # Shift signed charge support to non-negative integers for convolution.
    # Padding is physical charge ZERO, never the first/minimum charge category.
    rounded = charges.round().long()
    if not torch.equal(charges, rounded.to(charges)):
        raise ValueError("formal charges must be integers")
    qmin = min(int(rounded.min()), 0)
    qmax = max(int(rounded.max()), 0)
    charge_prob = node.new_zeros(g.num_nodes(), qmax - qmin + 1)
    charge_prob.index_add_(1, rounded - qmin, node)
    padded = node.new_zeros(g.batch_size, max_nodes, charge_prob.shape[-1])
    padded[:, :, -qmin] = 1.
    padded[node_batch, local] = charge_prob
    charge_dist = bounded_sum_distribution(padded, max_nodes * (qmax - qmin))
    target = true_charge.round().long() - max_nodes * qmin
    if not torch.equal(true_charge, true_charge.round()):
        raise ValueError("target net charges must be integers")
    if bool(((target < 0) | (target >= charge_dist.shape[-1] - 1)).any()):
        raise ValueError("target net charge is outside the representable range")
    global_probability = charge_dist.gather(1, target[:, None])[:, 0]
    return local_probability, global_probability
