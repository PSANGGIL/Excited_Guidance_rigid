"""Complete-graph batch budget, including the maximum fake-atom capacity.

Budget counts DIRECTED edges n(n-1), not chemical bonds or upper triangles.
Each example appears exactly once. Fixed cost-bin packing keeps epoch length
stable for Lightning/LR scheduling; equal-cost items and batch order reshuffle.
"""
import math

import torch
from torch.utils.data import Sampler


class EdgeBudgetBatchSampler(Sampler):
    def __init__(self, counts, batch_size, max_edges, fake_fraction=0.,
                 shuffle=True, seed=17041):
        counts = torch.as_tensor(counts)
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError("counts must be a nonempty vector")
        if not torch.isfinite(counts.float()).all() or not torch.equal(counts, counts.long()):
            raise ValueError("counts must be finite integers")
        if (counts < 1).any() or batch_size < 1 or max_edges < 1:
            raise ValueError("positive atom counts, batch size and edge budget required")
        if not math.isfinite(fake_fraction) or not 0 <= fake_fraction <= 1:
            raise ValueError("fake_fraction must be in [0, 1]")
        self.batch_size, self.max_edges = int(batch_size), int(max_edges)
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        slots = counts.long() + torch.ceil(counts.double() * fake_fraction).long()
        self.costs = slots * (slots - 1)
        if int(self.costs.max()) > self.max_edges:
            raise ValueError("one molecule exceeds max_num_edges, including fake slots")
        self.groups = {int(c): torch.where(self.costs == c)[0].tolist()
                       for c in torch.unique(self.costs)}
        # First-fit decreasing packing. Store costs, not identities: identities
        # of equal-cost molecules can be permuted without breaking the bound.
        bins, totals = [], []
        for cost in sorted(self.costs.tolist(), reverse=True):
            for batch, total_index in zip(bins, range(len(bins))):
                if len(batch) < self.batch_size and totals[total_index] + cost <= self.max_edges:
                    batch.append(cost)
                    totals[total_index] += cost
                    break
            else:
                bins.append([cost])
                totals.append(cost)
        self.bins = bins

    @classmethod
    def from_dataset(cls, dataset, batch_size, max_edges, **kwargs):
        counts = dataset.node_idx_array[:, 1] - dataset.node_idx_array[:, 0]
        return cls(counts, batch_size, max_edges,
                   fake_fraction=dataset.fake_atom_fraction, **kwargs)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.bins)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        pools = {}
        for cost, indices in self.groups.items():
            order = torch.randperm(len(indices), generator=generator).tolist() if self.shuffle else range(len(indices))
            pools[cost] = [indices[i] for i in order]
        batches = [[pools[cost].pop() for cost in costs] for costs in self.bins]
        order = torch.randperm(len(batches), generator=generator).tolist() if self.shuffle else range(len(batches))
        for index in order:
            yield batches[index]
