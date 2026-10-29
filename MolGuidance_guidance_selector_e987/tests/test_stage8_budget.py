import unittest

import torch

from molguidance.data_processing.edge_budget import EdgeBudgetBatchSampler


class BudgetTests(unittest.TestCase):
    def test_coverage_bound_stable_length_and_reproducible_shuffle(self):
        counts = [1, 2, 2, 2, 2, 5, 5, 6, 8, 9, 9, 10]
        sampler = EdgeBudgetBatchSampler(counts, 4, 210, fake_fraction=.2)
        first = list(sampler)
        for epoch in range(10):
            sampler.set_epoch(epoch)
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(counts))))
            for batch in batches:
                self.assertTrue(0 < len(batch) <= 4)
                self.assertLessEqual(int(sampler.costs[batch].sum()), 210)
            self.assertEqual(batches, list(sampler))
        self.assertNotEqual(first, list(sampler))

    def test_oversized_fake_capacity_fails_before_training(self):
        with self.assertRaisesRegex(ValueError, "one molecule"):
            EdgeBudgetBatchSampler([10], 4, 100, fake_fraction=.2)

    def test_one_atom_empty_edge_and_validation_order(self):
        sampler = EdgeBudgetBatchSampler([1, 1, 2], 2, 2, shuffle=False)
        self.assertEqual(list(sampler), list(sampler))
        self.assertEqual(sorted(sum(list(sampler), [])), [0, 1, 2])

    def test_invalid_counts(self):
        for counts in ([], [0], [2.5], [float("nan")]):
            with self.assertRaises(ValueError):
                EdgeBudgetBatchSampler(counts, 2, 100)


if __name__ == "__main__":
    unittest.main()
