import unittest

import torch
from torch.utils.data import Dataset, DataLoader

from tools.stage7_comparison import PairedDataset, PairedBatches


class _RandomDataset(Dataset):
    def __len__(self):
        return 13

    def __getitem__(self, index):
        return index, torch.randn(3)


class ComparisonTests(unittest.TestCase):
    def test_paired_priors_ignore_model_rng_but_change_between_epochs(self):
        dataset = PairedDataset(_RandomDataset(), 17041)
        state = torch.random.get_rng_state().clone()
        first = dataset[(0, 5)][1]
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        torch.manual_seed(999)
        torch.randn(173)
        self.assertTrue(torch.equal(first, dataset[(0, 5)][1]))
        self.assertFalse(torch.equal(first, dataset[(1, 5)][1]))

    def test_order_and_batch_membership_match_between_arms(self):
        def get_epochs():
            sampler = PairedBatches(13, 4, 71)
            loader = DataLoader(PairedDataset(_RandomDataset(), 81), batch_sampler=sampler)
            return [[(ids.tolist(), values.clone()) for ids, values in loader] for _ in range(2)]
        first = get_epochs()
        torch.manual_seed(47)
        second = get_epochs()
        for epoch_a, epoch_b in zip(first, second):
            ids = []
            for (ia, va), (ib, vb) in zip(epoch_a, epoch_b):
                self.assertEqual(ia, ib)
                torch.testing.assert_close(va, vb)
                ids.extend(ia)
            self.assertEqual(sorted(ids), list(range(13)))


if __name__ == "__main__":
    unittest.main()
