import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import dgl

from molguidance.data_processing.dataset import collate
from molguidance.data_processing.dataset import MoleculeDataset
from molguidance.data_processing.samplers import SameSizeMoleculeSampler, SameSizeDistributedMoleculeSampler
from molguidance.data_processing.edge_budget import EdgeBudgetBatchSampler

class MoleculeDataModule(pl.LightningDataModule):

    def __init__(self, dataset_config: dict, dm_prior_config: dict, batch_size: int, num_workers: int = 0, distributed: bool = False, max_num_edges: int | None = None):
        super().__init__()
        self.distributed = distributed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prior_config = dm_prior_config
        self.max_num_edges = max_num_edges
        self.save_hyperparameters()

    def setup(self, stage: str):

        if stage == 'fit':
            self.val_dataset = self.load_dataset('val')
            self.test_dataset = self.load_dataset('test')

            explicit_split = self.dataset_config.get('training_split')
            if explicit_split is not None:
                if explicit_split not in {'train', 'train_a', 'train_b'}:
                    raise ValueError(
                        "training_split must be one of 'train', 'train_a', or 'train_b'"
                    )
                training_split = explicit_split
            elif self.dataset_config.get('use_first_half_training_set', False):
                training_split = 'train_a'
            else:
                training_split = 'train_b'
            self.training_split = training_split
            self.train_dataset = self.load_dataset(training_split)

    def load_dataset(self, dataset_name: str):
        return MoleculeDataset(dataset_name, self.dataset_config, prior_config=self.prior_config)

    def train_dataloader(self):
        if self.max_num_edges is not None:
            return self._budget_loader(self.train_dataset, self.batch_size, True)
        dataloader = DataLoader(self.train_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                collate_fn=collate, 
                                num_workers=self.num_workers)

        return dataloader
    
    def val_dataloader(self):
        if self.max_num_edges is not None:
            return self._budget_loader(self.val_dataset, self.batch_size * 2, False)
        dataloader = DataLoader(self.val_dataset, 
                                batch_size=self.batch_size*2, 
                                shuffle=False, 
                                collate_fn=collate, 
                                num_workers=self.num_workers)
        return dataloader

    def test_dataloader(self):
        if self.max_num_edges is not None:
            return self._budget_loader(self.test_dataset, self.batch_size * 2, False)
        dataloader = DataLoader(self.test_dataset,
                                batch_size=self.batch_size*2, 
                                shuffle=False, 
                                collate_fn=collate, 
                                num_workers=self.num_workers)
        return dataloader

    def _budget_loader(self, dataset, batch_size, shuffle):
        if self.distributed:
            raise ValueError("edge-budget loader is single-device; refusing duplicated DDP data")
        sampler = EdgeBudgetBatchSampler.from_dataset(
            dataset, batch_size, self.max_num_edges, shuffle=shuffle,
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                          num_workers=self.num_workers)
