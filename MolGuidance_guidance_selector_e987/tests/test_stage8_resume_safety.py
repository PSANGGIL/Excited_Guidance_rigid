"""Recovery must advance even when the monitored validation loss gets worse."""
import tempfile
import unittest
from pathlib import Path

import dgl
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from tools.train_stage8 import training_checkpoints, TrainingStatus


class IncreasingValidationLoss(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.save_hyperparameters({"data_path": Path("/tmp/stage8_resume_fixture")})

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("val_total_loss", self.weight.detach() * 0 + self.current_epoch + 1.,
                 batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.01)


class ResumeSafetyTests(unittest.TestCase):
    def test_latest_checkpoint_is_current_even_when_validation_worsens(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            callbacks = [*training_checkpoints(output), TrainingStatus(output)]
            # Force rejection by the monitored callback after epoch zero.
            callbacks[0].save_top_k = 1
            trainer = pl.Trainer(
                accelerator="cpu", devices=1, max_epochs=3, callbacks=callbacks,
                logger=False, enable_progress_bar=False, enable_model_summary=False,
                num_sanity_val_steps=0, default_root_dir=output)
            loader = DataLoader([dgl.graph(([], []), num_nodes=1)], batch_size=1,
                                collate_fn=dgl.batch)
            trainer.fit(IncreasingValidationLoss(), loader, loader)
            best = torch.load(callbacks[0].best_model_path, map_location="cpu", weights_only=False)
            latest = torch.load(output / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(best["epoch"], 0)
            self.assertEqual(latest["epoch"], 2)
            self.assertEqual(latest["global_step"], 3)
            self.assertTrue(latest["optimizer_states"][0]["state"])

            class RestoredState(pl.Callback):
                def on_train_start(self, trainer, model):
                    self.step = int(trainer.global_step)
                    self.weight = model.weight.detach().clone()
                    state = next(iter(trainer.optimizers[0].state.values()))
                    self.optimizer_step = int(state["step"])

            restored = RestoredState()
            resumed_output = output / "resumed"
            resumed_output.mkdir()
            resumed = pl.Trainer(
                accelerator="cpu", devices=1, max_epochs=4,
                callbacks=[*training_checkpoints(resumed_output), TrainingStatus(resumed_output), restored],
                logger=False, enable_progress_bar=False, enable_model_summary=False,
                num_sanity_val_steps=0, default_root_dir=resumed_output)
            resumed.fit(IncreasingValidationLoss(), loader, loader,
                        ckpt_path=output / "checkpoints/last.ckpt", weights_only=False)
            self.assertEqual(restored.step, 3)
            self.assertEqual(restored.optimizer_step, 3)
            torch.testing.assert_close(restored.weight, latest["state_dict"]["weight"])
            after = torch.load(resumed_output / "checkpoints/last.ckpt", map_location="cpu", weights_only=False)
            self.assertEqual(after["epoch"], 3)
            self.assertEqual(after["global_step"], 4)
            import json
            rows = [json.loads(line) for line in (resumed_output / "epoch_history.jsonl").read_text().splitlines()]
            self.assertEqual([row["epoch"] for row in rows], [3])
            self.assertEqual(rows[0]["batches"], 1)


if __name__ == "__main__":
    unittest.main()
