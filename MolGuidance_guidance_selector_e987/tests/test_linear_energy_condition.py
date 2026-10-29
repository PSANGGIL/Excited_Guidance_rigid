"""Prevent EST / S1-2*T1 relabeling and fixed-zero training mistakes."""
import tempfile
from pathlib import Path
import unittest
import torch

from molguidance_hp1kg_baseline.convert_hp1kg_joint_kekule import property_value
from molguidance.analysis.generation_callback import GenerationValidation
from tests.test_stage7_evaluation import _PanelProbe


class LinearEnergyConditionTests(unittest.TestCase):
    def test_distinct_formulas_and_finite_values(self):
        record=dict(s1=4.0,t1=2.0)
        self.assertEqual(property_value(record,"s1_minus_2t1"),0.)
        self.assertEqual(property_value(record,"est"),2.)
        self.assertEqual(property_value(record,"s1"),4.)
        self.assertAlmostEqual(property_value(dict(s1=3.,t1=2.),"s1_minus_2t1"),-1.)
        with self.assertRaises(ValueError):
            property_value(dict(s1=float("nan"),t1=2.),"s1_minus_2t1")

    def test_sampling_and_metadata_use_new_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for split in ("train","val","test"):
                torch.save(dict(smiles=["O","CO"],node_idx_array=torch.tensor([[0,1],[1,3]]),
                    properties=torch.tensor([-3.,.2]),property_names="s1_minus_2t1"),path/f"{split}_data_processed.pt")
            model=_PanelProbe().train()
            callback=GenerationValidation(path/"out",path,n_samples=2,batch_size=1,n_timesteps=5,
                property_name="s1_minus_2t1",target_value_eV=0.)
            report=callback.evaluate(model,0,1)
            self.assertEqual(report["condition_name"],"s1_minus_2t1")
            self.assertNotIn("fixed_target_est_eV",report)
            self.assertTrue(all(r["target_s1_minus_2t1_eV"]==0. and "target_est_eV" not in r for r in report["records"]))
            self.assertTrue(all(a["property_name"]=="s1_minus_2t1" for a in model.arguments))
            stored=torch.load(path/"train_data_processed.pt",weights_only=False)
            torch.testing.assert_close(stored["properties"],torch.tensor([-3.,.2]))
            wrong=GenerationValidation(path/"wrong",path,n_samples=2)
            with self.assertRaisesRegex(ValueError,"differs from stored"):
                wrong.evaluate(model,0,1)

    def test_legacy_target_cannot_relabel_a_new_property(self):
        with self.assertRaises(ValueError):
            GenerationValidation("unused","unused",property_name="s1_minus_2t1",target_est_eV=0.)


if __name__ == "__main__":
    unittest.main()
