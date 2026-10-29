"""Condition bookkeeping regressions; no quantum calculations are launched."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


PATH = Path(__file__).resolve().parents[1] / "tools/run_stage8_xtb_stda.py"
SPEC = importlib.util.spec_from_file_location("stage8_xtb_driver", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class XtbTargetTests(unittest.TestCase):
    def test_both_manifest_schemas(self):
        est = dict(requested=1000, checkpoint_epoch_zero_based=3911, target_est_eV=0.)
        s1 = dict(n=1000, checkpoint_epoch=10, target_s1_eV=4., property="s1")
        self.assertEqual(driver.generation_spec(est)["condition_name"], "est")
        self.assertEqual(driver.generation_spec(s1)["condition_name"], "s1")
        self.assertEqual(driver.generation_spec(s1, target_s1=4.)["checkpoint_epoch"], 10)
        for manifest, kwargs in ((s1, dict(target_est=0.)), (est, dict(target_s1=4.)),
                                 (s1, dict(target_s1=3.)), (s1, dict(target_s1=float("nan")))):
            with self.assertRaises(ValueError):
                driver.generation_spec(manifest, **kwargs)

    def test_reject_ambiguous_or_incomplete_manifest(self):
        for manifest in (
            dict(requested=1000, checkpoint_epoch_zero_based=0),
            dict(n=1000, checkpoint_epoch=0, target_est_eV=0., target_s1_eV=4.),
            dict(n=1000, checkpoint_epoch=0, target_s1_eV=4., property="est"),
            dict(n=1000, checkpoint_epoch=0, target_s1_eV=float("inf")),
            dict(n=0, checkpoint_epoch=0, target_s1_eV=4.),
            dict(n=1000, target_s1_eV=4.),
        ):
            with self.assertRaises(ValueError):
                driver.generation_spec(manifest)

    @staticmethod
    def records():
        return [dict(generation_index=0, raw_valid=True, connected=True, eligible=True,
                     radicals=0, net_charge=0),
                dict(generation_index=1, raw_valid=False, connected=False, eligible=False)]

    def test_est_exact_boundary_and_full_ledger(self):
        # Binary subtraction gives 0.10000000000000009; the reported gap is 0.100.
        result = {0: dict(generation_index=0, status="ok", s1_energy_eV=2.2, t1_energy_eV=2.1)}
        with tempfile.TemporaryDirectory() as tmp:
            summary = driver.summarize(Path(tmp), self.records(), result, 0., finished=True)
        self.assertEqual(summary["hits"]["abs_error_le_0p1"], 1)
        self.assertEqual(summary["est_mae_eV"], .1)
        self.assertIsNone(summary["target_s1_eV"])
        self.assertIsNone(summary["s1_mae_eV"])
        self.assertEqual(summary["full_outcome_counts"], {"ok": 1, "not_attempted_invalid_graph": 1})

    def test_s1_does_not_invent_est_or_t1_targets(self):
        result = {0: dict(generation_index=0, status="ok", s1_energy_eV=3.9, t1_energy_eV=1.2)}
        with tempfile.TemporaryDirectory() as tmp:
            summary = driver.summarize(Path(tmp), self.records(), result, 4., condition_name="s1")
        self.assertEqual(summary["target_s1_eV"], 4.)
        self.assertEqual(summary["s1_mae_eV"], .1)
        self.assertEqual(summary["est_mean_eV"], 2.7)
        self.assertIsNone(summary["target_est_eV"])
        self.assertIsNone(summary["target_t1_eV"])
        self.assertIsNone(summary["est_mae_eV"])
        self.assertEqual(summary["hits"]["abs_error_le_0p1"], 1)

    def test_no_successes_are_pending_not_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = driver.summarize(Path(tmp), self.records(), {}, 4., condition_name="s1")
        self.assertEqual(summary["xtb_stda_successes"], 0)
        self.assertIsNone(summary["s1_mae_eV"])
        self.assertEqual(summary["full_outcome_counts"]["pending"], 1)
        self.assertEqual(summary["hits"]["abs_error_le_0p1"], 0)

    def test_s1_minus_2t1_is_not_est_and_checks_target_identity(self):
        manifest=dict(requested=1000,checkpoint_epoch_zero_based=4690,
            target_s1_minus_2t1_eV=0.,property="s1_minus_2t1",condition_name="s1_minus_2t1")
        self.assertEqual(driver.generation_spec(manifest,target_s1_minus_2t1=0.)["condition_name"],"s1_minus_2t1")
        with self.assertRaises(ValueError):
            driver.generation_spec(manifest,target_est=0.)
        with self.assertRaises(ValueError):
            driver.generation_spec({**manifest,"target_est_eV":0.})
        with self.assertRaises(ValueError):
            driver.generation_spec({**manifest,"condition_name":"est"})
        result={0:dict(generation_index=0,status="ok",s1_energy_eV=3.1,t1_energy_eV=1.5)}
        with tempfile.TemporaryDirectory() as tmp:
            summary=driver.summarize(Path(tmp),self.records(),result,0.,condition_name="s1_minus_2t1")
        self.assertEqual(summary["condition_mae_eV"],.1)
        self.assertEqual(summary["s1_minus_2t1_mean_eV"],.1)
        self.assertEqual(summary["est_mean_eV"],1.6)
        self.assertEqual(summary["hits"]["abs_error_le_0p1"],1)
        self.assertIsNone(summary["est_mae_eV"])
        self.assertIsNone(summary["target_est_eV"])
        self.assertIsNone(summary["target_s1_eV"])
        self.assertEqual(summary["target_s1_minus_2t1_eV"],0.)


if __name__ == "__main__":
    unittest.main()
