"""Pipeline orchestration and novel-only result export; no GPU or XTB jobs."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "tools/run_stage8_generation_xtb.py"
SPEC = importlib.util.spec_from_file_location("stage8_generation_xtb", PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class PipelineTests(unittest.TestCase):
    def test_atomic_json_and_digest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.json"
            driver.json_file(path, {"phase": "test", "n": 10000})
            self.assertEqual(json.loads(path.read_text())["n"], 10000)
            self.assertFalse(path.with_suffix(".json.tmp").exists())
            self.assertEqual(len(driver.digest(path)), 64)

    def test_failed_stage_raises_and_records_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = {}
            with patch.object(driver.subprocess, "Popen") as popen:
                popen.return_value.pid = 1234
                popen.return_value.wait.return_value = 2
                with self.assertRaises(RuntimeError):
                    driver.stage(["unused"], root, state, "generating", root / "stage.log", {})
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
                self.assertEqual(popen.call_args.kwargs["stdin"], driver.subprocess.DEVNULL)
            self.assertEqual(json.loads((root / "pipeline_status.json").read_text())["stage_exit_code"], 2)

    @staticmethod
    def fixture(root):
        raw = root / "raw_analysis"
        raw.mkdir()
        records = [dict(generation_index=i, eligible=True, db_member=db, smiles=smi,
                        aromatic=False, ring_sizes=[]) for i, (smi, db) in enumerate(
                            [("CC", False), ("CC", False), ("CCC", True), ("CO", False)])]
        driver.json_file(raw / "records.json", records)
        calc = root / "xtb"
        calc.mkdir()
        driver.json_file(calc / "summary.json", dict(complete=True, full_outcome_counts={"ok": 4}, xtb_stda_successes=4))
        (calc / "all_1000_outcomes.csv").write_text("generation_index,status\n")
        driver.json_file(calc / "source_all_1000_records.json", records)
        results = [dict(generation_index=r["generation_index"], status="ok", smiles=r["smiles"],
                        s1_energy_eV="2.200", t1_energy_eV="2.100", calculated_est_eV="0.100") for r in records]
        with (calc / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        return calc

    def test_exports_unique_novel_and_exact_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calc = self.fixture(root)
            with patch.object(driver, "render_spaced") as render:
                report = driver.postprocess(root, calc, 4)
            self.assertEqual(report["novel_unique_successes"], 2)
            self.assertEqual(report["excluded_db_matches"], 1)
            self.assertEqual(report["excluded_duplicate_successes"], 1)
            self.assertEqual(report["exact_abs_est_le_0p1_novel_unique"], 2)
            self.assertEqual(render.call_count, 2)
            self.assertTrue((calc / "all_4_outcomes.csv").exists())
            self.assertTrue((calc / "all_1000_outcomes.csv").exists())

    def test_rejects_incomplete_outcome_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            calc = self.fixture(root)
            driver.json_file(calc / "summary.json", dict(complete=True, full_outcome_counts={"ok": 3}, xtb_stda_successes=3))
            with self.assertRaises(ValueError):
                driver.postprocess(root, calc, 4)

    def test_renderer_uses_expected_spaced_layout(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "panel.png"
            rows = [dict(generation_index=42, canonical_smiles="c1ccccc1", s1_energy_eV="2.5",
                         t1_energy_eV="2.5", calculated_est_eV="0.0", abs_est_error_eV="0.0")]
            driver.render_spaced(rows, path, "Smoke test")
            with Image.open(path) as picture:
                self.assertEqual(picture.size, (3400, 840))

    def test_new_condition_export_does_not_rank_by_est(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            calc=self.fixture(root)
            summary=json.loads((calc/"summary.json").read_text())
            summary.update(condition_name="s1_minus_2t1",target_value_eV=0.)
            driver.json_file(calc/"summary.json",summary)
            with (calc/"results.csv").open() as handle:
                rows=list(csv.DictReader(handle))
            for row in rows:
                row.update(condition_name="s1_minus_2t1",t1_energy_eV="1.050",calculated_est_eV="1.150",
                           calculated_condition_eV="0.100")
            with (calc/"results.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with patch.object(driver,"render_spaced") as render:
                report=driver.postprocess(root,calc,4,"s1_minus_2t1",0.)
            self.assertEqual(report["exact_abs_condition_error_le_0p1_novel_unique"],2)
            self.assertNotIn("exact_abs_est_le_0p1_novel_unique",report)
            self.assertEqual(report["condition_name"],"s1_minus_2t1")
            self.assertEqual(render.call_args.args[3],"s1_minus_2t1")
            with (calc/"rdkit_2d_novel_unique_spaced/molecules.csv").open() as handle:
                exported=list(csv.DictReader(handle))
            self.assertTrue(all(row["abs_est_error_eV"] == "" for row in exported))
            self.assertTrue(all(float(row["calculated_condition_eV"])==.1 for row in exported))

    def test_property_manifest_rejects_est_data_for_s1_minus_2t1(self):
        with self.assertRaises(ValueError):
            driver.require_property("s1_minus_2t1","est")
        driver.require_property("s1_minus_2t1","s1_minus_2t1")


if __name__ == "__main__":
    unittest.main()
