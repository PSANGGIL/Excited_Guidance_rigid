import os
import tempfile
import unittest
from pathlib import Path

from sample_guided import build_dispatch, build_parser, validate_args


class GuidedSamplerCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def touch(self, name: str) -> Path:
        path = self.root / name
        path.write_bytes(b"checkpoint")
        return path

    @staticmethod
    def parse(*items: str):
        return build_parser().parse_args(list(items))

    def test_cfg_dispatch_and_method_specific_defaults(self):
        checkpoint = self.touch("cfg.ckpt")
        normalization = self.touch("normalization.pt")
        previous_directory = Path.cwd()
        os.chdir(self.root)
        try:
            args = self.parse(
                "--guidance",
                "cfg",
                "--model-checkpoint",
                str(checkpoint),
                "--property-name",
                "s1",
                "--properties-for-sampling",
                "3.64",
                "--normalization-file-path",
                str(normalization),
                "--dry-run",
            )
            method, command = build_dispatch(args)
        finally:
            os.chdir(previous_directory)

        self.assertEqual(method, "cfg")
        self.assertTrue(command[2].endswith("sample_cfg.py"))
        for feature in ("x", "a", "c", "e"):
            position = command.index(f"--guide_w_{feature}")
            self.assertEqual(command[position + 1], "2.0")
        output_position = command.index("--output_file")
        self.assertEqual(
            command[output_position + 1],
            str((self.root / "cfg_sampled_molecules.sdf").resolve()),
        )

    def test_classifier_guidance_accepts_generic_base_checkpoint(self):
        base = self.touch("flowmol.ckpt")
        classifier = self.touch("classifier.ckpt")
        args = self.parse(
            "--guidance",
            "classifier",
            "--model-checkpoint",
            str(base),
            "--classifier-checkpoint",
            str(classifier),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
        )

        method, command = build_dispatch(args)

        self.assertEqual(method, "cg")
        self.assertTrue(command[2].endswith("sample_cg.py"))
        self.assertEqual(
            command[command.index("--flowmol_checkpoint") + 1],
            str(base.resolve()),
        )
        for feature in ("x", "a", "c", "e"):
            position = command.index(f"--guide_w_{feature}")
            self.assertEqual(command[position + 1], "1.0")

    def test_autoguidance_requires_inferior_checkpoint(self):
        good = self.touch("good.ckpt")
        args = self.parse(
            "--guidance",
            "ag",
            "--good-model-checkpoint",
            str(good),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
        )

        with self.assertRaisesRegex(ValueError, "bad-model-checkpoint"):
            validate_args(args)

    def test_autoguidance_dispatches_both_models(self):
        good = self.touch("good.ckpt")
        bad = self.touch("bad.ckpt")
        args = self.parse(
            "--guidance",
            "autoguidance",
            "--good-model-checkpoint",
            str(good),
            "--bad-model-checkpoint",
            str(bad),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
        )

        method, command = build_dispatch(args)

        self.assertEqual(method, "ag")
        self.assertTrue(command[2].endswith("sample_ag.py"))
        self.assertEqual(
            command[command.index("--good_model_checkpoint") + 1],
            str(good.resolve()),
        )
        self.assertEqual(
            command[command.index("--bad_model_checkpoint") + 1],
            str(bad.resolve()),
        )

    def test_rejects_ambiguous_property_targets(self):
        checkpoint = self.touch("cfg.ckpt")
        targets = self.touch("targets.npy")
        args = self.parse(
            "--guidance",
            "cfg",
            "--model-checkpoint",
            str(checkpoint),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
            "--multiple-values-file",
            str(targets),
        )

        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_args(args)

    def test_model_guidance_forwards_corrected_cli_spelling(self):
        checkpoint = self.touch("mg.ckpt")
        args = self.parse(
            "--guidance-type",
            "model",
            "--model-checkpoint",
            str(checkpoint),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
            "--guide-weight",
            "1.75",
        )

        method, command = build_dispatch(args)

        self.assertEqual(method, "mg")
        self.assertTrue(command[2].endswith("sample_mg.py"))
        position = command.index("--guide_wight")
        self.assertEqual(command[position + 1], "1.75")

    def test_conditional_sampling_dispatch(self):
        checkpoint = self.touch("conditional.ckpt")
        args = self.parse(
            "--guidance",
            "none",
            "--model-checkpoint",
            str(checkpoint),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
            "--training-mode",
        )

        method, command = build_dispatch(args)

        self.assertEqual(method, "conditional")
        self.assertTrue(command[2].endswith("sample_condition.py"))
        self.assertIn("--training_mode", command)

    def test_rejects_feature_weights_for_model_guidance(self):
        checkpoint = self.touch("mg.ckpt")
        args = self.parse(
            "--guidance",
            "mg",
            "--model-checkpoint",
            str(checkpoint),
            "--property-name",
            "s1",
            "--properties-for-sampling",
            "3.64",
            "--guide-w-x",
            "2",
        )

        with self.assertRaisesRegex(ValueError, "valid only"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
