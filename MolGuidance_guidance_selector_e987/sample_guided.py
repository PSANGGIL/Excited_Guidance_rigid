#!/usr/bin/env python3
"""Unified command-line entry point for all MolGuidance samplers.

The numerical implementations remain in the original sampling scripts. This
module validates method-specific arguments and then replaces itself with the
selected sampler process, avoiding numerical drift between duplicate loops.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path
from typing import Iterable


GUIDANCE_ALIASES = {
    "cfg": "cfg",
    "classifier-free": "cfg",
    "cg": "cg",
    "classifier": "cg",
    "mg": "mg",
    "model": "mg",
    "ag": "ag",
    "auto": "ag",
    "autoguidance": "ag",
    "conditional": "conditional",
    "none": "conditional",
}

SAMPLER_SCRIPTS = {
    "cfg": "sample_cfg.py",
    "cg": "sample_cg.py",
    "mg": "sample_mg.py",
    "ag": "sample_ag.py",
    "conditional": "sample_condition.py",
}

FEATURES = ("x", "a", "c", "e")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample molecules with classifier-free (cfg), classifier (cg), "
            "model (mg), auto (ag), or unguided conditional sampling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--guidance",
        "--guidance-type",
        dest="guidance",
        required=True,
        choices=sorted(GUIDANCE_ALIASES),
        help="Guidance method (short and descriptive aliases are accepted)",
    )

    checkpoints = parser.add_argument_group("checkpoints")
    checkpoints.add_argument(
        "--model-checkpoint", "--model_checkpoint", type=Path,
        help=(
            "CFG/MG/conditional checkpoint; also accepted as the base/good "
            "checkpoint for CG/AG"
        ),
    )
    checkpoints.add_argument(
        "--flowmol-checkpoint", "--flowmol_checkpoint", type=Path,
        help="Base FlowMol checkpoint required by classifier guidance",
    )
    checkpoints.add_argument(
        "--classifier-checkpoint", "--classifier_checkpoint", type=Path,
        help="Property-classifier checkpoint required by classifier guidance",
    )
    checkpoints.add_argument(
        "--good-model-checkpoint", "--good_model_checkpoint", type=Path,
        help="Well-trained model required by autoguidance",
    )
    checkpoints.add_argument(
        "--bad-model-checkpoint", "--bad_model_checkpoint", type=Path,
        help="Inferior/early model required by autoguidance",
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--n-mols", "--n_mols", type=int, default=1000)
    sampling.add_argument(
        "--max-batch-size", "--max_batch_size", type=int, default=128
    )
    sampling.add_argument(
        "--n-atoms-per-mol", "--n_atoms_per_mol", type=int, default=None
    )
    sampling.add_argument("--n-timesteps", "--n_timesteps", type=int, default=100)
    sampling.add_argument("--stochasticity", type=float, default=None)
    sampling.add_argument("--hc-thresh", "--hc_thresh", type=float, default=None)
    sampling.add_argument("--xt-traj", "--xt_traj", action="store_true")
    sampling.add_argument("--ep-traj", "--ep_traj", action="store_true")

    conditioning = parser.add_argument_group("conditioning")
    conditioning.add_argument("--property-name", "--property_name")
    conditioning.add_argument(
        "--properties-for-sampling",
        "--properties_for_sampling",
        type=float,
        help="One scalar target shared by all generated molecules",
    )
    conditioning.add_argument(
        "--multiple-values-file",
        "--multilple-values-file",
        "--multilple_values_file",
        dest="multiple_values_file",
        type=Path,
        help="NumPy array with one property target per generated molecule",
    )
    conditioning.add_argument(
        "--number-of-atoms-file",
        "--number_of_atoms",
        dest="number_of_atoms_file",
        type=Path,
        help="NumPy array with one atom count per generated molecule",
    )
    conditioning.add_argument(
        "--normalization-file-path",
        "--normalization_file_path",
        type=Path,
    )
    conditioning.add_argument(
        "--properties-handle-method",
        "--properties_handle_method",
        default="concatenate_sum",
        choices=[
            "concatenate",
            "sum",
            "multiply",
            "concatenate_sum",
            "concatenate_multiply",
        ],
    )
    conditioning.add_argument(
        "--dataset-name",
        "--dataset_name",
        default="qme14s",
        choices=["qm9", "qme14s"],
    )

    guidance = parser.add_argument_group("guidance controls")
    for feature in FEATURES:
        guidance.add_argument(
            f"--guide-w-{feature}",
            f"--guide_w_{feature}",
            dest=f"guide_w_{feature}",
            type=float,
            default=None,
            help=(
                "Per-feature scale for CFG/CG/AG. Defaults to 2 for CFG/AG "
                "and 1 for CG"
            ),
        )
    guidance.add_argument(
        "--guide-weight",
        "--guide_wight",
        dest="guide_weight",
        type=float,
        default=1.0,
        help="Scalar model-guidance weight (MG only)",
    )
    guidance.add_argument(
        "--dfm-type",
        "--dfm_type",
        default="campbell",
        choices=["campbell", "campbell_rate_matrix"],
        help="Categorical integration method for CFG/AG",
    )
    guidance.add_argument(
        "--guidance-format",
        "--guidance_format",
        default="linear",
        choices=["linear", "log"],
        help="Combination rule for CFG/AG rate-matrix guidance",
    )
    guidance.add_argument(
        "--where-to-apply-guide",
        "--where_to_apply_guide",
        default="probabilities",
        choices=["probabilities", "rate_matrix"],
        help="CFG/AG categorical guidance location",
    )
    guidance.add_argument(
        "--training-mode",
        "--training_mode",
        action="store_true",
        help="Forwarded only to unguided conditional sampling",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--output-file", "--output_file", type=Path)
    output.add_argument("--analyze", action="store_true")
    output.add_argument("--gpu", type=int, default=0)
    output.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and print the delegated command without running it",
    )
    return parser


def canonical_guidance(value: str) -> str:
    return GUIDANCE_ALIASES[value]


def _require_file(path: Path | None, label: str, errors: list[str]) -> None:
    if path is None:
        errors.append(f"{label} is required")
    elif not path.is_file():
        errors.append(f"{label} does not exist or is not a file: {path}")


def validate_args(args: argparse.Namespace) -> str:
    """Validate cross-argument constraints and return canonical guidance name."""
    method = canonical_guidance(args.guidance)
    errors: list[str] = []

    if args.n_mols <= 0:
        errors.append("--n-mols must be positive")
    if args.max_batch_size <= 0:
        errors.append("--max-batch-size must be positive")
    if args.n_timesteps < 2:
        errors.append("--n-timesteps must be at least 2")
    if args.n_atoms_per_mol is not None and args.n_atoms_per_mol <= 0:
        errors.append("--n-atoms-per-mol must be positive")
    if args.n_atoms_per_mol is not None and args.number_of_atoms_file is not None:
        errors.append(
            "use either --n-atoms-per-mol or --number-of-atoms-file, not both"
        )

    single_target = args.properties_for_sampling is not None
    multiple_targets = args.multiple_values_file is not None
    if single_target == multiple_targets:
        errors.append(
            "provide exactly one of --properties-for-sampling and "
            "--multiple-values-file"
        )
    if not args.property_name:
        errors.append("--property-name is required for conditional generation")

    if args.multiple_values_file is not None:
        _require_file(args.multiple_values_file, "--multiple-values-file", errors)
    if args.number_of_atoms_file is not None:
        _require_file(args.number_of_atoms_file, "--number-of-atoms-file", errors)
    if args.normalization_file_path is not None:
        _require_file(
            args.normalization_file_path, "--normalization-file-path", errors
        )

    if method in {"cfg", "mg", "conditional"}:
        _require_file(args.model_checkpoint, "--model-checkpoint", errors)
    elif method == "cg":
        _require_file(
            args.flowmol_checkpoint or args.model_checkpoint,
            "--flowmol-checkpoint (or --model-checkpoint)",
            errors,
        )
        _require_file(args.classifier_checkpoint, "--classifier-checkpoint", errors)
    elif method == "ag":
        _require_file(
            args.good_model_checkpoint or args.model_checkpoint,
            "--good-model-checkpoint (or --model-checkpoint)",
            errors,
        )
        _require_file(args.bad_model_checkpoint, "--bad-model-checkpoint", errors)

    if method != "cg" and args.classifier_checkpoint is not None:
        errors.append("--classifier-checkpoint is valid only with --guidance cg")
    if method != "ag" and (
        args.good_model_checkpoint is not None or args.bad_model_checkpoint is not None
    ):
        errors.append(
            "--good-model-checkpoint/--bad-model-checkpoint are valid only with "
            "--guidance ag"
        )
    if method != "cg" and args.flowmol_checkpoint is not None:
        errors.append("--flowmol-checkpoint is valid only with --guidance cg")
    if method != "mg" and args.guide_weight != 1.0:
        errors.append("--guide-weight is valid only with --guidance mg")
    if method not in {"cfg", "cg", "ag"} and any(
        getattr(args, f"guide_w_{feature}") is not None for feature in FEATURES
    ):
        errors.append(
            "--guide-w-x/--guide-w-a/--guide-w-c/--guide-w-e are valid only "
            "with --guidance cfg, cg, or ag"
        )
    if method != "conditional" and args.training_mode:
        errors.append("--training-mode is valid only with --guidance conditional")

    if errors:
        raise ValueError("; ".join(errors))
    return method


def _append_value(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_switch(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def _common_arguments(args: argparse.Namespace, output_file: Path) -> list[str]:
    command: list[str] = []
    _append_value(command, "--n_mols", args.n_mols)
    _append_value(command, "--max_batch_size", args.max_batch_size)
    _append_value(command, "--n_atoms_per_mol", args.n_atoms_per_mol)
    _append_value(command, "--n_timesteps", args.n_timesteps)
    _append_value(command, "--stochasticity", args.stochasticity)
    _append_value(command, "--hc_thresh", args.hc_thresh)
    _append_switch(command, "--xt_traj", args.xt_traj)
    _append_switch(command, "--ep_traj", args.ep_traj)
    _append_value(command, "--property_name", args.property_name)
    _append_value(
        command, "--properties_for_sampling", args.properties_for_sampling
    )
    if args.multiple_values_file is not None:
        _append_value(
            command,
            "--multilple_values_file",
            _resolved(args.multiple_values_file),
        )
    if args.number_of_atoms_file is not None:
        _append_value(
            command, "--number_of_atoms", _resolved(args.number_of_atoms_file)
        )
    if args.normalization_file_path is not None:
        _append_value(
            command,
            "--normalization_file_path",
            _resolved(args.normalization_file_path),
        )
    _append_value(command, "--output_file", _resolved(output_file))
    _append_switch(command, "--analyze", args.analyze)
    _append_value(command, "--gpu", args.gpu)
    return command


def build_dispatch(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Return canonical method and the delegated Python command."""
    method = validate_args(args)
    script = Path(__file__).resolve().parent / SAMPLER_SCRIPTS[method]
    output_file = args.output_file or (
        Path.cwd() / f"{method}_sampled_molecules.sdf"
    )
    forwarded = _common_arguments(args, output_file)

    if method in {"cfg", "mg", "conditional"}:
        forwarded[0:0] = ["--model_checkpoint", _resolved(args.model_checkpoint)]
    elif method == "cg":
        base_checkpoint = args.flowmol_checkpoint or args.model_checkpoint
        forwarded[0:0] = [
            "--flowmol_checkpoint",
            _resolved(base_checkpoint),
            "--classifier_checkpoint",
            _resolved(args.classifier_checkpoint),
        ]
    elif method == "ag":
        good_checkpoint = args.good_model_checkpoint or args.model_checkpoint
        forwarded[0:0] = [
            "--good_model_checkpoint",
            _resolved(good_checkpoint),
            "--bad_model_checkpoint",
            _resolved(args.bad_model_checkpoint),
        ]

    if method in {"cfg", "ag"}:
        default_scale = 2.0
        for feature in FEATURES:
            value = getattr(args, f"guide_w_{feature}")
            _append_value(
                forwarded,
                f"--guide_w_{feature}",
                default_scale if value is None else value,
            )
        _append_value(forwarded, "--dfm_type", args.dfm_type)
        _append_value(forwarded, "--guidance_format", args.guidance_format)
        _append_value(
            forwarded, "--where_to_apply_guide", args.where_to_apply_guide
        )
        _append_value(
            forwarded, "--properties_handle_method", args.properties_handle_method
        )
        _append_value(forwarded, "--dataset_name", args.dataset_name)
    elif method == "cg":
        default_scale = 1.0
        for feature in FEATURES:
            value = getattr(args, f"guide_w_{feature}")
            _append_value(
                forwarded,
                f"--guide_w_{feature}",
                default_scale if value is None else value,
            )
    elif method == "mg":
        _append_value(forwarded, "--guide_wight", args.guide_weight)
        _append_value(
            forwarded, "--properties_handle_method", args.properties_handle_method
        )
        _append_value(forwarded, "--dataset_name", args.dataset_name)
    elif method == "conditional":
        _append_value(
            forwarded, "--properties_handle_method", args.properties_handle_method
        )
        _append_switch(forwarded, "--training_mode", args.training_mode)

    return method, [sys.executable, "-u", str(script), *forwarded]


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        method, command = build_dispatch(args)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Selected guidance: {method}", flush=True)
    print(f"Delegating command: {shlex.join(command)}", flush=True)
    if args.dry_run:
        return

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
