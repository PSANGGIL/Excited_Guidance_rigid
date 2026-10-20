# Stage 7 bounded GPU comparison — 2026-10-20

User requested an actual test after implementation, not another codec-only check.
This experiment tests early learning and rollout behavior. It does not establish
converged performance, prove global Kekulé validity, or justify a production run.

## Fixed protocol

- Same unchanged HP1KG 1,581 records and official optimized XYZ; train 1,264.
- Seven fresh arms, sequential on one B200:
  `00_stage6_control`, `01a_legacy_loss_control`, `01_contracts`,
  `01b_unweighted_loss_control`, `02_feedback`, `03_distribution`, `04_correction`.
- Full production-width models, FP32, no Stage 6 checkpoint initialization.
- 600 optimizer steps per arm; one training seed, 17041.
- Same batch size selected by a conservative GPU probe of the largest training
  molecules with correction and self-conditioning enabled.
- Probe candidates 256, 128, 64; per-process allocator cap 75% of GPU memory and
  accepted allocated peak at most 70%. This is a safety screen, not a promise of
  70% average GPU utilization or a complete worst-case memory proof.
- Common LR `0.0014 * batch_size / 640`, 100-step warmup, original Adam and
  gradient value clip 1.0. These are shared **screening overrides**, so this is
  not a reproduction of the full Stage 6 training schedule.
- Same epoch shuffle and same `(epoch, molecule index)` prior/fake augmentation.
  Reset model RNG by optimizer step. Verify first three input-batch hashes
  between arms. No claim of bitwise deterministic CUDA arithmetic.
- Fixed held-out loss/noise grid, strict rollout metrics; do not rank arms by
  total loss across different loss definitions.

## Generation and calculations

Use the **matched-step final checkpoint**, not each arm's best loss checkpoint.
Retain best-loss checkpoints separately for diagnosis.

For each arm:

1. 32 samples at validation-record EST targets, seed 17041.
2. Same 32 validation heavy-atom counts at EST=0, seed 17041.
3. Repeat both panels with seed 17042.

This gives 128 requested structures per arm and 896 across seven arms, CFG=1 and
100 time grid points (99 updates). Counts are fixed from validation for paired
comparisons, not independently drawn from a conditional count distribution.
Fake slots can change final counts; report retention and count errors.
The two generation seeds are not independent training replicates.

Preserve invalid endpoint tensors, raw sanitized-graph outcomes, fragmentation,
charge/radical status, aromaticity, ring-size warnings, novelty, count retention
and unoptimized coordinate screens. Save RDKit 2D PNGs for unique novel connected
molecules. Do not rescue failures by keeping the largest fragment, changing H
counts, re-embedding coordinates or projecting a different graph.

After all GPU arms finish, calculate eligible structures using four CPU workers,
one thread per external calculation:

```text
generated heavy XYZ
→ RDKit AddHs(addCoords=True), asserting heavy coordinates unchanged
→ xTB geometry optimization
→ xTB4sTDA
→ sTDA singlet/triplet
→ EST = S1 − T1 and target-error/yield reports
```

No ETKDG or force-field coordinate replacement. Each external command has a
600-second timeout; failures remain failures. XTB eligibility and preparation
failures are reported, not removed from the requested-sample denominator.
Geometry screens are not a physical stability certificate. sTDA values are not
identical-level substitutes for the original HP1KG labels.

## Execution and preservation

`tools/stage7_comparison.py --mode launch` creates a unique run directory, freezes
expanded configurations, records code/data hashes and starts an independent OS
session with stdin detached and file-based logs. GPU and CPU stages are child
processes of that detached worker, not children whose lifetime requires Codex.

Before each phase it checks source/data integrity. A changed source tree, failed
GPU arm, NaN or mismatched batch hashes stops the queue and records diagnostics.
It does not silently skip an arm or combine code revisions. Existing runs and
checkpoints are not overwritten.

Important files:

- `manifest.json`, `launch.json`, `status.json`, `events.jsonl`
- `probes/`, `selected_batch.json`
- `stages/<arm>/training_steps.jsonl`, `validation.jsonl`, `progress.json`
- `stages/<arm>/checkpoints/matched_steps_final.ckpt`
- `stages/<arm>/generation/<panel>_seed<seed>/.../report.json`
- generation folders' `xtb_stda/condition_results.json`
- `comparison.json`

The two-step, four-time-point harness smoke uses a separate directory. Its
untrained invalid samples are not a scientific result about model capability.

## Interpretation

The main question is whether feedback, probability-mass compatibility and
correction improve actual structural survival without destroying EST control.
Raw valence errors, aromatic-valid yield, count collapse, geometry quality,
XTB success and aromatic low-EST yield matter more than an auxiliary loss.

Do not conclude from a short, single-training-seed run that a model can never
learn. A favorable trend also needs longer matched-budget training and more
training seeds. If all arms remain untrained at 600 steps, report that the
screening budget was inconclusive instead of calling the modifications useless.
