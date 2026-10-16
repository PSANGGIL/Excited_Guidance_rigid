# Joint-chemistry/self-correction implementation log

This log records each architecture change independently so later regressions can
be attributed to one stage rather than to a bundle of simultaneous changes.

## Baseline

- Branch point: `a4c3f47`
- Baseline checkpoint: `epoch=1888-step=3778.ckpt`
- Baseline state: heavy-atom Cartesian flow plus independent atom type, formal
  charge, attached-H count, and five-class bond CTMC heads.
- Baseline generation bottlenecks are recorded in
  `runs/cfg_official_xyz_best_e1888_est0_n1000_xtb_20261016/detailed_analysis.md`.

## Stage 1: deterministic validation and conditional atom count

### Motivation

The previous validation metric mixed three random variables: sampled time,
categorical corruption, and classifier-free condition dropout.  Checkpoint
ordering could therefore change without a parameter update.  Generation also
sampled atom count from `p(n)` even when a fixed EST target was supplied, which
breaks the empirical joint distribution `p(n, c)`.  MolGuidance evaluates joint
size/property support, while this branch now explicitly samples `p(n | c)`.

References:

- MolGuidance: <https://arxiv.org/abs/2512.12198>
- Classifier-free guidance: <https://arxiv.org/abs/2207.12598>

### Changes

1. Validation uses the fixed grid `[0.05, 0.25, 0.5, 0.75, 0.95]`.
2. Categorical corruption uses a batch/grid-derived `torch.Generator` seed.
3. CFG condition dropout is disabled in validation but unchanged in training.
4. `val_condition_shuffle_gain` compares the true and one-step-rolled
   conditions using the exact same midpoint corruption.
5. A kernel estimator samples the empirical `p(n_atoms | condition)`.  The
   implementation supports vector conditions for the later S1/EST/fS1 stage.
6. The data configuration explicitly selects `training_split: train`; the
   legacy `train_a/train_b` flag remains backward compatible.

### Invariants and conflict checks

- No parameter tensor or state-dict key was added in this stage.
- Existing checkpoints default to the old marginal count mode and load without
  missing/unexpected keys.  Conditional count can be selected at inference.
- CTMC noising gained an optional generator; calls without it are unchanged.
- Endpoint/Dirichlet vector fields accept the same optional argument so the
  shared FlowMol API does not break non-CTMC configurations.
- Explicit atom-count files override either statistical count mode.
- Training condition dropout remains governed by `p_uncond`.

### Verification (2026-10-16)

- 17 unit tests passed, including existing EMA/edge-offset and sampler CLI tests.
- Existing H-count codec tests passed.
- HP1KG smoke model validation repeated exactly:
  `repeat_max_abs = 0.0` for every feature and identical shuffle gain.
- Explicit split resolved to `train`, with 1,264 train and 158 validation records.
- With identical random seed and 10,000 draws:
  - marginal mean/median count: `26.421 / 25`
  - EST=0 conditional mean/median count: `28.478 / 28`
- The baseline epoch-1888 checkpoint loaded with all 7,097,483 parameters and
  retained `mode=marginal`, confirming state-dict compatibility.

### Interpretation limit

The condition-shuffle value from the randomly initialized smoke model validates
determinism only; it is not evidence of learned condition fidelity.  Fidelity
must be measured on trained checkpoints over multiple target bins.
