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

## Stage 2: joint atom chemistry and Kekule edge state

### Motivation and literature check

The legacy endpoint factorized element, formal-charge, and attached-H heads.
That permits combinations that never occur in the training data and asks three
independent categorical samples to agree with the bond sample. It also treated
aromatic bond as an independent fifth edge class even though sanitizable
aromaticity is a cycle-level property.

The implemented representation follows the common part of four joint 2D/3D
generators: topology is generated with atom chemistry rather than inferred from
coordinates after generation.

- MiDi jointly denoises molecular graph and 3D geometry:
  <https://arxiv.org/abs/2302.09048>
- MolDiff identifies atom-bond inconsistency and jointly samples atoms and
  bonds: <https://arxiv.org/abs/2305.07508>
- JODO jointly models atom types, formal charges, bonds, and coordinates:
  <https://arxiv.org/abs/2305.12347>
- FlowMol supplies the mixed continuous/categorical flow formulation used by
  this codebase: <https://arxiv.org/abs/2404.19739>
- FlowMol3 motivates later, separately testable self-conditioning and fake-atom
  stages; those changes are deliberately not bundled into Stage 2:
  <https://arxiv.org/abs/2508.12629>

### Changes

1. A node endpoint is one categorical state
   `s = (element, formal_charge, attached_H_count)` instead of independent
   `a/c/h` samples.
2. The union codec has 37 lossless states. Generation is restricted to the 36
   states observed in the training split; held-out-only `Si(0, H=1)` remains
   decodable but cannot be sampled.
3. Edge endpoints are exactly `NONE/SINGLE/DOUBLE/TRIPLE`. Input aromatic
   structures are deterministically Kekulized and RDKit sanitization restores
   aromatic flags after decoding.
4. A reproducible converter emits the tensors, split SMILES, marginals,
   state-vocabulary metadata, support mask, and conversion report from the
   official HP1KG XYZ source.
5. Dataset, CTMC vector field, CFG/MG/AG samplers, and RDKit decoder
   dynamically use either the legacy feature set or the new `x/s/e` feature
   set. Classifier guidance rejects incompatible legacy classifiers early.
6. The edge prior now receives `n_bond_types`; its default remains five for
   legacy checkpoints.
7. Train-support projection is enforced only during reverse sampling: endpoint
   probabilities, CFG transition rates, last-step sampling, and endpoint
   trajectories. Training receives raw logits.

### Defects found by train-sample consistency testing

The first inference smoke test found that `edge_prior()` still hard-coded five
real edge classes. The new model therefore received six CTMC channels although
its embedding expected five (four classes plus mask). Training did not expose
this because the conditional path constructs `e_t` from the target. Both data
and inference paths now pass the configured vocabulary size.

The next multi-step test found held-out state 22 in two of six sampled nodes.
The support projector had used `-inf`; at CFG weight one, the last-step
expression evaluated `0 * (-inf) + 1 * (-inf)`, producing NaN before argmax.
Unsupported logits now use a finite sentinel, and support is also applied
explicitly after CFG probability/rate composition. An adversarial test that
makes the unsupported class logit 1000 confirms zero unsupported samples in all
four probability/rate-matrix and linear/log branches.

The initial support projector also masked logits during validation. A
held-out-only lossless state would therefore contribute an artificial CE near
10,000 and could corrupt checkpoint selection. Training/validation now use raw
logits, held-out-only targets are ignored by CE, and support projection is
restricted to reverse sampling. A real test molecule containing state 22
produced finite `x/s/e` losses (`0.507/1.591/0.958`) under the production
configuration.

### Invariants and conflict checks

- Joint mode is CTMC-only and requires fresh training because node/edge head
  shapes change from `a/c/h + 5 edges` to `37 states + 4 edges`.
- Legacy mode retains its exact canonical order `[x, a, c, h, e]`, input
  concatenation order `[a, time, c, h]`, default edge vocabulary, and state-dict
  keys.
- The codec vocabulary is the all-split union only to keep held-out decoding
  lossless; the train-support mask prevents test leakage into generation.
- Dataset/model `n_atom_states`, `n_bond_types`, joint-mode, and separate-H
  settings are cross-validated before either ordinary or guided model
  construction.
- Attached hydrogen count is part of atom chemistry, not an explicit hydrogen
  coordinate. Explicit H placement and its geometry evaluation remain a later
  stage.
- Radicals are rejected by this HP1KG converter. Stereochemistry remains out
  of scope because the source representation does not carry it.
- Valence, connectivity, and global aromatic-cycle objectives are not claimed
  in this stage; they are Stage 3 work.
- Model guidance uses the same dynamic joint-state contract. Auto-guidance
  requires identical feature schemas/codecs for its good and bad checkpoints
  and carries the train-support mask through categorical transitions.
  CFG and AG CLIs expose `--guide_w_s`; when omitted it retains the legacy atom
  guidance scale.
- Classifier guidance fails early in joint mode until its external property
  classifier is retrained on `s`; feeding `s` to a legacy `a/c` classifier
  would not be a valid comparison.

### Verification (2026-10-18)

- Raw conversion and independent dataset decode both achieved RDKit identity
  `1581/1581`: train `1264/1264`, validation `158/158`, test `159/159`.
- Aromatic molecule counts were train/validation/test `1086/135/141`.
- Production-size GPU forward/backward on the smallest training graph used six
  atoms and 30 directed candidate edges. Losses were finite (`x=0.09590`,
  `s=2.43118`, `e=0.58435`), gradient norm was `7.83294`, and every gradient
  was finite.
- Production dimensions were node input 39, node output 37, edge input 5, and
  edge output 4.
- The production-size joint MG path completed finite forward/backward on the
  six-atom graph with node/edge output widths 37/4; its sampling-mode unit test
  confirms state-22 suppression.
- A three-step joint Auto-Guidance reverse pass ended with no residual mask or
  held-out state and passed the 37-state codec into the RDKit decoder.
- A six-step CFG reverse smoke test ended with no residual masks and no
  unsupported state; the decoded RDKit molecule was constructible.
- The committed YAML resolved through the production guidance loader as
  `ClassifierFreeGuidance[x,s,e]` with split sizes `1264/158/159`, explicit
  `training_split=train`, and a passing schema cross-check.
- A clean reconversion reproduced all train/validation/test tensors, marginals,
  metadata, and report fields exactly (except the intentional output path), and
  independently repeated oracle identity `1581/1581`.
- 25 repository unittests (including eight new Stage 2 tests), plus three
  legacy H-count functions passed.
- The epoch-1888 legacy CFG checkpoint loaded strictly with canonical features
  `[x, a, c, h, e]`, node input 26, node output 22, and edge input 6.
- The epoch-1982 legacy MG checkpoint also loaded strictly with canonical
  features `[x, a, c, e]`, node input 21, node output 18, and edge input 6.

### Interpretation limit

Oracle identity proves representation and deterministic decoding, not learned
generation quality. The reverse smoke model was untrained and only validates
shape, support, mask termination, and code-path consistency. Long training is
deferred until the subsequent chemistry losses and self-correction stages have
their own independent regression checks.
