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

## Stage 3: differentiable joint-chemistry consistency

### Motivation and design boundary

Stage 2 removes impossible independent atom-state combinations but its endpoint
cross-entropies are local marginals. They do not directly penalize a prediction
in which individually plausible atoms and bonds form an over-valent,
disconnected, or non-Kekulizable graph. MiDi, MolDiff, JODO, and DiGress all
motivate learning atom and edge chemistry jointly rather than repairing an
independently sampled adjacency only after reverse diffusion.

- MiDi: <https://arxiv.org/abs/2302.09048>
- MolDiff: <https://arxiv.org/abs/2305.07508>
- JODO: <https://arxiv.org/abs/2305.12347>
- DiGress: <https://arxiv.org/abs/2209.14734>

This stage adds differentiable endpoint objectives only. It deliberately does
not add a hard intermediate projector, a second edge head, or a new sampler.
Consequently the CTMC forward kernel and reverse state space remain exactly the
Stage-2 x/s/e process.

### Effective CTMC endpoint

For an entry that is still masked at time t, the chemistry objective uses the
model endpoint probability. For an already revealed entry it uses the observed
one-hot state. Thus

p_eff = 1[masked] p_theta(x_1 | x_t,c) + 1[known] one_hot(x_t).

This is essential: applying every global term to predictions for already known
entries would create a duplicate copy objective and would disagree with the
masked-only CTMC cross-entropy.

### Added objectives

1. **Balanced edge existence.** From the same four-way edge softmax,
   p(edge)=1-p(NONE). Positive and negative masked pairs are averaged
   separately and then equally combined. No inconsistent second edge head is
   introduced.
2. **Joint-state-aware heavy valence.** The expected bond-order degree is
   v_i=sum_j sum_b order(b) p_eff(e_ij=b). For each atom state, loss is the
   squared distance to its closest heavy valence observed in the training
   split, followed by expectation over p_eff(s_i). Using an allowed set
   instead of a frequency-weighted mean avoids penalizing rare but valid P/S
   valences.
3. **Molecular charge.** The predicted total formal charge is matched to the
   target total charge with squared error divided by atom count. It does not
   force every atom neutral, so charged and zwitterionic extensions remain
   representable.
4. **Connectivity backbone.** For every true edge the converter stores its
   uniform-spanning-tree inclusion probability, equal to effective resistance
   in the unweighted molecular graph. The loss is a weighted
   -log p_eff(edge) backbone objective. Bridges receive weight one while
   symmetry-equivalent cycle edges share weight. Foster's identity
   sum_e R_eff(e)=n-1 is checked for every molecule.
5. **Aromatic/Kekule consistency.** Aromatic perimeter edges maximize
   p(SINGLE)+p(DOUBLE). A second term matches each aromatic atom's expected
   incident-double-bond degree to the target degree. This permits alternative
   Kekule matchings that preserve local degree instead of forcing one arbitrary
   alternating edge assignment.

All auxiliary reductions use mean(w_t * loss) when time-scaled loss is
enabled. An earlier draft divided by sum(w_t), which would cancel a common
timestep factor; a regression test now prevents that error.

### Data and configuration changes

- Metadata version 2 stores training-only heavy-valence counts for all 37
  lossless joint states. The held-out-only state retains an empty count map and
  is excluded from generative support.
- Processed molecular bonds now carry aligned bond_connectivity_weights and
  bond_aromatic_flags.
- Dataset packing maps both fields onto exactly the same complete upper/lower
  edge order as e_1_true.
- Enabling chemistry loss requires require_chemistry_auxiliary_labels: true.
  Schema validation rejects legacy data/config combinations before training.
- The isolated configuration is
  configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage3_chemistry.yaml.
  Its initial weights are edge existence 0.5, valence 0.05, charge 0.05,
  connectivity 0.25, and aromatic 0.2. The lower valence weight follows a
  production smoke measurement rather than assuming equal raw scales.
- The converted tensors are stored separately in
  molguidance_hp1kg_baseline/data_est_joint_kekule_chemistry_official_xyz;
  Stage-2 data and background outputs are untouched.

### Invariants and conflict checks

- The network architecture, endpoint vocabulary, CTMC corruption, and reverse
  sampler are unchanged. Stage-2 and Stage-3 models have identical 634
  state-dict tensor names and shapes; the auxiliary module has no persistent
  parameters or buffers.
- The training objective changes, so a fresh run is still preferred for a clean
  stage ablation even though a Stage-2 checkpoint can be structurally loaded.
- Base FlowMol, CFG, and MG call the same chemistry function and the same total
  loss reducer. An enabled loss missing from the returned dictionary raises
  instead of being silently omitted.
- Deterministic validation averages the full returned loss-key set, not only
  canonical x/s/e, and condition-shuffle gain includes enabled chemistry
  terms.
- Existing Stage-1/2 and legacy modes default to all auxiliary weights zero.
  They do not require the new tensors or metadata and preserve strict
  checkpoint loading.
- Connectivity and aromatic labels are training objectives only. Reverse
  sampling does not require target-only fields, so no target leakage enters
  generation.
- The objectives remain soft surrogates. They improve gradients but do not
  mathematically guarantee a connected, valence-valid sample; that distinction
  is retained for later sampler/projection evaluation.

### Verification (2026-10-18)

- Full conversion retained RDKit identity 1581/1581:
  train 1264, validation 158, test 159.
- Every molecule was connected. Per-molecule Foster error was at most
  7.63e-6; UST weights stayed in [0.55652, 1.0].
- Stored aromatic-edge counts were train/validation/test
  17333/2170/2453, and both auxiliary edge tensors matched bond tensors
  exactly.
- Independent post-conversion audit rechecked metadata support, all tensor
  lengths, and sum_e weight_e=n-1 molecule by molecule.
- A production-size six-atom CFG forward/backward at t=0.5 was finite.
  Raw losses were x=0.18707, s=1.87260, e=0.94220,
  edge existence 1.03049, valence 13.89089, charge 0.00252,
  connectivity 0.09392, and aromatic 0 for this non-aromatic sample.
  Gradient norm was finite at 179.70. With committed weights the auxiliary
  contribution is about 1.23 versus about 4.32 from weighted x/s/e on
  this smoke example.
- A reduced real MG model returned all eight expected loss keys, all finite,
  and completed backward with non-zero node-head gradient.
- Benzene oracle logits gave all five auxiliary losses below 1e-6.
  Removing all bonds increased existence, valence, connectivity, and aromatic
  losses. Predicting an all-single aromatic perimeter also increased the
  aromatic loss, confirming the local Kekule-degree term is active.
- Random logits produced finite, non-zero node and edge gradients. Fully
  revealed CTMC entries produced zero auxiliary loss independently of endpoint
  logits.
- Cycle and tree tests recovered UST weights 5/6 and 1, respectively.
- A held-out-state logit of 1000 still produced a normalized supported-state
  distribution with exactly zero held-out probability, covering the previous
  unsupported-class underflow/NaN failure mode.
- The final repository suite passed all 37 tests, and all changed Python files
  passed byte compilation and git diff --check.

### Interpretation limit

This stage validates objective wiring and mathematical invariants, not improved
generation. The loss weights are safe initial scales from an untrained smoke
batch, not an optimized Pareto point. Long training remains deferred until the
next isolated self-conditioning/recycling stage is implemented and checked for
train-sample consistency.


## Stage 4: residual endpoint self-conditioning

### Literature contract and distinction from hidden recycling

FlowMol3 defines self-conditioning as an optional previous endpoint estimate
fed back alongside the current noisy molecule. During training it uses a
detached first prediction on 50% of updates and computes loss only on a second
pass; during inference it always carries the endpoint from the preceding
integration step. Megalodon independently reports an outer residual
self-conditioning wrapper for joint coordinates and discrete molecular states.

- FlowMol3 paper: <https://arxiv.org/abs/2508.12629>
- FlowMol3 official implementation: <https://github.com/Dunni3/FlowMol>
- Megalodon official implementation: <https://github.com/NVIDIA-Digital-Bio/megalodon>

The existing `n_recycles` loop was audited before this change. It repeats the
same GVP/update layers over hidden features inside one network call, but never
feeds an endpoint prediction back into the network. It remains available as
weight-shared hidden depth and is not renamed or treated as
self-conditioning.

### Implemented state and residual

Let the first endpoint prediction be `z_hat_1`. The feedback state is

`z_sc = stop_gradient({x_hat_1, softmax(s_hat_1), softmax(e_hat_1)})`.

For legacy non-joint configurations the node probability block is the
concatenation of every enabled node category. For the Stage-4 joint codec it is
the 37-state atom/charge/H probability vector. Train-unsupported joint states
are masked before softmax, so self-conditioning cannot feed an excluded class
back into the network.

The residual node input concatenates the current node hidden representation,
the previous node endpoint probabilities, and an RBF expansion of
`||x_hat_1_previous - x_t||`. The residual edge input concatenates the current
directed-edge hidden representation, the previous undirected edge
probabilities, and the difference between previous-endpoint and current
pair-distance RBFs. Learned MLP outputs are added to the node and edge hidden
states immediately before the GVP update stack. Coordinates are not directly
overwritten.

Only distances and same-frame displacement norms enter scalar residuals.
Therefore applying one rigid rotation and translation to both current and
previous coordinates leaves the residual scalars unchanged and preserves the
E(3)-equivariance of the downstream GVP coordinate update.

### Train, validation, and reverse-sampling behavior

- `self_conditioning_rate` is a graph-batch-level Bernoulli choice. Stage 4
  sets it to 0.5.
- A selected training update corrupts the graph once, computes one endpoint
  under `no_grad`, canonicalizes/detaches it, and performs the gradient-bearing
  second pass on exactly the same noisy graph, timestep, condition embedding,
  CFG condition-dropout mask, and guidance scale.
- The unselected half performs one ordinary endpoint pass.
- Deterministic validation always uses the two-pass path so checkpoint metrics
  measure the inference-like architecture without a random mixture.
- Ordinary CTMC and scale-aware MG samplers carry the endpoint from one reverse
  step to the next. The first reverse step has no previous endpoint.
- CFG maintains separate unconditional and conditional endpoint histories.
  Feeding a guided mixture or one branch into the other would change the
  model's conditioning semantics and is explicitly avoided.
- Auto-guidance likewise maintains independent good-model and inferior-model
  endpoint histories. Classifier-guidance carries its base-model endpoint
  before applying external property gradients. The legacy hybrid sampler now
  maintains all four good/bad/CFG-unconditional/CFG-conditional histories
  independently and combines schedule-correct coordinate velocities rather
  than treating endpoint coordinates as velocities.
- Endpoint feedback is prepared before categorical temperature or stochastic
  state sampling. It therefore represents the network's endpoint belief, not
  one random reverse transition.

The schema guard requires `vector_field.self_conditioning: true` and a
positive `mol_fm.self_conditioning_rate` together, checks the rate is in
`[0,1]`, and currently restricts the feature to CTMC. All older configurations
default to both disabled.

### Isolation and checkpoint contract

The isolated configuration is
`configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage4_self_conditioning.yaml`.
It reuses the read-only Stage-3 data tensors and chemistry losses but writes to
a distinct Stage-4 run directory.

With self-conditioning disabled, no new module parameters exist and the
Stage-3 state-dict contract is unchanged. Enabling it adds exactly eight MLP
parameter tensors (186,880 parameters for the production dimensions). A direct
Stage-3/Stage-4 comparison found 634 common tensor names with identical shapes,
zero removed tensors, and only those eight new tensors. A fresh run is required
for the controlled Stage-4 ablation; silently initializing those tensors while
claiming exact checkpoint continuation would be misleading.

### Verification (2026-10-18)

- Eight focused tests passed: support projection and normalization, detached
  feedback, NaN/zero-mass probability fallback, disabled-architecture
  compatibility, E(3) invariance, no-gradient first pass with identical inputs,
  CFG branch separation, hybrid CFG branch separation, and schema guards.
- The complete repository suite passed all 45 tests after the integration.
- A rigid rotation plus translation changed residual node/edge outputs by less
  than 2e-6.
- The production Stage-4 YAML instantiated `ClassifierFreeGuidance` with a
  `CFGVectorField`, joint `[x,s,e]` state, rate 0.5, 7,294,617 total
  parameters, and 186,880 self-conditioner parameters.
- On a real six-atom HP1KG training graph at t=0.5, the full two-pass CFG
  forward and backward were finite. Total weighted loss was 4.03564 and all
  eight self-conditioner parameter tensors received finite gradient; their
  stacked norm was 1.66711.
- An untrained production model completed a three-timepoint CFG reverse smoke
  for a six-atom graph and terminated with the expected joint-state and edge
  tensor widths.

### Interpretation limit

This stage establishes train-sample consistency for self-conditioning; it does
not establish better chemistry or geometry. It adds one no-gradient network
pass on half of training updates. Inference still uses exactly one network
evaluation per reverse step; the first step simply has no endpoint history.
Generation-quality claims require a fresh Stage-4
training run and matched checkpoint evaluation. Fake atoms/count correction,
independent graph/geometry times, and bond-first geometry updates remain
separate later stages.


## Stage 5: fake-atom count correction

### Reference contract and local adaptation

FlowMol3 treats molecular size as part of the generated endpoint by appending
synthetic fake atoms during training and deleting nodes classified as fake
after reverse integration. Its reported augmentation samples a fake count,
anchors each fake node to a uniformly selected real atom, and adds isotropic
Gaussian coordinate noise with standard deviation one.

- FlowMol3 paper: <https://arxiv.org/abs/2508.12629>
- Official implementation: <https://github.com/Dunni3/FlowMol>
- Audited official commit: `77cae22174b7792b0e25e9e0414038420736d841`

For a molecule with N real heavy atoms, Stage 5 samples

`K ~ DiscreteUniform({0, ..., ceil(p_fake N)})`, with `p_fake = 0.3`,

then samples anchors `a_k ~ Uniform({1,...,N})` and coordinates

`x_fake,k = x_a_k + Normal(0, sigma_fake^2 I)`, with `sigma_fake = 1`.

The inclusive upper endpoint follows the paper-level support and remains
defined when the maximum is zero. Training uses worker-seeded randomness;
validation and test use a deterministic seed derived from split and molecule
index. All real and fake coordinates are centered together after augmentation.

HP1KG uses a lossless joint `(element, formal charge, explicit H count)` state,
so a separate element-only fake class would violate the codec. Stage 5 instead
appends one 38th joint state:

`(__FAKE__, charge=0, H=0, allowed heavy valence={0})`.

The static 37-state metadata and processed tensors are not rewritten. The
model appends the synthetic state in memory, marks it train-supported, and
expands the otherwise-unused CTMC state marginal for schema consistency.

### Edge, chemistry, and decoding semantics

- The complete graph is built after fake nodes are appended. Every target edge
  incident to a fake node is therefore `NONE`; connectivity and aromatic target
  fields are zero on the same pairs.
- The ordinary joint-state CE learns real versus fake identity. The existing
  heavy-valence objective assigns fake state valence zero, coupling fake-node
  probability to absence of incident bond order without adding a second,
  potentially inconsistent edge head.
- Molecular charge is still summed over all endpoint probabilities, but its
  squared error is normalized by the target real-atom count, not by the padded
  slot count. Dataset `is_real_1_true` is checked against the fake class rather
  than trusted as an independent label.
- Fake nodes remain in the graph throughout reverse CTMC/coordinate
  integration. Immediately before RDKit construction, the endpoint-induced
  real-node subgraph is taken. DGL relabels retained nodes and removes every
  fake-incident edge. The full slot graph remains available as `slot_graph`,
  while `g`, `num_atoms`, chemistry metrics, and RDKit see only decoded real
  atoms. An all-fake endpoint is explicitly invalid.
- Trajectory decoding repeats fake removal per frame. Rigid alignment is used
  only when a frame and final endpoint have the same real-node count, avoiding
  a shape error when node identity changes along the trajectory.

### Sampling-size contract

`sample(n_atoms)` continues to interpret `n_atoms` as exact graph slot counts.
`sample_random_sizes`, used by production CFG/MG sampling, first draws the
real-count proposal from the existing marginal or condition-aware kernel and
then adds fake capacity using the same integer-uniform law as training. This
preserves the joint training distribution over `(real count, slot count)`;
the model is not given the latent real-count proposal as an extra condition.

Three explicit capacity modes are available:

- `stochastic` (default): train-distribution-matched integer-uniform capacity;
- `maximum`: always add `ceil(p_fake N)` slots for diagnostic search capacity;
- `none`: treat the proposal as the exact slot count.

Base FlowMol, CFG, scale-aware MG, and auto-guidance random-size paths share
this helper. Fixed count requests initialize the same count diagnostics as
sampled counts. A separate API audit also made `properties_handle_method=None`
fall back to the model's configured method in every sampler instead of reaching
the vector field as an invalid mode.

The complete graph scales quadratically. A 30% node increase can produce up to
1.69 times as many pair states, so the isolated Stage-5 configuration starts at
batch size 640 rather than Stage 4's 1024. It writes to a distinct output path:

`configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage5_fake_atoms.yaml`.

### Checkpoint and stage isolation

Stage 5 changes the categorical state width from 37 to 38. Consequently the
node input embedding, node output head, and self-conditioning node-probability
layers change shape. Stage-4 checkpoints are not strictly compatible and must
not be presented as resumed Stage-5 training. The official XYZ coordinates,
Stage-3 processed database, all earlier checkpoints, and background outputs are
left untouched; Stage 5 requires a fresh run.

### Verification (2026-10-18)

- Ten focused fake-atom tests cover deterministic augmentation, inclusive
  capacity bounds, fake-edge targets, metadata/marginal expansion, chemistry
  oracle behavior, decoder relabelling, changing trajectory counts, all-fake
  rejection, and schema guards.
- The complete repository suite passes all 55 tests.
- Full HP1KG oracle decoding passed RDKit identity 1581/1581: train 1264,
  validation 158, and test 159. Fake nodes were actually present in 1398 of
  those deterministic/stochastic audit samples; no failure was hidden by an
  all-zero augmentation draw.
- Stage 4 and Stage 5 have the same 642 state-dict keys and no added/removed
  tensors. Exactly five tensors change shape: the conditional/unconditional
  node input projections, node output weight/bias, and self-conditioning node
  residual input. Parameter count rises only from 7,294,617 to 7,295,642
  (+1,025); every coordinate, edge, and GVP tensor shape is unchanged.
- On a real validation molecule augmented from 14 real atoms to 18 slots, the
  decoder returned exactly 14 RDKit atoms. A production Stage-5 CFG
  forward/backward with self-conditioning returned finite feature and chemistry
  losses, finite total loss 10.61636, and finite gradients for every parameter
  that received a gradient.
- An untrained production Stage-5 CFG model completed a three-timepoint reverse
  smoke through `sample_random_sizes`, emitted the expected 39-wide CTMC node
  tensor (38 endpoint states plus mask) and five-wide edge tensor (four Kekule
  states plus mask), and constructed a `SampledMolecule` without decoder error.
- Changed Python files pass byte compilation and `git diff --check`.

### Interpretation limit

This stage proves a train/sample-consistent mechanism for deleting surplus
slots and verifies that it is lossless on oracle endpoints. It does not prove
that a freshly trained model will calibrate the number of fake predictions or
improve chemical validity. Generated real-count histograms, all-fake rate,
fake-bond probability, validity, condition fidelity, and coordinate quality
must be measured from a new Stage-5 checkpoint before advancing to independent
graph/geometry time or bond-first updates.
