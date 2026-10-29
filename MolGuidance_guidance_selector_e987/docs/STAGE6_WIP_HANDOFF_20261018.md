# Stage 6 WIP handoff — 2026-10-18

## Pause state

- Branch: `feature/molguidance-joint-chemistry-self-correction`
- State: temporarily paused at the user's request.
- Stage 6 changes are intentionally left as uncommitted WIP.
- No long-running Stage 6 training or generation job was started.
- Existing files, checkpoints, datasets, and prior background results were not removed or overwritten.

## Completed implementation

### Independent modality time

- Added separate coordinate and graph times (`t_x`, `t_graph`).
- Cartesian coordinates are corrupted and weighted with `t_x`.
- Atom, charge, hydrogen, radical, and bond states are corrupted and weighted with `t_graph`.
- Training can independently sample the two times and, with a configurable probability, keep the graph clean (`t_graph = 1`) while coordinates remain noisy.
- Explicit single-time calls retain the diagonal path (`t_x = t_graph`) so the existing reverse sampler remains usable.
- Deterministic validation supports paired coordinate/graph time grids.

### Bond-first coordinate update

- Added a symmetric bond-conditioned coordinate update using:
  - node-state sum and absolute difference,
  - averaged directed edge hidden states,
  - distance RBF features,
  - an invariant scalar gate multiplied by the relative unit vector.
- Pair updates are antisymmetric and degree-normalized, preserving center of mass and O(3) equivariance.
- The final layer is zero-initialized so a fresh Stage 6 model starts from the legacy coordinate behavior and learns the new correction gradually.
- Fixed the Stage 6 update schedule so `convs_per_update = 1` performs an update at convolution 0; the legacy route retains its previous behavior.

### Integration and validation guards

- Routed graph time through CTMC corruption, CFG endpoint/self-conditioning passes, feature loss weighting, chemistry losses, and Model Guidance calls.
- Added schema checks for modality-time configuration and model/config flag agreement.
- Added the Stage 6 config:
  `configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage6_independent_time_bond_first.yaml`.
- Added Stage 6 tests in `tests/test_independent_time_bond_first.py`.

## Validation completed before the pause

- Python compilation passed for the initially modified Stage 6 files.
- `git diff --check` passed before the final two small follow-up edits.
- Full test discovery passed: **65/65** (55 existing + 10 Stage 6 tests).
- A real HP1KG one-graph CUDA forward/backward smoke test was finite:
  - nodes: 13
  - directed edges: 156
  - weighted total loss: 3.8016028404
  - gradient norm: 24.1517047882
  - bond-first module aggregate gradient L1: 10.3952645324
  - peak allocated CUDA memory: 77.792 MiB for this small smoke batch
- An untrained full production CFG one-step reverse smoke produced a 5-atom RDKit-valid graph.
- Bond-update tests passed for rotation equivariance, translation invariance, center-of-mass conservation, edge dependence, and gradient flow.
- Stage 5 versus Stage 6 state dictionaries:
  - Stage 5 keys: 642
  - Stage 6 keys: 674
  - common keys: 642
  - two changed input-layer shapes (`40 -> 41` inputs)
  - 32 new bond-update tensors
  - no removed tensor keys
  - strict Stage 5 checkpoint loading is therefore not compatible; fresh Stage 6 training is required.

## Reference consistency checked

- Megalodon paper: separate continuous/discrete times and mixed training with independently noised modalities versus clean graph/noisy structure.
- Official Megalodon/JODO implementation: graph hidden states are updated before equivariant coordinate updates, and coordinate displacement is an invariant scalar multiplied by a relative direction.
- EGNN and GVP principles were checked for translation/rotation behavior of the new coordinate route.

Reference links:

- https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00380f
- https://arxiv.org/abs/2505.18392
- https://arxiv.org/abs/2102.09844
- https://arxiv.org/abs/2009.01411

## Important limitations and pending checks

1. Reverse generation still follows the diagonal schedule `t_x = t_graph`; an explicit graph-first then geometry 2-D reverse schedule has not been implemented.
2. Better generation quality has not been demonstrated. It requires fresh Stage 6 training and matched checkpoint evaluation.
3. Production batch-size/GPU-memory tuning has not been run; batch 512 in the config is only a conservative starting value.
4. After the 65-test pass, two small edits were made:
   - the Model Guidance training forward path received explicit graph-time propagation;
   - duplicate feature precomputation in the vector field was removed.
   These final edits still require compilation, the complete test suite, and a real Model Guidance smoke test when work resumes.
5. A planned additional reflection/zero-initialization optimizer-step test was not added because work was paused. The existing rotation/translation/COM tests did pass.
6. No Stage 6 long training should be launched until item 4 is completed successfully.

## Recommended resume sequence

1. Inspect the current diff and run `git diff --check`.
2. Compile all touched Python files.
3. Run the full unit-test discovery suite again.
4. Run CFG and Model Guidance real-data forward/backward smoke tests.
5. Confirm clean-graph/noisy-coordinate and independent-time corruption statistics.
6. Commit Stage 6 only after those checks pass.
7. Start fresh training; do not resume from a Stage 5 checkpoint.

## Resume update — 2026-10-19

The pending verification sequence was completed.

- Final modified files compile.
- `git diff --check` passes.
- The complete suite passes **68/68** after adding reflection,
  zero-initialization/optimizer activation, and MG validation-dispatch tests.
- Real HP1KG CFG and scale-aware MG forward/backward smoke tests both produced
  finite losses, finite gradients, and nonzero bond-first gradients.
- A Model Guidance validation audit found and fixed one additional issue: MG
  had retained random validation time/dropout instead of the Stage-6 fixed
  paired grid. It now uses the common deterministic validation path with all
  training-only branches disabled.
- The real-data MG validation grid is repeatable within CUDA/DGL numerical
  tolerance: maximum repeat error `3.8147e-6` with tolerance `1e-5`.

The former pending items 4 and 5 are therefore closed. Long training has not
yet been started; the remaining next action is to commit the selected Stage-6
files and launch a fresh run rather than loading a Stage-5 checkpoint.

The initial Stage-6 commit is `af62e47`. Subsequent B200 optimizer-step probes
measured 51.0% reserved memory at batch 512 and 73.8% at batch 704. The latter
completed with finite loss and gradients, but real Lightning training later
reached 93.0% at batch 704. That run was stopped after epoch 6 and preserved;
batch 512 was restored as the production setting for the allocator audit.

## Production launch update — 2026-10-19

The apparent epoch-to-epoch memory growth was CUDA allocator fragmentation from
the varying complete-graph sizes, not retained model graphs. Without expandable
segments, the batch-512 Lightning run rose to 181774 MiB by epoch 11. With
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, batch 512 stabilized near
100020 MiB through the same interval. Both diagnostic runs were stopped and
their outputs were preserved.

The final production candidate uses batch 640 and the expandable-segment
allocator. At epoch 10 it occupied 125022 MiB of the 182631.6 MiB B200
(68.5%). Train loss fell from 14.9 to 3.49 and deterministic validation loss
from 13.5 to 3.14. The detached process was retained as the long run:

- PID at launch verification: `572069`
- run directory:
  `../runs/molguidance_hp1kg_joint_kekule_stage6_independent_time_bond_first/hp1kg-est-cfg-stage6-b640-expandable_a73c71od`
- launcher log:
  `../runs/molguidance_hp1kg_joint_kekule_stage6_independent_time_bond_first/launcher_b640_expandable_20261019.log`
- reproducible launcher: `scripts/train_stage6_b200.sh`

The launcher makes the allocator requirement explicit. Starting this config
without the launcher (or an equivalent environment setting) can again consume
nearly all device memory even at a smaller batch size.
