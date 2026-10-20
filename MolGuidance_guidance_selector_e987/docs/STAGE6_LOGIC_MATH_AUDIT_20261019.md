# Stage 6 logic and mathematics audit — 2026-10-19

## Audit snapshot and verdict

- Branch: `feature/molguidance-joint-chemistry-self-correction`
- Audited commit: `ded8ad1`
- Model: HP1KG EST CFG Stage 6 with joint atom chemistry, Kekule bonds,
  independent coordinate/graph times, self-conditioning, fake atoms, chemistry
  auxiliary losses, and bond-first coordinate updates.
- The detached training process was preserved. At the final audit check it was
  PID `572069`, near epoch 389, using 128292 MiB of GPU memory.
- No model, data, checkpoint, or training file was modified during the audit.

The implementation is executable and its current unit tests pass, but it is not
mathematically closed enough to call the model fully consistent. The codec,
tensor contracts, edge packing, rotation behavior, and local chemistry loss
implementation are sound. The remaining high-priority issues are the
categorical time-loss reduction, the difference between independent-time
training and diagonal reverse sampling, the mismatch between reflection-allowing
alignment and a reflection-sensitive full GVP, and optional property
normalization at inference.

## P0 categorical time-loss reduction

`molguidance/models/loss_utils.py` deliberately preserves the scalar objective
of the released broadcasting implementation. For a one-dimensional categorical
loss vector `L` and column time weights `w`, it computes

```text
mean(L) * mean(w)
```

rather than the sample-aligned objective

```text
mean(w_i * L_i).
```

For `L=[1,3]` and `w=[1,3]`, the current helper returns `4.0` with gradient
`[1.0,1.0]`. The aligned objective returns `5.0` with gradient `[0.5,1.5]`.
The current reduction therefore removes the correspondence between a sample's
timestep and its own loss.

Under the current `graph_clean_probability=0.5` distribution, a Monte Carlo
check assuming constant conditional CE on masked entries gave factorized scale
`0.30262` and aligned scale `0.15093`, a ratio of approximately `2.005`.
Clean-graph samples have zero categorical CE but their high weight still
multiplies other samples' loss.

The next implementation must explicitly choose either `mean(w_i*L_i)` over all
entries or a masked-entry-normalized estimator, document that choice, and test
values and gradients with heterogeneous weights. The current reduction is
useful only for exact released-baseline compatibility and is not consistent
with Stage 6 sample-wise time weighting.

## P0/P1 independent-time train–sample contract

Training samples coordinate and graph time independently and sets `t_graph=1`
with probability 0.5. The network receives both times and uses each schedule
column at the correct time.

Reverse generation nevertheless follows only `t_x=t_graph`. The clean-graph
branch can regularize coordinate denoising, but the sampler does not implement
the graph-first two-dimensional reverse path that motivated the branch. This is
not an invalid diagonal flow by itself, but the claimed causal generation
mechanism is not exercised during inference and the graph-clean mixture is used
only as auxiliary supervision.

The model should either implement and test an explicit graph-first 2-D reverse
schedule or describe and tune the clean-graph branch solely as denoising
regularization.

## P0 symmetry contract

`BondFirstCoordinateUpdate` alone is translation invariant, center-of-mass
preserving, and O(3)-equivariant. The full model is not O(3)-equivariant because
the production configuration uses `n_cp_feats=4`. GVP cross products are axial
vectors and are mixed with polar vectors.

Numerical results:

```text
single GVP, n_cp_feats=0: reflection vector error 0.0
single GVP, n_cp_feats=4: rotation vector error 1.68e-8
single GVP, n_cp_feats=4: reflection vector error 0.4402
two GVP layers, n_cp_feats=4: reflection scalar/vector error 0.0411/0.0706
```

The current Kabsch implementation has no determinant correction. A mirror test
gave `det(R)=-1.0000002` and mirror-fit RMS `2.22e-7`, confirming that alignment
can reflect the prior. Training can therefore present reflected alignments to a
model whose output depends on handedness.

One contract must be selected:

- For the current non-stereochemical representation, set `n_cp_feats=0` and
  retain O(3) equivariance.
- For a chiral model, restrict Kabsch to `det(R)=+1` and introduce explicit
  parity-aware features and stereochemical labels.

The existing reflection unit test covers the bond-first block, not the complete
multi-layer GVP.

## P0 inference property units

Training EST normalization is `mean=0.5285704`, `std=0.5375195`.
`sample_cfg.py` leaves `--normalization_file_path` optional. Passing raw EST=0
without it gives network input `0.0`; the correct normalized value is
`-0.98335`. Network input zero corresponds to raw EST near `0.52857`.

The inspected production generation scripts do pass the normalization file, so
their conditions were not necessarily mis-scaled. The public API remains
unsafe. A conditional checkpoint should resolve normalization metadata
automatically or reject calls without it.

The conditional atom-count kernel correctly uses raw EST values. The model
embedding and count kernel therefore require different representations of one
requested property, which the API should handle internally.

## P1 chemistry losses are marginal constraints

The edge-existence, valence, charge, connectivity, and aromatic terms are
finite, differentiable, and have passing oracle/gradient tests. They do not
guarantee that one jointly sampled discrete molecule obeys those constraints.

For benzene, setting every perimeter edge to
`P(single)=P(double)=0.5` produces zero perimeter loss and the correct expected
double degree at every carbon. Independent edge sampling yields one of the two
alternating Kekule patterns with probability only `2/64=3.125%`. Expected
valence and weighted connectivity have the same marginal-versus-discrete gap.

These losses are useful auxiliaries but cannot serve as hard validity
guarantees. A joint final decoder/solver, structured edge process, or
autoregressive closure stage remains necessary.

## P1 data loading and memory control

The YAML sets `training.max_num_edges=900000`, but `data_module_from_config`
does not pass it to `MoleculeDataModule`. The DataModule stores its default
`40000`, constructs ordinary shuffled DataLoaders, and never uses
`SameSizeMoleculeSampler`. The configuration field therefore has no effect.

This leaves complete-graph memory dependent on random batch composition. The
edge budget must be wired into a correct batch sampler or removed.

## P1 validity and checkpoint semantics

The low-level RDKit builder creates a conformer but does not sanitize the
molecule. `sample_cfg.py` writes every non-`None` molecule and labels the count
as valid.

The legacy `SampleAnalyzer.compute_validity` also:

1. sanitizes only the largest fragment, so disconnected graphs may count valid;
2. uses `except AtomKekulizeException or ValueError`, which does not catch
   `ValueError` as intended; and
3. can divide by zero on empty result sets.

`tools/analyze_generation.py` performs stricter sanitization and records
connectedness separately. Checkpoint selection still monitors only
`val_total_loss`; chemical-clean validity, connectedness, generated geometry,
and EST fidelity are not selection criteria. Condition shuffle gain is a
sensitivity diagnostic, not a physical fidelity metric.

## P1/P2 remaining limits

- GVP convolution dropout remains zero. `p_uncond=0.1` is condition dropout,
  not trunk regularization.
- Bond-first geometry processes every complete-graph pair and normalizes by
  complete-graph degree `n-1`; it is not explicitly gated by predicted bond
  probability.
- Optional `campbell_rate_matrix` CFG ignores `categorical_probability_mode`,
  clamps negative linear-guidance rates, and uses guided logits for the final
  residual masks. The default CLI path is the separate `campbell` route.
- Conversion reports store source/split SHA-256 values, but checkpoint loading
  does not enforce a data manifest hash.
- The deterministic validation grid differs from the stochastic training time
  distribution. It is useful for comparison but is not an unbiased estimator
  of the training objective.

## Components confirmed sound

- HP1KG codec oracle identity: `1581/1581`.
- Split identity: train `1264/1264`, validation `158/158`, test `159/159`.
- Joint atom state preserves element, formal charge, and attached-H count.
- Edge vocabulary is `NONE/SINGLE/DOUBLE/TRIPLE`; valid assignments restore
  aromaticity after sanitization.
- Optimized source XYZ coordinates and atom indexing are preserved.
- Train-support masking and fake-atom state handling are consistent.
- Fake-node capacity sampling matches training augmentation.
- Complete-graph edge packing and CFG self-conditioning branch separation are
  consistent.
- Condition dropout is per graph and disabled in deterministic validation.
- Bond-first gradients, rotation/translation behavior, and COM conservation
  pass.
- Full CPU unit-test discovery passed `68/68` with correct `PYTHONPATH`.
- `git diff --check` passed before documentation was added.

Passing tests establish stability for the tested contracts. They do not resolve
the counterexamples because the tests encode the factorized loss and test
reflection only on the bond-first submodule.

## Required correction order

1. Replace or quarantine the factorized categorical reduction and add
   heterogeneous-weight value/gradient tests.
2. Choose O(3) versus chiral SO(3) and make Kabsch, GVP, tests, and docs agree.
3. Implement a 2-D graph/coordinate reverse schedule or retune the clean-graph
   branch strictly as regularization.
4. Make condition normalization checkpoint-driven and mandatory.
5. Add joint discrete chemistry projection or a structured graph decoder.
6. Wire the edge-budget sampler into the DataModule.
7. Use strict whole-molecule validity, connectedness, geometry, and condition
   fidelity in checkpoint evaluation.
8. Verify data/converter manifest hashes on checkpoint load and generation.
9. After items 1–4, train a fresh matched checkpoint. The current run remains a
   useful diagnostic baseline but is not optimization-equivalent after fixing
   the loss.
