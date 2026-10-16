# HP1KG EST=0 generation audit (2026-10-07)

This document preserves the evidence and decisions that motivated the
`fix/molguidance-categorical-guidance-mask-safety` branch. It is intentionally
checkpoint-independent: existing checkpoints remain loadable, while sampling
semantics and invalid-state handling are made explicit.

## Outcome

The best current method is AG, but none of the four guidance methods reliably
generates chemically clean, HP1KG-like molecules at EST approximately zero.
Low predicted gaps are often obtained through ringless or abnormal-ring
structures instead of the aromatic, mostly five- and six-membered ring systems
present in the corresponding training subset.

| method | raw valid | connected | novel unique | XTB/STDA OK | clean proxy | strict target-like |
|---|---:|---:|---:|---:|---:|---:|
| CFG | 527 | 514 | 502 | 437 | 362 | 65/1000 |
| MG  | 705 | 339 | 336 | 259 | 247 | 29/1000 |
| CG  | 351 | 277 | 273 | 237 | 142 | 25/1000 |
| AG  | 549 | 532 | 521 | 466 | 418 | 82/1000 |

Strict target-like means all of: `abs(EST) <= 0.1 eV`, aromatic, only five- or
six-membered rings, and passing the chemical-clean proxy.

The generated survivors are smaller than the target subset. The training DB
has a mean of 26.38 heavy atoms; the actual `abs(EST) <= 0.1` subset has 28.34,
whereas surviving generated structures average 19.51--22.87. Independent
complete-graph edge errors therefore accumulate with molecular size.

## Structural evidence

The training DB contains 88.37% molecules whose rings are exclusively five or
six membered, 2.37% containing a three/four-membered ring, and 4.11% ringless
molecules. Generated novel connected structures instead show:

| method | only 5/6 | 3/4 ring | 7/8 ring | >=9 ring | ringless |
|---|---:|---:|---:|---:|---:|
| CFG | 68.73% | 13.15% | 10.36% | 3.78% | 5.58% |
| MG  | 35.42% | 0.89% | 5.36% | 3.57% | 55.95% |
| CG  | 41.76% | 24.91% | 24.18% | 22.34% | 6.59% |
| AG  | 71.59% | 8.25% | 13.44% | 2.50% | 6.14% |

MG's apparently high EST hit rate is largely a topology shortcut: successful
ringless molecules have median EST 0.033 eV, while its aromatic products have
median EST 0.328 eV. The real DB target subset is 84.35% aromatic and averages
3.75 rings, so ringless near-zero products do not reproduce the target domain.

## Geometry evidence

XTB geometry success is not evidence that the diffusion coordinates are good.
For structures that reached XTB, the median heavy-atom RMSD after optimization
is 0.739 A (AG), 0.796 A (CFG), 0.919 A (CG), and 1.104 A (MG), versus 0.258 A
for the DB control. Median stabilization energies are -67.8, -102.7, -190.5,
and -375.1 kcal/mol respectively, versus -12.7 kcal/mol for the DB control.
XTB is repairing a substantial part of the generated geometry.

## Direct implementation findings

1. The CFG/AG `campbell_rate_matrix` probability branch combines endpoint
   probabilities and then calls `softmax(probability / temperature)`. The base
   CTMC path applies temperature in log-probability/logit space. At the default
   categorical temperature of 0.05 these are materially different operations.
2. Residual CTMC masks are silently decoded as chemistry: an atom mask becomes
   Se, a charge mask becomes +4, and an edge mask becomes no bond. A residual
   latent state must instead be rejected and reported.
3. Atom, charge and bond heads are independent. There is no joint valence,
   connectivity or whole-cycle Kekule constraint in final decoding.
4. The edge distribution is 92.57% NONE, so one five-way softmax is severely
   imbalanced. Invalid generations are dominated by Kekulization failures.
5. Implicit hydrogen count is not modeled, causing unsupported P/S/Se/Si local
   environments to survive the neural prediction and fail later chemistry.
6. Validation selects checkpoints by denoising loss, not chemical cleanliness,
   target-like topology, condition fidelity or generated-coordinate quality.

## Ordered remediation

### Phase 0: no retraining

- Combine CFG/AG categorical distributions in log-probability space and apply
  temperature once: `softmax(((1-w) log p_u + w log p_c) / T)`.
- Keep an explicit legacy mode only for reproducibility.
- Expose categorical temperature at the CLI and sweep `1, 0.5, 0.2, 0.05`.
- Reject residual atom/charge/edge masks; record per-modality counts and a
  machine-readable sampling diagnostic sidecar.
- Sweep MG only inside its trained scale range first (`0, 0.5, 1.0`).
- Rank samplers with strict target-like and geometry metrics, not raw validity.

### Phase 1: retraining required

- Add implicit-H/valence state to the heavy-atom representation.
- Train aromatic systems as Kekule single/double assignments plus a separate
  whole-cycle aromatic target.
- Factor edge existence from conditional bond order and normalize each loss on
  its active items per graph.
- Add degree-aware message normalization, actual GVP dropout, self-conditioning,
  size-conditioned coordinate priors, and soft geometry losses.
- Pretrain on a substantially larger compatible chemical corpus before HP1KG
  fine-tuning; the present training split contains only 1,264 molecules.

### Phase 2: global chemical consistency

- Add graph-level degree/component/cycle/ring-size features to the denoiser.
- Use a final joint atom-state/bond solver enforcing valence, connectivity,
  charge and whole-cycle Kekule feasibility.
- Select checkpoints using chemical-clean validity, target topology, local DB
  support, condition fidelity, and coordinate-repair metrics.

## Evidence locations

- Generation comparison:
  `runs/guidance_est0_n1000_ema_edge_best_diffcoords_20261007/comparison/method_summary.csv`
- Corrected diffusion-coordinate XTB/STDA outputs:
  `runs/guidance_est0_n1000_ema_edge_best_diffcoords_20261007/`
- HP1KG EST distribution analysis:
  `analysis/hp1kg_est_distribution_20261001/`
- Training data:
  `molguidance_hp1kg_baseline/data_est/`

## Decision gate

Do not continue long training on the current sampler before Phase 0 A/B tests.
If corrected probability guidance and residual-mask rejection materially improve
strict target-like yield, retain the architecture for Phase 1. Otherwise port
the stabilization mechanisms (self-conditioning, dynamic/fake atoms and
train-time geometry distortion) before another long training run. The earlier
tree-plus-cycle-closure V9 experiment did not improve the result, so another
wholesale architecture replacement is not the first action.
