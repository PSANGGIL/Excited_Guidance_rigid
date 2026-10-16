# Heavy-atom attached-hydrogen state

This branch adds hydrogen information without adding hydrogen coordinate nodes.
Each heavy atom has a categorical state `h` with four endpoint classes: 0, 1,
2, or 3 attached hydrogens. CTMC corruption adds a fifth mask token only at
intermediate time. The active feature order is therefore `x, a, c, h, e`.

## Why this representation

Explicit hydrogen nodes would increase graph size and require the coordinate
model to generate every X-H vector. Leaving hydrogen entirely to RDKit lets
implicit-valence rules repair an inconsistent heavy-atom graph and hides the
model error. The attached-H state is the middle ground: it is jointly denoised
with atom type, charge, and bond state, while 3D hydrogen coordinates can still
be added after generation without moving the generated heavy atoms.

No aromatic auxiliary loss was added. Aromatic bonds retain the existing model
representation; the new state only makes per-atom valence/hydrogen occupancy an
explicit prediction target.

## Data and configuration

The reproducible converter is:

```bash
python molguidance_hp1kg_baseline/convert_hp1kg.py   --source joint_kekule_diffusion/data_hp1kg_nevpt2   --output molguidance_hp1kg_baseline/data_est_hcount   --property est
```

`est` is defined as `S1 - T1`, matching the previous EST dataset. The generated
training split contains 33,346 heavy atoms with the following H-counts:

- H0: 16,338
- H1: 11,921
- H2: 2,475
- H3: 2,612

The new CFG training config is
`configs/local/guidance_est/hp1kg_est_cfg_hcount.yaml`. It writes to a new run
directory and does not overwrite the previous dataset or checkpoints.

## Decoder behavior

When `h` is present, the RDKit decoder calls `SetNumExplicitHs` and
`SetNoImplicit(True)` on each heavy atom. A residual CTMC H mask rejects the
molecule instead of being interpreted as a chemical H count. When `h` is absent,
the old implicit-H decoder path is unchanged.

## Compatibility and validation

- Existing checkpoints remain loadable because `n_hydrogen_counts` defaults to
  zero. Epoch 987 was loaded strictly with feature order `x, a, c, e` and the
  original input/output dimensions.
- H-count checkpoints have enlarged input and node-output layers and therefore
  require fresh training; they are intentionally not shape-compatible with old
  weights.
- A real two-molecule CFG batch produced finite losses and nonzero gradient in
  the H output rows.
- A reverse-sampling smoke test generated an `h_1` tensor with five CTMC states
  and passed decoded H counts to RDKit.
- Full HP1KG oracle decoding sanitizes 1,581/1,581 and gives canonical identity
  for 1,580/1,581. The single mismatch is the existing nonstandard aromatic
  triple-bond record at train index 521, whose information is not losslessly
  represented by the current aromatic-edge schema.
