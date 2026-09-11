# FPA entity diffusion v2

The generated chemical state contains every heavy atom exactly once. Fused
aromatic rings additionally receive auxiliary cells. Cells are discarded after
sampling; hydrogens are added by RDKit.

## State and forward processes

- Atom coordinates use VP Gaussian diffusion in the zero-centre-of-mass
  subspace.
- Cell centres are expressed relative to the heavy-atom centre and diffused as
  equivariant vectors.
- Positive cell radius is diffused in log space.
- Cell orientation is a rotation matrix perturbed through the SO(3) exponential
  map. Ring-frame D_n gauge augmentations prevent dependence on an arbitrary
  first atom, traversal direction, or normal sign.
- Atom type, charge, bond, cell type, cell size, fusion, and membership use an
  absorbing-mask discrete process. Reverse reveal probabilities are computed
  from alpha-bar rather than treating classification as a complete sampler.

Fusion ports are not integer labels. The intersection of the two cells'
permutation-equivariant atom memberships defines their shared atom pair. This
supports general fusion graphs without cyclic-port label ambiguity.

## Geometry and chemistry

Messages use atom radii and cell radial/axial distances. Cell geometry is thus
disk-like rather than a spherical exclusion radius. Training additionally
checks cell centre, planarity, radial extent, membership cardinality, shared
atom count, bond length/type, aromatic bonds, valence, and isolated atoms.

## Dataset limitation

`data_qcdge_a10_full` contains 106,963 training molecules and every molecule
has exactly 10 heavy atoms. Its FPA-cell histogram is `{0: 106194, 2: 766,
3: 3}`; maximum fusion degree is 2 and there are no branched FPA examples.
The architecture permits degree-3/T-shaped fusion, but learning that behavior
requires an A18-or-larger dataset containing branched fused aromatics.

## Entry points

- Train: `train_fpa_v2.py --config scripts/qcdge_a10_fpa_v2.yaml`
- Sample: `sample_fpa_v2.py --checkpoint ... --conditions ... --output ...`

V2 checkpoints are intentionally incompatible with legacy rigid/hybrid and FPA
prototype checkpoints.
