# Template-free FPA diffusion model

## Purpose

The template-free FPA model preserves rings during Cartesian diffusion without
restricting generation to complete ring templates observed in the database.
Ring cells provide a larger geometric seed, while atom identities, formal
charges, hydrogens, radicals, bonds, and hetero-atom positions remain explicit
atom-level variables.

This implementation is separate from the trained legacy hybrid checkpoint. Its
input and output heads differ, so hybrid checkpoints must not be loaded into the
template-free model.

## Generative state

Each molecule contains one explicit node per heavy atom and zero or more
auxiliary ring cells.

Atom state:

- Cartesian position;
- element type and formal charge;
- attached-hydrogen count and radical-electron count;
- symmetric heavy-atom bond type matrix.

Cell state:

- ring size for 3--10 member rings, plus an unknown/out-of-range path;
- aromatic flag;
- centre, positive radius, and local orthonormal frame;
- exact cell-to-atom membership;
- pair relation: disjoint, directly linked, spiro, or the number of shared
  atoms for fused/overlapping cells.

Cells are auxiliary constraints rather than molecular fragments. They are
discarded after projection; the decoded molecule is defined by the atom graph.
This permits linear, angular, branched, and T-shaped fused-polycyclic layouts
without assigning one vocabulary token to every complete polyaromatic system.

## Diffusion and conditioning

- Atom coordinates and cell centres use continuous Gaussian diffusion.
- Cell frames use SO(3)-consistent rotation perturbations.
- Positive radii are represented in log space.
- Atom, bond, charge, hydrogen, radical, cell, relation, and membership states
  use masked categorical corruption and reverse prediction.
- The QCDGE spectrum vector contains masked singlet/triplet energy and
  oscillator-strength channels.
- Classifier-free condition dropout is applied during training. Conditional and
  null predictions can be compared through logged condition-gain and CFG-response
  diagnostics, including count heads.

## Projection and decoding

The projection stage enforces cell cardinality, predicted valid cycles, exact
cell overlaps, fused-edge consistency, direct linked-cell bonds, and atom
valence budgets. Hydrogens are not diffusion coordinates: attached-hydrogen
counts are decoded and RDKit performs final sanitization/export.

Source SMILES are graph-isomorphism mapped onto coordinate atom indices when
constructing hydrogen, radical, formal-charge, and bond sidecars. Rows whose
source SMILES cannot be parsed retain the original coordinate-derived graph and
are explicitly masked rather than silently receiving an incorrect atom mapping.

## Verified data contract

The encode/decode identity validator was run over all 133,704 model-visible
QCDGE A10 molecules:

- exact ring membership: 133,704 / 133,704;
- exact heavy-atom bond tensor: 133,704 / 133,704;
- canonical identity for RDKit-parseable references: 131,728 / 131,731;
- valid projection: 133,690 / 133,704;
- successful RDKit reconstruction: 133,650 / 133,704.

There are 1,973 source rows whose reference SMILES are not RDKit-parseable,
three parseable canonical-identity mismatches, fourteen projection exceptions,
and fifty-four RDKit reconstruction exceptions. These exceptions are retained
as explicit validation results; they are not counted as a perfect molecular
round trip.

## Entry points

```bash
PYTHONPATH=src python augment_qcdge_hydrogen_counts_v2.py --data data_qcdge_a10_full
PYTHONPATH=src python validate_fpa_identity.py --data data_qcdge_a10_full --split train --limit 999999
PYTHONPATH=src python train_fpa_template_free.py --config scripts/qcdge_a10_fpa_template_free.yaml
PYTHONPATH=src python sample_fpa_template_free.py --checkpoint <checkpoint> --output <samples.jsonl>
```

## Known limitations

- QCDGE A10 contains little supervision for large or highly branched fused
  systems; architectural support does not substitute for examples.
- Projection improves graph consistency but is non-differentiable and cannot
  guarantee realistic strain, conformers, aromatic electron assignment, or the
  requested excited-state property.
- Condition-response diagnostics measure model sensitivity to the condition;
  xTB/TD-DFT or another independent calculation is required to establish actual
  property attainment.
- The template-free model has passed unit, gradient, and database round-trip
  checks but has not yet undergone a full production training run.
