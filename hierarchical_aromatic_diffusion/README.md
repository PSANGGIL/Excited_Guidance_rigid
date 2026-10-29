# Hierarchical aromatic diffusion

This is an isolated scaffold-first replacement for the simultaneous atom/cell model. Existing runs and checkpoints are not imported or overwritten.

## Factorization

1. `P(ring-system topology | S1,T1)` predicts ring count, ordered perimeter size, system aromaticity, pair relations, junction ports and coarse geometry.
2. `P(scaffold chemistry | topology,S1,T1)` assigns C/N/O/F, charge, H/radical state and Kekule perimeter bonds to ordered slots. Junction slot pairs are unioned, so fused/spiro atoms cannot disagree.
3. `P(substituents,attachments | scaffold,S1,T1)` adds non-ring atoms and scaffold attachment bonds.
4. `P(coordinates | complete graph,S1,T1)` jointly refines Cartesian coordinates while graph topology remains fixed.

The same RBF S1/T1 embedding enters every stage. Aromaticity is not constructed by independently thresholding every ring edge: Kekule bond orders are generated and RDKit sanitization derives the final aromatic graph.

## Invariants before training

- RDKit identity round-trip on the full selected DB.
- Ordered ring slot rotation/reversal invariance.
- Exact junction reconstruction for fused, spiro and bridged systems.
- Oracle assembly is connected and sanitizable without topology repair.
- Generated cell count must equal realized ring-system count; a valid atom graph alone is never sufficient.

## Validation

```bash
cd hierarchical_aromatic_diffusion
PYTHONPATH=.. python -m unittest discover -s tests -v
python validate_codec.py --root ../data_qcdge_a10_full --aromatic-only --output aromatic_codec_validation.json
```

`model.py` defines the condition-shared four-stage network. The codec is template-free: it stores ordered ring slots and junction mappings, not a vocabulary of database ring templates.
