# Joint aromatic/Kekule diffusion v6

V6 is intentionally a new checkpoint schema (`schema_version=8`,
`model_kind=heavy_atom_joint_aromatic_d3pm_v6`). V5 checkpoints cannot be
resumed because the edge state and aromatic head shapes changed.

## Reverse state

The denoiser receives and predicts four categorical graph states plus Cartesian
noise:

- observed joint heavy-atom state;
- edge existence;
- Kekule bond order (single/double/triple);
- aromatic edge membership (regular/aromatic).

Aromatic membership is sampled with an exact D3PM posterior on every reverse
step. Aromatic edges are constrained to present, non-triple bonds after every
step. At sparse late reverse steps, cyclic biconnected components are selected
from the aromatic posterior and one global maximum matching assigns their
Kekule double bonds. Final strict projection is biased by the sampled `x0`
state, rather than discarding the reverse-chain sample.

## Database priors and support

Training metadata stores heavy-atom-count-conditioned joint topology samples
from the train split: cycle rank, maximum ring-system size, 3/4-member ring
count, bridgehead count, cumulene count and aromatic-ring count. Projection
uses the ring-size prior and a frequency-ratio acceptance prior for the three
historically overgenerated motifs.

Exact bond signatures and exact one-hop local atom environments are collected
from all DB splits. Strict generation rejects structures outside this support.
`--no-support-filter` and `--no-topology-prior` are diagnostic ablations, not
the production defaults.

## Checkpoint evaluation

The saved `evaluation.rollout` object reports:

- strict RDKit validity;
- chemically clean validity, separately from total validity;
- RDKit aromatic rate;
- ETKDGv3 success;
- MMFF94/UFF convergence;
- clash-free rate;
- mean/median ETKDG relaxation energy per heavy atom;
- failure and structural-warning counts.

Chemically clean means no unseen chemistry support, 3/4-member-ring,
bridgehead, cumulene, radical, disconnected, ETKDG, force-field convergence or
steric-clash warning.

## Commands

```bash
python train_joint_kekule_v6.py \
  --root joint_kekule_diffusion/data_hp1kg_nevpt2 \
  --run-dir runs/joint_kekule_hp1kg_v6

python sample_joint_kekule_v6.py \
  --checkpoint runs/joint_kekule_hp1kg_v6/checkpoints/latest.pt \
  --metadata runs/joint_kekule_hp1kg_v6/metadata.json \
  --output runs/joint_kekule_hp1kg_v6/generation_1000 \
  --num 1000 --evaluate-3d
```
