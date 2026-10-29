# Heavy-atom joint-state diffusion v5

V5 is checkpoint-incompatible with v4.  Existing v4 code, checkpoints, runs,
and generated structures remain untouched.

## Chemical representation

- Diffusion nodes are heavy atoms only.
- A single joint categorical state contains element, formal charge, total
  valence, and aromatic role.
- Hydrogen and radical are not independent diffusion heads.
- Hydrogen count is the residual between the joint-state total valence and the
  final heavy-atom bond-order sum.
- The state vocabulary is built automatically from all elements and chemical
  states present in the HP1KG cache and stored in checkpoint metadata.
- Aromatic hetero-H states such as `[nH]` have a dedicated aromatic role.

The database codec must reconstruct every molecule exactly before training.
The current HP1KG cache passes RDKit canonical identity for 1,581/1,581
molecules with 39 observed joint atom states.
The chemistry schema is label-independent; topology and ring priors are built
from the training split only and their provenance is stored in metadata.

## Strict graph projection

Generation solves the whole-molecule net-charge assignment with dynamic
programming. A connected, valence-safe predicted topology already inside the
training support is preserved. Otherwise it constructs a valence-constrained
spanning tree and projects the predicted cycle rank and maximum cyclic
biconnected-component size onto the training distribution conditional on
heavy-atom count. Out-of-support predictions are sampled from that empirical
distribution rather than clamped to an extreme. Additional closure edges are
ranked using the complete observed ring-size distribution. Every ring size has
finite smoothed support, so small, large, excessive, and fused/spiro rings are
controlled by data-derived priors rather than hand-written bans.

The final decoder requires all of the following:

- one connected component;
- net charge zero;
- zero radical electrons;
- RDKit hydrogen assignment exactly equal to residual valence;
- successful RDKit sanitization;
- RDKit `CalcNumAromaticRings` for aromatic evaluation.

## Hydrogen and xTB

The sampler writes a sanitized heavy-atom molecule with implicit hydrogen.  Use
`joint_xtb_stda_pipeline_v5.py` for electronic-structure calculations.  It
rejects charged/open-shell input, does not perform a SMILES round trip, and
calls `Chem.AddHs` exactly once before ETKDG/MMFF/UFF and xTB.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python -u \
train_joint_kekule_v5.py \
  --root joint_kekule_diffusion/data_hp1kg_nevpt2 \
  --run-dir runs/joint_kekule_hp1kg_joint_state_v5 \
  --epochs 10000 --batch-size 208 --precision fp32 --workers 4
```

## Generation

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python -u \
sample_joint_kekule_v5.py \
  --checkpoint runs/joint_kekule_hp1kg_joint_state_v5/checkpoints/latest.pt \
  --metadata runs/joint_kekule_hp1kg_joint_state_v5/metadata.json \
  --root joint_kekule_diffusion/data_hp1kg_nevpt2 \
  --output runs/joint_kekule_hp1kg_joint_state_v5/generation_test \
  --num 1000 --atoms 0 --s1 3.640037775 --t1 3.111467600 --cfg 1
```

Every generation run writes CSV, SDF, summary JSON, and RDKit 2D PNG pages.
