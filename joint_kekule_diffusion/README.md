# Joint Kekule molecular DDPM v4

Self-contained source bundle for the consistency-corrected aromatic QCDGE A+B model. The older A-only PID may keep running because Python already loaded its original modules; start v4 as a fresh run.

## Core guarantees

- Atom, charge, explicit-H, radical, single/double/triple bond and Cartesian coordinates share one diffusion timestep.
- Each undirected bond uses one symmetric corruption realization in training and sampling.
- Aromatic bonds are supervised edge-wise, including fused systems with explicit single fusion edges.
- Ring loss marginalizes over all non-adjacent Kekule placements with the observed number of double bonds; it does not force one RDKit Kekule phase.
- Valence, differentiable spanning-tree connectivity, net charge and radical-count consistency losses are included.
- Padding is isolated; atom permutation and SE(3) equivariance are regression-tested.
- Validation noise is deterministic; checkpoints are atomically replaced and include RNG state.
- Sampling validates count/CFG inputs, detects non-finite states, reports DB novelty separately from generated uniqueness, writes 2D PNG pages and a Cartesian SDF.

## Files

- `model.py`: joint DDPM, valid-Kekule marginal and global chemistry losses.
- `data.py`, `real_data.py`, `hybrid_data.py`, `fpa_data_template_free.py`: QCDGE loader chain.
- `train.py`: BF16 trainer, deterministic validation, atomic checkpoint/resume.
- `sample.py`: guarded reverse diffusion, strict RDKit decode, novelty, PNG and SDF export.
- `build_qcdge_ab_aromatic.py`: reproducible A+B aromatic conversion with strict codec identity filtering.
- `data_qcdge_ab_aromatic/`: 32,663 lossless, deduplicated aromatic molecules; see `validation_report.json`.
- `data_qcdge_a10_full/`: preserved A-only database used by the older run.
- `raw_qcdge/A_10.hdf5`, `raw_qcdge/B_10.hdf5`: preserved raw inputs.

## Train v4 from scratch

```bash
PYTHONPATH=. /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python -m joint_kekule_diffusion.train \
  --root joint_kekule_diffusion/data_hp1kg_nevpt2 --dataset-kind hp1kg --max-atoms 69 \
  --run-dir runs/joint_kekule_hp1kg_marginal_d3pm_v4 \
  --epochs 10000 --batch-size 320 --workers 4 --hidden 256 --layers 8
```

v4 optimizer resume is intentionally incompatible with v1/v2/v3 because the categorical transition kernels and training objective changed; train from scratch.

## Sample

```bash
PYTHONPATH=. /home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python -m joint_kekule_diffusion.sample \
  --checkpoint runs/joint_kekule_hp1kg_marginal_d3pm_v4/checkpoints/latest.pt \
  --metadata runs/joint_kekule_hp1kg_marginal_d3pm_v4/metadata.json \
  --root joint_kekule_diffusion/data_hp1kg_nevpt2 \
  --output runs/joint_kekule_hp1kg_marginal_d3pm_v4/sample_1000 --num 1000 --atoms 0
```

Outputs include `molecules.csv`, `summary.json`, `unique_novel_connected.sdf`, and `novel_valid_*.png`.


## HP1KG / BenchmarkSet1500 input

`hp1kg_data.py` adds a separate NEVPT2 adapter for the 1,581-molecule HP1KG dataset. It preserves all observed B/C/N/O/F/Si/P/S/Cl/As/Se/Br atom classes, formal charges, Kekule bond targets, aromatic-cycle supervision, negative S1-T1 gaps, and variable heavy-atom counts (6-69). Since the CSV has no Cartesian coordinates, deterministic ETKDGv3 conformers are cached; the manifest records MMFF/UFF/unoptimized/2D provenance.

Build and train with:

```bash
python -m joint_kekule_diffusion.hp1kg_data --csv /path/to/BenchmarkSet1500.csv --output joint_kekule_diffusion/data_hp1kg_nevpt2
python -m joint_kekule_diffusion.train --root joint_kekule_diffusion/data_hp1kg_nevpt2 --dataset-kind hp1kg --max-atoms 69 --run-dir runs/hp1kg_nevpt2
```

QCDGE and HP1KG excited-state labels must not be silently pooled: HP1KG conditioning uses NEVPT2 S1/T1 and carries its own train-split normalization.
## Chemistry-stable empirical-marginal D3PM v4

The v4 model uses empirical stationary categorical transitions, exact generalized reverse posteriors, hybrid x0-CE/VLB training, separate edge-existence and conditional bond-type channels, fused-component Kekule constraints, valence/connectivity projection, and periodic full reverse-rollout evaluation. It uses normalized graph-size losses, exact dynamic-programming Kekule marginalization, non-Kekule aromatic exceptions, charge-aware hetero matching, clipped x0 coordinate posteriors, and isolated rollout failures. It requires fresh training and intentionally rejects older optimizer checkpoints. Condition fidelity is monitored by a deterministic correct-condition versus shuffled-condition denoising-loss delta; final electronic-property fidelity still requires external quantum-chemistry evaluation.

Design references: Austin et al. (D3PM); Vignac et al. (DiGress, MiDi); Hoogeboom et al. (EDM); Jo et al. (GDSS); Xu et al. (GeoDiff); Peng et al. (MolDiff); Huang et al. (JODO); Satorras et al. (EGNN); Nichol and Dhariwal (Improved DDPM); RDKit aromaticity and Kekulization documentation.
