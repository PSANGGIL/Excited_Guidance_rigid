# MolGuidance official-code HP1KG baseline

The official repository is cloned, unmodified, at `../MolGuidance_official`.
This directory contains only the dataset adapter and external configuration
needed to run its original CTMC/flow-matching implementation on the existing
HP1KG NEVPT2 split.

The first controlled experiment uses S1 as the one scalar condition because
the official model's property embedder has input dimension one. No V8/V9
projection, tree/closure model, extra chemical loss, or custom sampler is used.

From the parent repository:

```bash
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python \
  molguidance_hp1kg_baseline/convert_hp1kg.py \
  --property s1 \
  --output molguidance_hp1kg_baseline/data_s1
```

Then, from `MolGuidance_official`:

```bash
WANDB_MODE=offline \
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python train_with_guidance.py \
  --config ../molguidance_hp1kg_baseline/hp1kg_s1_cfg.yaml
```

Important controlled deviations are data-only: the atom vocabulary and split
files follow HP1KG, and `train_a`/`train_b` are aliases of the existing full
training split because the unmodified official data module always selects one
of those names. The conversion report records unsupported source targets
(implicit hydrogen counts and radical electrons) rather than silently claiming
that the official schema models them.
