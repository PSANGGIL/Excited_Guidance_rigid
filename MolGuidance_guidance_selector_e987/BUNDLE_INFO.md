# MolGuidance Guidance Selector Bundle

This directory is a reproducible code snapshot for the HP1KG S1-conditioned
MolGuidance run and its guidance-selection sampling workflow.

## Provenance

- Source repository: `MolGuidance_official_loss_equiv`
- Source branch: `feature/hp1kg-original-loss-equivalent`
- Source commit: `d281705` (`feat: add selectable guidance sampling CLI`)
- Snapshot date: 2026-09-30 UTC

The directory intentionally has no `.git`, runtime cache, W&B run, generated
molecule, or training-log content.

## Entry points

- `train_with_guidance.py`: guided model training
- `sample_guided.py`: unified sampler selected with `--guidance`
- `sample_cfg.py`: classifier-free guidance implementation
- `sample_cg.py`: classifier guidance implementation
- `sample_mg.py`: model guidance implementation
- `sample_ag.py`: autoguidance implementation
- `sample_condition.py`: conditional sampling without additional guidance
- `configs/local/hp1kg_s1_original_loss_equiv_cfg_bs32_2k.yaml`: HP1KG CFG training configuration
- `tests/test_sample_guided_cli.py`: unified CLI dispatch and validation tests

Additional comparison utilities are under `tools/`:

- `analyze_generation.py`: RDKit validity, novelty, ring statistics, and 2D PNGs
- `joint_xtb_stda_pipeline_v5.py`: strict ETKDG + XTB/sTDA pipeline
- `joint_xtb_stda_pipeline.py`: shared XTB/sTDA execution and aggregation code
- `render_xtb_success_2d.py`: RDKit 2D pages for successful XTB results

## External resources

These large/runtime resources are not copied into this code-only snapshot:

- Verified epoch 987 checkpoint:
  `/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/runs/molguidance_official_hp1kg_s1_loss_equiv_cfg_bs32_2k/None_3meyuofd/checkpoints/epoch=987-step=39520.ckpt`
- HP1KG processed MolGuidance data and S1 normalization:
  `/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/molguidance_hp1kg_baseline/data_s1`
- Raw HP1KG data:
  `/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/joint_kekule_diffusion/data_hp1kg_nevpt2`
- Python/RDKit environment:
  `/home/jovyan/cpu-only-1-datavol-1/mg_b200`
- XTB environment:
  `/home/jovyan/cpu-only-1-datavol-1/xtb_env`

## Verified CFG invocation

```bash
/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python sample_guided.py \
  --guidance cfg \
  --model-checkpoint /home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/runs/molguidance_official_hp1kg_s1_loss_equiv_cfg_bs32_2k/None_3meyuofd/checkpoints/epoch=987-step=39520.ckpt \
  --property-name s1 \
  --properties-for-sampling 3.640037775039673 \
  --normalization-file-path /home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid/molguidance_hp1kg_baseline/data_s1/train_data_property_normalization.pt \
  --n-mols 1000 --max-batch-size 16 --n-timesteps 100 \
  --guide-w-x 2 --guide-w-a 2 --guide-w-c 2 --guide-w-e 2 \
  --output-file sampled_epoch987.sdf --gpu 0
```

The epoch 987 checkpoint is a CFG checkpoint. Other selector modes require
their matching model types: CG needs a base model plus classifier, MG needs a
ModelGuidance checkpoint, and AG needs good and inferior model checkpoints.

## Fresh four-guidance benchmark

`scripts/run_fresh_guidance_benchmark.sh` trains from scratch without loading an
existing checkpoint. CFG and model guidance use dedicated models. Classifier
guidance and adaptive guidance share the fresh `base_good` model; the fourth GPU
lane trains the classifier first and then the adaptive-guidance inferior model.
Use `scripts/status_fresh_guidance_benchmark.sh` to inspect processes, logs, GPU
usage, and checkpoints. Runtime outputs are kept outside this bundle under
`../runs/molguidance_guidance_fresh_20260930`.
