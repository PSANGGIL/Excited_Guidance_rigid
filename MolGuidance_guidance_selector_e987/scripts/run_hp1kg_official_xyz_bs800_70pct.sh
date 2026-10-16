#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid
REPO_ROOT=${TASK_ROOT}/MolGuidance_guidance_selector_e987
RUN_ROOT=${TASK_ROOT}/runs/molguidance_hp1kg_official_xyz_cfg_bs800_lr1p25e3_20261015
CONFIG=${REPO_ROOT}/configs/local/guidance_est/hp1kg_est_cfg_hcount_official_xyz_bs800_70pct.yaml
PYTHON_BIN=/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python

mkdir -p "${RUN_ROOT}/logs"
printf '%s\n' "$$" > "${RUN_ROOT}/training.pid"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/started_at_utc.txt"

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

exec "${PYTHON_BIN}" -u train.py \
    --config "${CONFIG}" \
    --seed 20261015
