#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT=/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid
REPO_ROOT=${TASK_ROOT}/MolGuidance_guidance_selector_e987
RUN_ROOT=${TASK_ROOT}/runs/molguidance_hp1kg_official_xyz_cfg_bs1264_lr1p5e3_20261015
CONFIG=${REPO_ROOT}/configs/local/guidance_est/hp1kg_est_cfg_hcount_official_xyz_bs1264_70pct.yaml
PYTHON_BIN=/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python

mkdir -p "${RUN_ROOT}/logs"
printf '%s\n' "$$" > "${RUN_ROOT}/training.pid"
printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/started_at_utc.txt"

on_exit() {
    status=$?
    printf '%s\n' "${status}" > "${RUN_ROOT}/exit_code.txt"
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${RUN_ROOT}/finished_at_utc.txt"
}
trap on_exit EXIT

cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

exec "${PYTHON_BIN}" -u train.py \
    --config "${CONFIG}" \
    --seed 20261015
