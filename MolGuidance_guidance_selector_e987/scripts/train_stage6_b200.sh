#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${STAGE6_PYTHON:-/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python}"
config="${STAGE6_CONFIG:-${project_dir}/configs/local/guidance_est/hp1kg_est_cfg_joint_kekule_stage6_independent_time_bond_first.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_NAME="${WANDB_NAME:-hp1kg-est-cfg-stage6-b640-expandable}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${project_dir}/..:${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${project_dir}"
exec "${python_bin}" -u train_with_guidance.py \
  --config "${config}" \
  --seed "${STAGE6_SEED:-17041}" \
  "$@"
