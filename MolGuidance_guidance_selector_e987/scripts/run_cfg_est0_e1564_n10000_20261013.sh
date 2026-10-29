#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/cfg_est0_e1564_n10000_corrected_20261013"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
RAW_TRAIN="$WORKSPACE_DIR/joint_kekule_diffusion/data_hp1kg_nevpt2/train.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_est"
NORM_FILE="$DB_DIR/train_data_property_normalization.pt"
CHECKPOINT="$WORKSPACE_DIR/runs/molguidance_guidance_est_20261001/cfg/None_i1zgib6q/checkpoints/epoch=1564-step=62600.ckpt"
TARGETS="$OUT_ROOT/targets_est_fixed.npy"
COUNTS="$OUT_ROOT/atom_counts_train_bootstrap.npy"
MANIFEST="$OUT_ROOT/condition_manifest.csv"
TARGET_EST="0.0"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/status" "$OUT_ROOT/analysis"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another CFG EST=0 epoch-1564 10k supervisor is active" >&2
  exit 73
fi
if [[ -e "$OUT_ROOT/started.utc" ]]; then
  echo "Run root was already started: $OUT_ROOT" >&2
  exit 72
fi

printf '%s\n' "$$" > "$OUT_ROOT/supervisor.pid"
date -u +%FT%TZ > "$OUT_ROOT/started.utc"
git -C "$WORKSPACE_DIR" rev-parse HEAD > "$OUT_ROOT/code_commit.txt"
git -C "$WORKSPACE_DIR" status --short > "$OUT_ROOT/code_status.txt"
sha256sum "$CHECKPOINT" > "$OUT_ROOT/checkpoint.sha256"

run_job() {
  local name="$1"; shift
  local log="$OUT_ROOT/logs/${name}.log"
  local status="$OUT_ROOT/status/${name}.status"
  local rc
  {
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'pid=%s\n' "$BASHPID"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } > "$status"
  "$@" > "$log" 2>&1
  rc=$?
  {
    printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'exit_code=%s\n' "$rc"
  } >> "$status"
  return "$rc"
}

cd "$BUNDLE_DIR"

MANIFEST_RC=1
GENERATION_RC=1
ANALYSIS_RC=1

run_job build_manifest "$PYTHON_BIN" -u tools/build_fixed_est_manifest.py \
  --train-data "$RAW_TRAIN" --out-dir "$OUT_ROOT" --n-mols 10000 \
  --target-est "$TARGET_EST" --seed 20261013
MANIFEST_RC=$?

if [[ "$MANIFEST_RC" -eq 0 ]]; then
  run_job generation "$PYTHON_BIN" -u sample_cfg.py \
    --model_checkpoint "$CHECKPOINT" \
    --n_mols 10000 --max_batch_size 32 --n_timesteps 100 \
    --property_name est --multilple_values_file "$TARGETS" \
    --number_of_atoms "$COUNTS" \
    --normalization_file_path "$NORM_FILE" \
    --output_file "$OUT_ROOT/samples.sdf" --gpu 0 \
    --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 --guide_w_e 1.5 \
    --dfm_type campbell_rate_matrix --guidance_format log \
    --where_to_apply_guide probabilities \
    --properties_handle_method concatenate_sum --dataset_name qme14s
  GENERATION_RC=$?
fi

if [[ "$GENERATION_RC" -eq 0 && -s "$OUT_ROOT/samples.sdf" ]]; then
  run_job analysis "$PYTHON_BIN" -u tools/analyze_generation.py \
    --sdf "$OUT_ROOT/samples.sdf" --db-dir "$DB_DIR" \
    --out-dir "$OUT_ROOT/analysis" --expected-count 10000 \
    --condition-manifest "$MANIFEST"
  ANALYSIS_RC=$?
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'manifest_exit=%s\n' "$MANIFEST_RC"
  printf 'generation_exit=%s\n' "$GENERATION_RC"
  printf 'analysis_exit=%s\n' "$ANALYSIS_RC"
} > "$OUT_ROOT/status/supervisor.status"
date -u +%FT%TZ > "$OUT_ROOT/finished.utc"

if [[ "$MANIFEST_RC" -ne 0 || "$GENERATION_RC" -ne 0 || "$ANALYSIS_RC" -ne 0 ]]; then
  exit 1
fi
