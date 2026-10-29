#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/guidance_compare_s1_3p640_20261001"
FRESH_ROOT="$WORKSPACE_DIR/runs/molguidance_guidance_fresh_20260930"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
GOOD_CKPT="$FRESH_ROOT/base_good/None_e8607ywj/checkpoints/epoch=1349-step=54000.ckpt"
CLASSIFIER_CKPT="$FRESH_ROOT/cg_classifier/property_predictor_s1/checkpoints/epoch=153-val_loss=0.4944.ckpt"
NORM_FILE="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_s1/train_data_property_normalization.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_s1"
COUNTS_FILE="$OUT_ROOT/atom_counts_seed1701.npy"
STATUS_FILE="$OUT_ROOT/status/cg_retry.status"
LOG_FILE="$OUT_ROOT/logs/cg_retry.log"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p "$OUT_ROOT/cg" "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status"
exec 9>"$OUT_ROOT/cg_retry.lock"
if ! flock -n 9; then
  echo "A CG retry is already active" >&2
  exit 73
fi

{
  printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'pid=%s\n' "$$"
} > "$STATUS_FILE"
printf '%s\n' "$$" > "$OUT_ROOT/pids/cg_retry.pid"

cd "$BUNDLE_DIR"
"$PYTHON_BIN" -u sample_guided.py --guidance cg \
  --flowmol-checkpoint "$GOOD_CKPT" \
  --classifier-checkpoint "$CLASSIFIER_CKPT" \
  --n-mols 1000 --n-timesteps 100 --max-batch-size 64 \
  --property-name s1 --properties-for-sampling 3.640037775039673 \
  --number-of-atoms-file "$COUNTS_FILE" \
  --normalization-file-path "$NORM_FILE" \
  --dataset-name qme14s --analyze \
  --guide-w-x 1 --guide-w-a 1 --guide-w-c 1 --guide-w-e 1 \
  --output-file "$OUT_ROOT/cg/samples.sdf" \
  > "$LOG_FILE" 2>&1
rc=$?

if (( rc == 0 )); then
  "$PYTHON_BIN" -u tools/analyze_generation.py \
    --sdf "$OUT_ROOT/cg/samples.sdf" --db-dir "$DB_DIR" \
    --out-dir "$OUT_ROOT/cg/analysis" --expected-count 1000 \
    > "$OUT_ROOT/logs/cg_analysis.log" 2>&1
  rc=$?
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'exit_code=%s\n' "$rc"
} >> "$STATUS_FILE"
exit "$rc"
