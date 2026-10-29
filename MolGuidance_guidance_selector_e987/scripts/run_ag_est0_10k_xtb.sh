#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/ag_est0_10k_finalbest_20261002"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
XTB_ENV="/home/jovyan/cpu-only-1-datavol-1/xtb_env"
RAW_TRAIN="$WORKSPACE_DIR/joint_kekule_diffusion/data_hp1kg_nevpt2/train.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_est"
NORM_FILE="$DB_DIR/train_data_property_normalization.pt"
SOURCE_SNAPSHOTS="$WORKSPACE_DIR/runs/guidance_compare_est_quintiles_finalbest_20261002/checkpoint_snapshots"
SNAPSHOTS="$OUT_ROOT/checkpoint_snapshots"
GOOD_CKPT="$SNAPSHOTS/base_good_epoch1349.ckpt"
BAD_CKPT="$SNAPSHOTS/ag_inferior_epoch291.ckpt"
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

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status" "$SNAPSHOTS" "$OUT_ROOT/ag"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another AG EST=0 10k supervisor is active" >&2
  exit 73
fi
printf '%s\n' "$$" > "$OUT_ROOT/pids/supervisor.pid"
date -u +%FT%TZ > "$OUT_ROOT/started.utc"
cd "$BUNDLE_DIR"

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

cp --reflink=auto --preserve=mode,timestamps \
  "$SOURCE_SNAPSHOTS/base_good_epoch1349.ckpt" "$GOOD_CKPT"
cp --reflink=auto --preserve=mode,timestamps \
  "$SOURCE_SNAPSHOTS/ag_inferior_epoch291.ckpt" "$BAD_CKPT"
sha256sum "$SNAPSHOTS"/*.ckpt > "$SNAPSHOTS/checkpoint_sha256.txt"

run_job build_manifest "$PYTHON_BIN" -u tools/build_fixed_est_manifest.py \
  --train-data "$RAW_TRAIN" --out-dir "$OUT_ROOT" --n-mols 10000 \
  --target-est "$TARGET_EST" --seed 1701
MANIFEST_RC=$?

GEN_RC=1
ANALYSIS_RC=1
XTB_RC=1
RENDER_RC=1
FINAL_RC=1
WINDOW_RC=1
if [[ "$MANIFEST_RC" -eq 0 ]]; then
  run_job ag_generation "$PYTHON_BIN" -u sample_guided.py \
    --guidance ag --good-model-checkpoint "$GOOD_CKPT" \
    --bad-model-checkpoint "$BAD_CKPT" --max-batch-size 64 \
    --guide-w-x 1.5 --guide-w-a 1.5 --guide-w-c 1.5 --guide-w-e 1.5 \
    --dfm-type campbell_rate_matrix --guidance-format log \
    --where-to-apply-guide probabilities \
    --properties-handle-method concatenate_sum \
    --n-mols 10000 --n-timesteps 100 --property-name est \
    --multiple-values-file "$TARGETS" --number-of-atoms-file "$COUNTS" \
    --normalization-file-path "$NORM_FILE" --dataset-name qme14s \
    --output-file "$OUT_ROOT/ag/samples.sdf"
  GEN_RC=$?
fi

if [[ "$GEN_RC" -eq 0 && -s "$OUT_ROOT/ag/samples.sdf" ]]; then
  run_job ag_analysis "$PYTHON_BIN" -u tools/analyze_generation.py \
    --sdf "$OUT_ROOT/ag/samples.sdf" --db-dir "$DB_DIR" \
    --out-dir "$OUT_ROOT/ag/analysis" --expected-count 10000 \
    --condition-manifest "$MANIFEST"
  ANALYSIS_RC=$?
fi

if [[ "$ANALYSIS_RC" -eq 0 && -s "$OUT_ROOT/ag/analysis/unique_novel_connected.sdf" ]]; then
  run_job ag_xtb "$PYTHON_BIN" -u tools/joint_xtb_stda_pipeline_v5.py \
    --input-sdf "$OUT_ROOT/ag/analysis/unique_novel_connected.sdf" \
    --output-dir "$OUT_ROOT/xtb/ag" --xtb-env "$XTB_ENV" \
    --workers 24 --threads 1 --target-s1 0.0 --target-t1 0.0 \
    --seed 20261002
  XTB_RC=$?
fi

RESULTS="$OUT_ROOT/xtb/ag/work/results.csv"
if [[ "$XTB_RC" -eq 0 && -s "$RESULTS" ]]; then
  run_job ag_render "$PYTHON_BIN" -u tools/render_xtb_success_2d.py \
    --results "$RESULTS" --out-dir "$OUT_ROOT/xtb/ag/rdkit_2d" \
    --target-s1 0.0
  RENDER_RC=$?
  run_job final_analysis "$PYTHON_BIN" -u tools/analyze_fixed_est_results.py \
    --generation-summary "$OUT_ROOT/ag/analysis/summary.json" \
    --molecules-csv "$OUT_ROOT/ag/analysis/molecules.csv" \
    --xtb-results "$RESULTS" --target-est "$TARGET_EST" \
    --out-dir "$OUT_ROOT/est_analysis"
  FINAL_RC=$?
  run_job est_window_render "$PYTHON_BIN" -u tools/render_est_window_2d.py \
    --results "$RESULTS" --out-dir "$OUT_ROOT/est_analysis/est_0_to_0p5_png" \
    --min-est 0.0 --max-est 0.5 --page-size 40
  WINDOW_RC=$?
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'manifest_exit=%s\n' "$MANIFEST_RC"
  printf 'generation_exit=%s\n' "$GEN_RC"
  printf 'analysis_exit=%s\n' "$ANALYSIS_RC"
  printf 'xtb_exit=%s\n' "$XTB_RC"
  printf 'render_exit=%s\n' "$RENDER_RC"
  printf 'final_analysis_exit=%s\n' "$FINAL_RC"
  printf 'est_window_render_exit=%s\n' "$WINDOW_RC"
} > "$OUT_ROOT/status/supervisor.status"
date -u +%FT%TZ > "$OUT_ROOT/finished.utc"

if [[ "$GEN_RC" -ne 0 || "$ANALYSIS_RC" -ne 0 || "$XTB_RC" -ne 0 || "$RENDER_RC" -ne 0 || "$FINAL_RC" -ne 0 || "$WINDOW_RC" -ne 0 ]]; then
  exit 1
fi
