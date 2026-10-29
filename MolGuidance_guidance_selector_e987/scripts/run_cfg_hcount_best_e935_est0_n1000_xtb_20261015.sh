#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/cfg_hcount_best_e935_est0_n1000_xtb_20261015"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
XTB_ENV="/home/jovyan/cpu-only-1-datavol-1/xtb_env"
RAW_TRAIN="$WORKSPACE_DIR/joint_kekule_diffusion/data_hp1kg_nevpt2/train.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_est_hcount"
NORM_FILE="$DB_DIR/train_data_property_normalization.pt"
SOURCE_CHECKPOINT="$WORKSPACE_DIR/runs/molguidance_guidance_est_hcount_20261014/cfg/None_ircuyahi/checkpoints/epoch=935-step=37440.ckpt"
CHECKPOINT="$OUT_ROOT/checkpoint/epoch=935-step=37440.ckpt"
TARGETS="$OUT_ROOT/targets_est_fixed.npy"
COUNTS="$OUT_ROOT/atom_counts_train_bootstrap.npy"
MANIFEST="$OUT_ROOT/condition_manifest.csv"
TARGET_EST="0.0"
SEED="20261015"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/status" "$OUT_ROOT/checkpoint"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another epoch-935 H-count supervisor is active" >&2
  exit 73
fi
if [[ -e "$OUT_ROOT/started.utc" ]]; then
  echo "Run root was already started: $OUT_ROOT" >&2
  exit 72
fi

printf '%s\n' "$$" > "$OUT_ROOT/supervisor.pid"
date -u +%FT%TZ > "$OUT_ROOT/started.utc"

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

for required in "$PYTHON_BIN" "$SOURCE_CHECKPOINT" "$RAW_TRAIN" "$NORM_FILE" "$XTB_ENV/bin/xtb"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 66
  fi
done

cp --reflink=auto --preserve=mode,timestamps "$SOURCE_CHECKPOINT" "$CHECKPOINT"
sha256sum "$CHECKPOINT" > "$OUT_ROOT/checkpoint/checkpoint.sha256"
git -C "$WORKSPACE_DIR" rev-parse HEAD > "$OUT_ROOT/code_commit.txt"
git -C "$WORKSPACE_DIR" status --short > "$OUT_ROOT/code_status.txt"
cp --preserve=mode,timestamps "$BASH_SOURCE" "$OUT_ROOT/pipeline.sh"
sha256sum "$OUT_ROOT/pipeline.sh" > "$OUT_ROOT/pipeline.sha256"

cd "$BUNDLE_DIR"
MANIFEST_RC=1
GENERATION_RC=1
ANALYSIS_RC=1
XTB_RC=1
RENDER_RC=1
EST_ANALYSIS_RC=1
EST_WINDOW_RC=1

run_job build_manifest "$PYTHON_BIN" -u tools/build_fixed_est_manifest.py \
  --train-data "$RAW_TRAIN" --out-dir "$OUT_ROOT" --n-mols 1000 \
  --target-est "$TARGET_EST" --seed "$SEED"
MANIFEST_RC=$?

if [[ "$MANIFEST_RC" -eq 0 ]]; then
  run_job generation "$PYTHON_BIN" -u sample_cfg.py \
    --model_checkpoint "$CHECKPOINT" \
    --n_mols 1000 --max_batch_size 32 --n_timesteps 100 \
    --property_name est --multilple_values_file "$TARGETS" \
    --number_of_atoms "$COUNTS" \
    --normalization_file_path "$NORM_FILE" \
    --output_file "$OUT_ROOT/samples.sdf" --gpu 0 \
    --guide_w_x 1.5 --guide_w_a 1.5 --guide_w_c 1.5 \
    --guide_w_h 1.5 --guide_w_e 1.5 \
    --dfm_type campbell_rate_matrix --guidance_format log \
    --where_to_apply_guide probabilities \
    --properties_handle_method concatenate_sum --dataset_name qme14s
  GENERATION_RC=$?
fi

if [[ "$GENERATION_RC" -eq 0 && -s "$OUT_ROOT/samples.sdf" ]]; then
  run_job analysis "$PYTHON_BIN" -u tools/analyze_generation.py \
    --sdf "$OUT_ROOT/samples.sdf" --db-dir "$DB_DIR" \
    --out-dir "$OUT_ROOT/analysis" --expected-count 1000 \
    --condition-manifest "$MANIFEST"
  ANALYSIS_RC=$?
fi

if [[ "$ANALYSIS_RC" -eq 0 && -s "$OUT_ROOT/analysis/unique_novel_connected.sdf" ]]; then
  run_job xtb_stda "$PYTHON_BIN" -u tools/joint_xtb_stda_pipeline_v5.py \
    --input-sdf "$OUT_ROOT/analysis/unique_novel_connected.sdf" \
    --output-dir "$OUT_ROOT/xtb" --xtb-env "$XTB_ENV" \
    --workers 24 --threads 1 --target-s1 0.0 --target-t1 0.0 \
    --seed "$SEED"
  XTB_RC=$?
fi

RESULTS="$OUT_ROOT/xtb/work/results.csv"
if [[ "$XTB_RC" -eq 0 && -s "$RESULTS" ]]; then
  run_job render_xtb "$PYTHON_BIN" -u tools/render_xtb_success_2d.py \
    --results "$RESULTS" --out-dir "$OUT_ROOT/xtb/rdkit_2d" \
    --target-s1 0.0
  RENDER_RC=$?

  run_job est_analysis "$PYTHON_BIN" -u tools/analyze_fixed_est_results.py \
    --generation-summary "$OUT_ROOT/analysis/summary.json" \
    --molecules-csv "$OUT_ROOT/analysis/molecules.csv" \
    --xtb-results "$RESULTS" --target-est "$TARGET_EST" \
    --out-dir "$OUT_ROOT/est_analysis"
  EST_ANALYSIS_RC=$?

  run_job est_window_render "$PYTHON_BIN" -u tools/render_est_window_2d.py \
    --results "$RESULTS" \
    --out-dir "$OUT_ROOT/est_analysis/est_0_to_0p5_png" \
    --min-est 0.0 --max-est 0.5 --page-size 40
  EST_WINDOW_RC=$?
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'manifest_exit=%s\n' "$MANIFEST_RC"
  printf 'generation_exit=%s\n' "$GENERATION_RC"
  printf 'analysis_exit=%s\n' "$ANALYSIS_RC"
  printf 'xtb_exit=%s\n' "$XTB_RC"
  printf 'render_exit=%s\n' "$RENDER_RC"
  printf 'est_analysis_exit=%s\n' "$EST_ANALYSIS_RC"
  printf 'est_window_render_exit=%s\n' "$EST_WINDOW_RC"
} > "$OUT_ROOT/status/supervisor.status"
date -u +%FT%TZ > "$OUT_ROOT/finished.utc"

for rc in "$MANIFEST_RC" "$GENERATION_RC" "$ANALYSIS_RC" "$XTB_RC" \
  "$RENDER_RC" "$EST_ANALYSIS_RC" "$EST_WINDOW_RC"; do
  [[ "$rc" -eq 0 ]] || exit 1
done
