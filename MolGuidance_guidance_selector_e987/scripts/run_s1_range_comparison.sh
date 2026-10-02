#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/guidance_compare_s1_quintiles_20261001"
FRESH_ROOT="$WORKSPACE_DIR/runs/molguidance_guidance_fresh_20260930"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
XTB_ENV="/home/jovyan/cpu-only-1-datavol-1/xtb_env"
NORM_FILE="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_s1/train_data_property_normalization.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_s1"
RAW_TRAIN="$WORKSPACE_DIR/joint_kekule_diffusion/data_hp1kg_nevpt2/train.pt"
TARGETS_FILE="$OUT_ROOT/targets_s1_seed1701.npy"
COUNTS_FILE="$OUT_ROOT/atom_counts_seed1701.npy"
MANIFEST="$OUT_ROOT/condition_manifest.csv"

CFG_CKPT="$FRESH_ROOT/cfg/None_3wupt02q/checkpoints/epoch=987-step=39520.ckpt"
MG_CKPT="$FRESH_ROOT/mg/None_mabj85bv/checkpoints/epoch=1938-step=77560.ckpt"
GOOD_CKPT="$FRESH_ROOT/base_good/None_e8607ywj/checkpoints/epoch=1349-step=54000.ckpt"
CLASSIFIER_CKPT="$FRESH_ROOT/cg_classifier/property_predictor_s1/checkpoints/epoch=153-val_loss=0.4944.ckpt"
BAD_CKPT="$FRESH_ROOT/ag_inferior/None_unv9xb53/checkpoints/epoch=291-step=11680.ckpt"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another S1 range supervisor is active" >&2
  exit 73
fi
printf '%s\n' "$$" > "$OUT_ROOT/pids/supervisor.pid"
date -u +%FT%TZ > "$OUT_ROOT/started.utc"

cd "$BUNDLE_DIR"

run_job() {
  local name="$1"
  shift
  local log="$OUT_ROOT/logs/${name}.log"
  local status="$OUT_ROOT/status/${name}.status"
  local rc
  {
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'pid=%s\n' "$BASHPID"
    printf 'command='; printf '%q ' "$@"; printf '\n'
  } > "$status"
  "$@" > "$log" 2>&1
  rc=$?
  {
    printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'exit_code=%s\n' "$rc"
  } >> "$status"
  return "$rc"
}

if [[ ! -s "$MANIFEST" || ! -s "$TARGETS_FILE" || ! -s "$COUNTS_FILE" ]]; then
  run_job build_manifest "$PYTHON_BIN" -u tools/build_s1_range_manifest.py \
    --train-data "$RAW_TRAIN" --out-dir "$OUT_ROOT" --per-range 200 --seed 1701 || exit 1
fi

COMMON=(
  --n-mols 1000 --n-timesteps 100 --property-name s1
  --multiple-values-file "$TARGETS_FILE"
  --number-of-atoms-file "$COUNTS_FILE"
  --normalization-file-path "$NORM_FILE"
  --dataset-name qme14s
)

declare -A SAMPLE_RC
declare -A XTB_PID
declare -A XTB_WORKERS=( [cfg]=5 [mg]=5 [cg]=5 [ag]=5 [conditional]=4 )

start_xtb() {
  local method="$1"
  local source="$OUT_ROOT/$method/analysis/unique_novel_connected.sdf"
  local xtb_out="$OUT_ROOT/xtb/$method"
  run_job "${method}_xtb" "$PYTHON_BIN" -u tools/joint_xtb_stda_pipeline_v5.py \
    --input-sdf "$source" --output-dir "$xtb_out" --xtb-env "$XTB_ENV" \
    --workers "${XTB_WORKERS[$method]}" --threads 1 \
    --target-s1 3.640037775039673 --target-t1 3.111 \
    --seed 20261001 &
  XTB_PID[$method]=$!
  printf '%s\n' "${XTB_PID[$method]}" > "$OUT_ROOT/pids/${method}_xtb.pid"
}

for method in cfg mg cg ag conditional; do
  mkdir -p "$OUT_ROOT/$method"
  command=("$PYTHON_BIN" -u sample_guided.py --guidance "$method")
  case "$method" in
    cfg)
      command+=(--model-checkpoint "$CFG_CKPT" --max-batch-size 32
        --guide-w-x 1.5 --guide-w-a 1.5 --guide-w-c 1.5 --guide-w-e 1.5
        --dfm-type campbell_rate_matrix --guidance-format log
        --where-to-apply-guide probabilities --properties-handle-method concatenate_sum)
      ;;
    mg)
      command+=(--model-checkpoint "$MG_CKPT" --max-batch-size 32
        --guide-weight 1.5 --properties-handle-method sum)
      ;;
    cg)
      command+=(--flowmol-checkpoint "$GOOD_CKPT"
        --classifier-checkpoint "$CLASSIFIER_CKPT" --max-batch-size 8
        --guide-w-x 1 --guide-w-a 1 --guide-w-c 1 --guide-w-e 1)
      ;;
    ag)
      command+=(--good-model-checkpoint "$GOOD_CKPT"
        --bad-model-checkpoint "$BAD_CKPT" --max-batch-size 16
        --guide-w-x 1.5 --guide-w-a 1.5 --guide-w-c 1.5 --guide-w-e 1.5
        --dfm-type campbell_rate_matrix --guidance-format log
        --where-to-apply-guide probabilities --properties-handle-method concatenate_sum)
      ;;
    conditional)
      command+=(--model-checkpoint "$GOOD_CKPT" --max-batch-size 32
        --properties-handle-method concatenate_sum)
      ;;
  esac
  command+=("${COMMON[@]}" --output-file "$OUT_ROOT/$method/samples.sdf")
  SAMPLE_RC[$method]=0
  run_job "${method}_generation" "${command[@]}" || SAMPLE_RC[$method]=$?
  if [[ "${SAMPLE_RC[$method]}" -eq 0 && -s "$OUT_ROOT/$method/samples.sdf" ]]; then
    run_job "${method}_analysis" "$PYTHON_BIN" -u tools/analyze_generation.py \
      --sdf "$OUT_ROOT/$method/samples.sdf" --db-dir "$DB_DIR" \
      --out-dir "$OUT_ROOT/$method/analysis" --expected-count 1000 \
      --condition-manifest "$MANIFEST" || SAMPLE_RC[$method]=1
  fi
  if [[ "${SAMPLE_RC[$method]}" -eq 0 && -s "$OUT_ROOT/$method/analysis/unique_novel_connected.sdf" ]]; then
    start_xtb "$method"
  fi
done

declare -A XTB_RC
for method in cfg mg cg ag conditional; do
  XTB_RC[$method]=1
  if [[ -n "${XTB_PID[$method]:-}" ]]; then
    XTB_RC[$method]=0
    wait "${XTB_PID[$method]}" || XTB_RC[$method]=$?
    results="$OUT_ROOT/xtb/$method/work/results.csv"
    if [[ -s "$results" ]]; then
      run_job "${method}_render" "$PYTHON_BIN" -u tools/render_xtb_success_2d.py \
        --results "$results" --out-dir "$OUT_ROOT/xtb/$method/rdkit_2d" \
        --target-s1 3.640037775039673 || XTB_RC[$method]=1
    fi
  fi
done

run_job aggregate "$PYTHON_BIN" -u tools/analyze_s1_range_comparison.py \
  --root "$OUT_ROOT" --manifest "$MANIFEST"
AGG_RC=$?

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  for method in cfg mg cg ag conditional; do
    printf '%s_generation_exit=%s\n' "$method" "${SAMPLE_RC[$method]}"
    printf '%s_xtb_exit=%s\n' "$method" "${XTB_RC[$method]}"
  done
  printf 'aggregate_exit=%s\n' "$AGG_RC"
} > "$OUT_ROOT/status/supervisor.status"

date -u +%FT%TZ > "$OUT_ROOT/finished.utc"
exit "$AGG_RC"
