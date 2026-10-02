#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/guidance_compare_s1_3p640_20261001"
FRESH_ROOT="$WORKSPACE_DIR/runs/molguidance_guidance_fresh_20260930"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
NORM_FILE="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_s1/train_data_property_normalization.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_s1"
COUNTS_FILE="$OUT_ROOT/atom_counts_seed1701.npy"

CFG_CKPT="$FRESH_ROOT/cfg/None_3wupt02q/checkpoints/epoch=987-step=39520.ckpt"
MG_CKPT="$FRESH_ROOT/mg/None_mabj85bv/checkpoints/epoch=1938-step=77560.ckpt"
GOOD_CKPT="$FRESH_ROOT/base_good/None_e8607ywj/checkpoints/epoch=1349-step=54000.ckpt"
CLASSIFIER_CKPT="$FRESH_ROOT/cg_classifier/property_predictor_s1/checkpoints/epoch=153-val_loss=0.4944.ckpt"
BAD_CKPT="$FRESH_ROOT/ag_inferior/None_unv9xb53/checkpoints/epoch=291-step=11680.ckpt"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another guidance-comparison supervisor is active" >&2
  exit 73
fi
if [[ -e "$OUT_ROOT/generation_started.utc" ]]; then
  echo "Comparison was already started: $OUT_ROOT" >&2
  exit 72
fi
date -u +%FT%TZ > "$OUT_ROOT/generation_started.utc"
printf '%s\n' "$$" > "$OUT_ROOT/pids/supervisor.pid"

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

COMMON=(
  --n-mols 1000 --n-timesteps 100
  --property-name s1 --properties-for-sampling 3.640037775039673
  --number-of-atoms-file "$COUNTS_FILE"
  --normalization-file-path "$NORM_FILE"
  --dataset-name qme14s --analyze
)

run_job cfg "$PYTHON_BIN" -u sample_guided.py --guidance cfg \
  --model-checkpoint "$CFG_CKPT" "${COMMON[@]}" --max-batch-size 32 \
  --guide-w-x 1.5 --guide-w-a 1.5 --guide-w-c 1.5 --guide-w-e 1.5 \
  --dfm-type campbell_rate_matrix --guidance-format log \
  --where-to-apply-guide probabilities --properties-handle-method concatenate_sum \
  --output-file "$OUT_ROOT/cfg/samples.sdf" &
p_cfg=$!; printf '%s\n' "$p_cfg" > "$OUT_ROOT/pids/cfg.pid"

run_job mg "$PYTHON_BIN" -u sample_guided.py --guidance mg \
  --model-checkpoint "$MG_CKPT" "${COMMON[@]}" --max-batch-size 32 \
  --guide-weight 1.5 --properties-handle-method sum \
  --output-file "$OUT_ROOT/mg/samples.sdf" &
p_mg=$!; printf '%s\n' "$p_mg" > "$OUT_ROOT/pids/mg.pid"

run_job cg "$PYTHON_BIN" -u sample_guided.py --guidance cg \
  --flowmol-checkpoint "$GOOD_CKPT" --classifier-checkpoint "$CLASSIFIER_CKPT" \
  "${COMMON[@]}" --max-batch-size 8 \
  --guide-w-x 1 --guide-w-a 1 --guide-w-c 1 --guide-w-e 1 \
  --output-file "$OUT_ROOT/cg/samples.sdf" &
p_cg=$!; printf '%s\n' "$p_cg" > "$OUT_ROOT/pids/cg.pid"

run_job ag "$PYTHON_BIN" -u sample_guided.py --guidance ag \
  --good-model-checkpoint "$GOOD_CKPT" --bad-model-checkpoint "$BAD_CKPT" \
  "${COMMON[@]}" --max-batch-size 16 \
  --guide-w-x 1.5 --guide-w-a 1.5 --guide-w-c 1.5 --guide-w-e 1.5 \
  --dfm-type campbell_rate_matrix --guidance-format log \
  --where-to-apply-guide probabilities --properties-handle-method concatenate_sum \
  --output-file "$OUT_ROOT/ag/samples.sdf" &
p_ag=$!; printf '%s\n' "$p_ag" > "$OUT_ROOT/pids/ag.pid"

run_job conditional "$PYTHON_BIN" -u sample_guided.py --guidance conditional \
  --model-checkpoint "$GOOD_CKPT" "${COMMON[@]}" --max-batch-size 32 \
  --properties-handle-method concatenate_sum \
  --output-file "$OUT_ROOT/conditional/samples.sdf" &
p_cond=$!; printf '%s\n' "$p_cond" > "$OUT_ROOT/pids/conditional.pid"

declare -A rc
for entry in "cfg:$p_cfg" "mg:$p_mg" "cg:$p_cg" "ag:$p_ag" "conditional:$p_cond"; do
  name="${entry%%:*}"
  pid="${entry##*:}"
  rc[$name]=0
  wait "$pid" || rc[$name]=$?
done

for name in cfg mg cg ag conditional; do
  if [[ -s "$OUT_ROOT/$name/samples.sdf" ]]; then
    "$PYTHON_BIN" -u tools/analyze_generation.py \
      --sdf "$OUT_ROOT/$name/samples.sdf" --db-dir "$DB_DIR" \
      --out-dir "$OUT_ROOT/$name/analysis" --expected-count 1000 \
      > "$OUT_ROOT/logs/${name}_analysis.log" 2>&1 || rc[$name]=1
  fi
done

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  for name in cfg mg cg ag conditional; do
    printf '%s_exit=%s\n' "$name" "${rc[$name]}"
  done
} > "$OUT_ROOT/status/supervisor.status"

for name in cfg mg cg ag conditional; do
  (( rc[$name] == 0 )) || exit 1
done
