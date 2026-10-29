#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
OUT_ROOT="$WORKSPACE_DIR/runs/guidance_est0_n1000_ema_edge_best_20261007"
PREVIOUS_ROOT="$WORKSPACE_DIR/runs/guidance_fixed_est0_n1000_finalbest_20261005"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
XTB_ENV="/home/jovyan/cpu-only-1-datavol-1/xtb_env"
RAW_TRAIN="$WORKSPACE_DIR/joint_kekule_diffusion/data_hp1kg_nevpt2/train.pt"
DB_DIR="$WORKSPACE_DIR/molguidance_hp1kg_baseline/data_est"
NORM_FILE="$DB_DIR/train_data_property_normalization.pt"
SNAPSHOTS="$OUT_ROOT/checkpoint_snapshots"
TARGETS="$OUT_ROOT/targets_est_fixed.npy"
COUNTS="$OUT_ROOT/atom_counts_train_bootstrap.npy"
MANIFEST="$OUT_ROOT/condition_manifest.csv"
TARGET_EST="0.0"

CFG_SOURCE="$PREVIOUS_ROOT/checkpoint_snapshots/cfg_epoch987.ckpt"
MG_SOURCE="$WORKSPACE_DIR/runs/molguidance_est_mg_ema_edge_fix_20261006/None_pbzlg5a9/checkpoints/epoch=1254-step=50200.ckpt"
GOOD_SOURCE="$PREVIOUS_ROOT/checkpoint_snapshots/base_good_epoch1349.ckpt"
CLASSIFIER_SOURCE="$PREVIOUS_ROOT/checkpoint_snapshots/cg_classifier_epoch346.ckpt"
BAD_SOURCE="$PREVIOUS_ROOT/checkpoint_snapshots/ag_inferior_epoch291.ckpt"

CFG_CKPT="$SNAPSHOTS/cfg_epoch987.ckpt"
MG_CKPT="$SNAPSHOTS/mg_ema_edge_epoch1254.ckpt"
GOOD_CKPT="$SNAPSHOTS/base_good_epoch1349.ckpt"
CLASSIFIER_CKPT="$SNAPSHOTS/cg_classifier_epoch346.ckpt"
BAD_CKPT="$SNAPSHOTS/ag_inferior_epoch291.ckpt"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status" "$SNAPSHOTS"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another guidance comparison supervisor is active" >&2
  exit 73
fi
if [[ -e "$OUT_ROOT/started.utc" ]]; then
  echo "Run root was already started: $OUT_ROOT" >&2
  exit 72
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

snapshot() {
  local source="$1" destination="$2"
  cp --reflink=auto --preserve=mode,timestamps "$source" "$destination"
}

snapshot "$CFG_SOURCE" "$CFG_CKPT"
snapshot "$MG_SOURCE" "$MG_CKPT"
snapshot "$GOOD_SOURCE" "$GOOD_CKPT"
snapshot "$CLASSIFIER_SOURCE" "$CLASSIFIER_CKPT"
snapshot "$BAD_SOURCE" "$BAD_CKPT"
sha256sum "$SNAPSHOTS"/*.ckpt > "$SNAPSHOTS/checkpoint_sha256.txt"
git -C "$WORKSPACE_DIR" rev-parse HEAD > "$OUT_ROOT/code_commit.txt"
git -C "$WORKSPACE_DIR" status --short > "$OUT_ROOT/code_status.txt"
git -C "$WORKSPACE_DIR" diff -- \
  MolGuidance_guidance_selector_e987/tools/analyze_generation.py \
  MolGuidance_guidance_selector_e987/tools/render_xtb_success_2d.py \
  > "$OUT_ROOT/code_changes.patch"
cp --preserve=mode,timestamps "$BASH_SOURCE" "$OUT_ROOT/pipeline.sh"
sha256sum "$OUT_ROOT/pipeline.sh" > "$OUT_ROOT/pipeline.sha256"

run_job build_manifest "$PYTHON_BIN" -u tools/build_fixed_est_manifest.py \
  --train-data "$RAW_TRAIN" --out-dir "$OUT_ROOT" --n-mols 1000 \
  --target-est "$TARGET_EST" --seed 20261005
MANIFEST_RC=$?

COMMON=(
  --n-mols 1000 --n-timesteps 100 --property-name est
  --multiple-values-file "$TARGETS" --number-of-atoms-file "$COUNTS"
  --normalization-file-path "$NORM_FILE" --dataset-name qme14s
)
declare -A GEN_RC XTB_PID XTB_RC POST_RC

start_xtb() {
  local method="$1"
  run_job "${method}_xtb" "$PYTHON_BIN" -u tools/joint_xtb_stda_pipeline_v5.py \
    --input-sdf "$OUT_ROOT/$method/analysis/unique_novel_connected.sdf" \
    --output-dir "$OUT_ROOT/$method/xtb" --xtb-env "$XTB_ENV" \
    --workers 6 --threads 1 --target-s1 0.0 --target-t1 0.0 \
    --seed 20261007 &
  XTB_PID[$method]=$!
  printf '%s\n' "${XTB_PID[$method]}" > "$OUT_ROOT/pids/${method}_xtb.pid"
}

if [[ "$MANIFEST_RC" -eq 0 ]]; then
  for method in cfg mg cg ag; do
    mkdir -p "$OUT_ROOT/$method"
    command=("$PYTHON_BIN" -u sample_guided.py --guidance "$method")
    case "$method" in
      cfg)
        command+=(--model-checkpoint "$CFG_CKPT" --max-batch-size 32
          --guide-w-x 1.5 --guide-w-a 1.5 --guide-w-c 1.5 --guide-w-e 1.5
          --dfm-type campbell_rate_matrix --guidance-format log
          --where-to-apply-guide probabilities
          --properties-handle-method concatenate_sum)
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
          --where-to-apply-guide probabilities
          --properties-handle-method concatenate_sum)
        ;;
    esac
    command+=("${COMMON[@]}" --output-file "$OUT_ROOT/$method/samples.sdf")
    GEN_RC[$method]=0
    run_job "${method}_generation" "${command[@]}" || GEN_RC[$method]=$?
    if [[ "${GEN_RC[$method]}" -eq 0 && -s "$OUT_ROOT/$method/samples.sdf" ]]; then
      run_job "${method}_analysis" "$PYTHON_BIN" -u tools/analyze_generation.py \
        --sdf "$OUT_ROOT/$method/samples.sdf" --db-dir "$DB_DIR" \
        --out-dir "$OUT_ROOT/$method/analysis" --expected-count 1000 \
        --condition-manifest "$MANIFEST" || GEN_RC[$method]=1
    fi
    if [[ "${GEN_RC[$method]}" -eq 0 && -s "$OUT_ROOT/$method/analysis/unique_novel_connected.sdf" ]]; then
      start_xtb "$method"
    fi
  done
fi

for method in cfg mg cg ag; do
  XTB_RC[$method]=1
  POST_RC[$method]=1
  if [[ -n "${XTB_PID[$method]:-}" ]]; then
    XTB_RC[$method]=0
    wait "${XTB_PID[$method]}" || XTB_RC[$method]=$?
  fi
  results="$OUT_ROOT/$method/xtb/work/results.csv"
  if [[ "${XTB_RC[$method]}" -eq 0 && -s "$results" ]]; then
    POST_RC[$method]=0
    run_job "${method}_render" "$PYTHON_BIN" -u tools/render_xtb_success_2d.py \
      --results "$results" --out-dir "$OUT_ROOT/$method/xtb/rdkit_2d" \
      --target-s1 0.0 || POST_RC[$method]=1
    run_job "${method}_final_analysis" "$PYTHON_BIN" -u tools/analyze_fixed_est_results.py \
      --generation-summary "$OUT_ROOT/$method/analysis/summary.json" \
      --molecules-csv "$OUT_ROOT/$method/analysis/molecules.csv" \
      --xtb-results "$results" --target-est "$TARGET_EST" \
      --out-dir "$OUT_ROOT/$method/est_analysis" || POST_RC[$method]=1
    run_job "${method}_est_window_render" "$PYTHON_BIN" -u tools/render_est_window_2d.py \
      --results "$results" \
      --out-dir "$OUT_ROOT/$method/est_analysis/est_0_to_0p5_png" \
      --min-est 0.0 --max-est 0.5 --page-size 40 || POST_RC[$method]=1
  fi
done

COMPARE_RC=1
if [[ "${POST_RC[cfg]}" -eq 0 && "${POST_RC[mg]}" -eq 0 && \
      "${POST_RC[cg]}" -eq 0 && "${POST_RC[ag]}" -eq 0 ]]; then
  run_job compare "$PYTHON_BIN" -u tools/compare_fixed_est_methods.py \
    --root "$OUT_ROOT" --methods cfg mg cg ag \
    --out-dir "$OUT_ROOT/comparison"
  COMPARE_RC=$?
fi

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'manifest_exit=%s\n' "$MANIFEST_RC"
  for method in cfg mg cg ag; do
    printf '%s_generation_exit=%s\n' "$method" "${GEN_RC[$method]:-1}"
    printf '%s_xtb_exit=%s\n' "$method" "${XTB_RC[$method]}"
    printf '%s_postprocess_exit=%s\n' "$method" "${POST_RC[$method]}"
  done
  printf 'comparison_exit=%s\n' "$COMPARE_RC"
} > "$OUT_ROOT/status/supervisor.status"
date -u +%FT%TZ > "$OUT_ROOT/finished.utc"

if [[ "$MANIFEST_RC" -ne 0 || "$COMPARE_RC" -ne 0 ]]; then
  exit 1
fi
for method in cfg mg cg ag; do
  [[ "${GEN_RC[$method]:-1}" -eq 0 ]] || exit 1
  [[ "${XTB_RC[$method]}" -eq 0 ]] || exit 1
  [[ "${POST_RC[$method]}" -eq 0 ]] || exit 1
done
