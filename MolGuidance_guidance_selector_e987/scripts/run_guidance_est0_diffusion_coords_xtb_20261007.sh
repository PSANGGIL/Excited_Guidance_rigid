#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
SOURCE_ROOT="$WORKSPACE_DIR/runs/guidance_est0_n1000_ema_edge_best_20261007"
OUT_ROOT="$WORKSPACE_DIR/runs/guidance_est0_n1000_ema_edge_best_diffcoords_20261007"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
XTB_ENV="/home/jovyan/cpu-only-1-datavol-1/xtb_env"
TARGET_EST="0.0"

export PYTHONPATH="$BUNDLE_DIR/tools:$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another diffusion-coordinate xTB supervisor is active" >&2
  exit 73
fi
if [[ -e "$OUT_ROOT/started.utc" ]]; then
  echo "Run root was already started: $OUT_ROOT" >&2
  exit 72
fi
printf '%s\n' "$$" > "$OUT_ROOT/pids/supervisor.pid"
date -u +%FT%TZ > "$OUT_ROOT/started.utc"
printf '%s\n' "$SOURCE_ROOT" > "$OUT_ROOT/source_generation_run.txt"
git -C "$WORKSPACE_DIR" rev-parse HEAD > "$OUT_ROOT/code_commit.txt"
git -C "$WORKSPACE_DIR" status --short > "$OUT_ROOT/code_status.txt"
cp --preserve=mode,timestamps \
  "$BUNDLE_DIR/tools/joint_xtb_stda_diffusion_coords.py" \
  "$OUT_ROOT/joint_xtb_stda_diffusion_coords.py"
cp --preserve=mode,timestamps "$BASH_SOURCE" "$OUT_ROOT/pipeline.sh"
sha256sum "$OUT_ROOT/joint_xtb_stda_diffusion_coords.py" \
  "$OUT_ROOT/pipeline.sh" > "$OUT_ROOT/code_sha256.txt"
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

declare -A XTB_PID XTB_RC POST_RC
for method in cfg mg cg ag; do
  mkdir -p "$OUT_ROOT/$method/analysis"
  cp --reflink=auto --preserve=mode,timestamps \
    "$SOURCE_ROOT/$method/analysis/summary.json" \
    "$OUT_ROOT/$method/analysis/summary.json"
  cp --reflink=auto --preserve=mode,timestamps \
    "$SOURCE_ROOT/$method/analysis/molecules.csv" \
    "$OUT_ROOT/$method/analysis/molecules.csv"
  cp --reflink=auto --preserve=mode,timestamps \
    "$SOURCE_ROOT/$method/analysis/unique_novel_connected.sdf" \
    "$OUT_ROOT/$method/analysis/unique_novel_connected.sdf"
  run_job "${method}_xtb" "$PYTHON_BIN" -u \
    tools/joint_xtb_stda_diffusion_coords.py \
    --input-sdf "$OUT_ROOT/$method/analysis/unique_novel_connected.sdf" \
    --output-dir "$OUT_ROOT/$method/xtb" --xtb-env "$XTB_ENV" \
    --workers 6 --threads 1 --target-s1 0.0 --target-t1 0.0 &
  XTB_PID[$method]=$!
  printf '%s\n' "${XTB_PID[$method]}" > "$OUT_ROOT/pids/${method}_xtb.pid"
done

for method in cfg mg cg ag; do
  XTB_RC[$method]=0
  wait "${XTB_PID[$method]}" || XTB_RC[$method]=$?
  POST_RC[$method]=1
  results="$OUT_ROOT/$method/xtb/work/results.csv"
  if [[ "${XTB_RC[$method]}" -eq 0 && -s "$results" ]]; then
    POST_RC[$method]=0
    run_job "${method}_render" "$PYTHON_BIN" -u \
      tools/render_xtb_success_2d.py \
      --results "$results" --out-dir "$OUT_ROOT/$method/xtb/rdkit_2d" \
      --target-s1 0.0 || POST_RC[$method]=1
    run_job "${method}_final_analysis" "$PYTHON_BIN" -u \
      tools/analyze_fixed_est_results.py \
      --generation-summary "$OUT_ROOT/$method/analysis/summary.json" \
      --molecules-csv "$OUT_ROOT/$method/analysis/molecules.csv" \
      --xtb-results "$results" --target-est "$TARGET_EST" \
      --out-dir "$OUT_ROOT/$method/est_analysis" || POST_RC[$method]=1
    run_job "${method}_est_window_render" "$PYTHON_BIN" -u \
      tools/render_est_window_2d.py --results "$results" \
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
  for method in cfg mg cg ag; do
    printf '%s_xtb_exit=%s\n' "$method" "${XTB_RC[$method]}"
    printf '%s_postprocess_exit=%s\n' "$method" "${POST_RC[$method]}"
  done
  printf 'comparison_exit=%s\n' "$COMPARE_RC"
} > "$OUT_ROOT/status/supervisor.status"
date -u +%FT%TZ > "$OUT_ROOT/finished.utc"

if [[ "$COMPARE_RC" -ne 0 ]]; then
  exit 1
fi
for method in cfg mg cg ag; do
  [[ "${XTB_RC[$method]}" -eq 0 ]] || exit 1
  [[ "${POST_RC[$method]}" -eq 0 ]] || exit 1
done
