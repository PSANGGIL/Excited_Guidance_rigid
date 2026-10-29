#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
GEN_ROOT="$WORKSPACE_DIR/runs/guidance_compare_s1_3p640_20261001"
OUT_ROOT="$GEN_ROOT/xtb_stda_v5_all_methods_24x1_20261001"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
XTB_ENV="/home/jovyan/cpu-only-1-datavol-1/xtb_env"
PIPELINE="$BUNDLE_DIR/tools/joint_xtb_stda_pipeline_v5.py"
RENDER="$BUNDLE_DIR/tools/render_xtb_success_2d.py"
TARGET_S1="3.640037775039673"
TARGET_T1="3.1114676"
SEED="20261001"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/pids" "$OUT_ROOT/status"
exec 9>"$OUT_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another all-guidance XTB/sTDA supervisor is active" >&2
  exit 73
fi
if [[ -e "$OUT_ROOT/started.utc" ]]; then
  echo "XTB/sTDA comparison was already started: $OUT_ROOT" >&2
  exit 72
fi
date -u +%FT%TZ > "$OUT_ROOT/started.utc"
printf '%s\n' "$$" > "$OUT_ROOT/pids/supervisor.pid"

run_method() {
  local name="$1"
  local workers="$2"
  local input="$GEN_ROOT/$name/analysis/unique_novel_connected.sdf"
  local output="$OUT_ROOT/$name"
  local status="$OUT_ROOT/status/$name.status"
  local rc

  {
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'pid=%s\n' "$BASHPID"
    printf 'workers=%s\n' "$workers"
    printf 'input=%s\n' "$input"
  } > "$status"

  if [[ ! -s "$input" ]]; then
    printf 'finished_utc=%s\nexit_code=66\nerror=missing_input\n' \
      "$(date -u +%FT%TZ)" >> "$status"
    return 66
  fi

  "$PYTHON_BIN" -u "$PIPELINE" \
    --input-sdf "$input" --output-dir "$output" --xtb-env "$XTB_ENV" \
    --workers "$workers" --threads 1 \
    --target-s1 "$TARGET_S1" --target-t1 "$TARGET_T1" --seed "$SEED" \
    > "$OUT_ROOT/logs/${name}.log" 2>&1
  rc=$?

  if (( rc == 0 )) && [[ -s "$output/work/results.csv" ]]; then
    "$PYTHON_BIN" -u "$RENDER" \
      --results "$output/work/results.csv" \
      --out-dir "$output/rdkit_2d" --target-s1 "$TARGET_S1" \
      > "$OUT_ROOT/logs/${name}_render.log" 2>&1
    rc=$?
  fi

  {
    printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'exit_code=%s\n' "$rc"
  } >> "$status"
  return "$rc"
}

# Five concurrent supervisors share 24 single-core quantum-chemistry workers.
run_method cfg 5 &
p_cfg=$!; printf '%s\n' "$p_cfg" > "$OUT_ROOT/pids/cfg.pid"
run_method mg 5 &
p_mg=$!; printf '%s\n' "$p_mg" > "$OUT_ROOT/pids/mg.pid"
run_method cg 5 &
p_cg=$!; printf '%s\n' "$p_cg" > "$OUT_ROOT/pids/cg.pid"
run_method ag 5 &
p_ag=$!; printf '%s\n' "$p_ag" > "$OUT_ROOT/pids/ag.pid"
run_method conditional 4 &
p_conditional=$!; printf '%s\n' "$p_conditional" > "$OUT_ROOT/pids/conditional.pid"

declare -A rc
for entry in \
  "cfg:$p_cfg" "mg:$p_mg" "cg:$p_cg" "ag:$p_ag" \
  "conditional:$p_conditional"; do
  name="${entry%%:*}"
  pid="${entry##*:}"
  rc[$name]=0
  wait "$pid" || rc[$name]=$?
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
