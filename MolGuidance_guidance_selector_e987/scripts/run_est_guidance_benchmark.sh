#!/usr/bin/env bash
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
RUN_ROOT="$WORKSPACE_DIR/runs/molguidance_guidance_est_20261001"
PYTHON_BIN="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
CONFIG_DIR="$BUNDLE_DIR/configs/local/guidance_est"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$BUNDLE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=offline
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/pids" "$RUN_ROOT/status"
exec 9>"$RUN_ROOT/supervisor.lock"
if ! flock -n 9; then
  echo "Another EST-guidance supervisor already owns the lock" >&2
  exit 73
fi
if [[ -e "$RUN_ROOT/started.utc" ]]; then
  echo "EST run root was already started: $RUN_ROOT" >&2
  exit 72
fi

date -u +%FT%TZ > "$RUN_ROOT/started.utc"
printf '%s\n' "$$" > "$RUN_ROOT/pids/supervisor.pid"
printf '%s\n' "$BUNDLE_DIR" > "$RUN_ROOT/bundle_path.txt"
printf '%s\n' \
  'property=est' \
  'definition=s1_eV-t1_eV' \
  'train_mean_eV=0.5285704135894775' \
  'train_std_eV=0.5375194549560547' \
  > "$RUN_ROOT/condition.txt"

cd "$BUNDLE_DIR"

run_job() {
  local name="$1"
  shift
  local log="$RUN_ROOT/logs/${name}.log"
  local status="$RUN_ROOT/status/${name}.status"
  local rc
  {
    printf 'started_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'pid=%s\n' "$BASHPID"
    printf 'command='
    printf '%q ' "$@"
    printf '\n'
  } > "$status"
  : > "$log"
  "$@" >> "$log" 2>&1
  rc=$?
  {
    printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'exit_code=%s\n' "$rc"
  } >> "$status"
  return "$rc"
}

run_job cfg \
  "$PYTHON_BIN" -u train_with_guidance.py \
  --config "$CONFIG_DIR/hp1kg_est_cfg_fresh.yaml" --seed 1701 &
cfg_pid=$!
printf '%s\n' "$cfg_pid" > "$RUN_ROOT/pids/cfg.pid"

run_job mg \
  "$PYTHON_BIN" -u train_with_guidance.py \
  --config "$CONFIG_DIR/hp1kg_est_mg_fresh.yaml" --seed 1701 &
mg_pid=$!
printf '%s\n' "$mg_pid" > "$RUN_ROOT/pids/mg.pid"

run_job base_good \
  "$PYTHON_BIN" -u train.py \
  --config "$CONFIG_DIR/hp1kg_est_base_fresh.yaml" --seed 1701 &
base_pid=$!
printf '%s\n' "$base_pid" > "$RUN_ROOT/pids/base_good.pid"

(
  classifier_rc=0
  inferior_rc=0
  run_job cg_classifier \
    "$PYTHON_BIN" -u molguidance/classifier/train_classifier.py \
    --config "$CONFIG_DIR/hp1kg_est_cg_classifier_fresh.yaml" \
    --flowmol_config "$CONFIG_DIR/hp1kg_est_base_fresh.yaml" \
    --seed 1701 || classifier_rc=$?
  run_job ag_inferior \
    "$PYTHON_BIN" -u train.py \
    --config "$CONFIG_DIR/hp1kg_est_inferior_fresh.yaml" --seed 1701 || inferior_rc=$?
  if (( classifier_rc != 0 || inferior_rc != 0 )); then
    exit 1
  fi
) &
aux_pid=$!
printf '%s\n' "$aux_pid" > "$RUN_ROOT/pids/cg_then_ag.pid"

cfg_rc=0
mg_rc=0
base_rc=0
aux_rc=0
wait "$cfg_pid" || cfg_rc=$?
wait "$mg_pid" || mg_rc=$?
wait "$base_pid" || base_rc=$?
wait "$aux_pid" || aux_rc=$?

{
  printf 'finished_utc=%s\n' "$(date -u +%FT%TZ)"
  printf 'cfg_exit=%s\nmg_exit=%s\nbase_good_exit=%s\ncg_then_ag_exit=%s\n' \
    "$cfg_rc" "$mg_rc" "$base_rc" "$aux_rc"
} > "$RUN_ROOT/status/supervisor.status"

if (( cfg_rc != 0 || mg_rc != 0 || base_rc != 0 || aux_rc != 0 )); then
  exit 1
fi
