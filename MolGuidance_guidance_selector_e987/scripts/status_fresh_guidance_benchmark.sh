#!/usr/bin/env bash
set -u

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_DIR="$(cd "$BUNDLE_DIR/.." && pwd)"
RUN_ROOT="$WORKSPACE_DIR/runs/molguidance_guidance_fresh_20260930"

echo "run_root=$RUN_ROOT"
echo
echo "[processes]"
for pid_file in "$RUN_ROOT"/pids/*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(<"$pid_file")"
  printf '%-24s ' "$(basename "$pid_file")=$pid"
  ps -p "$pid" -o pid=,ppid=,sid=,stat=,etime=,%cpu=,%mem=,args= 2>/dev/null || echo "not running"
done

echo
echo "[gpu]"
timeout 5s nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
timeout 5s nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo
echo "[status]"
for status_file in "$RUN_ROOT"/status/*.status; do
  [[ -e "$status_file" ]] || continue
  echo "--- $(basename "$status_file")"
  sed -n '1,20p' "$status_file"
done

echo
echo "[latest log lines]"
for log_file in "$RUN_ROOT"/logs/*.log; do
  [[ -e "$log_file" ]] || continue
  echo "--- $(basename "$log_file")"
  tail -n 8 "$log_file"
done

echo
echo "[checkpoints]"
find "$RUN_ROOT" -type f -name '*.ckpt' -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' 2>/dev/null | sort | tail -20
