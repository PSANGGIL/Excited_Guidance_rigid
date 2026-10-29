#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid
PYTHON_BIN=/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python
OUT="$ROOT/runs/aromatic_bias_diagnostics_20261019"
export AROMATIC_DIAG_ROOT="$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT"
mkdir -p "$OUT/smoke_verified_v2" "$OUT/full_v2"
cp diagnostics/aromatic_bias_20261019.py "$OUT/full_v2/diagnostic_source.py"
"$PYTHON_BIN" -u "$OUT/full_v2/diagnostic_source.py" --out "$OUT/smoke_verified_v2" --smoke --batch-size 4 > "$OUT/smoke_verified_v2/supervisor.log" 2>&1
"$PYTHON_BIN" -u "$OUT/full_v2/diagnostic_source.py" --out "$OUT/full_v2" --batch-size 16 > "$OUT/full_v2/supervisor.log" 2>&1
