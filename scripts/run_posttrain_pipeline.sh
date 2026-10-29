#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid
cd "$ROOT"
export PYTHONPATH=.deps:src
STATE=runs/posttrain_pipeline
mkdir -p "$STATE"
echo $$ > "$STATE/pipeline.pid"
date -u +%FT%TZ > "$STATE/started_at"

CHECKPOINT=runs/rigid_100k_chemical_finetune_1x_100ep_b200_final/best.pt
VALIDITY=runs/validity_scorer_stage3/best.pt
for seed in 101 202 303 404 505; do
  python -u sample_rigid.py \
    --checkpoint "$CHECKPOINT" \
    --targets 1.0 1.3 1.6 1.9 \
    --n-samples 250 --num-rings 13 --guidance 1.5 --seed "$seed" \
    --connection-attempts 32 --connection-noise 0.35 \
    --validity-scorer "$VALIDITY" --validity-weight 2 \
    --output "samples/rigid_1x_100ep_seed${seed}.jsonl" \
    > "$STATE/generate_seed${seed}.log" 2>&1
  python -u export_rigid_sdf.py \
    --input "samples/rigid_1x_100ep_seed${seed}.jsonl" \
    --output-dir "samples/rigid_1x_100ep_seed${seed}_sdf" \
    --per-target 250 \
    > "$STATE/export_seed${seed}.log" 2>&1
done

python - <<'PY'
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
out={'requested':5000,'seeds':{},'totals':Counter()}
for seed in (101,202,303,404,505):
 p=Path(f'samples/rigid_1x_100ep_seed{seed}.jsonl')
 rec=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 fast=sum(bool(x.get('connection_search',{}).get('selected_valid')) for x in rec)
 rows=list(csv.DictReader(open(f'samples/rigid_1x_100ep_seed{seed}_sdf/decode_manifest.csv')))
 sdf_ok=sum(x['status']=='ok' for x in rows)
 by=defaultdict(lambda:{'generated':0,'rdkit_valid':0})
 for x in rec:
  k=str(x['target_f_osc']);by[k]['generated']+=1;by[k]['rdkit_valid']+=bool(x.get('connection_search',{}).get('selected_valid'))
 out['seeds'][str(seed)]={'generated':len(rec),'rdkit_sanitize_valid':fast,'rdkit_sanitize_rate':fast/max(len(rec),1),'sdf_3d_success':sdf_ok,'sdf_3d_rate':sdf_ok/max(len(rec),1),'by_target':dict(by)}
 out['totals'].update(generated=len(rec),rdkit_sanitize_valid=fast,sdf_3d_success=sdf_ok)
out['totals']=dict(out['totals']);out['totals']['rdkit_sanitize_rate']=out['totals']['rdkit_sanitize_valid']/max(out['totals']['generated'],1);out['totals']['sdf_3d_rate']=out['totals']['sdf_3d_success']/max(out['totals']['generated'],1)
Path('runs/posttrain_pipeline/generation_analysis.json').write_text(json.dumps(out,indent=2))
PY
date -u +%FT%TZ > "$STATE/generation_complete"

python -u train_rigid.py \
  --config scripts/chemical_finetune_1x_10000ep_b200.yaml \
  --seed-model "$CHECKPOINT" \
  > "$STATE/train_10000ep.log" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$STATE/train_10000ep.pid"

XTB_ENV_PATH=${XBT_ENV_PATH:-${XTB_ENV:-}}
if [[ -z "$XTB_ENV_PATH" ]] && command -v xtb >/dev/null 2>&1; then
  XTB_ENV_PATH=$(dirname "$(dirname "$(command -v xtb)")")
fi
if [[ -n "$XTB_ENV_PATH" && -x "$XTB_ENV_PATH/bin/xtb" ]]; then
  for seed in 101 202 303 404 505; do
    python -u run_xtb_stda.py \
      --input-dir "samples/rigid_1x_100ep_seed${seed}_sdf" \
      --output-dir "samples/rigid_1x_100ep_seed${seed}_xtb" \
      --workers 3 --xtb-env "$XTB_ENV_PATH" \
      > "$STATE/xtb_seed${seed}.log" 2>&1
  done
  date -u +%FT%TZ > "$STATE/xtb_complete"
else
  printf 'xbt_env/xtb_env path was not visible; set XBT_ENV_PATH and relaunch xTB stage\n' > "$STATE/xtb_env_missing"
fi

wait "$TRAIN_PID"
date -u +%FT%TZ > "$STATE/training_10000ep_complete"
date -u +%FT%TZ > "$STATE/pipeline_complete"
