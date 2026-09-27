#!/usr/bin/env bash
set -u

root="/home/jovyan/cpu-only-1-datavol-1/psg/generation/Excited_Guidance_rigid"
out="$root/runs/qcdge_a10_fpa_template_free_scalar_epoch0030_mean_1000"
python_bin="/home/jovyan/cpu-only-1-datavol-1/mg_b200/bin/python"
checkpoint="$out/best_epoch0030.pt"

mkdir -p "$out"
for index in $(seq -w 0 9); do
    output="$out/batch_${index}.pt"
    if [ -s "$output" ]; then
        continue
    fi
    echo "START batch=$index $(date -u +%FT%TZ)"
    "$python_bin" "$root/sample_fpa_template_free.py" \
        --checkpoint "$checkpoint" \
        --num-samples 100 \
        --s1-energy 5.7580084800720215 \
        --t1-energy 4.1311564445495605 \
        --guidance 1.0 \
        --progress-every 100 \
        --output "$output"
    status=$?
    echo "DONE batch=$index status=$status $(date -u +%FT%TZ)"
    if [ "$status" -ne 0 ]; then
        exit "$status"
    fi
done
