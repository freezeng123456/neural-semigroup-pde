#!/usr/bin/env bash
set -euo pipefail

run_root=/data/runs/20260825-fisher-fair-variable-screen-seed42-c811cdb
cd "$run_root"
date --iso-8601=seconds > started_at
printf 'READY\n' > status
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader > gpu.txt

printf 'RUNNING\n' > status
if /data/miniconda/envs/syngen/bin/python experiments/run_fisher_fair.py \
    --regime variable \
    --models latent resnet fno \
    --output-dir outputs \
    --data-cache data/fisher_variable_screen.pt \
    --device cuda \
    --seed 42 \
    --data-seed 42 \
    --deterministic \
    --n-train 256 \
    --n-val 12 \
    --epochs 100 \
    --batch-size 64 \
    --eval-horizon 0.6 \
    --eval-taus 0.025,0.05,0.075,0.1,0.15,0.2 \
    --validation-interval 5 \
    --no-resume > run.log 2>&1; then
    printf 'COMPLETE\n' > status
else
    exit_code=$?
    printf 'FAILED exit=%s\n' "$exit_code" > status
    exit "$exit_code"
fi

date --iso-8601=seconds > finished_at
