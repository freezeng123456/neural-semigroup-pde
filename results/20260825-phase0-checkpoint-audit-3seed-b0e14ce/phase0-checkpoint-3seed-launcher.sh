#!/usr/bin/env bash
set -euo pipefail

run_root=/data/runs/20260825-phase0-checkpoint-audit-3seed-b0e14ce
source_root=/data/runs/20260824-fisher-coercive-3seed-fcd9e39
python_bin=/data/miniconda/envs/syngen/bin/python
cd "$run_root"
date --iso-8601=seconds > started_at
printf 'RUNNING\n' > status
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader > gpu.txt

for seed in 42 123 2026; do
    mkdir -p "results/seed_${seed}"
    "$python_bin" experiments/phase0_audit.py \
        --data "$source_root/seed_${seed}/checkpoints/data.pt" \
        --checkpoint "$source_root/seed_${seed}/checkpoints/latent_best.pt" \
        --physical-energy fisher-kpp \
        --output "results/seed_${seed}/phase0_audit.json" \
        --device cuda \
        --n-samples 50 \
        --sweep-samples 1 \
        --rollout-steps 20 \
        --max-pairs 20 \
        --ode-steps 5,10,15,30,60,120 \
        --equal-work-base-ode-steps 30 \
        > "results/seed_${seed}/run.log" 2>&1
    printf 'COMPLETE\n' > "results/seed_${seed}/status"
done

printf 'COMPLETE\n' > status
date --iso-8601=seconds > finished_at
