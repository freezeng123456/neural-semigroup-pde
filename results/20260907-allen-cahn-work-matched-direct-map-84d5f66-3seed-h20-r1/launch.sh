#!/bin/bash
set -euo pipefail
PYTHON=/jizhicfs/yuyechen/miniconda3/envs/cl/bin/python
REPO=/root/work/neural-semigroup-pde
ROOT=/jizhicfs/yuyechen/neural_semigroup/20260907-allen-cahn-work-matched-direct-map-h20-r1
cd "$REPO"
for seed in 42 137 2718; do
  echo "===== seed ${seed} start $(date -Is) ====="
  $PYTHON experiments/run_allen_cahn_work_matched_direct_map.py \
    --output-dir "$ROOT/s${seed}" \
    --data-cache "$ROOT/cache-s${seed}.pt" \
    --device cuda \
    --seed "$seed" \
    --data-seed "$seed" \
    --N 64 --n-train 1000 --n-val 50 --n-test 500 \
    --epochs 100 --batch-size 64 --validation-interval 5
  echo "===== seed ${seed} done $(date -Is) ====="
done
$PYTHON experiments/aggregate_allen_cahn_work_matched_direct_map.py \
  --root "$ROOT" \
  --output "$ROOT/aggregate.json"
echo "===== aggregate done $(date -Is) ====="
