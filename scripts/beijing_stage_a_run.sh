#!/usr/bin/env bash
set -e
source /etc/profile
set -uo pipefail
stage_root=${1:?absolute stage root required}
stage_python=${2:?absolute Python required}
cd "$stage_root/source"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$stage_python" experiments/verify_stage_a_bundle.py --root "$stage_root" --require-preflight
for stage_seed in 31415 271828 161803; do
  "$stage_python" experiments/run_burgers_stage_a.py --data-cache "$stage_root/cache.pt" --output-dir "$stage_root/study/seed-$stage_seed" --seed "$stage_seed" --device cuda
done
"$stage_python" experiments/aggregate_burgers_stage_a.py "$stage_root/study/seed-31415" "$stage_root/study/seed-271828" "$stage_root/study/seed-161803" --output-dir "$stage_root/aggregate" --data-cache "$stage_root/cache.pt" --protocol-file "$stage_root/source/docs/research/BURGERS_OSG_STAGE_A_PROTOCOL.md" --source-dir "$stage_root/source/experiments"
"$stage_python" experiments/verify_stage_a_bundle.py --root "$stage_root" --require-preflight
