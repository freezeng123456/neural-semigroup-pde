#!/usr/bin/env bash
set -uo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
/jizhicfs/yuyechen/miniconda3/envs/cl/bin/python -u /data/semigroup-h20-20260908/driver.py
result=$?
printf '%s\n' "$result" > /data/semigroup-h20-20260908/launcher.exit
exit "$result"
