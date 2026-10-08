#!/bin/bash
# Allocation is supplied by the reviewed submission command.
set -e
source /etc/profile
set -euo pipefail

STAGE_B_ROOT=${1:?Stage B root required}
STAGE_B_MODE=${2:?full or validate required}
test -n "${SLURM_JOB_ID:-}"
test -f "$STAGE_B_ROOT/PAYLOAD_MANIFEST.json"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps:$STAGE_B_ROOT/due"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$STAGE_B_ROOT"
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u \
  "$STAGE_B_ROOT/launch_task.py" --root "$STAGE_B_ROOT" --mode "$STAGE_B_MODE"
