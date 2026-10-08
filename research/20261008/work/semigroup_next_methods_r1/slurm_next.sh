#!/bin/bash
set -e
source /etc/profile
set -euo pipefail
SEMIGROUP_NEXT_ROOT=${1:?root required}
SEMIGROUP_NEXT_MODE=${2:?mode required}
test -n "${SLURM_JOB_ID:-}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps:$SEMIGROUP_NEXT_ROOT/due"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$SEMIGROUP_NEXT_ROOT"
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u \
  "$SEMIGROUP_NEXT_ROOT/launch_task.py" --root "$SEMIGROUP_NEXT_ROOT" --mode "$SEMIGROUP_NEXT_MODE"
