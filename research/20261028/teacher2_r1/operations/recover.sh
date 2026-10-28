#!/bin/bash
set -e
source /etc/profile
set -euo pipefail
SG_T2_RECOVERY_ROOT=${1:?experiment root required}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps:/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1/due"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$SG_T2_RECOVERY_ROOT"
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u "$SG_T2_RECOVERY_ROOT/operations/submit_recovery.py" --execute
