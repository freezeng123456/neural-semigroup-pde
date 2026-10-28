#!/bin/bash
set -e
source /etc/profile
set -euo pipefail
SG_BKS_ROOT=${1:?experiment root required}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$SG_BKS_ROOT"
cd "$SG_BKS_ROOT"
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u "$SG_BKS_ROOT/operations/submit_recovery.py" --execute
