#!/bin/bash
set -e
source /etc/profile
set -euo pipefail
SG_BKS_ROOT=${1:?experiment root required}
SG_BKS_MODE=${2:?mode required}
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="$SG_BKS_ROOT"
cd "$SG_BKS_ROOT"
if [ "$SG_BKS_MODE" = full ]; then
    SG_BKS_WAVE=${3:?fixed wave required}
    exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u "$SG_BKS_ROOT/operations/launch.py" full --wave "$SG_BKS_WAVE"
fi
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u "$SG_BKS_ROOT/operations/launch.py" preflight
