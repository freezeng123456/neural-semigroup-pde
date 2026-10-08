#!/bin/bash
# Resources are supplied by a separately reviewed Slurm submission.
set -e
source /etc/profile
set -euo pipefail

STAGE_B_TRAINING_ROOT=${1:?Training root required}
STAGE_B_EVALUATION_ROOT=${2:?Separate evaluation root required}
STAGE_B_ARRAY_JOB=${3:?Completed training array job required}
STAGE_B_ATTEMPT_ID=${4:?New evidence attempt ID required}
test -n "${SLURM_JOB_ID:-}"
test -f "$STAGE_B_TRAINING_ROOT/PAYLOAD_MANIFEST.json"
test -f "$STAGE_B_EVALUATION_ROOT/evaluation-operations/EVALUATOR_FREEZE.json"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps:$STAGE_B_TRAINING_ROOT/due"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$STAGE_B_TRAINING_ROOT"
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u \
  "$STAGE_B_EVALUATION_ROOT/evaluation-operations/evaluate_matrix.py" \
  --training-root "$STAGE_B_TRAINING_ROOT" --evaluation-root "$STAGE_B_EVALUATION_ROOT" \
  --training-array-job "$STAGE_B_ARRAY_JOB" --attempt-id "$STAGE_B_ATTEMPT_ID" --mode evaluate
