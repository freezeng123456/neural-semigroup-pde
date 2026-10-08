#!/bin/bash
# Allocation flags are fixed and recorded by the submission command.
set -e
source /etc/profile
set -euo pipefail

DUE_RUN_ROOT=${1:?Run root required}
DUE_RUN_MODE=${2:?preflight or full required}
DUE_RUN_NAME=${3:?Unique run name required}
case "$DUE_RUN_MODE" in preflight|full) ;; *) exit 64 ;; esac
test -n "${SLURM_JOB_ID:-}"
test -f "$DUE_RUN_ROOT/operations/setup_complete.json"
test ! -e "$DUE_RUN_ROOT/runs/$DUE_RUN_NAME"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DUE_RUN_ROOT/deps:$DUE_RUN_ROOT/due"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg
cd "$DUE_RUN_ROOT"
date -u +%Y-%m-%dT%H:%M:%SZ
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
exec /data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python -B -u \
  "$DUE_RUN_ROOT/operations/run_official_observed.py" \
  --source-root "$DUE_RUN_ROOT/due" \
  --data-dir "$DUE_RUN_ROOT/data" \
  --source-manifest "$DUE_RUN_ROOT/operations/official_source_manifest.json" \
  --run-root "$DUE_RUN_ROOT/runs/$DUE_RUN_NAME" \
  --mode "$DUE_RUN_MODE"
