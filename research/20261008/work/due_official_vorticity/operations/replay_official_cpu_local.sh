#!/bin/bash
# Local inference only. Existing Torch runtime and private dependency overlay.
set -euo pipefail
DUE_LOCAL_BASE=/Users/Apple_501/Documents/Codex/2026-10-03/cha
DUE_LOCAL_OPS="$DUE_LOCAL_BASE/work/due_official_vorticity/operations"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$DUE_LOCAL_OPS/local-import-deps"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg
exec /Users/Apple_501/Documents/Codex/2026-10-01/zha/work/apolarity-test-venv/bin/python -B -u \
  "$DUE_LOCAL_OPS/replay_official_cpu.py" \
  --source-root "$DUE_LOCAL_BASE/work/literature_baselines/due" \
  --source-manifest "$DUE_LOCAL_OPS/official_source_manifest.json" \
  --test-mat "$DUE_LOCAL_BASE/work/due_official_vorticity/data/VorticityOSG_test.mat" \
  "$@"
