#!/usr/bin/env bash
set -euo pipefail
RUN_ID=20260907-unknown-reaction-40616f3-t4b-full-r1
RUN_ROOT=/data/semigroup_runs/$RUN_ID
SOURCE=/data/worktrees/neural-semigroup-unknown-reaction-40616f3
CONSOLE=/data/semigroup_runs/$RUN_ID.console.log
exec 9>/data/neural_semigroup/unknown_reaction_gpu.lock
flock -n 9
test ! -e "$RUN_ROOT"
ACTIVE_GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$ACTIVE_GPU_PIDS" ]; then
  printf '%s\n' 'GPU busy; no experiment started.'
  exit 75
fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = 40616f3d59eb64f10b61a735a5966586c39146dc
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8
set +e
/data/miniconda/envs/syngen/bin/python -u "$SOURCE/experiments/run_unknown_reaction_matched.py" --output "$RUN_ROOT" >"$CONSOLE" 2>&1
EXIT_STATUS=$?
set -e
if [ -d "$RUN_ROOT" ]; then
  cp "$CONSOLE" "$RUN_ROOT/run.log"
  cp "$0" "$RUN_ROOT/launcher.sh"
  printf '%s\n' "$EXIT_STATUS" >"$RUN_ROOT/launcher.exit"
fi
exit "$EXIT_STATUS"
