#!/usr/bin/env bash
# Matched Burgers semigroup screen, CPU execution of the frozen three-seed
# matrix in docs/research/BURGERS_MATCHED_SEMIGROUP_SCREEN.md.
#
# Usage: burgers_screen_cpu.sh <work-root>
#
# The work root holds the immutable data cache and the per-seed cells.  Each
# seed runs as one independent task; inside a task the paired autonomous and
# query-time models train sequentially, so they never contend for the CPU
# allocation of the other model.  Single-threaded BLAS keeps the reductions
# bit-reproducible under torch.use_deterministic_algorithms.

set -euo pipefail

WORK_ROOT="${1:?work root required}"
EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../experiments" && pwd)"
RUNNER="${EXPERIMENTS_DIR}/run_burgers_semigroup_screen.py"
CACHE="${WORK_ROOT}/cache/burgers_cache.pt"
SEEDS=(31415 271828 161803)
DATA_SEED=20260902

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${WORK_ROOT}/cells"

if [[ ! -f "${CACHE}" ]]; then
  python3 "${RUNNER}" \
    --prepare-data-only \
    --output-dir "${WORK_ROOT}/data-prep" \
    --data-cache "${CACHE}" \
    --device cpu \
    --data-seed "${DATA_SEED}" \
    --n-train 1000 \
    --n-val 50 >"${WORK_ROOT}/data-prep.log" 2>&1
fi

sha256sum "${CACHE}" >"${WORK_ROOT}/inputs.sha256.before"

for seed in "${SEEDS[@]}"; do
  cell="${WORK_ROOT}/cells/s${seed}"
  mkdir -p "${cell}"
  # A cell that already carries its completion marker is never retrained, so
  # rerunning the launcher only re-checks input immutability and completion.
  if [[ -f "${cell}/done" ]]; then
    continue
  fi
  (
    if python3 "${RUNNER}" \
      --output-dir "${cell}" \
      --data-cache "${CACHE}" \
      --device cpu \
      --seed "${seed}" \
      --data-seed "${DATA_SEED}" \
      --n-train 1000 \
      --n-val 50 \
      --epochs 100 \
      --batch-size 64 \
      --eval-batch-size 10 \
      --validation-interval 10 \
      --ode-steps 12 \
      --hidden 32 >"${cell}/run.log" 2>&1; then
      printf 'EXIT_CODE=0\nDONE\n' >"${cell}/status"
      : >"${cell}/done"
    else
      printf 'EXIT_CODE=%d\nFAILED\n' "$?" >"${cell}/status"
      : >"${cell}/failed"
    fi
  ) &
done
wait

sha256sum "${CACHE}" >"${WORK_ROOT}/inputs.sha256.after"
cmp -s "${WORK_ROOT}/inputs.sha256.before" "${WORK_ROOT}/inputs.sha256.after" \
  || { echo "immutable data cache changed during execution" >&2; exit 1; }

# grep -L exits zero whenever it read its inputs, so gate on its output.
incomplete="$(grep -L DONE "${WORK_ROOT}"/cells/*/status || true)"
if [[ -n "${incomplete}" ]]; then
  echo "incomplete cells: ${incomplete}" >&2
  exit 1
fi
echo "all ${#SEEDS[@]} cells complete"
