#!/usr/bin/env bash
# Direct time-conditioned map control, per
# docs/research/BURGERS_DIRECT_MAP_CONTROL_PROTOCOL.md.
#
# Usage: burgers_direct_map_control_cpu.sh <work-root>
#
# Only model C is trained. Models A and B are loaded frozen from the committed
# minimal-generator lane and are never written to.

set -euo pipefail

WORK_ROOT="${1:?work root required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"
MINIMAL_ROOT="${REPO_ROOT}/results/20260905-burgers-minimal-generator-ab-3seed-cpu-r1"
CACHE="${WORK_ROOT}/cache/burgers_cache.pt"
CACHE_SHA256="30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d"
SEEDS=(31415 271828 161803)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${WORK_ROOT}/cells"

if [[ ! -f "${CACHE}" ]]; then
  python3 "${EXPERIMENTS_DIR}/run_burgers_semigroup_screen.py" \
    --prepare-data-only \
    --output-dir "${WORK_ROOT}/data-prep" \
    --data-cache "${CACHE}" \
    --device cpu \
    --data-seed 20260902 \
    --n-train 1000 \
    --n-val 50 >"${WORK_ROOT}/data-prep.log" 2>&1
fi
echo "${CACHE_SHA256}  ${CACHE}" | sha256sum --check --status \
  || { echo "regenerated cache does not match the frozen screen digest" >&2; exit 1; }

sha256sum "${CACHE}" >"${WORK_ROOT}/inputs.sha256.before"
find "${MINIMAL_ROOT}/cells" -name '*_best.pt' -print0 \
  | sort -z | xargs -0 sha256sum >"${WORK_ROOT}/frozen-checkpoints.sha256.before"

for seed in "${SEEDS[@]}"; do
  cell="${WORK_ROOT}/cells/s${seed}"
  mkdir -p "${cell}"
  if [[ -f "${cell}/done" ]]; then
    continue
  fi
  (
    if python3 "${EXPERIMENTS_DIR}/run_burgers_direct_map_control.py" \
      --output-dir "${cell}" \
      --data-cache "${CACHE}" \
      --expect-cache-sha256 "${CACHE_SHA256}" \
      --minimal-ab-root "${MINIMAL_ROOT}" \
      --seed "${seed}" \
      --device cpu >"${cell}/run.log" 2>&1; then
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
find "${MINIMAL_ROOT}/cells" -name '*_best.pt' -print0 \
  | sort -z | xargs -0 sha256sum >"${WORK_ROOT}/frozen-checkpoints.sha256.after"
cmp -s "${WORK_ROOT}/inputs.sha256.before" "${WORK_ROOT}/inputs.sha256.after" \
  || { echo "immutable data cache changed during execution" >&2; exit 1; }
cmp -s "${WORK_ROOT}/frozen-checkpoints.sha256.before" \
       "${WORK_ROOT}/frozen-checkpoints.sha256.after" \
  || { echo "frozen flow checkpoints changed during execution" >&2; exit 1; }

incomplete="$(grep -L DONE "${WORK_ROOT}"/cells/*/status || true)"
if [[ -n "${incomplete}" ]]; then
  echo "incomplete cells: ${incomplete}" >&2
  exit 1
fi

python3 "${EXPERIMENTS_DIR}/aggregate_burgers_direct_map_control.py" \
  "${WORK_ROOT}/cells/s31415/summary.json" \
  "${WORK_ROOT}/cells/s271828/summary.json" \
  "${WORK_ROOT}/cells/s161803/summary.json" \
  --output "${WORK_ROOT}/aggregate.json" >/dev/null

echo "all ${#SEEDS[@]} control cells complete"
