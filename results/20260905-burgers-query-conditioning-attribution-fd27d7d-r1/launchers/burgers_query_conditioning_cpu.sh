#!/usr/bin/env bash
# Checkpoint-only decomposition of the Burgers query-time composition defect,
# following docs/research/BURGERS_QUERY_CONDITIONING_ATTRIBUTION_PROTOCOL.md.
#
# Usage: burgers_query_conditioning_cpu.sh <work-root>
#
# The work root only holds the regenerated screen cache and the per-seed JSON
# outputs.  Nothing is trained and no checkpoint is written.

set -euo pipefail

WORK_ROOT="${1:?work root required}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXPERIMENTS_DIR="${REPO_ROOT}/experiments"
SCREEN_DIR="${REPO_ROOT}/results/20260905-burgers-matched-semigroup-screen-e479174-3seed-cpu-r1"
CACHE="${WORK_ROOT}/cache/burgers_cache.pt"
CACHE_SHA256="30c8443e37fa9b0d97e2b964699e8c0044f8f637232acf29abf915f19ac2de8d"
SEEDS=(31415 271828 161803)

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${WORK_ROOT}"

# The screen cache is not committed; regenerate it and require the digest that
# the screen recorded.  A mismatch aborts before any measurement.
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

for n_sample in 16 50; do
  for seed in "${SEEDS[@]}"; do
    python3 "${EXPERIMENTS_DIR}/evaluate_burgers_query_conditioning.py" \
      --autonomous-checkpoint "${SCREEN_DIR}/cells/s${seed}/checkpoints/a_autonomous_best.pt" \
      --query-checkpoint "${SCREEN_DIR}/cells/s${seed}/checkpoints/b_query_time_best.pt" \
      --data-cache "${CACHE}" \
      --expect-cache-sha256 "${CACHE_SHA256}" \
      --output "${WORK_ROOT}/s${seed}-n${n_sample}.json" \
      --seed "${seed}" \
      --n-sample "${n_sample}" \
      --device cpu >/dev/null
  done
  python3 "${EXPERIMENTS_DIR}/aggregate_burgers_query_conditioning.py" \
    "${WORK_ROOT}/s31415-n${n_sample}.json" \
    "${WORK_ROOT}/s271828-n${n_sample}.json" \
    "${WORK_ROOT}/s161803-n${n_sample}.json" \
    --output "${WORK_ROOT}/aggregate-n${n_sample}.json" >/dev/null
done

echo "attribution complete for ${#SEEDS[@]} seeds and both sample sets"
