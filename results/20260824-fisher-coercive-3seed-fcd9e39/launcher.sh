#!/usr/bin/env bash
set -u

RUN_ID="20260824-fisher-coercive-3seed-fcd9e39"
RUN_ROOT="/data/runs/${RUN_ID}"
CODE_ROOT="/data/worktrees/neural-semigroup-fcd9e39"
PYTHON="/data/miniconda/envs/syngen/bin/python"
COMMIT="fcd9e394ee87f229de57916271dcf42f0175bda4"
SEEDS=(42 123 2026)

mkdir -p "${RUN_ROOT}"
printf 'RUNNING\n' > "${RUN_ROOT}/status"

for SEED in "${SEEDS[@]}"; do
  CELL="${RUN_ROOT}/seed_${SEED}"
  mkdir -p "${CELL}/checkpoints" "${CELL}/results"
  rm -f "${CELL}/done" "${CELL}/failed"

  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
  GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)"
  CUDA_VISIBLE="${CUDA_VISIBLE_DEVICES:-0}"
  HOST="$(hostname)"
  CPU_COUNT="$(${PYTHON} -c 'import os; print(os.cpu_count())')"

  "${PYTHON}" - "${CELL}/config.json" "${RUN_ID}" "${SEED}" "${COMMIT}" "${HOST}" "${GPU_NAME}" "${GPU_UUID}" "${CUDA_VISIBLE}" "${CPU_COUNT}" <<'PY'
import json
import sys

path, run_id, seed, commit, host, gpu_name, gpu_uuid, cuda_visible, cpu_count = sys.argv[1:]
config = {
    "run_id": run_id,
    "repository": "freezeng123456/neural-semigroup-pde",
    "commit": commit,
    "entrypoint": "experiments/run_experiments.py",
    "python": "/data/miniconda/envs/syngen/bin/python",
    "host": host,
    "gpu_name": gpu_name,
    "gpu_uuid": gpu_uuid,
    "cuda_visible_devices": cuda_visible,
    "cpu_count_visible": int(cpu_count),
    "model_dtype": "float32",
    "seed": int(seed),
    "data_seed": 42,
    "deterministic": True,
    "architecture_only": True,
    "beta_v_floor": 0.1,
    "n_epochs": 100,
    "tau": 0.1,
    "reference_dt": 0.005,
    "ode_substep_counts": [15, 30, 60],
    "loss_weights": {
        "alpha_rollout": 0.0,
        "alpha_energy": 0.0,
        "latent_alpha_bound": 0.0,
        "baseline_alpha_bound": 0.1
    }
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

  printf 'RUNNING seed=%s started=%s\n' "${SEED}" "$(date -u +%FT%TZ)" > "${CELL}/status"
  cd "${CODE_ROOT}/experiments" || exit 1
  rm -f run_log.txt
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "${PYTHON}" run_experiments.py \
    --data-seed 42 \
    --seed "${SEED}" \
    --deterministic \
    --architecture-only \
    --beta-v-floor 0.1 \
    --no-resume \
    --checkpoint-dir "${CELL}/checkpoints" \
    --results-dir "${CELL}/results"
  RC=$?

  if [ -f run_log.txt ]; then
    cp run_log.txt "${CELL}/run.log"
  fi

  if [ "${RC}" -ne 0 ]; then
    printf 'FAILED seed=%s exit_code=%s ended=%s\n' "${SEED}" "${RC}" "$(date -u +%FT%TZ)" > "${CELL}/status"
    touch "${CELL}/failed"
    printf 'FAILED seed=%s\n' "${SEED}" > "${RUN_ROOT}/status"
    exit "${RC}"
  fi

  "${PYTHON}" - "${CELL}" <<'PY'
import json
import os
import sys
import torch

cell = sys.argv[1]
with open(os.path.join(cell, "config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
with open(os.path.join(cell, "results", "comparison.json"), encoding="utf-8") as handle:
    comparison = json.load(handle)
payload = torch.load(
    os.path.join(cell, "results", "evaluation_results.pt"),
    map_location="cpu",
    weights_only=False,
)
summary = {
    "seed": config["seed"],
    "data_seed": config["data_seed"],
    "commit": config["commit"],
    "latent": comparison["latent"],
    "baseline": comparison["baseline"],
    "detail_keys": sorted(payload.keys()),
}
with open(os.path.join(cell, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
PY

  printf 'COMPLETE seed=%s ended=%s\n' "${SEED}" "$(date -u +%FT%TZ)" > "${CELL}/status"
  touch "${CELL}/done"
done

printf 'COMPLETE seeds=3 ended=%s\n' "$(date -u +%FT%TZ)" > "${RUN_ROOT}/status"
touch "${RUN_ROOT}/done"
