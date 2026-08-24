#!/usr/bin/env bash
set -u

BENCH_ROOT="/data/runs/20260824-inference-optimization-benchmark-1fd45f0"
FORMAL_ROOT="/data/runs/20260824-fisher-coercive-3seed-fcd9e39"
OLD_ROOT="/data/worktrees/neural-semigroup-fcd9e39"
NEW_ROOT="/data/worktrees/neural-semigroup-1fd45f0"
PYTHON="/data/miniconda/envs/syngen/bin/python"
DATA="${FORMAL_ROOT}/seed_42/checkpoints/data.pt"
CHECKPOINT="${FORMAL_ROOT}/seed_42/checkpoints/latent_best.pt"

mkdir -p "${BENCH_ROOT}"
printf 'WAITING_FOR_FORMAL_RUN\n' > "${BENCH_ROOT}/status"
while [ ! -f "${FORMAL_ROOT}/done" ]; do
  sleep 60
done

printf 'WAITING_FOR_EXCLUSIVE_GPU\n' > "${BENCH_ROOT}/status"
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do
  sleep 30
done

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'RUNNING_BEFORE\n' > "${BENCH_ROOT}/status"
"${PYTHON}" "${NEW_ROOT}/experiments/benchmark_inference.py" \
  --implementation-root "${OLD_ROOT}" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${BENCH_ROOT}/before.json" \
  --device cuda \
  --n-samples 50 \
  --rollout-steps 20 \
  --repeats 1 \
  > "${BENCH_ROOT}/before.log" 2>&1
RC=$?
if [ "${RC}" -ne 0 ]; then
  printf 'FAILED_BEFORE exit_code=%s\n' "${RC}" > "${BENCH_ROOT}/status"
  touch "${BENCH_ROOT}/failed"
  exit "${RC}"
fi

printf 'RUNNING_AFTER\n' > "${BENCH_ROOT}/status"
"${PYTHON}" "${NEW_ROOT}/experiments/benchmark_inference.py" \
  --implementation-root "${NEW_ROOT}" \
  --data "${DATA}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${BENCH_ROOT}/after.json" \
  --device cuda \
  --n-samples 50 \
  --rollout-steps 20 \
  --repeats 1 \
  > "${BENCH_ROOT}/after.log" 2>&1
RC=$?
if [ "${RC}" -ne 0 ]; then
  printf 'FAILED_AFTER exit_code=%s\n' "${RC}" > "${BENCH_ROOT}/status"
  touch "${BENCH_ROOT}/failed"
  exit "${RC}"
fi

"${PYTHON}" - "${BENCH_ROOT}" <<'PY'
import json
import math
import os
import sys

root = sys.argv[1]
with open(os.path.join(root, "before.json"), encoding="utf-8") as handle:
    before = json.load(handle)
with open(os.path.join(root, "after.json"), encoding="utf-8") as handle:
    after = json.load(handle)

validation_before = before["validation_seconds"][0]
validation_after = after["validation_seconds"][0]
formal_before = before["formal_evaluation_seconds"][0]
formal_after = after["formal_evaluation_seconds"][0]

checks = {}
for section, keys in {
    "validation_metrics": (
        "rollout_mse",
        "bound_viol",
        "energy_mono_frac",
    ),
    "formal_metrics": (
        "rollout_mse_mean",
        "rollout_mse_std",
        "bound_viol_mean",
        "numerical_semigroup_defect_mse_mean",
        "numerical_semigroup_defect_abs_l2_mean",
        "numerical_semigroup_defect_rel_l2_mean",
        "learned_energy_mono_frac_mean",
        "physical_energy_mono_frac_mean",
        "latent_l2_norm_max",
        "latent_abs_max",
        "latent_vector_field_l2_norm_max",
    ),
}.items():
    for key in keys:
        old = float(before[section][key])
        new = float(after[section][key])
        abs_diff = abs(new - old)
        rel_diff = abs_diff / max(abs(old), 1e-30)
        checks[f"{section}.{key}"] = {
            "before": old,
            "after": new,
            "absolute_difference": abs_diff,
            "relative_difference": rel_diff,
            "within_tolerance": math.isclose(old, new, rel_tol=1e-5, abs_tol=1e-10),
        }

summary = {
    "status": "complete",
    "before_commit": "fcd9e394ee87f229de57916271dcf42f0175bda4",
    "after_commit": "1fd45f0b732852a7b8d720a3898fed6146815621",
    "n_samples": 50,
    "rollout_steps": 20,
    "validation_seconds_before": validation_before,
    "validation_seconds_after": validation_after,
    "validation_speedup": validation_before / validation_after,
    "formal_evaluation_seconds_before": formal_before,
    "formal_evaluation_seconds_after": formal_after,
    "formal_evaluation_speedup": formal_before / formal_after,
    "all_metric_checks_within_tolerance": all(
        check["within_tolerance"] for check in checks.values()
    ),
    "metric_checks": checks,
}
with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY

printf 'COMPLETE\n' > "${BENCH_ROOT}/status"
touch "${BENCH_ROOT}/done"
