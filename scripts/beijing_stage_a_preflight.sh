#!/usr/bin/env bash
set -e
source /etc/profile
set -uo pipefail
stage_root=${1:?absolute stage root required}
stage_python=${2:?absolute Python required}
cd "$stage_root/source"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$stage_python" experiments/verify_stage_a_bundle.py --root "$stage_root"
"$stage_python" -m pytest -q experiments/tests/test_osg_baseline.py experiments/tests/test_burgers_stage_a.py experiments/tests/test_burgers_stage_a_parameter_audit.py experiments/tests/test_burgers_stage_a_aggregation.py
"$stage_python" experiments/preflight_burgers_stage_a.py --data-cache "$stage_root/cache.pt" --output-dir "$stage_root/preflight" --device cuda
