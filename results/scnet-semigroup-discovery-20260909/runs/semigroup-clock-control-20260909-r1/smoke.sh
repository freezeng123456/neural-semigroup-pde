#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SEMIGROUP_SOURCE_COMMIT=87f2f49f721c0098d6f104a5343894684c25beef
cd /work/home/zenghang/semigroup-clock-bootstrap-20260909-r1/neural-semigroup-pde-87f2f49f721c0098d6f104a5343894684c25beef
py=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
"$py" experiments/run_semigroup_clock_control.py prepare --root /work/home/zenghang/semigroup-clock-control-20260909-r1/run --source-cache /work/home/zenghang/semigroup-discovery-20260909-r1/run/cache.pt
"$py" experiments/run_semigroup_clock_control.py run --root /work/home/zenghang/semigroup-clock-control-20260909-r1/run --cell 0 --updates 2 --smoke
printf 'passed\n' > /work/home/zenghang/semigroup-clock-control-20260909-r1/smoke_passed
