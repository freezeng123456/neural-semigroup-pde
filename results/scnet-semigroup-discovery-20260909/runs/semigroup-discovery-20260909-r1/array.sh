#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT='21bc488e479e47b74ae57e886ad9b7ceb6f2567c'
cd '/work/home/zenghang/semigroup-discovery-bootstrap-20260909-r1/neural-semigroup-pde-21bc488e479e47b74ae57e886ad9b7ceb6f2567c'
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_discovery.py run --root '/work/home/zenghang/semigroup-discovery-20260909-r1/run' --cell "$SLURM_ARRAY_TASK_ID" --updates 2000
