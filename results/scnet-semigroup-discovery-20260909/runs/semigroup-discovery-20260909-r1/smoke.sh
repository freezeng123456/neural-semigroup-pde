#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT='21bc488e479e47b74ae57e886ad9b7ceb6f2567c'
cd '/work/home/zenghang/semigroup-discovery-bootstrap-20260909-r1/neural-semigroup-pde-21bc488e479e47b74ae57e886ad9b7ceb6f2567c'
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/tests/test_semigroup_discovery.py
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_discovery.py prepare --root '/work/home/zenghang/semigroup-discovery-20260909-r1/run'
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_discovery.py run --root '/work/home/zenghang/semigroup-discovery-20260909-r1/run' --cell 0 --updates 2 --smoke
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_discovery.py run --root '/work/home/zenghang/semigroup-discovery-20260909-r1/run' --cell 9 --updates 2 --smoke
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_discovery.py run --root '/work/home/zenghang/semigroup-discovery-20260909-r1/run' --cell 10 --updates 2 --smoke
printf 'passed\n' > '/work/home/zenghang/semigroup-discovery-20260909-r1/smoke_passed'
