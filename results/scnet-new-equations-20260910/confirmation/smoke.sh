#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SEMIGROUP_SOURCE_COMMIT='78f5e27cf3d2f1f4c6cc26b2955529d19c155c1c'
cd '/work/home/zenghang/semigroup-confirmation-bootstrap-20260909-r1/neural-semigroup-pde-78f5e27cf3d2f1f4c6cc26b2955529d19c155c1c'
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_confirmation.py prepare --root '/work/home/zenghang/semigroup-confirmation-20260909-r1/run'
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_confirmation.py run --root '/work/home/zenghang/semigroup-confirmation-20260909-r1/run' --cell 0 --updates 2 --smoke
'/work/home/zenghang/miniconda3/envs/pytorch/bin/python' experiments/run_semigroup_confirmation.py run --root '/work/home/zenghang/semigroup-confirmation-20260909-r1/run' --cell 1 --updates 2 --smoke
printf 'passed\n' > '/work/home/zenghang/semigroup-confirmation-20260909-r1/smoke_passed'
