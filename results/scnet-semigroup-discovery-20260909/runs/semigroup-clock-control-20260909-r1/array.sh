#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SEMIGROUP_SOURCE_COMMIT=87f2f49f721c0098d6f104a5343894684c25beef
cd /work/home/zenghang/semigroup-clock-bootstrap-20260909-r1/neural-semigroup-pde-87f2f49f721c0098d6f104a5343894684c25beef
/work/home/zenghang/miniconda3/envs/pytorch/bin/python experiments/run_semigroup_clock_control.py run --root /work/home/zenghang/semigroup-clock-control-20260909-r1/run --cell "$SLURM_ARRAY_TASK_ID" --updates 2000
