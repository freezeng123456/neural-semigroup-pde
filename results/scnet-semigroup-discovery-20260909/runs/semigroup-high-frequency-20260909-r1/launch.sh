#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SEMIGROUP_EVALUATION_COMMIT=574ac25258494caafc16318bbd861b7c220e1f58
export PYTHONPATH=/work/home/zenghang/semigroup-clock-bootstrap-20260909-r1/neural-semigroup-pde-87f2f49f721c0098d6f104a5343894684c25beef/experiments
/work/home/zenghang/miniconda3/envs/pytorch/bin/python /work/home/zenghang/semigroup-high-frequency-20260909-r1/evaluation.py --root /work/home/zenghang/semigroup-high-frequency-20260909-r1/run --original /work/home/zenghang/semigroup-discovery-20260909-r1/run
