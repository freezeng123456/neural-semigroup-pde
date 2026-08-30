#!/bin/bash
#SBATCH --job-name=bnd-w2-aggregate
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/wave-148f494-r1/joblogs/aggregate-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/wave-148f494-r1/joblogs/aggregate-%j.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1
src="$base/semigroup-148f494-r2"
wave="$base/wave-148f494-r1"
root="$wave/aggregate-148f494-r1"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python

test ! -e "$root"
test "$(find "$wave/cells" -mindepth 2 -maxdepth 2 -name done | wc -l)" -eq 54
export PYTHONPATH="$src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=148f4946fa489ec2b0bf485cf76bc62c7456b3e3
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=708cef2741c564d3bb530c677f2cdc8898734371712ee76c81e0ae92aa6e53f6

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
"$python_bin" -c "import sys,numpy,torch; print('python',sys.executable); print('numpy',numpy.__version__); print('torch',torch.__version__)"

cd "$src"
"$python_bin" experiments/aggregate_boundary_family_semigroup.py \
  --wave-root "$wave" \
  --output-dir "$root" \
  | tee "$wave/aggregate.log"

test "$(cat "$root/done")" = passed
test -f "$root/receipt.json"
sha256sum "$root/aggregate.json" "$root/cell_summary.csv" "$root/family_seed_summary.csv" \
  | tee "$wave/aggregate-artifacts.sha256"
printf '__AGGREGATE_COMPLETE__\n'
