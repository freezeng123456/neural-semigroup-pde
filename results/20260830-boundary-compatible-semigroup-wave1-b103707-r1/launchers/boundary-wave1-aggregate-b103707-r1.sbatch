#!/bin/bash
#SBATCH --job-name=bnd-wave1-aggregate
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1/joblogs/aggregate-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1/joblogs/aggregate-%j.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2
src="$base/semigroup-b103707"
wave="$base/wave-b103707-r1"
root="$wave/aggregate-b103707-r1"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python

test ! -e "$root"
test "$(find "$wave/cells" -mindepth 2 -maxdepth 2 -name done | wc -l)" -eq 12
export PYTHONPATH="$src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
"$python_bin" -c "import sys,numpy; print('python',sys.executable); print('numpy',numpy.__version__)"

cd "$src"
"$python_bin" experiments/aggregate_boundary_semigroup.py \
  --wave-root "$wave" \
  --output-dir "$root" \
  | tee "$wave/aggregate.log"

test "$(cat "$root/done")" = passed
test -f "$root/receipt.json"
sha256sum "$root/aggregate.json" "$root/cell_summary.csv" "$root/seed_summary.csv" \
  | tee "$wave/aggregate-artifacts.sha256"
printf '__AGGREGATE_COMPLETE__\n'
