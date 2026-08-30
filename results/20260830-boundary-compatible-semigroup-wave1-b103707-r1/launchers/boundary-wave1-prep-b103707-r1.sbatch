#!/bin/bash
#SBATCH --job-name=bnd-wave1-prep
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1/joblogs/prep-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1/joblogs/prep-%j.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2
src="$base/semigroup-b103707"
wave="$base/wave-b103707-r1"
root="$wave/data-prep"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python

test -d "$wave/joblogs"
test ! -e "$root"
export PYTHONPATH="$src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=b1037072e3177365f2989019ada346f6f651dcf2
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=ce0d4a20cad470f7ece2543ebd8115057aa1a80acdf7d5bf0ff57c68a7b5c5c4

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$python_bin" -c "import sys,torch,numpy; print('python',sys.executable); print('torch',torch.__version__); print('torch_cuda',torch.version.cuda); print('numpy',numpy.__version__); print('cuda_available',torch.cuda.is_available()); print('gpu_count',torch.cuda.device_count())"

cd "$src"
"$python_bin" experiments/run_boundary_semigroup.py prepare-data \
  --output-dir "$root" \
  | tee "$wave/data-prep.log"

test -f "$root/receipt.json"
test "$(cat "$root/done")" = passed
sha256sum "$root/boundary_wave1_cache.pt" | tee "$wave/data-cache.sha256"
printf '__DATA_PREP_COMPLETE__\n'
