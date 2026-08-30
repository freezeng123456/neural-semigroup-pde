#!/bin/bash
#SBATCH --job-name=bnd-w2-prep
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/wave-148f494-r1/joblogs/prep-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/wave-148f494-r1/joblogs/prep-%j.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1
src="$base/semigroup-148f494-r2"
wave="$base/wave-148f494-r1"
root="$wave/data-prep"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python

test -d "$wave/joblogs"
test ! -e "$root"
mkdir -p "$root"
export PYTHONPATH="$src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=148f4946fa489ec2b0bf485cf76bc62c7456b3e3
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=708cef2741c564d3bb530c677f2cdc8898734371712ee76c81e0ae92aa6e53f6

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$python_bin" -c "import sys,torch,numpy; print('python',sys.executable); print('torch',torch.__version__); print('torch_cuda',torch.version.cuda); print('numpy',numpy.__version__); print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0)); print('gpu_count',torch.cuda.device_count())"

cd "$src"
families=(inhomogeneous_dirichlet homogeneous_neumann robin)
for family in "${families[@]}"; do
  "$python_bin" experiments/run_boundary_family_semigroup.py prepare-data \
    --output-dir "$root/$family" \
    --boundary-family "$family" \
    | tee "$wave/data-prep-$family.log"
done

sha256sum "$root"/*/boundary_family_wave2_*_cache.pt | tee "$wave/data-cache.sha256"
test "$(find "$root" -mindepth 2 -maxdepth 2 -name receipt.json | wc -l)" -eq 3
printf 'passed\n' > "$wave/data-prep.done"
printf '__DATA_PREP_COMPLETE__\n'
