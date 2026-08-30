#!/bin/bash
#SBATCH --job-name=bnd-wave1-smoke-r2
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/joblogs/smoke-r2-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/joblogs/smoke-r2-%j.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2
src="$base/semigroup-b103707"
root="$base/smoke-b103707-r2"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
pydeps=/work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-tests-3c46763-r2/pydeps

test ! -e "$root"
test -d "$pydeps/pytest"
mkdir -p "$root/cells"
export PYTHONPATH="$src:$pydeps"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=b1037072e3177365f2989019ada346f6f651dcf2
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=ce0d4a20cad470f7ece2543ebd8115057aa1a80acdf7d5bf0ff57c68a7b5c5c4

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  "$python_bin" -c "import sys,torch,numpy,pytest; print('python',sys.executable); print('torch',torch.__version__); print('torch_cuda',torch.version.cuda); print('numpy',numpy.__version__); print('pytest',pytest.__version__); print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0)); print('gpu_count',torch.cuda.device_count())"
} | tee "$root/gpu_probe.txt"

cd "$src"
"$python_bin" -m pytest -q experiments/tests/test_boundary_semigroup.py \
  | tee "$root/pytest.log"

"$python_bin" experiments/run_boundary_semigroup.py prepare-data \
  --output-dir "$root/data-prep" --smoke \
  | tee "$root/data-prep.log"
cache="$root/data-prep/boundary_wave1_cache.pt"

for boundary_mode in hard_dirichlet unconstrained; do
  for temporal_mode in autonomous query_time; do
    cell="$root/cells/${boundary_mode}-${temporal_mode}"
    "$python_bin" experiments/run_boundary_semigroup.py cell \
      --output-dir "$cell" \
      --data-cache "$cache" \
      --seed 31415 \
      --boundary-mode "$boundary_mode" \
      --temporal-mode "$temporal_mode" \
      --device cuda \
      --smoke \
      | tee "$root/${boundary_mode}-${temporal_mode}.log"
  done
done

test "$(find "$root/cells" -mindepth 2 -maxdepth 2 -name receipt.json | wc -l)" -eq 4
test "$(find "$root/cells" -mindepth 2 -maxdepth 2 -name done | wc -l)" -eq 4
printf 'passed\n' > "$root/smoke.done"
printf '__SMOKE_COMPLETE__\n'
