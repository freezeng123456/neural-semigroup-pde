#!/bin/bash
#SBATCH --job-name=bnd-w2-smoke
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/joblogs/smoke-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/joblogs/smoke-%j.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1
src="$base/semigroup-148f494-r2"
root="$base/smoke-148f494-r1"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
pydeps=/work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-tests-3c46763-r2/pydeps

test ! -e "$root"
test -d "$src"
test -d "$pydeps/pytest"
mkdir -p "$root/data-prep" "$root/cells"
export PYTHONPATH="$src:$pydeps"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=148f4946fa489ec2b0bf485cf76bc62c7456b3e3
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=708cef2741c564d3bb530c677f2cdc8898734371712ee76c81e0ae92aa6e53f6

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
  "$python_bin" -c "import sys,torch,numpy,pytest; print('python',sys.executable); print('torch',torch.__version__); print('torch_cuda',torch.version.cuda); print('numpy',numpy.__version__); print('pytest',pytest.__version__); print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0)); print('gpu_count',torch.cuda.device_count())"
} | tee "$root/gpu_probe.txt"

cd "$src"
"$python_bin" -m pytest -q experiments/tests/test_boundary_family_semigroup.py \
  | tee "$root/pytest.log"

families=(inhomogeneous_dirichlet homogeneous_neumann robin)
enforcements=(hard penalty unconstrained)
temporals=(autonomous query_time)

for family in "${families[@]}"; do
  prep="$root/data-prep/$family"
  "$python_bin" experiments/run_boundary_family_semigroup.py prepare-data \
    --output-dir "$prep" \
    --boundary-family "$family" \
    --smoke \
    | tee "$root/data-prep-$family.log"
done

for family in "${families[@]}"; do
  cache="$root/data-prep/$family/boundary_family_wave2_${family}_cache.pt"
  for enforcement in "${enforcements[@]}"; do
    for temporal in "${temporals[@]}"; do
      cell_name="${family}-${enforcement}-${temporal}"
      "$python_bin" experiments/run_boundary_family_semigroup.py cell \
        --output-dir "$root/cells/$cell_name" \
        --data-cache "$cache" \
        --seed 31415 \
        --boundary-family "$family" \
        --enforcement-mode "$enforcement" \
        --temporal-mode "$temporal" \
        --device cuda \
        --smoke \
        | tee "$root/$cell_name.log"
    done
  done
done

test "$(find "$root/cells" -mindepth 2 -maxdepth 2 -name receipt.json | wc -l)" -eq 18
test "$(find "$root/cells" -mindepth 2 -maxdepth 2 -name done | wc -l)" -eq 18
test "$(grep -l '"hard_boundary_pass": true' "$root"/cells/*-hard-*/receipt.json | wc -l)" -eq 6
printf 'passed\n' > "$root/smoke.done"
printf '__SMOKE_COMPLETE__\n'
