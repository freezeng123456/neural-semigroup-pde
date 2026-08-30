#!/bin/bash
#SBATCH --job-name=bnd-w2-a
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --array=0-17%18
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/wave-148f494-r1/joblogs/phase-a-%A_%a.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1/wave-148f494-r1/joblogs/phase-a-%A_%a.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-family-wave2-148f494-r1
src="$base/semigroup-148f494-r2"
wave="$base/wave-148f494-r1"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
families=(inhomogeneous_dirichlet homogeneous_neumann robin)
enforcements=(hard penalty unconstrained)
temporals=(autonomous query_time)

test -f "$wave/data-prep.done"
family_index=$((SLURM_ARRAY_TASK_ID / 6))
combo_index=$((SLURM_ARRAY_TASK_ID % 6))
enforcement_index=$((combo_index / 2))
temporal_index=$((combo_index % 2))
family=${families[$family_index]}
enforcement=${enforcements[$enforcement_index]}
temporal=${temporals[$temporal_index]}
seed=31415
cache="$wave/data-prep/$family/boundary_family_wave2_${family}_cache.pt"
cell_name="seed${seed}-${family}-${enforcement}-${temporal}"
cell="$wave/cells/$cell_name"

test -f "$cache"
test ! -e "$cell"
export PYTHONPATH="$src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=148f4946fa489ec2b0bf485cf76bc62c7456b3e3
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=708cef2741c564d3bb530c677f2cdc8898734371712ee76c81e0ae92aa6e53f6

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
printf 'slurm_array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
printf 'cell=%s\n' "$cell_name"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
"$python_bin" -c "import torch; print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0)); print('gpu_count',torch.cuda.device_count())"

cd "$src"
"$python_bin" experiments/run_boundary_family_semigroup.py cell \
  --output-dir "$cell" \
  --data-cache "$cache" \
  --seed "$seed" \
  --boundary-family "$family" \
  --enforcement-mode "$enforcement" \
  --temporal-mode "$temporal" \
  --device cuda

test "$(cat "$cell/done")" = passed
test -f "$cell/receipt.json"
printf '__CELL_COMPLETE__ %s\n' "$cell_name"
