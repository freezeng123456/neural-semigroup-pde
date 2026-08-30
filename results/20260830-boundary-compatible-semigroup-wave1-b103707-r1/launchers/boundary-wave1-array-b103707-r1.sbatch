#!/bin/bash
#SBATCH --job-name=bnd-wave1-cell
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-11%12
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1/joblogs/cell-%A_%a.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2/wave-b103707-r1/joblogs/cell-%A_%a.err

set -euo pipefail

base=/work/home/zenghang/semigroup_runs/20260830-boundary-wave1-b103707-r2
src="$base/semigroup-b103707"
wave="$base/wave-b103707-r1"
cache="$wave/data-prep/boundary_wave1_cache.pt"
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
seeds=(31415 271828 161803)
boundary_modes=(hard_dirichlet unconstrained)
temporal_modes=(autonomous query_time)

test -f "$cache"
test -f "$wave/data-prep/done"
seed_index=$((SLURM_ARRAY_TASK_ID / 4))
combo_index=$((SLURM_ARRAY_TASK_ID % 4))
boundary_index=$((combo_index / 2))
temporal_index=$((combo_index % 2))
seed=${seeds[$seed_index]}
boundary_mode=${boundary_modes[$boundary_index]}
temporal_mode=${temporal_modes[$temporal_index]}
cell_name="seed${seed}-${boundary_mode}-${temporal_mode}"
cell="$wave/cells/$cell_name"

test ! -e "$cell"
export PYTHONPATH="$src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export SEMIGROUP_SOURCE_COMMIT=b1037072e3177365f2989019ada346f6f651dcf2
export SEMIGROUP_SOURCE_ARCHIVE_SHA256=ce0d4a20cad470f7ece2543ebd8115057aa1a80acdf7d5bf0ff57c68a7b5c5c4

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
printf 'slurm_array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
printf 'cell=%s\n' "$cell_name"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
"$python_bin" -c "import torch; print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0)); print('gpu_count',torch.cuda.device_count())"

cd "$src"
"$python_bin" experiments/run_boundary_semigroup.py cell \
  --output-dir "$cell" \
  --data-cache "$cache" \
  --seed "$seed" \
  --boundary-mode "$boundary_mode" \
  --temporal-mode "$temporal_mode" \
  --device cuda

test "$(cat "$cell/done")" = passed
test -f "$cell/receipt.json"
printf '__CELL_COMPLETE__ %s\n' "$cell_name"
