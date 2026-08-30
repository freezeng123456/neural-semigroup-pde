#!/bin/bash
#SBATCH --job-name=fisher-gen-full
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1/logs/full-%A_%a.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1/logs/full-%A_%a.err

set -euo pipefail

run_root=/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1
source_root=/work/home/zenghang/semigroup_worktrees/e60a142-source
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
cache=/work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/locked-input/fisher_kpp_n64_test_s314163_h4p8.pt
formal_archive=/work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/source.tar.gz

seeds=(31415 31415 271828 271828 161803 161803)
models=(latent latent_query_time latent latent_query_time latent latent_query_time)
checkpoints=(
  /work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-frozen-inputs-s31415-r1/latent_best.pt
  /work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-frozen-inputs-s31415-r1/latent_query_time_best.pt
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s271828/cells/latent/output/latent/checkpoints/latent_best.pt
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s271828/cells/latent_query_time/output/latent_query_time/checkpoints/latent_query_time_best.pt
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s161803/cells/latent/output/latent/checkpoints/latent_best.pt
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s161803/cells/latent_query_time/output/latent_query_time/checkpoints/latent_query_time_best.pt
)
checkpoint_hashes=(
  4330c5760e5ec28d5150db8d7876282f83a474d3a62836988c92919f55441adb
  e57f1eccb1714357a832f5f9fa1e805b2e26869078e7429fdd81a7c22f991cd6
  0fa5f3067ef58999e575705db516cab9b94481a4a768eae7ced58034bd8ac1cb
  c87e71f8a11c5c848802880b8f2f685d8f6778d9534ce41746ba20b75ae51f42
  84dcd009982fbc2167aa0f08233ceb2af1ca8cc059623bf29078f64799bf3f9e
  f580c9c91441565932a706de4044ab1d62992673ba1e649eb67fd8ed3a9d673a
)

index=$SLURM_ARRAY_TASK_ID
seed=${seeds[$index]}
model=${models[$index]}
checkpoint=${checkpoints[$index]}
checkpoint_hash=${checkpoint_hashes[$index]}
cell_name="s${seed}-${model}"
output_root="$run_root/cells/$cell_name"

test -f "$run_root/smoke.done"
test -d "$source_root"
test ! -e "$output_root"
export PYTHONPATH="$source_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_array_job_id=%s\n' "$SLURM_ARRAY_JOB_ID"
printf 'slurm_array_task_id=%s\n' "$SLURM_ARRAY_TASK_ID"
printf 'cell=%s\n' "$cell_name"
printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
"$python_bin" -c "import sys,torch,numpy; print('python',sys.version); print('torch',torch.__version__); print('torch_cuda',torch.version.cuda); print('numpy',numpy.__version__); print('cuda_available',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0)); print('gpu_count',torch.cuda.device_count())"

cd "$source_root"
"$python_bin" experiments/evaluate_fisher_generator_stability.py \
  --model "$model" \
  --seed "$seed" \
  --checkpoint "$checkpoint" \
  --checkpoint-sha256 "$checkpoint_hash" \
  --test-cache "$cache" \
  --cache-sha256 29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e \
  --source-archive "$formal_archive" \
  --source-archive-sha256 a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5 \
  --expected-source-commit 637345584dc2db8ddccf9116a995615c3c036104 \
  --output-dir "$output_root" \
  --device cuda \
  --deterministic \
  --batch-size 64

test "$(cat "$output_root/done")" = passed
test -f "$output_root/receipt.json"
printf '__CELL_COMPLETE__ %s\n' "$cell_name"
