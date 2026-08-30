#!/bin/bash
#SBATCH --job-name=fisher-gen-aggregate
#SBATCH --partition=xhhgnormal
#SBATCH --account=acu722p2q8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1/logs/aggregate-%j.out
#SBATCH --error=/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1/logs/aggregate-%j.err

set -euo pipefail

run_root=/work/home/zenghang/semigroup_runs/20260831-fisher-generator-stability-e60a142-3seed-r1
source_root=/work/home/zenghang/semigroup_worktrees/e60a142-source
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
aggregate_root="$run_root/aggregate-e60a142-r1"

test ! -e "$aggregate_root"
test "$(find "$run_root/cells" -mindepth 2 -maxdepth 2 -name done | wc -l)" -eq 6
export PYTHONPATH="$source_root"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

printf 'hostname=%s\n' "$(hostname)"
printf 'slurm_job_id=%s\n' "$SLURM_JOB_ID"
cd "$source_root"
"$python_bin" experiments/aggregate_fisher_generator_stability.py \
  --cell-root "$run_root/cells/s31415-latent" \
  --cell-root "$run_root/cells/s31415-latent_query_time" \
  --cell-root "$run_root/cells/s271828-latent" \
  --cell-root "$run_root/cells/s271828-latent_query_time" \
  --cell-root "$run_root/cells/s161803-latent" \
  --cell-root "$run_root/cells/s161803-latent_query_time" \
  --output-dir "$aggregate_root" \
  | tee "$run_root/aggregate.log"

sha256sum \
  /work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-frozen-inputs-s31415-r1/latent_best.pt \
  /work/home/zenghang/semigroup_runs/20260830-semigroup-novelty-frozen-inputs-s31415-r1/latent_query_time_best.pt \
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s271828/cells/latent/output/latent/checkpoints/latent_best.pt \
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s271828/cells/latent_query_time/output/latent_query_time/checkpoints/latent_query_time_best.pt \
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s161803/cells/latent/output/latent/checkpoints/latent_best.pt \
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/seeds/s161803/cells/latent_query_time/output/latent_query_time/checkpoints/latent_query_time_best.pt \
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/locked-input/fisher_kpp_n64_test_s314163_h4p8.pt \
  /work/home/zenghang/semigroup_runs/20260830-fisher-kpp-formal-s271828-s161803-6373455-scnet-r1-a72ad8/source.tar.gz \
  > "$run_root/input-sha256-after.txt"
cmp "$run_root/input-sha256-before.txt" "$run_root/input-sha256-after.txt"
sha256sum "$aggregate_root/aggregate.json" "$aggregate_root/cell_summary.csv" "$aggregate_root/paired_summary.csv" "$aggregate_root/receipt.json" "$aggregate_root/exploratory_manifest.json" \
  | tee "$run_root/aggregate-artifacts.sha256"
printf 'passed\n' > "$run_root/full.done"
printf '__AGGREGATE_COMPLETE__\n'
