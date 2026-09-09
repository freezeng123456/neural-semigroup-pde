#!/usr/bin/env bash
# Arguments: smoke|submit ROOT ABS_PYTHON PARTITION GPU_BUDGET
set -eu
stage=$1
run_root=$2
python_bin=$3
partition=$4
gpu_budget=$5
code_dir=$(cd "$(dirname "$0")/../.." && pwd)
case "$run_root:$python_bin" in /*:/*) ;; *) echo 'Absolute paths required' >&2; exit 2;; esac
case "$gpu_budget" in 1|2|3|4) ;; *) echo 'This screen is bounded to 1-4 allocated GPUs' >&2; exit 2;; esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "$stage" = smoke ]; then
    [ ! -e "$run_root" ] || { echo 'Run root already exists; preserve it' >&2; exit 2; }
    mkdir -p "$run_root/logs"
    (cd "$code_dir" && git rev-parse HEAD) > "$run_root/source_commit.txt"
    sinfo -p "$partition" -o '%P %a %l %D %G' > "$run_root/partition.txt"
    squeue -u "$(id -un)" > "$run_root/preexisting_jobs.txt"
    cat > "$run_root/smoke.sh" <<EOF
#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd '$code_dir'
'$python_bin' experiments/tests/test_semigroup_discovery.py
'$python_bin' experiments/run_semigroup_discovery.py prepare --root '$run_root/run'
'$python_bin' experiments/run_semigroup_discovery.py run --root '$run_root/run' --cell 0 --updates 2 --smoke
'$python_bin' experiments/run_semigroup_discovery.py run --root '$run_root/run' --cell 9 --updates 2 --smoke
'$python_bin' experiments/run_semigroup_discovery.py run --root '$run_root/run' --cell 10 --updates 2 --smoke
printf 'passed\n' > '$run_root/smoke_passed'
EOF
    sbatch --parsable --partition="$partition" --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --time=00:30:00 --output="$run_root/logs/smoke-%j.log" "$run_root/smoke.sh" > "$run_root/smoke_job_id.txt"
    cat "$run_root/smoke_job_id.txt"
elif [ "$stage" = submit ]; then
    [ -f "$run_root/smoke_passed" ] || { echo 'Successful GPU smoke required' >&2; exit 2; }
    [ ! -e "$run_root/array_job_id.txt" ] || { echo 'Already submitted' >&2; exit 2; }
    current_commit=$(cd "$code_dir" && git rev-parse HEAD)
    [ "$current_commit" = "$(cat "$run_root/source_commit.txt")" ] || { echo 'Source changed since smoke' >&2; exit 2; }
    cat > "$run_root/array.sh" <<EOF
#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd '$code_dir'
'$python_bin' experiments/run_semigroup_discovery.py run --root '$run_root/run' --cell "\$SLURM_ARRAY_TASK_ID" --updates 2000
EOF
    sbatch --parsable --partition="$partition" --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --array="0-32%$gpu_budget" --time=00:30:00 --output="$run_root/logs/cell-%A-%a.log" "$run_root/array.sh" > "$run_root/array_job_id.txt"
    cat "$run_root/array_job_id.txt"
else
    echo 'Choose smoke or submit' >&2; exit 2
fi
