#!/usr/bin/env bash
# Run once on the login node; all computation runs in Slurm.
set -eu
root=$1
code_dir=$(cd "$(dirname "$0")/../.." && pwd)
python_bin=/work/home/zenghang/miniconda3/envs/pytorch/bin/python
: "${SEMIGROUP_SOURCE_COMMIT:?Frozen source commit required}"
[ ! -e "$root" ] || { echo 'Existing run preserved'; exit 2; }
mkdir -p "$root/logs"
printf '%s\n' "$SEMIGROUP_SOURCE_COMMIT" > "$root/source_commit.txt"
cp "$code_dir/docs/research/SCNET_NEW_EQUATIONS_PROTOCOL.md" "$root/protocol.md"
sinfo -p xhhgnormal -o '%P %a %l %D %G' > "$root/partition.txt"
squeue -u "$(id -un)" > "$root/preexisting_jobs.txt"
sacctmgr -nP show assoc where user="$(id -un)" format=Account,Partition,GrpTRES,GrpSubmitJobs > "$root/association.txt"
cat > "$root/smoke.sh" <<INNER
#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SEMIGROUP_SOURCE_COMMIT='$SEMIGROUP_SOURCE_COMMIT'
cd '$code_dir'
'$python_bin' experiments/tests/test_new_equations.py
'$python_bin' experiments/run_new_equations.py prepare --root '$root/run'
'$python_bin' experiments/run_new_equations.py run --root '$root/run' --cell 0 --updates 2 --smoke
'$python_bin' experiments/run_new_equations.py run --root '$root/run' --cell 1 --updates 2 --smoke
'$python_bin' experiments/run_new_equations.py run --root '$root/run' --cell 12 --updates 2 --smoke
'$python_bin' experiments/run_new_equations.py run --root '$root/run' --cell 13 --updates 2 --smoke
printf 'passed\n' > '$root/smoke_passed'
INNER
cat > "$root/array.sh" <<INNER
#!/usr/bin/env bash
set -eu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SEMIGROUP_SOURCE_COMMIT='$SEMIGROUP_SOURCE_COMMIT'
[ -f '$root/smoke_passed' ]
cd '$code_dir'
'$python_bin' experiments/run_new_equations.py run --root '$root/run' --cell "\$SLURM_ARRAY_TASK_ID" --updates 5000
INNER
smoke_id=$(sbatch --parsable --partition=xhhgnormal --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --time=01:00:00 --output="$root/logs/smoke-%j.log" "$root/smoke.sh")
printf '%s\n' "$smoke_id" > "$root/smoke_job_id.txt"
array_id=$(sbatch --parsable --dependency="afterok:$smoke_id" --kill-on-invalid-dep=yes --partition=xhhgnormal --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --array=0-23%4 --time=00:45:00 --output="$root/logs/cell-%A-%a.log" "$root/array.sh")
printf '%s\n' "$array_id" > "$root/array_job_id.txt"
printf 'SMOKE=%s ARRAY=%s ROOT=%s\n' "$smoke_id" "$array_id" "$root"
