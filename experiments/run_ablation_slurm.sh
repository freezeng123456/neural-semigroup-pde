#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=xhhgnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=ablation_%j.out
#SBATCH --error=ablation_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'no GPU')"
echo "Start: $(date)"

cd ~/projects/DC-PINNs_LaTeX_arxiv_2604.13723/experiments

source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate research

echo "Python: $(which python3)"
python3 --version

# Force unbuffered output
export PYTHONUNBUFFERED=1

python3 run_ablation_only.py 2>&1

echo "End: $(date)"
