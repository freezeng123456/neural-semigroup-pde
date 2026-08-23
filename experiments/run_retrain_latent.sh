#!/bin/bash
#SBATCH --job-name=retrain_latent
#SBATCH --partition=xhhgnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=retrain_latent_%j.out
#SBATCH --error=retrain_latent_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L 2>/dev/null || echo 'no GPU')"
echo "Start: $(date)"

cd ~/projects/DC-PINNs_LaTeX_arxiv_2604.13723/experiments

# Activate conda environment (scnet compute nodes)
source /public/software/apps/anaconda3/2023.09/etc/profile.d/conda.sh
conda activate research

echo "Python: $(which python3 2>/dev/null || which python 2>/dev/null || echo 'NOT FOUND')"
python3 --version 2>/dev/null || python --version 2>/dev/null

# Verify fixed models.py
echo "Verifying models.py fix..."
python3 -c "
from models import LatentSemigroupNet
import torch
m = LatentSemigroupNet(N=4, hidden_V=[4], hidden_K=[4])
z = torch.randn(1, 4)
g = m.grad_psi(z)
# The gradient should be finite and non-zero
assert g.isfinite().all(), 'grad_psi produced non-finite values!'
assert g.abs().sum() > 0, 'grad_psi is zero!'
print(f'grad_psi check PASSED: {g[0].tolist()}')
"

# Run all experiments
python3 run_all_latent.py 2>&1

echo "End: $(date)"
