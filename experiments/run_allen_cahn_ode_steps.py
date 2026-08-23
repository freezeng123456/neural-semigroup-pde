#!/usr/bin/env python3
"""
Evaluate Allen-Cahn Latent with increased ODE substeps (30 → 100).
No retraining needed — just change ode_steps at inference time.
"""
import torch
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LatentSemigroupNetBounded
from evaluate import evaluate_full


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    N = 64
    tau = 0.1

    # Load data
    data = torch.load("checkpoints/allen_cahn/data_N64.pt", map_location="cpu", weights_only=False)
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    print(f"Val: {val_u0.shape}")

    # Load checkpoint
    ckpt_path = "checkpoints/allen_cahn_new/latent_N64_best.pt"
    model = LatentSemigroupNetBounded(
        N=N, m=-1.0, M=1.0,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded from epoch {ckpt['epoch']}, val_mse={ckpt['best_val_mse']:.4e}")

    results = {}

    # Evaluate with different ode_steps
    for steps in [30, 50, 100, 200]:
        model.ode_steps = steps
        metrics, _ = evaluate_full(
            model, val_u0, val_trajs, tau=tau,
            rollout_steps=20, device=device, model_name=f"Latent_M{steps}",
            lower_bound=-1.0, upper_bound=1.0,
        )
        results[f"ode_steps_{steps}"] = metrics
        print(f"  ode_steps={steps:>3d}: MSE={metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}  BV={metrics['bound_viol_mean']:.2e}  SD={metrics['semigroup_defect_mean']:.2e}")

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/allen_cahn_latent_ode_steps.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results/allen_cahn_latent_ode_steps.json")


if __name__ == "__main__":
    main()
