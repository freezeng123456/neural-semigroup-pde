#!/usr/bin/env python3
"""
Retrain Allen-Cahn Latent model from existing checkpoint with higher LR.
Goal: escape plateau at val_mse=2.21e-2.
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LatentSemigroupNetBounded
from training import train_model
from evaluate import evaluate_full


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    N = 64
    tau = 0.1
    n_epochs = 200
    checkpoint_dir = "checkpoints/allen_cahn_tune"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data
    data = torch.load("checkpoints/allen_cahn/data_N64.pt", map_location="cpu", weights_only=False)
    train_u0, train_ut = data["train_u0"], data["train_ut"]
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    # Load existing checkpoint
    ckpt_path = "checkpoints/allen_cahn_new/latent_N64_best.pt"
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = LatentSemigroupNetBounded(
        N=N, m=-1.0, M=1.0,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    print(f"Loaded from epoch {ckpt['epoch']}, val_mse={ckpt['best_val_mse']:.4e}")

    # Train with rollout loss + energy loss (matching run_allen_cahn.py)
    # Fresh optimizer with lr=1e-3, 200 epochs
    print(f"\n{'='*70}")
    print(f"RETRAINING Allen-Cahn Latent: 200 epochs, lr=1e-3, rollout+energy loss")
    print(f"{'='*70}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name="latent_tune", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )
    elapsed = time.time() - t0
    print(f"\nTraining: {elapsed:.0f}s")

    # Evaluate best checkpoint
    best_ckpt = torch.load(os.path.join(checkpoint_dir, "latent_tune_best.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="Latent",
        lower_bound=-1.0, upper_bound=1.0,
    )
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Rollout MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"  Bound Viol:  {metrics['bound_viol_mean']:.4e}")
    print(f"  SG Defect:   {metrics['semigroup_defect_mean']:.4e}")
    print(f"  Energy Mono: {metrics.get('energy_mono_frac_mean', 0):.2f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/allen_cahn_latent_tune.json", "w") as f:
        json.dump({"n_params": sum(p.numel() for p in model.parameters()), **metrics}, f, indent=2)
    print(f"\nSaved results/allen_cahn_latent_tune.json")


if __name__ == "__main__":
    main()
