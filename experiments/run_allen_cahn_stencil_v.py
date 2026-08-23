#!/usr/bin/env python3
"""
Train Allen-Cahn with stencil-based V_θ (Direction 2).
V_θ now takes local neighborhood as input, enabling spatial context.
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models_stencil_v import LatentSemigroupNetStencilV
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
    checkpoint_dir = "checkpoints/allen_cahn_stencil_v"
    os.makedirs(checkpoint_dir, exist_ok=True)

    data = torch.load("checkpoints/allen_cahn/data_N64.pt", map_location="cpu", weights_only=False)
    train_u0, train_ut = data["train_u0"], data["train_ut"]
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    # Stencil V model: radius=2 (5-point stencil for V)
    model = LatentSemigroupNetStencilV(
        N=N, m=-1.0, M=1.0,
        stencil_radius_V=2,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius_K=3,
        interaction_radius=2,
        beta_V=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")

    # Verify grad_V is non-zero after a few random inputs
    with torch.enable_grad():
        u_sample = train_u0[:4]
        z = model.encode(u_sample)
        z.requires_grad_(True)
        V = model.V_net(z)
        grad_V = torch.autograd.grad(V.sum(), z, create_graph=False, retain_graph=False)[0]
        print(f"Pre-training grad_V: mean_abs={grad_V.abs().mean():.6f}")

    print(f"\n{'='*70}")
    print(f"TRAINING Allen-Cahn Stencil-V: {n_epochs} epochs")
    print(f"{'='*70}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name="latent_stencil_v", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )
    elapsed = time.time() - t0
    print(f"\nTraining: {elapsed:.0f}s")

    # Evaluate
    best_ckpt = torch.load(os.path.join(checkpoint_dir, "latent_stencil_v_best.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    # Check learned grad_V
    with torch.enable_grad():
        u_sample = val_u0[:4].to(device)
        z = model.encode(u_sample)
        z.requires_grad_(True)
        V = model.V_net(z)
        grad_V = torch.autograd.grad(V.sum(), z, create_graph=False, retain_graph=False)[0]
        print(f"\nLearned grad_V: mean_abs={grad_V.abs().mean():.4f}, std={grad_V.std():.4f}")

    metrics, _ = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="LatentStencilV",
        lower_bound=-1.0, upper_bound=1.0,
    )
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"  Rollout MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"  Bound Viol:  {metrics['bound_viol_mean']:.4e}")
    print(f"  SG Defect:   {metrics['semigroup_defect_mean']:.4e}")
    print(f"  Energy Mono: {metrics.get('energy_mono_frac_mean', 0):.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/allen_cahn_stencil_v.json", "w") as f:
        json.dump({"n_params": n_params, **metrics}, f, indent=2)
    print(f"\nSaved results/allen_cahn_stencil_v.json")


if __name__ == "__main__":
    main()
