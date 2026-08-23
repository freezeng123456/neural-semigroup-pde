#!/usr/bin/env python3
"""
Train Allen-Cahn with quartic interaction latent model.
Tests Direction 1: enhanced energy functional with (z_i-z_j)^4 term.
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models_quartic import LatentSemigroupNetQuartic
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
    checkpoint_dir = "checkpoints/allen_cahn_quartic"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load Allen-Cahn data
    data = torch.load("checkpoints/allen_cahn/data_N64.pt", map_location="cpu", weights_only=False)
    train_u0, train_ut = data["train_u0"], data["train_ut"]
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    # Quartic model
    model = LatentSemigroupNetQuartic(
        N=N, m=-1.0, M=1.0,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")

    # Quick check: energy before training
    with torch.no_grad():
        u_sample = train_u0[:4]
        E = model.energy(u_sample)
        print(f"Initial energy check: {E.mean():.4f}")

    # Train
    print(f"\n{'='*70}")
    print(f"TRAINING Allen-Cahn Quartic: {n_epochs} epochs, lr=1e-3")
    print(f"{'='*70}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name="latent_quartic", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )
    elapsed = time.time() - t0
    print(f"\nTraining: {elapsed:.0f}s")

    # Evaluate
    best_ckpt = torch.load(os.path.join(checkpoint_dir, "latent_quartic_best.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    # Check learned coefficients
    with torch.no_grad():
        a_full = torch.nn.functional.softplus(torch.mm(model.emb_a, model.emb_a.t()))
        a_ij = a_full * model.interaction_mask
        b_full = torch.nn.functional.softplus(torch.mm(model.emb_b, model.emb_b.t()))
        b_ij = b_full * model.interaction_mask
        u_sample = train_u0[:4].to(device)
        z = model.encode(u_sample)
        K = torch.nn.functional.softplus(model.K_net(z)) + 5e-3
        print(f"\nLearned parameters:")
        print(f"  a_ij: mean={a_ij[a_ij>0].mean():.4f}, std={a_ij[a_ij>0].std():.4f}")
        print(f"  b_ij: mean={b_ij[b_ij>0].mean():.4f}, std={b_ij[b_ij>0].std():.4f}")
        print(f"  K: mean={K.mean():.4f}, std={K.std():.4f}")

    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="LatentQuartic",
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
    with open("results/allen_cahn_quartic.json", "w") as f:
        json.dump({"n_params": n_params, **metrics}, f, indent=2)
    print(f"\nSaved results/allen_cahn_quartic.json")


if __name__ == "__main__":
    main()
