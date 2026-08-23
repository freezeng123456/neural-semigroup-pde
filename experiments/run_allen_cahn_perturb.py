#!/usr/bin/env python3
"""
Retrain Allen-Cahn Latent with weight perturbation to escape degenerate local minimum.
The current model has constant K and a_ij — needs perturbation to break symmetry.
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

    N = 64
    tau = 0.1
    n_epochs = 200
    checkpoint_dir = "checkpoints/allen_cahn_perturb"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data
    data = torch.load("checkpoints/allen_cahn/data_N64.pt", map_location="cpu", weights_only=False)
    train_u0, train_ut = data["train_u0"], data["train_ut"]
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    # Load checkpoint and perturb weights
    ckpt_path = "checkpoints/allen_cahn_new/latent_N64_best.pt"
    model = LatentSemigroupNetBounded(
        N=N, m=-1.0, M=1.0,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Perturb weights: add noise to break symmetry
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'emb' in name:
                # Large perturbation to embeddings (interaction coefficients)
                param.add_(torch.randn_like(param) * 0.1)
            elif 'K_net' in name:
                # Perturb mobility network
                param.add_(torch.randn_like(param) * 0.05)
            elif 'V_net' in name:
                # Smaller perturbation to potential network
                param.add_(torch.randn_like(param) * 0.01)

    model.to(device)

    # Verify perturbation worked
    with torch.no_grad():
        a_full = torch.nn.functional.softplus(torch.mm(model.emb, model.emb.t()))
        a_ij = a_full * model.interaction_mask
        u = -0.5 + torch.rand(4, N, device=device)
        z = model.encode(u)
        K = torch.nn.functional.softplus(model.K_net(z)) + 5e-3
        print(f"After perturbation:")
        print(f"  a_ij: mean={a_ij[a_ij>0].mean():.4f}, std={a_ij[a_ij>0].std():.4f}")
        print(f"  K: mean={K.mean():.4f}, std={K.std():.4f}")

    # Train with fresh optimizer
    print(f"\n{'='*70}")
    print(f"RETRAINING with perturbed weights: {n_epochs} epochs, lr=1e-3")
    print(f"{'='*70}")

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name="latent_perturb", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )
    print(f"\nTraining: {time.time()-t0:.0f}s")

    # Evaluate
    best_ckpt = torch.load(os.path.join(checkpoint_dir, "latent_perturb_best.pt"),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(device)

    metrics, _ = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="Latent_perturb",
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
    with open("results/allen_cahn_latent_perturb.json", "w") as f:
        json.dump({"n_params": sum(p.numel() for p in model.parameters()), **metrics}, f, indent=2)
    print(f"\nSaved results/allen_cahn_latent_perturb.json")


if __name__ == "__main__":
    main()
