#!/usr/bin/env python3
"""
Experiment B4: Ablation Study on Fisher-KPP.

Tests:
1. Full latent model (baseline)
2. No interaction (a_ij = 0): only local potential V_theta
3. No mobility (K = I): gradient flow without spatially-varying mobility
4. Small beta_V=0 vs beta_V=0.1: effect of quadratic growth on global flow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import FisherKPPSolver, generate_initial_conditions, generate_training_data
from models import LatentSemigroupNet, ScalarMLP, StencilMLP
from training import train_model
from evaluate import evaluate_full


# ============================================================
# Ablation model variants
# ============================================================

class LatentNoInteraction(LatentSemigroupNet):
    """Latent model with no spatial interaction (a_ij = 0)."""
    def __init__(self, N=64, **kwargs):
        super().__init__(N=N, interaction_radius=0, **kwargs)
        # Zero out embeddings
        self.emb.data.zero_()
        # Freeze embeddings
        self.emb.requires_grad_(False)


class LatentUnitMobility(LatentSemigroupNet):
    """Latent model with K = I (identity mobility)."""
    def __init__(self, N=64, **kwargs):
        super().__init__(N=N, **kwargs)

    def K_diag(self, z):
        """Override to return identity (K=1 everywhere)."""
        return torch.ones_like(z)


def run_ablation_variant(variant_name, model, train_u0, train_ut, val_u0, val_trajs,
                         tau, n_epochs, batch_size, lr, device, checkpoint_dir):
    """Train and evaluate a single ablation variant."""
    print(f"\n--- {variant_name} ---")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params}")

    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name=variant_name, device=device,
    )

    # Load best checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{variant_name}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metrics, _ = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=variant_name,
    )

    return {"n_params": n_params, **metrics}


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_ablation_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/ablation", exist_ok=True)

    N = 64
    tau = 0.1
    n_epochs = 50
    batch_size = 64
    lr = 1e-3

    # Load or generate data
    data_path = "checkpoints/data.pt"
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        train_u0 = data["train_u0"]
        train_ut = data["train_ut"]
        val_u0 = data["val_u0"]
        val_trajs = data["val_trajs"]
    else:
        train_u0, train_ut, val_u0, val_trajs = generate_training_data(
            N=N, n_train=1000, n_val=50, L=10.0, nu=0.1, r=1.0,
            dt=0.005, T_max=2.0, tau=tau
        )

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    results = {}

    # 1. Full latent model (beta_V=0)
    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    t0 = time.time()

    model_full = LatentSemigroupNet(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    results["full"] = run_ablation_variant(
        "full", model_full, train_u0, train_ut, val_u0, val_trajs,
        tau, n_epochs, batch_size, lr, device, "checkpoints/ablation"
    )
    print(f"  Full model rollout_mse: {results['full']['rollout_mse_mean']:.4e}")

    # 2. No interaction
    model_no_int = LatentNoInteraction(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    results["no_interaction"] = run_ablation_variant(
        "no_interaction", model_no_int, train_u0, train_ut, val_u0, val_trajs,
        tau, n_epochs, batch_size, lr, device, "checkpoints/ablation"
    )
    print(f"  No interaction rollout_mse: {results['no_interaction']['rollout_mse_mean']:.4e}")

    # 3. No mobility (K=I)
    model_no_mob = LatentUnitMobility(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    results["no_mobility"] = run_ablation_variant(
        "no_mobility", model_no_mob, train_u0, train_ut, val_u0, val_trajs,
        tau, n_epochs, batch_size, lr, device, "checkpoints/ablation"
    )
    print(f"  No mobility rollout_mse: {results['no_mobility']['rollout_mse_mean']:.4e}")

    # 4. Full with beta_V=0.1
    model_beta = LatentSemigroupNet(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.1,
    )
    results["beta_V_0.1"] = run_ablation_variant(
        "beta_V_0.1", model_beta, train_u0, train_ut, val_u0, val_trajs,
        tau, n_epochs, batch_size, lr, device, "checkpoints/ablation"
    )
    print(f"  beta_V=0.1 rollout_mse: {results['beta_V_0.1']['rollout_mse_mean']:.4e}")

    print(f"\nAblation study: {time.time()-t0:.0f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<20} {'Params':>8} {'Rollout MSE':>15} {'Bound Viol':>12} {'SG Defect':>12} {'Energy Mono':>12}")
    print("-" * 80)
    for name, data in results.items():
        print(f"{name:<20} {data['n_params']:>8d} {data['rollout_mse_mean']:>15.4e} "
              f"{data['bound_viol_mean']:>12.4e} {data['semigroup_defect_mean']:>12.4e} "
              f"{data.get('energy_mono_frac_mean', 0.0):>12.4f}")

    with open("results/ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/ablation.json")


if __name__ == "__main__":
    main()
