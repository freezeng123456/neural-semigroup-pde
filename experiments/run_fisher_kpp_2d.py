#!/usr/bin/env python3
"""
Experiment P9: 2D Fisher-KPP Equation.
Compares LatentSemigroupNetBounded([0,1]) vs BaselineResNet.

2D Fisher-KPP PDE:
    du/dt = nu * Laplacian(u) + r * u * (1-u),  (x,y) in [0,L]x[0,L]
    periodic BC, admissible set K = [0, 1]^{Nx*Ny}

The 2D state is flattened to N=Nx*Ny for the 1D models.

Usage:
    source ~/pyenvs/research/bin/activate
    python run_fisher_kpp_2d.py
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import (
    FisherKPP2DSolver,
    generate_fisher_kpp_2d_initial_conditions,
    generate_fisher_kpp_2d_training_data,
)
from models import LatentSemigroupNetBounded, BaselineResNet
from training import train_model
from evaluate import evaluate_full


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_fisher_kpp_2d_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/fisher_kpp_2d", exist_ok=True)

    # Parameters
    Nx, Ny = 32, 32
    N = Nx * Ny  # 1024
    L = 2 * np.pi
    nu = 0.1
    r = 1.0
    tau = 0.1
    T_max = 2.0
    dt_ref = 0.005
    n_train = 500
    n_val = 50
    n_epochs = 30
    batch_size = 32
    lr = 1e-3
    checkpoint_dir = "checkpoints/fisher_kpp_2d"

    print(f"\n{'='*70}")
    print(f"2D FISHER-KPP EXPERIMENT: Nx={Nx}, Ny={Ny}, N={N}")
    print(f"{'='*70}")

    t_total_start = time.time()

    # ============================================================
    # Generate data
    # ============================================================
    data_path = os.path.join(checkpoint_dir, f"data_N{Nx}x{Ny}.pt")
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        train_u0 = data["train_u0"]
        train_ut = data["train_ut"]
        val_u0 = data["val_u0"]
        val_trajs = data["val_trajs"]
    else:
        t0 = time.time()
        train_u0, train_ut, val_u0, val_trajs = generate_fisher_kpp_2d_training_data(
            n_train=n_train, n_val=n_val, Nx=Nx, Ny=Ny, L=L,
            nu=nu, r=r, dt=dt_ref, T_max=T_max, tau=tau
        )
        print(f"Data generation: {time.time()-t0:.0f}s")
        torch.save({
            "train_u0": train_u0, "train_ut": train_ut,
            "val_u0": val_u0, "val_trajs": val_trajs,
        }, data_path)

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    results = {}

    # ============================================================
    # LatentSemigroupNetBounded
    # ============================================================
    print(f"\n--- Training LatentSemigroupNetBounded (N={N}, [0,1]) ---")
    latent_model = LatentSemigroupNetBounded(
        N=N, m=0.0, M=1.0,
        hidden_V=[32, 32], hidden_K=[32, 32],
        stencil_radius=3, beta_V=0.0,
        interaction_radius=2,
    )
    n_params = sum(p.numel() for p in latent_model.parameters())
    print(f"Parameters: {n_params}")

    ckpt_path = os.path.join(checkpoint_dir, "latent_best.pt")
    resume = ckpt_path if os.path.exists(ckpt_path) else None

    t0 = time.time()
    history_latent = train_model(
        latent_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name="latent", device=device, resume_from=resume,
        lower_bound=0.0, upper_bound=1.0,
    )
    print(f"Latent training time: {time.time()-t0:.0f}s")

    # Load best
    ckpt = torch.load(os.path.join(checkpoint_dir, "latent_best.pt"),
                       map_location=device, weights_only=False)
    latent_model.load_state_dict(ckpt["model_state_dict"])
    latent_model.to(device)

    t0 = time.time()
    latent_metrics, _ = evaluate_full(
        latent_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="Latent_2D",
        lower_bound=0.0, upper_bound=1.0,
    )
    print(f"Latent evaluation: {time.time()-t0:.0f}s")
    results["latent"] = latent_metrics
    print(f"  Latent rollout_mse: {latent_metrics['rollout_mse_mean']:.4e}")
    print(f"  Latent bound_viol: {latent_metrics['bound_viol_mean']:.4e}")
    print(f"  Latent sg_defect:  {latent_metrics['semigroup_defect_mean']:.4e}")

    # ============================================================
    # BaselineResNet
    # ============================================================
    print(f"\n--- Training BaselineResNet (N={N}) ---")
    baseline_model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
    n_params_b = sum(p.numel() for p in baseline_model.parameters())
    print(f"Parameters: {n_params_b}")

    t0 = time.time()
    history_baseline = train_model(
        baseline_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=batch_size, lr=lr,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name="baseline", device=device,
        lower_bound=0.0, upper_bound=1.0,
    )
    print(f"Baseline training time: {time.time()-t0:.0f}s")

    ckpt_b = torch.load(os.path.join(checkpoint_dir, "baseline_best.pt"),
                          map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_b["model_state_dict"])
    baseline_model.to(device)

    t0 = time.time()
    baseline_metrics, _ = evaluate_full(
        baseline_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="Baseline_2D",
        lower_bound=0.0, upper_bound=1.0,
    )
    print(f"Baseline evaluation: {time.time()-t0:.0f}s")
    results["baseline"] = baseline_metrics
    print(f"  Baseline rollout_mse: {baseline_metrics['rollout_mse_mean']:.4e}")
    print(f"  Baseline bound_viol: {baseline_metrics['bound_viol_mean']:.4e}")
    print(f"  Baseline sg_defect:  {baseline_metrics['semigroup_defect_mean']:.4e}")

    # ============================================================
    # Save results
    # ============================================================
    with open("results/fisher_kpp_2d_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/fisher_kpp_2d_comparison.json")

    # Print summary table
    print("\n" + "=" * 70)
    print("2D FISHER-KPP RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Latent':>15} {'Baseline':>15}")
    print("-" * 65)
    keys = [
        ("rollout_mse_mean", "Rollout MSE"),
        ("bound_viol_mean", "Bound violation"),
        ("semigroup_defect_mean", "Semigroup defect"),
        ("energy_mono_frac_mean", "Energy monotonicity"),
    ]
    for key, label in keys:
        v_lat = results["latent"].get(key, float("nan"))
        v_base = results["baseline"].get(key, float("nan"))
        print(f"{label:<35} {v_lat:>15.4e} {v_base:>15.4e}")
    print("=" * 65)

    total_time = time.time() - t_total_start
    print(f"\nTotal time: {total_time:.0f}s")
    print("2D FISHER-KPP EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
