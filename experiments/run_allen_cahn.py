#!/usr/bin/env python3
"""
Experiment A1: Allen-Cahn Equation.
Compares LatentSemigroupNetBounded([-1,1]) vs BaselineResNet vs FNOBaseline.

Allen-Cahn PDE:
    du/dt = eps^2 * d2u/dx2 + u(1 - u^2),  x in [0, 2*pi]
    periodic BC, admissible set K = [-1, 1]^N

Usage:
    source ~/pyenvs/research/bin/activate
    python run_allen_cahn.py
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import (
    AllenCahnSolver,
    generate_allen_cahn_initial_conditions,
    generate_allen_cahn_training_data,
)
from models import LatentSemigroupNetBounded, BaselineResNet, FNOBaseline
from training import train_model
from evaluate import evaluate_full


def run_allen_cahn_single(N=64, n_train=1000, n_val=50, n_epochs=50, device="cuda",
                          checkpoint_dir="checkpoints/allen_cahn"):
    """Run Allen-Cahn experiment for a single N."""
    L = 2 * np.pi
    eps = 0.1
    tau = 0.1
    T_max = 2.0
    dt_ref = 0.001  # fine reference solver time step

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"ALLEN-CAHN EXPERIMENT: N={N}")
    print(f"{'='*70}")

    # Generate data
    data_path = os.path.join(checkpoint_dir, f"data_N{N}.pt")
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        train_u0 = data["train_u0"]
        train_ut = data["train_ut"]
        val_u0 = data["val_u0"]
        val_trajs = data["val_trajs"]
    else:
        train_u0, train_ut, val_u0, val_trajs = generate_allen_cahn_training_data(
            N=N, n_train=n_train, n_val=n_val, L=L, eps=eps,
            dt=dt_ref, T_max=T_max, tau=tau
        )
        torch.save({
            "train_u0": train_u0, "train_ut": train_ut,
            "val_u0": val_u0, "val_trajs": val_trajs,
        }, data_path)

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    results = {}

    # --- LatentSemigroupNetBounded ---
    print(f"\n--- Training LatentSemigroupNetBounded (N={N}) ---")
    latent_model = LatentSemigroupNetBounded(
        N=N, m=-1.0, M=1.0,
        hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    n_params = sum(p.numel() for p in latent_model.parameters())
    print(f"Parameters: {n_params}")

    ckpt_path = os.path.join(checkpoint_dir, f"latent_N{N}_best.pt")
    resume = ckpt_path if os.path.exists(ckpt_path) else None

    history_latent = train_model(
        latent_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name=f"latent_N{N}", device=device, resume_from=resume,
        lower_bound=-1.0, upper_bound=1.0,
    )

    # Load best
    ckpt = torch.load(os.path.join(checkpoint_dir, f"latent_N{N}_best.pt"),
                       map_location=device, weights_only=False)
    latent_model.load_state_dict(ckpt["model_state_dict"])
    latent_model.to(device)

    latent_metrics, _ = evaluate_full(
        latent_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"Latent_N{N}",
        lower_bound=-1.0, upper_bound=1.0,
    )
    results["latent"] = latent_metrics
    print(f"  Latent rollout_mse: {latent_metrics['rollout_mse_mean']:.4e}")

    # --- BaselineResNet ---
    print(f"\n--- Training BaselineResNet (N={N}) ---")
    baseline_model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
    n_params_b = sum(p.numel() for p in baseline_model.parameters())
    print(f"Parameters: {n_params_b}")

    history_baseline = train_model(
        baseline_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name=f"baseline_N{N}", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )

    ckpt_b = torch.load(os.path.join(checkpoint_dir, f"baseline_N{N}_best.pt"),
                          map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_b["model_state_dict"])
    baseline_model.to(device)

    baseline_metrics, _ = evaluate_full(
        baseline_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"Baseline_N{N}",
        lower_bound=-1.0, upper_bound=1.0,
    )
    results["baseline"] = baseline_metrics
    print(f"  Baseline rollout_mse: {baseline_metrics['rollout_mse_mean']:.4e}")

    # --- FNOBaseline ---
    print(f"\n--- Training FNOBaseline (N={N}) ---")
    fno_model = FNOBaseline(N=N, width=16, n_modes=8, n_layers=4)
    n_params_f = sum(p.numel() for p in fno_model.parameters())
    print(f"Parameters: {n_params_f}")

    history_fno = train_model(
        fno_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name=f"fno_N{N}", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )

    ckpt_f = torch.load(os.path.join(checkpoint_dir, f"fno_N{N}_best.pt"),
                          map_location=device, weights_only=False)
    fno_model.load_state_dict(ckpt_f["model_state_dict"])
    fno_model.to(device)

    fno_metrics, _ = evaluate_full(
        fno_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"FNO_N{N}",
        lower_bound=-1.0, upper_bound=1.0,
    )
    results["fno"] = fno_metrics
    print(f"  FNO rollout_mse: {fno_metrics['rollout_mse_mean']:.4e}")

    return results


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_allen_cahn_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("results", exist_ok=True)

    # ============================================================
    # Main comparison: N=64
    # ============================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT A1: ALLEN-CAHN EQUATION")
    print("=" * 70)

    t0 = time.time()
    main_results = run_allen_cahn_single(
        N=64, n_train=1000, n_val=50, n_epochs=50, device=device,
        checkpoint_dir="checkpoints/allen_cahn",
    )
    print(f"\nMain Allen-Cahn experiment: {time.time()-t0:.0f}s")

    with open("results/allen_cahn_comparison.json", "w") as f:
        json.dump(main_results, f, indent=2)
    print("Saved results/allen_cahn_comparison.json")

    # Print summary table
    print("\n" + "=" * 70)
    print("ALLEN-CAHN RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'Latent':>15} {'Baseline':>15} {'FNO':>15}")
    print("-" * 75)
    keys = [
        ("rollout_mse_mean", "Rollout MSE"),
        ("bound_viol_mean", "Bound violation"),
        ("semigroup_defect_mean", "Semigroup defect"),
        ("energy_mono_frac_mean", "Energy monotonicity"),
    ]
    for key, label in keys:
        v_lat = main_results["latent"].get(key, float("nan"))
        v_base = main_results["baseline"].get(key, float("nan"))
        v_fno = main_results["fno"].get(key, float("nan"))
        print(f"{label:<30} {v_lat:>15.4e} {v_base:>15.4e} {v_fno:>15.4e}")
    print("=" * 75)

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("ALLEN-CAHN EXPERIMENT COMPLETE")


if __name__ == "__main__":
    main()
