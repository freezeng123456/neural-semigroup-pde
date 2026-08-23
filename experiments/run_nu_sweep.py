#!/usr/bin/env python3
"""
Sweep diffusion coefficient nu for Fisher-KPP equation.

For each nu value:
  1. Generate training data (spectral ETD-RK4 reference solver)
  2. Train LatentSemigroupNet
  3. Evaluate rollout MSE, one-step MSE, bound violation, semigroup defect

Usage:
    python run_nu_sweep.py
"""
import torch
import numpy as np
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import FisherKPPSolver, generate_initial_conditions
from models import LatentSemigroupNet
from training import train_model
from evaluate import evaluate_full

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_data_for_nu(N, nu, r, L, dt_pde, tau, T_max, n_train, n_val):
    """Generate training and validation data for a given nu."""
    solver = FisherKPPSolver(N=N, L=L, nu=nu, r=r, dt=dt_pde)
    n_tau_steps = int(tau / dt_pde)

    print(f"  Generating {n_train} training samples (nu={nu})...")
    train_u0 = generate_initial_conditions(N, n_train, L)
    train_ut = torch.zeros_like(train_u0)

    for i in range(n_train):
        t, u = solver.solve(train_u0[i], tau, save_every=n_tau_steps)
        train_ut[i] = u[-1]

    print(f"  Generating {n_val} validation trajectories...")
    val_u0 = generate_initial_conditions(N, n_val, L)
    val_trajs = []
    for i in range(n_val):
        t, u = solver.solve(val_u0[i], T_max, save_every=1)
        val_trajs.append((t, u))

    return train_u0, train_ut, val_u0, val_trajs


def run_single_nu(nu, N=64, L=10.0, r=1.0, dt_pde=0.005, tau=0.1,
                  T_max=2.0, n_train=1000, n_val=50, n_epochs=100):
    """Train and evaluate the latent model for a single nu value."""

    # For very small nu, use finer time step for reference solver stability
    if nu <= 0.002:
        dt_pde = 0.001
        print(f"  Using finer dt_pde={dt_pde} for small nu={nu}")

    print(f"\n{'='*60}")
    print(f"  nu = {nu}, N = {N}, tau = {tau}, dt_pde = {dt_pde}")
    print(f"{'='*60}")

    # 1. Generate data
    t0 = time.time()
    train_u0, train_ut, val_u0, val_trajs = generate_data_for_nu(
        N=N, nu=nu, r=r, L=L, dt_pde=dt_pde, tau=tau, T_max=T_max,
        n_train=n_train, n_val=n_val
    )
    print(f"  Data generation: {time.time()-t0:.0f}s")

    # 2. Train latent model
    model = LatentSemigroupNet(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")

    ckpt_dir = f"checkpoints/nu_{nu}"
    os.makedirs(ckpt_dir, exist_ok=True)

    t0 = time.time()
    history = train_model(
        model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1, alpha_V=0.0,
        weight_decay=1e-5, checkpoint_dir=ckpt_dir,
        model_name="latent", device=DEVICE
    )
    train_time = time.time() - t0
    print(f"  Training: {train_time:.0f}s")

    # 3. Evaluate
    best_ckpt = torch.load(os.path.join(ckpt_dir, "latent_best.pt"),
                           map_location=DEVICE, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.to(DEVICE)

    metrics, details = evaluate_full(
        model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=DEVICE, model_name="Latent",
        lower_bound=0.0, upper_bound=1.0
    )

    print(f"\n  Results for nu={nu}:")
    print(f"    Rollout MSE:   {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"    Bound viol:    {metrics['bound_viol_mean']:.4e}")
    print(f"    Semigrp defect:{metrics['semigroup_defect_mean']:.4e}")
    print(f"    Energy mono:   {metrics['energy_mono_frac_mean']:.4f}")

    return {
        "nu": nu,
        "n_params": n_params,
        "train_time_s": train_time,
        "rollout_mse_mean": metrics["rollout_mse_mean"],
        "rollout_mse_std": metrics["rollout_mse_std"],
        "bound_viol_mean": metrics["bound_viol_mean"],
        "semigroup_defect_mean": metrics["semigroup_defect_mean"],
        "energy_mono_frac_mean": metrics["energy_mono_frac_mean"],
        "per_step_mse": details.get("per_step_mse", []).tolist()
            if hasattr(details.get("per_step_mse", []), "tolist") else details.get("per_step_mse", []),
    }


def main():
    # Redirect output to log file
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_nu_sweep_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Sweep nu values: from 0.1 (current) down to 0.001
    nu_values = [0.1, 0.05, 0.01, 0.005, 0.001]

    results = []
    for nu in nu_values:
        try:
            res = run_single_nu(
                nu=nu, N=64, L=10.0, r=1.0,
                dt_pde=0.005, tau=0.1, T_max=2.0,
                n_train=1000, n_val=50, n_epochs=100
            )
            results.append(res)
        except Exception as e:
            print(f"  ERROR for nu={nu}: {e}")
            results.append({"nu": nu, "error": str(e)})

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Diffusion Coefficient Sweep (Fisher-KPP)")
    print("=" * 70)
    print(f"{'nu':>8s} | {'Rollout MSE':>14s} | {'Bound Viol':>12s} | {'SG Defect':>12s} | {'Energy Mono':>11s} | {'Time':>6s}")
    print("-" * 70)
    for r in results:
        if "error" in r:
            print(f"{r['nu']:8.4f} | {'ERROR':>14s}")
        else:
            print(f"{r['nu']:8.4f} | {r['rollout_mse_mean']:14.4e} | {r['bound_viol_mean']:12.4e} | "
                  f"{r['semigroup_defect_mean']:12.4e} | {r['energy_mono_frac_mean']:11.4f} | {r['train_time_s']:5.0f}s")

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "nu_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 70)
    print("SWEEP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
