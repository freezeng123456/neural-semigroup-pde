#!/usr/bin/env python3
"""
Sweep diffusion coefficient nu for Fisher-KPP (continued from nu=0.1).
Runs nu = 0.05, 0.01, 0.005, 0.001 with 30 epochs each.
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
                  T_max=2.0, n_train=1000, n_val=50, n_epochs=30):
    # Smaller nu -> stiffer PDE, finer time step needed
    if nu <= 0.002:
        dt_pde = 0.001
        print(f"  Using finer dt_pde={dt_pde} for small nu={nu}")

    print(f"\n{'='*60}")
    print(f"  nu = {nu}, N = {N}, tau = {tau}, dt_pde = {dt_pde}")
    print(f"{'='*60}")

    t0 = time.time()
    train_u0, train_ut, val_u0, val_trajs = generate_data_for_nu(
        N=N, nu=nu, r=r, L=L, dt_pde=dt_pde, tau=tau, T_max=T_max,
        n_train=n_train, n_val=n_val
    )
    print(f"  Data generation: {time.time()-t0:.0f}s")

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

    # Evaluate with 20-step rollout
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
    print(f"    Rollout MSE:    {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    print(f"    Bound viol:     {metrics['bound_viol_mean']:.4e}")
    print(f"    Semigrp defect: {metrics['semigroup_defect_mean']:.4e}")
    print(f"    Energy mono:    {metrics['energy_mono_frac_mean']:.4f}")

    return {
        "nu": nu,
        "n_params": n_params,
        "train_time_s": train_time,
        "best_epoch": best_ckpt.get("epoch", -1),
        "best_val_mse_train": best_ckpt.get("best_val_mse", -1),
        "rollout_mse_mean": metrics["rollout_mse_mean"],
        "rollout_mse_std": metrics["rollout_mse_std"],
        "one_step_mse_mean": metrics.get("one_step_mse_mean", -1),
        "bound_viol_mean": metrics["bound_viol_mean"],
        "semigroup_defect_mean": metrics["semigroup_defect_mean"],
        "energy_mono_frac_mean": metrics["energy_mono_frac_mean"],
    }


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_nu_sweep_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Include already-computed nu=0.1 result
    results = [{
        "nu": 0.1,
        "n_params": 9603,
        "best_epoch": 7,
        "best_val_mse_train": 2.8715e-05,
        "rollout_mse_mean": None,  # will evaluate below
        "status": "checkpoint_saved"
    }]

    # Evaluate existing nu=0.1 checkpoint with full 20-step rollout
    print("\n" + "="*60)
    print("  Evaluating existing nu=0.1 checkpoint (20-step rollout)")
    print("="*60)
    data_path = "checkpoints/data.pt"
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    else:
        # Generate validation data
        _, _, val_u0, val_trajs = generate_data_for_nu(
            N=64, nu=0.1, r=1.0, L=10.0, dt_pde=0.005, tau=0.1,
            T_max=2.0, n_train=1000, n_val=50
        )

    model = LatentSemigroupNet(N=64, hidden_V=[64, 64], hidden_K=[64, 64],
                                stencil_radius=3, beta_V=0.0)
    ckpt = torch.load("checkpoints/nu_0.1/latent_best.pt",
                       map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    metrics, _ = evaluate_full(
        model, val_u0, val_trajs, tau=0.1,
        rollout_steps=20, device=DEVICE, model_name="Latent",
        lower_bound=0.0, upper_bound=1.0
    )
    results[0]["rollout_mse_mean"] = metrics["rollout_mse_mean"]
    results[0]["rollout_mse_std"] = metrics["rollout_mse_std"]
    results[0]["bound_viol_mean"] = metrics["bound_viol_mean"]
    results[0]["semigroup_defect_mean"] = metrics["semigroup_defect_mean"]
    results[0]["energy_mono_frac_mean"] = metrics["energy_mono_frac_mean"]
    print(f"  nu=0.1: rollout_mse={metrics['rollout_mse_mean']:.4e}")

    # Run remaining nu values
    nu_values = [0.05, 0.01, 0.005, 0.001]
    for nu in nu_values:
        try:
            res = run_single_nu(
                nu=nu, N=64, L=10.0, r=1.0,
                dt_pde=0.005, tau=0.1, T_max=2.0,
                n_train=1000, n_val=50, n_epochs=30
            )
            results.append(res)
        except Exception as e:
            print(f"  ERROR for nu={nu}: {e}")
            import traceback; traceback.print_exc()
            results.append({"nu": nu, "error": str(e)})

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Diffusion Coefficient Sweep (Fisher-KPP)")
    print("="*80)
    print(f"{'nu':>8s} | {'Rollout MSE':>14s} | {'± Std':>10s} | {'Bound Viol':>10s} | {'SG Defect':>12s} | {'Best Ep':>7s}")
    print("-"*80)
    for r in results:
        if "error" in r:
            print(f"{r['nu']:8.4f} | {'ERROR':>14s}")
        elif r.get("rollout_mse_mean") is not None:
            print(f"{r['nu']:8.4f} | {r['rollout_mse_mean']:14.4e} | {r.get('rollout_mse_std',0):10.4e} | "
                  f"{r.get('bound_viol_mean',0):10.4e} | {r.get('semigroup_defect_mean',0):12.4e} | {r.get('best_epoch','?'):>7}")

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "nu_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    print("\nSWEEP COMPLETE")


if __name__ == "__main__":
    main()
