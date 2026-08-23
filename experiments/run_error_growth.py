#!/usr/bin/env python3
"""
Experiment B3: Error Growth Analysis.
Long-horizon rollout (T=20, 200 steps) on Fisher-KPP.
Records MSE at each time step for Latent, ResNet, FNO.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import FisherKPPSolver, generate_initial_conditions
from models import LatentSemigroupNet, BaselineResNet, FNOBaseline


@torch.no_grad()
def compute_error_growth(model, val_u0, val_trajs, tau, n_steps, device="cuda"):
    """Rollout model for n_steps and record MSE at each step."""
    model.eval()
    n_samples = len(val_u0)
    all_mse = np.zeros((n_samples, n_steps))
    all_bound_viol = np.zeros((n_samples, n_steps))

    for i in range(n_samples):
        u0 = val_u0[i:i+1].to(device)
        t_true, u_true = val_trajs[i]
        u_true = u_true.to(device)
        T = len(t_true) - 1
        steps = min(n_steps, T)
        if steps < 2:
            continue

        u_pred = u0
        for step in range(steps):
            u_pred = model(u_pred, tau)
            u_ref = u_true[step + 1:step + 2]
            all_mse[i, step] = torch.mean((u_pred - u_ref) ** 2).item()
            all_bound_viol[i, step] = (
                torch.mean(F.relu(-u_pred)) + torch.mean(F.relu(u_pred - 1.0))
            ).item()

    return all_mse, all_bound_viol


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_error_growth_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    N = 64
    tau = 0.1
    n_steps = 200  # T=20.0 / tau=0.1

    # Generate validation trajectories for T=20
    print("Generating validation trajectories for T=20...")
    solver = FisherKPPSolver(N=N, L=10.0, nu=0.1, r=1.0, dt=0.005)
    n_val = 20  # fewer samples for long trajectories
    val_u0 = generate_initial_conditions(N, n_val, L=10.0)
    val_trajs = []
    for i in range(n_val):
        t, u = solver.solve(val_u0[i], T=20.0, save_every=int(tau/0.005))
        val_trajs.append((t, u))
        if (i + 1) % 5 == 0:
            print(f"  val: {i+1}/{n_val}")

    print(f"Validation trajectories: {len(val_trajs)}, steps per traj: {len(val_trajs[0][0])-1}")

    # --- Load trained models ---
    # Load or regenerate training data for training
    data_path = "checkpoints/data.pt"
    if os.path.exists(data_path):
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        train_u0 = data["train_u0"]
        train_ut = data["train_ut"]
    else:
        from pde_solver import generate_training_data
        train_u0, train_ut, _, _ = generate_training_data(
            N=N, n_train=1000, n_val=50, L=10.0, nu=0.1, r=1.0,
            dt=0.005, T_max=2.0, tau=tau
        )

    results = {}

    # --- Latent Model ---
    print("\nLoading LatentSemigroupNet...")
    latent_model = LatentSemigroupNet(N=N, hidden_V=[64, 64], hidden_K=[64, 64])
    ckpt_path = "checkpoints/latent_best.pt"
    if not os.path.exists(ckpt_path):
        print("Training latent model...")
        from training import train_model
        train_model(latent_model, train_u0, train_ut, val_u0[:10],
                    val_trajs[:10], tau=tau, n_epochs=100, batch_size=64, lr=1e-3,
                    alpha_rollout=0.1, alpha_energy=0.01, alpha_bound=0.0,
                    checkpoint_dir="checkpoints", model_name="latent", device=device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    latent_model.load_state_dict(ckpt["model_state_dict"])
    latent_model.to(device)

    print("Computing error growth for Latent...")
    mse_latent, viol_latent = compute_error_growth(
        latent_model, val_u0, val_trajs, tau, n_steps, device
    )
    results["latent"] = {
        "mse_per_step": mse_latent.mean(axis=0).tolist(),
        "mse_std_per_step": mse_latent.std(axis=0).tolist(),
        "bound_viol_per_step": viol_latent.mean(axis=0).tolist(),
    }
    print(f"  Final MSE (step 200): {mse_latent[:, -1].mean():.4e}")

    # --- BaselineResNet ---
    print("\nLoading BaselineResNet...")
    baseline_model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
    ckpt_path_b = "checkpoints/baseline_best.pt"
    if not os.path.exists(ckpt_path_b):
        print("Training baseline model...")
        from training import train_model
        train_model(baseline_model, train_u0, train_ut, val_u0[:10],
                    val_trajs[:10], tau=tau, n_epochs=100, batch_size=64, lr=1e-3,
                    alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
                    checkpoint_dir="checkpoints", model_name="baseline", device=device)
    ckpt_b = torch.load(ckpt_path_b, map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_b["model_state_dict"])
    baseline_model.to(device)

    print("Computing error growth for BaselineResNet...")
    mse_baseline, viol_baseline = compute_error_growth(
        baseline_model, val_u0, val_trajs, tau, n_steps, device
    )
    results["baseline_resnet"] = {
        "mse_per_step": mse_baseline.mean(axis=0).tolist(),
        "mse_std_per_step": mse_baseline.std(axis=0).tolist(),
        "bound_viol_per_step": viol_baseline.mean(axis=0).tolist(),
    }
    print(f"  Final MSE (step 200): {mse_baseline[:, -1].mean():.4e}")

    # --- FNO ---
    print("\nLoading FNO...")
    fno_model = FNOBaseline(N=N, width=16, n_modes=16, n_layers=4)
    ckpt_path_f = "checkpoints/fno/fno_best.pt"
    if not os.path.exists(ckpt_path_f):
        print("Training FNO model...")
        from training import train_model
        train_model(fno_model, train_u0, train_ut, val_u0[:10],
                    val_trajs[:10], tau=tau, n_epochs=100, batch_size=64, lr=1e-3,
                    alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
                    checkpoint_dir="checkpoints/fno", model_name="fno", device=device)
    ckpt_f = torch.load(ckpt_path_f, map_location=device, weights_only=False)
    fno_model.load_state_dict(ckpt_f["model_state_dict"])
    fno_model.to(device)

    print("Computing error growth for FNO...")
    mse_fno, viol_fno = compute_error_growth(
        fno_model, val_u0, val_trajs, tau, n_steps, device
    )
    results["fno"] = {
        "mse_per_step": mse_fno.mean(axis=0).tolist(),
        "mse_std_per_step": mse_fno.std(axis=0).tolist(),
        "bound_viol_per_step": viol_fno.mean(axis=0).tolist(),
    }
    print(f"  Final MSE (step 200): {mse_fno[:, -1].mean():.4e}")

    # Add metadata
    results["metadata"] = {
        "n_steps": n_steps,
        "tau": tau,
        "T_max": 20.0,
        "n_val": n_val,
        "N": N,
    }

    with open("results/error_growth.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/error_growth.json")

    # Print summary table at key time steps
    print("\n" + "=" * 70)
    print("ERROR GROWTH SUMMARY (MSE at selected time steps)")
    print("=" * 70)
    key_steps = [0, 9, 19, 49, 99, 149, 199]
    print(f"{'Step':>6} {'Time':>6} {'Latent':>12} {'ResNet':>12} {'FNO':>12}")
    print("-" * 50)
    for s in key_steps:
        if s < n_steps:
            t = (s + 1) * tau
            ml = results["latent"]["mse_per_step"][s]
            mb = results["baseline_resnet"]["mse_per_step"][s]
            mf = results["fno"]["mse_per_step"][s]
            print(f"{s+1:>6d} {t:>6.1f} {ml:>12.4e} {mb:>12.4e} {mf:>12.4e}")


if __name__ == "__main__":
    main()
