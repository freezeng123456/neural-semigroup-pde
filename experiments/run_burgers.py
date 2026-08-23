#!/usr/bin/env python3
"""
Experiment B1: Viscous Burgers Equation.
Compares LatentSemigroupNet ([-1,1]) vs BaselineResNet.
Also runs convergence study with N=32, 64, 128.
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import BurgersSolver, generate_burgers_initial_conditions, generate_burgers_training_data
from models import LatentSemigroupNetBounded, BaselineResNet, FNOBaseline
from training import train_model
from evaluate import evaluate_full


def run_burgers_single(N=64, n_train=1000, n_val=50, n_epochs=50, device="cuda",
                       checkpoint_dir="checkpoints/burgers", results_prefix="burgers"):
    """Run Burgers experiment for a single N."""
    L = 2 * np.pi
    nu = 0.01
    tau = 0.05
    T_max = 1.0

    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"BURGERS EXPERIMENT: N={N}")
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
        train_u0, train_ut, val_u0, val_trajs = generate_burgers_training_data(
            N=N, n_train=n_train, n_val=n_val, L=L, nu=nu,
            dt=0.001, T_max=T_max, tau=tau
        )
        torch.save({
            "train_u0": train_u0, "train_ut": train_ut,
            "val_u0": val_u0, "val_trajs": val_trajs,
        }, data_path)

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    results = {}

    # --- LatentSemigroupNet ---
    print(f"\n--- Training LatentSemigroupNet (N={N}) ---")
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
    )

    # Load best
    ckpt = torch.load(os.path.join(checkpoint_dir, f"latent_N{N}_best.pt"),
                       map_location=device, weights_only=False)
    latent_model.load_state_dict(ckpt["model_state_dict"])
    latent_model.to(device)

    latent_metrics, _ = evaluate_full(
        latent_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"Latent_N{N}",
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
    )

    ckpt_b = torch.load(os.path.join(checkpoint_dir, f"baseline_N{N}_best.pt"),
                          map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_b["model_state_dict"])
    baseline_model.to(device)

    baseline_metrics, _ = evaluate_full(
        baseline_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"Baseline_N{N}",
    )
    results["baseline"] = baseline_metrics
    print(f"  Baseline rollout_mse: {baseline_metrics['rollout_mse_mean']:.4e}")

    return results


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_burgers_log.txt")
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
    print("EXPERIMENT B1: VISCOUS BURGERS EQUATION")
    print("=" * 70)

    t0 = time.time()
    main_results = run_burgers_single(
        N=64, n_train=1000, n_val=50, n_epochs=50, device=device,
        checkpoint_dir="checkpoints/burgers", results_prefix="burgers"
    )
    print(f"\nMain Burgers experiment: {time.time()-t0:.0f}s")

    with open("results/burgers_comparison.json", "w") as f:
        json.dump(main_results, f, indent=2)
    print("Saved results/burgers_comparison.json")

    # ============================================================
    # Convergence study: N=32, 64, 128
    # ============================================================
    print("\n" + "=" * 70)
    print("BURGERS CONVERGENCE STUDY")
    print("=" * 70)

    convergence_results = {}
    for N in [32, 64, 128]:
        t1 = time.time()
        res = run_burgers_single(
            N=N, n_train=1000, n_val=30, n_epochs=40, device=device,
            checkpoint_dir=f"checkpoints/burgers_N{N}", results_prefix=f"burgers_N{N}"
        )
        convergence_results[f"N={N}"] = res
        print(f"  N={N} done in {time.time()-t1:.0f}s")

    with open("results/burgers_convergence.json", "w") as f:
        json.dump(convergence_results, f, indent=2)
    print("Saved results/burgers_convergence.json")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print("BURGERS EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
