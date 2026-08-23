#!/usr/bin/env python3
"""
Train BaselineResNet and FNO for Allen-Cahn benchmark.
Latent model already trained — only fill in the missing baselines.
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


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "run_allen_cahn_baselines_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    N = 64
    tau = 0.1
    n_epochs = 50
    checkpoint_dir = "checkpoints/allen_cahn"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    data_path = os.path.join(checkpoint_dir, f"data_N{N}.pt")
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        print("Generate data first with run_allen_cahn.py")
        return

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    train_u0 = data["train_u0"]
    train_ut = data["train_ut"]
    val_u0 = data["val_u0"]
    val_trajs = data["val_trajs"]
    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")

    results = {}

    # --- BaselineResNet ---
    print(f"\n{'='*70}")
    print(f"TRAINING BaselineResNet (Allen-Cahn, N={N})")
    print(f"{'='*70}")

    baseline_model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
    n_params_b = sum(p.numel() for p in baseline_model.parameters())
    print(f"Parameters: {n_params_b}")

    t0 = time.time()
    train_model(
        baseline_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name=f"baseline_N{N}", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )
    print(f"Baseline training: {time.time()-t0:.0f}s")

    ckpt_b = torch.load(os.path.join(checkpoint_dir, f"baseline_N{N}_best.pt"),
                          map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_b["model_state_dict"])
    baseline_model.to(device)

    baseline_metrics, _ = evaluate_full(
        baseline_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"Baseline_N{N}",
        lower_bound=-1.0, upper_bound=1.0,
    )
    results["baseline"] = {"n_params": n_params_b, **baseline_metrics}
    print(f"  Baseline MSE: {baseline_metrics['rollout_mse_mean']:.4e}")

    # --- FNOBaseline ---
    print(f"\n{'='*70}")
    print(f"TRAINING FNOBaseline (Allen-Cahn, N={N})")
    print(f"{'='*70}")

    fno_model = FNOBaseline(N=N, width=16, n_modes=8, n_layers=4)
    n_params_f = sum(p.numel() for p in fno_model.parameters())
    print(f"Parameters: {n_params_f}")

    t0 = time.time()
    train_model(
        fno_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=n_epochs, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
        weight_decay=1e-5, checkpoint_dir=checkpoint_dir,
        model_name=f"fno_N{N}", device=device,
        lower_bound=-1.0, upper_bound=1.0,
    )
    print(f"FNO training: {time.time()-t0:.0f}s")

    ckpt_f = torch.load(os.path.join(checkpoint_dir, f"fno_N{N}_best.pt"),
                          map_location=device, weights_only=False)
    fno_model.load_state_dict(ckpt_f["model_state_dict"])
    fno_model.to(device)

    fno_metrics, _ = evaluate_full(
        fno_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name=f"FNO_N{N}",
        lower_bound=-1.0, upper_bound=1.0,
    )
    results["fno"] = {"n_params": n_params_f, **fno_metrics}
    print(f"  FNO MSE: {fno_metrics['rollout_mse_mean']:.4e}")

    # Save results
    out_path = "results/allen_cahn_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")

    print("\n" + "=" * 70)
    print("ALLEN-CAHN BASELINES SUMMARY")
    print("=" * 70)
    for name, m in results.items():
        print(f"  {name:<20} MSE={m['rollout_mse_mean']:.4e}  BV={m['bound_viol_mean']:.2e}")


if __name__ == "__main__":
    main()
