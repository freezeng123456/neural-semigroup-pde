#!/usr/bin/env python3
"""
Experiment B2: FNO Baseline Comparison on Fisher-KPP.
Compares LatentSemigroupNet, BaselineResNet, and FNO.
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import FisherKPPSolver, generate_initial_conditions, generate_training_data
from models import LatentSemigroupNet, BaselineResNet, FNOBaseline
from training import train_model
from evaluate import evaluate_full


def main():
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_fno_log.txt")
    log_f = open(log_path, "w", buffering=1)
    sys.stdout = log_f
    sys.stderr = log_f

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints/fno", exist_ok=True)

    # Load Fisher-KPP data
    data_path = "checkpoints/data.pt"
    print(f"Loading cached data from {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    train_u0 = data["train_u0"]
    train_ut = data["train_ut"]
    val_u0 = data["val_u0"]
    val_trajs = data["val_trajs"]

    print(f"Train: {train_u0.shape}, Val: {val_u0.shape}")
    tau = 0.1
    N = 64

    results = {}

    # --- LatentSemigroupNet: load existing checkpoint ---
    print("\n" + "=" * 70)
    print("LOADING LatentSemigroupNet (Fisher-KPP)")
    print("=" * 70)

    latent_model = LatentSemigroupNet(
        N=N, hidden_V=[64, 64], hidden_K=[64, 64],
        stencil_radius=3, beta_V=0.0,
    )
    n_params = sum(p.numel() for p in latent_model.parameters())
    print(f"Parameters: {n_params}")

    ckpt = torch.load("checkpoints/latent_best.pt", map_location=device, weights_only=False)
    latent_model.load_state_dict(ckpt["model_state_dict"])
    latent_model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_mse={ckpt['best_val_mse']:.4e}")

    latent_metrics, _ = evaluate_full(
        latent_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="Latent",
    )
    results["latent"] = {"n_params": n_params, **latent_metrics}
    print(f"  Rollout MSE: {latent_metrics['rollout_mse_mean']:.4e}")

    # --- BaselineResNet: load existing checkpoint ---
    print("\n" + "=" * 70)
    print("LOADING BaselineResNet (Fisher-KPP)")
    print("=" * 70)

    baseline_model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
    n_params_b = sum(p.numel() for p in baseline_model.parameters())
    print(f"Parameters: {n_params_b}")

    ckpt_b = torch.load("checkpoints/baseline_best.pt", map_location=device, weights_only=False)
    baseline_model.load_state_dict(ckpt_b["model_state_dict"])
    baseline_model.to(device)
    print(f"Loaded checkpoint from epoch {ckpt_b['epoch']}, val_mse={ckpt_b['best_val_mse']:.4e}")

    baseline_metrics, _ = evaluate_full(
        baseline_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="BaselineResNet",
    )
    results["baseline_resnet"] = {"n_params": n_params_b, **baseline_metrics}
    print(f"  Rollout MSE: {baseline_metrics['rollout_mse_mean']:.4e}")

    # --- FNO Baseline: train from scratch ---
    print("\n" + "=" * 70)
    print("TRAINING FNO Baseline (Fisher-KPP)")
    print("=" * 70)

    fno_model = FNOBaseline(N=N, width=16, n_modes=16, n_layers=4)
    n_params_fno = sum(p.numel() for p in fno_model.parameters())
    print(f"Parameters: {n_params_fno}")

    t0 = time.time()
    train_model(
        fno_model, train_u0, train_ut, val_u0, val_trajs,
        tau=tau, n_epochs=100, batch_size=64, lr=1e-3,
        alpha_rollout=0.0, alpha_energy=0.0, alpha_bound=0.1,
        weight_decay=1e-5, checkpoint_dir="checkpoints/fno",
        model_name="fno", device=device,
    )
    print(f"FNO training: {time.time()-t0:.0f}s")

    ckpt_fno = torch.load("checkpoints/fno/fno_best.pt", map_location=device, weights_only=False)
    fno_model.load_state_dict(ckpt_fno["model_state_dict"])
    fno_model.to(device)

    fno_metrics, _ = evaluate_full(
        fno_model, val_u0, val_trajs, tau=tau,
        rollout_steps=20, device=device, model_name="FNO",
    )
    results["fno"] = {"n_params": n_params_fno, **fno_metrics}
    print(f"  Rollout MSE: {fno_metrics['rollout_mse_mean']:.4e}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("FNO COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Params':>8} {'Rollout MSE':>15} {'Bound Viol':>12} {'SG Defect':>12}")
    print("-" * 70)
    for name, data in results.items():
        print(f"{name:<25} {data['n_params']:>8d} {data['rollout_mse_mean']:>15.4e} "
              f"{data['bound_viol_mean']:>12.4e} {data['semigroup_defect_mean']:>12.4e}")

    with open("results/fno_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/fno_comparison.json")


if __name__ == "__main__":
    main()
