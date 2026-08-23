#!/usr/bin/env python3
"""
Main experiment script: Fisher-KPP semigroup learning benchmark.

Compares:
    1. LatentSemigroupNet (structure-preserving)
    2. BaselineResNet (no structure)

Usage:
    source ~/pyenvs/research/bin/activate
    python run_experiments.py
"""
import torch
import numpy as np
import os
import sys
import time
import json

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import FisherKPPSolver, generate_initial_conditions, generate_training_data
from models import LatentSemigroupNet, BaselineResNet
from training import train_model
from evaluate import evaluate_full, compare_models


def main():
    # Setup direct log file (bypass shell buffering)
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_log.txt")
    log_f = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = log_f
    sys.stderr = log_f
    # ============================================================
    # Configuration
    # ============================================================
    config = {
        # PDE
        "N": 64,              # grid points
        "L": 10.0,            # domain length
        "nu": 0.1,            # diffusion coefficient
        "r": 1.0,             # reaction rate
        "dt_pde": 0.005,      # PDE solver time step

        # Training data
        "n_train": 1000,      # training samples
        "n_val": 50,          # validation trajectories
        "tau": 0.1,           # learning time step
        "T_max": 2.0,         # max time for validation trajectories

        # Model
        "hidden_V": [64, 64],
        "hidden_K": [64, 64],
        "stencil_radius": 3,
        "beta_V": 0.0,

        # Training
        "n_epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,
        "alpha_rollout": 0.1,
        "alpha_energy": 0.01,
        "alpha_bound": 0.1,
        "weight_decay": 1e-5,

        # Evaluation
        "rollout_steps": 20,

        # Paths
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
    }

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # Save config
    with open(os.path.join(config["results_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ============================================================
    # Generate data
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING DATA")
    print("=" * 70)

    data_path = os.path.join(config["checkpoint_dir"], "data.pt")
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        train_u0 = data["train_u0"]
        train_ut = data["train_ut"]
        val_u0 = data["val_u0"]
        val_trajs = data["val_trajs"]
    else:
        train_u0, train_ut, val_u0, val_trajs = generate_training_data(
            N=config["N"], n_train=config["n_train"], n_val=config["n_val"],
            L=config["L"], nu=config["nu"], r=config["r"],
            dt=config["dt_pde"], T_max=config["T_max"], tau=config["tau"],
        )
        torch.save({
            "train_u0": train_u0,
            "train_ut": train_ut,
            "val_u0": val_u0,
            "val_trajs": val_trajs,
        }, data_path)
        print(f"Data saved to {data_path}")

    print(f"Train: {train_u0.shape[0]} samples")
    print(f"Val: {val_u0.shape[0]} trajectories")

    # ============================================================
    # Train LatentSemigroupNet
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING LatentSemigroupNet")
    print("=" * 70)

    latent_model = LatentSemigroupNet(
        N=config["N"],
        hidden_V=config["hidden_V"],
        hidden_K=config["hidden_K"],
        stencil_radius=config["stencil_radius"],
        beta_V=config["beta_V"],
    )
    n_params = sum(p.numel() for p in latent_model.parameters())
    print(f"Parameters: {n_params}")

    latent_ckpt_path = os.path.join(config["checkpoint_dir"], "latent_best.pt")
    resume_latent = latent_ckpt_path if os.path.exists(latent_ckpt_path) else None

    history_latent = train_model(
        latent_model, train_u0, train_ut, val_u0, val_trajs,
        tau=config["tau"], n_epochs=config["n_epochs"],
        batch_size=config["batch_size"], lr=config["lr"],
        alpha_rollout=config["alpha_rollout"],
        alpha_energy=config["alpha_energy"],
        alpha_bound=0.0,  # LatentNet has bounds by construction
        weight_decay=config["weight_decay"],
        checkpoint_dir=config["checkpoint_dir"],
        model_name="latent",
        device=device,
        resume_from=resume_latent,
    )

    # Load best checkpoint
    latent_ckpt = torch.load(
        os.path.join(config["checkpoint_dir"], "latent_best.pt"),
        map_location=device, weights_only=False
    )
    latent_model.load_state_dict(latent_ckpt["model_state_dict"])
    latent_model.to(device)

    # ============================================================
    # Train BaselineResNet
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING BaselineResNet")
    print("=" * 70)

    baseline_model = BaselineResNet(N=config["N"], hidden_dim=32, n_blocks=4)
    n_params_b = sum(p.numel() for p in baseline_model.parameters())
    print(f"Parameters: {n_params_b}")

    history_baseline = train_model(
        baseline_model, train_u0, train_ut, val_u0, val_trajs,
        tau=config["tau"], n_epochs=config["n_epochs"],
        batch_size=config["batch_size"], lr=config["lr"],
        alpha_rollout=0.0,  # ResNet can't handle rollout loss well
        alpha_energy=0.0,   # No energy function
        alpha_bound=config["alpha_bound"],
        weight_decay=config["weight_decay"],
        checkpoint_dir=config["checkpoint_dir"],
        model_name="baseline",
        device=device,
    )

    # Load best checkpoint
    baseline_ckpt = torch.load(
        os.path.join(config["checkpoint_dir"], "baseline_best.pt"),
        map_location=device, weights_only=False
    )
    baseline_model.load_state_dict(baseline_ckpt["model_state_dict"])
    baseline_model.to(device)

    # ============================================================
    # Evaluate both models
    # ============================================================
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    print("\n--- LatentSemigroupNet ---")
    latent_metrics, latent_details = evaluate_full(
        latent_model, val_u0, val_trajs,
        tau=config["tau"], rollout_steps=config["rollout_steps"],
        device=device, model_name="LatentSemigroupNet",
    )

    for k, v in latent_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")

    print("\n--- BaselineResNet ---")
    baseline_metrics, baseline_details = evaluate_full(
        baseline_model, val_u0, val_trajs,
        tau=config["tau"], rollout_steps=config["rollout_steps"],
        device=device, model_name="BaselineResNet",
    )

    for k, v in baseline_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")

    # Save detailed results
    torch.save({
        "latent_metrics": latent_metrics,
        "baseline_metrics": baseline_metrics,
        "latent_details": latent_details,
        "baseline_details": baseline_details,
    }, os.path.join(config["results_dir"], "evaluation_results.pt"))

    # Comparison
    compare_models(
        latent_metrics, baseline_metrics,
        os.path.join(config["results_dir"], "comparison.json"),
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Checkpoints: {os.path.abspath(config['checkpoint_dir'])}")
    print(f"Results: {os.path.abspath(config['results_dir'])}")


if __name__ == "__main__":
    main()
