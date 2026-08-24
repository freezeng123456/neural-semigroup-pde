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
import argparse
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
from seed_utils import set_global_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the one-dimensional Fisher--KPP benchmark."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed", type=int, default=42,
        help="seed for generating the frozen training/validation data split",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="request deterministic PyTorch algorithms when available",
    )
    parser.add_argument(
        "--beta-v-floor", type=float, default=0.0,
        help="fixed non-trainable lower bound on the quadratic potential coefficient",
    )
    parser.add_argument(
        "--architecture-only", action="store_true",
        help="disable auxiliary rollout and learned-energy losses",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=5,
        help="run full rollout validation every N epochs (final epoch is always validated)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def fisher_kpp_discrete_energy(u, *, L, nu, r):
    """Periodic finite-difference proxy for the physical Fisher--KPP energy."""
    dx = float(L) / u.shape[-1]
    grad = (torch.roll(u, shifts=-1, dims=-1) - u) / dx
    density = 0.5 * float(nu) * grad.square()
    density = density - float(r) * (0.5 * u.square() - u.pow(3) / 3.0)
    return dx * density.sum(dim=-1)


def main():
    args = parse_args()
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
        "beta_V_floor": args.beta_v_floor,

        # Training
        "n_epochs": 100,
        "batch_size": 64,
        "lr": 1e-3,
        "alpha_rollout": 0.0 if args.architecture_only else 0.1,
        "alpha_energy": 0.0 if args.architecture_only else 0.01,
        "alpha_bound": 0.1,
        "weight_decay": 1e-5,
        "validation_interval": args.validation_interval,

        # Evaluation
        "rollout_steps": 20,
        "ode_substep_counts": [15, 30, 60],
        "collect_latent_diagnostics": True,

        # Reproducibility
        "seed": args.seed,
        "data_seed": args.data_seed,
        "deterministic": args.deterministic,
        "architecture_only": args.architecture_only,

        # Paths
        "checkpoint_dir": args.checkpoint_dir,
        "results_dir": args.results_dir,
    }

    if config["beta_V_floor"] < 0:
        raise ValueError("--beta-v-floor must be non-negative")
    if config["validation_interval"] <= 0:
        raise ValueError("--validation-interval must be positive")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # ============================================================
    # Generate data
    # ============================================================
    print("\n" + "=" * 70)
    print("GENERATING DATA")
    print("=" * 70)

    set_global_seed(config["data_seed"], deterministic=config["deterministic"])
    data_generation_config = {
        key: config[key]
        for key in (
            "data_seed", "N", "L", "nu", "r", "dt_pde", "n_train",
            "n_val", "tau", "T_max"
        )
    }

    data_path = os.path.join(config["checkpoint_dir"], "data.pt")
    if os.path.exists(data_path):
        print(f"Loading cached data from {data_path}")
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        train_u0 = data["train_u0"]
        train_ut = data["train_ut"]
        val_u0 = data["val_u0"]
        val_trajs = data["val_trajs"]
        cached_config = data.get("data_generation_config")
        if cached_config is None:
            config["data_cache_provenance"] = "legacy_unverified"
            print("WARNING: cached data predate recorded data-generation metadata")
        elif cached_config != data_generation_config:
            raise ValueError(
                "cached data configuration does not match the requested run; "
                "use an empty checkpoint directory or the matching --data-seed"
            )
        else:
            config["data_cache_provenance"] = "verified"
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
            "data_generation_config": data_generation_config,
        }, data_path)
        config["data_cache_provenance"] = "generated"
        print(f"Data saved to {data_path}")

    print(f"Train: {train_u0.shape[0]} samples")
    print(f"Val: {val_u0.shape[0]} trajectories")

    # Reset RNGs after data creation so model initialization and training are
    # controlled only by the training seed on a frozen data split.
    set_global_seed(config["seed"], deterministic=config["deterministic"])

    with open(os.path.join(config["results_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

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
        beta_V_floor=config["beta_V_floor"],
    )
    n_params = sum(p.numel() for p in latent_model.parameters())
    print(f"Parameters: {n_params}")

    latent_ckpt_path = os.path.join(config["checkpoint_dir"], "latent_best.pt")
    if os.path.exists(latent_ckpt_path) and not args.no_resume:
        if args.architecture_only or config["beta_V_floor"] > 0:
            raise ValueError(
                "architecture-only or positive-beta-floor runs must not resume an "
                "unverified checkpoint; pass --no-resume and use an isolated "
                "checkpoint directory"
            )
    resume_latent = (
        latent_ckpt_path
        if os.path.exists(latent_ckpt_path) and not args.no_resume
        else None
    )

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
        reference_dt=config["dt_pde"],
        run_metadata=config,
        validation_interval=config["validation_interval"],
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
        reference_dt=config["dt_pde"],
        run_metadata=config,
        validation_interval=config["validation_interval"],
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
    physical_energy = lambda u: fisher_kpp_discrete_energy(
        u, L=config["L"], nu=config["nu"], r=config["r"]
    )
    latent_metrics, latent_details = evaluate_full(
        latent_model, val_u0, val_trajs,
        tau=config["tau"], rollout_steps=config["rollout_steps"],
        device=device, model_name="LatentSemigroupNet",
        reference_dt=config["dt_pde"],
        physical_energy_fn=physical_energy,
        ode_substep_counts=config["ode_substep_counts"],
        collect_latent_diagnostics=config["collect_latent_diagnostics"],
    )
    latent_metrics["beta_V_floor"] = config["beta_V_floor"]
    latent_metrics["beta_V_effective"] = float(
        latent_model.V_net.effective_beta.detach().item()
    )

    for k, v in latent_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4e}")

    print("\n--- BaselineResNet ---")
    baseline_metrics, baseline_details = evaluate_full(
        baseline_model, val_u0, val_trajs,
        tau=config["tau"], rollout_steps=config["rollout_steps"],
        device=device, model_name="BaselineResNet",
        reference_dt=config["dt_pde"],
        physical_energy_fn=physical_energy,
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
