#!/usr/bin/env python3
"""
Evaluate all trained models (Latent, Baseline, FNO) across all PDE benchmarks.
Loads existing checkpoints, evaluates, and saves consolidated results.
"""
import torch
import numpy as np
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import LatentSemigroupNet, LatentSemigroupNetBounded, BaselineResNet, FNOBaseline
from evaluate import evaluate_full
from seed_utils import set_global_seed


def periodic_gradient(u, domain_length):
    dx = float(domain_length) / u.shape[-1]
    return (torch.roll(u, shifts=-1, dims=-1) - u) / dx, dx


def fisher_kpp_energy(u):
    grad, dx = periodic_gradient(u, 10.0)
    density = 0.5 * 0.1 * grad.square() - (0.5 * u.square() - u.pow(3) / 3.0)
    return dx * density.sum(dim=-1)


def allen_cahn_energy(u):
    grad, dx = periodic_gradient(u, 2.0 * np.pi)
    density = 0.5 * (0.1 ** 2) * grad.square() + 0.25 * (1.0 - u.square()).square()
    return dx * density.sum(dim=-1)


def burgers_l2_energy(u):
    dx = 2.0 * np.pi / u.shape[-1]
    return 0.5 * dx * u.square().sum(dim=-1)


def load_checkpoint(model, path, device):
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    epoch = ckpt.get("epoch", "?")
    mse = ckpt.get("best_val_mse", ckpt.get("val_metrics", {}).get("rollout_mse", "?"))
    print(f"  Loaded {path} (epoch={epoch}, val_mse={mse})")
    return model


def evaluate_fisher_kpp(device):
    """Evaluate all models on Fisher-KPP benchmark."""
    print("\n" + "=" * 70)
    print("Fisher-KPP (N=64, tau=0.1, T_max=2.0)")
    print("=" * 70)

    data = torch.load("checkpoints/data.pt", map_location="cpu", weights_only=False)
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    tau, reference_dt, N = 0.1, 0.005, 64
    results = {}

    # Latent
    if os.path.exists("checkpoints/latent_best.pt"):
        model = LatentSemigroupNet(N=N, hidden_V=[64, 64], hidden_K=[64, 64],
                                    stencil_radius=3, beta_V=0.0)
        model = load_checkpoint(model, "checkpoints/latent_best.pt", device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="Latent",
                                    reference_dt=reference_dt,
                                    physical_energy_fn=fisher_kpp_energy)
        results["latent"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  Latent MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: latent_best.pt not found")

    # BaselineResNet
    if os.path.exists("checkpoints/baseline_best.pt"):
        model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
        model = load_checkpoint(model, "checkpoints/baseline_best.pt", device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="BaselineResNet",
                                    lower_bound=0.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=fisher_kpp_energy)
        results["baseline"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  Baseline MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: baseline_best.pt not found")

    # FNO (trained with width=16 in run_fno_comparison.py)
    if os.path.exists("checkpoints/fno/fno_best.pt"):
        model = FNOBaseline(N=N, width=16, n_modes=16, n_layers=4)
        model = load_checkpoint(model, "checkpoints/fno/fno_best.pt", device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="FNO",
                                    lower_bound=0.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=fisher_kpp_energy)
        results["fno"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  FNO MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: fno/fno_best.pt not found")

    return results


def evaluate_allen_cahn(device):
    """Evaluate all models on Allen-Cahn benchmark."""
    print("\n" + "=" * 70)
    print("Allen-Cahn (N=64, tau=0.1, T_max=2.0)")
    print("=" * 70)

    data_path = "checkpoints/allen_cahn/data_N64.pt"
    if not os.path.exists(data_path):
        print(f"  ERROR: Data file not found: {data_path}")
        return {}

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    # Allen--Cahn validation data are stored every model step, not every
    # internal reference-solver step.
    tau, reference_dt, N = 0.1, 0.1, 64
    checkpoint_dir = "checkpoints/allen_cahn"
    results = {}
    ckpt_path = "checkpoints/allen_cahn_new/latent_N64_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/allen_cahn/latent_N64_best.pt"
    if os.path.exists(ckpt_path):
        model = LatentSemigroupNetBounded(N=N, m=-1.0, M=1.0, hidden_V=[64, 64],
                                            hidden_K=[64, 64], stencil_radius=3, beta_V=0.0)
        model = load_checkpoint(model, ckpt_path, device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="Latent",
                                    lower_bound=-1.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=allen_cahn_energy)
        results["latent"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  Latent MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: No latent checkpoint found")

    # BaselineResNet (trained as baseline_N64_best.pt by run_allen_cahn_baselines.py)
    ckpt_path = os.path.join(checkpoint_dir, f"baseline_N{N}_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/allen_cahn/baseline_resnet_best.pt"
    if os.path.exists(ckpt_path):
        model = BaselineResNet(N=N, hidden_dim=32, n_blocks=4)
        model = load_checkpoint(model, ckpt_path, device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="BaselineResNet",
                                    lower_bound=-1.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=allen_cahn_energy)
        results["baseline"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  Baseline MSE: {metrics['rollout_mse_mean']:.4e}")
    else:
        print("  MISSING: baseline — needs training")

    # FNO (trained as fno_N64_best.pt by run_allen_cahn_baselines.py)
    ckpt_path = os.path.join(checkpoint_dir, f"fno_N{N}_best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/allen_cahn/fno_best.pt"
    if os.path.exists(ckpt_path):
        model = FNOBaseline(N=N, width=16, n_modes=8, n_layers=4)
        model = load_checkpoint(model, ckpt_path, device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="FNO",
                                    lower_bound=-1.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=allen_cahn_energy)
        results["fno"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  FNO MSE: {metrics['rollout_mse_mean']:.4e}")
    else:
        print("  MISSING: fno — needs training")

    return results


def evaluate_burgers(device):
    """Evaluate all models on Burgers benchmark."""
    print("\n" + "=" * 70)
    print("Burgers (N=64, tau=0.05, T_max=1.0)")
    print("=" * 70)

    data_path = "checkpoints/burgers/data_N64.pt"
    if not os.path.exists(data_path):
        print(f"  ERROR: Data file not found: {data_path}")
        return {}

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    val_u0, val_trajs = data["val_u0"], data["val_trajs"]
    # Burgers validation data are likewise stored every tau.
    tau, reference_dt, N = 0.05, 0.05, 64
    results = {}

    # Latent
    ckpt_path = "checkpoints/burgers/latent_N64_best.pt"
    if os.path.exists(ckpt_path):
        model = LatentSemigroupNetBounded(N=N, m=-1.0, M=1.0, hidden_V=[64, 64],
                                            hidden_K=[64, 64], stencil_radius=3, beta_V=0.0)
        model = load_checkpoint(model, ckpt_path, device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="Latent",
                                    lower_bound=-1.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=burgers_l2_energy)
        results["latent"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  Latent MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: latent_N64_best.pt not found")

    # BaselineResNet (trained with hidden_dim=64, n_blocks=4)
    ckpt_path = "checkpoints/burgers/baseline_resnet_best.pt"
    if os.path.exists(ckpt_path):
        model = BaselineResNet(N=N, hidden_dim=64, n_blocks=4)
        model = load_checkpoint(model, ckpt_path, device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="BaselineResNet",
                                    lower_bound=-1.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=burgers_l2_energy)
        results["baseline"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  Baseline MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: baseline_resnet_best.pt not found")

    # FNO (trained with width=32, n_modes=8)
    ckpt_path = "checkpoints/burgers/fno_best.pt"
    if os.path.exists(ckpt_path):
        model = FNOBaseline(N=N, width=32, n_modes=8, n_layers=4)
        model = load_checkpoint(model, ckpt_path, device)
        metrics, _ = evaluate_full(model, val_u0, val_trajs, tau=tau,
                                    rollout_steps=20, device=device, model_name="FNO",
                                    lower_bound=-1.0, upper_bound=1.0,
                                    reference_dt=reference_dt,
                                    physical_energy_fn=burgers_l2_energy)
        results["fno"] = {"n_params": sum(p.numel() for p in model.parameters()), **metrics}
        print(f"  FNO MSE: {metrics['rollout_mse_mean']:.4e} ± {metrics['rollout_mse_std']:.4e}")
    else:
        print("  WARNING: fno_best.pt not found")

    return results


def main():
    set_global_seed(42, deterministic=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {"metadata": {"seed": 42}}

    # Fisher-KPP
    all_results["fisher_kpp"] = evaluate_fisher_kpp(device)

    # Allen-Cahn
    all_results["allen_cahn"] = evaluate_allen_cahn(device)

    # Burgers
    all_results["burgers"] = evaluate_burgers(device)

    # Save consolidated results
    os.makedirs("results", exist_ok=True)
    out_path = "results/all_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved consolidated results to {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for pde_name, pde_results in all_results.items():
        if pde_name == "metadata":
            continue
        print(f"\n{pde_name}:")
        if not pde_results:
            print("  (no results)")
            continue
        for model_name, m in pde_results.items():
            mse = m.get("rollout_mse_mean", float("nan"))
            std = m.get("rollout_mse_std", float("nan"))
            bv = m.get("bound_viol_mean", float("nan"))
            sd = m.get("semigroup_defect_mean", float("nan"))
            print(f"  {model_name:<20} MSE={mse:.4e}±{std:.4e}  BV={bv:.2e}  SD={sd:.2e}")


if __name__ == "__main__":
    main()
