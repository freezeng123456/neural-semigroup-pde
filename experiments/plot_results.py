"""
Visualization for semigroup learning experiments.

Produces:
    1. Rollout MSE vs time
    2. Bound violation vs time
    3. Energy monotonicity comparison
    4. Semigroup defect comparison
    5. Training curves
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def plot_results(results_dir="results", output_dir="figures"):
    """Plot all experiment results."""
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    eval_path = os.path.join(results_dir, "evaluation_results.pt")
    if not os.path.exists(eval_path):
        print(f"No results found at {eval_path}")
        return

    data = torch.load(eval_path, map_location="cpu", weights_only=False)
    latent_metrics = data["latent_metrics"]
    baseline_metrics = data["baseline_metrics"]
    latent_details = data["latent_details"]
    baseline_details = data["baseline_details"]

    # Load config
    config_path = os.path.join(results_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    tau = config.get("tau", 0.1)
    n_steps = len(latent_details.get("per_step_mse", []))

    # Colors
    c_latent = "#e74c3c"
    c_baseline = "#3498db"

    # ============================================================
    # Figure 1: Rollout MSE vs step
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t = np.arange(1, n_steps + 1) * tau

    # Subplot 1: Rollout MSE
    ax = axes[0]
    ax.semilogy(t, latent_details["per_step_mse"], "o-", color=c_latent,
                label="LatentSemigroupNet", markersize=4)
    ax.semilogy(t, baseline_details["per_step_mse"], "s-", color=c_baseline,
                label="BaselineResNet", markersize=4)
    ax.set_xlabel("Time")
    ax.set_ylabel("MSE")
    ax.set_title("Rollout MSE vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Bound violation
    ax = axes[1]
    ax.plot(t, latent_details["per_step_viol"], "o-", color=c_latent,
            label="LatentSemigroupNet", markersize=4)
    ax.plot(t, baseline_details["per_step_viol"], "s-", color=c_baseline,
            label="BaselineResNet", markersize=4)
    ax.set_xlabel("Time")
    ax.set_ylabel("Bound violation")
    ax.set_title("[0,1] Bound Violation vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Semigroup defect
    ax = axes[2]
    metrics_labels = ["Rollout\nMSE", "Bound\nViolation", "Semigroup\nDefect"]
    x = np.arange(len(metrics_labels))
    width = 0.35

    latent_vals = [
        latent_metrics["rollout_mse_mean"],
        latent_metrics["bound_viol_mean"],
        latent_metrics["semigroup_defect_mean"],
    ]
    baseline_vals = [
        baseline_metrics["rollout_mse_mean"],
        baseline_metrics["bound_viol_mean"],
        baseline_metrics["semigroup_defect_mean"],
    ]

    bars1 = ax.bar(x - width/2, latent_vals, width, color=c_latent, label="LatentNet")
    bars2 = ax.bar(x + width/2, baseline_vals, width, color=c_baseline, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)
    ax.set_ylabel("Value (log scale)")
    ax.set_yscale("log")
    ax.set_title("Aggregate Metrics Comparison")
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "metrics_comparison.png"), dpi=150)
    print(f"Saved: metrics_comparison.png")

    # ============================================================
    # Figure 2: Training curves
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Load training histories
    latent_hist = torch.load(
        os.path.join("checkpoints", "latent_history.pt"),
        map_location="cpu", weights_only=False
    )
    baseline_hist = torch.load(
        os.path.join("checkpoints", "baseline_history.pt"),
        map_location="cpu", weights_only=False
    )

    epochs = latent_hist["epoch"]

    # Subplot 1: Training loss
    ax = axes[0]
    ax.semilogy(epochs, latent_hist["L_step"], color=c_latent, label="LatentNet L_step")
    ax.semilogy(epochs, baseline_hist["L_step"], "--", color=c_baseline, label="Baseline L_step")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Validation MSE
    ax = axes[1]
    ax.semilogy(epochs, latent_hist["val_mse"], color=c_latent, label="LatentNet")
    ax.semilogy(epochs, baseline_hist["val_mse"], "--", color=c_baseline, label="Baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Rollout MSE")
    ax.set_title("Validation MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 3: Bound violation during training
    ax = axes[2]
    ax.semilogy(epochs, latent_hist["val_bound_viol"], color=c_latent, label="LatentNet")
    ax.semilogy(epochs, baseline_hist["val_bound_viol"], "--", color=c_baseline, label="Baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Bound Violation")
    ax.set_title("Validation Bound Violation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    print(f"Saved: training_curves.png")

    # ============================================================
    # Figure 3: Energy monotonicity (if available)
    # ============================================================
    if "energy_mono_frac_mean" in latent_metrics:
        fig, ax = plt.subplots(figsize=(6, 5))
        models = ["LatentSemigroupNet", "BaselineResNet"]
        values = [
            latent_metrics.get("energy_mono_frac_mean", 0),
            baseline_metrics.get("energy_mono_frac_mean", 0),
        ]

        ax.bar(models, values, color=[c_latent, c_baseline])
        ax.set_ylabel("Fraction of steps monotone")
        ax.set_title("Energy Monotonicity")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect")
        ax.legend()

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "energy_monotonicity.png"), dpi=150)
        print(f"Saved: energy_monotonicity.png")

    print(f"\nAll figures saved to {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    plot_results()
