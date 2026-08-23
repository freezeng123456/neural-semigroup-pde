"""
Evaluation metrics for structure-preserving semigroup learners.

Metrics:
    1. Rollout MSE over long horizon
    2. [0,1] bound violation rate
    3. Energy monotonicity
    4. Semigroup defect: ||Phi(u, t+s) - Phi(Phi(u, t), s)||
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
import os


@torch.no_grad()
def evaluate_full(model, val_u0, val_trajs, tau, rollout_steps=20,
                  device="cuda", model_name="model",
                  lower_bound=0.0, upper_bound=1.0,
                  reference_dt=None):
    """
    Comprehensive evaluation.

    Args:
        model: trained model
        val_u0: (N_val, N) initial conditions
        val_trajs: list of (t_true, u_true) from PDE solver
        tau: model rollout time step
        rollout_steps: number of rollout steps for evaluation
        reference_dt: time spacing of stored PDE trajectory snapshots.
            If provided, evaluation uses the matching snapshot index
            round(k*tau/reference_dt) instead of assuming one snapshot per rollout.

    Returns:
        metrics: dict of scalar metrics
        details: dict of per-sample arrays
    """
    model.eval()
    N_val = len(val_u0)
    results = {
        "rollout_mse": [],
        "bound_viol": [],
        "energy_mono_frac": [],
        "semigroup_defect": [],
        "per_step_mse": np.zeros(rollout_steps),
        "per_step_viol": np.zeros(rollout_steps),
    }
    n_valid = 0

    for i in range(N_val):
        u0 = val_u0[i:i+1].to(device)
        t_true, u_true = val_trajs[i]
        u_true = u_true.to(device)
        T = len(t_true) - 1
        steps = min(rollout_steps, T)
        if steps < 2:
            continue
        n_valid += 1

        u_pred = u0
        E0 = model.energy(u_pred).item() if hasattr(model, 'energy') else None
        energy_ok = 0
        mse_total = 0.0
        viol_total = 0.0

        # Per-step rollout
        u_history = [u0.cpu()]

        for step in range(steps):
            u_pred = model(u_pred, tau)
            u_history.append(u_pred.cpu())
            if reference_dt is None:
                ref_idx = step + 1
            else:
                ref_idx = int(round((step + 1) * tau / reference_dt))
            if ref_idx >= len(u_true):
                break
            u_ref = u_true[ref_idx:ref_idx + 1]

            step_mse = torch.mean((u_pred - u_ref) ** 2).item()
            mse_total += step_mse
            viol_step = (
                torch.mean(F.relu(lower_bound - u_pred)) + torch.mean(F.relu(u_pred - upper_bound))
            ).item()
            viol_total += viol_step

            results["per_step_mse"][step] += step_mse
            results["per_step_viol"][step] += viol_step

            if E0 is not None:
                E_new = model.energy(u_pred).item()
                if E_new <= E0 + 1e-6:
                    energy_ok += 1
                E0 = E_new

        results["rollout_mse"].append(mse_total / steps)
        results["bound_viol"].append(viol_total / steps)
        if E0 is not None:
            results["energy_mono_frac"].append(energy_ok / steps)

        # Semigroup defect
        u_history_t = torch.stack(u_history, dim=0).to(device)  # (steps+1, 1, N)
        sg_defect = compute_semigroup_defect(
            model, u_history_t, tau, steps, device
        )
        results["semigroup_defect"].append(sg_defect)

    if n_valid == 0:
        return {"error": "no valid trajectories"}, {}

    # Aggregate
    metrics = {
        "model": model_name,
        "rollout_mse_mean": float(np.mean(results["rollout_mse"])),
        "rollout_mse_std": float(np.std(results["rollout_mse"])),
        "bound_viol_mean": float(np.mean(results["bound_viol"])),
        "bound_viol_std": float(np.std(results["bound_viol"])),
        "semigroup_defect_mean": float(np.mean(results["semigroup_defect"])),
        "semigroup_defect_std": float(np.std(results["semigroup_defect"])),
        "n_valid": n_valid,
    }
    if results["energy_mono_frac"]:
        metrics["energy_mono_frac_mean"] = float(np.mean(results["energy_mono_frac"]))
        metrics["energy_mono_frac_std"] = float(np.std(results["energy_mono_frac"]))

    # Per-step averages
    for step in range(rollout_steps):
        results[f"per_step_mse"][step] /= n_valid
        results[f"per_step_viol"][step] /= n_valid

    # Details for plotting
    details = {
        "per_step_mse": results["per_step_mse"].tolist(),
        "per_step_viol": results["per_step_viol"].tolist(),
    }

    return metrics, details


@torch.no_grad()
def compute_semigroup_defect(model, u_history, tau, steps, device):
    """
    Compute semigroup defect:
        || Phi(u0, t_i + t_j) - Phi(Phi(u0, t_i), t_j) ||
    averaged over pairs.

    Uses the stored history to avoid re-computing.
    """
    defects = []
    # Sample pairs
    max_pairs = 20
    n_pairs = 0

    for i in range(0, steps - 1, max(1, steps // max_pairs)):
        for j in range(1, min(4, steps - i)):
            if i + j >= len(u_history):
                continue
            t_i = i * tau
            t_j = j * tau
            t_sum = (i + j) * tau

            # Phi(Phi(u0, ti), tj)
            u_ti = model(u_history[0].to(device), t_i)
            u_composed = model(u_ti, t_j)

            # Phi(u0, ti + tj)
            u_direct = model(u_history[0].to(device), t_sum)

            defect = torch.mean((u_composed - u_direct) ** 2).item()
            defects.append(defect)
            n_pairs += 1

            if n_pairs >= max_pairs:
                break
        if n_pairs >= max_pairs:
            break

    return float(np.mean(defects)) if defects else 0.0


def compare_models(latent_metrics, baseline_metrics, output_path):
    """Print comparison table and save JSON."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'LatentNet':>15} {'Baseline':>15}")
    print("-" * 70)

    keys = [
        ("rollout_mse_mean", "Rollout MSE"),
        ("bound_viol_mean", "Bound violation"),
        ("semigroup_defect_mean", "Semigroup defect"),
    ]
    if "energy_mono_frac_mean" in latent_metrics:
        keys.append(("energy_mono_frac_mean", "Energy monotonicity"))

    for key, label in keys:
        v_lat = latent_metrics.get(key, float("nan"))
        v_base = baseline_metrics.get(key, float("nan"))
        print(f"{label:<30} {v_lat:>15.4e} {v_base:>15.4e}")

    print("=" * 70)

    # Save JSON
    comparison = {
        "latent": latent_metrics,
        "baseline": baseline_metrics,
    }
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation module loaded.")
