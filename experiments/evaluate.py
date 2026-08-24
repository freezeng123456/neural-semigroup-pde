"""Evaluation metrics for structure-preserving semigroup learners.

The evaluator distinguishes rollout accuracy, numerical semigroup defects,
learned-energy monotonicity, and optional physical-energy monotonicity.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def resolve_reference_stride(tau, reference_dt, *, rtol=1e-7, atol=1e-10):
    """Return the exact stored-snapshot stride for one model step.

    ``reference_dt=None`` preserves the legacy one-snapshot-per-step assumption.
    Otherwise, silent rounding is forbidden.
    """
    tau = float(tau)
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError(f"tau must be a positive finite number, got {tau!r}")
    if reference_dt is None:
        return 1

    reference_dt = float(reference_dt)
    if not np.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError(
            f"reference_dt must be a positive finite number, got {reference_dt!r}"
        )

    ratio = tau / reference_dt
    stride = int(round(ratio))
    if stride < 1 or not np.isclose(ratio, stride, rtol=rtol, atol=atol):
        raise ValueError(
            "model and reference times are not exactly alignable: "
            f"tau/reference_dt={ratio:.17g}, expected a positive integer "
            f"within rtol={rtol:g}, atol={atol:g}"
        )
    return stride


def _as_float(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            value = value.mean()
        return float(value.detach().item())
    return float(value)


def reference_start_time(t_true):
    """Return the start time of a stored reference trajectory."""
    try:
        if len(t_true) == 0:
            raise ValueError("reference trajectory has no timestamps")
        return _as_float(t_true[0])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError("reference trajectory has no usable start timestamp") from exc


def validate_reference_timestamp(t_true, ref_idx, expected_time, *, rtol, atol):
    """Require a stored timestamp to match the requested physical time."""
    try:
        actual_time = _as_float(t_true[ref_idx])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"stored reference timestamp is missing at index {ref_idx}"
        ) from exc
    if not np.isclose(actual_time, expected_time, rtol=rtol, atol=atol):
        raise ValueError(
            "stored reference timestamp does not match the requested model time: "
            f"t_true[{ref_idx}]={actual_time:.17g}, expected {expected_time:.17g}"
        )


def prepare_aligned_rollout_batches(
    val_u0,
    val_trajs,
    tau,
    rollout_steps,
    reference_dt=None,
    *,
    alignment_rtol=1e-7,
    alignment_atol=1e-10,
):
    """Validate references and group compatible trajectories for batched rollout.

    Grouping by both the aligned horizon and state shape preserves support for
    variable-length trajectories without padding model states.  All timestamps
    are validated before inference starts, so batching cannot hide a malformed
    reference trajectory.
    """
    if len(val_u0) != len(val_trajs):
        raise ValueError(
            "validation initial states and trajectories must have the same "
            f"length: got {len(val_u0)} and {len(val_trajs)}"
        )
    rollout_steps = int(rollout_steps)
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")

    reference_stride = resolve_reference_stride(
        tau,
        reference_dt,
        rtol=alignment_rtol,
        atol=alignment_atol,
    )
    grouped = {}
    evaluated_steps = []
    for sample_idx, (t_true, u_true) in enumerate(val_trajs):
        if len(t_true) != len(u_true):
            raise ValueError(
                "reference timestamps and states must have the same length: "
                f"got {len(t_true)} and {len(u_true)} for trajectory {sample_idx}"
            )
        start_time = reference_start_time(t_true)
        steps = min(rollout_steps, (len(t_true) - 1) // reference_stride)
        if steps < 1:
            continue

        references = []
        for step in range(steps):
            ref_idx = (step + 1) * reference_stride
            expected_time = start_time + (step + 1) * float(tau)
            validate_reference_timestamp(
                t_true,
                ref_idx,
                expected_time,
                rtol=alignment_rtol,
                atol=alignment_atol,
            )
            reference_state = u_true[ref_idx]
            if tuple(reference_state.shape) != tuple(val_u0[sample_idx].shape):
                raise ValueError(
                    "validation initial state and reference state shapes must match: "
                    f"got {tuple(val_u0[sample_idx].shape)} and "
                    f"{tuple(reference_state.shape)} for trajectory {sample_idx}"
                )
            references.append(reference_state)

        state_shape = tuple(val_u0[sample_idx].shape)
        key = (steps, state_shape)
        group = grouped.setdefault(
            key,
            {"steps": steps, "indices": [], "references": []},
        )
        group["indices"].append(sample_idx)
        group["references"].append(torch.stack(references, dim=0))
        evaluated_steps.append((sample_idx, steps))

    batches = []
    for group in grouped.values():
        indices = group["indices"]
        batches.append(
            {
                "steps": group["steps"],
                "indices": indices,
                "initial_states": val_u0[indices],
                # (batch, steps, *state_shape) -> (steps, batch, *state_shape)
                "references": torch.stack(group["references"], dim=0).transpose(0, 1),
            }
        )
    batches.sort(key=lambda group: group["indices"][0])
    evaluated_steps.sort()
    return reference_stride, batches, [steps for _, steps in evaluated_steps]


def evaluate_per_sample(function, batch):
    """Evaluate a scalar functional and return one value per batch item.

    Well-vectorized functionals return a leading batch dimension.  A scalar
    result for a multi-item batch is ambiguous, so this helper falls back to
    per-item calls for compatibility with older energy implementations.
    """
    value = function(batch)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=batch.device)
    batch_size = batch.shape[0]
    if value.ndim > 0 and value.shape[0] == batch_size:
        return value.reshape(batch_size, -1).mean(dim=1)
    if value.numel() == batch_size:
        return value.reshape(batch_size)
    if value.numel() == 1 and batch_size == 1:
        return value.reshape(1)
    if value.numel() == 1:
        values = []
        for sample_idx in range(batch_size):
            sample_value = function(batch[sample_idx : sample_idx + 1])
            if not torch.is_tensor(sample_value):
                sample_value = torch.as_tensor(sample_value, device=batch.device)
            if sample_value.numel() != 1:
                raise ValueError("functional must return one scalar per sample")
            values.append(sample_value.reshape(()))
        return torch.stack(values)
    raise ValueError("functional must return one scalar per sample")


def _latent_norm_snapshot(model, u):
    """Return latent-state and vector-field norms for every batch item."""
    encode = getattr(model, "encode", None)
    dynamics = getattr(model, "latent_dynamics", None)
    if not callable(encode) or not callable(dynamics):
        return None
    z = encode(u)
    dz = dynamics(z)
    z_flat = z.reshape(z.shape[0], -1)
    dz_flat = dz.reshape(dz.shape[0], -1)
    return {
        "latent_l2": torch.linalg.vector_norm(z_flat, dim=1),
        "latent_abs": z_flat.abs().max(dim=1).values,
        "vector_field_l2": torch.linalg.vector_norm(dz_flat, dim=1),
    }


@torch.no_grad()
def compute_numerical_semigroup_defect(
    model,
    u0,
    tau,
    steps,
    device="cpu",
    *,
    max_pairs=20,
    relative_eps=1e-12,
):
    """Measure composition error of the model's numerical time integrator."""
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if relative_eps <= 0:
        raise ValueError("relative_eps must be positive")
    if steps < 2:
        return {
            "status": "not_evaluated",
            "reason": "at least two rollout steps are required",
            "n_pairs": 0,
            "mse": float("nan"),
            "absolute_l2": float("nan"),
            "relative_l2": float("nan"),
        }

    u0 = u0.to(device)
    mse_values = []
    abs_values = []
    rel_values = []
    n_pairs = 0
    pair_stride = max(1, steps // max_pairs)

    # Start at one model step so the diagnostic does not spend pairs testing
    # only the numerical identity at t=0.
    for i in range(1, steps, pair_stride):
        for j in range(1, min(4, steps - i + 1)):
            if i + j > steps:
                continue
            direct = model(u0, (i + j) * tau)
            composed = model(model(u0, i * tau), j * tau)
            diff = composed - direct

            batch_size = diff.shape[0]
            mse_values.append(diff.reshape(batch_size, -1).square().mean(dim=1))
            diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
            direct_norm = torch.linalg.vector_norm(
                direct.reshape(direct.shape[0], -1), dim=1
            )
            abs_values.append(diff_norm)
            rel_values.append(diff_norm / torch.clamp(direct_norm, min=relative_eps))
            n_pairs += 1
            if n_pairs >= max_pairs:
                break
        if n_pairs >= max_pairs:
            break

    if not mse_values:
        return {
            "status": "not_evaluated",
            "reason": "no valid composition pairs",
            "n_pairs": 0,
            "mse": float("nan"),
            "absolute_l2": float("nan"),
            "relative_l2": float("nan"),
        }

    per_sample_mse = torch.stack(mse_values, dim=0).mean(dim=0)
    per_sample_abs = torch.stack(abs_values, dim=0).mean(dim=0)
    per_sample_rel = torch.stack(rel_values, dim=0).mean(dim=0)
    return {
        "status": "evaluated",
        "n_pairs": n_pairs,
        "mse": float(per_sample_mse.mean().item()),
        "absolute_l2": float(per_sample_abs.mean().item()),
        "relative_l2": float(per_sample_rel.mean().item()),
        "per_sample_mse": per_sample_mse.detach().cpu().tolist(),
        "per_sample_absolute_l2": per_sample_abs.detach().cpu().tolist(),
        "per_sample_relative_l2": per_sample_rel.detach().cpu().tolist(),
    }


@torch.no_grad()
def evaluate_ode_substep_convergence(
    model,
    u0,
    tau,
    substep_counts,
    device="cpu",
    *,
    relative_eps=1e-12,
):
    """Compare outputs at several ``model.ode_steps`` resolutions."""
    if not hasattr(model, "ode_steps"):
        return {
            "status": "not_evaluated",
            "reason": "model does not expose ode_steps",
            "comparisons": [],
        }

    counts = sorted({int(count) for count in substep_counts})
    if len(counts) < 2 or counts[0] <= 0:
        raise ValueError("substep_counts must contain at least two positive integers")

    original_steps = model.ode_steps
    outputs = {}
    try:
        for count in counts:
            model.ode_steps = count
            outputs[count] = model(u0.to(device), tau).detach().clone()
    finally:
        model.ode_steps = original_steps

    finest = counts[-1]
    reference = outputs[finest]
    reference_norm = torch.linalg.vector_norm(
        reference.reshape(reference.shape[0], -1), dim=1
    )
    comparisons = []
    for count in counts[:-1]:
        diff = outputs[count] - reference
        diff_norm = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
        comparisons.append(
            {
                "ode_steps": count,
                "reference_ode_steps": finest,
                "absolute_l2": torch.mean(diff_norm).item(),
                "relative_l2": torch.mean(
                    diff_norm / torch.clamp(reference_norm, min=relative_eps)
                ).item(),
            }
        )

    return {
        "status": "evaluated",
        "original_ode_steps": int(original_steps),
        "reference_ode_steps": finest,
        "comparisons": comparisons,
    }


@torch.no_grad()
def evaluate_full(
    model,
    val_u0,
    val_trajs,
    tau,
    rollout_steps=20,
    device="cuda",
    model_name="model",
    lower_bound=0.0,
    upper_bound=1.0,
    reference_dt=None,
    physical_energy_fn=None,
    ode_substep_counts=None,
    collect_latent_diagnostics=False,
    alignment_rtol=1e-7,
    alignment_atol=1e-10,
):
    """Evaluate rollouts with strict physical-time alignment."""
    rollout_steps = int(rollout_steps)
    reference_stride, rollout_batches, evaluated_steps = prepare_aligned_rollout_batches(
        val_u0,
        val_trajs,
        tau,
        rollout_steps,
        reference_dt,
        alignment_rtol=alignment_rtol,
        alignment_atol=alignment_atol,
    )
    model.eval()
    per_step_mse = np.zeros(rollout_steps, dtype=float)
    per_step_viol = np.zeros(rollout_steps, dtype=float)
    per_step_counts = np.zeros(rollout_steps, dtype=int)
    rollout_mse = []
    bound_viol = []
    learned_energy_mono = []
    physical_energy_mono = []
    sg_mse = []
    sg_abs = []
    sg_rel = []
    latent_l2_maxima = []
    latent_abs_maxima = []
    vector_field_l2_maxima = []
    has_learned_energy = callable(getattr(model, "energy", None))

    for rollout_batch in rollout_batches:
        steps = rollout_batch["steps"]
        u0 = rollout_batch["initial_states"].to(device)
        references = rollout_batch["references"].to(device)
        batch_size = u0.shape[0]
        u_pred = u0
        sample_mse = torch.zeros(batch_size, dtype=torch.float64, device=device)
        sample_viol = torch.zeros(batch_size, dtype=torch.float64, device=device)
        learned_ok = torch.zeros(batch_size, dtype=torch.int64, device=device)
        physical_ok = torch.zeros(batch_size, dtype=torch.int64, device=device)
        sample_latent = None
        if collect_latent_diagnostics:
            snapshot = _latent_norm_snapshot(model, u_pred)
            if snapshot is not None:
                sample_latent = {key: value.clone() for key, value in snapshot.items()}
        learned_previous = (
            evaluate_per_sample(model.energy, u_pred) if has_learned_energy else None
        )
        physical_previous = (
            evaluate_per_sample(physical_energy_fn, u_pred)
            if physical_energy_fn is not None
            else None
        )
        for step in range(steps):
            u_pred = model(u_pred, tau)
            if collect_latent_diagnostics:
                current_latent = _latent_norm_snapshot(model, u_pred)
                if current_latent is not None:
                    if sample_latent is None:
                        sample_latent = current_latent
                    else:
                        for key in sample_latent:
                            sample_latent[key] = torch.maximum(
                                sample_latent[key], current_latent[key]
                            )
            u_ref = references[step]
            step_mse = (u_pred - u_ref).reshape(batch_size, -1).square().mean(dim=1)
            step_viol = F.relu(lower_bound - u_pred).reshape(batch_size, -1).mean(dim=1)
            step_viol = step_viol + F.relu(u_pred - upper_bound).reshape(
                batch_size, -1
            ).mean(dim=1)
            sample_mse += step_mse
            sample_viol += step_viol
            per_step_mse[step] += float(step_mse.sum().item())
            per_step_viol[step] += float(step_viol.sum().item())
            per_step_counts[step] += batch_size

            if learned_previous is not None:
                learned_current = evaluate_per_sample(model.energy, u_pred)
                learned_ok += learned_current <= learned_previous + 1e-6
                learned_previous = learned_current
            if physical_previous is not None:
                physical_current = evaluate_per_sample(physical_energy_fn, u_pred)
                physical_ok += physical_current <= physical_previous + 1e-6
                physical_previous = physical_current

        rollout_mse.extend((sample_mse / steps).detach().cpu().tolist())
        bound_viol.extend((sample_viol / steps).detach().cpu().tolist())
        if sample_latent is not None:
            latent_l2_maxima.extend(sample_latent["latent_l2"].detach().cpu().tolist())
            latent_abs_maxima.extend(sample_latent["latent_abs"].detach().cpu().tolist())
            vector_field_l2_maxima.extend(
                sample_latent["vector_field_l2"].detach().cpu().tolist()
            )
        if learned_previous is not None:
            learned_energy_mono.extend((learned_ok.float() / steps).cpu().tolist())
        if physical_previous is not None:
            physical_energy_mono.extend((physical_ok.float() / steps).cpu().tolist())

        defect = compute_numerical_semigroup_defect(
            model, u0, tau, steps, device=device
        )
        if defect["status"] == "evaluated":
            sg_mse.extend(defect["per_sample_mse"])
            sg_abs.extend(defect["per_sample_absolute_l2"])
            sg_rel.extend(defect["per_sample_relative_l2"])

    n_valid = len(rollout_mse)
    if n_valid == 0:
        return {
            "error": "no valid trajectories with at least one aligned model step",
            "reference_stride": reference_stride,
        }, {}

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    def std_or_nan(values):
        return float(np.std(values)) if values else float("nan")

    metrics = {
        "model": model_name,
        "tau": float(tau),
        "reference_dt": None if reference_dt is None else float(reference_dt),
        "reference_stride": reference_stride,
        "rollout_mse_mean": float(np.mean(rollout_mse)),
        "rollout_mse_std": float(np.std(rollout_mse)),
        "bound_viol_mean": float(np.mean(bound_viol)),
        "bound_viol_std": float(np.std(bound_viol)),
        "numerical_semigroup_defect_mse_mean": mean_or_nan(sg_mse),
        "numerical_semigroup_defect_mse_std": std_or_nan(sg_mse),
        "numerical_semigroup_defect_abs_l2_mean": mean_or_nan(sg_abs),
        "numerical_semigroup_defect_abs_l2_std": std_or_nan(sg_abs),
        "numerical_semigroup_defect_rel_l2_mean": mean_or_nan(sg_rel),
        "numerical_semigroup_defect_rel_l2_std": std_or_nan(sg_rel),
        "semigroup_defect_mean": mean_or_nan(sg_mse),
        "semigroup_defect_std": std_or_nan(sg_mse),
        "semigroup_defect_semantics": (
            "legacy alias for numerical_semigroup_defect_mse; not an exact-flow guarantee"
        ),
        "numerical_semigroup_defect_status": (
            "evaluated" if sg_mse else "not_evaluated"
        ),
        "numerical_semigroup_relative_denominator": (
            "L2 norm of direct output, clamped below at 1e-12"
        ),
        "learned_energy_status": "evaluated" if learned_energy_mono else "not_evaluated",
        "physical_energy_status": "evaluated" if physical_energy_mono else "not_evaluated",
        "learned_energy_semantics": "model.energy; not assumed to equal physical PDE energy",
        "latent_diagnostics_status": (
            "evaluated" if latent_l2_maxima else "not_evaluated"
        ),
        "n_valid": n_valid,
        "min_evaluated_steps": int(min(evaluated_steps)),
        "max_evaluated_steps": int(max(evaluated_steps)),
    }
    if learned_energy_mono:
        learned_mean = float(np.mean(learned_energy_mono))
        learned_std = float(np.std(learned_energy_mono))
        metrics["learned_energy_mono_frac_mean"] = learned_mean
        metrics["learned_energy_mono_frac_std"] = learned_std
        metrics["energy_mono_frac_mean"] = learned_mean
        metrics["energy_mono_frac_std"] = learned_std
    if physical_energy_mono:
        metrics["physical_energy_mono_frac_mean"] = float(
            np.mean(physical_energy_mono)
        )
        metrics["physical_energy_mono_frac_std"] = float(np.std(physical_energy_mono))
    if latent_l2_maxima:
        metrics["latent_l2_norm_max"] = float(np.max(latent_l2_maxima))
        metrics["latent_abs_max"] = float(np.max(latent_abs_maxima))
        metrics["latent_vector_field_l2_norm_max"] = float(
            np.max(vector_field_l2_maxima)
        )

    valid_mask = per_step_counts > 0
    per_step_mse[valid_mask] /= per_step_counts[valid_mask]
    per_step_viol[valid_mask] /= per_step_counts[valid_mask]
    if ode_substep_counts is not None:
        metrics["ode_substep_convergence"] = evaluate_ode_substep_convergence(
            model, val_u0[:1], tau, ode_substep_counts, device=device
        )
    if hasattr(model, "ode_steps"):
        metrics["ode_steps"] = int(model.ode_steps)

    details = {
        "per_step_mse": per_step_mse.tolist(),
        "per_step_viol": per_step_viol.tolist(),
        "per_step_counts": per_step_counts.tolist(),
        "evaluated_steps": evaluated_steps,
    }
    return metrics, details


@torch.no_grad()
def compute_semigroup_defect(model, u_history, tau, steps, device):
    """Backward-compatible scalar wrapper for the legacy API."""
    return compute_numerical_semigroup_defect(
        model, u_history[0].to(device), tau, steps, device=device
    )["mse"]


def compare_models(latent_metrics, baseline_metrics, output_path):
    """Print a comparison table and save JSON."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<36} {'LatentNet':>15} {'Baseline':>15}")
    print("-" * 70)
    keys = [
        ("rollout_mse_mean", "Rollout MSE"),
        ("bound_viol_mean", "Bound violation"),
        ("numerical_semigroup_defect_abs_l2_mean", "Numerical SG defect (L2)"),
    ]
    if "learned_energy_mono_frac_mean" in latent_metrics:
        keys.append(("learned_energy_mono_frac_mean", "Learned-energy monotonicity"))
    if "physical_energy_mono_frac_mean" in latent_metrics:
        keys.append(("physical_energy_mono_frac_mean", "Physical-energy monotonicity"))

    for key, label in keys:
        latent_value = latent_metrics.get(key, float("nan"))
        baseline_value = baseline_metrics.get(key, float("nan"))
        print(f"{label:<36} {latent_value:>15.4e} {baseline_value:>15.4e}")
    print("=" * 70)

    comparison = {"latent": latent_metrics, "baseline": baseline_metrics}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as output_file:
        json.dump(comparison, output_file, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation module loaded.")
