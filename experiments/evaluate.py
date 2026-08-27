"""Evaluation metrics for structure-preserving semigroup learners.

The evaluator distinguishes rollout accuracy, numerical semigroup defects,
learned-energy monotonicity, and optional physical-energy monotonicity.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_PHASE0_ODE_STEPS = (5, 10, 15, 30, 60, 120)


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


def _latent_norm_snapshot(model, u, *, tau=None):
    """Return latent-state and vector-field norms for every batch item.

    Most latent models expose an autonomous ``latent_dynamics(z)``.  A
    query-time-conditioned control can instead explicitly advertise that its
    dynamics require the evaluated ``tau``.  Passing that query here keeps the
    diagnostic honest without changing the autonomous-model API.
    """
    encode = getattr(model, "encode", None)
    dynamics = getattr(model, "latent_dynamics", None)
    if not callable(encode) or not callable(dynamics):
        return None
    z = encode(u)
    if getattr(model, "latent_dynamics_requires_tau", False):
        if tau is None:
            return None
        dz = dynamics(z, tau=tau)
    else:
        dz = dynamics(z)
    z_flat = z.reshape(z.shape[0], -1)
    dz_flat = dz.reshape(dz.shape[0], -1)
    return {
        "latent_l2": torch.linalg.vector_norm(z_flat, dim=1),
        "latent_abs": z_flat.abs().max(dim=1).values,
        "vector_field_l2": torch.linalg.vector_norm(dz_flat, dim=1),
    }


def normalize_ode_steps(substep_counts):
    """Normalize and validate an RK4 ``ode_steps`` sweep.

    The helper is public so the Phase-0 CLI and tests use exactly the same
    validation rules.  Counts are returned in increasing order with
    duplicates removed.  Non-integral values are rejected instead of being
    silently truncated by ``int``.
    """
    try:
        raw_counts = list(substep_counts)
    except TypeError as exc:
        raise ValueError("substep_counts must be an iterable of positive integers") from exc
    if not raw_counts:
        raise ValueError("substep_counts must contain at least one positive integer")

    counts = set()
    for raw_count in raw_counts:
        try:
            numeric_count = float(raw_count)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"ode_steps must be a positive integer, got {raw_count!r}"
            ) from exc
        if not np.isfinite(numeric_count) or numeric_count <= 0:
            raise ValueError(
                f"ode_steps must be a positive integer, got {raw_count!r}"
            )
        count = int(numeric_count)
        if numeric_count != count:
            raise ValueError(
                f"ode_steps must be a positive integer, got {raw_count!r}"
            )
        counts.add(count)
    return tuple(sorted(counts))


def _empty_semigroup_defect(status, reason, *, semantics, base_ode_steps=None):
    """Create a schema-stable empty semigroup diagnostic."""
    return {
        "status": status,
        "reason": reason,
        "semantics": semantics,
        "base_ode_steps": base_ode_steps,
        "n_pairs": 0,
        "mse": float("nan"),
        "absolute_l2": float("nan"),
        "relative_l2": float("nan"),
    }


def _iter_semigroup_pairs(steps, max_pairs):
    """Yield the same bounded, non-zero-time pair schedule used by production."""
    steps = int(steps)
    pair_stride = max(1, steps // max_pairs)
    n_pairs = 0
    for i in range(1, steps, pair_stride):
        for j in range(1, min(4, steps - i + 1)):
            if i + j > steps:
                continue
            yield i, j
            n_pairs += 1
            if n_pairs >= max_pairs:
                return


def _forward_at_ode_steps(model, u, tau, ode_steps):
    """Run one model call at a temporary ``ode_steps`` value and restore it."""
    if not hasattr(model, "ode_steps"):
        raise AttributeError("model does not expose ode_steps")
    original_steps = model.ode_steps
    model.ode_steps = int(ode_steps)
    try:
        return model(u, tau)
    finally:
        model.ode_steps = original_steps


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
    """Measure the production fixed-``ode_steps`` composition error.

    This is the historical diagnostic.  Every call to ``model`` receives its
    requested physical time while the model's own fixed ``ode_steps`` value
    is left unchanged.  Consequently, a direct ``(i + j) * tau`` call uses
    ``model.ode_steps`` RK4 steps, whereas the composed path uses twice that
    amount.  It is retained for backward compatibility and is intentionally
    distinct from :func:`compute_equal_work_semigroup_defect`.
    """
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if relative_eps <= 0:
        raise ValueError("relative_eps must be positive")
    steps = int(steps)
    if steps < 2:
        return _empty_semigroup_defect(
            "not_evaluated",
            "at least two rollout steps are required",
            semantics="production_fixed_ode_steps",
        )

    u0 = u0.to(device)
    mse_values = []
    abs_values = []
    rel_values = []
    n_pairs = 0
    # Start at one model step so the diagnostic does not spend pairs testing
    # only the numerical identity at t=0.
    for i, j in _iter_semigroup_pairs(steps, max_pairs):
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

    if not mse_values:
        return _empty_semigroup_defect(
            "not_evaluated",
            "no valid composition pairs",
            semantics="production_fixed_ode_steps",
        )

    per_sample_mse = torch.stack(mse_values, dim=0).mean(dim=0)
    per_sample_abs = torch.stack(abs_values, dim=0).mean(dim=0)
    per_sample_rel = torch.stack(rel_values, dim=0).mean(dim=0)
    return {
        "status": "evaluated",
        "semantics": "production_fixed_ode_steps",
        "base_ode_steps": None,
        "n_pairs": n_pairs,
        "mse": float(per_sample_mse.mean().item()),
        "absolute_l2": float(per_sample_abs.mean().item()),
        "relative_l2": float(per_sample_rel.mean().item()),
        "per_sample_mse": per_sample_mse.detach().cpu().tolist(),
        "per_sample_absolute_l2": per_sample_abs.detach().cpu().tolist(),
        "per_sample_relative_l2": per_sample_rel.detach().cpu().tolist(),
    }


@torch.no_grad()
def compute_equal_work_semigroup_defect(
    model,
    u0,
    tau,
    steps,
    device="cpu",
    *,
    base_ode_steps=None,
    max_pairs=20,
    relative_eps=1e-12,
):
    """Measure composition error with an equal RK4 base step/work budget.

    ``base_ode_steps`` is the number of RK4 substeps used for one model
    interval ``tau``.  A pair ``(i, j)`` therefore evaluates the direct path
    with ``(i + j) * base_ode_steps`` substeps, and the composed path with
    ``i * base_ode_steps`` followed by ``j * base_ode_steps`` substeps.  Both
    paths use exactly ``(i + j) * base_ode_steps`` RK4 substeps in total.

    The model's original ``ode_steps`` value is restored even when a forward
    call raises.  This diagnostic is deliberately separate from
    :func:`compute_numerical_semigroup_defect`, whose production semantics use
    the model's fixed step count for every requested duration.
    """
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if relative_eps <= 0:
        raise ValueError("relative_eps must be positive")
    steps = int(steps)
    if not hasattr(model, "ode_steps"):
        return _empty_semigroup_defect(
            "not_evaluated",
            "model does not expose ode_steps",
            semantics="equal_work_base_step",
            base_ode_steps=base_ode_steps,
        )

    original_steps = model.ode_steps
    if base_ode_steps is None:
        base_ode_steps = original_steps
    normalized_base = normalize_ode_steps((base_ode_steps,))[0]
    base_ode_steps = normalized_base
    if steps < 2:
        return _empty_semigroup_defect(
            "not_evaluated",
            "at least two rollout steps are required",
            semantics="equal_work_base_step",
            base_ode_steps=base_ode_steps,
        )

    u0 = u0.to(device)
    mse_values = []
    abs_values = []
    rel_values = []
    pair_work = []
    n_pairs = 0
    for i, j in _iter_semigroup_pairs(steps, max_pairs):
        direct_steps = (i + j) * base_ode_steps
        first_steps = i * base_ode_steps
        second_steps = j * base_ode_steps

        direct = _forward_at_ode_steps(model, u0, (i + j) * tau, direct_steps)
        first = _forward_at_ode_steps(model, u0, i * tau, first_steps)
        composed = _forward_at_ode_steps(model, first, j * tau, second_steps)
        diff = composed - direct

        batch_size = diff.shape[0]
        mse_values.append(diff.reshape(batch_size, -1).square().mean(dim=1))
        diff_norm = torch.linalg.vector_norm(diff.reshape(batch_size, -1), dim=1)
        direct_norm = torch.linalg.vector_norm(
            direct.reshape(direct.shape[0], -1), dim=1
        )
        abs_values.append(diff_norm)
        rel_values.append(diff_norm / torch.clamp(direct_norm, min=relative_eps))
        pair_work.append(
            {
                "i": i,
                "j": j,
                "direct_ode_steps": direct_steps,
                "composed_ode_steps": [first_steps, second_steps],
                "direct_total_ode_steps": direct_steps,
                "composed_total_ode_steps": first_steps + second_steps,
            }
        )
        n_pairs += 1

    # All calls above restore independently; this explicit assignment also
    # protects against future changes that might introduce a non-local call.
    model.ode_steps = original_steps

    if not mse_values:
        return _empty_semigroup_defect(
            "not_evaluated",
            "no valid composition pairs",
            semantics="equal_work_base_step",
            base_ode_steps=base_ode_steps,
        )

    per_sample_mse = torch.stack(mse_values, dim=0).mean(dim=0)
    per_sample_abs = torch.stack(abs_values, dim=0).mean(dim=0)
    per_sample_rel = torch.stack(rel_values, dim=0).mean(dim=0)
    return {
        "status": "evaluated",
        "semantics": "equal_work_base_step",
        "base_ode_steps": base_ode_steps,
        "n_pairs": n_pairs,
        "mse": float(per_sample_mse.mean().item()),
        "absolute_l2": float(per_sample_abs.mean().item()),
        "relative_l2": float(per_sample_rel.mean().item()),
        "pair_work": pair_work,
        "per_sample_mse": per_sample_mse.detach().cpu().tolist(),
        "per_sample_absolute_l2": per_sample_abs.detach().cpu().tolist(),
        "per_sample_relative_l2": per_sample_rel.detach().cpu().tolist(),
    }


@torch.no_grad()
def evaluate_ode_steps_sweep(
    model,
    u0,
    tau,
    substep_counts,
    device="cpu",
    *,
    relative_eps=1e-12,
    semigroup_steps=None,
    max_pairs=20,
    include_defects=False,
):
    """Run a no-retraining RK4 ``ode_steps`` convergence audit.

    The same checkpoint/model parameters are reused for every count.  The
    model's original ``ode_steps`` value is restored on success and failure.
    When ``include_defects`` is true, each row also contains the production
    fixed-``ode_steps`` defect and the equal-work base-step defect at that
    count.  ``semigroup_steps`` is required in that mode because defect
    diagnostics need a finite pair horizon.
    """
    if not hasattr(model, "ode_steps"):
        return {
            "status": "not_evaluated",
            "reason": "model does not expose ode_steps",
            "comparisons": [],
            "sweep": [],
        }

    counts = normalize_ode_steps(substep_counts)
    if len(counts) < 2 or counts[0] <= 0:
        raise ValueError("substep_counts must contain at least two positive integers")
    if include_defects and semigroup_steps is None:
        raise ValueError("semigroup_steps is required when include_defects=True")
    if include_defects and max_pairs <= 0:
        raise ValueError("max_pairs must be positive")

    original_steps = model.ode_steps
    outputs = {}
    try:
        for count in counts:
            outputs[count] = _forward_at_ode_steps(
                model, u0.to(device), tau, count
            ).detach().clone()

        finest = counts[-1]
        reference = outputs[finest]
        reference_norm = torch.linalg.vector_norm(
            reference.reshape(reference.shape[0], -1), dim=1
        )
        comparisons = []
        for count in counts:
            diff = outputs[count] - reference
            diff_norm = torch.linalg.vector_norm(
                diff.reshape(diff.shape[0], -1), dim=1
            )
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

        sweep = []
        for comparison in comparisons:
            row = dict(comparison)
            if include_defects:
                count = comparison["ode_steps"]
                current_steps = model.ode_steps
                model.ode_steps = count
                try:
                    production_defect = compute_numerical_semigroup_defect(
                        model,
                        u0,
                        tau,
                        semigroup_steps,
                        device=device,
                        max_pairs=max_pairs,
                        relative_eps=relative_eps,
                    )
                finally:
                    model.ode_steps = current_steps
                equal_work_defect = compute_equal_work_semigroup_defect(
                    model,
                    u0,
                    tau,
                    semigroup_steps,
                    device=device,
                    base_ode_steps=count,
                    max_pairs=max_pairs,
                    relative_eps=relative_eps,
                )
                row["production_fixed_ode_steps_defect"] = production_defect
                row["equal_work_base_step_defect"] = equal_work_defect
            sweep.append(row)
    finally:
        model.ode_steps = original_steps

    return {
        "status": "evaluated",
        "original_ode_steps": int(original_steps),
        "ode_steps": list(counts),
        "reference_ode_steps": counts[-1],
        "semantics": (
            "same checkpoint reused; only ode_steps changes temporarily; no retraining"
        ),
        "comparisons": comparisons,
        "sweep": sweep,
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
    """Backward-compatible wrapper for the original convergence API."""
    audit = evaluate_ode_steps_sweep(
        model,
        u0,
        tau,
        substep_counts,
        device=device,
        relative_eps=relative_eps,
        include_defects=False,
    )
    return {
        "status": audit["status"],
        **({"reason": audit["reason"]} if "reason" in audit else {}),
        **(
            {
                "original_ode_steps": audit["original_ode_steps"],
                "reference_ode_steps": audit["reference_ode_steps"],
                "comparisons": audit["comparisons"][:-1],
                "semantics": audit["semantics"],
            }
            if audit["status"] == "evaluated"
            else {"comparisons": []}
        ),
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
    equal_work_base_ode_steps=None,
):
    """Evaluate rollouts with strict physical-time alignment.

    Existing ``numerical_semigroup_defect_*`` fields retain their production
    fixed-``ode_steps`` meaning.  Passing ``equal_work_base_ode_steps`` adds a
    separate ``equal_work_semigroup_defect_*`` family whose direct and
    composed paths use the same total RK4 work.
    """
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
    per_step_rel_l2 = np.zeros(rollout_steps, dtype=float)
    per_step_viol = np.zeros(rollout_steps, dtype=float)
    per_step_bound_max = np.zeros(rollout_steps, dtype=float)
    per_step_counts = np.zeros(rollout_steps, dtype=int)
    rollout_mse = []
    rollout_rel_l2 = []
    bound_viol = []
    bound_viol_max = []
    learned_energy_mono = []
    physical_energy_mono = []
    sg_mse = []
    sg_abs = []
    sg_rel = []
    equal_work_sg_mse = []
    equal_work_sg_abs = []
    equal_work_sg_rel = []
    latent_l2_maxima = []
    latent_abs_maxima = []
    vector_field_l2_maxima = []
    learned_energy_positive_increments = []
    physical_energy_positive_increments = []
    learned_positive_per_step_sum = np.zeros(rollout_steps, dtype=float)
    physical_positive_per_step_sum = np.zeros(rollout_steps, dtype=float)
    learned_positive_per_step_max = np.full(rollout_steps, np.nan, dtype=float)
    physical_positive_per_step_max = np.full(rollout_steps, np.nan, dtype=float)
    energy_per_step_counts = np.zeros(rollout_steps, dtype=int)
    has_learned_energy = callable(getattr(model, "energy", None))
    equal_work_requested = equal_work_base_ode_steps is not None
    if equal_work_requested:
        equal_work_base_ode_steps = normalize_ode_steps(
            (equal_work_base_ode_steps,)
        )[0]

    for rollout_batch in rollout_batches:
        steps = rollout_batch["steps"]
        u0 = rollout_batch["initial_states"].to(device)
        references = rollout_batch["references"].to(device)
        batch_size = u0.shape[0]
        u_pred = u0
        sample_mse = torch.zeros(batch_size, dtype=torch.float64, device=device)
        sample_rel_l2 = torch.zeros(batch_size, dtype=torch.float64, device=device)
        sample_viol = torch.zeros(batch_size, dtype=torch.float64, device=device)
        sample_bound_max = torch.zeros(batch_size, dtype=torch.float64, device=device)
        learned_ok = torch.zeros(batch_size, dtype=torch.int64, device=device)
        physical_ok = torch.zeros(batch_size, dtype=torch.int64, device=device)
        sample_latent = None
        if collect_latent_diagnostics:
            snapshot = _latent_norm_snapshot(model, u_pred, tau=tau)
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
                current_latent = _latent_norm_snapshot(model, u_pred, tau=tau)
                if current_latent is not None:
                    if sample_latent is None:
                        sample_latent = current_latent
                    else:
                        for key in sample_latent:
                            sample_latent[key] = torch.maximum(
                                sample_latent[key], current_latent[key]
                            )
            u_ref = references[step]
            flat_diff = (u_pred - u_ref).reshape(batch_size, -1)
            flat_ref = u_ref.reshape(batch_size, -1)
            step_mse = flat_diff.square().mean(dim=1)
            step_rel_l2 = torch.linalg.vector_norm(flat_diff, dim=1) / torch.clamp(
                torch.linalg.vector_norm(flat_ref, dim=1), min=1e-12
            )
            step_viol = F.relu(lower_bound - u_pred).reshape(batch_size, -1).mean(dim=1)
            step_viol = step_viol + F.relu(u_pred - upper_bound).reshape(
                batch_size, -1
            ).mean(dim=1)
            step_bound_max = torch.maximum(
                F.relu(lower_bound - u_pred).reshape(batch_size, -1).max(dim=1).values,
                F.relu(u_pred - upper_bound).reshape(batch_size, -1).max(dim=1).values,
            )
            sample_mse += step_mse
            sample_rel_l2 += step_rel_l2
            sample_viol += step_viol
            sample_bound_max = torch.maximum(sample_bound_max, step_bound_max)
            per_step_mse[step] += float(step_mse.sum().item())
            per_step_rel_l2[step] += float(step_rel_l2.sum().item())
            per_step_viol[step] += float(step_viol.sum().item())
            per_step_bound_max[step] = max(
                per_step_bound_max[step], float(step_bound_max.max().item())
            )
            per_step_counts[step] += batch_size

            if learned_previous is not None:
                learned_current = evaluate_per_sample(model.energy, u_pred)
                learned_delta = learned_current - learned_previous
                learned_positive = torch.clamp_min(learned_delta, 0.0)
                learned_ok += learned_delta <= 1e-6
                learned_energy_positive_increments.extend(
                    learned_positive.detach().cpu().tolist()
                )
                learned_positive_per_step_sum[step] += float(
                    learned_positive.sum().item()
                )
                learned_positive_per_step_max[step] = np.nanmax(
                    [
                        learned_positive_per_step_max[step],
                        float(learned_positive.max().item()),
                    ]
                )
                learned_previous = learned_current
            if physical_previous is not None:
                physical_current = evaluate_per_sample(physical_energy_fn, u_pred)
                physical_delta = physical_current - physical_previous
                physical_positive = torch.clamp_min(physical_delta, 0.0)
                physical_ok += physical_delta <= 1e-6
                physical_energy_positive_increments.extend(
                    physical_positive.detach().cpu().tolist()
                )
                physical_positive_per_step_sum[step] += float(
                    physical_positive.sum().item()
                )
                physical_positive_per_step_max[step] = np.nanmax(
                    [
                        physical_positive_per_step_max[step],
                        float(physical_positive.max().item()),
                    ]
                )
                physical_previous = physical_current

            if learned_previous is not None or physical_previous is not None:
                energy_per_step_counts[step] += batch_size

        rollout_mse.extend((sample_mse / steps).detach().cpu().tolist())
        rollout_rel_l2.extend((sample_rel_l2 / steps).detach().cpu().tolist())
        bound_viol.extend((sample_viol / steps).detach().cpu().tolist())
        bound_viol_max.extend(sample_bound_max.detach().cpu().tolist())
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
        if equal_work_requested:
            equal_work_defect = compute_equal_work_semigroup_defect(
                model,
                u0,
                tau,
                steps,
                device=device,
                base_ode_steps=equal_work_base_ode_steps,
            )
            if equal_work_defect["status"] == "evaluated":
                equal_work_sg_mse.extend(equal_work_defect["per_sample_mse"])
                equal_work_sg_abs.extend(equal_work_defect["per_sample_absolute_l2"])
                equal_work_sg_rel.extend(equal_work_defect["per_sample_relative_l2"])

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
        "rollout_rel_l2_mean": float(np.mean(rollout_rel_l2)),
        "rollout_rel_l2_std": float(np.std(rollout_rel_l2)),
        "bound_viol_mean": float(np.mean(bound_viol)),
        "bound_viol_std": float(np.std(bound_viol)),
        "bound_viol_max": float(np.max(bound_viol_max)),
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
        "numerical_semigroup_defect_semantics": (
            "production_fixed_ode_steps: every direct/composed model call uses "
            "model.ode_steps; a two-call composition therefore receives twice "
            "the RK4 work of its direct counterpart"
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
        "learned_energy_positive_increment_semantics": (
            "max(E[k+1] - E[k], 0) per valid sample-step transition; "
            "computed without the 1e-6 monotonicity tolerance"
        ),
        "physical_energy_positive_increment_semantics": (
            "max(E_physical[k+1] - E_physical[k], 0) per valid sample-step "
            "transition; computed without the 1e-6 monotonicity tolerance"
        ),
        "equal_work_semigroup_defect_status": (
            "not_requested" if not equal_work_requested else (
                "evaluated" if equal_work_sg_mse else "not_evaluated"
            )
        ),
        "equal_work_semigroup_defect_semantics": (
            "equal_work_base_step: for pair (i,j), direct uses "
            "(i+j)*base_ode_steps RK4 substeps and composition uses "
            "i*base_ode_steps followed by j*base_ode_steps; total work matches"
        ),
        "equal_work_semigroup_defect_base_ode_steps": (
            None if equal_work_base_ode_steps is None else int(equal_work_base_ode_steps)
        ),
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
    if learned_energy_positive_increments:
        metrics["learned_energy_positive_increment_mean"] = float(
            np.mean(learned_energy_positive_increments)
        )
        metrics["learned_energy_positive_increment_max"] = float(
            np.max(learned_energy_positive_increments)
        )
    if physical_energy_positive_increments:
        metrics["physical_energy_positive_increment_mean"] = float(
            np.mean(physical_energy_positive_increments)
        )
        metrics["physical_energy_positive_increment_max"] = float(
            np.max(physical_energy_positive_increments)
        )
    if equal_work_requested:
        metrics["equal_work_semigroup_defect_mse_mean"] = mean_or_nan(
            equal_work_sg_mse
        )
        metrics["equal_work_semigroup_defect_mse_std"] = std_or_nan(
            equal_work_sg_mse
        )
        metrics["equal_work_semigroup_defect_abs_l2_mean"] = mean_or_nan(
            equal_work_sg_abs
        )
        metrics["equal_work_semigroup_defect_abs_l2_std"] = std_or_nan(
            equal_work_sg_abs
        )
        metrics["equal_work_semigroup_defect_rel_l2_mean"] = mean_or_nan(
            equal_work_sg_rel
        )
        metrics["equal_work_semigroup_defect_rel_l2_std"] = std_or_nan(
            equal_work_sg_rel
        )
    if latent_l2_maxima:
        metrics["latent_l2_norm_max"] = float(np.max(latent_l2_maxima))
        metrics["latent_abs_max"] = float(np.max(latent_abs_maxima))
        metrics["latent_vector_field_l2_norm_max"] = float(
            np.max(vector_field_l2_maxima)
        )

    valid_mask = per_step_counts > 0
    per_step_mse[valid_mask] /= per_step_counts[valid_mask]
    per_step_rel_l2[valid_mask] /= per_step_counts[valid_mask]
    per_step_viol[valid_mask] /= per_step_counts[valid_mask]
    learned_positive_per_step = np.full(rollout_steps, np.nan, dtype=float)
    physical_positive_per_step = np.full(rollout_steps, np.nan, dtype=float)
    energy_valid_mask = energy_per_step_counts > 0
    learned_positive_per_step[energy_valid_mask] = (
        learned_positive_per_step_sum[energy_valid_mask]
        / energy_per_step_counts[energy_valid_mask]
    )
    physical_positive_per_step[energy_valid_mask] = (
        physical_positive_per_step_sum[energy_valid_mask]
        / energy_per_step_counts[energy_valid_mask]
    )
    if ode_substep_counts is not None:
        metrics["ode_substep_convergence"] = evaluate_ode_substep_convergence(
            model, val_u0[:1], tau, ode_substep_counts, device=device
        )
    if hasattr(model, "ode_steps"):
        metrics["ode_steps"] = int(model.ode_steps)

    details = {
        "per_step_mse": per_step_mse.tolist(),
        "per_step_rel_l2": per_step_rel_l2.tolist(),
        "per_step_viol": per_step_viol.tolist(),
        "per_step_bound_max": per_step_bound_max.tolist(),
        "per_step_counts": per_step_counts.tolist(),
        "evaluated_steps": evaluated_steps,
        "energy_positive_increment_semantics": (
            "per-step mean of max(E[k+1] - E[k], 0) over valid sample transitions; "
            "NaN marks an unavailable step"
        ),
    }
    if learned_energy_positive_increments:
        details["learned_energy_positive_increment_per_step"] = (
            learned_positive_per_step.tolist()
        )
        details["learned_energy_max_positive_increment_per_step"] = (
            learned_positive_per_step_max.tolist()
        )
    if physical_energy_positive_increments:
        details["physical_energy_positive_increment_per_step"] = (
            physical_positive_per_step.tolist()
        )
        details["physical_energy_max_positive_increment_per_step"] = (
            physical_positive_per_step_max.tolist()
        )
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
