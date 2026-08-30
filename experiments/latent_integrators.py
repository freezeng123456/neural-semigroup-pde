"""Side-effect-free ODE integration for frozen latent-flow checkpoints.

Historical models own an RK4-only ``forward`` method and expose the number of
substeps through mutable ``model.ode_steps`` state.  Exploratory numerical
controls need Euler, midpoint/RK2, and RK4 under exact right-hand-side (RHS)
budgets without mutating the checkpoint model.  This module provides that
single seam.

The same explicit integrator kernel is used by the learned-flow evaluator and
the analytic advection--diffusion calibration.  As a result, work accounting
is executable rather than a label written by the launcher.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F


INTEGRATOR_RHS_EVALUATIONS = {
    "euler": 1,
    "rk2": 2,
    "rk4": 4,
}


def normalize_integrator(name: str) -> str:
    """Return one canonical explicit-integrator name."""

    normalized = str(name).strip().lower()
    aliases = {
        "forward_euler": "euler",
        "midpoint": "rk2",
        "explicit_midpoint": "rk2",
        "classical_rk4": "rk4",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in INTEGRATOR_RHS_EVALUATIONS:
        choices = ", ".join(sorted(INTEGRATOR_RHS_EVALUATIONS))
        raise ValueError(f"unsupported integrator {name!r}; expected one of {choices}")
    return normalized


def normalize_integrators(names: Iterable[str]) -> tuple[str, ...]:
    """Normalize an ordered, duplicate-free integrator collection."""

    result: list[str] = []
    for name in names:
        normalized = normalize_integrator(name)
        if normalized in result:
            raise ValueError(f"integrator list contains duplicate {normalized!r}")
        result.append(normalized)
    if not result:
        raise ValueError("at least one integrator is required")
    return tuple(result)


def rhs_evaluations_per_substep(method: str) -> int:
    """Return the exact RHS calls made by one explicit substep."""

    return INTEGRATOR_RHS_EVALUATIONS[normalize_integrator(method)]


def substeps_for_rhs_budget(method: str, rhs_budget: int) -> int:
    """Convert an exact RHS budget to substeps, rejecting fractional work."""

    budget = int(rhs_budget)
    if budget <= 0:
        raise ValueError("RHS budget must be positive")
    per_step = rhs_evaluations_per_substep(method)
    if budget % per_step:
        raise ValueError(
            f"RHS budget {budget} is not divisible by {per_step} for {method}"
        )
    return budget // per_step


def integration_work(method: str, substeps: int) -> dict[str, int | str]:
    """Return executable work accounting for one integration call."""

    normalized = normalize_integrator(method)
    count = int(substeps)
    if count <= 0:
        raise ValueError("substeps must be positive")
    per_step = rhs_evaluations_per_substep(normalized)
    return {
        "integrator": normalized,
        "substeps": count,
        "rhs_evaluations_per_substep": per_step,
        "rhs_evaluations": count * per_step,
    }


def _duration_step(
    duration: float | torch.Tensor,
    state: torch.Tensor,
    substeps: int,
) -> torch.Tensor:
    duration_tensor = torch.as_tensor(duration, dtype=state.dtype, device=state.device)
    if not torch.isfinite(duration_tensor).all() or (duration_tensor <= 0).any():
        raise ValueError("integration duration must contain finite positive values")
    if duration_tensor.ndim == 0:
        return duration_tensor / int(substeps)
    if duration_tensor.ndim == 2 and duration_tensor.shape[1] == 1:
        duration_tensor = duration_tensor[:, 0]
    if duration_tensor.ndim != 1 or duration_tensor.shape[0] != state.shape[0]:
        raise ValueError(
            "batched duration must be scalar or have one value per state: "
            f"duration={tuple(duration_tensor.shape)}, state={tuple(state.shape)}"
        )
    shape = (duration_tensor.shape[0],) + (1,) * (state.ndim - 1)
    return duration_tensor.reshape(shape) / int(substeps)


def explicit_step(
    rhs: Callable[[torch.Tensor], torch.Tensor],
    state: torch.Tensor,
    dt: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Advance one Euler, midpoint/RK2, or classical RK4 substep."""

    normalized = normalize_integrator(method)
    k1 = rhs(state)
    if normalized == "euler":
        return state + dt * k1
    if normalized == "rk2":
        k2 = rhs(state + 0.5 * dt * k1)
        return state + dt * k2
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_explicit(
    rhs: Callable[[torch.Tensor], torch.Tensor],
    initial_state: torch.Tensor,
    duration: float | torch.Tensor,
    *,
    substeps: int,
    method: str,
    clamp: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Integrate one tensor-valued autonomous RHS without external mutation."""

    count = int(substeps)
    if count <= 0:
        raise ValueError("substeps must be positive")
    normalized = normalize_integrator(method)
    state = initial_state
    dt = _duration_step(duration, state, count)
    for _ in range(count):
        state = explicit_step(rhs, state, dt, normalized)
        if clamp is not None:
            state = torch.clamp(state, float(clamp[0]), float(clamp[1]))
    return state


def _positive_batch_time(
    value: float | torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Return one positive conditioning value for every batch item."""

    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.ndim == 0:
        tensor = tensor.expand(state.shape[0])
    elif tensor.ndim == 2 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 1 or tensor.shape[0] != state.shape[0]:
        raise ValueError(
            "conditioning time must be scalar or have one value per state: "
            f"conditioning={tuple(tensor.shape)}, state={tuple(state.shape)}"
        )
    if not torch.isfinite(tensor).all() or (tensor <= 0).any():
        raise ValueError("conditioning time must contain finite positive values")
    return tensor


def interaction_matrix(model: torch.nn.Module) -> torch.Tensor:
    """Build one checkpoint model's periodic interaction matrix once per call."""

    custom = getattr(model, "_interaction_matrix", None)
    if callable(custom):
        return custom()
    if not hasattr(model, "emb") or not hasattr(model, "interaction_mask"):
        raise TypeError("latent model lacks interaction parameters")
    embeddings = getattr(model, "emb")
    mask = getattr(model, "interaction_mask")
    return F.softplus(torch.mm(embeddings, embeddings.t())) * mask


@torch.no_grad()
def evolve_latent_flow(
    model: torch.nn.Module,
    state: torch.Tensor,
    evolution_time: float | torch.Tensor,
    *,
    method: str,
    substeps: int,
    conditioning_time: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Evolve a frozen A or B latent model with an explicit integrator.

    ``conditioning_time`` defaults to the physical call duration.  It is
    ignored for the autonomous A model and passed to every RHS evaluation for
    query-time-conditioned B.  The model's ``ode_steps`` attribute and state
    dictionary are never changed.
    """

    encode = getattr(model, "encode", None)
    decode = getattr(model, "decode", None)
    dynamics = getattr(model, "latent_dynamics", None)
    if not callable(encode) or not callable(decode) or not callable(dynamics):
        raise TypeError("model must expose encode, decode, and latent_dynamics")
    latent = encode(state)
    interactions = interaction_matrix(model)
    condition = _positive_batch_time(
        evolution_time if conditioning_time is None else conditioning_time,
        latent,
    )

    if getattr(model, "latent_dynamics_requires_tau", False):

        def rhs(value: torch.Tensor) -> torch.Tensor:
            return dynamics(value, interactions, tau=condition)

    else:

        def rhs(value: torch.Tensor) -> torch.Tensor:
            return dynamics(value, interactions)

    latent = integrate_explicit(
        rhs,
        latent,
        evolution_time,
        substeps=int(substeps),
        method=method,
        clamp=(-20.0, 20.0),
    )
    return decode(latent)


def integer_depth(horizon: float, lag: float) -> int:
    """Return ``horizon / lag`` and reject a non-integral composition depth."""

    ratio = float(horizon) / float(lag)
    depth = int(round(ratio))
    if (
        not math.isfinite(ratio)
        or depth <= 0
        or not math.isclose(ratio, depth, rel_tol=1e-10, abs_tol=1e-10)
    ):
        raise ValueError(
            "horizon must be a positive integer multiple of lag: "
            f"horizon={horizon}, lag={lag}, ratio={ratio:.17g}"
        )
    return depth


__all__ = [
    "INTEGRATOR_RHS_EVALUATIONS",
    "evolve_latent_flow",
    "explicit_step",
    "integer_depth",
    "integrate_explicit",
    "integration_work",
    "interaction_matrix",
    "normalize_integrator",
    "normalize_integrators",
    "rhs_evaluations_per_substep",
    "substeps_for_rhs_budget",
]
