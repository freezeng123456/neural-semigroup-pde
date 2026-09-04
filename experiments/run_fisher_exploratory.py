#!/usr/bin/env python3
"""Small, explicitly exploratory Fisher--KPP controls.

This runner is intentionally separate from :mod:`run_fisher_fair` and from
all formal/locked evaluation roots.  It provides the first inexpensive screen
requested in the semigroup transfer plan:

``capacity``
    Compare an autonomous ``A-wide`` latent flow with a
    ``latent_query_time`` control.  At the protocol setting ``N=64`` the
    autonomous interaction embedding is widened by one coordinate, giving
    exactly the 9,667 parameters of the query-time control.  The actual count
    and target/difference are always written to the result, including for
    small smoke grids where the counts need not be equal.

``b-soft``
    Train ``latent_query_time`` with the existing
    :func:`training.rollout_loss` through ``training.train_model``.  The
    supported fixed weights are ``1e-3``, ``1e-2`` and ``1e-1``; each weight
    gets its own model/checkpoint directory while all variants consume one
    shared exploratory cache.

``non-autonomous``
    Compare the same autonomous A-wide model with a latent flow whose
    mobility reads absolute physical time.  The reference equation uses
    ``r(t) = 1 + 0.5*sin(omega*t)``.  Both models receive the same frozen
    initial conditions, start times, labels, seed, update budget and
    validation trajectories.  The autonomous model has no absolute-time
    input in its forward path; the control receives ``absolute_time``
    explicitly and evaluates that time at every RK4 stage.

All artifacts produced by this module carry ``exploratory: true`` and
``do_not_use_for_formal: true``.  Existing caches are accepted only when they
carry the matching exploratory schema and configuration.  Paths containing
formal/locked roots are rejected before any write, and this module never
touches ``source.tar.gz`` or ``results-recovery``.

Example CPU smoke runs::

    python experiments/run_fisher_exploratory.py \
        --experiment capacity --device cpu --output-dir /tmp/fisher-exploratory-capacity \
        --data-cache /tmp/fisher-exploratory-capacity/data.pt \
        --N 8 --reference-dt 0.01 --fixed-tau 0.02 --eval-horizon 0.04 \
        --n-train 8 --n-val 2 --epochs 1 --batch-size 4 --ode-steps 2 \
        --deterministic --no-resume

    python experiments/run_fisher_exploratory.py \
        --experiment b-soft --device cpu --output-dir /tmp/fisher-exploratory-bsoft \
        --data-cache /tmp/fisher-exploratory-bsoft/data.pt \
        --N 8 --reference-dt 0.01 --fixed-tau 0.02 --eval-horizon 0.04 \
        --n-train 8 --n-val 2 --epochs 1 --batch-size 4 --ode-steps 2 \
        --deterministic --no-resume

    python experiments/run_fisher_exploratory.py \
        --experiment non-autonomous --device cpu --output-dir /tmp/fisher-exploratory-nonaut \
        --data-cache /tmp/fisher-exploratory-nonaut/data.pt \
        --N 8 --reference-dt 0.01 --fixed-tau 0.02 --eval-horizon 0.04 \
        --n-train 8 --n-val 2 --epochs 1 --batch-size 4 --ode-steps 2 \
        --omega 1.5 --start-time-max 0.2 --deterministic --no-resume

The default values are deliberately the same order of magnitude as the
formal Fisher configuration, but are not formal settings and must not be
pooled with formal/locked results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import training  # noqa: E402
from evaluate import evaluate_full  # noqa: E402
from models import (  # noqa: E402
    LatentSemigroupNet,
    QueryTimeConditionedLatentFlow,
    _broadcast_positive_tau,
)
from pde_solver import FisherKPPSolver, generate_initial_conditions  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402


EXPLORATORY_SCHEMA_VERSION = 1
EXPLORATORY_TRACK = "fisher_semigroup_first_wave"
AUTONOMOUS_DATA_KIND = "autonomous_fisher"
NON_AUTONOMOUS_DATA_KIND = "non_autonomous_fisher"
SUPPORTED_B_SOFT_LAMBDAS = (1e-3, 1e-2, 1e-1)
DEFAULT_TRAINING_LAMBDAS = SUPPORTED_B_SOFT_LAMBDAS
EXPERIMENT_ALIASES = {
    "capacity-matched": "capacity",
    "capacity_matched": "capacity",
    "b_soft": "b-soft",
    "non_autonomous": "non-autonomous",
    "nonauto": "non-autonomous",
}
DEFAULT_LATENT_KWARGS = {
    "hidden_V": [64, 64],
    "hidden_K": [64, 64],
    "stencil_radius": 3,
    "interaction_radius": 2,
    "beta_V": 0.0,
    "beta_V_floor": 0.1,
}


def _count_parameters(model: nn.Module) -> int:
    """Return the total number of parameters, including frozen parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters()))


def _count_trainable_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


class CapacityMatchedAutonomousLatentFlow(LatentSemigroupNet):
    """Autonomous A-wide control for the 9,667-parameter query-time model.

    ``LatentSemigroupNet`` uses an ``(N, 8)`` interaction embedding.  The
    query-time mobility layer adds one 64-wide input column, i.e. 64
    parameters at the default Fisher width.  Adding one interaction-embedding
    coordinate is a meaningful autonomous architectural widening and adds
    exactly ``N`` parameters.  Therefore the protocol setting ``N=64`` is an
    exact match.  For a tiny smoke grid the count is intentionally reported,
    rather than silently claiming equality.
    """

    def __init__(self, *args, interaction_embedding_dim: int = 9, **kwargs):
        if int(interaction_embedding_dim) < 8:
            raise ValueError("interaction_embedding_dim must be at least 8")
        super().__init__(*args, **kwargs)
        # Assignment replaces the base parameter in the module registry.  The
        # interaction matrix remains the same positive Gram construction, but
        # its autonomous capacity is widened without introducing a time input.
        self.emb = nn.Parameter(
            torch.randn(self.N, int(interaction_embedding_dim)) * 0.001
        )
        self.interaction_embedding_dim = int(interaction_embedding_dim)


def _broadcast_nonnegative_time(absolute_time: Any, u: torch.Tensor) -> torch.Tensor:
    """Broadcast a finite non-negative absolute time to ``(B, 1, N)``.

    This deliberately differs from ``_broadcast_positive_tau``: physical time
    zero is a valid absolute clock value even though a query increment must be
    strictly positive.
    """

    if u.ndim != 2:
        raise ValueError(f"u must have shape (B, N), got {tuple(u.shape)}")
    batch_size, n_sites = u.shape
    try:
        time_tensor = torch.as_tensor(absolute_time, device=u.device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "absolute_time must be a scalar or a tensor with shape (B,)"
        ) from exc
    if time_tensor.is_complex():
        raise ValueError("absolute_time must be real-valued")
    if time_tensor.ndim == 0:
        time_batch = time_tensor.expand(batch_size)
    elif time_tensor.ndim == 1 and time_tensor.shape[0] == batch_size:
        time_batch = time_tensor
    else:
        raise ValueError(
            f"absolute_time must be a scalar or have shape ({batch_size},), "
            f"got {tuple(time_tensor.shape)}"
        )
    time_batch = time_batch.to(dtype=u.dtype)
    valid = torch.isfinite(time_batch) & (time_batch >= 0)
    if not bool(valid.all().item()):
        raise ValueError("absolute_time must contain finite non-negative values")
    return time_batch.reshape(batch_size, 1, 1).expand(-1, 1, n_sites)


class AbsoluteTimeConditionedStencilMLP(nn.Module):
    """Periodic stencil MLP with an explicit absolute-time feature."""

    def __init__(self, radius: int = 3, hidden_dims: Iterable[int] = (32, 32)):
        super().__init__()
        self.radius = int(radius)
        hidden_dims = list(hidden_dims)
        in_dim = 2 * self.radius + 2
        dims = [in_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for index in range(len(dims) - 1):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < len(dims) - 2:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, absolute_time: Any) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError(f"z must have shape (B, N), got {tuple(z.shape)}")
        radius = self.radius
        z_pad = F.pad(z.unsqueeze(1), (radius, radius), mode="circular").squeeze(1)
        patches = z_pad.unfold(1, 2 * radius + 1, 1)
        time_feature = _broadcast_nonnegative_time(absolute_time, z)
        time_feature = time_feature.squeeze(1).unsqueeze(-1)
        return self.net(torch.cat((patches, time_feature), dim=-1)).squeeze(-1)


class NonAutonomousLatentFlow(LatentSemigroupNet):
    """Latent gradient flow whose mobility reads absolute physical time.

    The query-time control in ``models.py`` conditions on the requested lag,
    which is a different intervention.  This class instead passes the clock
    ``t`` through every RK4 stage, making the negative control genuinely
    non-autonomous: ``dz/dt = -K_theta(z, t) grad Psi_theta(z)``.
    """

    latent_dynamics_requires_absolute_time = True

    def __init__(
        self,
        N: int = 64,
        hidden_V: Iterable[int] = (32, 32),
        hidden_K: Iterable[int] = (32, 32),
        stencil_radius: int = 3,
        beta_V: float = 0.0,
        beta_V_floor: float = 0.0,
        interaction_radius: int = 2,
    ):
        super().__init__(
            N=N,
            hidden_V=list(hidden_V),
            hidden_K=list(hidden_K),
            stencil_radius=stencil_radius,
            beta_V=beta_V,
            beta_V_floor=beta_V_floor,
            interaction_radius=interaction_radius,
        )
        autonomous_k_net = self.K_net
        conditioned_k_net = AbsoluteTimeConditionedStencilMLP(
            radius=stencil_radius,
            hidden_dims=list(hidden_K),
        )
        autonomous_linear = [
            layer
            for layer in autonomous_k_net.net
            if isinstance(layer, nn.Linear)
        ]
        conditioned_linear = [
            layer
            for layer in conditioned_k_net.net
            if isinstance(layer, nn.Linear)
        ]
        if len(autonomous_linear) != len(conditioned_linear):
            raise RuntimeError("mobility MLP layouts must match")
        with torch.no_grad():
            for index, (source, target) in enumerate(
                zip(autonomous_linear, conditioned_linear)
            ):
                if index == 0:
                    target.weight[:, :-1].copy_(source.weight)
                    # The new time column is initially neutral.  Training can
                    # learn the explicitly exposed non-autonomous control.
                    target.weight[:, -1].zero_()
                else:
                    target.weight.copy_(source.weight)
                target.bias.copy_(source.bias)
        self.K_net = conditioned_k_net

    def _dynamics_and_grad(
        self,
        z: torch.Tensor,
        a_ij: torch.Tensor | None,
        absolute_time: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask
        mobility = F.softplus(self.K_net(z, absolute_time)) + 5e-3
        grad_psi = self.grad_psi(z, a_ij=a_ij)
        return mobility, grad_psi

    def latent_dynamics(
        self,
        z: torch.Tensor,
        a_ij: torch.Tensor | None = None,
        *,
        absolute_time: Any = None,
    ) -> torch.Tensor:
        if absolute_time is None:
            raise ValueError(
                "non-autonomous dynamics require the non-negative absolute_time"
            )
        if a_ij is None:
            a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
            a_ij = a_full * self.interaction_mask
        mobility, grad_psi = self._dynamics_and_grad(z, a_ij, absolute_time)
        return -mobility * grad_psi

    def rk4_step(
        self,
        z: torch.Tensor,
        dt: torch.Tensor,
        absolute_time: Any,
        a_ij: torch.Tensor,
    ) -> torch.Tensor:
        dt_batch = dt.reshape(-1)
        time_batch = _broadcast_nonnegative_time(absolute_time, z)[:, 0, 0]
        f1 = self.latent_dynamics(z, a_ij, absolute_time=time_batch)
        f2 = self.latent_dynamics(
            z + 0.5 * dt * f1,
            a_ij,
            absolute_time=time_batch + 0.5 * dt_batch,
        )
        f3 = self.latent_dynamics(
            z + 0.5 * dt * f2,
            a_ij,
            absolute_time=time_batch + 0.5 * dt_batch,
        )
        f4 = self.latent_dynamics(
            z + dt * f3,
            a_ij,
            absolute_time=time_batch + dt_batch,
        )
        z_next = z + dt / 6.0 * (f1 + 2.0 * f2 + 2.0 * f3 + f4)
        return torch.clamp(z_next, -20.0, 20.0)

    def forward(
        self,
        u: torch.Tensor,
        tau: Any,
        absolute_time: Any = 0.0,
    ) -> torch.Tensor:
        z = self.encode(u)
        a_full = F.softplus(torch.mm(self.emb, self.emb.t()))
        a_ij = a_full * self.interaction_mask
        tau_batch = _broadcast_positive_tau(tau, z)[:, 0, 0]
        time_batch = _broadcast_nonnegative_time(absolute_time, z)[:, 0, 0]
        dt = (tau_batch / self.ode_steps).reshape(-1, 1)
        for _ in range(self.ode_steps):
            z = self.rk4_step(z, dt, time_batch, a_ij)
            time_batch = time_batch + dt[:, 0]
        return self.decode(z)


class NonAutonomousFisherKPPSolver:
    """Reference solver for Fisher--KPP with ``r(t)`` modulation.

    The spatial discretisation mirrors ``FisherKPPSolver`` (periodic RFFT),
    while the time integrator is an explicit RK4 whose four stages receive
    their corresponding absolute physical time.  This is intentionally a
    small, transparent reference for the exploratory negative control.
    """

    def __init__(
        self,
        N: int = 64,
        L: float = 10.0,
        nu: float = 0.1,
        base_rate: float = 1.0,
        modulation: float = 0.5,
        omega: float = 1.0,
        dt: float = 0.005,
        dtype: torch.dtype = torch.float32,
    ):
        if int(N) <= 1:
            raise ValueError("N must be greater than one")
        for name, value in (
            ("L", L),
            ("nu", nu),
            ("base_rate", base_rate),
            ("modulation", modulation),
            ("omega", omega),
            ("dt", dt),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(L) <= 0 or float(nu) <= 0 or float(omega) <= 0 or float(dt) <= 0:
            raise ValueError("L, nu, omega, and dt must be strictly positive")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("NonAutonomousFisherKPPSolver supports float32/float64")
        self.N = int(N)
        self.L = float(L)
        self.nu = float(nu)
        self.base_rate = float(base_rate)
        self.modulation = float(modulation)
        self.omega = float(omega)
        self.dt = float(dt)
        self.dtype = dtype
        self.dx = self.L / self.N
        k_rfft = 2.0 * np.pi * np.fft.rfftfreq(self.N, d=self.dx)
        self.k2 = torch.tensor(k_rfft, dtype=dtype).square()

    def rate(self, absolute_time: Any) -> torch.Tensor | float:
        """Return the exact configured ``r(t)`` schedule."""

        if torch.is_tensor(absolute_time):
            return self.base_rate + self.modulation * torch.sin(
                self.omega * absolute_time
            )
        return self.base_rate + self.modulation * math.sin(
            self.omega * float(absolute_time)
        )

    def _rhs(self, u_hat: torch.Tensor, absolute_time: Any) -> torch.Tensor:
        u_phys = torch.fft.irfft(u_hat, n=self.N)
        reaction_rate = self.rate(absolute_time)
        reaction = reaction_rate * torch.fft.rfft(u_phys * (1.0 - u_phys))
        return -self.nu * self.k2.to(u_hat.device) * u_hat + reaction

    def step(self, u_hat: torch.Tensor, absolute_time: Any) -> torch.Tensor:
        dt = self.dt
        k1 = self._rhs(u_hat, absolute_time)
        k2 = self._rhs(u_hat + 0.5 * dt * k1, float(absolute_time) + 0.5 * dt)
        k3 = self._rhs(u_hat + 0.5 * dt * k2, float(absolute_time) + 0.5 * dt)
        k4 = self._rhs(u_hat + dt * k3, float(absolute_time) + dt)
        return u_hat + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def solve(
        self,
        u0: torch.Tensor,
        T: float,
        *,
        start_time: float = 0.0,
        save_every: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if u0.ndim != 1 or u0.shape[-1] != self.N:
            raise ValueError(f"u0 must have shape ({self.N},)")
        if not math.isfinite(float(start_time)) or float(start_time) < 0:
            raise ValueError("start_time must be finite and non-negative")
        ratio = float(T) / self.dt
        n_steps = int(round(ratio))
        if n_steps <= 0 or not math.isclose(
            ratio, n_steps, rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError("T must be a positive integer multiple of dt")
        if int(save_every) <= 0:
            raise ValueError("save_every must be positive")
        save_every = int(save_every)
        n_save = n_steps // save_every + 1
        times = torch.empty(n_save, dtype=self.dtype)
        states = torch.empty(n_save, self.N, dtype=self.dtype)
        u_hat = torch.fft.rfft(u0.to(self.dtype))
        times[0] = float(start_time)
        states[0] = u0.to(self.dtype)
        save_index = 1
        current_time = float(start_time)
        for step_index in range(1, n_steps + 1):
            u_hat = self.step(u_hat, current_time)
            current_time += self.dt
            if step_index % save_every == 0:
                times[save_index] = current_time
                states[save_index] = torch.fft.irfft(u_hat, n=self.N)
                save_index += 1
        return times, states


def non_autonomous_rate(
    absolute_time: Any,
    *,
    base_rate: float = 1.0,
    modulation: float = 0.5,
    omega: float = 1.0,
) -> torch.Tensor | float:
    """Standalone schedule used by the solver and tests."""

    if torch.is_tensor(absolute_time):
        return float(base_rate) + float(modulation) * torch.sin(
            float(omega) * absolute_time
        )
    return float(base_rate) + float(modulation) * math.sin(
        float(omega) * float(absolute_time)
    )


def parse_positive_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive floats") from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be finite and positive")
    return values


def parse_nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a finite non-negative float") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative float")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        "--mode",
        dest="experiment",
        choices=(
            "capacity",
            "capacity-matched",
            "capacity_matched",
            "b-soft",
            "b_soft",
            "non-autonomous",
            "non_autonomous",
            "nonauto",
        ),
        required=True,
        help="exploratory control to run; every choice writes exploratory artifacts",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--prepare-data-only", action="store_true")

    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=1.0)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--fixed-tau", type=float, default=0.1)
    parser.add_argument("--eval-horizon", type=float, default=1.2)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument(
        "--start-time-max",
        type=parse_nonnegative_float,
        default=1.0,
        help="maximum seeded absolute start time for the non-autonomous cache",
    )
    parser.add_argument(
        "--lambdas",
        type=parse_positive_float_list,
        default=DEFAULT_TRAINING_LAMBDAS,
        help="B-soft rollout weights; supported values are 1e-3, 1e-2, 1e-1",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "non-autonomous validation batch size; defaults to --batch-size "
            "to bound checkpoint-evaluation GPU memory"
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--ode-steps", type=int, default=30)
    parser.add_argument("--beta-v-floor", type=float, default=0.1)
    return parser


def reference_steps_for_duration(duration: float, reference_dt: float) -> int:
    ratio = float(duration) / float(reference_dt)
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(
        ratio, steps, rel_tol=1e-10, abs_tol=1e-12
    ):
        raise ValueError(
            "duration must be a positive integer multiple of reference_dt; "
            f"received duration/reference_dt={ratio:.17g}"
        )
    return steps


def rollout_steps_for_horizon(horizon: float, tau: float) -> int:
    ratio = float(horizon) / float(tau)
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(
        ratio, steps, rel_tol=1e-9, abs_tol=1e-10
    ):
        raise ValueError(f"eval_horizon={horizon} is not an integer multiple of tau={tau}")
    return steps


def validate_args(args: argparse.Namespace) -> None:
    args.experiment = EXPERIMENT_ALIASES.get(args.experiment, args.experiment)
    if int(args.seed) < 0 or int(args.data_seed) < 0:
        raise ValueError("seed and data-seed must be non-negative")
    if int(args.N) <= 1:
        raise ValueError("N must be greater than one")
    for name in ("n_train", "n_val", "epochs", "batch_size", "ode_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.eval_batch_size is not None and int(args.eval_batch_size) <= 0:
        raise ValueError("eval-batch-size must be positive when provided")
    for name in ("L", "nu", "reaction_rate", "reference_dt", "fixed_tau", "eval_horizon", "lr", "weight_decay"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if int(args.validation_interval) <= 0:
        raise ValueError("validation-interval must be positive")
    if not math.isfinite(float(args.beta_v_floor)) or float(args.beta_v_floor) < 0:
        raise ValueError("beta-v-floor must be finite and non-negative")
    if not math.isfinite(float(args.omega)) or float(args.omega) <= 0:
        raise ValueError("omega must be finite and strictly positive")
    if args.experiment == "non-autonomous" and float(args.start_time_max) <= 0:
        raise ValueError("non-autonomous start-time-max must be strictly positive")
    if args.experiment == "non-autonomous" and not math.isclose(
        float(args.reaction_rate), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "non-autonomous exploratory control fixes the reference schedule "
            "to r(t)=1+0.5*sin(omega*t), so reaction-rate must be 1"
        )
    reference_steps_for_duration(args.fixed_tau, args.reference_dt)
    reference_steps_for_duration(args.eval_horizon, args.reference_dt)
    rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)
    if args.experiment == "b-soft":
        for value in args.lambdas:
            if not any(math.isclose(value, allowed, rel_tol=1e-12, abs_tol=1e-15) for allowed in SUPPORTED_B_SOFT_LAMBDAS):
                raise ValueError(
                    "b-soft supports only fixed lambda values "
                    "1e-3, 1e-2, and 1e-1"
                )
        if len(set(float(value) for value in args.lambdas)) != len(args.lambdas):
            raise ValueError("b-soft lambdas must not contain duplicates")
    _assert_exploratory_path(args.output_dir, "output directory")
    _assert_exploratory_path(args.data_cache, "data cache")


_FORBIDDEN_PATH_COMPONENTS = {
    "formal",
    "locked",
    "results-recovery",
    "results_recovery",
    "source.tar.gz",
}


def _assert_exploratory_path(path: os.PathLike[str] | str, label: str) -> Path:
    """Reject obvious formal/locked roots before any exploratory write."""

    resolved = Path(path).expanduser().resolve()
    lowered_parts = {part.lower() for part in resolved.parts}
    if lowered_parts.intersection(_FORBIDDEN_PATH_COMPONENTS):
        raise ValueError(
            f"{label} points at a forbidden formal/locked/results-recovery path: {resolved}"
        )
    lowered_text = str(resolved).lower()
    if "source.tar.gz" in lowered_text:
        raise ValueError(f"{label} may not refer to source.tar.gz")
    parts = [part.lower() for part in resolved.parts]
    for index in range(len(parts) - 1):
        if parts[index : index + 2] == ["experiments", "results"]:
            raise ValueError(
                f"{label} may not reuse the tracked experiments/results tree: {resolved}"
            )
    return resolved


def _assert_existing_output_is_exploratory(output_dir: Path) -> None:
    marker = output_dir / "exploratory_manifest.json"
    if not marker.exists():
        # A non-empty result/provenance directory without our marker may be a
        # formal root.  Refuse to overwrite it instead of trying to infer its
        # provenance from filenames.
        if any(
            (output_dir / filename).exists()
            for filename in ("summary.json", "provenance.json", "result.json", "data_provenance.json")
        ):
            raise ValueError(
                "existing output directory lacks an exploratory manifest; "
                "refusing to reuse it"
            )
        return
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing exploratory manifest: {marker}") from exc
    if payload.get("exploratory") is not True or payload.get("track") != EXPLORATORY_TRACK:
        raise ValueError(f"existing output directory is not an exploratory root: {output_dir}")


def _sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        EXPERIMENTS_DIR / "training.py",
        EXPERIMENTS_DIR / "models.py",
        EXPERIMENTS_DIR / "evaluate.py",
        EXPERIMENTS_DIR / "pde_solver.py",
        EXPERIMENTS_DIR / "seed_utils.py",
    )
    return {
        str(path.relative_to(REPOSITORY_ROOT)): _sha256_file(path)
        for path in paths
        if path.exists()
    }


def _device_provenance(device: torch.device) -> dict[str, Any]:
    return {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _history_for_json(history: Mapping[str, Any]) -> dict[str, Any]:
    """Replace only scheduled-validation gaps with JSON ``null`` values."""
    normalized = {
        key: list(value) if isinstance(value, list) else value
        for key, value in history.items()
    }
    validation_performed = normalized.get("validation_performed")
    if not isinstance(validation_performed, list):
        return normalized
    for key in ("val_mse", "val_bound_viol", "val_energy_mono"):
        values = normalized.get(key)
        if not isinstance(values, list) or len(values) != len(validation_performed):
            continue
        for index, performed in enumerate(validation_performed):
            if not performed and isinstance(values[index], float) and not math.isfinite(values[index]):
                values[index] = None
    return normalized


def _atomic_torch_save(payload: Any, path: os.PathLike[str] | str) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mark_exploratory_torch_artifact(
    path: os.PathLike[str] | str,
    *,
    artifact: str,
    model: str | None = None,
) -> None:
    """Add an explicit exploratory marker to every generated torch artifact."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        payload.update(
            {
                "schema_version": EXPLORATORY_SCHEMA_VERSION,
                "exploratory": True,
                "do_not_use_for_formal": True,
                "track": EXPLORATORY_TRACK,
                "artifact": artifact,
            }
        )
        if model is not None:
            payload["model"] = model
        _atomic_torch_save(payload, path)
    else:
        raise ValueError(f"expected a mapping torch artifact at {path}")


def _common_data_config(args: argparse.Namespace, data_kind: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "track": EXPLORATORY_TRACK,
        "experiment": "fisher_semigroup_first_wave",
        "data_kind": data_kind,
        "pde": "fisher-kpp",
        "data_seed": int(args.data_seed),
        "deterministic": bool(args.deterministic),
        "N": int(args.N),
        "L": float(args.L),
        "nu": float(args.nu),
        "reference_dt": float(args.reference_dt),
        "fixed_tau": float(args.fixed_tau),
        "eval_horizon": float(args.eval_horizon),
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "initial_condition_policy": "fisher_low_frequency_fourier_modes",
    }
    if data_kind == AUTONOMOUS_DATA_KIND:
        config.update(
            {
                "reaction_schedule": {
                    "kind": "constant",
                    "r": float(args.reaction_rate),
                },
                "absolute_time_input": False,
            }
        )
    else:
        config.update(
            {
                "reaction_schedule": {
                    "kind": "sinusoidal_absolute_time",
                    "base_rate": 1.0,
                    "modulation": 0.5,
                    "omega": float(args.omega),
                    "formula": "r(t)=1+0.5*sin(omega*t)",
                },
                "start_time_policy": "seeded_uniform_on_[0,start_time_max]",
                "start_time_max": float(args.start_time_max),
                "absolute_time_input": True,
            }
        )
    return config


def _generate_autonomous_data(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.data_seed, deterministic=args.deterministic)
    train_u0 = generate_initial_conditions(args.N, args.n_train, args.L)
    solver = FisherKPPSolver(
        N=args.N,
        L=args.L,
        nu=args.nu,
        r=args.reaction_rate,
        dt=args.reference_dt,
    )
    train_ut = solver.solve_batch_final(train_u0, args.fixed_tau)
    val_u0 = generate_initial_conditions(args.N, args.n_val, args.L)
    val_trajs = [
        solver.solve(initial, args.eval_horizon, save_every=1)
        for initial in val_u0
    ]
    return {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "data_kind": AUTONOMOUS_DATA_KIND,
        "train_u0": train_u0,
        "train_t0": None,
        "train_ut": train_ut,
        "val_u0": val_u0,
        "val_t0": torch.zeros(args.n_val, dtype=val_u0.dtype),
        "val_trajs": val_trajs,
        "data_generation_config": _common_data_config(args, AUTONOMOUS_DATA_KIND),
    }


def _generate_non_autonomous_data(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(args.data_seed, deterministic=args.deterministic)
    train_u0 = generate_initial_conditions(args.N, args.n_train, args.L)
    train_t0 = torch.rand(args.n_train, dtype=train_u0.dtype) * float(args.start_time_max)
    solver = NonAutonomousFisherKPPSolver(
        N=args.N,
        L=args.L,
        nu=args.nu,
        base_rate=1.0,
        modulation=0.5,
        omega=args.omega,
        dt=args.reference_dt,
        dtype=train_u0.dtype,
    )
    train_ut = torch.stack(
        [
            solver.solve(
                train_u0[index],
                args.fixed_tau,
                start_time=float(train_t0[index]),
                save_every=reference_steps_for_duration(args.fixed_tau, args.reference_dt),
            )[1][-1]
            for index in range(args.n_train)
        ]
    )
    val_u0 = generate_initial_conditions(args.N, args.n_val, args.L)
    val_t0 = torch.rand(args.n_val, dtype=val_u0.dtype) * float(args.start_time_max)
    val_trajs = [
        solver.solve(
            val_u0[index],
            args.eval_horizon,
            start_time=float(val_t0[index]),
            save_every=1,
        )
        for index in range(args.n_val)
    ]
    return {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "data_kind": NON_AUTONOMOUS_DATA_KIND,
        "train_u0": train_u0,
        "train_t0": train_t0,
        "train_ut": train_ut,
        "val_u0": val_u0,
        "val_t0": val_t0,
        "val_trajs": val_trajs,
        "data_generation_config": _common_data_config(args, NON_AUTONOMOUS_DATA_KIND),
    }


def _validate_timestamp_grid(
    times: torch.Tensor,
    *,
    start_time: float,
    n_steps: int,
    reference_dt: float,
) -> None:
    if tuple(times.shape) != (n_steps + 1,):
        raise ValueError(
            f"validation times must have shape {(n_steps + 1,)}, got {tuple(times.shape)}"
        )
    expected = float(start_time) + torch.arange(n_steps + 1, dtype=times.dtype) * float(reference_dt)
    if not torch.allclose(times, expected, rtol=1e-6, atol=1e-7):
        raise ValueError("validation timestamps are not aligned to reference_dt")


def validate_data_cache(data: Mapping[str, Any], args: argparse.Namespace, data_kind: str) -> dict[str, Any]:
    if data.get("schema_version") != EXPLORATORY_SCHEMA_VERSION:
        raise ValueError("cache is not an exploratory schema-v1 cache")
    if data.get("exploratory") is not True or data.get("do_not_use_for_formal") is not True:
        raise ValueError("cache is not explicitly marked exploratory")
    if data.get("track") != EXPLORATORY_TRACK or data.get("data_kind") != data_kind:
        raise ValueError("cache exploratory track/data kind does not match this run")
    expected_config = _common_data_config(args, data_kind)
    if data.get("data_generation_config") != expected_config:
        raise ValueError("exploratory cache configuration does not match this run")
    required = {"train_u0", "train_t0", "train_ut", "val_u0", "val_t0", "val_trajs"}
    missing = sorted(required.difference(data))
    if missing:
        raise ValueError(f"exploratory cache is missing fields: {missing}")
    train_u0 = data["train_u0"]
    train_ut = data["train_ut"]
    val_u0 = data["val_u0"]
    train_t0 = data["train_t0"]
    val_t0 = data["val_t0"]
    for name, tensor in (("train_u0", train_u0), ("train_ut", train_ut), ("val_u0", val_u0), ("val_t0", val_t0)):
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be a finite tensor")
    if tuple(train_u0.shape) != (args.n_train, args.N) or tuple(train_ut.shape) != tuple(train_u0.shape):
        raise ValueError("exploratory training tensors have unexpected shapes")
    if tuple(val_u0.shape) != (args.n_val, args.N) or tuple(val_t0.shape) != (args.n_val,):
        raise ValueError("exploratory validation tensors have unexpected shapes")
    if data_kind == AUTONOMOUS_DATA_KIND:
        if train_t0 is not None or not torch.allclose(val_t0, torch.zeros_like(val_t0)):
            raise ValueError("autonomous exploratory cache must not carry absolute start times")
    else:
        if not torch.is_tensor(train_t0) or tuple(train_t0.shape) != (args.n_train,):
            raise ValueError("non-autonomous cache must carry one train start time per sample")
        if not torch.isfinite(train_t0).all() or not (train_t0 >= 0).all():
            raise ValueError("non-autonomous train start times must be finite and non-negative")
        if not torch.all(train_t0 <= float(args.start_time_max)):
            raise ValueError("non-autonomous train start times exceed start-time-max")
        if not torch.all(val_t0 >= 0) or not torch.all(val_t0 <= float(args.start_time_max)):
            raise ValueError("non-autonomous validation start times are out of range")
    if not torch.isfinite(train_u0).all() or not torch.isfinite(train_ut).all() or not torch.isfinite(val_u0).all():
        raise ValueError("exploratory cache contains non-finite states")

    expected_steps = reference_steps_for_duration(args.eval_horizon, args.reference_dt)
    val_trajs = data["val_trajs"]
    if not isinstance(val_trajs, (list, tuple)) or len(val_trajs) != args.n_val:
        raise ValueError("exploratory cache has the wrong validation trajectory count")
    for index, trajectory in enumerate(val_trajs):
        if not isinstance(trajectory, (list, tuple)) or len(trajectory) != 2:
            raise ValueError(f"validation trajectory {index} must be a (times, states) pair")
        times, states = trajectory
        if not torch.is_tensor(times) or not torch.is_tensor(states):
            raise ValueError(f"validation trajectory {index} must contain tensors")
        if tuple(states.shape) != (expected_steps + 1, args.N):
            raise ValueError(f"validation trajectory {index} has unexpected state shape")
        if not torch.isfinite(times).all() or not torch.isfinite(states).all():
            raise ValueError(f"validation trajectory {index} contains non-finite values")
        _validate_timestamp_grid(
            times,
            start_time=float(val_t0[index]),
            n_steps=expected_steps,
            reference_dt=args.reference_dt,
        )
    return {
        "train_shape": list(train_u0.shape),
        "validation_shape": list(val_u0.shape),
        "validation_trajectory_length": expected_steps + 1,
        "train_state_range": [float(train_u0.min()), float(train_u0.max())],
        "train_target_range": [float(train_ut.min()), float(train_ut.max())],
        "validation_start_time_range": [float(val_t0.min()), float(val_t0.max())],
        "train_start_time_range": (
            None if train_t0 is None else [float(train_t0.min()), float(train_t0.max())]
        ),
    }


def load_or_generate_data(
    args: argparse.Namespace,
    *,
    data_kind: str,
    allow_generate: bool = True,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    cache_path = _assert_exploratory_path(args.data_cache, "data cache")
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        validation = validate_data_cache(data, args, data_kind)
        return data, "loaded", validation
    if not allow_generate:
        raise FileNotFoundError(
            "exploratory data cache does not exist; use --prepare-data-only first: "
            f"{cache_path}"
        )
    data = (
        _generate_autonomous_data(args)
        if data_kind == AUTONOMOUS_DATA_KIND
        else _generate_non_autonomous_data(args)
    )
    _atomic_torch_save(data, cache_path)
    validation = validate_data_cache(data, args, data_kind)
    return data, "generated", validation


def _latent_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs = dict(DEFAULT_LATENT_KWARGS)
    kwargs["hidden_V"] = list(DEFAULT_LATENT_KWARGS["hidden_V"])
    kwargs["hidden_K"] = list(DEFAULT_LATENT_KWARGS["hidden_K"])
    kwargs["N"] = int(args.N)
    kwargs["beta_V_floor"] = float(args.beta_v_floor)
    return kwargs


def build_query_time_model(args: argparse.Namespace) -> QueryTimeConditionedLatentFlow:
    return QueryTimeConditionedLatentFlow(**_latent_kwargs(args))


def build_capacity_matched_autonomous_model(
    args: argparse.Namespace,
) -> CapacityMatchedAutonomousLatentFlow:
    return CapacityMatchedAutonomousLatentFlow(
        **_latent_kwargs(args),
        interaction_embedding_dim=9,
    )


def build_non_autonomous_model(args: argparse.Namespace) -> NonAutonomousLatentFlow:
    return NonAutonomousLatentFlow(**_latent_kwargs(args))


def parameter_match_metadata(
    autonomous_model: nn.Module,
    query_time_model: nn.Module,
) -> dict[str, Any]:
    target = _count_parameters(query_time_model)
    actual = _count_parameters(autonomous_model)
    return {
        "target_model": "latent_query_time",
        "target_parameter_count": target,
        "actual_parameter_count": actual,
        "difference_actual_minus_target": actual - target,
        "exact_match": actual == target,
        "strategy": "one_extra_autonomous_interaction_embedding_coordinate",
        "autonomous_interaction_embedding_dim": int(
            getattr(autonomous_model, "interaction_embedding_dim", 8)
        ),
    }


def fisher_energy(args: argparse.Namespace):
    def energy(u: torch.Tensor) -> torch.Tensor:
        dx = float(args.L) / u.shape[-1]
        grad = (torch.roll(u, shifts=-1, dims=-1) - u) / dx
        density = 0.5 * float(args.nu) * grad.square()
        density -= float(args.reaction_rate) * (0.5 * u.square() - u.pow(3) / 3.0)
        return dx * density.sum(dim=-1)

    return energy


def _autonomous_run_metadata(
    args: argparse.Namespace,
    *,
    model_name: str,
    model: nn.Module,
    data_cache: Path,
    alpha_rollout: float,
    parameter_match: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": args.experiment,
        "model": model_name,
        "temporal_structure": {
            "kind": (
                "autonomous_time_homogeneous_gradient_flow"
                if model_name == "a_wide"
                else "query_time_conditioned_gradient_flow"
            ),
            "absolute_time_input": False,
            "continuous_cross_tau_semigroup": model_name == "a_wide",
        },
        "parameter_count": _count_parameters(model),
        "trainable_parameter_count": _count_trainable_parameters(model),
        "training_seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "deterministic": bool(args.deterministic),
        "fixed_tau": float(args.fixed_tau),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size or args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "ode_steps": int(args.ode_steps),
        "validation_interval": int(args.validation_interval),
        "loss_weights": {
            "alpha_rollout": float(alpha_rollout),
            "alpha_energy": 0.0,
            "alpha_bound": 0.0,
            "alpha_V": 0.0,
        },
        "composition_loss_source": (
            "training.rollout_loss" if alpha_rollout > 0 else "not_enabled"
        ),
        "lambda": None if alpha_rollout <= 0 else float(alpha_rollout),
        "lambda_semantics": (
            None
            if alpha_rollout <= 0
            else "alpha_rollout in L_step + lambda * training.rollout_loss"
        ),
        "data_cache_sha256": _sha256_file(data_cache),
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
    }
    if parameter_match is not None:
        metadata["parameter_match"] = parameter_match
    return metadata


def _train_autonomous_model(
    model: nn.Module,
    *,
    model_name: str,
    args: argparse.Namespace,
    data: Mapping[str, Any],
    data_cache: Path,
    alpha_rollout: float = 0.0,
    parameter_match: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_global_seed(args.seed, deterministic=args.deterministic)
    model.ode_steps = int(args.ode_steps)
    model_dir = Path(args.output_dir).absolute() / model_name
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = _autonomous_run_metadata(
        args,
        model_name=model_name,
        model=model,
        data_cache=data_cache,
        alpha_rollout=alpha_rollout,
        parameter_match=parameter_match,
    )
    history = training.train_model(
        model,
        data["train_u0"],
        data["train_ut"],
        data["val_u0"],
        data["val_trajs"],
        tau=args.fixed_tau,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_rollout=float(alpha_rollout),
        alpha_energy=0.0,
        alpha_bound=0.0,
        alpha_V=0.0,
        weight_decay=args.weight_decay,
        checkpoint_dir=str(checkpoint_dir),
        model_name=model_name,
        device=args.device,
        resume_from=(
            None
            if args.no_resume
            else str(checkpoint_dir / f"{model_name}_best.pt")
        ),
        reference_dt=args.reference_dt,
        run_metadata=run_metadata,
        validation_interval=args.validation_interval,
    )
    # ``training.train_model`` is shared with the formal runners and therefore
    # deliberately has no exploratory schema knowledge.  Mark only the files
    # inside this explicitly exploratory output root after it returns.
    for artifact_path, artifact_name in (
        (checkpoint_dir / f"{model_name}_best.pt", "checkpoint_best"),
        (checkpoint_dir / f"{model_name}_final.pt", "checkpoint_final"),
        (checkpoint_dir / f"{model_name}_history.pt", "training_history"),
    ):
        if artifact_path.exists():
            _mark_exploratory_torch_artifact(
                artifact_path,
                artifact=artifact_name,
                model=model_name,
            )
    best_path = checkpoint_dir / f"{model_name}_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"exploratory best checkpoint was not written: {best_path}")
    checkpoint = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(args.device).eval()
    rollout_steps = rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)
    metrics, details = evaluate_full(
        model,
        data["val_u0"],
        data["val_trajs"],
        tau=args.fixed_tau,
        rollout_steps=rollout_steps,
        device=args.device,
        model_name=model_name,
        reference_dt=args.reference_dt,
        physical_energy_fn=fisher_energy(args),
        collect_latent_diagnostics=True,
        equal_work_base_ode_steps=min(30, int(args.ode_steps)),
    )
    result = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": args.experiment,
        "model": model_name,
        "parameter_count": _count_parameters(model),
        "trainable_parameter_count": _count_trainable_parameters(model),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_path": str(best_path.absolute()),
        "checkpoint_sha256": _sha256_file(best_path),
        "loss_weights": run_metadata["loss_weights"],
        "composition_loss_source": run_metadata["composition_loss_source"],
        "history": _history_for_json(history),
        "evaluations": {
            str(args.fixed_tau): {
                "metrics": metrics,
                "details": details,
            }
        },
        "run_metadata": run_metadata,
    }
    _write_json(model_dir / "result.json", result)
    return result, run_metadata


def run_capacity_experiment(
    args: argparse.Namespace,
    data: Mapping[str, Any],
    data_cache: Path,
) -> dict[str, Any]:
    set_global_seed(args.seed, deterministic=args.deterministic)
    query_model = build_query_time_model(args)
    set_global_seed(args.seed, deterministic=args.deterministic)
    autonomous_model = build_capacity_matched_autonomous_model(args)
    match = parameter_match_metadata(autonomous_model, query_model)
    autonomous_result, _ = _train_autonomous_model(
        autonomous_model,
        model_name="a_wide",
        args=args,
        data=data,
        data_cache=data_cache,
        alpha_rollout=0.0,
        parameter_match=match,
    )
    query_result, _ = _train_autonomous_model(
        query_model,
        model_name="b_query_time",
        args=args,
        data=data,
        data_cache=data_cache,
        alpha_rollout=0.0,
        parameter_match=match,
    )
    return {
        "parameter_match": match,
        "models": {
            "a_wide": autonomous_result,
            "b_query_time": query_result,
        },
    }


def run_b_soft_experiment(
    args: argparse.Namespace,
    data: Mapping[str, Any],
    data_cache: Path,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    set_global_seed(args.seed, deterministic=args.deterministic)
    query_parameter_count = _count_parameters(build_query_time_model(args))
    for value in args.lambdas:
        value = float(value)
        label = f"b_soft_lambda_{value:.0e}".replace("-", "m")
        set_global_seed(args.seed, deterministic=args.deterministic)
        model = build_query_time_model(args)
        result, metadata = _train_autonomous_model(
            model,
            model_name=label,
            args=args,
            data=data,
            data_cache=data_cache,
            alpha_rollout=value,
        )
        # Keep lambda machine-readable even if a future filename formatter
        # changes.  The training history's L_rollout is the existing shared
        # rollout_loss value, not a reimplemented composition objective.
        result["lambda"] = value
        result["lambda_semantics"] = {
            "parameter": "alpha_rollout",
            "loss": "training.rollout_loss",
            "total_objective": "L_step + lambda * L_rollout",
            "fixed_supported_values": list(SUPPORTED_B_SOFT_LAMBDAS),
        }
        result["parameter_count"] = query_parameter_count
        result["run_metadata"] = metadata
        _write_json(Path(args.output_dir).absolute() / label / "result.json", result)
        results[f"{value:.12g}"] = result
    return {
        "supported_lambdas": list(SUPPORTED_B_SOFT_LAMBDAS),
        "requested_lambdas": [float(value) for value in args.lambdas],
        "models": results,
    }


@torch.no_grad()
def _non_autonomous_evaluate(
    model: nn.Module,
    data: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    absolute_time_input: bool,
) -> dict[str, Any]:
    model.eval()
    device = args.device
    # Keep the validation trajectory on CPU and move one bounded batch at a
    # time.  With 30 RK4 steps and a multi-step rollout, evaluating all
    # validation trajectories at once can exhaust a 10 GB GPU even though the
    # training mini-batches fit.  The per-sample MSE statistics are unchanged
    # by this chunking, while peak memory is controlled by eval-batch-size.
    val_u0 = data["val_u0"]
    val_t0 = data["val_t0"]
    references = torch.stack([trajectory[1] for trajectory in data["val_trajs"]])
    eval_batch_size = int(args.eval_batch_size or args.batch_size)
    if eval_batch_size <= 0:
        raise ValueError("eval batch size must be positive")
    n_steps = rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)
    stride = reference_steps_for_duration(args.fixed_tau, args.reference_dt)
    mse_values: list[float] = []
    for batch_start in range(0, int(val_u0.shape[0]), eval_batch_size):
        batch_end = min(batch_start + eval_batch_size, int(val_u0.shape[0]))
        state = val_u0[batch_start:batch_end].to(device)
        start_time = val_t0[batch_start:batch_end].to(device)
        reference_batch = references[batch_start:batch_end].to(device)
        for step_index in range(n_steps):
            if absolute_time_input:
                state = model(
                    state,
                    args.fixed_tau,
                    absolute_time=start_time + step_index * float(args.fixed_tau),
                )
            else:
                state = model(state, args.fixed_tau)
            reference = reference_batch[:, (step_index + 1) * stride]
            mse_values.extend(
                (state - reference)
                .reshape(state.shape[0], -1)
                .square()
                .mean(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
        del state, start_time, reference_batch
    return {
        "model": "b_absolute_time" if absolute_time_input else "a_wide",
        "tau": float(args.fixed_tau),
        "reference_dt": float(args.reference_dt),
        "rollout_steps": n_steps,
        "eval_batch_size": eval_batch_size,
        "rollout_mse_mean": float(np.mean(mse_values)),
        "rollout_mse_std": float(np.std(mse_values)),
        "n_valid": len(mse_values),
        "absolute_time_input": bool(absolute_time_input),
        "rate_schedule": "r(t)=1+0.5*sin(omega*t)",
    }


def _train_non_autonomous_model(
    model: nn.Module,
    *,
    model_name: str,
    args: argparse.Namespace,
    data: Mapping[str, Any],
    data_cache: Path,
    absolute_time_input: bool,
    parameter_match: dict[str, Any],
) -> dict[str, Any]:
    set_global_seed(args.seed, deterministic=args.deterministic)
    model.ode_steps = int(args.ode_steps)
    model_dir = Path(args.output_dir).absolute() / model_name
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(args.device)
    train_u0 = data["train_u0"].to(args.device)
    train_t0 = data["train_t0"].to(args.device)
    train_ut = data["train_ut"].to(args.device)
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    loader = DataLoader(
        TensorDataset(train_u0.cpu(), train_t0.cpu(), train_ut.cpu()),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    run_metadata = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": "non-autonomous",
        "model": model_name,
        "temporal_structure": {
            "kind": "autonomous_time_homogeneous_gradient_flow"
            if not absolute_time_input
            else "non_autonomous_absolute_time_conditioned_gradient_flow",
            "absolute_time_input": bool(absolute_time_input),
            "absolute_time_argument": None if not absolute_time_input else "absolute_time",
            "rate_schedule": "r(t)=1+0.5*sin(omega*t)",
        },
        "parameter_count": _count_parameters(model),
        "trainable_parameter_count": _count_trainable_parameters(model),
        "parameter_match": parameter_match,
        "training_seed": int(args.seed),
        "data_seed": int(args.data_seed),
        "fixed_tau": float(args.fixed_tau),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size or args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "ode_steps": int(args.ode_steps),
        "validation_interval": int(args.validation_interval),
        "loss_weights": {"alpha_rollout": 0.0},
        "data_cache_sha256": _sha256_file(data_cache),
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
    }
    history: dict[str, list[float | int | bool | None]] = {
        "epoch": [],
        "L_step": [],
        "val_mse": [],
        "validation_performed": [],
    }
    best_val = float("inf")
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for u0_batch, t0_batch, ut_batch in loader:
            u0_batch = u0_batch.to(args.device)
            t0_batch = t0_batch.to(args.device)
            ut_batch = ut_batch.to(args.device)
            if absolute_time_input:
                prediction = model(
                    u0_batch,
                    args.fixed_tau,
                    absolute_time=t0_batch,
                )
            else:
                prediction = model(u0_batch, args.fixed_tau)
            loss = torch.mean((prediction - ut_batch) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        should_validate = epoch % args.validation_interval == 0 or epoch == args.epochs
        validation = _non_autonomous_evaluate(
            model,
            data,
            args,
            absolute_time_input=absolute_time_input,
        ) if should_validate else None
        history["epoch"].append(epoch)
        history["L_step"].append(float(np.mean(losses)))
        history["validation_performed"].append(should_validate)
        history["val_mse"].append(
            float(validation["rollout_mse_mean"])
            if validation is not None
            else None
        )
        if validation is not None and validation["rollout_mse_mean"] < best_val:
            best_val = float(validation["rollout_mse_mean"])
            torch.save(
                {
                    "schema_version": EXPLORATORY_SCHEMA_VERSION,
                    "exploratory": True,
                    "do_not_use_for_formal": True,
                    "track": EXPLORATORY_TRACK,
                    "artifact": "checkpoint_best",
                    "model": model_name,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_mse": best_val,
                    "history": history,
                    "run_metadata": run_metadata,
                },
                checkpoint_dir / f"{model_name}_best.pt",
            )
    final_path = checkpoint_dir / f"{model_name}_final.pt"
    torch.save(
        {
            "schema_version": EXPLORATORY_SCHEMA_VERSION,
            "exploratory": True,
            "do_not_use_for_formal": True,
            "track": EXPLORATORY_TRACK,
            "artifact": "checkpoint_final",
            "model": model_name,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "run_metadata": run_metadata,
        },
        final_path,
    )
    best_path = checkpoint_dir / f"{model_name}_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"non-autonomous best checkpoint was not written: {best_path}")
    best = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    model.eval()
    evaluation = _non_autonomous_evaluate(
        model,
        data,
        args,
        absolute_time_input=absolute_time_input,
    )
    result = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": "non-autonomous",
        "model": model_name,
        "parameter_count": _count_parameters(model),
        "trainable_parameter_count": _count_trainable_parameters(model),
        "checkpoint_epoch": best.get("epoch"),
        "checkpoint_path": str(best_path.absolute()),
        "checkpoint_sha256": _sha256_file(best_path),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "evaluation": evaluation,
        "run_metadata": run_metadata,
    }
    _write_json(model_dir / "result.json", result)
    return result


def run_non_autonomous_experiment(
    args: argparse.Namespace,
    data: Mapping[str, Any],
    data_cache: Path,
) -> dict[str, Any]:
    set_global_seed(args.seed, deterministic=args.deterministic)
    autonomous_model = build_capacity_matched_autonomous_model(args)
    set_global_seed(args.seed, deterministic=args.deterministic)
    non_autonomous_model = build_non_autonomous_model(args)
    match = parameter_match_metadata(autonomous_model, non_autonomous_model)
    # Both models receive the same frozen data object.  The difference is only
    # whether the model call consumes the per-sample absolute start time.
    autonomous_result = _train_non_autonomous_model(
        autonomous_model,
        model_name="a_wide",
        args=args,
        data=data,
        data_cache=data_cache,
        absolute_time_input=False,
        parameter_match=match,
    )
    non_autonomous_result = _train_non_autonomous_model(
        non_autonomous_model,
        model_name="b_absolute_time",
        args=args,
        data=data,
        data_cache=data_cache,
        absolute_time_input=True,
        parameter_match=match,
    )
    return {
        "rate_schedule": {
            "formula": "r(t)=1+0.5*sin(omega*t)",
            "base_rate": float(args.reaction_rate),
            "modulation": 0.5,
            "omega": float(args.omega),
        },
        "shared_data": {
            "data_kind": NON_AUTONOMOUS_DATA_KIND,
            "same_cache_for_models": True,
            "train_start_time_field": "train_t0",
            "validation_start_time_field": "val_t0",
        },
        "parameter_match": match,
        "models": {
            "a_wide": autonomous_result,
            "b_absolute_time": non_autonomous_result,
        },
    }


def _write_exploratory_manifest(args: argparse.Namespace, data_cache: Path) -> Path:
    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "exploratory_manifest.json"
    if path.exists():
        _assert_existing_output_is_exploratory(output_dir)
    payload = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": args.experiment,
        "created_at_unix": time.time(),
        "config": vars(args),
        "data_cache": str(data_cache),
        "data_cache_sha256": _sha256_file(data_cache) if data_cache.exists() else None,
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
        "scope_guard": {
            "formal_roots_reused": False,
            "locked_test_cache_reused": False,
            "source_archive_modified": False,
            "results_recovery_modified": False,
            "scnet_or_t4_login": False,
        },
    }
    _write_json(path, payload)
    return path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    output_dir = _assert_exploratory_path(args.output_dir, "output directory")
    data_cache = _assert_exploratory_path(args.data_cache, "data cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    _assert_existing_output_is_exploratory(output_dir)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    data_kind = (
        NON_AUTONOMOUS_DATA_KIND
        if args.experiment == "non-autonomous"
        else AUTONOMOUS_DATA_KIND
    )
    data, cache_status, data_validation = load_or_generate_data(
        args,
        data_kind=data_kind,
        allow_generate=True,
    )
    manifest_path = _write_exploratory_manifest(args, data_cache)
    provenance = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": args.experiment,
        "status": "data_prepared" if args.prepare_data_only else "started",
        "cache_status": cache_status,
        "data_cache": {
            "path": str(data_cache),
            "sha256": _sha256_file(data_cache),
        },
        "data_validation": data_validation,
        "data_generation_config": data["data_generation_config"],
        "config": vars(args),
        "manifest_path": str(manifest_path),
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
        "device": _device_provenance(torch.device(args.device)),
    }
    _write_json(output_dir / "data_provenance.json", provenance)
    if args.prepare_data_only:
        _write_json(output_dir / "summary.json", provenance)
        print(json.dumps({"exploratory": True, "mode": "data_prepared", "data_cache": str(data_cache)}, indent=2))
        return

    if args.experiment == "capacity":
        results = run_capacity_experiment(args, data, data_cache)
    elif args.experiment == "b-soft":
        results = run_b_soft_experiment(args, data, data_cache)
    else:
        results = run_non_autonomous_experiment(args, data, data_cache)
    summary = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "experiment": args.experiment,
        "status": "completed",
        "config": vars(args),
        "data_cache": {
            "path": str(data_cache),
            "sha256": _sha256_file(data_cache),
        },
        "data_validation": data_validation,
        "results": results,
        "git_commit": _git_commit(),
        "source_hashes": _source_hashes(),
        "device": _device_provenance(torch.device(args.device)),
    }
    provenance.update({"status": "completed", "summary_path": str((output_dir / "summary.json").absolute())})
    _write_json(output_dir / "data_provenance.json", provenance)
    _write_json(output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "exploratory": True,
                "experiment": args.experiment,
                "output_dir": str(output_dir),
                "data_cache": str(data_cache),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
