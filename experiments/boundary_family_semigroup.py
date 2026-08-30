"""Boundary-family neural flows for the exploratory Wave 2 experiment.

Spatial admissibility and temporal conditioning are separate interventions.
The boundary map owns no trainable parameter; the neural vector field has the
same tensors in every matrix cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


BOUNDARY_FAMILIES = (
    "inhomogeneous_dirichlet",
    "homogeneous_neumann",
    "robin",
)
ENFORCEMENT_MODES = ("hard", "penalty", "unconstrained")
TEMPORAL_MODES = ("autonomous", "query_time")
TRAINING_SEEDS = (31415, 271828, 161803)
UNIFORM_PRIMARY_PAIRS = (
    (0.02, 0.08),
    (0.02, 0.16),
    (0.04, 0.08),
    (0.04, 0.16),
    (0.08, 0.16),
)
LONG_ROLLOUT_PAIRS = (
    (0.03, 0.24),
    (0.03, 0.48),
    (0.06, 0.24),
    (0.06, 0.48),
)
NONUNIFORM_PARTITIONS = (
    (0.02, 0.14),
    (0.04, 0.12),
    (0.02, 0.04, 0.10),
    (0.01, 0.03, 0.04, 0.08),
)
REFINEMENT_STEPS = (4, 8, 16)
BOUNDARY_THRESHOLD = 5e-6
METRIC_FLOOR = 1e-16


@dataclass(frozen=True)
class BoundaryFamilySpec:
    """One fixed discrete affine boundary condition."""

    family: str
    dx: float
    dirichlet_left: float = 0.20
    dirichlet_right: float = -0.10
    neumann_left: float = 0.0
    neumann_right: float = 0.0
    robin_alpha: float = 1.0
    robin_beta: float = 0.1
    robin_gamma_left: float = 0.15
    robin_gamma_right: float = -0.05

    def validate(self) -> None:
        if self.family not in BOUNDARY_FAMILIES:
            raise ValueError(f"unsupported boundary family: {self.family}")
        values = asdict(self)
        for name, value in values.items():
            if name == "family":
                continue
            if not math.isfinite(float(value)):
                raise ValueError(f"non-finite boundary value: {name}")
        if self.dx <= 0:
            raise ValueError("dx must be positive")
        if self.family == "homogeneous_neumann" and (
            self.neumann_left != 0.0 or self.neumann_right != 0.0
        ):
            raise ValueError("the frozen Neumann family must be homogeneous")
        if self.family == "robin" and self.robin_alpha + self.robin_beta / self.dx == 0:
            raise ValueError("singular Robin endpoint reconstruction")

    @property
    def robin_ratio(self) -> float:
        return (self.robin_beta / self.dx) / (
            self.robin_alpha + self.robin_beta / self.dx
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        """Apply the affine state reconstruction in grid-value units."""

        _check_state(state)
        projected = state.clone()
        if self.family == "inhomogeneous_dirichlet":
            projected[:, 0] = self.dirichlet_left
            projected[:, -1] = self.dirichlet_right
        elif self.family == "homogeneous_neumann":
            projected[:, 0] = projected[:, 1]
            projected[:, -1] = projected[:, -2]
        else:
            scale = self.robin_beta / self.dx
            denominator = self.robin_alpha + scale
            projected[:, 0] = (
                self.robin_gamma_left + scale * projected[:, 1]
            ) / denominator
            projected[:, -1] = (
                self.robin_gamma_right + scale * projected[:, -2]
            ) / denominator
        return projected

    def project_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        """Project a vector onto the homogeneous tangent boundary space."""

        _check_state(tangent)
        projected = tangent.clone()
        if self.family == "inhomogeneous_dirichlet":
            projected[:, (0, -1)] = 0.0
        elif self.family == "homogeneous_neumann":
            projected[:, 0] = projected[:, 1]
            projected[:, -1] = projected[:, -2]
        else:
            projected[:, 0] = self.robin_ratio * projected[:, 1]
            projected[:, -1] = self.robin_ratio * projected[:, -2]
        return projected

    def state_residual(self, state: torch.Tensor) -> torch.Tensor:
        """Return two endpoint reconstruction residuals in state units."""

        _check_state(state)
        if self.family == "inhomogeneous_dirichlet":
            left = state[:, 0] - self.dirichlet_left
            right = state[:, -1] - self.dirichlet_right
        elif self.family == "homogeneous_neumann":
            left = state[:, 0] - state[:, 1]
            right = state[:, -1] - state[:, -2]
        else:
            scale = self.robin_beta / self.dx
            denominator = self.robin_alpha + scale
            expected_left = (self.robin_gamma_left + scale * state[:, 1]) / denominator
            expected_right = (
                self.robin_gamma_right + scale * state[:, -2]
            ) / denominator
            left = state[:, 0] - expected_left
            right = state[:, -1] - expected_right
        return torch.stack((left, right), dim=1)

    def tangent_residual(self, tangent: torch.Tensor) -> torch.Tensor:
        """Return the homogeneous boundary residual of an RHS vector."""

        _check_state(tangent)
        if self.family == "inhomogeneous_dirichlet":
            left = tangent[:, 0]
            right = tangent[:, -1]
        elif self.family == "homogeneous_neumann":
            left = tangent[:, 0] - tangent[:, 1]
            right = tangent[:, -1] - tangent[:, -2]
        else:
            left = tangent[:, 0] - self.robin_ratio * tangent[:, 1]
            right = tangent[:, -1] - self.robin_ratio * tangent[:, -2]
        return torch.stack((left, right), dim=1)

    def equation_residual(self, state: torch.Tensor) -> torch.Tensor:
        """Return the two raw discrete boundary-equation residuals."""

        _check_state(state)
        if self.family == "inhomogeneous_dirichlet":
            return self.state_residual(state)
        if self.family == "homogeneous_neumann":
            left = -(state[:, 1] - state[:, 0]) / self.dx
            right = (state[:, -1] - state[:, -2]) / self.dx
            return torch.stack((left, right), dim=1)
        left_normal = -(state[:, 1] - state[:, 0]) / self.dx
        right_normal = (state[:, -1] - state[:, -2]) / self.dx
        left = (
            self.robin_alpha * state[:, 0]
            + self.robin_beta * left_normal
            - self.robin_gamma_left
        )
        right = (
            self.robin_alpha * state[:, -1]
            + self.robin_beta * right_normal
            - self.robin_gamma_right
        )
        return torch.stack((left, right), dim=1)


@dataclass(frozen=True)
class BoundaryFamilyWaveConfig:
    """Frozen numerical and training settings for one Wave 2 family."""

    boundary_family: str
    schema_version: int = 1
    experiment: str = "boundary_family_semigroup_wave2"
    exploratory: bool = True
    smoke_only: bool = False
    data_seed: int = 515151
    n_grid: int = 33
    domain_length: float = 1.0
    diffusivity: float = 0.02
    reference_dt: float = 0.001
    sine_modes: int = 8
    n_train: int = 512
    n_val: int = 64
    n_test: int = 128
    train_taus: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16)
    hidden_width: int = 32
    query_time_scale: float = 0.08
    neural_rk4_steps: int = 8
    epochs: int = 150
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    validation_interval: int = 5
    penalty_lambda: float = 1.0
    stress_samples: int = 64
    stress_amplitude: float = 0.25
    stress_tau: float = 0.04
    stress_horizon: float = 0.16
    boundary_threshold: float = BOUNDARY_THRESHOLD

    def validate(self) -> None:
        if self.boundary_family not in BOUNDARY_FAMILIES:
            raise ValueError(f"unsupported boundary family: {self.boundary_family}")
        if self.n_grid < 5 or self.n_grid % 2 == 0:
            raise ValueError("n_grid must be an odd integer at least five")
        positive = {
            "domain_length": self.domain_length,
            "diffusivity": self.diffusivity,
            "reference_dt": self.reference_dt,
            "sine_modes": self.sine_modes,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "hidden_width": self.hidden_width,
            "query_time_scale": self.query_time_scale,
            "neural_rk4_steps": self.neural_rk4_steps,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_interval": self.validation_interval,
            "penalty_lambda": self.penalty_lambda,
            "stress_samples": self.stress_samples,
            "stress_amplitude": self.stress_amplitude,
            "stress_tau": self.stress_tau,
            "stress_horizon": self.stress_horizon,
            "boundary_threshold": self.boundary_threshold,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if self.n_train % len(self.train_taus) != 0:
            raise ValueError("n_train must balance training durations")
        if self.stress_samples > self.n_test:
            raise ValueError("stress_samples may not exceed n_test")
        self.boundary_spec.validate()
        for duration in self.reference_horizons:
            _require_reference_alignment(duration, self.reference_dt)
        for tau, horizon in (*UNIFORM_PRIMARY_PAIRS, *LONG_ROLLOUT_PAIRS):
            composition_depth(tau, horizon)
        for partition in NONUNIFORM_PARTITIONS:
            if not math.isclose(sum(partition), 0.16, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("nonuniform partition must sum to 0.16")

    @property
    def dx(self) -> float:
        return self.domain_length / (self.n_grid - 1)

    @property
    def boundary_spec(self) -> BoundaryFamilySpec:
        return BoundaryFamilySpec(family=self.boundary_family, dx=self.dx)

    @property
    def reference_horizons(self) -> tuple[float, ...]:
        values = set(self.train_taus)
        values.update(horizon for _tau, horizon in UNIFORM_PRIMARY_PAIRS)
        values.update(horizon for _tau, horizon in LONG_ROLLOUT_PAIRS)
        values.add(0.16)
        return tuple(sorted(values))

    def data_identity(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "smoke_only": self.smoke_only,
            "data_seed": self.data_seed,
            "boundary_family": self.boundary_family,
            "boundary_spec": self.boundary_spec.as_dict(),
            "n_grid": self.n_grid,
            "domain_length": self.domain_length,
            "diffusivity": self.diffusivity,
            "reference_dt": self.reference_dt,
            "sine_modes": self.sine_modes,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "train_taus": list(self.train_taus),
            "reference_horizons": list(self.reference_horizons),
        }

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["train_taus"] = list(self.train_taus)
        payload["dx"] = self.dx
        payload["boundary_spec"] = self.boundary_spec.as_dict()
        payload["uniform_primary_pairs"] = [
            list(pair) for pair in UNIFORM_PRIMARY_PAIRS
        ]
        payload["long_rollout_pairs"] = [list(pair) for pair in LONG_ROLLOUT_PAIRS]
        payload["nonuniform_partitions"] = [
            list(part) for part in NONUNIFORM_PARTITIONS
        ]
        payload["refinement_steps"] = list(REFINEMENT_STEPS)
        payload["reference_horizons"] = list(self.reference_horizons)
        return payload


def wave_config(
    boundary_family: str, *, smoke_only: bool = False
) -> BoundaryFamilyWaveConfig:
    config = BoundaryFamilyWaveConfig(boundary_family=boundary_family)
    if smoke_only:
        config = replace(
            config,
            smoke_only=True,
            n_grid=17,
            reference_dt=0.002,
            sine_modes=4,
            n_train=32,
            n_val=8,
            n_test=8,
            hidden_width=8,
            neural_rk4_steps=2,
            epochs=2,
            batch_size=8,
            validation_interval=1,
            stress_samples=8,
        )
    config.validate()
    return config


def _check_state(state: torch.Tensor) -> None:
    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError("state must have shape [batch, grid] with grid >= 3")


def time_key(value: float) -> str:
    return format(float(value), ".12g")


def composition_depth(tau: float, horizon: float) -> int:
    ratio = float(horizon) / float(tau)
    rounded = round(ratio)
    if rounded <= 0 or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"horizon={horizon} is not an integer multiple of tau={tau}")
    return int(rounded)


def _require_reference_alignment(duration: float, reference_dt: float) -> int:
    steps = round(float(duration) / float(reference_dt))
    if steps <= 0 or not math.isclose(
        steps * float(reference_dt), float(duration), rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError(
            f"duration={duration} is not aligned with reference_dt={reference_dt}"
        )
    return int(steps)


def residual_max(residual: torch.Tensor) -> torch.Tensor:
    return residual.abs().max()


def residual_rms(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().mean().sqrt()


class BoundaryFamilyFlow(nn.Module):
    """A local neural ODE with orthogonal boundary and temporal interventions."""

    def __init__(
        self,
        *,
        n_grid: int,
        hidden_width: int,
        boundary_spec: BoundaryFamilySpec,
        enforcement_mode: str,
        temporal_mode: str,
        query_time_scale: float,
        rk4_steps: int,
    ) -> None:
        super().__init__()
        boundary_spec.validate()
        if enforcement_mode not in ENFORCEMENT_MODES:
            raise ValueError(f"unsupported enforcement mode: {enforcement_mode}")
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"unsupported temporal mode: {temporal_mode}")
        if n_grid < 3 or hidden_width <= 0 or query_time_scale <= 0 or rk4_steps <= 0:
            raise ValueError("invalid model dimensions or integration settings")
        self.n_grid = int(n_grid)
        self.hidden_width = int(hidden_width)
        self.boundary_spec = boundary_spec
        self.enforcement_mode = str(enforcement_mode)
        self.temporal_mode = str(temporal_mode)
        self.query_time_scale = float(query_time_scale)
        self.rk4_steps = int(rk4_steps)
        self.vector_field_net = nn.Sequential(
            nn.Linear(4, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 1),
        )

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.enforcement_mode == "hard":
            return self.boundary_spec.project_state(state)
        return state

    def _duration_tensor(
        self, duration: float | torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        value = torch.as_tensor(duration, dtype=state.dtype, device=state.device)
        if value.ndim == 0:
            value = value.expand(state.shape[0])
        elif value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1 or value.shape[0] != state.shape[0]:
            raise ValueError("duration must be scalar or one value per batch item")
        if bool((value <= 0).any()):
            raise ValueError("duration values must be positive")
        return value.reshape(-1, 1)

    def vector_field(
        self, state: torch.Tensor, map_duration: float | torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.n_grid:
            raise ValueError(
                f"expected state shape [batch, {self.n_grid}], got {tuple(state.shape)}"
            )
        current = self.project_state(state)
        patches = F.pad(current, (1, 1), mode="constant", value=0.0).unfold(
            dimension=1, size=3, step=1
        )
        duration = self._duration_tensor(map_duration, current)
        if self.temporal_mode == "autonomous":
            temporal_feature = torch.ones_like(duration)
        else:
            temporal_feature = duration / self.query_time_scale
        temporal_feature = temporal_feature.unsqueeze(1).expand(-1, self.n_grid, -1)
        features = torch.cat((patches, temporal_feature), dim=-1)
        derivative = self.vector_field_net(features).squeeze(-1)
        if self.enforcement_mode == "hard":
            derivative = self.boundary_spec.project_tangent(derivative)
        return derivative

    def _rk4_step(
        self,
        state: torch.Tensor,
        dt: torch.Tensor,
        map_duration: float | torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        k1 = self.vector_field(state, map_duration)
        stage2 = self.project_state(state + 0.5 * dt * k1)
        k2 = self.vector_field(stage2, map_duration)
        stage3 = self.project_state(state + 0.5 * dt * k2)
        k3 = self.vector_field(stage3, map_duration)
        stage4 = self.project_state(state + dt * k3)
        k4 = self.vector_field(stage4, map_duration)
        next_state = self.project_state(
            state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        )
        return next_state, (stage2, stage3, stage4, next_state), (k1, k2, k3, k4)

    def integrate(
        self,
        state: torch.Tensor,
        duration: float | torch.Tensor,
        *,
        map_duration: float | torch.Tensor | None = None,
        steps: int | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        step_count = self.rk4_steps if steps is None else int(steps)
        if step_count <= 0:
            raise ValueError("steps must be positive")
        requested_duration = duration if map_duration is None else map_duration
        duration_tensor = self._duration_tensor(duration, state)
        dt = duration_tensor / step_count
        current = self.project_state(state)
        state_max = residual_max(self.boundary_spec.state_residual(current))
        equation_max = residual_max(self.boundary_spec.equation_residual(current))
        tangent_max = torch.zeros((), dtype=state.dtype, device=state.device)
        for _ in range(step_count):
            current, stages, derivatives = self._rk4_step(
                current, dt, requested_duration
            )
            for stage in stages:
                state_max = torch.maximum(
                    state_max,
                    residual_max(self.boundary_spec.state_residual(stage)),
                )
                equation_max = torch.maximum(
                    equation_max,
                    residual_max(self.boundary_spec.equation_residual(stage)),
                )
            for derivative in derivatives:
                tangent_max = torch.maximum(
                    tangent_max,
                    residual_max(self.boundary_spec.tangent_residual(derivative)),
                )
        if not return_diagnostics:
            return current
        final_residual = self.boundary_spec.state_residual(current)
        diagnostics = {
            "max_stage_boundary_state": float(state_max.detach().cpu()),
            "max_stage_boundary_equation": float(equation_max.detach().cpu()),
            "max_rhs_boundary_tangent": float(tangent_max.detach().cpu()),
            "final_boundary_state_max": float(
                residual_max(final_residual).detach().cpu()
            ),
            "final_boundary_state_rms": float(
                residual_rms(final_residual).detach().cpu()
            ),
            "final_boundary_equation_max": float(
                residual_max(self.boundary_spec.equation_residual(current))
                .detach()
                .cpu()
            ),
            "rk4_steps": step_count,
            "rhs_evaluations": 4 * step_count,
        }
        return current, diagnostics

    def forward(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        output = self.integrate(state, duration)
        if isinstance(output, tuple):
            raise AssertionError("forward unexpectedly returned diagnostics")
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _reference_project_state(
    state: torch.Tensor, spec: BoundaryFamilySpec
) -> torch.Tensor:
    """Independent explicit boundary reconstruction for reference data."""

    _check_state(state)
    projected = state.clone()
    if spec.family == "inhomogeneous_dirichlet":
        projected[:, 0] = spec.dirichlet_left
        projected[:, -1] = spec.dirichlet_right
    elif spec.family == "homogeneous_neumann":
        projected[:, 0] = projected[:, 1]
        projected[:, -1] = projected[:, -2]
    elif spec.family == "robin":
        scale = spec.robin_beta / spec.dx
        denominator = spec.robin_alpha + scale
        projected[:, 0] = (
            spec.robin_gamma_left + scale * projected[:, 1]
        ) / denominator
        projected[:, -1] = (
            spec.robin_gamma_right + scale * projected[:, -2]
        ) / denominator
    else:
        raise ValueError(f"unsupported reference boundary family: {spec.family}")
    return projected


def _reference_project_tangent(
    tangent: torch.Tensor, spec: BoundaryFamilySpec
) -> torch.Tensor:
    """Independent tangent reconstruction for the reference method of lines."""

    _check_state(tangent)
    projected = tangent.clone()
    if spec.family == "inhomogeneous_dirichlet":
        projected[:, (0, -1)] = 0.0
    elif spec.family == "homogeneous_neumann":
        projected[:, 0] = projected[:, 1]
        projected[:, -1] = projected[:, -2]
    elif spec.family == "robin":
        scale = spec.robin_beta / spec.dx
        ratio = scale / (spec.robin_alpha + scale)
        projected[:, 0] = ratio * projected[:, 1]
        projected[:, -1] = ratio * projected[:, -2]
    else:
        raise ValueError(f"unsupported reference boundary family: {spec.family}")
    return projected


def sample_initial_conditions(
    count: int,
    *,
    n_grid: int,
    modes: int,
    seed: int,
    boundary_spec: BoundaryFamilySpec,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample matched smooth interiors, then impose one affine boundary."""

    if count <= 0 or n_grid < 3 or modes <= 0:
        raise ValueError("invalid initial-condition dimensions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
    frequencies = torch.arange(1, modes + 1, dtype=torch.float64)
    basis = torch.sin(math.pi * frequencies[:, None] * x[None, :])
    decay = frequencies.pow(-1.25)
    coefficients = (
        torch.randn((count, modes), generator=generator, dtype=torch.float64)
        * decay[None, :]
    )
    states = coefficients @ basis
    maximum = states.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    amplitudes = 0.25 + 0.60 * torch.rand(
        (count, 1), generator=generator, dtype=torch.float64
    )
    states = states / maximum * amplitudes
    states = _reference_project_state(states, boundary_spec)
    return states.to(dtype=dtype)


def reference_rhs(
    state: torch.Tensor, *, diffusivity: float, boundary_spec: BoundaryFamilySpec
) -> torch.Tensor:
    """Independent centered-difference Allen-type reference vector field."""

    _check_state(state)
    derivative = torch.zeros_like(state)
    interior = state[:, 1:-1]
    laplacian = (state[:, 2:] - 2.0 * interior + state[:, :-2]) / (boundary_spec.dx**2)
    derivative[:, 1:-1] = float(diffusivity) * laplacian + interior - interior.pow(3)
    return _reference_project_tangent(derivative, boundary_spec)


def reference_solve(
    initial_state: torch.Tensor,
    duration: float,
    *,
    diffusivity: float,
    reference_dt: float,
    boundary_spec: BoundaryFamilySpec,
) -> torch.Tensor:
    """Advance the independent finite-difference reference with fixed RK4."""

    steps = _require_reference_alignment(duration, reference_dt)
    state = _reference_project_state(
        initial_state.detach().clone().to(dtype=torch.float64, device="cpu"),
        boundary_spec,
    )
    dt = float(duration) / steps
    for _ in range(steps):
        k1 = reference_rhs(state, diffusivity=diffusivity, boundary_spec=boundary_spec)
        stage2 = _reference_project_state(state + 0.5 * dt * k1, boundary_spec)
        k2 = reference_rhs(stage2, diffusivity=diffusivity, boundary_spec=boundary_spec)
        stage3 = _reference_project_state(state + 0.5 * dt * k2, boundary_spec)
        k3 = reference_rhs(stage3, diffusivity=diffusivity, boundary_spec=boundary_spec)
        stage4 = _reference_project_state(state + dt * k3, boundary_spec)
        k4 = reference_rhs(stage4, diffusivity=diffusivity, boundary_spec=boundary_spec)
        state = _reference_project_state(
            state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4),
            boundary_spec,
        )
    return state.to(dtype=initial_state.dtype)


def build_reference_cache(config: BoundaryFamilyWaveConfig) -> dict[str, Any]:
    """Generate one immutable cache for one boundary family."""

    config.validate()
    spec = config.boundary_spec
    train_u0 = sample_initial_conditions(
        config.n_train,
        n_grid=config.n_grid,
        modes=config.sine_modes,
        seed=config.data_seed + 11,
        boundary_spec=spec,
    )
    val_u0 = sample_initial_conditions(
        config.n_val,
        n_grid=config.n_grid,
        modes=config.sine_modes,
        seed=config.data_seed + 23,
        boundary_spec=spec,
    )
    test_u0 = sample_initial_conditions(
        config.n_test,
        n_grid=config.n_grid,
        modes=config.sine_modes,
        seed=config.data_seed + 37,
        boundary_spec=spec,
    )
    repeats = config.n_train // len(config.train_taus)
    train_tau = torch.tensor(list(config.train_taus) * repeats, dtype=torch.float32)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.data_seed + 41)
    permutation = torch.randperm(config.n_train, generator=generator)
    train_tau = train_tau[permutation]
    train_target = torch.empty_like(train_u0)
    for tau in config.train_taus:
        mask = torch.isclose(train_tau, torch.tensor(tau, dtype=train_tau.dtype))
        train_target[mask] = reference_solve(
            train_u0[mask],
            tau,
            diffusivity=config.diffusivity,
            reference_dt=config.reference_dt,
            boundary_spec=spec,
        )
    val_targets = {
        time_key(tau): reference_solve(
            val_u0,
            tau,
            diffusivity=config.diffusivity,
            reference_dt=config.reference_dt,
            boundary_spec=spec,
        )
        for tau in config.train_taus
    }
    test_targets = {
        time_key(horizon): reference_solve(
            test_u0,
            horizon,
            diffusivity=config.diffusivity,
            reference_dt=config.reference_dt,
            boundary_spec=spec,
        )
        for horizon in config.reference_horizons
    }
    cache = {
        "schema_version": config.schema_version,
        "experiment": config.experiment,
        "data_identity": config.data_identity(),
        "train_u0": train_u0,
        "train_tau": train_tau,
        "train_target": train_target,
        "val_u0": val_u0,
        "val_targets": val_targets,
        "test_u0": test_u0,
        "test_targets": test_targets,
    }
    validate_reference_cache(cache, config)
    return cache


def validate_reference_cache(
    cache: dict[str, Any], config: BoundaryFamilyWaveConfig
) -> dict[str, Any]:
    if cache.get("data_identity") != config.data_identity():
        raise ValueError("reference cache identity differs from selected profile")
    expected_shapes = {
        "train_u0": (config.n_train, config.n_grid),
        "train_tau": (config.n_train,),
        "train_target": (config.n_train, config.n_grid),
        "val_u0": (config.n_val, config.n_grid),
        "test_u0": (config.n_test, config.n_grid),
    }
    for name, shape in expected_shapes.items():
        value = cache.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"cache tensor {name} has unexpected shape")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"cache tensor {name} contains non-finite values")
    expected_val = {time_key(tau) for tau in config.train_taus}
    expected_test = {time_key(value) for value in config.reference_horizons}
    if set(cache.get("val_targets", {})) != expected_val:
        raise ValueError("validation target grid is incomplete")
    if set(cache.get("test_targets", {})) != expected_test:
        raise ValueError("test target grid is incomplete")
    state_tensors: list[torch.Tensor] = [
        cache["train_u0"],
        cache["train_target"],
        cache["val_u0"],
        cache["test_u0"],
        *cache["val_targets"].values(),
        *cache["test_targets"].values(),
    ]
    maximum_state_residual = max(
        float(residual_max(config.boundary_spec.state_residual(value)))
        for value in state_tensors
    )
    maximum_equation_residual = max(
        float(residual_max(config.boundary_spec.equation_residual(value)))
        for value in state_tensors
    )
    if (
        max(maximum_state_residual, maximum_equation_residual)
        > config.boundary_threshold
    ):
        raise ValueError("reference cache violates its frozen boundary condition")
    counts = {
        time_key(tau): int(
            torch.isclose(
                cache["train_tau"],
                torch.tensor(tau, dtype=cache["train_tau"].dtype),
            ).sum()
        )
        for tau in config.train_taus
    }
    if set(counts.values()) != {config.n_train // len(config.train_taus)}:
        raise ValueError("training duration allocation is not balanced")
    return {
        "boundary_family": config.boundary_family,
        "train_shape": list(cache["train_u0"].shape),
        "validation_shape": list(cache["val_u0"].shape),
        "test_shape": list(cache["test_u0"].shape),
        "duration_counts": counts,
        "maximum_reference_state_residual": maximum_state_residual,
        "maximum_reference_equation_residual": maximum_equation_residual,
    }


def mean_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction - target).flatten(1).norm(dim=1)
    denominator = target.flatten(1).norm(dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean().detach().cpu())


def geometric_mean(values: Iterable[float], *, floor: float = METRIC_FLOOR) -> float:
    sequence = [float(value) for value in values]
    if not sequence:
        raise ValueError("geometric mean requires values")
    if any(not math.isfinite(value) or value < 0 for value in sequence):
        raise ValueError("geometric mean received an invalid metric")
    return math.exp(
        sum(math.log(max(value, floor)) for value in sequence) / len(sequence)
    )


def partition_key(partition: Sequence[float]) -> str:
    return "+".join(time_key(value) for value in partition)


def composed_partition(
    model: BoundaryFamilyFlow,
    initial_state: torch.Tensor,
    *,
    partition: Sequence[float],
    steps_per_segment: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not partition or any(float(value) <= 0 for value in partition):
        raise ValueError("partition durations must be positive")
    current = initial_state
    diagnostics = {
        "max_stage_boundary_state": 0.0,
        "max_stage_boundary_equation": 0.0,
        "max_rhs_boundary_tangent": 0.0,
        "rk4_steps": 0,
        "rhs_evaluations": 0,
    }
    for segment in partition:
        result = model.integrate(
            current,
            float(segment),
            map_duration=float(segment),
            steps=steps_per_segment,
            return_diagnostics=True,
        )
        if not isinstance(result, tuple):
            raise AssertionError("partition integration did not return diagnostics")
        current, segment_diagnostics = result
        for name in (
            "max_stage_boundary_state",
            "max_stage_boundary_equation",
            "max_rhs_boundary_tangent",
        ):
            diagnostics[name] = max(diagnostics[name], segment_diagnostics[name])
        diagnostics["rk4_steps"] += int(segment_diagnostics["rk4_steps"])
        diagnostics["rhs_evaluations"] += int(segment_diagnostics["rhs_evaluations"])
    final_residual = model.boundary_spec.state_residual(current)
    diagnostics.update(
        {
            "final_boundary_state_max": float(
                residual_max(final_residual).detach().cpu()
            ),
            "final_boundary_state_rms": float(
                residual_rms(final_residual).detach().cpu()
            ),
            "final_boundary_equation_max": float(
                residual_max(model.boundary_spec.equation_residual(current))
                .detach()
                .cpu()
            ),
        }
    )
    return current, diagnostics


METRIC_FIELDS = (
    "evaluation_kind",
    "partition_id",
    "partition_json",
    "horizon",
    "segment_count",
    "steps_per_segment",
    "sample_count",
    "equal_work_composition_defect",
    "production_work_composition_defect",
    "rollout_mse",
    "rollout_interior_mse",
    "rollout_relative_l2",
    "direct_equal_work_mse",
    "composed_boundary_state_max",
    "composed_boundary_state_rms",
    "composed_boundary_equation_max",
    "max_stage_boundary_state",
    "max_stage_boundary_equation",
    "rhs_boundary_tangent_max",
    "composed_rhs_evaluations",
    "direct_equal_rhs_evaluations",
    "direct_production_rhs_evaluations",
    "equal_rhs_work",
)


def expected_metric_identities(
    uniform_steps: int = 8,
) -> set[tuple[str, str, int]]:
    identities: set[tuple[str, str, int]] = set()
    for tau, horizon in UNIFORM_PRIMARY_PAIRS:
        partition = (tau,) * composition_depth(tau, horizon)
        identities.add(("uniform_primary", partition_key(partition), uniform_steps))
    for tau, horizon in LONG_ROLLOUT_PAIRS:
        partition = (tau,) * composition_depth(tau, horizon)
        identities.add(("long_rollout", partition_key(partition), uniform_steps))
    for partition in NONUNIFORM_PARTITIONS:
        for steps in REFINEMENT_STEPS:
            identities.add(("nonuniform_refinement", partition_key(partition), steps))
    return identities


def _evaluate_partition(
    model: BoundaryFamilyFlow,
    initial_state: torch.Tensor,
    reference: torch.Tensor,
    *,
    evaluation_kind: str,
    partition: Sequence[float],
    steps_per_segment: int,
) -> dict[str, Any]:
    horizon = float(sum(partition))
    composed, composed_diagnostics = composed_partition(
        model,
        initial_state,
        partition=partition,
        steps_per_segment=steps_per_segment,
    )
    direct_equal_result = model.integrate(
        initial_state,
        horizon,
        map_duration=horizon,
        steps=steps_per_segment * len(partition),
        return_diagnostics=True,
    )
    direct_production_result = model.integrate(
        initial_state,
        horizon,
        map_duration=horizon,
        steps=steps_per_segment,
        return_diagnostics=True,
    )
    if not isinstance(direct_equal_result, tuple) or not isinstance(
        direct_production_result, tuple
    ):
        raise AssertionError("direct integration did not return diagnostics")
    direct_equal, direct_equal_diagnostics = direct_equal_result
    direct_production, direct_production_diagnostics = direct_production_result
    return {
        "evaluation_kind": evaluation_kind,
        "partition_id": partition_key(partition),
        "partition_json": json.dumps(list(partition), separators=(",", ":")),
        "horizon": horizon,
        "segment_count": len(partition),
        "steps_per_segment": steps_per_segment,
        "sample_count": int(initial_state.shape[0]),
        "equal_work_composition_defect": mean_relative_l2(direct_equal, composed),
        "production_work_composition_defect": mean_relative_l2(
            direct_production, composed
        ),
        "rollout_mse": float(F.mse_loss(composed, reference).detach().cpu()),
        "rollout_interior_mse": float(
            F.mse_loss(composed[:, 1:-1], reference[:, 1:-1]).detach().cpu()
        ),
        "rollout_relative_l2": mean_relative_l2(composed, reference),
        "direct_equal_work_mse": float(
            F.mse_loss(direct_equal, reference).detach().cpu()
        ),
        "composed_boundary_state_max": composed_diagnostics["final_boundary_state_max"],
        "composed_boundary_state_rms": composed_diagnostics["final_boundary_state_rms"],
        "composed_boundary_equation_max": composed_diagnostics[
            "final_boundary_equation_max"
        ],
        "max_stage_boundary_state": max(
            composed_diagnostics["max_stage_boundary_state"],
            direct_equal_diagnostics["max_stage_boundary_state"],
            direct_production_diagnostics["max_stage_boundary_state"],
        ),
        "max_stage_boundary_equation": max(
            composed_diagnostics["max_stage_boundary_equation"],
            direct_equal_diagnostics["max_stage_boundary_equation"],
            direct_production_diagnostics["max_stage_boundary_equation"],
        ),
        "rhs_boundary_tangent_max": max(
            composed_diagnostics["max_rhs_boundary_tangent"],
            direct_equal_diagnostics["max_rhs_boundary_tangent"],
            direct_production_diagnostics["max_rhs_boundary_tangent"],
        ),
        "composed_rhs_evaluations": composed_diagnostics["rhs_evaluations"],
        "direct_equal_rhs_evaluations": direct_equal_diagnostics["rhs_evaluations"],
        "direct_production_rhs_evaluations": direct_production_diagnostics[
            "rhs_evaluations"
        ],
        "equal_rhs_work": composed_diagnostics["rhs_evaluations"]
        == direct_equal_diagnostics["rhs_evaluations"],
    }


def evaluate_model(
    model: BoundaryFamilyFlow,
    cache: dict[str, Any],
    config: BoundaryFamilyWaveConfig,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate uniform, nonuniform, long-rollout, and boundary-stress grids."""

    model.eval()
    test_u0 = cache["test_u0"].to(device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for evaluation_kind, pairs in (
            ("uniform_primary", UNIFORM_PRIMARY_PAIRS),
            ("long_rollout", LONG_ROLLOUT_PAIRS),
        ):
            for tau, horizon in pairs:
                partition = (tau,) * composition_depth(tau, horizon)
                reference = cache["test_targets"][time_key(horizon)].to(device)
                rows.append(
                    _evaluate_partition(
                        model,
                        test_u0,
                        reference,
                        evaluation_kind=evaluation_kind,
                        partition=partition,
                        steps_per_segment=config.neural_rk4_steps,
                    )
                )
        reference_016 = cache["test_targets"][time_key(0.16)].to(device)
        for partition in NONUNIFORM_PARTITIONS:
            for steps in REFINEMENT_STEPS:
                rows.append(
                    _evaluate_partition(
                        model,
                        test_u0,
                        reference_016,
                        evaluation_kind="nonuniform_refinement",
                        partition=partition,
                        steps_per_segment=steps,
                    )
                )

        stress = test_u0[: config.stress_samples].clone()
        stress[:, 0] += config.stress_amplitude
        stress[:, -1] -= config.stress_amplitude
        perturbed_residual = model.boundary_spec.state_residual(stress)
        projected_stress = model.project_state(stress)
        projected_residual = model.boundary_spec.state_residual(projected_stress)
        stress_partition = (config.stress_tau,) * composition_depth(
            config.stress_tau, config.stress_horizon
        )
        stress_output, stress_diagnostics = composed_partition(
            model,
            stress,
            partition=stress_partition,
            steps_per_segment=config.neural_rk4_steps,
        )
        stress_summary = {
            "sample_count": config.stress_samples,
            "amplitude": config.stress_amplitude,
            "partition": list(stress_partition),
            "perturbed_input_state_max": float(
                residual_max(perturbed_residual).detach().cpu()
            ),
            "projected_input_state_max": float(
                residual_max(projected_residual).detach().cpu()
            ),
            "final_state_max": float(
                residual_max(model.boundary_spec.state_residual(stress_output))
                .detach()
                .cpu()
            ),
            "final_equation_max": float(
                residual_max(model.boundary_spec.equation_residual(stress_output))
                .detach()
                .cpu()
            ),
            "max_stage_boundary_state": stress_diagnostics["max_stage_boundary_state"],
            "max_stage_boundary_equation": stress_diagnostics[
                "max_stage_boundary_equation"
            ],
            "max_rhs_boundary_tangent": stress_diagnostics["max_rhs_boundary_tangent"],
        }

    identities = {
        (row["evaluation_kind"], row["partition_id"], row["steps_per_segment"])
        for row in rows
    }
    if identities != expected_metric_identities(config.neural_rk4_steps) or len(
        rows
    ) != len(identities):
        raise AssertionError("evaluation grid is incomplete or duplicated")
    if not all(row["equal_rhs_work"] for row in rows):
        raise AssertionError("equal-work RHS counts differ")
    uniform_primary = [
        row for row in rows if row["evaluation_kind"] == "uniform_primary"
    ]
    long_rollout = [row for row in rows if row["evaluation_kind"] == "long_rollout"]
    nonuniform_primary = [
        row
        for row in rows
        if row["evaluation_kind"] == "nonuniform_refinement"
        and row["steps_per_segment"] == 8
    ]
    refinement = {
        str(steps): geometric_mean(
            row["equal_work_composition_defect"]
            for row in rows
            if row["evaluation_kind"] == "nonuniform_refinement"
            and row["steps_per_segment"] == steps
        )
        for steps in REFINEMENT_STEPS
    }
    normal_boundary_evidence = max(
        max(
            row["composed_boundary_state_max"],
            row["composed_boundary_equation_max"],
            row["max_stage_boundary_state"],
            row["max_stage_boundary_equation"],
            row["rhs_boundary_tangent_max"],
        )
        for row in rows
    )
    stress_boundary_evidence = max(
        stress_summary["projected_input_state_max"],
        stress_summary["final_state_max"],
        stress_summary["final_equation_max"],
        stress_summary["max_stage_boundary_state"],
        stress_summary["max_stage_boundary_equation"],
        stress_summary["max_rhs_boundary_tangent"],
    )
    summary = {
        "uniform_primary_defect_geometric_mean": geometric_mean(
            row["equal_work_composition_defect"] for row in uniform_primary
        ),
        "primary_nonuniform_defect_geometric_mean": geometric_mean(
            row["equal_work_composition_defect"] for row in nonuniform_primary
        ),
        "long_rollout_mse_geometric_mean": geometric_mean(
            row["rollout_mse"] for row in long_rollout
        ),
        "nonuniform_refinement_defect_geometric_means": refinement,
        "maximum_normal_boundary_evidence": normal_boundary_evidence,
        "maximum_hard_boundary_evidence": max(
            normal_boundary_evidence, stress_boundary_evidence
        ),
        "boundary_stress": stress_summary,
    }
    return rows, summary
