"""Boundary-compatible neural flows for the exploratory Wave 1 experiment.

The module has one narrow responsibility: represent the orthogonal product of
a spatial boundary intervention and a temporal-conditioning intervention.  It
also owns the independent reference discretization and metric kernels needed
to test those two properties without importing the repository's periodic PDE
models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


BOUNDARY_MODES = ("hard_dirichlet", "unconstrained")
TEMPORAL_MODES = ("autonomous", "query_time")
TRAINING_SEEDS = (31415, 271828, 161803)
PRIMARY_SEMIGROUP_PAIRS = (
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
BOUNDARY_THRESHOLD = 1e-12
METRIC_FLOOR = 1e-16


@dataclass(frozen=True)
class BoundaryWaveConfig:
    """Frozen numerical and training settings for one Wave 1 profile."""

    schema_version: int = 1
    experiment: str = "boundary_compatible_semigroup_wave1"
    exploratory: bool = True
    smoke_only: bool = False
    data_seed: int = 424242
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
    stress_samples: int = 64
    stress_amplitude: float = 0.25
    stress_tau: float = 0.04
    stress_horizon: float = 0.16
    boundary_threshold: float = BOUNDARY_THRESHOLD

    def validate(self) -> None:
        """Reject settings that would change the meaning of the frozen grid."""

        if self.n_grid < 5 or self.n_grid % 2 == 0:
            raise ValueError("n_grid must be an odd integer at least five")
        positive = {
            "domain_length": self.domain_length,
            "diffusivity": self.diffusivity,
            "reference_dt": self.reference_dt,
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
            "stress_samples": self.stress_samples,
            "stress_amplitude": self.stress_amplitude,
            "stress_tau": self.stress_tau,
            "stress_horizon": self.stress_horizon,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if self.n_train % len(self.train_taus) != 0:
            raise ValueError("n_train must balance the frozen training durations")
        if self.stress_samples > self.n_test:
            raise ValueError("stress_samples may not exceed n_test")
        for tau in self.train_taus:
            _require_reference_alignment(tau, self.reference_dt)
        for tau, horizon in (*PRIMARY_SEMIGROUP_PAIRS, *LONG_ROLLOUT_PAIRS):
            composition_depth(tau, horizon)
            _require_reference_alignment(horizon, self.reference_dt)
        composition_depth(self.stress_tau, self.stress_horizon)

    @property
    def dx(self) -> float:
        return self.domain_length / (self.n_grid - 1)

    @property
    def reference_horizons(self) -> tuple[float, ...]:
        values = set(self.train_taus)
        values.update(horizon for _tau, horizon in PRIMARY_SEMIGROUP_PAIRS)
        values.update(horizon for _tau, horizon in LONG_ROLLOUT_PAIRS)
        return tuple(sorted(values))

    def data_identity(self) -> dict[str, Any]:
        """Return exactly the fields that define the immutable data cache."""

        return {
            "schema_version": self.schema_version,
            "experiment": self.experiment,
            "smoke_only": self.smoke_only,
            "data_seed": self.data_seed,
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
        payload["primary_semigroup_pairs"] = [
            list(pair) for pair in PRIMARY_SEMIGROUP_PAIRS
        ]
        payload["long_rollout_pairs"] = [list(pair) for pair in LONG_ROLLOUT_PAIRS]
        payload["reference_horizons"] = list(self.reference_horizons)
        payload["dx"] = self.dx
        return payload


def wave_config(*, smoke_only: bool = False) -> BoundaryWaveConfig:
    """Return the full frozen profile or a visibly reduced smoke profile."""

    config = BoundaryWaveConfig()
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


def time_key(value: float) -> str:
    """Use one stable textual key for cache times and CSV identities."""

    return format(float(value), ".12g")


def composition_depth(tau: float, horizon: float) -> int:
    """Return the integer number of base maps in one rollout."""

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


def project_homogeneous_dirichlet(state: torch.Tensor) -> torch.Tensor:
    """Project a batched grid state onto the zero-endpoint subspace."""

    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError("state must have shape [batch, grid] with grid >= 3")
    mask = torch.ones((1, state.shape[1]), dtype=state.dtype, device=state.device)
    mask[:, (0, -1)] = 0
    return state * mask


def endpoint_max(state: torch.Tensor) -> torch.Tensor:
    """Return the maximum absolute endpoint magnitude of a batched state."""

    return state[:, (0, -1)].abs().max()


def endpoint_rms(state: torch.Tensor) -> torch.Tensor:
    """Return the RMS endpoint magnitude of a batched state."""

    return state[:, (0, -1)].square().mean().sqrt()


class BoundarySemigroupFlow(nn.Module):
    """A local neural ODE with independent spatial and temporal interventions.

    All four experiment cells instantiate the exact same parameter tensors.
    Hard Dirichlet compatibility is implemented only through a fixed mask, and
    temporal conditioning changes only the value of one already-present input
    feature.
    """

    def __init__(
        self,
        *,
        n_grid: int,
        hidden_width: int,
        boundary_mode: str,
        temporal_mode: str,
        query_time_scale: float,
        rk4_steps: int,
    ) -> None:
        super().__init__()
        if boundary_mode not in BOUNDARY_MODES:
            raise ValueError(f"unsupported boundary_mode: {boundary_mode}")
        if temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"unsupported temporal_mode: {temporal_mode}")
        if n_grid < 3 or hidden_width <= 0 or query_time_scale <= 0 or rk4_steps <= 0:
            raise ValueError("invalid model dimensions or integration settings")
        self.n_grid = int(n_grid)
        self.hidden_width = int(hidden_width)
        self.boundary_mode = str(boundary_mode)
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
        mask = torch.ones((1, n_grid), dtype=torch.float32)
        mask[:, (0, -1)] = 0
        self.register_buffer("dirichlet_mask", mask, persistent=False)

    def project_state(self, state: torch.Tensor) -> torch.Tensor:
        if self.boundary_mode == "hard_dirichlet":
            return state * self.dirichlet_mask.to(dtype=state.dtype)
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
            raise ValueError("duration values must be strictly positive")
        return value.reshape(-1, 1)

    def vector_field(
        self, state: torch.Tensor, map_duration: float | torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.n_grid:
            raise ValueError(
                f"expected state shape [batch, {self.n_grid}], got {tuple(state.shape)}"
            )
        admissible_state = self.project_state(state)
        patches = F.pad(admissible_state, (1, 1), mode="constant", value=0.0).unfold(
            dimension=1, size=3, step=1
        )
        duration = self._duration_tensor(map_duration, admissible_state)
        if self.temporal_mode == "autonomous":
            temporal_feature = torch.ones_like(duration)
        else:
            temporal_feature = duration / self.query_time_scale
        temporal_feature = temporal_feature.unsqueeze(1).expand(-1, self.n_grid, -1)
        features = torch.cat((patches, temporal_feature), dim=-1)
        derivative = self.vector_field_net(features).squeeze(-1)
        if self.boundary_mode == "hard_dirichlet":
            derivative = derivative * self.dirichlet_mask.to(dtype=derivative.dtype)
        return derivative

    def _rk4_step(
        self,
        state: torch.Tensor,
        dt: torch.Tensor,
        map_duration: float | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
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
        return next_state, (stage2, stage3, stage4, next_state)

    def integrate(
        self,
        state: torch.Tensor,
        duration: float | torch.Tensor,
        *,
        map_duration: float | torch.Tensor | None = None,
        steps: int | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Integrate one requested map, optionally reporting stage residuals."""

        step_count = self.rk4_steps if steps is None else int(steps)
        if step_count <= 0:
            raise ValueError("steps must be strictly positive")
        requested_duration = duration if map_duration is None else map_duration
        duration_tensor = self._duration_tensor(duration, state)
        dt = duration_tensor / step_count
        current = self.project_state(state)
        max_stage_boundary = endpoint_max(current)
        for _ in range(step_count):
            current, stages = self._rk4_step(current, dt, requested_duration)
            for stage in stages:
                max_stage_boundary = torch.maximum(
                    max_stage_boundary, endpoint_max(stage)
                )
        if not return_diagnostics:
            return current
        diagnostics = {
            "max_stage_boundary": float(max_stage_boundary.detach().cpu()),
            "final_boundary_max": float(endpoint_max(current).detach().cpu()),
            "final_boundary_rms": float(endpoint_rms(current).detach().cpu()),
            "rk4_steps": step_count,
            "rhs_evaluations": 4 * step_count,
        }
        return current, diagnostics

    def forward(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        output = self.integrate(state, duration)
        if isinstance(output, tuple):
            raise AssertionError("forward unexpectedly received diagnostics")
        return output


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def parameter_fingerprint(model: nn.Module) -> str:
    """Hash tensor names, shapes, dtypes, and bytes without archive metadata."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(cpu_tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def sample_sine_initial_conditions(
    count: int,
    *,
    n_grid: int,
    modes: int,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample mixed-frequency zero-Dirichlet states reproducibly on CPU."""

    if count <= 0 or n_grid < 3 or modes <= 0:
        raise ValueError("invalid initial-condition dimensions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
    frequencies = torch.arange(1, modes + 1, dtype=torch.float64)
    basis = torch.sin(math.pi * frequencies[:, None] * x[None, :])
    decay = frequencies.pow(-1.25)
    coefficients = torch.randn(
        (count, modes), generator=generator, dtype=torch.float64
    ) * decay[None, :]
    states = coefficients @ basis
    maximum = states.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    amplitudes = 0.25 + 0.60 * torch.rand(
        (count, 1), generator=generator, dtype=torch.float64
    )
    states = states / maximum * amplitudes
    states[:, 0] = 0.0
    states[:, -1] = 0.0
    return states.to(dtype=dtype)


def reference_reaction_diffusion_rhs(
    state: torch.Tensor, *, dx: float, diffusivity: float
) -> torch.Tensor:
    """Independent centered-difference Allen-type reference vector field."""

    if state.ndim != 2 or state.shape[1] < 3:
        raise ValueError("reference state must have shape [batch, grid]")
    derivative = torch.zeros_like(state)
    interior = state[:, 1:-1]
    laplacian = (
        state[:, 2:] - 2.0 * interior + state[:, :-2]
    ) / (float(dx) ** 2)
    derivative[:, 1:-1] = float(diffusivity) * laplacian + interior - interior.pow(3)
    return derivative


def reference_solve(
    initial_state: torch.Tensor,
    duration: float,
    *,
    dx: float,
    diffusivity: float,
    reference_dt: float,
) -> torch.Tensor:
    """Advance the independent finite-difference reference with fixed RK4."""

    steps = _require_reference_alignment(duration, reference_dt)
    state = initial_state.detach().clone().to(dtype=torch.float64, device="cpu")
    state[:, 0] = 0.0
    state[:, -1] = 0.0
    dt = float(duration) / steps
    for _ in range(steps):
        k1 = reference_reaction_diffusion_rhs(
            state, dx=dx, diffusivity=diffusivity
        )
        stage2 = state + 0.5 * dt * k1
        stage2[:, (0, -1)] = 0.0
        k2 = reference_reaction_diffusion_rhs(
            stage2, dx=dx, diffusivity=diffusivity
        )
        stage3 = state + 0.5 * dt * k2
        stage3[:, (0, -1)] = 0.0
        k3 = reference_reaction_diffusion_rhs(
            stage3, dx=dx, diffusivity=diffusivity
        )
        stage4 = state + dt * k3
        stage4[:, (0, -1)] = 0.0
        k4 = reference_reaction_diffusion_rhs(
            stage4, dx=dx, diffusivity=diffusivity
        )
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state[:, (0, -1)] = 0.0
    return state.to(dtype=initial_state.dtype)


def build_reference_cache(config: BoundaryWaveConfig) -> dict[str, Any]:
    """Generate the one immutable cache shared by all 12 paired cells."""

    config.validate()
    train_u0 = sample_sine_initial_conditions(
        config.n_train,
        n_grid=config.n_grid,
        modes=config.sine_modes,
        seed=config.data_seed + 11,
    )
    val_u0 = sample_sine_initial_conditions(
        config.n_val,
        n_grid=config.n_grid,
        modes=config.sine_modes,
        seed=config.data_seed + 23,
    )
    test_u0 = sample_sine_initial_conditions(
        config.n_test,
        n_grid=config.n_grid,
        modes=config.sine_modes,
        seed=config.data_seed + 37,
    )
    repeats = config.n_train // len(config.train_taus)
    train_tau = torch.tensor(
        list(config.train_taus) * repeats, dtype=torch.float32
    )
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
            dx=config.dx,
            diffusivity=config.diffusivity,
            reference_dt=config.reference_dt,
        )
    val_targets = {
        time_key(tau): reference_solve(
            val_u0,
            tau,
            dx=config.dx,
            diffusivity=config.diffusivity,
            reference_dt=config.reference_dt,
        )
        for tau in config.train_taus
    }
    test_targets = {
        time_key(horizon): reference_solve(
            test_u0,
            horizon,
            dx=config.dx,
            diffusivity=config.diffusivity,
            reference_dt=config.reference_dt,
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
    cache: dict[str, Any], config: BoundaryWaveConfig
) -> dict[str, Any]:
    """Validate identity, shape, finiteness, and zero reference endpoints."""

    if cache.get("data_identity") != config.data_identity():
        raise ValueError("reference cache identity differs from the selected profile")
    tensors = {
        "train_u0": (config.n_train, config.n_grid),
        "train_tau": (config.n_train,),
        "train_target": (config.n_train, config.n_grid),
        "val_u0": (config.n_val, config.n_grid),
        "test_u0": (config.n_test, config.n_grid),
    }
    for name, shape in tensors.items():
        value = cache.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != shape:
            raise ValueError(f"cache tensor {name} has an unexpected shape")
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
    if any(not bool(torch.isfinite(value).all()) for value in state_tensors):
        raise ValueError("reference cache contains non-finite states")
    maximum_endpoint = max(float(endpoint_max(value)) for value in state_tensors)
    if maximum_endpoint != 0.0:
        raise ValueError("reference cache does not satisfy exact zero endpoints")
    counts = {
        time_key(tau): int(
            torch.isclose(
                cache["train_tau"],
                torch.tensor(tau, dtype=cache["train_tau"].dtype),
            ).sum()
        )
        for tau in config.train_taus
    }
    expected_count = config.n_train // len(config.train_taus)
    if set(counts.values()) != {expected_count}:
        raise ValueError("training duration allocation is not exactly balanced")
    return {
        "train_shape": list(cache["train_u0"].shape),
        "validation_shape": list(cache["val_u0"].shape),
        "test_shape": list(cache["test_u0"].shape),
        "duration_counts": counts,
        "maximum_reference_endpoint": maximum_endpoint,
    }


def composed_rollout(
    model: BoundarySemigroupFlow,
    initial_state: torch.Tensor,
    *,
    tau: float,
    horizon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one base map repeatedly and accumulate stage diagnostics."""

    depth = composition_depth(tau, horizon)
    current = initial_state
    max_stage_boundary = 0.0
    total_rhs = 0
    for _ in range(depth):
        output = model.integrate(
            current,
            tau,
            map_duration=tau,
            steps=model.rk4_steps,
            return_diagnostics=True,
        )
        if not isinstance(output, tuple):
            raise AssertionError("composed rollout did not return diagnostics")
        current, diagnostics = output
        max_stage_boundary = max(
            max_stage_boundary, float(diagnostics["max_stage_boundary"])
        )
        total_rhs += int(diagnostics["rhs_evaluations"])
    return current, {
        "composition_depth": depth,
        "max_stage_boundary": max_stage_boundary,
        "final_boundary_max": float(endpoint_max(current).detach().cpu()),
        "final_boundary_rms": float(endpoint_rms(current).detach().cpu()),
        "rk4_steps": depth * model.rk4_steps,
        "rhs_evaluations": total_rhs,
    }


def mean_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction - target).flatten(1).norm(dim=1)
    denominator = target.flatten(1).norm(dim=1).clamp_min(1e-12)
    return float((numerator / denominator).mean().detach().cpu())


def geometric_mean(values: Iterable[float], *, floor: float = METRIC_FLOOR) -> float:
    sequence = [float(value) for value in values]
    if not sequence:
        raise ValueError("geometric mean requires at least one value")
    if any(not math.isfinite(value) or value < 0 for value in sequence):
        raise ValueError("geometric mean received an invalid metric")
    return math.exp(sum(math.log(max(value, floor)) for value in sequence) / len(sequence))


def evaluate_model(
    model: BoundarySemigroupFlow,
    cache: dict[str, Any],
    config: BoundaryWaveConfig,
    *,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the frozen semigroup, rollout, and boundary-stress grids."""

    model.eval()
    test_u0 = cache["test_u0"].to(device)
    rows: list[dict[str, Any]] = []
    pairs: Sequence[tuple[str, tuple[float, float]]] = (
        *(('primary_semigroup', pair) for pair in PRIMARY_SEMIGROUP_PAIRS),
        *(('long_rollout', pair) for pair in LONG_ROLLOUT_PAIRS),
    )
    with torch.no_grad():
        for evaluation_kind, (tau, horizon) in pairs:
            depth = composition_depth(tau, horizon)
            reference = cache["test_targets"][time_key(horizon)].to(device)
            composed, composed_diagnostics = composed_rollout(
                model, test_u0, tau=tau, horizon=horizon
            )
            direct_equal_output = model.integrate(
                test_u0,
                horizon,
                map_duration=horizon,
                steps=model.rk4_steps * depth,
                return_diagnostics=True,
            )
            direct_production_output = model.integrate(
                test_u0,
                horizon,
                map_duration=horizon,
                steps=model.rk4_steps,
                return_diagnostics=True,
            )
            if not isinstance(direct_equal_output, tuple) or not isinstance(
                direct_production_output, tuple
            ):
                raise AssertionError("direct integration did not return diagnostics")
            direct_equal, direct_equal_diagnostics = direct_equal_output
            direct_production, direct_production_diagnostics = direct_production_output
            equal_work_defect = mean_relative_l2(direct_equal, composed)
            production_work_defect = mean_relative_l2(direct_production, composed)
            full_mse = float(F.mse_loss(composed, reference).detach().cpu())
            interior_mse = float(
                F.mse_loss(composed[:, 1:-1], reference[:, 1:-1]).detach().cpu()
            )
            rhs_boundary = float(
                endpoint_max(model.vector_field(test_u0, tau)).detach().cpu()
            )
            rows.append(
                {
                    "evaluation_kind": evaluation_kind,
                    "tau": tau,
                    "horizon": horizon,
                    "composition_depth": depth,
                    "sample_count": config.n_test,
                    "equal_work_composition_defect": equal_work_defect,
                    "production_work_composition_defect": production_work_defect,
                    "rollout_mse": full_mse,
                    "rollout_interior_mse": interior_mse,
                    "rollout_relative_l2": mean_relative_l2(composed, reference),
                    "direct_equal_work_mse": float(
                        F.mse_loss(direct_equal, reference).detach().cpu()
                    ),
                    "composed_boundary_max": composed_diagnostics[
                        "final_boundary_max"
                    ],
                    "composed_boundary_rms": composed_diagnostics[
                        "final_boundary_rms"
                    ],
                    "max_stage_boundary": max(
                        composed_diagnostics["max_stage_boundary"],
                        direct_equal_diagnostics["max_stage_boundary"],
                        direct_production_diagnostics["max_stage_boundary"],
                    ),
                    "rhs_boundary_max": rhs_boundary,
                    "composed_rhs_evaluations": composed_diagnostics[
                        "rhs_evaluations"
                    ],
                    "direct_equal_rhs_evaluations": direct_equal_diagnostics[
                        "rhs_evaluations"
                    ],
                    "direct_production_rhs_evaluations": direct_production_diagnostics[
                        "rhs_evaluations"
                    ],
                    "equal_rhs_work": (
                        composed_diagnostics["rhs_evaluations"]
                        == direct_equal_diagnostics["rhs_evaluations"]
                    ),
                }
            )

        stress = test_u0[: config.stress_samples].clone()
        signs = torch.where(
            torch.arange(config.stress_samples, device=device) % 2 == 0,
            torch.ones(config.stress_samples, device=device),
            -torch.ones(config.stress_samples, device=device),
        )
        stress[:, 0] = config.stress_amplitude * signs
        stress[:, -1] = -config.stress_amplitude * signs
        first_output = model.integrate(
            stress,
            config.stress_tau,
            map_duration=config.stress_tau,
            return_diagnostics=True,
        )
        stress_output, stress_diagnostics = composed_rollout(
            model,
            stress,
            tau=config.stress_tau,
            horizon=config.stress_horizon,
        )
        if not isinstance(first_output, tuple):
            raise AssertionError("stress integration did not return diagnostics")
        first_state, first_diagnostics = first_output
        stress_rhs_boundary = float(
            endpoint_max(model.vector_field(stress, config.stress_tau)).detach().cpu()
        )

    primary_rows = [row for row in rows if row["evaluation_kind"] == "primary_semigroup"]
    long_rows = [row for row in rows if row["evaluation_kind"] == "long_rollout"]
    summary = {
        "primary_equal_work_defect_geometric_mean": geometric_mean(
            row["equal_work_composition_defect"] for row in primary_rows
        ),
        "primary_production_work_defect_geometric_mean": geometric_mean(
            row["production_work_composition_defect"] for row in primary_rows
        ),
        "long_rollout_mse_geometric_mean": geometric_mean(
            row["rollout_mse"] for row in long_rows
        ),
        "long_rollout_relative_l2_geometric_mean": geometric_mean(
            row["rollout_relative_l2"] for row in long_rows
        ),
        "maximum_normal_boundary_residual": max(
            max(
                float(row["composed_boundary_max"]),
                float(row["max_stage_boundary"]),
                float(row["rhs_boundary_max"]),
            )
            for row in rows
        ),
        "all_equal_work_counts_match": all(bool(row["equal_rhs_work"]) for row in rows),
        "boundary_stress": {
            "sample_count": config.stress_samples,
            "initial_endpoint_max": float(endpoint_max(stress).detach().cpu()),
            "first_call_endpoint_max": float(endpoint_max(first_state).detach().cpu()),
            "first_call_stage_boundary_max": first_diagnostics["max_stage_boundary"],
            "final_endpoint_max": float(endpoint_max(stress_output).detach().cpu()),
            "final_endpoint_rms": float(endpoint_rms(stress_output).detach().cpu()),
            "max_stage_boundary": stress_diagnostics["max_stage_boundary"],
            "rhs_boundary_max_at_input": stress_rhs_boundary,
        },
    }
    summary["maximum_hard_boundary_evidence"] = max(
        summary["maximum_normal_boundary_residual"],
        summary["boundary_stress"]["first_call_endpoint_max"],
        summary["boundary_stress"]["first_call_stage_boundary_max"],
        summary["boundary_stress"]["final_endpoint_max"],
        summary["boundary_stress"]["max_stage_boundary"],
        summary["boundary_stress"]["rhs_boundary_max_at_input"],
    )
    return rows, summary


METRIC_FIELDS = (
    "evaluation_kind",
    "tau",
    "horizon",
    "composition_depth",
    "sample_count",
    "equal_work_composition_defect",
    "production_work_composition_defect",
    "rollout_mse",
    "rollout_interior_mse",
    "rollout_relative_l2",
    "direct_equal_work_mse",
    "composed_boundary_max",
    "composed_boundary_rms",
    "max_stage_boundary",
    "rhs_boundary_max",
    "composed_rhs_evaluations",
    "direct_equal_rhs_evaluations",
    "direct_production_rhs_evaluations",
    "equal_rhs_work",
)


def expected_metric_identities() -> set[tuple[str, float, float]]:
    return {
        (kind, float(tau), float(horizon))
        for kind, pairs in (
            ("primary_semigroup", PRIMARY_SEMIGROUP_PAIRS),
            ("long_rollout", LONG_ROLLOUT_PAIRS),
        )
        for tau, horizon in pairs
    }


__all__ = [
    "BOUNDARY_MODES",
    "BOUNDARY_THRESHOLD",
    "BoundarySemigroupFlow",
    "BoundaryWaveConfig",
    "LONG_ROLLOUT_PAIRS",
    "METRIC_FIELDS",
    "METRIC_FLOOR",
    "PRIMARY_SEMIGROUP_PAIRS",
    "TEMPORAL_MODES",
    "TRAINING_SEEDS",
    "build_reference_cache",
    "composed_rollout",
    "composition_depth",
    "endpoint_max",
    "endpoint_rms",
    "evaluate_model",
    "expected_metric_identities",
    "geometric_mean",
    "mean_relative_l2",
    "parameter_count",
    "parameter_fingerprint",
    "project_homogeneous_dirichlet",
    "reference_reaction_diffusion_rhs",
    "reference_solve",
    "sample_sine_initial_conditions",
    "time_key",
    "validate_reference_cache",
    "wave_config",
]
