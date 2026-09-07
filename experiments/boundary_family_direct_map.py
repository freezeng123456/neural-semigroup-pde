"""Work-matched Wave 2 models: Euler flows and a one-shot residual map.

The stencil MLP, hard boundary reconstruction, and duration feature are
Wave 2's.  The temporal object is the only intervention.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from boundary_family_semigroup import (
    BoundaryFamilyFlow,
    BoundaryFamilySpec,
    residual_max,
)


EXPECTED_PARAMETERS = 1249
MATCHED_NET_EVALUATIONS = 1
RK4_NET_EVALUATIONS = 32
REFINEMENT_SWEEP = (1, 2, 4, 8, 16, 32, 64, 128)
CROSS_LAG_FINE = 0.04
CROSS_LAG_COARSE = 0.08
CROSS_LAG_HORIZON = 0.16
MODEL_LABELS = (
    "a_rk4_autonomous",
    "b_rk4_query_time",
    "a_euler_autonomous",
    "b_euler_query_time",
    "c_direct_map",
)
MATCHED_LABELS = (
    "a_euler_autonomous",
    "b_euler_query_time",
    "c_direct_map",
)


def build_stencil_mlp(hidden_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(4, hidden_width),
        nn.Tanh(),
        nn.Linear(hidden_width, hidden_width),
        nn.Tanh(),
        nn.Linear(hidden_width, 1),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def duration_feature(
    duration: float | torch.Tensor,
    state: torch.Tensor,
    *,
    scale: float,
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
    return (value / float(scale)).reshape(-1, 1, 1).expand(-1, state.shape[1], -1)


def stencil_features(
    state: torch.Tensor,
    duration: float | torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    patches = F.pad(state, (1, 1), mode="constant", value=0.0).unfold(
        dimension=1, size=3, step=1
    )
    return torch.cat((patches, duration_feature(duration, state, scale=scale)), dim=-1)


class EulerBoundaryFamilyFlow(nn.Module):
    """Wave 2 stencil field integrated by explicit Euler."""

    def __init__(self, flow: BoundaryFamilyFlow) -> None:
        super().__init__()
        if flow.enforcement_mode != "hard":
            raise ValueError("this lane uses hard enforcement only")
        self.flow = flow

    @property
    def boundary_spec(self) -> BoundaryFamilySpec:
        return self.flow.boundary_spec

    @property
    def temporal_mode(self) -> str:
        return self.flow.temporal_mode

    def vector_field(
        self, state: torch.Tensor, map_duration: float | torch.Tensor
    ) -> torch.Tensor:
        return self.flow.vector_field(state, map_duration)

    def integrate(
        self,
        state: torch.Tensor,
        duration: float | torch.Tensor,
        *,
        substeps: int,
        map_duration: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if int(substeps) <= 0:
            raise ValueError("substeps must be positive")
        requested = duration if map_duration is None else map_duration
        duration_tensor = self.flow._duration_tensor(duration, state)
        step = duration_tensor / int(substeps)
        current = self.flow.project_state(state)
        for _ in range(int(substeps)):
            current = self.flow.project_state(
                current + step * self.vector_field(current, requested)
            )
        return current

    def forward(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        return self.integrate(state, duration, substeps=1)

    def boundary_residual_max(self, state: torch.Tensor) -> float:
        projected = self.flow.project_state(state)
        return float(residual_max(self.boundary_spec.state_residual(projected)))


class BoundaryFamilyDirectMap(nn.Module):
    """One-shot residual operator with Wave 2's stencil and hard projection.

    The network output is a state residual, not a vector field, and is not
    multiplied by the requested duration.  Different lags are therefore not
    iterates of one flow.
    """

    def __init__(
        self,
        *,
        n_grid: int,
        hidden_width: int,
        boundary_spec: BoundaryFamilySpec,
        query_time_scale: float,
    ) -> None:
        super().__init__()
        boundary_spec.validate()
        if n_grid < 3 or hidden_width <= 0 or query_time_scale <= 0:
            raise ValueError("invalid direct-map dimensions")
        self.n_grid = int(n_grid)
        self.hidden_width = int(hidden_width)
        self.boundary_spec = boundary_spec
        self.query_time_scale = float(query_time_scale)
        self.vector_field_net = build_stencil_mlp(hidden_width)

    def load_field_from(self, flow: BoundaryFamilyFlow) -> None:
        self.vector_field_net.load_state_dict(
            flow.vector_field_net.state_dict(), strict=True
        )

    def residual(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        current = self.boundary_spec.project_state(state)
        features = stencil_features(
            current, duration, scale=self.query_time_scale
        )
        return self.vector_field_net(features).squeeze(-1)

    def forward(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        current = self.boundary_spec.project_state(state)
        return self.boundary_spec.project_state(current + self.residual(current, duration))

    def boundary_residual_max(self, state: torch.Tensor) -> float:
        return float(residual_max(self.boundary_spec.state_residual(self.forward(state, 0.04))))


def compose(
    model: nn.Module,
    state: torch.Tensor,
    *,
    tau: float,
    depth: int,
) -> torch.Tensor:
    current = state
    for _ in range(int(depth)):
        current = model(current, tau)
    return current


def rms_defect(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left - right).square().mean(dim=1)).mean().item())


@torch.no_grad()
def cross_lag_defect(
    evolve,
    sample: torch.Tensor,
    *,
    fine: float = CROSS_LAG_FINE,
    coarse: float = CROSS_LAG_COARSE,
    horizon: float = CROSS_LAG_HORIZON,
) -> dict[str, object]:
    fine_depth = int(round(horizon / fine))
    coarse_depth = int(round(horizon / coarse))
    fine_state = sample
    for _ in range(fine_depth):
        fine_state = evolve(fine_state, fine)
    coarse_state = sample
    for _ in range(coarse_depth):
        coarse_state = evolve(coarse_state, coarse)
    return {
        "rms_defect_mean": rms_defect(fine_state, coarse_state),
        "fine_tau": fine,
        "coarse_tau": coarse,
        "horizon": horizon,
        "fine_depth": fine_depth,
        "coarse_depth": coarse_depth,
        "composed_state_rms": float(
            torch.sqrt(fine_state.square().mean(dim=1)).mean().item()
        ),
    }


def flow_cross_lag(
    model: EulerBoundaryFamilyFlow, sample: torch.Tensor, substeps: int
) -> dict[str, object]:
    return cross_lag_defect(
        lambda state, tau: model.integrate(state, tau, substeps=substeps),
        sample,
    ) | {"substeps_per_call": int(substeps)}


def flow_cross_lag_equal_substeps(
    model: EulerBoundaryFamilyFlow, sample: torch.Tensor, substeps: int
) -> dict[str, object]:
    ratio = int(round(CROSS_LAG_COARSE / CROSS_LAG_FINE))
    fine_state = sample
    for _ in range(int(round(CROSS_LAG_HORIZON / CROSS_LAG_FINE))):
        fine_state = model.integrate(
            fine_state, CROSS_LAG_FINE, substeps=substeps
        )
    coarse_state = sample
    for _ in range(int(round(CROSS_LAG_HORIZON / CROSS_LAG_COARSE))):
        coarse_state = model.integrate(
            coarse_state, CROSS_LAG_COARSE, substeps=substeps * ratio
        )
    return {
        "rms_defect_mean": rms_defect(fine_state, coarse_state),
        "fine_substeps_per_call": int(substeps),
        "coarse_substeps_per_call": int(substeps * ratio),
    }


def direct_map_cross_lag(
    model: BoundaryFamilyDirectMap, sample: torch.Tensor
) -> dict[str, object]:
    return cross_lag_defect(model, sample) | {"substeps_per_call": None}


def build_paired_models(
    *,
    n_grid: int,
    hidden_width: int,
    boundary_spec: BoundaryFamilySpec,
    query_time_scale: float,
    rk4_steps: int,
) -> dict[str, nn.Module]:
    """One fingerprint, five temporal objects."""

    autonomous = BoundaryFamilyFlow(
        n_grid=n_grid,
        hidden_width=hidden_width,
        boundary_spec=boundary_spec,
        enforcement_mode="hard",
        temporal_mode="autonomous",
        query_time_scale=query_time_scale,
        rk4_steps=rk4_steps,
    )
    query = BoundaryFamilyFlow(
        n_grid=n_grid,
        hidden_width=hidden_width,
        boundary_spec=boundary_spec,
        enforcement_mode="hard",
        temporal_mode="query_time",
        query_time_scale=query_time_scale,
        rk4_steps=rk4_steps,
    )
    query.load_state_dict(autonomous.state_dict(), strict=True)
    euler_autonomous = EulerBoundaryFamilyFlow(
        BoundaryFamilyFlow(
            n_grid=n_grid,
            hidden_width=hidden_width,
            boundary_spec=boundary_spec,
            enforcement_mode="hard",
            temporal_mode="autonomous",
            query_time_scale=query_time_scale,
            rk4_steps=1,
        )
    )
    euler_query = EulerBoundaryFamilyFlow(
        BoundaryFamilyFlow(
            n_grid=n_grid,
            hidden_width=hidden_width,
            boundary_spec=boundary_spec,
            enforcement_mode="hard",
            temporal_mode="query_time",
            query_time_scale=query_time_scale,
            rk4_steps=1,
        )
    )
    euler_autonomous.flow.load_state_dict(autonomous.state_dict(), strict=True)
    euler_query.flow.load_state_dict(autonomous.state_dict(), strict=True)
    direct = BoundaryFamilyDirectMap(
        n_grid=n_grid,
        hidden_width=hidden_width,
        boundary_spec=boundary_spec,
        query_time_scale=query_time_scale,
    )
    direct.load_field_from(autonomous)
    models = {
        "a_rk4_autonomous": autonomous,
        "b_rk4_query_time": query,
        "a_euler_autonomous": euler_autonomous,
        "b_euler_query_time": euler_query,
        "c_direct_map": direct,
    }
    counts = {label: parameter_count(model) for label, model in models.items()}
    unique = set(counts.values())
    if len(unique) != 1:
        raise RuntimeError(f"parameter counts must match across the five models: {counts}")
    if hidden_width == 32 and unique != {EXPECTED_PARAMETERS}:
        raise RuntimeError(
            f"Wave 2 width-32 models must have {EXPECTED_PARAMETERS} parameters: {counts}"
        )
    return models


def trainable(model: nn.Module) -> nn.Module:
    if isinstance(model, EulerBoundaryFamilyFlow):
        return model.flow
    return model


def net_evaluations_per_call(label: str) -> int:
    if label.startswith("a_rk4") or label.startswith("b_rk4"):
        return RK4_NET_EVALUATIONS
    return MATCHED_NET_EVALUATIONS


def refinement_gains(defects: Sequence[float], sweep: Sequence[int]) -> dict[str, float]:
    if len(defects) != len(sweep):
        raise ValueError("refinement series length mismatch")
    if defects[0] <= 0:
        raise ValueError("the coarsest defect must be positive to form a gain")
    return {
        "gain_1_to_last": float(defects[0] / max(defects[-1], 1e-16)),
        "first_defect": float(defects[0]),
        "last_defect": float(defects[-1]),
        "last_substeps": int(sweep[-1]),
    }
