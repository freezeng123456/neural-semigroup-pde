"""Exact boundary-aware heat semigroup plus a one-shot reaction increment."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from boundary_family_direct_map import (  # noqa: E402
    EXPECTED_PARAMETERS,
    build_stencil_mlp,
    parameter_count,
    stencil_features,
)
from boundary_family_semigroup import BoundaryFamilySpec  # noqa: E402


def linear_heat_rhs(
    state: torch.Tensor,
    *,
    diffusivity: float,
    spec: BoundaryFamilySpec,
) -> torch.Tensor:
    """Centered Laplacian with Wave 2's tangent projection, no reaction."""

    projected = spec.project_state(state)
    derivative = torch.zeros_like(projected)
    interior = projected[:, 1:-1]
    laplacian = (projected[:, 2:] - 2.0 * interior + projected[:, :-2]) / spec.dx**2
    derivative[:, 1:-1] = float(diffusivity) * laplacian
    return spec.project_tangent(derivative)


def assemble_affine_heat_generator(
    spec: BoundaryFamilySpec,
    *,
    n_grid: int,
    diffusivity: float,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return A, offset, and equilibrium so F(u) = A u + offset on R^N."""

    zeros = torch.zeros(1, n_grid, dtype=dtype, device=device)
    offset = linear_heat_rhs(zeros, diffusivity=diffusivity, spec=spec).squeeze(0)
    columns = []
    for index in range(n_grid):
        basis = torch.zeros(1, n_grid, dtype=dtype, device=device)
        basis[0, index] = 1.0
        column = linear_heat_rhs(basis, diffusivity=diffusivity, spec=spec).squeeze(0)
        columns.append(column - offset)
    generator = torch.stack(columns, dim=1)
    if float(offset.norm()) <= 1e-10:
        equilibrium = torch.zeros(n_grid, dtype=dtype, device=device)
    else:
        equilibrium = torch.linalg.lstsq(
            generator, -offset.unsqueeze(1)
        ).solution.squeeze(1)
    return generator, offset, equilibrium


def duration_vector(
    duration: float | torch.Tensor, state: torch.Tensor
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
    return value


def apply_heat_semigroup(
    state: torch.Tensor,
    duration: float | torch.Tensor,
    *,
    generator: torch.Tensor,
    equilibrium: torch.Tensor,
) -> torch.Tensor:
    taus = duration_vector(duration, state)
    centered = state - equilibrium
    matrices = torch.linalg.matrix_exp(taus.reshape(-1, 1, 1) * generator)
    evolved = torch.bmm(centered.unsqueeze(1), matrices.transpose(-1, -2)).squeeze(1)
    return evolved + equilibrium


class BoundaryFamilyPhysicsSplitMap(nn.Module):
    """One-shot map: exact heat on X_B plus tau times a learned reaction."""

    def __init__(
        self,
        *,
        n_grid: int,
        hidden_width: int,
        boundary_spec: BoundaryFamilySpec,
        query_time_scale: float,
        diffusivity: float,
    ) -> None:
        super().__init__()
        boundary_spec.validate()
        if n_grid < 3 or hidden_width <= 0 or query_time_scale <= 0 or diffusivity <= 0:
            raise ValueError("invalid physics-split dimensions")
        self.n_grid = int(n_grid)
        self.hidden_width = int(hidden_width)
        self.boundary_spec = boundary_spec
        self.query_time_scale = float(query_time_scale)
        self.diffusivity = float(diffusivity)
        self.reaction_net = build_stencil_mlp(hidden_width)
        generator, offset, equilibrium = assemble_affine_heat_generator(
            boundary_spec,
            n_grid=n_grid,
            diffusivity=diffusivity,
            dtype=torch.get_default_dtype(),
            device=torch.device("cpu"),
        )
        self.register_buffer("heat_generator", generator)
        self.register_buffer("heat_offset", offset)
        self.register_buffer("heat_equilibrium", equilibrium)

    def heat(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        current = self.boundary_spec.project_state(state)
        evolved = apply_heat_semigroup(
            current,
            duration,
            generator=self.heat_generator,
            equilibrium=self.heat_equilibrium,
        )
        return self.boundary_spec.project_state(evolved)

    def reaction(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        current = self.boundary_spec.project_state(state)
        features = stencil_features(
            current, duration, scale=self.query_time_scale
        )
        return self.boundary_spec.project_tangent(
            self.reaction_net(features).squeeze(-1)
        )

    def forward(
        self, state: torch.Tensor, duration: float | torch.Tensor
    ) -> torch.Tensor:
        current = self.boundary_spec.project_state(state)
        taus = duration_vector(duration, current).unsqueeze(1)
        return self.boundary_spec.project_state(
            self.heat(current, duration) + taus * self.reaction(current, duration)
        )


def build_physics_split_map(
    *,
    n_grid: int,
    hidden_width: int,
    boundary_spec: BoundaryFamilySpec,
    query_time_scale: float,
    diffusivity: float,
) -> BoundaryFamilyPhysicsSplitMap:
    model = BoundaryFamilyPhysicsSplitMap(
        n_grid=n_grid,
        hidden_width=hidden_width,
        boundary_spec=boundary_spec,
        query_time_scale=query_time_scale,
        diffusivity=diffusivity,
    )
    if hidden_width == 32 and parameter_count(model) != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"physics-split map must have {EXPECTED_PARAMETERS} parameters"
        )
    return model
