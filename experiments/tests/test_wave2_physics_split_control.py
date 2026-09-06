"""Contracts for the Wave 2 physics-split direct-map lane."""

from __future__ import annotations

import torch

from experiments.boundary_family_direct_map import (
    EXPECTED_PARAMETERS,
    BoundaryFamilyDirectMap,
    parameter_count,
)
from experiments.boundary_family_physics_split import (
    apply_heat_semigroup,
    assemble_affine_heat_generator,
    build_physics_split_map,
    linear_heat_rhs,
)
from experiments.boundary_family_semigroup import (
    BOUNDARY_FAMILIES,
    BoundaryFamilySpec,
    residual_max,
)


def _spec(family: str, n_grid: int = 17) -> BoundaryFamilySpec:
    return BoundaryFamilySpec(family=family, dx=1.0 / (n_grid - 1))


def test_production_width_keeps_the_wave2_parameter_count():
    model = build_physics_split_map(
        n_grid=33,
        hidden_width=32,
        boundary_spec=BoundaryFamilySpec(family="robin", dx=1.0 / 32),
        query_time_scale=0.08,
        diffusivity=0.02,
    )
    assert parameter_count(model) == EXPECTED_PARAMETERS


def test_heat_semigroup_composes_and_preserves_every_family():
    for family in BOUNDARY_FAMILIES:
        spec = _spec(family)
        generator, _offset, equilibrium = assemble_affine_heat_generator(
            spec, n_grid=17, diffusivity=0.02, dtype=torch.float64, device=torch.device("cpu")
        )
        state = spec.project_state(torch.randn(4, 17, dtype=torch.float64))
        once = apply_heat_semigroup(
            state, 0.12, generator=generator, equilibrium=equilibrium
        )
        composed = apply_heat_semigroup(
            apply_heat_semigroup(
                state, 0.05, generator=generator, equilibrium=equilibrium
            ),
            0.07,
            generator=generator,
            equilibrium=equilibrium,
        )
        assert torch.allclose(once, composed, atol=1e-8, rtol=1e-8)
        assert float(residual_max(spec.state_residual(once))) <= 5e-8


def test_heat_rhs_has_no_reaction():
    spec = _spec("homogeneous_neumann")
    state = spec.project_state(torch.full((2, 17), 0.5))
    rhs = linear_heat_rhs(state, diffusivity=0.02, spec=spec)
    # Constant Neumann state is discrete-harmonic, so the linear rhs vanishes.
    assert torch.allclose(rhs, torch.zeros_like(rhs), atol=1e-6)


def test_physics_split_is_not_the_residual_map():
    spec = _spec("inhomogeneous_dirichlet")
    torch.manual_seed(6)
    physics = build_physics_split_map(
        n_grid=17,
        hidden_width=8,
        boundary_spec=spec,
        query_time_scale=0.08,
        diffusivity=0.02,
    )
    residual = BoundaryFamilyDirectMap(
        n_grid=17,
        hidden_width=8,
        boundary_spec=spec,
        query_time_scale=0.08,
    )
    residual.vector_field_net.load_state_dict(physics.reaction_net.state_dict())
    state = spec.project_state(torch.randn(3, 17))
    with torch.no_grad():
        left = physics(state, 0.08)
        right = residual(state, 0.08)
    assert not torch.allclose(left, right, atol=1e-4)
    assert float(residual_max(spec.state_residual(left))) <= 5e-6
