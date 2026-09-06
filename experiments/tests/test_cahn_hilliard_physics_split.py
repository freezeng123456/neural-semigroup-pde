"""Checks for the zero-parameter Cahn--Hilliard conservative split."""

from __future__ import annotations

import math

import torch

from experiments.run_cahn_hilliard_physics_split import (
    CahnHilliardPhysicsSplit,
    CahnHilliardSolver,
    generate_initial_conditions,
    parameter_count,
)


def test_physics_split_has_zero_parameters():
    model = CahnHilliardPhysicsSplit(n_grid=16)
    assert parameter_count(model) == 0


def test_linear_semigroup_composes():
    model = CahnHilliardPhysicsSplit(n_grid=16, epsilon=0.1)
    state = torch.randn(3, 16) * 0.2
    with torch.no_grad():
        one = model.linear_semigroup(state, 0.15)
        two = model.linear_semigroup(model.linear_semigroup(state, 0.075), 0.075)
    assert torch.allclose(one, two, atol=1e-6, rtol=1e-6)


def test_split_conserves_mass():
    model = CahnHilliardPhysicsSplit(n_grid=32, epsilon=0.1)
    state = torch.randn(5, 32) * 0.3
    state = state - state.mean(dim=-1, keepdim=True) + 0.2
    with torch.no_grad():
        evolved = model(state, 0.1)
    assert torch.allclose(evolved.mean(dim=-1), state.mean(dim=-1), atol=1e-6)


def test_constant_state_is_stationary():
    model = CahnHilliardPhysicsSplit(n_grid=32, epsilon=0.1)
    state = torch.full((4, 32), 0.25)
    with torch.no_grad():
        evolved = model(state, 0.2)
    assert torch.allclose(evolved, state, atol=1e-6)


def test_solver_step_conserves_mass():
    solver = CahnHilliardSolver(n_grid=32, epsilon=0.1, dt=5e-4, device="cpu")
    initial = torch.randn(2, 32) * 0.25
    initial = initial - initial.mean(dim=-1, keepdim=True)
    times, trajectory = solver.solve_batch(initial, 0.01)
    assert times.numel() == 21
    assert torch.allclose(
        trajectory[:, -1].mean(dim=-1),
        initial.mean(dim=-1),
        atol=1e-6,
    )


def test_initial_conditions_hit_the_requested_masses():
    states, labels = generate_initial_conditions(
        16, 2.0 * math.pi, (-0.3, 0.0, 0.3), 4, 7
    )
    assert states.shape == (12, 16)
    for mass in (-0.3, 0.0, 0.3):
        selected = states[labels == mass]
        assert selected.shape[0] == 4
        assert torch.allclose(selected.mean(dim=-1), torch.full((4,), mass), atol=1e-6)
