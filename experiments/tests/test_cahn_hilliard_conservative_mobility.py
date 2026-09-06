"""Checks for the conservative Cahn--Hilliard mobility."""

from __future__ import annotations

import torch

from experiments.run_cahn_hilliard_conservative_mobility import (
    ConservativeCahnHilliardFlow,
    parameter_count,
)


def test_model_has_learned_weights():
    model = ConservativeCahnHilliardFlow(n_grid=16)
    assert parameter_count(model) > 1000


def test_vector_field_has_zero_mean():
    model = ConservativeCahnHilliardFlow(n_grid=32)
    state = torch.randn(4, 32) * 0.3
    with torch.no_grad():
        field = model.vector_field(state)
    assert torch.allclose(field.mean(dim=-1), torch.zeros(4), atol=1e-5)


def test_forward_conserves_mass():
    model = ConservativeCahnHilliardFlow(n_grid=32, ode_steps=4)
    state = torch.randn(3, 32) * 0.25
    state = state - state.mean(dim=-1, keepdim=True) + 0.2
    with torch.no_grad():
        evolved = model(state, 0.05)
    assert torch.allclose(evolved.mean(dim=-1), state.mean(dim=-1), atol=1e-5)


def test_initialization_is_near_the_physical_field():
    model = ConservativeCahnHilliardFlow(n_grid=32, ode_steps=2)
    state = torch.randn(3, 32) * 0.2
    with torch.no_grad():
        residual = model.residual(state.unsqueeze(-1)).squeeze(-1)
        mobility = model.mobility(state)
    assert float(residual.abs().max()) < 1e-5
    assert float(mobility.max()) < 0.05


def test_zero_residual_keeps_constants_stationary():
    model = ConservativeCahnHilliardFlow(n_grid=32, ode_steps=2)
    with torch.no_grad():
        for parameter in model.residual.parameters():
            parameter.zero_()
        state = torch.full((2, 32), 0.3)
        evolved = model(state, 0.1)
    assert torch.allclose(evolved, state, atol=1e-5)
