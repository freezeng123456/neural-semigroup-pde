"""Checks for the exact-biharmonic Cahn--Hilliard increment."""

from __future__ import annotations

import math

import torch

from experiments.run_cahn_hilliard_exact_biharmonic import (
    ExactBiharmonicConservativeIncrement,
)
from experiments.run_cahn_hilliard_conservative_mobility import parameter_count


def test_increment_has_weights():
    model = ExactBiharmonicConservativeIncrement(n_grid=16)
    assert parameter_count(model) > 1000


def test_linear_semigroup_composes():
    model = ExactBiharmonicConservativeIncrement(n_grid=16, epsilon=0.1)
    state = torch.randn(3, 16) * 0.2
    with torch.no_grad():
        one = model.linear_semigroup(state, 0.15)
        two = model.linear_semigroup(model.linear_semigroup(state, 0.075), 0.075)
    assert torch.allclose(one, two, atol=1e-6, rtol=1e-6)


def test_increment_has_no_linear_laplacian():
    model = ExactBiharmonicConservativeIncrement(n_grid=32, epsilon=0.1)
    state = torch.randn(4, 32) * 0.25
    with torch.no_grad():
        potential = model.nonlinear_potential(state)
        # The linear operator is absent: a constant has constant potential.
        constant = torch.full((2, 32), 0.2)
        assert torch.allclose(
            model.nonlinear_potential(constant),
            constant.pow(3) - constant + model.residual(constant.unsqueeze(-1)).squeeze(-1),
            atol=1e-6,
        )
        field = model.vector_field(state)
    assert torch.allclose(field.mean(dim=-1), torch.zeros(4), atol=1e-5)
    assert potential.shape == state.shape


def test_forward_conserves_mass():
    model = ExactBiharmonicConservativeIncrement(n_grid=32)
    state = torch.randn(3, 32) * 0.2
    state = state - state.mean(dim=-1, keepdim=True) + 0.15
    with torch.no_grad():
        evolved = model(state, 0.075)
    assert torch.allclose(evolved.mean(dim=-1), state.mean(dim=-1), atol=1e-5)
    assert math.isfinite(float(evolved.square().mean()))
