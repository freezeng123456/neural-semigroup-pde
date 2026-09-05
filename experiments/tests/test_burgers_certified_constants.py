"""Checks that the certified Burgers constants really are upper bounds."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.certify_burgers_generator_constants import (
    DTYPE,
    SECH2_DERIVATIVE_SUP,
    ScalarFlux,
    operator_norms,
    periodic_operators,
)

N_GRID = 32
LENGTH = 2.0 * math.pi


def build_flux(seed=5):
    torch.manual_seed(seed)
    return ScalarFlux(
        {
            "flux.0.weight": torch.randn(8, 2, dtype=DTYPE),
            "flux.0.bias": torch.randn(8, dtype=DTYPE),
            "flux.2.weight": torch.randn(1, 8, dtype=DTYPE),
            "flux.2.bias": torch.randn(1, dtype=DTYPE),
        }
    )


def test_sech_squared_derivative_constant():
    grid = torch.linspace(-40.0, 40.0, 400_001, dtype=DTYPE)
    gate = 1.0 - torch.tanh(grid) ** 2
    derivative = -2.0 * torch.tanh(grid) * gate
    assert float(derivative.abs().max().item()) <= SECH2_DERIVATIVE_SUP + 1e-9


def test_analytic_derivative_bound_dominates_autograd():
    flux = build_flux()
    grid = torch.linspace(-3.0, 3.0, 20_001, dtype=DTYPE, requires_grad=True)
    values = flux(grid)
    (gradient,) = torch.autograd.grad(values.sum(), grid)
    assert float(gradient.abs().max().item()) <= flux.derivative_sup_analytic()
    assert torch.allclose(gradient, flux.derivative(grid.detach()), atol=1e-10)


def test_covering_bound_dominates_a_finer_grid():
    flux = build_flux()
    bound = 1.5
    covering = flux.derivative_sup_covering(bound, 20_001)
    fine = torch.linspace(-bound, bound, 400_001, dtype=DTYPE)
    assert float(flux.derivative(fine).abs().max().item()) <= covering


def test_flux_error_bound_is_gauge_invariant_and_dominating():
    flux = build_flux()
    bound = 1.2
    report = flux.flux_error_sup_covering(bound, 20_001, lipschitz=10.0)
    fine = torch.linspace(-bound, bound, 400_001, dtype=DTYPE)
    difference = flux(fine) - 0.5 * fine**2
    centered = difference - report["optimal_constant"]
    assert float(centered.abs().max().item()) <= report["gauge_invariant_sup"]
    # Adding a constant to the flux must not change the certified quantity,
    # because a discrete divergence annihilates constants.
    shifted = build_flux()
    shifted.b2 += 7.0
    shifted_report = shifted.flux_error_sup_covering(bound, 20_001, lipschitz=10.0)
    assert shifted_report["gauge_invariant_sup"] == pytest.approx(
        report["gauge_invariant_sup"]
    )
    assert shifted_report["optimal_constant"] == pytest.approx(
        report["optimal_constant"] + 7.0
    )


def test_operator_norms_match_the_explicit_matrices():
    divergence, laplacian, dx = periodic_operators(N_GRID, LENGTH)
    norms = operator_norms(N_GRID, dx)
    assert float(torch.linalg.matrix_norm(divergence, ord=2)) == pytest.approx(
        norms["divergence_spectral_norm"], rel=1e-10
    )
    assert float(torch.linalg.matrix_norm(laplacian, ord=2)) == pytest.approx(
        norms["laplacian_spectral_norm"], rel=1e-10
    )


def test_divergence_is_skew_and_laplacian_is_negative_semidefinite():
    divergence, laplacian, _dx = periodic_operators(N_GRID, LENGTH)
    assert torch.equal(divergence, -divergence.T)
    assert torch.equal(laplacian, laplacian.T)
    assert float(torch.linalg.eigvalsh(laplacian).max().item()) <= 1e-10


def test_one_sided_bound_dominates_sampled_quotients():
    """The certified rate must dominate the sampled quotient it replaces."""

    flux = build_flux()
    divergence, laplacian, dx = periodic_operators(N_GRID, LENGTH)
    norms = operator_norms(N_GRID, dx)
    bound = 1.0
    lipschitz = flux.derivative_sup_covering(bound, 20_001)
    certified = norms["divergence_spectral_norm"] * lipschitz

    torch.manual_seed(2)
    left = torch.rand(64, N_GRID, dtype=DTYPE) * 2.0 * bound - bound
    right = torch.rand(64, N_GRID, dtype=DTYPE) * 2.0 * bound - bound

    def field(state):
        return -(divergence @ flux(state.T)).T + 0.01 * (laplacian @ state.T).T

    difference = left - right
    numerator = ((field(left) - field(right)) * difference).sum(dim=-1)
    denominator = difference.square().sum(dim=-1)
    assert float((numerator / denominator).max().item()) <= certified
