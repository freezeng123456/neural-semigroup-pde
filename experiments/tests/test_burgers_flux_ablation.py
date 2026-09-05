"""Structural checks for the Burgers flux-generator ablation."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.run_burgers_flux_ablation import (
    VARIANTS,
    AblatedBurgersFluxFlow,
    build_variant,
)

EXPECTED_PARAMETERS = {
    "r2_h32_l2_anchor": 1313,
    "r1_h32_l2_anchor": 1249,
    "r0_h32_l2_anchor": 1185,
    "r0_h8_l2_anchor": 105,
    "r0_h8_l1_anchor": 33,
    "r2_h32_l2_noanchor": 1313,
    "r0_h32_l2_noanchor": 1185,
}


def make(label, n_grid=16):
    torch.manual_seed(5)
    return build_variant(
        label,
        n_grid=n_grid,
        length=2.0 * math.pi,
        viscosity=0.01,
        ode_steps=2,
    )


def test_frozen_grid_parameter_counts():
    assert set(VARIANTS) == set(EXPECTED_PARAMETERS)
    for label, expected in EXPECTED_PARAMETERS.items():
        model = make(label)
        assert sum(p.numel() for p in model.parameters()) == expected


def test_every_variant_preserves_the_spatial_mean():
    """All variants stay in divergence form, so the mean cannot drift."""

    state = torch.randn(4, 16)
    for label in VARIANTS:
        model = make(label)
        rhs = model.rhs(state, 0.04)
        assert torch.allclose(rhs.mean(dim=1), torch.zeros(4), atol=1e-6)
        evolved = model(state, 0.04)
        assert torch.allclose(evolved.mean(dim=1), state.mean(dim=1), atol=1e-6)


def test_anchor_removal_zeroes_only_the_viscous_term():
    anchored = make("r2_h32_l2_anchor")
    bare = make("r2_h32_l2_noanchor")
    assert anchored.viscosity == pytest.approx(0.01)
    assert bare.viscosity == 0.0
    state = torch.randn(2, 16)
    with torch.no_grad():
        flux_difference = anchored.local_flux(state, 0.04) - bare.local_flux(
            state, 0.04
        )
    assert torch.allclose(flux_difference, torch.zeros_like(flux_difference))


def test_pointwise_flux_sees_only_the_local_state():
    """Radius zero must make the flux independent of every neighbour."""

    model = make("r0_h32_l2_anchor")
    left = torch.zeros(1, 16)
    right = torch.zeros(1, 16)
    left[0, 0] = 0.3
    right[0, 0] = 0.3
    right[0, 5] = 0.9
    with torch.no_grad():
        flux_left = model.local_flux(left, 0.04)
        flux_right = model.local_flux(right, 0.04)
    assert flux_left[0, 0] == pytest.approx(flux_right[0, 0].item())


def test_all_variants_are_autonomous():
    for label in VARIANTS:
        model = make(label)
        assert model.query_conditioned is False
        state = torch.randn(2, 16)
        with torch.no_grad():
            assert torch.allclose(
                model.rhs(state, 0.04), model.rhs(state, 0.8)
            )


def test_invalid_layer_count_is_rejected():
    with pytest.raises(ValueError, match="hidden_layers must be 1 or 2"):
        AblatedBurgersFluxFlow(
            n_grid=16,
            length=2.0 * math.pi,
            viscosity=0.01,
            hidden=8,
            hidden_layers=3,
            ode_steps=2,
        )
