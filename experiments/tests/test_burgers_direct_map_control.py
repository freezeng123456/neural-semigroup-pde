"""Structural checks for the Burgers direct time-conditioned map control."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.run_burgers_direct_map_control import (
    PeriodicBurgersDirectMap,
    compose,
    measure_direct_map_composition,
)
from experiments.run_burgers_minimal_ab import EXPECTED_PARAMETERS, build_pair
from experiments.seed_utils import set_global_seed

GRID = {"N": 32, "L": 2.0 * math.pi, "nu": 0.01}


def make_direct(n_grid=32):
    return PeriodicBurgersDirectMap(
        n_grid=n_grid, length=2.0 * math.pi, viscosity=0.01
    )


def test_direct_map_matches_the_minimal_parameter_budget():
    assert sum(p.numel() for p in make_direct().parameters()) == EXPECTED_PARAMETERS


def test_direct_map_shares_the_autonomous_initialization_stream():
    """C must start from the same flux weights as A under the same seed."""

    set_global_seed(11, deterministic=True)
    direct = make_direct()
    set_global_seed(11, deterministic=True)
    autonomous, _query = build_pair(GRID, 12)
    for left, right in zip(
        direct.field.parameters(), autonomous.parameters()
    ):
        assert torch.equal(left, right)


def test_direct_map_preserves_the_spatial_mean():
    model = make_direct()
    state = torch.randn(4, 32)
    with torch.no_grad():
        for tau in (0.04, 0.08, 0.4):
            evolved = model(state, tau)
            assert torch.allclose(
                evolved.mean(dim=1), state.mean(dim=1), atol=1e-6
            )


def test_viscous_part_is_an_exact_semigroup():
    """The linear factor must compose exactly; only the learned part cannot."""

    model = make_direct()
    state = torch.randn(3, 32)
    with torch.no_grad():
        composed = model.viscous_semigroup(
            model.viscous_semigroup(state, 0.04), 0.08
        )
        direct = model.viscous_semigroup(state, 0.12)
    assert torch.allclose(composed, direct, atol=1e-6)


def test_direct_map_is_not_the_flow_of_one_generator():
    """Two in-range lags reaching one horizon must not agree for C."""

    torch.manual_seed(4)
    model = make_direct()
    state = torch.randn(3, 32)
    with torch.no_grad():
        fine = compose(model, state, tau=0.04, depth=10)
        coarse = compose(model, state, tau=0.08, depth=5)
    assert not torch.allclose(fine, coarse, atol=1e-5)


def test_composition_report_records_flux_evaluation_counts():
    torch.manual_seed(4)
    model = make_direct()
    measured = measure_direct_map_composition(model, torch.randn(2, 32))
    for row in measured["direct_versus_composed"].values():
        assert row["direct_flux_evaluations"] == 1
        assert row["composed_flux_evaluations"] == row["composition_depth"]
    for row in measured["in_range_cross_lag"].values():
        assert row["fine_flux_evaluations"] == row["fine_depth"]
        assert row["coarse_flux_evaluations"] == row["coarse_depth"]
        assert row["composed_state_rms"] > 0.0


def test_lag_enters_only_through_the_conditioning_channel():
    """Zeroing the control column must make C's nonlinear part lag-linear."""

    model = make_direct()
    with torch.no_grad():
        model.field.flux[0].weight[:, -1] = 0.0
    state = torch.randn(2, 32)
    with torch.no_grad():
        half = model(state, 0.05) - model.viscous_semigroup(state, 0.05)
        full = model(state, 0.10) - model.viscous_semigroup(state, 0.10)
    assert torch.allclose(2.0 * half, full, atol=1e-6)


def test_grid_mismatch_is_caught_by_strict_loading():
    model = make_direct(n_grid=32)
    other = make_direct(n_grid=64)
    with pytest.raises(RuntimeError):
        model.load_state_dict(other.state_dict(), strict=True)
