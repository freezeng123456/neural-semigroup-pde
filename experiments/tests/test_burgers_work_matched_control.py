"""Checks for the work-matched Burgers flow-versus-direct-map lane."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.run_burgers_direct_map_control import PeriodicBurgersDirectMap
from experiments.run_burgers_minimal_ab import EXPECTED_PARAMETERS
from experiments.run_burgers_work_matched_control import (
    EulerBurgersFluxFlow,
    build_pair,
    flow_cross_lag,
    flow_cross_lag_equal_substeps,
)

GRID = {"N": 32, "L": 2.0 * math.pi, "nu": 0.01}


def test_matched_budget_across_all_three_models():
    torch.manual_seed(6)
    autonomous, query = build_pair(GRID, 1)
    direct = PeriodicBurgersDirectMap(
        n_grid=GRID["N"], length=GRID["L"], viscosity=GRID["nu"]
    )
    for model in (autonomous, query, direct):
        assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS
    assert autonomous.ode_steps == 1
    for left, right in zip(autonomous.parameters(), query.parameters()):
        assert torch.equal(left, right)


def test_one_euler_substep_costs_one_flux_evaluation():
    """The deployed map must be a single field evaluation, not four stages."""

    torch.manual_seed(6)
    autonomous, _query = build_pair(GRID, 1)
    state = torch.randn(3, GRID["N"])
    calls = []
    original = autonomous.rhs

    def counting(*call_args, **call_kwargs):
        calls.append(1)
        return original(*call_args, **call_kwargs)

    autonomous.rhs = counting
    with torch.no_grad():
        evolved = autonomous(state, 0.04)
    assert len(calls) == 1
    with torch.no_grad():
        assert torch.allclose(evolved, state + 0.04 * original(state, 0.04))


def test_euler_flow_preserves_the_spatial_mean():
    torch.manual_seed(6)
    autonomous, query = build_pair(GRID, 1)
    state = torch.randn(4, GRID["N"])
    for model in (autonomous, query):
        with torch.no_grad():
            evolved = model(state, 0.08)
        assert torch.allclose(evolved.mean(dim=1), state.mean(dim=1), atol=1e-6)


def test_autonomous_field_composes_exactly_at_equal_substeps():
    torch.manual_seed(6)
    autonomous, _query = build_pair(GRID, 1)
    rows = flow_cross_lag_equal_substeps(autonomous, torch.randn(3, GRID["N"]), 1)
    for row in rows.values():
        assert row["rms_defect_mean"] == 0.0
        assert row["coarse_substeps_per_call"] == 2 * row["fine_substeps_per_call"]


def test_deployed_budget_defect_refines_toward_zero():
    """Refining substeps must shrink the flow's cross-lag defect."""

    torch.manual_seed(6)
    autonomous, _query = build_pair(GRID, 1)
    sample = torch.randn(3, GRID["N"]) * 0.3
    series = [
        flow_cross_lag(autonomous, sample, substeps)["horizon=0.4"][
            "rms_defect_mean"
        ]
        for substeps in (1, 2, 4, 8, 16)
    ]
    assert all(later < earlier for earlier, later in zip(series, series[1:]))
    assert series[0] > 10.0 * series[-1]


def test_flux_evaluation_counts_are_reported_for_the_flow_sweep():
    torch.manual_seed(6)
    autonomous, _query = build_pair(GRID, 1)
    for substeps in (1, 4):
        rows = flow_cross_lag(autonomous, torch.randn(2, GRID["N"]), substeps)
        for horizon, row in rows.items():
            depth_ratio = row["fine_flux_evaluations"] / row["coarse_flux_evaluations"]
            assert row["substeps_per_call"] == substeps
            assert depth_ratio == pytest.approx(2.0)
