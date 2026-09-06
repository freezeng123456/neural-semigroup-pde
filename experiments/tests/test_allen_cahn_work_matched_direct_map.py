"""Checks for the Allen--Cahn work-matched direct-map lane."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.run_allen_cahn_work_matched_direct_map import (
    AllenCahnDirectHeatMap,
    EulerAutonomousFlow,
    EulerQueryTimeFlow,
    QUERY_EXTRA_PARAMETERS,
    architecture_kwargs,
    copy_autonomous_into_query,
    cross_lag_defect,
    equal_substep_autonomous_defect,
    parameter_count,
)


def _pair(n_grid: int = 16):
    torch.manual_seed(7)
    kwargs = architecture_kwargs(n_grid)
    autonomous = EulerAutonomousFlow(**kwargs, ode_steps=1)
    query = EulerQueryTimeFlow(**kwargs, ode_steps=1)
    direct = AllenCahnDirectHeatMap(**kwargs, epsilon=0.1, length=2.0 * math.pi)
    copy_autonomous_into_query(autonomous, query)
    copy_autonomous_into_query(autonomous, direct)
    return autonomous, query, direct


def test_query_and_direct_add_only_the_tau_column():
    autonomous, query, direct = _pair()
    extra_query = parameter_count(query) - parameter_count(autonomous)
    extra_direct = parameter_count(direct) - parameter_count(autonomous)
    assert extra_query == QUERY_EXTRA_PARAMETERS
    assert extra_direct == QUERY_EXTRA_PARAMETERS


def test_one_euler_step_is_one_vector_field_evaluation():
    autonomous, _query, _direct = _pair()
    state = torch.randn(3, 16)
    calls = []
    original = autonomous.vector_field

    def counting(latent):
        calls.append(1)
        return original(latent)

    autonomous.vector_field = counting
    with torch.no_grad():
        evolved = autonomous(state, 0.1)
        expected = autonomous.decode(
            autonomous.encode(state) + 0.1 * original(autonomous.encode(state))
        )
    assert len(calls) == 1
    assert torch.allclose(evolved, expected, atol=1e-5, rtol=1e-5)


def test_heat_semigroup_composes_exactly():
    _autonomous, _query, direct = _pair()
    state = torch.randn(4, 16)
    with torch.no_grad():
        one = direct.heat_semigroup(state, 0.15)
        two = direct.heat_semigroup(direct.heat_semigroup(state, 0.075), 0.075)
    assert torch.allclose(one, two, atol=1e-6, rtol=1e-6)


def test_autonomous_equal_substep_composition_is_numerically_zero():
    autonomous, _query, _direct = _pair()
    row = equal_substep_autonomous_defect(autonomous, torch.randn(3, 16) * 0.3)
    assert row["rms_defect_mean"] == pytest.approx(0.0, abs=1e-6)


def test_direct_map_uses_exactly_one_learned_evaluation():
    _autonomous, _query, direct = _pair()
    state = torch.rand(2, 16) * 1.6 - 0.8
    calls = []
    original = direct.vector_field

    def counting(latent, tau):
        calls.append(1)
        return original(latent, tau)

    direct.vector_field = counting
    with torch.no_grad():
        direct(state, 0.075)
    assert len(calls) == 1


def test_deployed_cross_lag_keys_are_the_unseen_pair():
    autonomous, _query, _direct = _pair()
    row = cross_lag_defect(autonomous, torch.randn(2, 16) * 0.2, 0.075, 0.15)
    assert row["tau_fine"] == 0.075
    assert row["tau_coarse"] == 0.15
