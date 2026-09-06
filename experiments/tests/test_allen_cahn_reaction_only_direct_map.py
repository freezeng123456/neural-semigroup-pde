"""Checks for the reaction-only Allen--Cahn direct map."""

from __future__ import annotations

import math

import torch

from experiments.run_allen_cahn_reaction_only_direct_map import (
    AllenCahnPhysicsSplit,
    AllenCahnReactionOnlyDirectMap,
)
from experiments.run_allen_cahn_work_matched_direct_map import parameter_count


def test_reaction_only_has_no_interaction_weights():
    model = AllenCahnReactionOnlyDirectMap(
        n_grid=16, epsilon=0.1, length=2.0 * math.pi
    )
    names = set(model.state_dict())
    assert not any("interaction" in name for name in names)
    assert parameter_count(model) > 8000


def test_one_call_is_one_residual_evaluation():
    model = AllenCahnReactionOnlyDirectMap(
        n_grid=16, epsilon=0.1, length=2.0 * math.pi
    )
    state = torch.rand(3, 16) * 1.6 - 0.8
    calls = []
    original = model.residual

    def counting(current, tau):
        calls.append(1)
        return original(current, tau)

    model.residual = counting
    with torch.no_grad():
        model(state, 0.1)
    assert len(calls) == 1


def test_physics_split_matches_zero_residual():
    torch.manual_seed(3)
    learned = AllenCahnReactionOnlyDirectMap(
        n_grid=16, epsilon=0.1, length=2.0 * math.pi
    )
    with torch.no_grad():
        for parameter in learned.parameters():
            parameter.zero_()
        physics = AllenCahnPhysicsSplit(
            n_grid=16, epsilon=0.1, length=2.0 * math.pi
        )
        state = torch.rand(4, 16) * 1.4 - 0.7
        assert torch.allclose(learned(state, 0.08), physics(state, 0.08), atol=1e-6)


def test_heat_semigroup_still_composes():
    model = AllenCahnReactionOnlyDirectMap(
        n_grid=16, epsilon=0.1, length=2.0 * math.pi
    )
    state = torch.randn(3, 16)
    with torch.no_grad():
        one = model.heat_semigroup(state, 0.15)
        two = model.heat_semigroup(model.heat_semigroup(state, 0.075), 0.075)
    assert torch.allclose(one, two, atol=1e-6, rtol=1e-6)
