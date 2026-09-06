"""Checks for the zero-parameter Fisher--KPP physics split."""

from __future__ import annotations

import math

import torch

from experiments.run_fisher_physics_split import (
    FisherPhysicsSplit,
    compare_to_formal_a,
    parameter_count,
)


def test_physics_split_has_zero_parameters():
    model = FisherPhysicsSplit(n_grid=16, length=10.0)
    assert parameter_count(model) == 0


def test_heat_semigroup_composes():
    model = FisherPhysicsSplit(n_grid=16, length=10.0, nu=0.1)
    state = torch.rand(3, 16) * 0.8 + 0.1
    with torch.no_grad():
        one = model.heat_semigroup(state, 0.15)
        two = model.heat_semigroup(model.heat_semigroup(state, 0.075), 0.075)
    assert torch.allclose(one, two, atol=1e-6, rtol=1e-6)


def test_forward_is_projected_heat_plus_reaction():
    model = FisherPhysicsSplit(n_grid=16, length=10.0, nu=0.2, reaction_rate=1.5)
    state = torch.rand(4, 16) * 0.7 + 0.15
    tau = 0.08
    with torch.no_grad():
        linear = model.heat_semigroup(state, tau)
        expected = model.project(linear + tau * model.increment(linear))
        assert torch.allclose(model(state, tau), expected, atol=1e-6)
        assert float(model(state, tau).min()) > 0.0
        assert float(model(state, tau).max()) < 1.0


def test_comparison_uses_the_frozen_threshold():
    physics = {
        "tau=0.075,T=1.2": {"rollout_mse_mean": 0.09},
        "tau=0.075,T=2.4": {"rollout_mse_mean": 0.09},
        "tau=0.075,T=4.8": {"rollout_mse_mean": 0.09},
        "tau=0.15,T=1.2": {"rollout_mse_mean": 0.09},
        "tau=0.15,T=2.4": {"rollout_mse_mean": 0.09},
        "tau=0.15,T=4.8": {"rollout_mse_mean": 0.09},
    }
    published = [
        {"seed": 1, "tau": 0.075, "horizon": horizon, "a_mse": 0.10}
        for horizon in (1.2, 2.4, 4.8)
    ] + [
        {"seed": 1, "tau": 0.15, "horizon": horizon, "a_mse": 0.10}
        for horizon in (1.2, 2.4, 4.8)
    ]
    result = compare_to_formal_a(physics, published)
    assert math.isclose(result["geometric_mean_c_phys_over_a"], 0.9)
    assert result["decision"] == "fisher_physics_split_dominates_formal_A"
