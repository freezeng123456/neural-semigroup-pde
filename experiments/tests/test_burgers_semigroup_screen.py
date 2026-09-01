"""Focused structural checks for the matched Burgers screen."""

from __future__ import annotations

import torch

from experiments.run_burgers_semigroup_screen import PeriodicBurgersFluxFlow


def build_pair():
    torch.manual_seed(7)
    autonomous = PeriodicBurgersFluxFlow(
        n_grid=16,
        length=2.0 * torch.pi,
        viscosity=0.01,
        hidden=8,
        ode_steps=2,
        query_conditioned=False,
    )
    query = PeriodicBurgersFluxFlow(
        n_grid=16,
        length=2.0 * torch.pi,
        viscosity=0.01,
        hidden=8,
        ode_steps=2,
        query_conditioned=True,
    )
    query.load_state_dict(autonomous.state_dict(), strict=True)
    return autonomous, query


def test_matched_models_have_identical_initial_parameters():
    autonomous, query = build_pair()
    assert sum(p.numel() for p in autonomous.parameters()) == sum(
        p.numel() for p in query.parameters()
    )
    for left, right in zip(autonomous.parameters(), query.parameters()):
        assert torch.equal(left, right)


def test_periodic_split_generator_preserves_spatial_mean():
    autonomous, query = build_pair()
    state = torch.randn(4, 16)
    for model in (autonomous, query):
        rhs = model.rhs(state, 0.04)
        assert torch.allclose(rhs.mean(dim=1), torch.zeros(4), atol=1e-7)
        evolved = model(state, 0.04)
        assert torch.allclose(evolved.mean(dim=1), state.mean(dim=1), atol=1e-6)


def test_autonomous_equal_work_composition_is_numerically_consistent():
    autonomous, _query = build_pair()
    state = torch.randn(2, 16)
    composed = autonomous(autonomous(state, 0.04), 0.04)
    direct = autonomous.integrate(
        state,
        0.08,
        substeps=4,
        conditioning_tau=0.08,
    )
    assert torch.allclose(composed, direct, rtol=1e-5, atol=1e-6)
