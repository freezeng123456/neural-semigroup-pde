"""Contracts for the Wave 2 work-matched direct-map lane."""

from __future__ import annotations

import torch

from experiments.boundary_family_direct_map import (
    EXPECTED_PARAMETERS,
    MATCHED_NET_EVALUATIONS,
    REFINEMENT_SWEEP,
    BoundaryFamilyDirectMap,
    build_paired_models,
    compose,
    flow_cross_lag,
    flow_cross_lag_equal_substeps,
    parameter_count,
)
from experiments.boundary_family_semigroup import BoundaryFamilySpec, residual_max


def _spec(n_grid: int = 17) -> BoundaryFamilySpec:
    return BoundaryFamilySpec(family="homogeneous_neumann", dx=1.0 / (n_grid - 1))


def _models(n_grid: int = 17):
    torch.manual_seed(6)
    return build_paired_models(
        n_grid=n_grid,
        hidden_width=8,
        boundary_spec=_spec(n_grid),
        query_time_scale=0.08,
        rk4_steps=2,
    )


def test_all_five_models_share_the_wave2_parameter_count_on_production_width():
    torch.manual_seed(6)
    models = build_paired_models(
        n_grid=33,
        hidden_width=32,
        boundary_spec=BoundaryFamilySpec(family="robin", dx=1.0 / 32),
        query_time_scale=0.08,
        rk4_steps=8,
    )
    for model in models.values():
        assert parameter_count(model) == EXPECTED_PARAMETERS


def test_paired_initialization_copies_one_fingerprint():
    models = _models()
    reference = list(models["a_rk4_autonomous"].vector_field_net.parameters())
    for label in (
        "b_rk4_query_time",
        "a_euler_autonomous",
        "b_euler_query_time",
        "c_direct_map",
    ):
        candidate = models[label]
        net = (
            candidate.vector_field_net
            if isinstance(candidate, BoundaryFamilyDirectMap)
            else candidate.flow.vector_field_net
            if hasattr(candidate, "flow")
            else candidate.vector_field_net
        )
        for left, right in zip(reference, net.parameters()):
            assert torch.equal(left, right)


def test_direct_map_is_not_an_euler_step():
    models = _models()
    state = torch.randn(4, 17)
    tau = 0.08
    direct = models["c_direct_map"]
    euler = models["b_euler_query_time"]
    with torch.no_grad():
        residual_map = direct(state, tau)
        euler_step = euler(state, tau)
        projected = euler.flow.project_state(state)
        field_scale = euler.flow.project_state(
            projected + tau * euler.vector_field(projected, tau)
        )
    assert not torch.allclose(residual_map, euler_step, atol=1e-5)
    assert torch.allclose(euler_step, field_scale, atol=1e-6)


def test_one_euler_call_is_one_field_evaluation():
    models = _models()
    model = models["a_euler_autonomous"]
    state = torch.randn(3, 17)
    calls = []
    original = model.vector_field

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    model.vector_field = counting
    with torch.no_grad():
        evolved = model(state, 0.04)
    assert len(calls) == MATCHED_NET_EVALUATIONS
    with torch.no_grad():
        assert torch.allclose(evolved, model.flow.project_state(
            model.flow.project_state(state)
            + 0.04 * original(model.flow.project_state(state), 0.04)
        ))


def test_hard_maps_stay_on_the_boundary_space():
    models = _models()
    state = torch.randn(5, 17)
    for model in models.values():
        with torch.no_grad():
            evolved = model(state, 0.04)
        residual = model.boundary_spec.state_residual(evolved)
        assert float(residual_max(residual)) <= 5e-6


def test_autonomous_field_composes_exactly_at_equal_substeps():
    models = _models()
    rows = flow_cross_lag_equal_substeps(
        models["a_euler_autonomous"], torch.randn(3, 17), 1
    )
    assert rows["rms_defect_mean"] == 0.0
    assert rows["coarse_substeps_per_call"] == 2 * rows["fine_substeps_per_call"]


def test_autonomous_cross_lag_defect_refines_toward_zero():
    models = _models()
    sample = torch.randn(3, 17) * 0.2
    coarse = flow_cross_lag(models["a_euler_autonomous"], sample, 1)
    fine = flow_cross_lag(models["a_euler_autonomous"], sample, 16)
    assert fine["rms_defect_mean"] < coarse["rms_defect_mean"]
    assert set(str(value) for value in REFINEMENT_SWEEP) >= {"1", "16", "128"}


def test_compose_depth_matches_horizon():
    models = _models()
    state = torch.zeros(2, 17)
    with torch.no_grad():
        once = models["c_direct_map"](state, 0.04)
        twice = compose(models["c_direct_map"], state, tau=0.04, depth=2)
    assert once.shape == state.shape
    assert twice.shape == state.shape
