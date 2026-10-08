"""Identify functionally inactive flow parameters without rewriting old counts.

Stage A records two distinct, analytic redundancies: autonomous first-layer
control weights and the scalar output bias of a periodically differentiated
flux. Subtracting these gives an upper bound on functional parameter count,
not a proof that all remaining network coordinates are identifiable. Historical
``run_burgers_validated.parameter_counts`` values retain their original meaning.
"""
from __future__ import annotations

import copy
import math

import pytest
import torch

from experiments import run_burgers_validated as historical


def flow_and_state(architecture: str, query: bool):
    # Float64 isolates the mathematical invariance from float32 cancellation.
    torch.manual_seed(1847)
    config = {"N": 64, "L": 2 * math.pi, "nu": 0.01}
    model = historical.build_model(architecture, query, config, ode_steps=2).double()
    grid = torch.arange(64, dtype=torch.float64) * (2 * math.pi / 64)
    state = torch.stack((0.35 * torch.sin(grid) + 0.15 * torch.cos(2 * grid),
                         0.27 * torch.cos(grid) - 0.18 * torch.sin(3 * grid)))
    # A spatially varying loss avoids measuring the conserved mean alone.
    weights = 0.8 + 0.25 * torch.cos(2 * grid) + 0.15 * torch.sin(3 * grid)
    return model, state, weights


@pytest.mark.parametrize("architecture,nominal,control_width", [
    ("small", 33, 8), ("large", 1313, 32),
])
@pytest.mark.parametrize("query", [False, True])
def test_constant_flux_bias_is_a_gauge_for_rhs_rollout_and_gradient(
        architecture, nominal, control_width, query):
    model, state, weights = flow_and_state(architecture, query)
    shifted = copy.deepcopy(model)
    with torch.no_grad():
        shifted.flux[-1].bias.add_(0.37)

    assert torch.allclose(model.rhs(state, 0.075), shifted.rhs(state, 0.075),
                          rtol=0.0, atol=2e-14)
    prediction = model(state, 0.075)
    assert torch.allclose(prediction, shifted(state, 0.075), rtol=0.0, atol=2e-14)
    (prediction * weights).square().mean().backward()
    assert model.flux[-1].bias.grad is not None
    assert model.flux[-1].bias.grad.abs().max().item() < 2e-14
    assert model.flux[-1].weight.grad.abs().max().item() > 1e-10

    # These are separate counts in the NEW study's audit, not replacements
    # for the historical nominal/control-only counting convention.
    inactive_control_weight_count = 0 if query else control_width
    flux_constant_gauge_parameter_count = model.flux[-1].bias.numel()
    actual_nominal = sum(parameter.numel() for parameter in model.parameters())
    assert actual_nominal == nominal
    assert flux_constant_gauge_parameter_count == 1
    functional_parameter_count_upper_bound = (
        actual_nominal - inactive_control_weight_count - flux_constant_gauge_parameter_count
    )
    expected = {("small", False): 24, ("small", True): 32,
                ("large", False): 1280, ("large", True): 1312}
    assert functional_parameter_count_upper_bound == expected[architecture, query]


@pytest.mark.parametrize("architecture", ["small", "large"])
def test_autonomous_control_weights_have_zero_gradient_and_no_functional_effect(architecture):
    model, state, weights = flow_and_state(architecture, False)
    shifted = copy.deepcopy(model)
    with torch.no_grad():
        shifted.flux[0].weight[:, -1].add_(
            torch.linspace(-0.8, 0.6, shifted.flux[0].out_features, dtype=torch.float64)
        )
    prediction = model(state, 0.075)
    assert torch.equal(prediction, shifted(state, 0.075))
    assert torch.equal(model.rhs(state, 0.025), model.rhs(state, 0.1))
    (prediction * weights).square().mean().backward()
    control_gradient = model.flux[0].weight.grad[:, -1]
    assert torch.equal(control_gradient, torch.zeros_like(control_gradient))
    assert model.flux[0].weight.grad[:, :-1].abs().max().item() > 1e-10


@pytest.mark.parametrize("architecture", ["small", "large"])
def test_query_control_weights_are_not_counted_as_inactive(architecture):
    model, state, weights = flow_and_state(architecture, True)
    shifted = copy.deepcopy(model)
    with torch.no_grad():
        shifted.flux[0].weight[:, -1].add_(
            torch.linspace(-0.8, 0.6, shifted.flux[0].out_features, dtype=torch.float64)
        )
    prediction = model(state, 0.075)
    assert (prediction - shifted(state, 0.075)).abs().max().item() > 1e-8
    (prediction * weights).square().mean().backward()
    assert model.flux[0].weight.grad[:, -1].abs().max().item() > 1e-10


@pytest.mark.parametrize("architecture", ["small", "large"])
def test_zero_query_control_recovers_the_identically_initialized_autonomous_field(architecture):
    autonomous, state, _ = flow_and_state(architecture, False)
    query, _, _ = flow_and_state(architecture, True)
    assert historical.state_digest(autonomous.state_dict()) == historical.state_digest(query.state_dict())
    assert torch.equal(autonomous.rhs(state, 0.075), query.rhs(state, 0.0))
