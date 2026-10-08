"""Scientific contracts for the independently adapted OSG/GDSG baseline."""
import copy
import math

import pytest
import torch

from experiments.osg_baseline import (
    PeriodicOSGNet, gdsg_loss, osg_objective, sample_gdsg_times,
    training_state_affine,
)


@pytest.mark.parametrize("hidden,layers,count", [(4, 1, 33), (32, 2, 1313)])
def test_capacity_identity_and_periodic_equivariance(hidden, layers, count):
    torch.manual_seed(7)
    model = PeriodicOSGNet(n_grid=16, hidden=hidden, hidden_layers=layers).double()
    u = torch.randn(3, 16, dtype=torch.float64)
    tau = torch.tensor([0.025, 0.05, 0.1], dtype=torch.float64)
    assert sum(p.numel() for p in model.parameters()) == count
    assert torch.equal(model(u, 0), u)
    expected = model(u, tau).roll(5, 1)
    torch.testing.assert_close(model(u.roll(5, 1), tau[:, None]), expected, rtol=1e-13, atol=1e-13)
    per_sample = torch.cat([model(u[i:i+1], tau[i]) for i in range(3)])
    torch.testing.assert_close(per_sample, model(u, tau), rtol=1e-13, atol=1e-13)


def test_normalization_is_training_global_and_no_clipping_or_per_sample_fit():
    first = torch.tensor([[-2., 0.], [2., 4.]])
    second = torch.tensor([[6., 8.]])
    assert training_state_affine(first, second) == (3., 5.)
    assert training_state_affine(torch.full((2, 3), 4.)) == (4., 1.)
    model = PeriodicOSGNet(n_grid=8, radius=2, hidden=4, hidden_layers=1).double()
    converted = copy.deepcopy(model)
    converted.state_center.fill_(3.)
    converted.state_scale.fill_(5.)
    # Include states beyond the fitted [-1,1] interval; affine covariance
    # would fail if either trajectory normalization or clipping were added.
    u = torch.linspace(-3, 3, 16, dtype=torch.float64).reshape(2, 8)
    torch.testing.assert_close(converted(5*u+3, .04), 5*model(u, .04)+3, rtol=1e-13, atol=1e-13)


def test_time_feature_and_outer_factor_use_distinct_units():
    class Capture(torch.nn.Module):
        def forward(self, features):
            self.features = features
            return torch.ones_like(features[..., :1])
    model = PeriodicOSGNet(n_grid=8, state_center=2, state_scale=4,
                           tau_min=.025, tau_max=.1).double()
    model.increment = Capture()
    u = torch.arange(16, dtype=torch.float64).reshape(2, 8)
    tau = torch.tensor([.025, .1], dtype=torch.float64)
    torch.testing.assert_close(model(u, tau), u + 4*tau[:, None])
    torch.testing.assert_close(model.increment.features[..., -1], torch.tensor([[-1.]*8, [1.]*8], dtype=torch.float64))
    torch.testing.assert_close(model.increment.features[..., 2], (u-2)/4)
    # At the left boundary, the patch wraps to the last two grid points.
    torch.testing.assert_close(model.increment.features[0, 0, :5], (u[0, [6, 7, 0, 1, 2]]-2)/4)


def test_gdsg_is_zero_for_an_exact_flow():
    def flow(u, tau):
        return u * torch.exp(-torch.as_tensor(tau, dtype=u.dtype).reshape(-1, 1))
    u = torch.randn(4, 8, dtype=torch.float64)
    assert gdsg_loss(flow, u, .025, .05).item() < 1e-30


def test_gdsg_has_the_analytic_residual_and_full_nested_gradient():
    class Euler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(.7, dtype=torch.float64))
        def forward(self, u, tau):
            return (1+self.a*torch.as_tensor(tau, dtype=u.dtype).reshape(-1, 1))*u
    model = Euler()
    u = torch.tensor([[1., 2.], [-1., 3.]], dtype=torch.float64)
    loss = gdsg_loss(model, u, .025, .05)
    expected = model.a**4 * .025**2 * .05**2 * u.square().mean()
    torch.testing.assert_close(loss, expected, rtol=1e-11, atol=1e-16)
    loss.backward()
    torch.testing.assert_close(model.a.grad, 4*model.a.detach()**3*.025**2*.05**2*u.square().mean(), rtol=1e-11, atol=1e-16)


def test_both_composition_orders_are_used():
    def nonlinear(u, lag):
        lag = torch.as_tensor(lag, dtype=u.dtype).reshape(-1, 1)
        return u + lag*u.square() + lag.square()
    u = torch.tensor([[.1, .4], [.2, .8]], dtype=torch.float64)
    s, t = .025, .065
    direct = nonlinear(u, s+t)
    forward = (nonlinear(nonlinear(u, s), t)-direct).square().mean()
    reverse = (nonlinear(nonlinear(u, t), s)-direct).square().mean()
    assert not math.isclose(forward.item(), reverse.item(), rel_tol=1e-6)
    torch.testing.assert_close(gdsg_loss(nonlinear, u, s, t), .5*(forward+reverse))


def test_lambda_zero_uses_identical_architecture_one_call_and_physical_loss():
    class Spy(PeriodicOSGNet):
        def forward(self, *args):
            self.calls += 1
            return super().forward(*args)
    model = Spy(n_grid=8, hidden=4, hidden_layers=1, state_scale=4.).double()
    model.calls = 0
    u = torch.randn(3, 8, dtype=torch.float64)
    target = .9*u
    expected = (model(u, .05)-target).square().mean()
    model.calls = 0
    result = osg_objective(model, u, .05, target, sg_weight=0)
    assert model.calls == 1 and result["network_evaluations"] == len(u)
    assert result["total"] is result["data"]
    torch.testing.assert_close(result["total"], expected)
    assert result["semigroup"].item() == 0
    result["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_objective_diagnostics_weighting_and_normalized_option():
    model = PeriodicOSGNet(n_grid=8, hidden=4, hidden_layers=1).double()
    u = torch.randn(3, 8, dtype=torch.float64)
    target = .9*u
    result = osg_objective(model, u, .05, target, sg_weight=2,
                           sg_states=u[:2], sg_s=.025, sg_t=.05)
    torch.testing.assert_close(result["total"], (result["data"]+2*result["semigroup"])/3)
    assert result["network_evaluations"] == 3+5*2
    normalized = osg_objective(model, u, .05, target, sg_weight=2,
                               sg_states=u[:2], sg_s=.025, sg_t=.05, state_scale=4)
    for key in ("total", "data", "semigroup"):
        torch.testing.assert_close(normalized[key], result[key]/16)
    result["total"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_in_range_sampling_is_reproducible_and_uses_separate_rng():
    global_state = torch.random.get_rng_state().clone()
    args = dict(n_samples=4096, tau_min=.025, tau_max=.1, dtype=torch.float64)
    s, t = sample_gdsg_times(**args, generator=torch.Generator().manual_seed(123))
    s2, t2 = sample_gdsg_times(**args, generator=torch.Generator().manual_seed(123))
    assert torch.equal(s, s2) and torch.equal(t, t2)
    assert torch.equal(global_state, torch.random.get_rng_state())
    for lag in (s, t, s+t):
        assert (lag >= .025-1e-15).all() and (lag <= .1+1e-15).all()
    s, t = sample_gdsg_times(3, tau_min=.025, tau_max=.05,
                             generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    torch.testing.assert_close(s, torch.full((3,), .025, dtype=torch.float64))
    torch.testing.assert_close(s, t)


@pytest.mark.parametrize("options", [{"state_scale":0}, {"state_center":math.nan},
    {"tau_min":.1,"tau_max":.1}, {"radius":4}, {"hidden":True}, {"hidden_layers":0}])
def test_invalid_model_contract(options):
    with pytest.raises(ValueError):
        PeriodicOSGNet(n_grid=8, **options)


@pytest.mark.parametrize("lag", [-.1, math.nan, math.inf, [0.1, 0.2]])
def test_invalid_lags(lag):
    model = PeriodicOSGNet(n_grid=8)
    with pytest.raises(ValueError):
        model(torch.ones(3, 8), lag)


def test_invalid_loss_and_sampling_contracts():
    u = torch.ones(2, 8)
    model = PeriodicOSGNet(n_grid=8)
    with pytest.raises(ValueError, match="target"):
        osg_objective(model, u, .05, u[:, :1])
    with pytest.raises(ValueError, match="requires"):
        osg_objective(model, u, .05, u, sg_weight=1)
    with pytest.raises(ValueError, match="nonnegative"):
        osg_objective(model, u, .05, u, sg_weight=-1)
    with pytest.raises(ValueError, match="state_scale"):
        gdsg_loss(model, u, .02, .03, state_scale=torch.tensor([1., 2.]))
    with pytest.raises(ValueError, match="same"):
        gdsg_loss(lambda state, lag: state[:, :1], u, .02, .03)
    with pytest.raises(ValueError, match="in-range"):
        sample_gdsg_times(2, tau_min=.06, tau_max=.1, generator=torch.Generator())
    with pytest.raises(ValueError, match="finite"):
        training_state_affine(torch.tensor([float("nan")]))


def test_normalization_buffers_survive_checkpoint_roundtrip():
    model = PeriodicOSGNet(n_grid=8, state_center=.25, state_scale=.75,
                           tau_min=.02, tau_max=.2).double()
    restored = PeriodicOSGNet(n_grid=8).double()
    restored.load_state_dict(model.state_dict())
    u = torch.randn(2, 8, dtype=torch.float64)
    assert torch.equal(model(u, .04), restored(u, .04))
