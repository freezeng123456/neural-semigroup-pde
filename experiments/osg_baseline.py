"""Independent OSG-Net / GDSG adaptation for a periodic scalar Burgers grid.

References: Chen & Wu, Deep-OSG, JCP 493 (2023), 112498, Eq. (8),
https://arxiv.org/html/2302.03358v2; Chen, Wu & Xiu, DUE, SIAM Review
67(4) (2025), 873--902, https://doi.org/10.1137/24M1671827.

Author code inspected: https://github.com/AI4Equations/due at commit
1e629104a12a9c3e3064996cbd4e301cec641ad8 (2026-08-17).
Its LICENSE is GNU LGPL v2.1, SHA256
20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331.
Relevant files: due/networks/fcn.py::osgnet, due/models/ode_osg.py::train,
due/models/pde_osg.py::train and due/datasets/pde.py::normalize.

This module independently implements the published equations. It neither
imports nor vendors DUE source, and is an adapted baseline, not a reproduction
of the original paper's experiments. Differences from that implementation:

* A shared MLP on circular spatial patches replaces its whole-state MLP/FNO.
* The public API uses physical states and time lags; state normalization is a
  single training-only affine transform shared across all spatial locations.
  No clipping, per-trajectory fitting, logarithmic time transform, or test-data
  statistics are used. The time feature is affine, the outer lag is physical.
* The caller supplies GDSG states, allowing smooth training states instead of
  independent uniform values at every grid point. Our optional lag sampler
  keeps both sublags and their sum within the supervised lag interval; DUE's
  default independently sampled sublags can have an out-of-range sum.
* Both composition orders and every nested call remain differentiable. As in
  DUE, the loss is (data + lambda * GDSG) / (1 + lambda). Our default is physical
  MSE for comparison with the existing generator; DUE uses normalized states.
  An explicit state_scale can reproduce the normalized loss convention.
  Lambda changes the loss only, never the architecture, and scales the direct
  supervised gradient by 1/(1+lambda) as well as adding the GDSG term.

Zero-time identity and circular translation equivariance hold by construction.
Neither mean conservation nor energy dissipation nor an exact semigroup law
is imposed. Comparisons to a conservative, viscosity-anchored generator must
account for that different physical prior as well as capacity and work.
"""
from __future__ import annotations

import math
from numbers import Integral
from typing import Callable

import torch
from torch import nn


DUE_COMMIT = "1e629104a12a9c3e3064996cbd4e301cec641ad8"
DUE_LICENSE = "GNU Lesser General Public License v2.1"
DUE_LICENSE_SHA256 = "20c17d8b8c48a600800dfd14f95d5cb9ff47066a9641ddeab48dc54aec96e331"


def _finite_number(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _state_batch(state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor) or state.ndim != 2 or min(state.shape) == 0:
        raise ValueError("state must have nonempty shape [batch, grid]")
    if not state.is_floating_point():
        raise ValueError("state must have a floating-point dtype")


def _lag_column(tau, state: torch.Tensor) -> torch.Tensor:
    lag = torch.as_tensor(tau, dtype=state.dtype, device=state.device)
    if lag.ndim == 0:
        lag = lag.expand(state.shape[0])
    if lag.shape == (state.shape[0], 1):
        lag = lag[:, 0]
    if lag.shape != (state.shape[0],):
        raise ValueError("tau must be scalar or have one value per sample")
    if not torch.isfinite(lag).all() or (lag < 0).any():
        raise ValueError("tau must be finite and nonnegative")
    return lag[:, None]


def training_state_affine(*training_tensors: torch.Tensor) -> tuple[float, float]:
    """Return global center/half-range from explicitly supplied training data.

    The caller owns split isolation: pass training inputs and targets only.
    A constant training set gets scale one instead of a zero denominator.
    Values outside the fitted range are intentionally not clipped.
    """
    if not training_tensors:
        raise ValueError("provide at least one training tensor")
    extrema = []
    for tensor in training_tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or not tensor.is_floating_point():
            raise ValueError("training tensors must be nonempty floating-point tensors")
        tensor = tensor.detach().to(dtype=torch.float64)
        if not torch.isfinite(tensor).all():
            raise ValueError("training tensors must be finite")
        extrema.append((tensor.min().item(), tensor.max().item()))
    low, high = min(pair[0] for pair in extrema), max(pair[1] for pair in extrema)
    center, scale = low / 2 + high / 2, high / 2 - low / 2
    return center, scale if scale > 0 else 1.0


class PeriodicOSGNet(nn.Module):
    """``Phi_tau(u) = u + tau * scale * N(normalized_patch, time)``.

    The default radius=2, width=32, depth=2 has 1,313 parameters. They all
    participate in the parameterization; there is no disabled time channel.
    A single call uses one network evaluation, independent of the requested
    lag. Lambda=0 and lambda>0 should use identical copied initial state dicts.
    """

    def __init__(self, *, n_grid: int, radius: int = 2, hidden: int = 32,
                 hidden_layers: int = 2, state_center: float = 0.0,
                 state_scale: float = 1.0, tau_min: float = 0.025,
                 tau_max: float = 0.1):
        super().__init__()
        for name, value in (("n_grid", n_grid), ("hidden", hidden), ("hidden_layers", hidden_layers)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 0 or 2 * radius + 1 > n_grid:
            raise ValueError("radius must be nonnegative with a patch no wider than the grid")
        center = _finite_number(state_center, "state_center")
        scale = _finite_number(state_scale, "state_scale", positive=True)
        tmin, tmax = _finite_number(tau_min, "tau_min"), _finite_number(tau_max, "tau_max")
        if not 0 <= tmin < tmax:
            raise ValueError("require 0 <= tau_min < tau_max")
        self.n_grid, self.radius = int(n_grid), int(radius)
        self.hidden, self.hidden_layers = int(hidden), int(hidden_layers)
        # Float64 scalar metadata retain constructor precision in state_dict;
        # forward casts them to the actual state dtype/device.
        for name, value in (("state_center", center), ("state_scale", scale),
                            ("tau_min", tmin), ("tau_max", tmax)):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        layers = [nn.Linear(2 * self.radius + 2, self.hidden), nn.Tanh()]
        for _ in range(self.hidden_layers - 1):
            layers.extend((nn.Linear(self.hidden, self.hidden), nn.Tanh()))
        layers.append(nn.Linear(self.hidden, 1))
        self.increment = nn.Sequential(*layers)
        for layer in self.increment:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.2)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        _state_batch(state)
        if state.shape[1] != self.n_grid:
            raise ValueError(f"state grid must contain {self.n_grid} points")
        lag = _lag_column(tau, state)
        center, scale = self.state_center.to(state), self.state_scale.to(state)
        tmin, tmax = self.tau_min.to(state), self.tau_max.to(state)
        normalized = (state - center) / scale
        patches = torch.stack([torch.roll(normalized, -offset, dims=1)
                               for offset in range(-self.radius, self.radius + 1)], dim=-1)
        time_feature = (2 * lag - tmax - tmin) / (tmax - tmin)
        inputs = torch.cat((patches, time_feature[:, None].expand(-1, self.n_grid, -1)), dim=-1)
        return state + lag * scale * self.increment(inputs).squeeze(-1)


def _prediction(model: Callable, state: torch.Tensor, tau) -> torch.Tensor:
    prediction = model(state, tau)
    if not isinstance(prediction, torch.Tensor) or prediction.shape != state.shape:
        raise ValueError("model must return the same [batch, grid] shape as its input")
    return prediction


def _scale(state_scale, reference: torch.Tensor) -> torch.Tensor:
    scale = torch.as_tensor(state_scale, dtype=reference.dtype, device=reference.device)
    if scale.ndim != 0 or not torch.isfinite(scale) or scale <= 0:
        raise ValueError("state_scale must be a finite positive scalar")
    return scale


def gdsg_loss(model: Callable, states: torch.Tensor, s, t, *, state_scale=1.0) -> torch.Tensor:
    """Mean two-order GDSG residual; all five network calls receive gradients.

    ``states`` may contain B*Q flattened collocation fields. The caller chooses
    their distribution and records it; no validation/test states are selected
    or accessed here. Sublags and their sum are not silently clipped.
    """
    _state_batch(states)
    s, t = _lag_column(s, states), _lag_column(t, states)
    scale = _scale(state_scale, states)
    direct = _prediction(model, states, s + t)
    first_s = _prediction(model, states, s)
    s_then_t = _prediction(model, first_s, t)
    first_t = _prediction(model, states, t)
    t_then_s = _prediction(model, first_t, s)
    return 0.5 * (((s_then_t - direct) / scale).square().mean()
                  + ((t_then_s - direct) / scale).square().mean())


def osg_objective(model: Callable, initial: torch.Tensor, tau, target: torch.Tensor,
                  *, sg_weight: float = 0.0, sg_states: torch.Tensor | None = None,
                  sg_s=None, sg_t=None, state_scale=1.0) -> dict[str, torch.Tensor | int]:
    """DUE-weighted objective with separate data/GDSG diagnostics.

    Return keys: total, data, semigroup, network_evaluations. The last counts
    field evaluations rather than Python calls: B + 5*B_sg when lambda>0.
    By default both losses use physical MSE, even when the model internally
    normalizes states. Explicit scale divides all residuals for comparisons
    that intentionally use normalized MSE; it must be shared across models.
    With lambda=0 collocation inputs are unused and require no computation.
    """
    _state_batch(initial)
    if not isinstance(target, torch.Tensor) or target.shape != initial.shape:
        raise ValueError("target must have the same [batch, grid] shape as initial")
    weight = _finite_number(sg_weight, "sg_weight")
    if weight < 0:
        raise ValueError("sg_weight must be nonnegative")
    scale = _scale(state_scale, initial)
    data = ((_prediction(model, initial, tau) - target) / scale).square().mean()
    if weight == 0:
        return {"total": data, "data": data, "semigroup": data.new_zeros(()),
                "network_evaluations": initial.shape[0]}
    if sg_states is None or sg_s is None or sg_t is None:
        raise ValueError("positive sg_weight requires sg_states, sg_s and sg_t")
    residual = gdsg_loss(model, sg_states, sg_s, sg_t, state_scale=scale)
    return {"total": (data + weight * residual) / (1 + weight),
            "data": data, "semigroup": residual,
            "network_evaluations": initial.shape[0] + 5 * sg_states.shape[0]}


def sample_gdsg_times(n_samples: int, *, tau_min: float, tau_max: float,
                      generator: torch.Generator, device=None,
                      dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample s,t with s,t,s+t all in [tau_min,tau_max].

    The total is uniform on [2*tau_min,tau_max], then s is uniform on
    [tau_min,total-tau_min]. The explicit generator isolates these draws from
    initialization and minibatch RNGs. Sampling happens on its device and is
    optionally copied to the requested model device.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    low, high = _finite_number(tau_min, "tau_min", positive=True), _finite_number(tau_max, "tau_max", positive=True)
    if high < 2 * low:
        raise ValueError("in-range composition requires tau_max >= 2*tau_min")
    if not isinstance(generator, torch.Generator):
        raise ValueError("an explicit torch.Generator is required")
    if not dtype.is_floating_point:
        raise ValueError("dtype must be floating point")
    draws = torch.rand((2, n_samples), generator=generator, device=generator.device, dtype=dtype)
    total = 2 * low + (high - 2 * low) * draws[0]
    first = low + (total - 2 * low) * draws[1]
    return first.to(device=device), (total - first).to(device=device)
