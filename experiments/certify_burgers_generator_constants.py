#!/usr/bin/env python3
"""Certified constants and rollout bound for the minimal Burgers generator.

Every constant here is a bound over a continuum, obtained from the weight
matrices and from exact operator spectra, not a maximum over sampled states.
``docs/research/FISHER_KPP_THEOREM_CARD.md`` records that the project's
generator-matching and one-sided-stability constants had only finite-sample
estimates, which "cannot be substituted into a theorem as certified uniform
constants without an additional covering, interval-bound, or analytic
argument".  The adopted 33-parameter generator is small enough to supply those
arguments: its flux is a pointwise one-hidden-layer tanh map, so its
derivative bounds are analytic, and a covering argument turns a grid maximum
into a certified supremum.

The bound proved here is for the exact flow of the learned semidiscrete
generator against the exact flow of the finite-difference semidiscrete Burgers
generator.  Time-integration error and the finite-difference-versus-spectral
spatial consistency term are reported separately and are *not* certified here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import torch_load_compat  # noqa: E402

DTYPE = torch.float64
# sup |d/dx sech^2 x| = max over t=tanh(x) of 2|t|(1-t^2) at t = 1/sqrt(3).
SECH2_DERIVATIVE_SUP = 4.0 / (3.0 * math.sqrt(3.0))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ScalarFlux:
    """The autonomous flux as a scalar map, with certified derivative bounds.

    With patch radius zero and a zeroed control channel the flux network reads
    a single number, so ``f(s) = W2 tanh(w s + b1) + b2`` with
    ``w = W1[:, 0]``.
    """

    def __init__(self, state_dict: dict):
        first = state_dict["flux.0.weight"].to(DTYPE)
        self.w = first[:, 0].clone()
        self.control_column = first[:, 1].clone()
        self.b1 = state_dict["flux.0.bias"].to(DTYPE).clone()
        self.w2 = state_dict["flux.2.weight"].to(DTYPE).reshape(-1).clone()
        self.b2 = float(state_dict["flux.2.bias"].to(DTYPE).item())

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        activation = torch.tanh(state.unsqueeze(-1) * self.w + self.b1)
        return activation @ self.w2 + self.b2

    def derivative(self, state: torch.Tensor) -> torch.Tensor:
        pre = state.unsqueeze(-1) * self.w + self.b1
        gate = 1.0 - torch.tanh(pre) ** 2
        return gate @ (self.w2 * self.w)

    def derivative_sup_analytic(self) -> float:
        """0 < sech^2 <= 1 gives an unconditional bound on |f'|."""

        return float(torch.abs(self.w2 * self.w).sum().item())

    def second_derivative_bound(self) -> float:
        """Lipschitz bound for f', used by the covering argument."""

        return float(
            SECH2_DERIVATIVE_SUP
            * torch.abs(self.w2 * self.w * self.w).sum().item()
        )

    def derivative_sup_covering(self, bound: float, samples: int) -> float:
        grid = torch.linspace(-bound, bound, samples, dtype=DTYPE)
        spacing = float(2.0 * bound / (samples - 1))
        grid_max = float(self.derivative(grid).abs().max().item())
        return grid_max + self.second_derivative_bound() * spacing / 2.0

    def flux_error_sup_covering(
        self, bound: float, samples: int, lipschitz: float
    ) -> dict[str, float]:
        """Certified gauge-invariant sup of ``f(s) - s^2/2`` on [-bound, bound].

        A discrete divergence annihilates constants, so the flux is only
        determined up to an additive constant and the generator error depends
        on ``f - s^2/2`` only through its variation.  The certified quantity is
        therefore the half-range after subtracting the optimal constant, not
        the raw supremum.

        The difference has derivative bounded by ``lipschitz + bound``, so a
        grid extremum plus half a spacing times that bound brackets the true
        extremum over the whole interval.
        """

        grid = torch.linspace(-bound, bound, samples, dtype=DTYPE)
        spacing = float(2.0 * bound / (samples - 1))
        remainder = (lipschitz + bound) * spacing / 2.0
        difference = self(grid) - 0.5 * grid**2
        grid_max = float(difference.max().item())
        grid_min = float(difference.min().item())
        return {
            "gauge_invariant_sup": (grid_max - grid_min) / 2.0 + remainder,
            "grid_half_range": (grid_max - grid_min) / 2.0,
            "optimal_constant": (grid_max + grid_min) / 2.0,
            "raw_grid_sup": float(difference.abs().max().item()),
            "covering_remainder": remainder,
        }


def periodic_operators(n_grid: int, length: float):
    dx = length / n_grid
    divergence = torch.zeros((n_grid, n_grid), dtype=DTYPE)
    laplacian = torch.zeros((n_grid, n_grid), dtype=DTYPE)
    for index in range(n_grid):
        divergence[index, (index + 1) % n_grid] += 1.0 / (2.0 * dx)
        divergence[index, (index - 1) % n_grid] -= 1.0 / (2.0 * dx)
        laplacian[index, (index + 1) % n_grid] += 1.0 / dx**2
        laplacian[index, index] -= 2.0 / dx**2
        laplacian[index, (index - 1) % n_grid] += 1.0 / dx**2
    return divergence, laplacian, dx


def operator_norms(n_grid: int, dx: float) -> dict[str, float]:
    """Exact spectral norms of the two circulant operators."""

    wavenumber = torch.arange(n_grid, dtype=DTYPE)
    angle = 2.0 * math.pi * wavenumber / n_grid
    return {
        "divergence_spectral_norm": float(
            (torch.sin(angle).abs().max() / dx).item()
        ),
        "laplacian_spectral_norm": float(
            ((2.0 - 2.0 * torch.cos(angle)).abs().max() / dx**2).item()
        ),
    }


def learned_generator(state: torch.Tensor, flux: ScalarFlux, divergence, laplacian, viscosity):
    return -(divergence @ flux(state.T)).T + viscosity * (laplacian @ state.T).T


def reference_generator(state: torch.Tensor, divergence, laplacian, viscosity):
    return -(divergence @ (0.5 * state.T**2)).T + viscosity * (laplacian @ state.T).T


def rk4(field, state: torch.Tensor, horizon: float, steps: int) -> torch.Tensor:
    dt = horizon / steps
    for _ in range(steps):
        k1 = field(state)
        k2 = field(state + 0.5 * dt * k1)
        k3 = field(state + 0.5 * dt * k2)
        k4 = field(state + dt * k3)
        state = state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return state


def mesh_norm(vectors: torch.Tensor, dx: float) -> torch.Tensor:
    return torch.sqrt(dx * vectors.square().sum(dim=-1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--expect-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-trajectory", type=int, default=50)
    parser.add_argument("--covering-samples", type=int, default=2_000_001)
    parser.add_argument("--range-margin", type=float, default=0.05)
    parser.add_argument("--reference-steps-per-unit-time", type=int, default=10_000)
    parser.add_argument("--horizons", type=float, nargs="*", default=[0.4, 0.8])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    digest = sha256_file(args.data_cache)
    if digest != args.expect_cache_sha256:
        raise RuntimeError(
            f"data cache digest mismatch: {digest} != {args.expect_cache_sha256}"
        )
    cache = torch_load_compat(args.data_cache, map_location="cpu")
    config = cache["config"]
    n_grid, length, viscosity = int(config["N"]), float(config["L"]), float(config["nu"])
    payload = torch_load_compat(args.checkpoint, map_location="cpu")
    flux = ScalarFlux(payload["model_state_dict"])
    divergence, laplacian, dx = periodic_operators(n_grid, length)
    norms = operator_norms(n_grid, dx)

    # The zeroed control channel is what makes the flux a scalar map. Record
    # the discarded column so the reduction is auditable rather than assumed.
    control_column_norm = float(flux.control_column.norm().item())

    initial = cache["val_u0"][: args.n_trajectory].to(DTYPE)

    def learned(state):
        return learned_generator(state, flux, divergence, laplacian, viscosity)

    def reference(state):
        return reference_generator(state, divergence, laplacian, viscosity)

    # One pass at the longest horizon fixes the state range the bound is
    # conditional on; both trajectories must stay inside it.
    longest = max(args.horizons)
    steps = int(round(longest * args.reference_steps_per_unit_time))
    observed_max = 0.0
    trajectory_errors = {}
    for horizon in sorted(args.horizons):
        substeps = int(round(horizon * args.reference_steps_per_unit_time))
        learned_state = rk4(learned, initial, horizon, substeps)
        reference_state = rk4(reference, initial, horizon, substeps)
        observed_max = max(
            observed_max,
            float(learned_state.abs().max().item()),
            float(reference_state.abs().max().item()),
        )
        trajectory_errors[str(horizon)] = {
            "mesh_norm_mean": float(
                mesh_norm(learned_state - reference_state, dx).mean().item()
            ),
            "mesh_norm_max": float(
                mesh_norm(learned_state - reference_state, dx).max().item()
            ),
            "reference_rk4_substeps": substeps,
        }
    state_bound = max(
        observed_max, float(initial.abs().max().item())
    ) + args.range_margin

    analytic_lipschitz = flux.derivative_sup_analytic()
    covering_lipschitz = flux.derivative_sup_covering(
        state_bound, args.covering_samples
    )
    flux_lipschitz = min(analytic_lipschitz, covering_lipschitz)
    flux_error_report = flux.flux_error_sup_covering(
        state_bound, args.covering_samples, flux_lipschitz
    )
    flux_error = flux_error_report["gauge_invariant_sup"]

    learned_lipschitz = (
        norms["divergence_spectral_norm"] * flux_lipschitz
        + viscosity * norms["laplacian_spectral_norm"]
    )
    learned_one_sided = norms["divergence_spectral_norm"] * flux_lipschitz
    reference_one_sided = norms["divergence_spectral_norm"] * state_bound
    generator_matching = (
        norms["divergence_spectral_norm"] * flux_error * math.sqrt(length)
    )

    # Sampled diagnostics, computed before the bound so the decomposition can
    # show what a sharper one-sided estimate would buy. These are maxima over
    # sampled pairs and are NOT certified.
    snapshots = torch.cat(
        [initial]
        + [
            rk4(reference, initial, horizon, int(round(horizon * 500)))
            for horizon in sorted(args.horizons)
        ]
    )
    empirical_one_sided = -math.inf
    for offset in (1, 7, 23):
        left = snapshots
        right = torch.roll(snapshots, shifts=offset, dims=0)
        difference = left - right
        keep = mesh_norm(difference, dx) > 1e-9
        if not bool(keep.any()):
            continue
        difference = difference[keep]
        quotient = (
            dx
            * ((reference(left[keep]) - reference(right[keep])) * difference).sum(
                dim=-1
            )
        ) / (dx * difference.square().sum(dim=-1))
        empirical_one_sided = max(
            empirical_one_sided, float(quotient.max().item())
        )
    discrete_gradient_sup = float((divergence @ snapshots.T).abs().max().item())

    def chi(rate: float, horizon: float) -> float:
        return (math.expm1(rate * horizon)) / rate if rate > 0 else horizon

    bounds = {}
    for horizon in sorted(args.horizons):
        certified = generator_matching * chi(reference_one_sided, horizon)
        observed = trajectory_errors[str(horizon)]["mesh_norm_max"]
        # Attribute the looseness. The zero-rate variant isolates what the
        # generator-matching term alone costs; the sampled-rate variant shows
        # what a sharper one-sided estimate would buy. Only the first row is
        # certified.
        without_amplification = generator_matching * horizon
        with_sampled_rate = (
            generator_matching * chi(empirical_one_sided, horizon)
            if empirical_one_sided > 0
            else None
        )
        bounds[str(horizon)] = {
            "certified_mesh_norm_bound": certified,
            "observed_mesh_norm_max": observed,
            "looseness_factor": certified / observed if observed > 0 else None,
            "amplification_factor": math.exp(reference_one_sided * horizon),
            "decomposition": {
                "bound_with_zero_rate": without_amplification,
                "bound_with_sampled_rate_not_certified": with_sampled_rate,
                "cost_of_amplification": (
                    certified / without_amplification
                    if without_amplification > 0
                    else None
                ),
                "cost_of_generator_matching": (
                    without_amplification / observed if observed > 0 else None
                ),
            },
        }

    # Self-checks on the operator algebra the one-sided bound relies on.
    skew_residual = float(
        (divergence + divergence.T).abs().max().item()
    )
    symmetry_residual = float((laplacian - laplacian.T).abs().max().item())
    laplacian_max_eigenvalue = float(
        torch.linalg.eigvalsh(laplacian).max().item()
    )

    result = {
        "experiment": "burgers_certified_generator_constants",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "note": "docs/research/BURGERS_CERTIFIED_CONSTANTS.md",
        "architecture": "r0_h8_l1_anchor",
        "seed": int(args.seed),
        "inputs": {
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "data_cache_sha256": digest,
        },
        "grid": {"N": n_grid, "L": length, "dx": dx, "nu": viscosity},
        "operator_norms": norms,
        "operator_self_checks": {
            "divergence_skew_symmetry_residual": skew_residual,
            "laplacian_symmetry_residual": symmetry_residual,
            "laplacian_max_eigenvalue": laplacian_max_eigenvalue,
        },
        "scalar_flux_reduction": {
            "zeroed_control_column_norm": control_column_norm,
            "reduction_is_exact_because_control_is_zeroed": True,
        },
        "state_range": {
            "observed_max_abs": observed_max,
            "margin": args.range_margin,
            "certified_on_interval": [-state_bound, state_bound],
        },
        "certified_constants": {
            "flux_lipschitz_analytic": analytic_lipschitz,
            "flux_lipschitz_covering": covering_lipschitz,
            "flux_lipschitz_used": flux_lipschitz,
            "flux_second_derivative_bound": flux.second_derivative_bound(),
            "flux_error_gauge_invariant_sup": flux_error,
            "flux_error_detail": flux_error_report,
            "covering_samples": int(args.covering_samples),
            "learned_generator_lipschitz": learned_lipschitz,
            "learned_generator_one_sided": learned_one_sided,
            "reference_generator_one_sided": reference_one_sided,
            "generator_matching_mesh_norm": generator_matching,
        },
        "sampled_diagnostics_not_certified": {
            "empirical_reference_one_sided_quotient": empirical_one_sided,
            "certified_over_empirical_one_sided": (
                reference_one_sided / empirical_one_sided
                if empirical_one_sided > 0
                else None
            ),
            "discrete_gradient_sup_on_snapshots": discrete_gradient_sup,
        },
        "exact_flow_trajectory_errors": trajectory_errors,
        "layered_bound": bounds,
        "not_certified_here": [
            "time-integration error of the production 12-substep RK4 map",
            "finite-difference versus spectral spatial consistency",
            "invariance of the state interval, which is observed rather than proved",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "certified_constants": result["certified_constants"],
        "state_range": result["state_range"],
        "layered_bound": result["layered_bound"],
    }, indent=2))


if __name__ == "__main__":
    main()
