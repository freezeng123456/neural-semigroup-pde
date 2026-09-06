#!/usr/bin/env python3
"""Zero-parameter Cahn--Hilliard conservative-split evaluation.

Implements ``docs/research/CAHN_HILLIARD_PHYSICS_SPLIT_PROTOCOL.md``.
The protocol commit predates every number this runner writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate import evaluate_full  # noqa: E402
from run_fisher_fair import rollout_steps_for_horizon  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402

TEST_TAUS = (0.075, 0.15)
HORIZONS = (0.6, 1.2, 2.4)
MASSES = (-0.3, 0.0, 0.3)
N_PER_MASS = 96
DATA_SEED = 20260907
MASS_DRIFT_LIMIT = 1e-8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_mean(values) -> float:
    logs = [math.log(max(float(value), 1e-30)) for value in values]
    return math.exp(sum(logs) / len(logs))


def tau_column(tau, state: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(tau):
        duration = tau.to(device=state.device, dtype=state.dtype)
    else:
        duration = torch.as_tensor(float(tau), device=state.device, dtype=state.dtype)
    while duration.ndim < state.ndim:
        duration = duration.unsqueeze(-1)
    return duration


class CahnHilliardSolver:
    """Batched spectral ETD-RK4 for periodic Cahn--Hilliard."""

    def __init__(self, *, n_grid=64, length=2.0 * math.pi, epsilon=0.1, dt=5e-4, device="cpu"):
        self.N = int(n_grid)
        self.length = float(length)
        self.epsilon = float(epsilon)
        self.dt = float(dt)
        self.device = torch.device(device)
        spacing = self.length / self.N
        wavenumbers = 2.0 * math.pi * torch.fft.rfftfreq(self.N, d=spacing, device=self.device)
        laplacian = -(wavenumbers ** 2)
        biharmonic = wavenumbers ** 4
        linear = -(self.epsilon ** 2) * biharmonic
        self.laplacian_symbol = laplacian.to(dtype=torch.float64)
        self.linear_symbol = linear.to(dtype=torch.float64)
        self.integrating_factor = torch.exp(self.linear_symbol * self.dt)
        self.half_integrating_factor = torch.exp(self.linear_symbol * self.dt / 2.0)

    def nonlinear_hat(self, state: torch.Tensor) -> torch.Tensor:
        chemical = state.to(dtype=torch.float64).pow(3) - state.to(dtype=torch.float64)
        return self.laplacian_symbol * torch.fft.rfft(chemical, dim=-1)

    def step(self, state_hat: torch.Tensor) -> torch.Tensor:
        dt = self.dt
        factor = self.integrating_factor
        half = self.half_integrating_factor
        state = torch.fft.irfft(state_hat, n=self.N, dim=-1)
        n1 = self.nonlinear_hat(state)
        stage_a = torch.fft.irfft(half * state_hat + (dt / 2.0) * half * n1, n=self.N, dim=-1)
        n2 = self.nonlinear_hat(stage_a)
        stage_b = torch.fft.irfft(half * state_hat + (dt / 2.0) * n2, n=self.N, dim=-1)
        n3 = self.nonlinear_hat(stage_b)
        stage_c_hat = half * (half * state_hat + (dt / 2.0) * half * n1) + (dt / 2.0) * (2.0 * n3 - n1)
        stage_c = torch.fft.irfft(stage_c_hat, n=self.N, dim=-1)
        n4 = self.nonlinear_hat(stage_c)
        return factor * state_hat + (dt / 6.0) * (factor * n1 + 2.0 * half * (n2 + n3) + n4)

    def solve_batch(self, initial: torch.Tensor, horizon: float) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = float(horizon) / self.dt
        steps = int(round(ratio))
        if steps <= 0 or not math.isclose(ratio, steps, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"horizon {horizon} is not an integer multiple of dt={self.dt}")
        state = initial.to(device=self.device, dtype=torch.float64)
        trajectory = torch.empty(state.shape[0], steps + 1, self.N, dtype=torch.float32, device=self.device)
        trajectory[:, 0] = state.to(dtype=torch.float32)
        state_hat = torch.fft.rfft(state, dim=-1)
        times = torch.arange(steps + 1, device=self.device, dtype=torch.float32) * self.dt
        for step in range(1, steps + 1):
            state_hat = self.step(state_hat)
            trajectory[:, step] = torch.fft.irfft(state_hat, n=self.N, dim=-1).to(dtype=torch.float32)
        return times.cpu(), trajectory.cpu()


class CahnHilliardPhysicsSplit(nn.Module):
    """Exact biharmonic semigroup plus one conservative nonlinear increment."""

    def __init__(self, *, n_grid=64, length=2.0 * math.pi, epsilon=0.1):
        super().__init__()
        self.N = int(n_grid)
        self.length = float(length)
        self.epsilon = float(epsilon)
        spacing = self.length / self.N
        wavenumbers = 2.0 * math.pi * torch.fft.rfftfreq(self.N, d=spacing)
        self.register_buffer("linear_symbol", -(self.epsilon ** 2) * (wavenumbers ** 4))
        self.register_buffer("laplacian_symbol", -(wavenumbers ** 2))

    def linear_semigroup(self, state: torch.Tensor, tau) -> torch.Tensor:
        duration = tau_column(tau, state)
        decay = torch.exp(duration * self.linear_symbol)
        transformed = torch.fft.rfft(state.to(dtype=torch.float64), dim=-1)
        evolved = torch.fft.irfft(transformed * decay.to(dtype=transformed.dtype), n=self.N, dim=-1)
        return evolved.to(dtype=state.dtype)

    def increment(self, state: torch.Tensor) -> torch.Tensor:
        chemical = state.to(dtype=torch.float64).pow(3) - state.to(dtype=torch.float64)
        return torch.fft.irfft(
            self.laplacian_symbol.to(dtype=torch.float64) * torch.fft.rfft(chemical, dim=-1),
            n=self.N,
            dim=-1,
        ).to(dtype=state.dtype)

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        linear = self.linear_semigroup(state, tau)
        step = tau_column(tau, linear)
        return linear + step * self.increment(linear)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def generate_initial_conditions(n_grid: int, length: float, masses, n_per_mass: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    set_global_seed(seed, deterministic=True)
    grid = torch.linspace(0.0, length, n_grid + 1)[:n_grid]
    states = []
    labels = []
    for mass in masses:
        for _ in range(int(n_per_mass)):
            n_modes = np.random.randint(2, 6)
            field = torch.zeros(n_grid)
            for _mode in range(n_modes):
                wavenumber = np.random.randint(1, 4)
                phase = np.random.uniform(0.0, 2.0 * math.pi)
                amplitude = np.random.uniform(0.1, 0.5)
                field = field + amplitude * torch.sin(2.0 * math.pi * wavenumber * grid / length + phase)
            field = field / (field.abs().max() + 1e-8) * 0.6
            field = field - field.mean() + float(mass)
            states.append(field)
            labels.append(float(mass))
    return torch.stack(states), torch.tensor(labels, dtype=torch.float32)


def free_energy(state: torch.Tensor, *, epsilon: float, length: float) -> torch.Tensor:
    spacing = float(length) / state.shape[-1]
    gradient = (torch.roll(state, shifts=-1, dims=-1) - state) / spacing
    potential = 0.25 * (state.square() - 1.0).square()
    density = 0.5 * (epsilon ** 2) * gradient.square() + potential
    return spacing * density.sum(dim=-1)


def generate_locked_test(args) -> dict:
    initials, masses = generate_initial_conditions(
        args.N, args.L, MASSES, args.n_per_mass, args.data_seed
    )
    solver = CahnHilliardSolver(
        n_grid=args.N,
        length=args.L,
        epsilon=args.epsilon,
        dt=args.reference_dt,
        device=args.device,
    )
    horizon = max(HORIZONS)
    times, trajectories = solver.solve_batch(initials, horizon)
    return {
        "u0": initials,
        "masses": masses,
        "times": times,
        "trajectories": trajectories,
        "config": {
            "N": int(args.N),
            "L": float(args.L),
            "epsilon": float(args.epsilon),
            "reference_dt": float(args.reference_dt),
            "data_seed": int(args.data_seed),
            "masses": list(MASSES),
            "n_per_mass": int(args.n_per_mass),
            "horizon": horizon,
        },
    }


def cache_prefixes(locked: dict, horizon: float, reference_dt: float):
    steps = int(round(float(horizon) / float(reference_dt)))
    times = locked["times"][: steps + 1]
    return [
        (times, locked["trajectories"][index, : steps + 1])
        for index in range(locked["trajectories"].shape[0])
    ]


def mass_drift(prediction: torch.Tensor, initial: torch.Tensor) -> torch.Tensor:
    return (prediction.mean(dim=-1) - initial.mean(dim=-1)).abs()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * math.pi)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=5e-4)
    parser.add_argument("--n-per-mass", type=int, default=N_PER_MASS)
    parser.add_argument("--data-seed", type=int, default=DATA_SEED)
    return parser.parse_args(argv)


@torch.no_grad()
def evaluate_split(args, locked: dict) -> dict:
    device = torch.device(args.device)
    model = CahnHilliardPhysicsSplit(
        n_grid=args.N, length=args.L, epsilon=args.epsilon
    ).to(device)
    model.eval()
    if parameter_count(model) != 0:
        raise RuntimeError("CahnHilliardPhysicsSplit must have zero trainable parameters")
    energy = lambda state: free_energy(state, epsilon=args.epsilon, length=args.L)
    cells = {}
    for tau in TEST_TAUS:
        for horizon in HORIZONS:
            depth = rollout_steps_for_horizon(horizon, tau)
            prefixes = cache_prefixes(locked, horizon, args.reference_dt)
            metrics, _details = evaluate_full(
                model,
                locked["u0"],
                prefixes,
                tau=tau,
                rollout_steps=depth,
                device=device,
                model_name="c_physics_split",
                lower_bound=-1.0,
                upper_bound=1.0,
                reference_dt=args.reference_dt,
                physical_energy_fn=energy,
            )
            prediction = locked["u0"].to(device)
            for _ in range(depth):
                prediction = model(prediction, tau)
            drift = mass_drift(prediction, locked["u0"].to(device))
            cells[f"tau={tau},T={horizon}"] = {
                "tau": float(tau),
                "horizon": float(horizon),
                "rollout_steps": depth,
                "rollout_mse_mean": float(metrics["rollout_mse_mean"]),
                "rollout_rel_l2_mean": float(metrics["rollout_rel_l2_mean"]),
                "bound_viol_mean": float(metrics["bound_viol_mean"]),
                "physical_energy_mono_frac": float(
                    metrics.get("physical_energy_mono_frac_mean", float("nan"))
                ),
                "semigroup_defect_mean": float(metrics["semigroup_defect_mean"]),
                "mass_drift_mean": float(drift.mean().item()),
                "mass_drift_max": float(drift.max().item()),
            }
    drifts = [row["mass_drift_max"] for row in cells.values()]
    mses = [row["rollout_mse_mean"] for row in cells.values()]
    mass_ok = all(value <= MASS_DRIFT_LIMIT for value in drifts)
    usable = mass_ok and all(math.isfinite(value) for value in mses)
    return {
        "cells": cells,
        "geometric_mean_mse": geometric_mean(mses),
        "max_mass_drift": max(drifts),
        "conservative_split_conserves_mass": mass_ok,
        "conservative_split_is_a_usable_ch_baseline": usable,
        "mass_drift_limit": MASS_DRIFT_LIMIT,
        "parameter_count": 0,
    }


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    if args.cache is not None and Path(args.cache).is_file():
        locked = torch.load(args.cache, map_location="cpu", weights_only=False)
        cache_status = "loaded"
        cache_path = Path(args.cache)
    else:
        locked = generate_locked_test(args)
        cache_path = Path(args.output).with_name("locked_test.pt")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(locked, cache_path)
        cache_status = "created"
    comparison = evaluate_split(args, locked)
    payload = {
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/CAHN_HILLIARD_PHYSICS_SPLIT_PROTOCOL.md",
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": sha256_file(cache_path),
        "cache_status": cache_status,
        "seconds": time.perf_counter() - started,
        "device": str(args.device),
        "config": locked["config"],
        **comparison,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "geometric_mean_mse": payload["geometric_mean_mse"],
                "max_mass_drift": payload["max_mass_drift"],
                "conserves_mass": payload["conservative_split_conserves_mass"],
                "usable_baseline": payload["conservative_split_is_a_usable_ch_baseline"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
