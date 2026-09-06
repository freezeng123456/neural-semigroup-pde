#!/usr/bin/env python3
"""Conservative-mobility Cahn--Hilliard generator.

Implements ``docs/research/CAHN_HILLIARD_CONSERVATIVE_MOBILITY_PROTOCOL.md``.
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

import torch
from torch import nn
from torch.nn import functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate import evaluate_full  # noqa: E402
from experiment_artifacts import torch_load_compat  # noqa: E402
from models import ScalarMLP, StencilMLP  # noqa: E402
from run_cahn_hilliard_physics_split import (  # noqa: E402
    CahnHilliardSolver,
    HORIZONS,
    MASSES,
    cache_prefixes,
    free_energy,
    generate_initial_conditions,
    mass_drift,
)
from run_fisher_fair import rollout_steps_for_horizon  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402
from training import train_model  # noqa: E402

SEEDS = (42, 137, 2718)
TRAIN_TAUS = (0.025, 0.05, 0.10, 0.20)
TEST_TAUS = (0.075, 0.15)
SELECTION_TAU = 0.10
HIDDEN = [64, 64]
STENCIL_RADIUS = 3
ODE_STEPS = 30
MASS_DRIFT_LIMIT = 1e-6
DEFAULT_LOCKED = Path(
    "/jizhicfs/yuyechen/neural_semigroup/"
    "20260907-cahn-hilliard-physics-split-h20-r1/locked_test.pt"
)


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


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


class ConservativeCahnHilliardFlow(nn.Module):
    """Physical-space autonomous CH generator with mobility -D^T M D."""

    def __init__(self, *, n_grid=64, length=2.0 * math.pi, epsilon=0.1, ode_steps=ODE_STEPS):
        super().__init__()
        self.N = int(n_grid)
        self.length = float(length)
        self.epsilon = float(epsilon)
        self.ode_steps = int(ode_steps)
        self.dx = self.length / self.N
        self.residual = ScalarMLP(HIDDEN, beta=0.0, beta_floor=0.0)
        self.mobility_net = StencilMLP(radius=STENCIL_RADIUS, hidden_dims=HIDDEN)
        spacing = self.length / self.N
        wavenumbers = 2.0 * math.pi * torch.fft.rfftfreq(self.N, d=spacing)
        self.register_buffer("laplacian_symbol", -(wavenumbers ** 2))

    def laplacian(self, state: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfft(state, dim=-1)
        return torch.fft.irfft(self.laplacian_symbol * transformed, n=self.N, dim=-1)

    def chemical_potential(self, state: torch.Tensor) -> torch.Tensor:
        residual = self.residual(state.unsqueeze(-1)).squeeze(-1)
        return -(self.epsilon ** 2) * self.laplacian(state) + state.pow(3) - state + residual

    def gradient(self, field: torch.Tensor) -> torch.Tensor:
        return (torch.roll(field, shifts=-1, dims=-1) - field) / self.dx

    def divergence(self, flux: torch.Tensor) -> torch.Tensor:
        """Discrete `-D^T`, the adjoint divergence for the periodic gradient."""
        return (flux - torch.roll(flux, shifts=1, dims=-1)) / self.dx

    def mobility(self, state: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.mobility_net(state)) + 5e-3

    def vector_field(self, state: torch.Tensor) -> torch.Tensor:
        mobility = self.mobility(state)
        return self.divergence(mobility * self.gradient(self.chemical_potential(state)))

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        current = state
        step = tau_column(tau, current) / self.ode_steps
        for _ in range(self.ode_steps):
            k1 = self.vector_field(current)
            k2 = self.vector_field(current + 0.5 * step * k1)
            k3 = self.vector_field(current + 0.5 * step * k2)
            k4 = self.vector_field(current + step * k3)
            current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return current


def generate_training(args, seed: int) -> dict:
    initials, _masses = generate_initial_conditions(
        args.N, args.L, MASSES, args.n_train_per_mass, seed
    )
    taus = torch.tensor(
        [TRAIN_TAUS[index % len(TRAIN_TAUS)] for index in range(initials.shape[0])],
        dtype=torch.float32,
    )
    solver = CahnHilliardSolver(
        n_grid=args.N,
        length=args.L,
        epsilon=args.epsilon,
        dt=args.reference_dt,
        device=args.device,
    )
    targets = torch.empty_like(initials)
    for tau in TRAIN_TAUS:
        selected = (taus - float(tau)).abs() < 1e-12
        if not bool(selected.any()):
            continue
        _times, trajectory = solver.solve_batch(initials[selected], float(tau))
        targets[selected] = trajectory[:, -1]
    val_initials, _val_masses = generate_initial_conditions(
        args.N, args.L, MASSES, args.n_val_per_mass, seed + 17
    )
    val_times, val_trajectory = solver.solve_batch(val_initials, 1.2)
    val_trajs = [
        (val_times, val_trajectory[index]) for index in range(val_initials.shape[0])
    ]
    return {
        "train_u0": initials,
        "train_ut": targets,
        "train_tau": taus,
        "val_u0": val_initials,
        "val_trajs": val_trajs,
    }


def evaluate_seed(model, locked: dict, args) -> dict:
    device = torch.device(args.device)
    model = model.to(device).eval()
    energy = lambda state: free_energy(state, epsilon=args.epsilon, length=args.L)
    cells = {}
    with torch.no_grad():
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
                    model_name="a_conservative",
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
    mses = [row["rollout_mse_mean"] for row in cells.values()]
    drifts = [row["mass_drift_max"] for row in cells.values()]
    return {
        "cells": cells,
        "geometric_mean_mse": geometric_mean(mses),
        "max_mass_drift": max(drifts),
        "conserves_mass": all(value <= MASS_DRIFT_LIMIT for value in drifts),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--locked-cache", type=Path, default=DEFAULT_LOCKED)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * math.pi)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=5e-4)
    parser.add_argument("--n-train-per-mass", type=int, default=64)
    parser.add_argument("--n-val-per-mass", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args(argv)


def run_one_seed(seed: int, locked: dict, args) -> dict:
    set_global_seed(seed, deterministic=True)
    model = ConservativeCahnHilliardFlow(
        n_grid=args.N, length=args.L, epsilon=args.epsilon, ode_steps=ODE_STEPS
    )
    data = generate_training(args, seed + 1000)
    checkpoint_dir = Path(args.output_dir) / "checkpoints" / f"s{seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_model(
        model,
        data["train_u0"],
        data["train_ut"],
        data["val_u0"],
        data["val_trajs"],
        train_tau=data["train_tau"],
        tau=SELECTION_TAU,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_rollout=0.0,
        alpha_energy=0.0,
        alpha_bound=0.0,
        weight_decay=1e-5,
        checkpoint_dir=str(checkpoint_dir),
        model_name="a_conservative",
        device=args.device,
        lower_bound=-1.0,
        upper_bound=1.0,
        reference_dt=args.reference_dt,
        validation_interval=args.validation_interval,
        resume_from=None,
    )
    training_seconds = time.perf_counter() - started
    best_path = checkpoint_dir / "a_conservative_best.pt"
    checkpoint = torch_load_compat(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = evaluate_seed(model, locked, args)
    return {
        "seed": int(seed),
        "parameter_count": parameter_count(model),
        "training_seconds": float(training_seconds),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        **evaluation,
    }


def main(argv=None):
    args = parse_args(argv)
    locked_path = Path(args.locked_cache).resolve()
    if not locked_path.is_file():
        raise FileNotFoundError(f"locked Cahn--Hilliard cache does not exist: {locked_path}")
    locked = torch.load(locked_path, map_location="cpu", weights_only=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for seed in args.seeds:
        row = run_one_seed(int(seed), locked, args)
        seed_path = output_dir / f"s{seed}.json"
        seed_path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        seed_rows.append(row)
        print(json.dumps({"seed": seed, "mse": row["geometric_mean_mse"], "mass": row["max_mass_drift"]}, indent=2))
    pooled_mse = geometric_mean(row["geometric_mean_mse"] for row in seed_rows)
    mass_ok = all(row["conserves_mass"] for row in seed_rows)
    payload = {
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/CAHN_HILLIARD_CONSERVATIVE_MOBILITY_PROTOCOL.md",
        "locked_cache": str(locked_path),
        "locked_cache_sha256": sha256_file(locked_path),
        "mass_drift_limit": MASS_DRIFT_LIMIT,
        "ode_steps": ODE_STEPS,
        "conservative_mobility_conserves_mass": mass_ok,
        "geometric_mean_mse": pooled_mse,
        "seeds": seed_rows,
    }
    decision_path = output_dir / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(decision_path),
                "geometric_mean_mse": pooled_mse,
                "conserves_mass": mass_ok,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
