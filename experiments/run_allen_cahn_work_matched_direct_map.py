#!/usr/bin/env python3
"""Allen--Cahn work-matched A/C lane.

Implements ``docs/research/ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_PROTOCOL.md``.
The protocol was committed before this runner produced a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch import nn

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import torch_load_compat  # noqa: E402
from models import (  # noqa: E402
    PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded,
    QueryTimeConditionedPhysicsAnchoredPeriodicLatentFlowBounded,
    _broadcast_positive_tau,
)
from pde_solver import (  # noqa: E402
    AllenCahnSolver,
    generate_allen_cahn_initial_conditions,
)
from run_allen_cahn_fair import (  # noqa: E402
    LOWER_BOUND,
    UPPER_BOUND,
    generate_data,
    reference_steps_for_duration,
    rollout_steps_for_horizon,
    _solve_batch_final,
)
from seed_utils import set_global_seed  # noqa: E402
from training import train_model  # noqa: E402

SEEDS = (42, 137, 2718)
TRAIN_TAUS = (0.025, 0.05, 0.10, 0.20)
TEST_TAUS = (0.075, 0.15)
HORIZONS = (1.2, 2.4, 4.8)
SELECTION_TAU = 0.1
HIDDEN = [64, 64]
STENCIL_RADIUS = 3
INTERACTION_RADIUS = 2
QUERY_EXTRA_PARAMETERS = 64
REFINEMENT_SUBSTEPS = (1, 2, 4, 8, 16)
TEST_SEED_OFFSET = 1_000_003
N_TEST = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tau_column(tau, state: torch.Tensor) -> torch.Tensor:
    return _broadcast_positive_tau(tau, state)[:, 0, :1]


def architecture_kwargs(n_grid: int) -> dict:
    return {
        "N": int(n_grid),
        "m": LOWER_BOUND,
        "M": UPPER_BOUND,
        "hidden_V": list(HIDDEN),
        "hidden_K": list(HIDDEN),
        "stencil_radius": STENCIL_RADIUS,
        "interaction_radius": INTERACTION_RADIUS,
        "beta_V": 0.0,
        "beta_V_floor": 0.0,
    }


class EulerAutonomousFlow(
    PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded
):
    """Autonomous physics-anchored field integrated by explicit Euler."""

    def __init__(self, *args, ode_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_steps = int(ode_steps)
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be a positive integer")

    def vector_field(self, latent: torch.Tensor) -> torch.Tensor:
        return self.latent_dynamics(latent, self._interaction_matrix())

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        latent = self.encode(state)
        step = tau_column(tau, latent) / self.ode_steps
        for _ in range(self.ode_steps):
            latent = torch.clamp(latent + step * self.vector_field(latent), -20.0, 20.0)
        return self.decode(latent)


class EulerQueryTimeFlow(QueryTimeConditionedPhysicsAnchoredPeriodicLatentFlowBounded):
    """Query-conditioned sibling integrated by explicit Euler."""

    def __init__(self, *args, ode_steps: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_steps = int(ode_steps)
        if self.ode_steps < 1:
            raise ValueError("ode_steps must be a positive integer")

    def vector_field(self, latent: torch.Tensor, tau) -> torch.Tensor:
        return self.latent_dynamics(
            latent, self._interaction_matrix(), tau=tau
        )

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        latent = self.encode(state)
        step = tau_column(tau, latent) / self.ode_steps
        for _ in range(self.ode_steps):
            latent = torch.clamp(
                latent + step * self.vector_field(latent, tau), -20.0, 20.0
            )
        return self.decode(latent)


class AllenCahnDirectHeatMap(QueryTimeConditionedPhysicsAnchoredPeriodicLatentFlowBounded):
    """Exact periodic heat semigroup plus one query-conditioned increment."""

    def __init__(self, *args, epsilon: float = 0.1, length: float = 2.0 * math.pi, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_steps = 1
        self.epsilon = float(epsilon)
        self.length = float(length)
        spacing = self.length / self.N
        wavenumbers = 2.0 * math.pi * torch.fft.fftfreq(self.N, d=spacing)
        self.register_buffer("laplacian_eigenvalues", -(wavenumbers ** 2))

    def heat_semigroup(self, state: torch.Tensor, tau) -> torch.Tensor:
        duration = tau_column(tau, state)
        decay = torch.exp(
            duration * (self.epsilon ** 2) * self.laplacian_eigenvalues
        )
        transformed = torch.fft.fft(state.to(dtype=torch.float64), dim=-1)
        evolved = torch.fft.ifft(transformed * decay.to(dtype=transformed.dtype), dim=-1)
        return evolved.real.to(dtype=state.dtype)

    def vector_field(self, latent: torch.Tensor, tau) -> torch.Tensor:
        return self.latent_dynamics(
            latent, self._interaction_matrix(), tau=tau
        )

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        linear = self.heat_semigroup(state, tau)
        latent = self.encode(linear)
        step = tau_column(tau, latent)
        latent = torch.clamp(latent + step * self.vector_field(latent, tau), -20.0, 20.0)
        return self.decode(latent)


def copy_autonomous_into_query(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    transferred = {}
    for name, tensor in target_state.items():
        if name in source_state and source_state[name].shape == tensor.shape:
            transferred[name] = source_state[name].clone()
        elif (
            name == "K_net.net.0.weight"
            and "K_net.net.0.weight" in source_state
            and source_state[name].shape[0] == tensor.shape[0]
            and source_state[name].shape[1] + 1 == tensor.shape[1]
        ):
            padded = tensor.clone()
            padded[:, :-1] = source_state[name]
            padded[:, -1] = 0
            transferred[name] = padded
        else:
            transferred[name] = tensor
    target.load_state_dict(transferred)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def compose(model: nn.Module, state: torch.Tensor, tau: float, depth: int) -> torch.Tensor:
    evolved = state
    for _ in range(int(depth)):
        evolved = model(evolved, tau)
    return evolved


def physical_energy(state: torch.Tensor, *, epsilon: float, length: float) -> torch.Tensor:
    spacing = length / state.shape[-1]
    gradient = (torch.roll(state, shifts=-1, dims=-1) - torch.roll(state, shifts=1, dims=-1)) / (
        2.0 * spacing
    )
    density = 0.5 * (epsilon ** 2) * gradient.square() + 0.25 * (1.0 - state.square()).square()
    return density.sum(dim=-1) * spacing


def rms_defect(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left - right).square().mean(dim=1)).mean().item())


def fair_data_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        regime="variable",
        data_seed=int(args.data_seed),
        deterministic=True,
        N=int(args.N),
        L=float(args.L),
        epsilon=float(args.epsilon),
        reference_dt=float(args.reference_dt),
        fixed_tau=SELECTION_TAU,
        train_taus=TRAIN_TAUS,
        eval_horizon=HORIZONS[0],
        n_train=int(args.n_train),
        n_val=int(args.n_val),
        trajectory_horizon=0.0,
    )


def load_or_build_cache(args, cache_path: Path) -> dict:
    if cache_path.exists():
        payload = torch_load_compat(cache_path, map_location="cpu")
        return payload
    data = generate_data(fair_data_args(args))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache_path)
    return data


def build_models(args, init_seed: int):
    set_global_seed(init_seed, deterministic=True)
    kwargs = architecture_kwargs(args.N)
    unmatched = PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded(
        **kwargs
    )
    unmatched.ode_steps = 30
    autonomous = EulerAutonomousFlow(**kwargs, ode_steps=1)
    query = EulerQueryTimeFlow(**kwargs, ode_steps=1)
    direct = AllenCahnDirectHeatMap(
        **kwargs, epsilon=args.epsilon, length=args.L
    )
    copy_autonomous_into_query(autonomous, query)
    copy_autonomous_into_query(autonomous, direct)
    return {
        "a_rk4_autonomous": unmatched,
        "a_euler_autonomous": autonomous,
        "b_euler_query_time": query,
        "c_direct_heat_map": direct,
    }


def train_one(model, label: str, data: dict, args) -> dict:
    checkpoint_dir = Path(args.output_dir) / "checkpoints" / label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed, deterministic=True)
    started = time.perf_counter()
    history = train_model(
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
        alpha_V=0.0,
        weight_decay=1e-5,
        checkpoint_dir=str(checkpoint_dir),
        model_name=label,
        device=args.device,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        reference_dt=args.reference_dt,
        validation_interval=args.validation_interval,
        resume_from=None,
    )
    elapsed = time.perf_counter() - started
    best_path = checkpoint_dir / f"{label}_best.pt"
    checkpoint = torch_load_compat(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    return {
        "label": label,
        "parameter_count": parameter_count(model),
        "training_seconds": float(elapsed),
        "checkpoint_epoch": int(checkpoint.get("epoch", args.epochs)),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "history_keys": sorted(history.keys()) if isinstance(history, dict) else [],
    }


def generate_locked_test(args) -> dict:
    set_global_seed(int(args.data_seed) + TEST_SEED_OFFSET, deterministic=True)
    solver = AllenCahnSolver(
        N=args.N, L=args.L, eps=args.epsilon, dt=args.reference_dt
    )
    initial = generate_allen_cahn_initial_conditions(args.N, args.n_test, args.L)
    targets = {}
    for horizon in HORIZONS:
        steps = reference_steps_for_duration(horizon, args.reference_dt)
        targets[str(horizon)] = _solve_batch_final(solver, initial, steps)
    return {"u0": initial, "targets": targets}


@torch.no_grad()
def evaluate_rollouts(model, locked: dict, args) -> dict:
    model.eval()
    device = args.device
    initial = locked["u0"].to(device)
    rows = {}
    for tau in TEST_TAUS:
        for horizon in HORIZONS:
            depth = rollout_steps_for_horizon(horizon, tau)
            prediction = compose(model, initial, tau, depth)
            target = locked["targets"][str(horizon)].to(device)
            error = (prediction - target).square().mean(dim=1)
            energy_pred = physical_energy(
                prediction, epsilon=args.epsilon, length=args.L
            )
            energy_init = physical_energy(
                initial, epsilon=args.epsilon, length=args.L
            )
            rows[f"tau={tau},T={horizon}"] = {
                "rollout_mse_mean": float(error.mean().item()),
                "relative_l2_mean": float(
                    (torch.linalg.vector_norm(prediction - target, dim=1)
                     / torch.linalg.vector_norm(target, dim=1).clamp_min(1e-12)
                     ).mean().item()
                ),
                "bound_violation_mean": float(
                    (torch.relu(prediction - UPPER_BOUND) + torch.relu(LOWER_BOUND - prediction))
                    .mean()
                    .item()
                ),
                "energy_monotone_fraction": float(
                    (energy_pred <= energy_init + 1e-8).float().mean().item()
                ),
                "state_rms_mean": float(torch.sqrt(prediction.square().mean(dim=1)).mean().item()),
                "depth": depth,
            }
    return rows


@torch.no_grad()
def cross_lag_defect(model, sample: torch.Tensor, tau_fine: float, tau_coarse: float) -> dict:
    model.eval()
    horizon = tau_coarse
    if not math.isclose(2.0 * tau_fine, tau_coarse, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("in-range pair must satisfy 2 * tau_fine = tau_coarse")
    direct = model(sample, tau_coarse)
    composed = model(model(sample, tau_fine), tau_fine)
    defect = rms_defect(direct, composed)
    state_rms = float(torch.sqrt(sample.square().mean(dim=1)).mean().item())
    return {
        "horizon": horizon,
        "tau_fine": tau_fine,
        "tau_coarse": tau_coarse,
        "rms_defect_mean": defect,
        "relative_to_state_rms": defect / max(state_rms, 1e-12),
        "state_rms_mean": state_rms,
    }


@torch.no_grad()
def equal_substep_autonomous_defect(model: EulerAutonomousFlow, sample: torch.Tensor) -> dict:
    original = model.ode_steps
    try:
        model.ode_steps = 2
        fine = model(sample, 0.15)
        model.ode_steps = 1
        left = model(sample, 0.075)
        right = model(left, 0.075)
        defect = rms_defect(fine, right)
    finally:
        model.ode_steps = original
    return {"rms_defect_mean": defect, "fine_substeps": 2, "coarse_substeps_per_call": 1}


@torch.no_grad()
def refinement_sweep(model, sample: torch.Tensor) -> list[dict]:
    original = model.ode_steps
    series = []
    try:
        for substeps in REFINEMENT_SUBSTEPS:
            model.ode_steps = int(substeps)
            row = cross_lag_defect(model, sample, 0.075, 0.15)
            row["substeps_per_call"] = int(substeps)
            series.append(row)
    finally:
        model.ode_steps = original
    return series


def geometric_mean(values) -> float:
    logs = [math.log(max(float(value), 1e-30)) for value in values]
    return math.exp(sum(logs) / len(logs))


def ratio_table(rollouts: dict, numerator: str, denominator: str) -> dict:
    cells = {}
    for key in rollouts[denominator]:
        cells[key] = (
            rollouts[numerator][key]["rollout_mse_mean"]
            / rollouts[denominator][key]["rollout_mse_mean"]
        )
    return {
        "cells": cells,
        "geometric_mean": geometric_mean(cells.values()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * math.pi)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "a_rk4_autonomous",
            "a_euler_autonomous",
            "b_euler_query_time",
            "c_direct_heat_map",
        ],
    )
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.data_cache)
    data = load_or_build_cache(args, cache_path)
    models = build_models(args, args.seed)
    trained = {}
    for label in args.models:
        trained[label] = train_one(models[label], label, data, args)

    locked = generate_locked_test(args)
    torch.save(locked, output / "locked_test.pt")
    rollouts = {
        label: evaluate_rollouts(models[label], locked, args) for label in args.models
    }
    sample = locked["u0"][:64].to(args.device)
    defects = {
        label: cross_lag_defect(models[label], sample, 0.075, 0.15)
        for label in args.models
    }
    refinements = {}
    equal_substep = None
    if "a_euler_autonomous" in models:
        refinements["a_euler_autonomous"] = refinement_sweep(
            models["a_euler_autonomous"], sample
        )
        equal_substep = equal_substep_autonomous_defect(
            models["a_euler_autonomous"], sample
        )
    if "b_euler_query_time" in models:
        refinements["b_euler_query_time"] = refinement_sweep(
            models["b_euler_query_time"], sample
        )

    summary = {
        "experiment": "allen_cahn_work_matched_direct_map",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "cache_sha256": sha256_file(cache_path),
        "protocol": "docs/research/ALLEN_CAHN_WORK_MATCHED_DIRECT_MAP_PROTOCOL.md",
        "models": trained,
        "rollouts": rollouts,
        "deployed_budget_cross_lag": defects,
        "refinement": refinements,
        "equal_substep_autonomous": equal_substep,
        "ratios": {
            "mse_a_prime_over_c": (
                ratio_table(rollouts, "a_euler_autonomous", "c_direct_heat_map")
                if "a_euler_autonomous" in rollouts and "c_direct_heat_map" in rollouts
                else None
            ),
            "mse_b_prime_over_c": (
                ratio_table(rollouts, "b_euler_query_time", "c_direct_heat_map")
                if "b_euler_query_time" in rollouts and "c_direct_heat_map" in rollouts
                else None
            ),
            "mse_a_rk4_over_c": (
                ratio_table(rollouts, "a_rk4_autonomous", "c_direct_heat_map")
                if "a_rk4_autonomous" in rollouts and "c_direct_heat_map" in rollouts
                else None
            ),
        },
        "learned_evaluations_per_call": {
            "a_rk4_autonomous": 120,
            "a_euler_autonomous": 1,
            "b_euler_query_time": 1,
            "c_direct_heat_map": 1,
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
