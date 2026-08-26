#!/usr/bin/env python3
"""Run the standardized Phase-4 Allen--Cahn fair comparison.

The runner mirrors :mod:`run_fisher_fair` while keeping the Allen--Cahn
benchmark self-contained.  It supports fixed- and variable-time training,
strictly aligned reference trajectories, architecture-only training by
default, and a ``--prepare-data-only`` mode for shared-cluster workflows.

The three compared maps are parameter-matched at the N=64 protocol setting:

* ``latent``: ``LatentSemigroupNetBounded`` with admissible range [-1, 1];
* ``resnet``: time-conditioned ``TimeConditionedResNet``;
* ``fno``: time-conditioned ``TimeConditionedFNO``.

Typical CPU smoke run::

    python experiments/run_allen_cahn_fair.py \
        --regime fixed --models latent resnet fno --device cpu \
        --output-dir /tmp/ac-smoke --data-cache /tmp/ac-smoke-data.pt \
        --N 16 --reference-dt 0.01 --fixed-tau 0.04 \
        --eval-horizon 0.16 --n-train 32 --n-val 4 --epochs 2 \
        --batch-size 8 --deterministic --no-resume

Prepare a shared cache once, without creating model checkpoints::

    python experiments/run_allen_cahn_fair.py \
        --regime variable --device cpu --prepare-data-only \
        --output-dir /path/to/data-provenance \
        --data-cache /path/to/shared/allen-cahn-data.pt \
        --deterministic --no-resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate import evaluate_full  # noqa: E402
from models import (  # noqa: E402
    DecodedInteractionLatentSemigroupNetBounded,
    DecodedInteractionJacobianMobilityLatentSemigroupNetBounded,
    DecodedStateEnergyLatentSemigroupNetBounded,
    PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded,
    PeriodicStencilDecodedInteractionLatentSemigroupNetBounded,
    LatentSemigroupNetBounded,
    TimeConditionedFNO,
    TimeConditionedResNet,
)
from pde_solver import (  # noqa: E402
    AllenCahnSolver,
    generate_allen_cahn_initial_conditions,
)
from seed_utils import set_global_seed  # noqa: E402
from training import train_model  # noqa: E402


MODEL_NAMES = (
    "latent",
    "latent_decoded_interaction",
    "latent_decoded_interaction_jacobian_mobility",
    "latent_decoded_energy",
    "latent_periodic_decoded_interaction",
    "latent_physics_anchored_periodic",
    "resnet",
    "fno",
)
DEFAULT_MODEL_NAMES = ("latent", "resnet", "fno")
LOWER_BOUND = -1.0
UPPER_BOUND = 1.0
DEFAULT_TRAIN_TAUS = (0.025, 0.05, 0.1, 0.2)
DEFAULT_EVAL_TAUS = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
PARAMETER_TOLERANCES = {"resnet": 550, "fno": 300}


def parse_float_list(value):
    """Parse a comma-separated list of positive finite times."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated float list"
        ) from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "times must be finite and strictly positive"
        )
    return values


def parse_nonnegative_float(value):
    """Parse a finite non-negative loss weight."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a finite non-negative float") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("expected a finite non-negative float")
    return parsed


def build_parser():
    """Build the CLI, keeping the Fisher fair-runner interface aligned."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("fixed", "variable"), required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_NAMES,
        default=list(DEFAULT_MODEL_NAMES),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * math.pi)
    parser.add_argument(
        "--eps", "--epsilon", dest="epsilon", type=float, default=0.1
    )
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--fixed-tau", type=float, default=0.1)
    parser.add_argument(
        "--train-taus",
        type=parse_float_list,
        default=DEFAULT_TRAIN_TAUS,
    )
    parser.add_argument(
        "--eval-taus",
        type=parse_float_list,
        default=DEFAULT_EVAL_TAUS,
    )
    parser.add_argument("--eval-horizon", type=float, default=1.2)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument(
        "--beta-v-floor",
        type=float,
        default=0.1,
        help=(
            "non-negative fixed quadratic floor for the latent potential; "
            "zero is an Allen--Cahn diagnostic setting without the "
            "positive z-coercivity guarantee"
        ),
    )
    parser.add_argument(
        "--alpha-rollout", type=parse_nonnegative_float, default=0.0
    )
    parser.add_argument(
        "--alpha-trajectory",
        type=parse_nonnegative_float,
        default=0.0,
        help=(
            "weight of the reference-supervised repeated-step endpoint loss; "
            "requires fixed-time training and --trajectory-horizon"
        ),
    )
    parser.add_argument(
        "--trajectory-horizon",
        type=float,
        default=0.0,
        help=(
            "physical endpoint used by the reference-supervised repeated-step "
            "loss; must be an exact multiple of fixed-tau and reference-dt"
        ),
    )
    parser.add_argument(
        "--alpha-energy", type=parse_nonnegative_float, default=0.0
    )
    parser.add_argument("--alpha-bound", type=parse_nonnegative_float, default=0.0)
    parser.add_argument("--alpha-v", type=parse_nonnegative_float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help=(
            "generate/load and validate the frozen Allen-Cahn data cache, "
            "write data_provenance.json, then exit without training or checkpoints"
        ),
    )
    return parser


def rollout_steps_for_horizon(horizon, tau):
    """Return exact model steps for a physical horizon and increment."""
    ratio = float(horizon) / float(tau)
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(ratio, steps, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(
            f"eval_horizon={horizon} is not an integer multiple of tau={tau}"
        )
    return steps


def reference_steps_for_duration(duration, reference_dt):
    """Convert a duration to an exact integer reference-solver step count."""
    duration = float(duration)
    reference_dt = float(reference_dt)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and strictly positive")
    if not math.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("reference_dt must be finite and strictly positive")
    ratio = duration / reference_dt
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(ratio, steps, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(
            "duration must be an integer multiple of reference_dt; "
            f"received duration/reference_dt={ratio:.17g}"
        )
    return steps


def data_config(args):
    """Return the complete frozen data-generation/split configuration."""
    config = {
        "pde": "allen-cahn",
        "regime": args.regime,
        "data_seed": args.data_seed,
        "deterministic": bool(args.deterministic),
        "split_policy": "train_then_validation_from_one_seeded_stream",
        "initial_condition_policy": "low_frequency_fourier_modes_scaled_to_0.9",
        "N": args.N,
        "L": args.L,
        "epsilon": args.epsilon,
        "reference_dt": args.reference_dt,
        "fixed_tau": args.fixed_tau,
        "train_taus": list(args.train_taus),
        "eval_horizon": args.eval_horizon,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "bounds": [LOWER_BOUND, UPPER_BOUND],
    }
    # Keep legacy step-only cache identities stable.  A positive trajectory
    # horizon changes the frozen reference targets and therefore belongs to
    # the cache key.
    if args.trajectory_horizon > 0:
        config["trajectory_horizon"] = args.trajectory_horizon
    return config


def _validate_unique_times(values, name):
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate values")


def validate_args(args):
    """Validate protocol invariants before touching the cache or outputs."""
    if args.seed < 0 or args.data_seed < 0:
        raise ValueError("seed and data-seed must be non-negative")
    if args.N <= 1:
        raise ValueError("N must be greater than one")
    if args.n_train <= 0 or args.n_val <= 0 or args.epochs <= 0:
        raise ValueError("sample and epoch counts must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if not math.isfinite(args.L) or args.L <= 0:
        raise ValueError("L must be finite and strictly positive")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        raise ValueError("eps/epsilon must be finite and strictly positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and strictly positive")
    if args.validation_interval <= 0:
        raise ValueError("validation interval must be positive")
    if not math.isfinite(args.beta_v_floor) or args.beta_v_floor < 0:
        raise ValueError("beta-v-floor must be finite and non-negative")
    _validate_unique_times(args.train_taus, "train-taus")
    _validate_unique_times(args.eval_taus, "eval-taus")
    reference_steps_for_duration(args.eval_horizon, args.reference_dt)
    reference_steps_for_duration(args.fixed_tau, args.reference_dt)
    if args.regime == "variable":
        for tau in args.train_taus:
            reference_steps_for_duration(tau, args.reference_dt)
    eval_taus = (args.fixed_tau,) if args.regime == "fixed" else args.eval_taus
    for tau in eval_taus:
        rollout_steps_for_horizon(args.eval_horizon, tau)
        reference_steps_for_duration(tau, args.reference_dt)
    for value_name in (
        "alpha_rollout",
        "alpha_trajectory",
        "alpha_energy",
        "alpha_bound",
        "alpha_v",
    ):
        value = getattr(args, value_name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{value_name} must be finite and non-negative")
    if not math.isfinite(args.trajectory_horizon) or args.trajectory_horizon < 0:
        raise ValueError("trajectory-horizon must be finite and non-negative")
    if args.alpha_trajectory > 0:
        if args.regime != "fixed":
            raise ValueError(
                "alpha-trajectory currently requires --regime fixed"
            )
        trajectory_steps = rollout_steps_for_horizon(
            args.trajectory_horizon, args.fixed_tau
        )
        if trajectory_steps < 2:
            raise ValueError(
                "trajectory-horizon must cover at least two fixed-tau steps"
            )
        reference_steps_for_duration(args.trajectory_horizon, args.reference_dt)
    elif args.trajectory_horizon != 0:
        raise ValueError(
            "trajectory-horizon requires a positive alpha-trajectory"
        )


def _new_solver(args):
    return AllenCahnSolver(
        N=args.N,
        L=args.L,
        eps=args.epsilon,
        dt=args.reference_dt,
    )


def _solve_batch_final(solver, u0, n_steps):
    """Run the Allen--Cahn reference solver on a batch for exactly n_steps."""
    if u0.ndim != 2 or u0.shape[-1] != solver.N:
        raise ValueError(
            f"u0 must have shape (batch, {solver.N}), got {tuple(u0.shape)}"
        )
    u_hat = torch.fft.fft(u0.to(torch.float64), dim=-1)
    for _ in range(int(n_steps)):
        u_hat = solver.rk4_step(u_hat)
    return torch.fft.ifft(u_hat, dim=-1).real.to(dtype=u0.dtype)


def _solve_batch_trajectory(solver, u0, n_steps):
    """Return timestamps and trajectories with an exact reference grid."""
    if u0.ndim != 2 or u0.shape[-1] != solver.N:
        raise ValueError(
            f"u0 must have shape (batch, {solver.N}), got {tuple(u0.shape)}"
        )
    u_hat = torch.fft.fft(u0.to(torch.float64), dim=-1)
    states = [u0.clone()]
    for _ in range(int(n_steps)):
        u_hat = solver.rk4_step(u_hat)
        states.append(torch.fft.ifft(u_hat, dim=-1).real.to(dtype=u0.dtype))
    trajectories = torch.stack(states, dim=1)
    times = torch.arange(
        int(n_steps) + 1, dtype=torch.float64
    ) * float(solver.dt)
    return times, trajectories


def _generate_variable_training_data(args, solver, train_u0):
    tau_values = tuple(float(value) for value in args.train_taus)
    assignment = torch.arange(args.n_train, dtype=torch.long) % len(tau_values)
    assignment = assignment[torch.randperm(args.n_train)]
    train_tau = torch.tensor(tau_values, dtype=train_u0.dtype)[assignment]
    train_ut = torch.empty_like(train_u0)
    for tau_index, tau in enumerate(tau_values):
        mask = assignment == tau_index
        n_steps = reference_steps_for_duration(tau, args.reference_dt)
        train_ut[mask] = _solve_batch_final(solver, train_u0[mask], n_steps)
    return train_tau, train_ut


def generate_data(args):
    """Generate one deterministic, frozen train/validation split."""
    set_global_seed(args.data_seed, deterministic=args.deterministic)
    solver = _new_solver(args)
    train_u0 = generate_allen_cahn_initial_conditions(
        args.N, args.n_train, args.L
    )
    if args.regime == "fixed":
        train_tau = None
        train_ut = _solve_batch_final(
            solver,
            train_u0,
            reference_steps_for_duration(args.fixed_tau, args.reference_dt),
        )
        train_rollout_ut = None
        if args.alpha_trajectory > 0:
            train_rollout_ut = _solve_batch_final(
                solver,
                train_u0,
                reference_steps_for_duration(
                    args.trajectory_horizon, args.reference_dt
                ),
            )
    else:
        train_tau, train_ut = _generate_variable_training_data(
            args, solver, train_u0
        )
        train_rollout_ut = None

    # The validation stream is generated after the training stream, so the
    # data seed fixes both the split and the order of every initial condition.
    val_u0 = generate_allen_cahn_initial_conditions(args.N, args.n_val, args.L)
    val_steps = reference_steps_for_duration(args.eval_horizon, args.reference_dt)
    times, val_trajectories = _solve_batch_trajectory(solver, val_u0, val_steps)
    val_trajs = [
        (times.clone(), val_trajectories[index].clone())
        for index in range(args.n_val)
    ]
    return {
        "train_u0": train_u0,
        "train_tau": train_tau,
        "train_ut": train_ut,
        "train_rollout_ut": train_rollout_ut,
        "val_u0": val_u0,
        "val_trajs": val_trajs,
        "data_generation_config": data_config(args),
    }


def _atomic_torch_save(data, path):
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    try:
        torch.save(data, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_timestamp_grid(times, expected_steps, reference_dt):
    if len(times) != expected_steps + 1:
        raise ValueError(
            f"expected {expected_steps + 1} validation timestamps, got {len(times)}"
        )
    for index, timestamp in enumerate(times):
        expected = index * float(reference_dt)
        actual = float(timestamp)
        if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError(
                "validation timestamps are not strictly aligned to reference_dt: "
                f"index={index}, actual={actual:.17g}, expected={expected:.17g}"
            )


def validate_data_cache(data, args):
    """Validate cache configuration, shapes, finiteness, and time alignment."""
    expected_config = data_config(args)
    if data.get("data_generation_config") != expected_config:
        raise ValueError("data cache configuration does not match this run")
    required = ("train_u0", "train_ut", "val_u0", "val_trajs")
    if args.alpha_trajectory > 0:
        required = (*required, "train_rollout_ut")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"data cache is missing required fields: {missing}")

    train_u0 = data["train_u0"]
    train_ut = data["train_ut"]
    train_rollout_ut = data.get("train_rollout_ut")
    val_u0 = data["val_u0"]
    if tuple(train_u0.shape) != (args.n_train, args.N):
        raise ValueError(f"unexpected train_u0 shape: {tuple(train_u0.shape)}")
    if tuple(train_ut.shape) != (args.n_train, args.N):
        raise ValueError(f"unexpected train_ut shape: {tuple(train_ut.shape)}")
    if args.alpha_trajectory > 0:
        if tuple(train_rollout_ut.shape) != (args.n_train, args.N):
            raise ValueError(
                "unexpected train_rollout_ut shape: "
                f"{tuple(train_rollout_ut.shape)}"
            )
    elif train_rollout_ut is not None:
        raise ValueError(
            "step-only cache must not contain train_rollout_ut"
        )
    if tuple(val_u0.shape) != (args.n_val, args.N):
        raise ValueError(f"unexpected val_u0 shape: {tuple(val_u0.shape)}")
    for name, tensor in (("train_u0", train_u0), ("train_ut", train_ut), ("val_u0", val_u0)):
        if not torch.is_tensor(tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"{name} must be a finite tensor")
    for name, tensor in (("train_u0", train_u0), ("train_ut", train_ut), ("val_u0", val_u0)):
        if float(tensor.min()) < LOWER_BOUND - 1e-4 or float(tensor.max()) > UPPER_BOUND + 1e-4:
            raise ValueError(f"{name} leaves the Allen-Cahn admissible bounds [-1, 1]")
    if args.alpha_trajectory > 0:
        if not torch.is_tensor(train_rollout_ut) or not torch.isfinite(train_rollout_ut).all():
            raise ValueError("train_rollout_ut must be a finite tensor")
        if (
            float(train_rollout_ut.min()) < LOWER_BOUND - 1e-4
            or float(train_rollout_ut.max()) > UPPER_BOUND + 1e-4
        ):
            raise ValueError(
                "train_rollout_ut leaves the Allen-Cahn admissible bounds [-1, 1]"
            )

    train_tau = data.get("train_tau")
    if args.regime == "fixed":
        if train_tau is not None:
            raise ValueError("fixed-time cache must not contain train_tau")
    else:
        if train_tau is None or tuple(train_tau.shape) != (args.n_train,):
            raise ValueError("variable-time cache must contain one train_tau per pair")
        if not torch.isfinite(train_tau).all() or not (train_tau > 0).all():
            raise ValueError("train_tau must contain finite positive values")
        requested_taus = tuple(float(value) for value in args.train_taus)
        counts = []
        for tau in requested_taus:
            matches = torch.isclose(
                train_tau,
                torch.as_tensor(tau, dtype=train_tau.dtype),
                rtol=1e-6,
                atol=1e-8,
            )
            counts.append(int(matches.sum().item()))
            if counts[-1] == 0:
                raise ValueError(f"train_tau cache does not contain requested tau={tau}")
        if sum(counts) != args.n_train or max(counts) - min(counts) > 1:
            raise ValueError("variable-time training increments are not balanced")

    expected_steps = reference_steps_for_duration(args.eval_horizon, args.reference_dt)
    val_trajs = data["val_trajs"]
    if len(val_trajs) != args.n_val:
        raise ValueError(f"expected {args.n_val} validation trajectories, got {len(val_trajs)}")
    for index, trajectory in enumerate(val_trajs):
        if not isinstance(trajectory, (tuple, list)) or len(trajectory) != 2:
            raise ValueError(f"validation trajectory {index} must be a (times, states) pair")
        times, states = trajectory
        if tuple(states.shape) != (expected_steps + 1, args.N):
            raise ValueError(
                f"unexpected validation state shape for trajectory {index}: {tuple(states.shape)}"
            )
        if not torch.isfinite(times).all() or not torch.isfinite(states).all():
            raise ValueError(f"validation trajectory {index} contains non-finite values")
        if float(states.min()) < LOWER_BOUND - 1e-4 or float(states.max()) > UPPER_BOUND + 1e-4:
            raise ValueError(f"validation trajectory {index} leaves [-1, 1]")
        _validate_timestamp_grid(times, expected_steps, args.reference_dt)

    train_tau_counts = None
    if train_tau is not None:
        train_tau_counts = {
            str(float(tau)): int(
                torch.isclose(
                    train_tau,
                    torch.as_tensor(float(tau), dtype=train_tau.dtype),
                    rtol=1e-6,
                    atol=1e-8,
                ).sum().item()
            )
            for tau in args.train_taus
        }
    return {
        "train_u0_shape": list(train_u0.shape),
        "train_ut_shape": list(train_ut.shape),
        "train_rollout_ut_shape": (
            list(train_rollout_ut.shape)
            if train_rollout_ut is not None
            else None
        ),
        "val_u0_shape": list(val_u0.shape),
        "validation_trajectory_length": expected_steps + 1,
        "train_tau_counts": train_tau_counts,
        "train_u0_range": [float(train_u0.min()), float(train_u0.max())],
        "train_ut_range": [float(train_ut.min()), float(train_ut.max())],
        "train_rollout_ut_range": (
            [float(train_rollout_ut.min()), float(train_rollout_ut.max())]
            if train_rollout_ut is not None
            else None
        ),
        "val_u0_range": [float(val_u0.min()), float(val_u0.max())],
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_generate_data(args, *, allow_generate=True):
    """Load a validated cache, or create it only in the preparation stage."""
    cache_path = os.path.abspath(args.data_cache)
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        validation = validate_data_cache(data, args)
        return data, "loaded", validation

    if not allow_generate:
        raise FileNotFoundError(
            "frozen Allen-Cahn data cache does not exist; run once with "
            f"--prepare-data-only before launching seed jobs: {cache_path}"
        )

    data = generate_data(args)
    validation = validate_data_cache(data, args)
    _atomic_torch_save(data, cache_path)
    # Re-open the serialized artifact so the preparation path validates the
    # exact bytes that later seed jobs will read.
    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
    validation = validate_data_cache(saved, args)
    return saved, "created", validation


def allen_cahn_free_energy(u, L, epsilon):
    """Return the physical Allen--Cahn free energy per sample.

    For ``u_t = epsilon^2 u_xx + u - u^3`` this is

    ``E(u) = integral [epsilon^2/2 |u_x|^2 + (1-u^2)^2/4] dx``.

    It is intentionally separate from ``model.energy``: the latter is the
    learned latent energy and is not assumed to equal the PDE energy.
    """
    if u.ndim == 1:
        u = u.unsqueeze(0)
        squeeze = True
    elif u.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"u must have shape (batch, N) or (N,), got {tuple(u.shape)}")
    n_sites = u.shape[-1]
    dx = float(L) / n_sites
    k = 2.0 * math.pi * torch.fft.fftfreq(
        n_sites,
        d=dx,
        device=u.device,
        dtype=torch.float64,
    )
    u_double = u.to(torch.float64)
    u_hat = torch.fft.fft(u_double, dim=-1)
    # torch.fft.fft is unnormalized. Parseval therefore gives
    # integral |u_x|^2 dx = L / N^2 * sum_k |k * u_hat_k|^2.
    grad_sq_integral = (
        float(L)
        * (k.square() * u_hat.abs().square()).sum(dim=-1)
        / n_sites**2
    )
    potential = ((1.0 - u_double.square()).square() / 4.0).sum(dim=-1) * dx
    energy = 0.5 * float(epsilon) ** 2 * grad_sq_integral + potential
    energy = energy.to(dtype=u.dtype)
    return energy[0] if squeeze else energy


def latent_structure_metadata(args):
    """Record the latent constraints independently of the data cache.

    The frozen reference data do not depend on this choice, but it materially
    changes the structural prior used by the latent model.  In particular,
    an Allen--Cahn diagnostic with a zero floor intentionally gives up the
    positive quadratic coercivity-in-z guarantee.
    """
    beta_v_floor = float(args.beta_v_floor)
    has_floor = beta_v_floor > 0.0
    return {
        "state_bounds": [LOWER_BOUND, UPPER_BOUND],
        "decoder": "u=m+(M-m)*sigmoid(z)",
        "beta_v_floor": beta_v_floor,
        "has_positive_z_coercivity_floor": has_floor,
        "z_coercivity_status": (
            "positive_quadratic_floor" if has_floor else "disabled_diagnostic"
        ),
        "mobility_stencil_boundary": "circular_periodic",
    }


def model_config(name, args):
    if name == "latent":
        return {
            "class": "LatentSemigroupNetBounded",
            "kwargs": {
                "N": args.N,
                "m": LOWER_BOUND,
                "M": UPPER_BOUND,
                "hidden_V": [64, 64],
                "hidden_K": [64, 64],
                "stencil_radius": 3,
                "interaction_radius": 2,
                "beta_V": 0.0,
                "beta_V_floor": args.beta_v_floor,
            },
        }
    if name == "latent_decoded_interaction":
        return {
            "class": "DecodedInteractionLatentSemigroupNetBounded",
            "kwargs": {
                "N": args.N,
                "m": LOWER_BOUND,
                "M": UPPER_BOUND,
                "hidden_V": [64, 64],
                "hidden_K": [64, 64],
                "stencil_radius": 3,
                "interaction_radius": 2,
                "beta_V": 0.0,
                "beta_V_floor": args.beta_v_floor,
            },
            "interaction_energy": "quadratic_decoded_state_differences",
        }
    if name == "latent_decoded_interaction_jacobian_mobility":
        return {
            "class": "DecodedInteractionJacobianMobilityLatentSemigroupNetBounded",
            "kwargs": {
                "N": args.N,
                "m": LOWER_BOUND,
                "M": UPPER_BOUND,
                "hidden_V": [64, 64],
                "hidden_K": [64, 64],
                "stencil_radius": 3,
                "interaction_radius": 2,
                "beta_V": 0.0,
                "beta_V_floor": args.beta_v_floor,
            },
            "interaction_energy": "quadratic_decoded_state_differences",
            "mobility": "positive_stencil_over_decoder_jacobian_squared",
        }
    if name == "latent_decoded_energy":
        return {
            "class": "DecodedStateEnergyLatentSemigroupNetBounded",
            "kwargs": {
                "N": args.N,
                "m": LOWER_BOUND,
                "M": UPPER_BOUND,
                "hidden_V": [64, 64],
                "hidden_K": [64, 64],
                "stencil_radius": 3,
                "interaction_radius": 2,
                "beta_V": 0.0,
                "beta_V_floor": args.beta_v_floor,
            },
            "interaction_energy": "quadratic_decoded_state_differences",
            "potential_coordinate": "decoded_state_with_chain_rule",
        }
    if name == "latent_periodic_decoded_interaction":
        return {
            "class": "PeriodicStencilDecodedInteractionLatentSemigroupNetBounded",
            "kwargs": {
                "N": args.N,
                "m": LOWER_BOUND,
                "M": UPPER_BOUND,
                "hidden_V": [64, 64],
                "hidden_K": [64, 64],
                "stencil_radius": 3,
                "interaction_radius": 2,
                "beta_V": 0.0,
                "beta_V_floor": args.beta_v_floor,
            },
            "interaction_energy": "quadratic_decoded_state_differences",
            "interaction_parameterization": "shared_periodic_distance_stencil",
        }
    if name == "latent_physics_anchored_periodic":
        return {
            "class": "PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded",
            "kwargs": {
                "N": args.N,
                "m": LOWER_BOUND,
                "M": UPPER_BOUND,
                "hidden_V": [64, 64],
                "hidden_K": [64, 64],
                "stencil_radius": 3,
                "interaction_radius": 2,
                "beta_V": 0.0,
                "beta_V_floor": args.beta_v_floor,
            },
            "interaction_energy": "quadratic_decoded_state_differences",
            "interaction_parameterization": "shared_periodic_distance_stencil",
            "fixed_physical_potential": "(1-u^2)^2/4",
        }
    if name == "resnet":
        return {
            "class": "TimeConditionedResNet",
            "kwargs": {"N": args.N, "width": 18, "blocks": 3},
        }
    if name == "fno":
        return {
            "class": "TimeConditionedFNO",
            "kwargs": {"N": args.N, "width": 16, "modes": 8, "layers": 4},
        }
    raise ValueError(f"unsupported model: {name}")


def build_model(name, args):
    config = model_config(name, args)
    kwargs = config["kwargs"]
    if name == "latent":
        return LatentSemigroupNetBounded(**kwargs)
    if name == "latent_decoded_interaction":
        return DecodedInteractionLatentSemigroupNetBounded(**kwargs)
    if name == "latent_decoded_interaction_jacobian_mobility":
        return DecodedInteractionJacobianMobilityLatentSemigroupNetBounded(**kwargs)
    if name == "latent_decoded_energy":
        return DecodedStateEnergyLatentSemigroupNetBounded(**kwargs)
    if name == "latent_periodic_decoded_interaction":
        return PeriodicStencilDecodedInteractionLatentSemigroupNetBounded(**kwargs)
    if name == "latent_physics_anchored_periodic":
        return PhysicsAnchoredPeriodicDecodedInteractionLatentSemigroupNetBounded(**kwargs)
    if name == "resnet":
        return TimeConditionedResNet(**kwargs)
    if name == "fno":
        return TimeConditionedFNO(**kwargs)
    raise ValueError(f"unsupported model: {name}")


def _architecture_only(args):
    return all(
        getattr(args, name) == 0.0
        for name in (
            "alpha_rollout",
            "alpha_trajectory",
            "alpha_energy",
            "alpha_bound",
            "alpha_v",
        )
    )


def _device_provenance(device):
    gpu_name = None
    cuda_version = getattr(torch.version, "cuda", None)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": cuda_version,
        "device": str(device),
        "gpu": gpu_name,
    }


def _git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _source_hashes():
    paths = {
        "run_allen_cahn_fair.py": Path(__file__),
        "models.py": EXPERIMENTS_DIR / "models.py",
        "training.py": EXPERIMENTS_DIR / "training.py",
        "evaluate.py": EXPERIMENTS_DIR / "evaluate.py",
        "pde_solver.py": EXPERIMENTS_DIR / "pde_solver.py",
        "seed_utils.py": EXPERIMENTS_DIR / "seed_utils.py",
    }
    return {
        name: _sha256_file(path) for name, path in paths.items() if path.exists()
    }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def _write_metrics_csv(path, results):
    """Write one flat, machine-readable evaluation row per model and tau.

    ``result.json`` deliberately retains the complete nested evaluation trace,
    whereas this file is the compact cell-level artifact consumed by generic
    launchers and result collectors.  Keeping both avoids making a successful
    training run depend on a launcher knowing the runner's private directory
    layout.
    """
    identity_fields = (
        "model",
        "tau",
        "parameter_count",
        "checkpoint_epoch",
        "training_seconds",
        "training_peak_memory_bytes",
        "training_examples_per_second",
        "optimizer_updates",
        "examples_seen",
        "evaluation_seconds",
        "evaluation_peak_memory_bytes",
    )
    rows = []
    for model_name, result in results.items():
        evaluations = result.get("evaluations", {})
        if not evaluations:
            raise ValueError(f"cannot write metrics.csv: {model_name} has no evaluations")
        for tau, evaluation in evaluations.items():
            metrics = evaluation.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError(
                    f"cannot write metrics.csv: {model_name} tau={tau} has no metrics"
                )
            row = {
                "model": model_name,
                "tau": float(tau),
                "parameter_count": result["parameter_count"],
                "checkpoint_epoch": result["checkpoint_epoch"],
                "training_seconds": result["training_seconds"],
                "training_peak_memory_bytes": result["training_peak_memory_bytes"],
                "training_examples_per_second": result[
                    "training_examples_per_second"
                ],
                "optimizer_updates": result["optimizer_updates"],
                "examples_seen": result["examples_seen"],
                "evaluation_seconds": evaluation["evaluation_seconds"],
                "evaluation_peak_memory_bytes": evaluation["peak_memory_bytes"],
            }
            for key, value in metrics.items():
                if key not in row:
                    row[key] = value
            rows.append(row)

    if not rows:
        raise ValueError("cannot write metrics.csv: no model evaluations were recorded")
    metric_fields = sorted(
        {key for row in rows for key in row}.difference(identity_fields)
    )
    path = Path(path)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[*identity_fields, *metric_fields],
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_data_provenance(args, cache_path, cache_status, validation):
    """Write the intentionally short provenance artifact for cache preparation."""
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "schema_version": 1,
        "experiment": "allen_cahn_parameter_matched_fair",
        "mode": "prepare_data_only",
        "cache_status": cache_status,
        "data_cache": os.path.abspath(cache_path),
        "data_cache_sha256": _sha256_file(cache_path),
        "data_generation_config": data_config(args),
        "latent_structure": latent_structure_metadata(args),
        "validation": validation,
        "source": {
            "git_commit": _git_commit(),
            "runner_sha256": _sha256_file(Path(__file__)),
        },
    }
    path = os.path.join(args.output_dir, "data_provenance.json")
    _write_json(path, payload)
    return path, payload


def run_model(name, args, data, device, provenance):
    """Train and evaluate one model in an independent output subtree."""
    set_global_seed(args.seed, deterministic=args.deterministic)
    model = build_model(name, args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_dir = os.path.join(args.output_dir, name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    selection_tau = args.fixed_tau
    architecture_only = _architecture_only(args)
    run_metadata = {
        **data_config(args),
        "training_seed": args.seed,
        "model": name,
        "model_config": model_config(name, args),
        "parameter_count": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "validation_interval": args.validation_interval,
        "bounds": [LOWER_BOUND, UPPER_BOUND],
        "beta_V_floor": args.beta_v_floor,
        "latent_structure": latent_structure_metadata(args),
        "architecture_only": architecture_only,
        "auxiliary_loss_weights": {
            "alpha_rollout": args.alpha_rollout,
            "alpha_trajectory": args.alpha_trajectory,
            "alpha_energy": args.alpha_energy,
            "alpha_bound": args.alpha_bound,
            "alpha_v": args.alpha_v,
        },
        "no_resume": args.no_resume,
        "trajectory_supervision": {
            "enabled": args.alpha_trajectory > 0,
            "horizon": args.trajectory_horizon,
            "model_steps": (
                rollout_steps_for_horizon(
                    args.trajectory_horizon, args.fixed_tau
                )
                if args.alpha_trajectory > 0
                else None
            ),
            "target": "frozen_reference_endpoint",
        },
        "provenance": provenance,
    }

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train_model(
        model,
        data["train_u0"],
        data["train_ut"],
        data["val_u0"],
        data["val_trajs"],
        train_tau=data["train_tau"],
        tau=selection_tau,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_rollout=args.alpha_rollout,
        alpha_trajectory=args.alpha_trajectory,
        train_rollout_ut=data.get("train_rollout_ut"),
        trajectory_steps=(
            rollout_steps_for_horizon(args.trajectory_horizon, args.fixed_tau)
            if args.alpha_trajectory > 0
            else None
        ),
        alpha_energy=args.alpha_energy,
        alpha_bound=args.alpha_bound,
        alpha_V=args.alpha_v,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        model_name=name,
        device=device,
        resume_from=(
            None
            if args.no_resume
            else os.path.join(checkpoint_dir, f"{name}_best.pt")
        ),
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        reference_dt=args.reference_dt,
        run_metadata=run_metadata,
        validation_interval=args.validation_interval,
    )
    if is_cuda:
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - start
    training_peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if is_cuda else None
    )

    best_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    physical_energy = lambda state: allen_cahn_free_energy(  # noqa: E731
        state, L=args.L, epsilon=args.epsilon
    )
    eval_taus = (args.fixed_tau,) if args.regime == "fixed" else args.eval_taus
    evaluations = {}
    for eval_tau in eval_taus:
        rollout_steps = rollout_steps_for_horizon(args.eval_horizon, eval_tau)
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        evaluation_start = time.perf_counter()
        metrics, details = evaluate_full(
            model,
            data["val_u0"],
            data["val_trajs"],
            tau=eval_tau,
            rollout_steps=rollout_steps,
            device=device,
            model_name=name,
            lower_bound=LOWER_BOUND,
            upper_bound=UPPER_BOUND,
            reference_dt=args.reference_dt,
            physical_energy_fn=physical_energy,
            collect_latent_diagnostics=name in (
                "latent",
                "latent_decoded_interaction",
                "latent_decoded_interaction_jacobian_mobility",
                "latent_decoded_energy",
                "latent_periodic_decoded_interaction",
                "latent_physics_anchored_periodic",
            ),
            equal_work_base_ode_steps=(
                30
                if name in (
                    "latent",
                    "latent_decoded_interaction",
                    "latent_decoded_interaction_jacobian_mobility",
                    "latent_decoded_energy",
                    "latent_periodic_decoded_interaction",
                    "latent_physics_anchored_periodic",
                )
                else None
            ),
        )
        if is_cuda:
            torch.cuda.synchronize(device)
        evaluation_seconds = time.perf_counter() - evaluation_start
        evaluations[str(eval_tau)] = {
            "metrics": metrics,
            "details": details,
            "evaluation_seconds": evaluation_seconds,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if is_cuda else None
            ),
        }

    result = {
        "model": name,
        "model_config": model_config(name, args),
        "parameter_count": parameter_count,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_seconds": training_seconds,
        "training_peak_memory_bytes": training_peak_memory_bytes,
        "training_examples_per_second": args.epochs * args.n_train / training_seconds,
        "optimizer_updates": args.epochs * math.ceil(args.n_train / args.batch_size),
        "examples_seen": args.epochs * args.n_train,
        "architecture_only": architecture_only,
        "latent_structure": latent_structure_metadata(args),
        "history": history,
        "evaluations": evaluations,
    }
    _write_json(os.path.join(model_dir, "result.json"), result)
    return result


def _parameter_budget_report(names, counts, *, spatial_grid=None):
    report = {name: {"parameter_count": counts[name]} for name in names}
    report["spatial_grid"] = spatial_grid
    if "latent" not in counts:
        return {"status": "not_checked_without_latent_reference", "models": report}
    if spatial_grid is not None and int(spatial_grid) != 64:
        return {
            "status": "recorded_without_enforcement_outside_protocol_N64",
            "models": report,
        }
    for name, tolerance in PARAMETER_TOLERANCES.items():
        if name in counts:
            report[name]["latent_difference"] = counts[name] - counts["latent"]
            report[name]["allowed_absolute_difference"] = tolerance
    report["status"] = (
        "within_protocol_tolerance"
        if all(
            abs(counts[name] - counts["latent"]) <= tolerance
            for name, tolerance in PARAMETER_TOLERANCES.items()
            if name in counts
        )
        else "outside_protocol_tolerance"
    )
    return report


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    data, cache_status, validation = load_or_generate_data(
        args, allow_generate=args.prepare_data_only
    )
    data_provenance_path, data_provenance = write_data_provenance(
        args, args.data_cache, cache_status, validation
    )

    if args.prepare_data_only:
        print(
            json.dumps(
                {
                    "mode": "prepare_data_only",
                    "data_cache": os.path.abspath(args.data_cache),
                    "data_provenance": os.path.abspath(data_provenance_path),
                    "cache_status": cache_status,
                    "validation": validation,
                },
                indent=2,
            )
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    provenance = {
        "schema_version": 1,
        "experiment": "allen_cahn_parameter_matched_fair",
        "git_commit": _git_commit(),
        "config": vars(args),
        "data_generation_config": data["data_generation_config"],
        "data_cache": os.path.abspath(args.data_cache),
        "data_cache_sha256": _sha256_file(args.data_cache),
        "data_cache_status": cache_status,
        "data_validation": validation,
        "environment": _device_provenance(device),
        "source_hashes": _source_hashes(),
        "data_provenance": os.path.abspath(data_provenance_path),
        "latent_structure": latent_structure_metadata(args),
    }
    _write_json(os.path.join(args.output_dir, "provenance.json"), provenance)

    counts = {}
    results = {}
    for name in dict.fromkeys(args.models):
        # Re-seeding inside run_model gives every model the same data-order RNG
        # stream and an independent, reproducible initialization.
        model = build_model(name, args)
        counts[name] = sum(parameter.numel() for parameter in model.parameters())
        del model
    parameter_report = _parameter_budget_report(
        tuple(dict.fromkeys(args.models)), counts, spatial_grid=args.N
    )
    if parameter_report.get("status") == "outside_protocol_tolerance":
        raise ValueError(
            "model parameter counts are outside the Phase-1/Phase-4 matched budget: "
            f"{counts}"
        )
    for name in dict.fromkeys(args.models):
        results[name] = run_model(name, args, data, device, provenance)

    summary = {
        "schema_version": 1,
        "experiment": "allen_cahn_parameter_matched_fair",
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "torch_version": torch.__version__,
        "config": vars(args),
        "data_generation_config": data["data_generation_config"],
        "parameter_budget": parameter_report,
        "data_provenance": data_provenance,
        "latent_structure": latent_structure_metadata(args),
        "results": results,
    }
    _write_json(os.path.join(args.output_dir, "summary.json"), summary)
    _write_metrics_csv(os.path.join(args.output_dir, "metrics.csv"), results)
    print(
        json.dumps(
            {
                "output_dir": os.path.abspath(args.output_dir),
                "models": {name: result["parameter_count"] for name, result in results.items()},
                "optimizer_updates_per_model": results[next(iter(results))][
                    "optimizer_updates"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
