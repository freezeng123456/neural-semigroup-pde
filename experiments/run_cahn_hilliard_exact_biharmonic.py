#!/usr/bin/env python3
"""Exact-biharmonic conservative increment for Cahn--Hilliard.

Implements ``docs/research/CAHN_HILLIARD_EXACT_BIHARMONIC_PROTOCOL.md``.
The protocol commit predates every number this runner writes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from run_cahn_hilliard_conservative_mobility import (  # noqa: E402
    ConservativeCahnHilliardFlow,
    DEFAULT_LOCKED,
    MASS_DRIFT_LIMIT,
    SEEDS,
    evaluate_seed,
    generate_training,
    geometric_mean,
    parameter_count,
    sha256_file,
    tau_column,
)
from seed_utils import set_global_seed  # noqa: E402
from training import train_model  # noqa: E402
from experiment_artifacts import torch_load_compat  # noqa: E402


class ExactBiharmonicConservativeIncrement(ConservativeCahnHilliardFlow):
    """Exact linear semigroup plus one conservative nonlinear increment."""

    def __init__(self, *, n_grid=64, length=2.0 * math.pi, epsilon=0.1):
        super().__init__(n_grid=n_grid, length=length, epsilon=epsilon, ode_steps=1)
        wavenumbers = 2.0 * math.pi * torch.fft.rfftfreq(self.N, d=self.dx)
        self.register_buffer("linear_symbol", -(self.epsilon ** 2) * (wavenumbers ** 4))

    def linear_semigroup(self, state: torch.Tensor, tau) -> torch.Tensor:
        duration = tau_column(tau, state)
        decay = torch.exp(duration * self.linear_symbol)
        transformed = torch.fft.rfft(state.to(dtype=torch.float64), dim=-1)
        evolved = torch.fft.irfft(
            transformed * decay.to(dtype=transformed.dtype), n=self.N, dim=-1
        )
        return evolved.to(dtype=state.dtype)

    def nonlinear_potential(self, state: torch.Tensor) -> torch.Tensor:
        residual = self.residual(state.unsqueeze(-1)).squeeze(-1)
        return state.pow(3) - state + residual

    def vector_field(self, state: torch.Tensor) -> torch.Tensor:
        mobility = self.mobility(state)
        return self.divergence(mobility * self.gradient(self.nonlinear_potential(state)))

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        linear = self.linear_semigroup(state, tau)
        step = tau_column(tau, linear)
        return linear + step * self.vector_field(linear)


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


def run_exact_seed(seed: int, locked: dict, args) -> dict:
    set_global_seed(seed, deterministic=True)
    model = ExactBiharmonicConservativeIncrement(
        n_grid=args.N, length=args.L, epsilon=args.epsilon
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
        tau=0.10,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_rollout=0.0,
        alpha_energy=0.0,
        alpha_bound=0.0,
        weight_decay=1e-5,
        checkpoint_dir=str(checkpoint_dir),
        model_name="c_exact_biharmonic",
        device=args.device,
        lower_bound=-1.0,
        upper_bound=1.0,
        reference_dt=args.reference_dt,
        validation_interval=args.validation_interval,
        resume_from=None,
    )
    training_seconds = time.perf_counter() - started
    best_path = checkpoint_dir / "c_exact_biharmonic_best.pt"
    if not best_path.is_file():
        return {
            "seed": int(seed),
            "parameter_count": parameter_count(model),
            "training_seconds": float(training_seconds),
            "best_checkpoint": None,
            "trained": False,
            "conserves_mass": False,
            "geometric_mean_mse": float("inf"),
        }
    checkpoint = torch_load_compat(best_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluation = evaluate_seed(model, locked, args)
    return {
        "seed": int(seed),
        "parameter_count": parameter_count(model),
        "training_seconds": float(training_seconds),
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "trained": True,
        **evaluation,
    }


def main(argv=None):
    args = parse_args(argv)
    locked_path = Path(args.locked_cache).resolve()
    locked = torch.load(locked_path, map_location="cpu", weights_only=False)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows = []
    for seed in args.seeds:
        row = run_exact_seed(int(seed), locked, args)
        (output_dir / f"s{seed}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
        seed_rows.append(row)
        print(json.dumps({"seed": seed, "trained": row.get("trained"), "mse": row.get("geometric_mean_mse")}, indent=2))
    trained = all(row.get("trained") for row in seed_rows)
    mass_ok = all(row.get("conserves_mass") for row in seed_rows)
    finite_mses = [row["geometric_mean_mse"] for row in seed_rows if row.get("trained")]
    payload = {
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/CAHN_HILLIARD_EXACT_BIHARMONIC_PROTOCOL.md",
        "locked_cache": str(locked_path),
        "locked_cache_sha256": sha256_file(locked_path),
        "mass_drift_limit": MASS_DRIFT_LIMIT,
        "exact_biharmonic_trains": trained,
        "exact_biharmonic_conserves_mass": mass_ok,
        "geometric_mean_mse": geometric_mean(finite_mses) if finite_mses else float("inf"),
        "seeds": seed_rows,
    }
    decision_path = output_dir / "decision.json"
    decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(decision_path), **{k: payload[k] for k in ("exact_biharmonic_trains", "exact_biharmonic_conserves_mass", "geometric_mean_mse")}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
