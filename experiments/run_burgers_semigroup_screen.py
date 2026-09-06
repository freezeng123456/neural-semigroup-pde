#!/usr/bin/env python3
"""Matched autonomous/query-conditioned screen for periodic viscous Burgers."""

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
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import torch_load_compat  # noqa: E402
from pde_solver import BurgersSolver, generate_burgers_initial_conditions  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402


TRAIN_TAUS = (0.025, 0.05, 0.075, 0.1)
TEST_TAUS = (0.04, 0.08)
HORIZONS = (0.4, 0.8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tau_batch(tau, state: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(tau, dtype=state.dtype, device=state.device)
    if value.ndim == 0:
        value = value.expand(state.shape[0])
    if value.ndim != 1 or value.shape[0] != state.shape[0]:
        raise ValueError("tau must be scalar or have one value per sample")
    return value.reshape(-1, 1)


class PeriodicBurgersFluxFlow(nn.Module):
    """Periodic conservative flux plus fixed viscous dissipation.

    Both variants have the same parameterization.  The autonomous model feeds
    a zero control channel; the query-conditioned model feeds the requested
    duration.  The spatial mean is preserved because every learned term is a
    periodic discrete divergence and the fixed Laplacian also sums to zero.
    """

    def __init__(
        self,
        *,
        n_grid: int,
        length: float,
        viscosity: float,
        radius: int = 2,
        hidden: int = 32,
        ode_steps: int = 12,
        query_conditioned: bool = False,
    ):
        super().__init__()
        self.n_grid = int(n_grid)
        self.length = float(length)
        self.viscosity = float(viscosity)
        self.radius = int(radius)
        self.ode_steps = int(ode_steps)
        self.query_conditioned = bool(query_conditioned)
        width = 2 * self.radius + 2
        self.flux = nn.Sequential(
            nn.Linear(width, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.2)
                nn.init.zeros_(module.bias)

    @property
    def dx(self) -> float:
        return self.length / self.n_grid

    def local_flux(self, state: torch.Tensor, conditioning_tau) -> torch.Tensor:
        patches = torch.stack(
            [
                torch.roll(state, shifts=-offset, dims=1)
                for offset in range(-self.radius, self.radius + 1)
            ],
            dim=-1,
        )
        control = tau_batch(conditioning_tau, state)
        if not self.query_conditioned:
            control = torch.zeros_like(control)
        control = (control / max(TEST_TAUS)).unsqueeze(1).expand(-1, state.shape[1], -1)
        return self.flux(torch.cat((patches, control), dim=-1)).squeeze(-1)

    def rhs(self, state: torch.Tensor, conditioning_tau) -> torch.Tensor:
        flux = self.local_flux(state, conditioning_tau)
        divergence = (
            torch.roll(flux, shifts=-1, dims=1)
            - torch.roll(flux, shifts=1, dims=1)
        ) / (2.0 * self.dx)
        laplacian = (
            torch.roll(state, shifts=-1, dims=1)
            - 2.0 * state
            + torch.roll(state, shifts=1, dims=1)
        ) / (self.dx**2)
        return -divergence + self.viscosity * laplacian

    def integrate(
        self,
        state: torch.Tensor,
        duration,
        *,
        substeps: int,
        conditioning_tau,
    ) -> torch.Tensor:
        dt = tau_batch(duration, state) / int(substeps)
        value = state
        for _ in range(int(substeps)):
            k1 = self.rhs(value, conditioning_tau)
            k2 = self.rhs(value + 0.5 * dt * k1, conditioning_tau)
            k3 = self.rhs(value + 0.5 * dt * k2, conditioning_tau)
            k4 = self.rhs(value + dt * k3, conditioning_tau)
            value = value + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return value

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        return self.integrate(
            state,
            tau,
            substeps=self.ode_steps,
            conditioning_tau=tau,
        )


def generate_cache(args: argparse.Namespace) -> dict[str, object]:
    set_global_seed(args.data_seed, deterministic=True)
    np.random.seed(args.data_seed)
    solver = BurgersSolver(
        N=args.N,
        L=args.L,
        nu=args.nu,
        dt=args.reference_dt,
    )
    train_u0 = generate_burgers_initial_conditions(args.N, args.n_train, args.L)
    generator = torch.Generator().manual_seed(args.data_seed)
    tau_index = torch.randint(
        len(TRAIN_TAUS),
        (args.n_train,),
        generator=generator,
    )
    train_tau = torch.tensor(TRAIN_TAUS)[tau_index]
    train_ut = torch.empty_like(train_u0)
    for index in range(args.n_train):
        tau = float(train_tau[index].item())
        _times, states = solver.solve(
            train_u0[index],
            tau,
            save_every=max(1, int(round(tau / args.reference_dt))),
        )
        train_ut[index] = states[-1]

    val_u0 = generate_burgers_initial_conditions(args.N, args.n_val, args.L)
    val_trajectories = []
    for initial in val_u0:
        times, states = solver.solve(initial, max(HORIZONS), save_every=1)
        val_trajectories.append((times, states))
    return {
        "schema_version": 1,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "pde": "viscous_burgers_periodic",
        "config": {
            "N": int(args.N),
            "L": float(args.L),
            "nu": float(args.nu),
            "reference_dt": float(args.reference_dt),
            "train_taus": list(TRAIN_TAUS),
            "test_taus": list(TEST_TAUS),
            "horizons": list(HORIZONS),
            "data_seed": int(args.data_seed),
        },
        "train_u0": train_u0,
        "train_tau": train_tau,
        "train_ut": train_ut,
        "val_u0": val_u0,
        "val_trajectories": val_trajectories,
    }


@torch.no_grad()
def evaluate_model(
    model: PeriodicBurgersFluxFlow,
    data: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    model.eval()
    val_u0 = data["val_u0"]
    trajectories = data["val_trajectories"]
    rows = {}
    for tau in TEST_TAUS:
        depth_to_horizon = {
            int(round(horizon / tau)): horizon for horizon in HORIZONS
        }
        mse = {horizon: [] for horizon in HORIZONS}
        mean_drift = {horizon: [] for horizon in HORIZONS}
        energy_change = {horizon: [] for horizon in HORIZONS}
        for start in range(0, len(val_u0), args.eval_batch_size):
            initial = val_u0[start : start + args.eval_batch_size].to(args.device)
            state = initial
            initial_mean = initial.mean(dim=1)
            initial_energy = 0.5 * initial.square().mean(dim=1)
            batch_trajectories = trajectories[start : start + len(initial)]
            for depth in range(1, max(depth_to_horizon) + 1):
                state = model(state, tau)
                if depth in depth_to_horizon:
                    horizon = depth_to_horizon[depth]
                    index = int(round(horizon / args.reference_dt))
                    reference = torch.stack(
                        [states[index] for _times, states in batch_trajectories]
                    ).to(args.device)
                    mse[horizon].append(
                        (state - reference).square().mean(dim=1).cpu()
                    )
                    mean_drift[horizon].append(
                        (state.mean(dim=1) - initial_mean).abs().cpu()
                    )
                    energy_change[horizon].append(
                        (0.5 * state.square().mean(dim=1) - initial_energy).cpu()
                    )
        for horizon in HORIZONS:
            key = f"tau={tau}:horizon={horizon}"
            mse_values = torch.cat(mse[horizon]).to(torch.float64)
            drift_values = torch.cat(mean_drift[horizon]).to(torch.float64)
            energy_values = torch.cat(energy_change[horizon]).to(torch.float64)
            rows[key] = {
                "rollout_mse_mean": float(mse_values.mean().item()),
                "mean_drift_max": float(drift_values.max().item()),
                "energy_change_mean": float(energy_values.mean().item()),
                "energy_increase_fraction": float(
                    (energy_values > 1e-8).to(torch.float64).mean().item()
                ),
            }

    composition = {}
    sample = val_u0[: min(16, len(val_u0))].to(args.device)
    for tau in TEST_TAUS:
        for horizon in HORIZONS:
            depth = int(round(horizon / tau))
            composed = sample
            for _ in range(depth):
                composed = model(composed, tau)
            direct = model.integrate(
                sample,
                horizon,
                substeps=depth * model.ode_steps,
                conditioning_tau=horizon,
            )
            defect = torch.sqrt((direct - composed).square().mean(dim=1))
            composition[f"tau={tau}:horizon={horizon}"] = {
                "equal_work_rms_defect_mean": float(defect.mean().item()),
                "composition_depth": depth,
                "rhs_evaluations_each_path": 4 * depth * model.ode_steps,
            }
    return {"rollouts": rows, "composition": composition}


@torch.no_grad()
def validation_mse(
    model: PeriodicBurgersFluxFlow,
    data: dict[str, object],
    args: argparse.Namespace,
) -> float:
    """Select checkpoints without observing the unseen-lag evaluation cells."""

    model.eval()
    val_u0 = data["val_u0"]
    trajectories = data["val_trajectories"]
    values = []
    for tau in TRAIN_TAUS:
        reference_index = int(round(tau / args.reference_dt))
        for start in range(0, len(val_u0), args.eval_batch_size):
            initial = val_u0[start : start + args.eval_batch_size].to(args.device)
            prediction = model(initial, tau)
            batch_trajectories = trajectories[start : start + len(initial)]
            reference = torch.stack(
                [states[reference_index] for _times, states in batch_trajectories]
            ).to(args.device)
            values.append((prediction - reference).square().mean(dim=1).cpu())
    return float(torch.cat(values).to(torch.float64).mean().item())


def train_model(
    model: PeriodicBurgersFluxFlow,
    *,
    label: str,
    data: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    model = model.to(args.device)
    dataset = TensorDataset(data["train_u0"], data["train_tau"], data["train_ut"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    model_dir = Path(args.output_dir) / label
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_value = float("inf")
    best_path = checkpoint_dir / f"{label}_best.pt"
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for initial, tau, target in loader:
            initial = initial.to(args.device)
            tau = tau.to(args.device)
            target = target.to(args.device)
            prediction = model(initial, tau)
            loss = (prediction - target).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        should_validate = epoch % args.validation_interval == 0 or epoch == args.epochs
        score = validation_mse(model, data, args) if should_validate else None
        history.append(
            {"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": score}
        )
        if score is not None and score < best_value:
            best_value = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_mse": best_value,
                    "label": label,
                },
                best_path,
            )
    best = torch_load_compat(best_path, map_location=args.device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    evaluation = evaluate_model(model, data, args)
    result = {
        "label": label,
        "query_conditioned": model.query_conditioned,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": int(best["epoch"]),
        "checkpoint": str(best_path.absolute()),
        "checkpoint_sha256": sha256_file(best_path),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "evaluation": evaluation,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--data-seed", type=int, default=20260902)
    parser.add_argument("--prepare-data-only", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * math.pi)
    parser.add_argument("--nu", type=float, default=0.01)
    parser.add_argument("--reference-dt", type=float, default=0.001)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=32)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).absolute()
    cache_path = Path(args.data_cache).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_data_only:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = generate_cache(args)
        torch.save(data, cache_path)
        receipt = {
            "experiment": "burgers_semigroup_screen_data",
            "exploratory": True,
            "do_not_use_for_formal": True,
            "cache": str(cache_path),
            "cache_sha256": sha256_file(cache_path),
            "config": data["config"],
        }
        (output_dir / "summary.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2))
        return

    data = torch_load_compat(cache_path, map_location="cpu")
    expected_cache_config = {
        "N": int(args.N),
        "L": float(args.L),
        "nu": float(args.nu),
        "reference_dt": float(args.reference_dt),
        "train_taus": list(TRAIN_TAUS),
        "test_taus": list(TEST_TAUS),
        "horizons": list(HORIZONS),
        "data_seed": int(args.data_seed),
    }
    if data.get("config") != expected_cache_config:
        raise RuntimeError("data cache configuration does not match this run")
    set_global_seed(args.seed, deterministic=True)
    autonomous = PeriodicBurgersFluxFlow(
        n_grid=args.N,
        length=args.L,
        viscosity=args.nu,
        hidden=args.hidden,
        ode_steps=args.ode_steps,
        query_conditioned=False,
    )
    initial_state = {name: value.clone() for name, value in autonomous.state_dict().items()}
    query = PeriodicBurgersFluxFlow(
        n_grid=args.N,
        length=args.L,
        viscosity=args.nu,
        hidden=args.hidden,
        ode_steps=args.ode_steps,
        query_conditioned=True,
    )
    query.load_state_dict(initial_state, strict=True)
    count_a = sum(parameter.numel() for parameter in autonomous.parameters())
    count_b = sum(parameter.numel() for parameter in query.parameters())
    if count_a != count_b:
        raise RuntimeError("matched Burgers models must have identical parameter counts")

    set_global_seed(args.seed, deterministic=True)
    result_a = train_model(autonomous, label="a_autonomous", data=data, args=args)
    set_global_seed(args.seed, deterministic=True)
    result_b = train_model(query, label="b_query_time", data=data, args=args)
    ratios = {
        key: result_a["evaluation"]["rollouts"][key]["rollout_mse_mean"]
        / result_b["evaluation"]["rollouts"][key]["rollout_mse_mean"]
        for key in result_a["evaluation"]["rollouts"]
    }
    summary = {
        "experiment": "burgers_matched_semigroup_screen",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": int(args.seed),
        "data_cache": str(cache_path),
        "data_cache_sha256": sha256_file(cache_path),
        "parameter_count_each": count_a,
        "runtime": {
            "device": str(args.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "validation_interval": int(args.validation_interval),
            "ode_steps": int(args.ode_steps),
            "hidden": int(args.hidden),
            "checkpoint_selection": "mean one-step validation MSE over training lags only",
        },
        "models": {"a_autonomous": result_a, "b_query_time": result_b},
        "comparison": {
            "mse_a_over_b": ratios,
            "mse_a_over_b_geometric_mean": math.exp(
                sum(math.log(value) for value in ratios.values()) / len(ratios)
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
