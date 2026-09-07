#!/usr/bin/env python3
"""Direct time-conditioned map control for the Burgers semigroup lane.

Implements ``docs/research/BURGERS_DIRECT_MAP_CONTROL_PROTOCOL.md``: the
A-versus-C comparison named as indispensable by
``docs/research/CURRENT_METHOD_NOVELTY_PRIOR_ART.md``.  Model C keeps the
physics and the 33-parameter budget of the adopted minimal generator and
removes the flow.  Models A and B are loaded frozen and never retrained.
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
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate_burgers_query_conditioning import classify  # noqa: E402
from experiment_artifacts import torch_load_compat  # noqa: E402
from run_burgers_flux_ablation import (  # noqa: E402
    AblatedBurgersFluxFlow,
    validation_mse,
)
from run_burgers_minimal_ab import (  # noqa: E402
    EXPECTED_PARAMETERS,
    HIDDEN_LAYERS,
    HIDDEN_WIDTH,
    PATCH_RADIUS,
    build_pair,
)
from run_burgers_semigroup_screen import HORIZONS, TEST_TAUS, tau_batch  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402

ARCHITECTURE = "r0_h8_l1_anchor"
FLOW_FLUX_EVALUATIONS_PER_CALL = 48  # 12 RK4 substeps of four stages
DIRECT_FLUX_EVALUATIONS_PER_CALL = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms_defect(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt((left - right).square().mean(dim=1)).mean().item())


class PeriodicBurgersDirectMap(nn.Module):
    """One-shot time-conditioned operator with the exact viscous semigroup.

    The linear part is the exponential of the same three-point periodic
    Laplacian the flow models use, so it composes exactly.  The nonlinear part
    is a single learned divergence increment scaled by the requested lag, so
    maps at different lags are not iterates of one flow.  The flux network is
    the adopted minimal generator's, which fixes the parameter budget.
    """

    def __init__(self, *, n_grid: int, length: float, viscosity: float):
        super().__init__()
        self.n_grid = int(n_grid)
        self.length = float(length)
        self.viscosity = float(viscosity)
        self.field = AblatedBurgersFluxFlow(
            n_grid=self.n_grid,
            length=self.length,
            viscosity=self.viscosity,
            radius=PATCH_RADIUS,
            hidden=HIDDEN_WIDTH,
            hidden_layers=HIDDEN_LAYERS,
            ode_steps=1,
            query_conditioned=True,
        )
        wavenumber = torch.arange(self.n_grid, dtype=torch.get_default_dtype())
        eigenvalues = (
            2.0 * torch.cos(2.0 * math.pi * wavenumber / self.n_grid) - 2.0
        ) / self.dx**2
        self.register_buffer("laplacian_eigenvalues", eigenvalues)

    @property
    def dx(self) -> float:
        return self.length / self.n_grid

    def viscous_semigroup(self, state: torch.Tensor, duration) -> torch.Tensor:
        exponent = tau_batch(duration, state) * self.viscosity
        decay = torch.exp(exponent * self.laplacian_eigenvalues)
        return torch.fft.ifft(torch.fft.fft(state) * decay).real

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        flux = self.field.local_flux(state, tau)
        divergence = (
            torch.roll(flux, shifts=-1, dims=1) - torch.roll(flux, shifts=1, dims=1)
        ) / (2.0 * self.dx)
        increment = -tau_batch(tau, state) * divergence
        return self.viscous_semigroup(state, tau) + increment


def compose(model, state: torch.Tensor, *, tau: float, depth: int) -> torch.Tensor:
    for _ in range(depth):
        state = model(state, tau)
    return state


@torch.no_grad()
def evaluate_direct_map(model, data: dict, args) -> dict[str, object]:
    model.eval()
    val_u0 = data["val_u0"]
    trajectories = data["val_trajectories"]
    rollouts = {}
    for tau in TEST_TAUS:
        depth_to_horizon = {int(round(h / tau)): h for h in HORIZONS}
        mse = {h: [] for h in HORIZONS}
        drift = {h: [] for h in HORIZONS}
        energy = {h: [] for h in HORIZONS}
        for start in range(0, len(val_u0), args.eval_batch_size):
            initial = val_u0[start : start + args.eval_batch_size].to(args.device)
            state = initial
            initial_mean = initial.mean(dim=1)
            initial_energy = 0.5 * initial.square().mean(dim=1)
            batch = trajectories[start : start + len(initial)]
            for depth in range(1, max(depth_to_horizon) + 1):
                state = model(state, tau)
                if depth in depth_to_horizon:
                    horizon = depth_to_horizon[depth]
                    index = int(round(horizon / args.reference_dt))
                    reference = torch.stack(
                        [states[index] for _t, states in batch]
                    ).to(args.device)
                    mse[horizon].append(
                        (state - reference).square().mean(dim=1).cpu()
                    )
                    drift[horizon].append(
                        (state.mean(dim=1) - initial_mean).abs().cpu()
                    )
                    energy[horizon].append(
                        (0.5 * state.square().mean(dim=1) - initial_energy).cpu()
                    )
        for horizon in HORIZONS:
            rollouts[f"tau={tau}:horizon={horizon}"] = {
                "rollout_mse_mean": float(
                    torch.cat(mse[horizon]).to(torch.float64).mean().item()
                ),
                "mean_drift_max": float(
                    torch.cat(drift[horizon]).to(torch.float64).max().item()
                ),
                "energy_change_mean": float(
                    torch.cat(energy[horizon]).to(torch.float64).mean().item()
                ),
            }
    return rollouts


@torch.no_grad()
def measure_direct_map_composition(model, sample: torch.Tensor) -> dict[str, object]:
    direct_versus_composed = {}
    for tau in TEST_TAUS:
        for horizon in HORIZONS:
            depth = int(round(horizon / tau))
            composed = compose(model, sample, tau=tau, depth=depth)
            direct = model(sample, horizon)
            direct_versus_composed[f"tau={tau}:horizon={horizon}"] = {
                "rms_defect_mean": rms_defect(direct, composed),
                "composition_depth": depth,
                "composed_flux_evaluations": depth * DIRECT_FLUX_EVALUATIONS_PER_CALL,
                "direct_flux_evaluations": DIRECT_FLUX_EVALUATIONS_PER_CALL,
            }

    fine, coarse = min(TEST_TAUS), max(TEST_TAUS)
    cross_lag = {}
    for horizon in HORIZONS:
        fine_depth = int(round(horizon / fine))
        coarse_depth = int(round(horizon / coarse))
        fine_state = compose(model, sample, tau=fine, depth=fine_depth)
        coarse_state = compose(model, sample, tau=coarse, depth=coarse_depth)
        cross_lag[f"horizon={horizon}"] = {
            "rms_defect_mean": rms_defect(fine_state, coarse_state),
            "fine_lag": fine,
            "coarse_lag": coarse,
            "fine_depth": fine_depth,
            "coarse_depth": coarse_depth,
            "fine_flux_evaluations": fine_depth,
            "coarse_flux_evaluations": coarse_depth,
            "composed_state_rms": float(
                torch.sqrt(fine_state.square().mean(dim=1)).mean().item()
            ),
        }
    return {
        "direct_versus_composed": direct_versus_composed,
        "in_range_cross_lag": cross_lag,
    }


def train_direct_map(model, *, data: dict, args):
    model = model.to(args.device)
    loader = DataLoader(
        TensorDataset(data["train_u0"], data["train_tau"], data["train_ut"]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    checkpoint_dir = Path(args.output_dir) / "c_direct_map" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "c_direct_map_best.pt"
    best_value = float("inf")
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for initial, tau, target in loader:
            prediction = model(initial.to(args.device), tau.to(args.device))
            loss = (prediction - target.to(args.device)).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        validate = epoch % args.validation_interval == 0 or epoch == args.epochs
        score = validation_mse(model, data, args) if validate else None
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
                    "label": "c_direct_map",
                },
                best_path,
            )
    best = torch_load_compat(best_path, map_location=args.device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    return model, {
        "label": "c_direct_map",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": int(best["epoch"]),
        "best_val_mse": float(best["best_val_mse"]),
        "checkpoint": str(best_path.absolute()),
        "checkpoint_sha256": sha256_file(best_path),
        "training_seconds": time.perf_counter() - started,
        "flux_evaluations_per_call": DIRECT_FLUX_EVALUATIONS_PER_CALL,
        "history": history,
    }


def load_frozen_flow_pair(minimal_root: Path, seed: int, config: dict, ode_steps: int):
    autonomous, query = build_pair(config, ode_steps)
    checkpoints = minimal_root / f"cells/s{seed}/checkpoints"
    for model, name in ((autonomous, "a_autonomous"), (query, "b_query_time")):
        payload = torch_load_compat(
            checkpoints / f"{name}_best.pt", map_location="cpu"
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
    return autonomous, query, {
        "a_autonomous_checkpoint_sha256": sha256_file(
            checkpoints / "a_autonomous_best.pt"
        ),
        "b_query_time_checkpoint_sha256": sha256_file(
            checkpoints / "b_query_time_best.pt"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--expect-cache-sha256", required=True)
    parser.add_argument("--minimal-ab-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-dt", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=10)
    parser.add_argument("--n-sample", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=12)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cache_path = Path(args.data_cache)
    digest = sha256_file(cache_path)
    if digest != args.expect_cache_sha256:
        raise RuntimeError(
            "data cache digest does not match the frozen screen cache: "
            f"{digest} != {args.expect_cache_sha256}"
        )
    data = torch_load_compat(cache_path, map_location="cpu")
    config = data["config"]

    set_global_seed(args.seed, deterministic=True)
    direct = PeriodicBurgersDirectMap(
        n_grid=int(config["N"]),
        length=float(config["L"]),
        viscosity=float(config["nu"]),
    )
    count = sum(p.numel() for p in direct.parameters())
    if count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"the direct map must match the minimal generator at "
            f"{EXPECTED_PARAMETERS} parameters, got {count}"
        )

    autonomous, query, frozen_digests = load_frozen_flow_pair(
        Path(args.minimal_ab_root), args.seed, config, args.ode_steps
    )
    autonomous = autonomous.to(args.device)
    query = query.to(args.device)

    set_global_seed(args.seed, deterministic=True)
    direct, result_c = train_direct_map(direct, data=data, args=args)

    rollouts = {
        "a_autonomous": evaluate_direct_map(autonomous, data, args),
        "b_query_time": evaluate_direct_map(query, data, args),
        "c_direct_map": evaluate_direct_map(direct, data, args),
    }
    sample = data["val_u0"][: args.n_sample].to(args.device)
    composition = measure_direct_map_composition(direct, sample)

    floor_source = json.loads(
        (Path(args.minimal_ab_root) / f"cells/s{args.seed}/summary.json").read_text(
            encoding="utf-8"
        )
    )
    floors = floor_source["autonomous_integrator_floor"]
    cross_lag_verdict = {
        key: classify(
            row["rms_defect_mean"],
            max(floors[f"tau=0.04:{key}"], floors[f"tau=0.08:{key}"]),
        )
        for key, row in composition["in_range_cross_lag"].items()
    }

    def ratios(numerator: str):
        cells = {
            key: rollouts[numerator][key]["rollout_mse_mean"]
            / rollouts["c_direct_map"][key]["rollout_mse_mean"]
            for key in rollouts["c_direct_map"]
        }
        return {
            "cells": cells,
            "geometric_mean": math.exp(
                sum(math.log(value) for value in cells.values()) / len(cells)
            ),
        }

    def lag_spread(label: str) -> float:
        spreads = []
        for horizon in HORIZONS:
            values = [
                rollouts[label][f"tau={tau}:horizon={horizon}"]["rollout_mse_mean"]
                for tau in TEST_TAUS
            ]
            spreads.append((max(values) - min(values)) / min(values))
        return max(spreads)

    summary = {
        "experiment": "burgers_direct_map_control",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/BURGERS_DIRECT_MAP_CONTROL_PROTOCOL.md",
        "architecture": ARCHITECTURE,
        "seed": int(args.seed),
        "n_sample": int(args.n_sample),
        "data_cache_sha256": digest,
        "parameter_count_each": EXPECTED_PARAMETERS,
        "flux_evaluations_per_call": {
            "a_autonomous": FLOW_FLUX_EVALUATIONS_PER_CALL,
            "b_query_time": FLOW_FLUX_EVALUATIONS_PER_CALL,
            "c_direct_map": DIRECT_FLUX_EVALUATIONS_PER_CALL,
        },
        "frozen_flow_checkpoints": frozen_digests,
        "autonomous_integrator_floor": floors,
        "runtime": {
            "device": str(args.device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "validation_interval": int(args.validation_interval),
            "checkpoint_selection": "mean one-step validation MSE over training lags only",
        },
        "models": {"c_direct_map": result_c},
        "rollouts": rollouts,
        "direct_map_composition": composition,
        "in_range_cross_lag_verdict": cross_lag_verdict,
        "unseen_lag_mse_spread_max": {
            label: lag_spread(label) for label in rollouts
        },
        "comparison": {
            "mse_a_over_c": ratios("a_autonomous"),
            "mse_b_over_c": ratios("b_query_time"),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "mse_a_over_c": summary["comparison"]["mse_a_over_c"][
                    "geometric_mean"
                ],
                "mse_b_over_c": summary["comparison"]["mse_b_over_c"][
                    "geometric_mean"
                ],
                "in_range_cross_lag_verdict": cross_lag_verdict,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
