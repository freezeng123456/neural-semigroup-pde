#!/usr/bin/env python3
"""Flux-generator ablation for the matched Burgers semigroup screen.

Implements the frozen grid in
``docs/research/BURGERS_FLUX_ABLATION_PROTOCOL.md``.  Every condition outside
the architecture is imported from the screen runner, which is left untouched
so the committed screen results stay reproducible from their source digests.
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

from experiment_artifacts import torch_load_compat  # noqa: E402
from run_burgers_semigroup_screen import (  # noqa: E402
    HORIZONS,
    TEST_TAUS,
    TRAIN_TAUS,
    PeriodicBurgersFluxFlow,
)
from seed_utils import set_global_seed  # noqa: E402

# label -> patch radius, hidden width, hidden layers, viscous anchor
VARIANTS = {
    "r2_h32_l2_anchor": (2, 32, 2, True),
    "r1_h32_l2_anchor": (1, 32, 2, True),
    "r0_h32_l2_anchor": (0, 32, 2, True),
    "r0_h8_l2_anchor": (0, 8, 2, True),
    "r0_h8_l1_anchor": (0, 8, 1, True),
    "r2_h32_l2_noanchor": (2, 32, 2, False),
    "r0_h32_l2_noanchor": (0, 32, 2, False),
}
REFERENCE_VARIANT = "r2_h32_l2_anchor"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AblatedBurgersFluxFlow(PeriodicBurgersFluxFlow):
    """Screen generator with a configurable flux depth.

    Patch radius, hidden width, and the viscous anchor are already constructor
    arguments of the screen class; only the layer count needs a new option.
    """

    def __init__(self, *, hidden: int, hidden_layers: int, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        if hidden_layers not in (1, 2):
            raise ValueError("hidden_layers must be 1 or 2")
        self.hidden_layers = int(hidden_layers)
        if self.hidden_layers == 1:
            width = 2 * self.radius + 2
            self.flux = nn.Sequential(
                nn.Linear(width, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            for module in self.flux.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight, gain=0.2)
                    nn.init.zeros_(module.bias)


def build_variant(label: str, *, n_grid: int, length: float, viscosity: float, ode_steps: int):
    radius, hidden, layers, anchor = VARIANTS[label]
    return AblatedBurgersFluxFlow(
        n_grid=n_grid,
        length=length,
        viscosity=viscosity if anchor else 0.0,
        radius=radius,
        hidden=hidden,
        hidden_layers=layers,
        ode_steps=ode_steps,
        query_conditioned=False,
    )


@torch.no_grad()
def validation_mse(model, data, args) -> float:
    """Selection metric: one-step MSE on the training lags only."""

    model.eval()
    val_u0 = data["val_u0"]
    trajectories = data["val_trajectories"]
    values = []
    for tau in TRAIN_TAUS:
        index = int(round(tau / args.reference_dt))
        for start in range(0, len(val_u0), args.eval_batch_size):
            initial = val_u0[start : start + args.eval_batch_size].to(args.device)
            prediction = model(initial, tau)
            batch = trajectories[start : start + len(initial)]
            reference = torch.stack([states[index] for _t, states in batch]).to(
                args.device
            )
            values.append((prediction - reference).square().mean(dim=1).cpu())
    return float(torch.cat(values).to(torch.float64).mean().item())


@torch.no_grad()
def evaluate(model, data, args) -> dict[str, object]:
    model.eval()
    val_u0 = data["val_u0"]
    trajectories = data["val_trajectories"]
    rollouts = {}
    composition = {}
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
            key = f"tau={tau}:horizon={horizon}"
            rollouts[key] = {
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

    # Matched conditioning: both paths present the same value to the control
    # channel, so the defect is nonuniform RK4 truncation only.
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
                conditioning_tau=tau,
            )
            composition[f"tau={tau}:horizon={horizon}"] = {
                "equal_work_rms_defect_mean": float(
                    torch.sqrt((direct - composed).square().mean(dim=1))
                    .mean()
                    .item()
                ),
                "rhs_evaluations_each_path": 4 * depth * model.ode_steps,
            }
    return {"rollouts": rollouts, "composition": composition}


def train_variant(label: str, data: dict, args) -> dict[str, object]:
    config = data["config"]
    set_global_seed(args.seed, deterministic=True)
    model = build_variant(
        label,
        n_grid=int(config["N"]),
        length=float(config["L"]),
        viscosity=float(config["nu"]),
        ode_steps=args.ode_steps,
    ).to(args.device)
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
    cell_dir = Path(args.output_dir) / label
    (cell_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    best_path = cell_dir / "checkpoints" / f"{label}_best.pt"
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
                    "label": label,
                },
                best_path,
            )
    best = torch_load_compat(best_path, map_location=args.device)
    model.load_state_dict(best["model_state_dict"], strict=True)
    radius, hidden, layers, anchor = VARIANTS[label]
    return {
        "label": label,
        "patch_radius": radius,
        "hidden_width": hidden,
        "hidden_layers": layers,
        "viscous_anchor": anchor,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": int(best["epoch"]),
        "best_val_mse": float(best["best_val_mse"]),
        "checkpoint_sha256": sha256_file(best_path),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "evaluation": evaluate(model, data, args),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--expect-cache-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variants", nargs="*", default=sorted(VARIANTS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference-dt", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=10)
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
    unknown = [label for label in args.variants if label not in VARIANTS]
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    data = torch_load_compat(cache_path, map_location="cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = {label: train_variant(label, data, args) for label in args.variants}
    reference = variants.get(REFERENCE_VARIANT)
    ratios = {}
    if reference is not None:
        for label, result in variants.items():
            cells = {
                key: result["evaluation"]["rollouts"][key]["rollout_mse_mean"]
                / reference["evaluation"]["rollouts"][key]["rollout_mse_mean"]
                for key in result["evaluation"]["rollouts"]
            }
            ratios[label] = {
                "cells": cells,
                "geometric_mean": math.exp(
                    sum(math.log(value) for value in cells.values()) / len(cells)
                ),
            }

    summary = {
        "experiment": "burgers_flux_ablation",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/BURGERS_FLUX_ABLATION_PROTOCOL.md",
        "seed": int(args.seed),
        "data_cache_sha256": digest,
        "reference_variant": REFERENCE_VARIANT,
        "runtime": {"device": str(args.device), "cuda_available": bool(torch.cuda.is_available())},
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "validation_interval": int(args.validation_interval),
            "ode_steps": int(args.ode_steps),
            "checkpoint_selection": "mean one-step validation MSE over training lags only",
        },
        "variants": variants,
        "rollout_mse_ratio_to_reference": ratios,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                label: {
                    "parameters": variants[label]["parameter_count"],
                    "gm_ratio": ratios.get(label, {}).get("geometric_mean"),
                }
                for label in variants
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
