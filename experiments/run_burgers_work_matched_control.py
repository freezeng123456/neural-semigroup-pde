#!/usr/bin/env python3
"""Work-matched flow versus direct-map comparison for Burgers.

Implements ``docs/research/BURGERS_WORK_MATCHED_CONTROL_PROTOCOL.md``.  The
flow models are integrated by explicit Euler with one substep, which costs
exactly one flux evaluation per call and therefore matches the direct map.
Model C is loaded frozen from the unmatched-work lane and never retrained.
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
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import torch_load_compat  # noqa: E402
from run_burgers_direct_map_control import (  # noqa: E402
    PeriodicBurgersDirectMap,
    compose,
    rms_defect,
)
from run_burgers_flux_ablation import AblatedBurgersFluxFlow, validation_mse  # noqa: E402
from run_burgers_minimal_ab import (  # noqa: E402
    EXPECTED_PARAMETERS,
    HIDDEN_LAYERS,
    HIDDEN_WIDTH,
    PATCH_RADIUS,
)
from run_burgers_semigroup_screen import HORIZONS, TEST_TAUS, tau_batch  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402

ARCHITECTURE = "r0_h8_l1_anchor"
MATCHED_FLUX_EVALUATIONS_PER_CALL = 1
REFINEMENT_SWEEP = (1, 2, 4, 8, 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EulerBurgersFluxFlow(AblatedBurgersFluxFlow):
    """The minimal generator integrated by explicit Euler.

    One substep costs one flux evaluation, so the deployed map matches the
    direct map's budget exactly.  The underlying object is still a vector
    field, which is what makes the composition defect refinable.
    """

    def integrate(self, state, duration, *, substeps: int, conditioning_tau):
        dt = tau_batch(duration, state) / int(substeps)
        for _ in range(int(substeps)):
            state = state + dt * self.rhs(state, conditioning_tau)
        return state


def build_pair(config: dict, ode_steps: int):
    def make(query_conditioned: bool):
        return EulerBurgersFluxFlow(
            n_grid=int(config["N"]),
            length=float(config["L"]),
            viscosity=float(config["nu"]),
            radius=PATCH_RADIUS,
            hidden=HIDDEN_WIDTH,
            hidden_layers=HIDDEN_LAYERS,
            ode_steps=ode_steps,
            query_conditioned=query_conditioned,
        )

    autonomous = make(False)
    query = make(True)
    query.load_state_dict(autonomous.state_dict(), strict=True)
    return autonomous, query


@torch.no_grad()
def evaluate_rollouts(model, data: dict, args) -> dict[str, object]:
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
                    mse[horizon].append((state - reference).square().mean(dim=1).cpu())
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
def flow_cross_lag(model, sample: torch.Tensor, substeps: int) -> dict[str, object]:
    """In-range cross-lag defect for a flow at a fixed per-call substep count.

    Euler costs one flux evaluation per substep, so the evaluation count is the
    depth times the substep count.
    """

    fine, coarse = min(TEST_TAUS), max(TEST_TAUS)
    rows = {}
    for horizon in HORIZONS:
        fine_depth = int(round(horizon / fine))
        coarse_depth = int(round(horizon / coarse))
        fine_state = sample
        for _ in range(fine_depth):
            fine_state = model.integrate(
                fine_state, fine, substeps=substeps, conditioning_tau=fine
            )
        coarse_state = sample
        for _ in range(coarse_depth):
            coarse_state = model.integrate(
                coarse_state, coarse, substeps=substeps, conditioning_tau=coarse
            )
        rows[f"horizon={horizon}"] = {
            "rms_defect_mean": rms_defect(fine_state, coarse_state),
            "substeps_per_call": substeps,
            "fine_flux_evaluations": fine_depth * substeps,
            "coarse_flux_evaluations": coarse_depth * substeps,
            "composed_state_rms": float(
                torch.sqrt(fine_state.square().mean(dim=1)).mean().item()
            ),
        }
    return rows


@torch.no_grad()
def flow_cross_lag_equal_substeps(model, sample: torch.Tensor, substeps: int):
    """Cross-lag defect when both paths are given the same substep size.

    For an autonomous field this must vanish exactly: the two paths become one
    Euler trajectory.
    """

    fine, coarse = min(TEST_TAUS), max(TEST_TAUS)
    ratio = int(round(coarse / fine))
    rows = {}
    for horizon in HORIZONS:
        fine_state = sample
        for _ in range(int(round(horizon / fine))):
            fine_state = model.integrate(
                fine_state, fine, substeps=substeps, conditioning_tau=fine
            )
        coarse_state = sample
        for _ in range(int(round(horizon / coarse))):
            coarse_state = model.integrate(
                coarse_state, coarse, substeps=substeps * ratio, conditioning_tau=coarse
            )
        rows[f"horizon={horizon}"] = {
            "rms_defect_mean": rms_defect(fine_state, coarse_state),
            "fine_substeps_per_call": substeps,
            "coarse_substeps_per_call": substeps * ratio,
        }
    return rows


@torch.no_grad()
def direct_map_cross_lag(model, sample: torch.Tensor) -> dict[str, object]:
    fine, coarse = min(TEST_TAUS), max(TEST_TAUS)
    rows = {}
    for horizon in HORIZONS:
        fine_depth = int(round(horizon / fine))
        coarse_depth = int(round(horizon / coarse))
        fine_state = compose(model, sample, tau=fine, depth=fine_depth)
        coarse_state = compose(model, sample, tau=coarse, depth=coarse_depth)
        rows[f"horizon={horizon}"] = {
            "rms_defect_mean": rms_defect(fine_state, coarse_state),
            "substeps_per_call": None,
            "fine_flux_evaluations": fine_depth,
            "coarse_flux_evaluations": coarse_depth,
            "composed_state_rms": float(
                torch.sqrt(fine_state.square().mean(dim=1)).mean().item()
            ),
        }
    return rows


def train(model, *, label: str, data: dict, args):
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
    checkpoint_dir = Path(args.output_dir) / label / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{label}_best.pt"
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
    return model, {
        "label": label,
        "integrator": "explicit_euler",
        "substeps_per_call": int(model.ode_steps),
        "flux_evaluations_per_call": int(model.ode_steps),
        "query_conditioned": model.query_conditioned,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": int(best["epoch"]),
        "best_val_mse": float(best["best_val_mse"]),
        "checkpoint_sha256": sha256_file(best_path),
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--expect-cache-sha256", required=True)
    parser.add_argument("--direct-map-root", required=True)
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
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cache_path = Path(args.data_cache)
    digest = sha256_file(cache_path)
    if digest != args.expect_cache_sha256:
        raise RuntimeError(
            f"data cache digest mismatch: {digest} != {args.expect_cache_sha256}"
        )
    data = torch_load_compat(cache_path, map_location="cpu")
    config = data["config"]

    direct_checkpoint = (
        Path(args.direct_map_root)
        / f"cells/s{args.seed}/checkpoints/c_direct_map_best.pt"
    )
    direct = PeriodicBurgersDirectMap(
        n_grid=int(config["N"]),
        length=float(config["L"]),
        viscosity=float(config["nu"]),
    )
    direct.load_state_dict(
        torch_load_compat(direct_checkpoint, map_location="cpu")["model_state_dict"],
        strict=True,
    )
    direct = direct.to(args.device).eval()

    set_global_seed(args.seed, deterministic=True)
    autonomous, query = build_pair(config, MATCHED_FLUX_EVALUATIONS_PER_CALL)
    counts = {
        sum(p.numel() for p in model.parameters())
        for model in (autonomous, query, direct)
    }
    if counts != {EXPECTED_PARAMETERS}:
        raise RuntimeError(
            f"all three models must have {EXPECTED_PARAMETERS} parameters, got {counts}"
        )

    set_global_seed(args.seed, deterministic=True)
    autonomous, result_a = train(
        autonomous, label="a_euler_autonomous", data=data, args=args
    )
    set_global_seed(args.seed, deterministic=True)
    query, result_b = train(query, label="b_euler_query_time", data=data, args=args)

    rollouts = {
        "a_euler_autonomous": evaluate_rollouts(autonomous, data, args),
        "b_euler_query_time": evaluate_rollouts(query, data, args),
        "c_direct_map": evaluate_rollouts(direct, data, args),
    }
    sample = data["val_u0"][: args.n_sample].to(args.device)
    deployed = {
        "a_euler_autonomous": flow_cross_lag(
            autonomous, sample, MATCHED_FLUX_EVALUATIONS_PER_CALL
        ),
        "b_euler_query_time": flow_cross_lag(
            query, sample, MATCHED_FLUX_EVALUATIONS_PER_CALL
        ),
        "c_direct_map": direct_map_cross_lag(direct, sample),
    }
    refinement = {
        label: {
            str(substeps): flow_cross_lag(model, sample, substeps)
            for substeps in REFINEMENT_SWEEP
        }
        for label, model in (
            ("a_euler_autonomous", autonomous),
            ("b_euler_query_time", query),
        )
    }
    equal_substeps = {
        label: flow_cross_lag_equal_substeps(model, sample, 1)
        for label, model in (
            ("a_euler_autonomous", autonomous),
            ("b_euler_query_time", query),
        )
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

    summary = {
        "experiment": "burgers_work_matched_control",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/BURGERS_WORK_MATCHED_CONTROL_PROTOCOL.md",
        "architecture": ARCHITECTURE,
        "seed": int(args.seed),
        "n_sample": int(args.n_sample),
        "data_cache_sha256": digest,
        "parameter_count_each": EXPECTED_PARAMETERS,
        "flux_evaluations_per_call": {
            "a_euler_autonomous": MATCHED_FLUX_EVALUATIONS_PER_CALL,
            "b_euler_query_time": MATCHED_FLUX_EVALUATIONS_PER_CALL,
            "c_direct_map": MATCHED_FLUX_EVALUATIONS_PER_CALL,
        },
        "frozen_direct_map_checkpoint_sha256": sha256_file(direct_checkpoint),
        "runtime": {
            "device": str(args.device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "training_config": {
            "integrator": "explicit_euler",
            "substeps_per_call": MATCHED_FLUX_EVALUATIONS_PER_CALL,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "learning_rate": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "validation_interval": int(args.validation_interval),
            "checkpoint_selection": "mean one-step validation MSE over training lags only",
        },
        "models": {"a_euler_autonomous": result_a, "b_euler_query_time": result_b},
        "rollouts": rollouts,
        "deployed_budget_cross_lag": deployed,
        "refinement_sweep_cross_lag": refinement,
        "equal_substep_cross_lag": equal_substeps,
        "comparison": {
            "mse_a_over_c": ratios("a_euler_autonomous"),
            "mse_b_over_c": ratios("b_euler_query_time"),
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
                "mse_a_over_c": summary["comparison"]["mse_a_over_c"]["geometric_mean"],
                "mse_b_over_c": summary["comparison"]["mse_b_over_c"]["geometric_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
