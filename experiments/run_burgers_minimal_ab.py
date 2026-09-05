#!/usr/bin/env python3
"""Paired A/B comparison at the recommended minimal Burgers generator.

Implements ``docs/research/BURGERS_MINIMAL_GENERATOR_AB_PROTOCOL.md``.  The
architecture is the ablation's recommended variant; every other condition is
the screen's.  Composition is measured with the attribution lane's own
measurement function so the three paths keep identical semantics.
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

from evaluate_burgers_query_conditioning import classify, measure_model  # noqa: E402
from experiment_artifacts import torch_load_compat  # noqa: E402
from run_burgers_flux_ablation import (  # noqa: E402
    AblatedBurgersFluxFlow,
    evaluate,
    validation_mse,
)
from seed_utils import set_global_seed  # noqa: E402

ARCHITECTURE = "r0_h8_l1_anchor"
PATCH_RADIUS = 0
HIDDEN_WIDTH = 8
HIDDEN_LAYERS = 1
EXPECTED_PARAMETERS = 33


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_pair(config: dict, ode_steps: int):
    """Model B is initialized from A, so only the control channel differs."""

    def make(query_conditioned: bool):
        return AblatedBurgersFluxFlow(
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


def train(model, *, label: str, data: dict, args) -> dict[str, object]:
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
        "query_conditioned": model.query_conditioned,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "selected_epoch": int(best["epoch"]),
        "best_val_mse": float(best["best_val_mse"]),
        "checkpoint": str(best_path.absolute()),
        "checkpoint_sha256": sha256_file(best_path),
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "evaluation": {"rollouts": evaluate(model, data, args)["rollouts"]},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--expect-cache-sha256", required=True)
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
    autonomous, query = build_pair(config, args.ode_steps)
    counts = {
        sum(p.numel() for p in model.parameters()) for model in (autonomous, query)
    }
    if len(counts) != 1 or counts != {EXPECTED_PARAMETERS}:
        raise RuntimeError(
            f"paired minimal models must both have {EXPECTED_PARAMETERS} parameters"
        )

    set_global_seed(args.seed, deterministic=True)
    autonomous, result_a = train(autonomous, label="a_autonomous", data=data, args=args)
    set_global_seed(args.seed, deterministic=True)
    query, result_b = train(query, label="b_query_time", data=data, args=args)

    sample = data["val_u0"][: args.n_sample].to(args.device)
    paths = {
        "a_autonomous": measure_model(autonomous, sample),
        "b_query_time": measure_model(query, sample),
    }
    floors = {
        key: row["equal_work_rms_defect_mean"]
        for key, row in paths["a_autonomous"]["path2_matched_conditioning"].items()
    }
    autonomous_cross_lag = [
        row["equal_work_rms_defect_mean"]
        for row in paths["a_autonomous"]["path3_in_range_cross_lag"].values()
    ]
    if any(value != 0.0 for value in autonomous_cross_lag):
        raise RuntimeError("autonomous cross-lag paths must agree exactly")

    cross_lag_verdict = {}
    for key, row in paths["b_query_time"]["path3_in_range_cross_lag"].items():
        floor = max(
            floors[f"tau=0.04:{key}"],
            floors[f"tau=0.08:{key}"],
        )
        cross_lag_verdict[key] = classify(
            row["equal_work_rms_defect_mean"], floor
        )

    ratios = {
        key: result_a["evaluation"]["rollouts"][key]["rollout_mse_mean"]
        / result_b["evaluation"]["rollouts"][key]["rollout_mse_mean"]
        for key in result_a["evaluation"]["rollouts"]
    }
    summary = {
        "experiment": "burgers_minimal_generator_ab",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/BURGERS_MINIMAL_GENERATOR_AB_PROTOCOL.md",
        "architecture": ARCHITECTURE,
        "architecture_detail": {
            "patch_radius": PATCH_RADIUS,
            "hidden_width": HIDDEN_WIDTH,
            "hidden_layers": HIDDEN_LAYERS,
            "viscous_anchor": True,
        },
        "seed": int(args.seed),
        "n_sample": int(args.n_sample),
        "data_cache_sha256": digest,
        "parameter_count_each": EXPECTED_PARAMETERS,
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
            "ode_steps": int(args.ode_steps),
            "checkpoint_selection": "mean one-step validation MSE over training lags only",
        },
        "models": {"a_autonomous": result_a, "b_query_time": result_b},
        "composition_paths": paths,
        "autonomous_integrator_floor": floors,
        "in_range_cross_lag_verdict": cross_lag_verdict,
        "comparison": {
            "mse_a_over_b": ratios,
            "mse_a_over_b_geometric_mean": math.exp(
                sum(math.log(value) for value in ratios.values()) / len(ratios)
            ),
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
                "mse_a_over_b_geometric_mean": summary["comparison"][
                    "mse_a_over_b_geometric_mean"
                ],
                "in_range_cross_lag_verdict": cross_lag_verdict,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
