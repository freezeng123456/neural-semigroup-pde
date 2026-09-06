#!/usr/bin/env python3
"""Wave 2 A/B/C control at matched work.

Implements ``docs/research/WAVE2_DIRECT_MAP_WORK_MATCHED_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundary_family_direct_map import (  # noqa: E402
    EXPECTED_PARAMETERS,
    MATCHED_LABELS,
    MATCHED_NET_EVALUATIONS,
    MODEL_LABELS,
    REFINEMENT_SWEEP,
    BoundaryFamilyDirectMap,
    EulerBoundaryFamilyFlow,
    build_paired_models,
    compose,
    direct_map_cross_lag,
    flow_cross_lag,
    flow_cross_lag_equal_substeps,
    net_evaluations_per_call,
    parameter_count,
    trainable,
)
from boundary_family_semigroup import (  # noqa: E402
    BOUNDARY_FAMILIES,
    LONG_ROLLOUT_PAIRS,
    TRAINING_SEEDS,
    composition_depth,
    residual_max,
    time_key,
    validate_reference_cache,
    wave_config,
)
from experiment_artifacts import torch_load_compat  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402


PROTOCOL_PATH = (
    REPOSITORY_ROOT / "docs/research/WAVE2_DIRECT_MAP_WORK_MATCHED_PROTOCOL.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_mean(values) -> float:
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


@torch.no_grad()
def validation_mse(model, cache: dict[str, Any], train_taus, device) -> float:
    model.eval()
    val_u0 = cache["val_u0"].to(device)
    losses = []
    for tau in train_taus:
        target = cache["val_targets"][time_key(tau)].to(device)
        losses.append(F.mse_loss(model(val_u0, tau), target))
    return float(torch.stack(losses).mean().item())


def train_model(model, *, label: str, cache, config, args, seed: int):
    device = torch.device(args.device)
    model = model.to(device)
    train_u0 = cache["train_u0"].to(device)
    train_tau = cache["train_tau"].to(device)
    train_target = cache["train_target"].to(device)
    optimizer = torch.optim.Adam(
        trainable(model).parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(seed + 7001)
    best_value = math.inf
    best_epoch = 0
    best_state = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(config.n_train, generator=permutation_generator)
        train_sum = 0.0
        for start in range(0, config.n_train, config.batch_size):
            indices = permutation[start : start + config.batch_size].to(device)
            prediction = model(train_u0[indices], train_tau[indices])
            loss = F.mse_loss(prediction, train_target[indices])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss for {label} at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * int(indices.numel())
        score = None
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            score = validation_mse(model, cache, config.train_taus, device)
            if score < best_value:
                best_value = score
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in trainable(model).state_dict().items()
                }
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_sum / config.n_train,
                "val_mse": score,
            }
        )
    if best_state is None:
        raise RuntimeError(f"{label} did not produce a validation checkpoint")
    trainable(model).load_state_dict(best_state, strict=True)
    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{label}_best.pt"
    torch.save(
        {
            "label": label,
            "epoch": best_epoch,
            "best_val_mse": best_value,
            "model_state_dict": best_state,
        },
        checkpoint_path,
    )
    return model, {
        "label": label,
        "parameter_count": parameter_count(model),
        "net_evaluations_per_call": net_evaluations_per_call(label),
        "selected_epoch": int(best_epoch),
        "best_val_mse": float(best_value),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_seconds": time.perf_counter() - started,
        "history": history,
    }


@torch.no_grad()
def evaluate_long_rollouts(model, cache, device) -> dict[str, dict[str, float]]:
    model.eval()
    test_u0 = cache["test_u0"].to(device)
    rows = {}
    for tau, horizon in LONG_ROLLOUT_PAIRS:
        depth = composition_depth(tau, horizon)
        prediction = compose(model, test_u0, tau=tau, depth=depth)
        reference = cache["test_targets"][time_key(horizon)].to(device)
        residual = model.boundary_spec.state_residual(prediction)
        rows[f"tau={tau}:horizon={horizon}"] = {
            "rollout_mse": float(F.mse_loss(prediction, reference).item()),
            "rollout_interior_mse": float(
                F.mse_loss(prediction[:, 1:-1], reference[:, 1:-1]).item()
            ),
            "boundary_state_max": float(residual_max(residual).item()),
        }
    return rows


@torch.no_grad()
def evaluate_stress(model, cache, config, device) -> dict[str, float]:
    test_u0 = cache["test_u0"][: config.stress_samples].clone()
    test_u0[:, 0] = test_u0[:, 0] + config.stress_amplitude
    test_u0[:, -1] = test_u0[:, -1] - config.stress_amplitude
    test_u0 = test_u0.to(device)
    depth = composition_depth(config.stress_tau, config.stress_horizon)
    prediction = compose(model, test_u0, tau=config.stress_tau, depth=depth)
    residual = model.boundary_spec.state_residual(prediction)
    return {
        "stress_boundary_state_max": float(residual_max(residual).item()),
        "stress_samples": int(config.stress_samples),
    }


def mse_ratio(numerator: dict[str, dict[str, float]], denominator: dict[str, dict[str, float]]):
    cells = {
        key: numerator[key]["rollout_mse"] / denominator[key]["rollout_mse"]
        for key in denominator
    }
    return {"cells": cells, "geometric_mean": geometric_mean(cells.values())}


def run_cell(args) -> dict[str, Any]:
    config = wave_config(args.boundary_family, smoke_only=bool(args.smoke))
    cache_path = Path(args.data_cache).expanduser().resolve()
    digest = sha256_file(cache_path)
    if args.expect_cache_sha256 and digest != args.expect_cache_sha256:
        raise RuntimeError(
            f"data cache digest mismatch: {digest} != {args.expect_cache_sha256}"
        )
    cache = torch_load_compat(cache_path, map_location="cpu")
    validate_reference_cache(cache, config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    set_global_seed(args.seed, deterministic=True)
    models = build_paired_models(
        n_grid=config.n_grid,
        hidden_width=config.hidden_width,
        boundary_spec=config.boundary_spec,
        query_time_scale=config.query_time_scale,
        rk4_steps=config.neural_rk4_steps,
    )
    trained = {}
    rollouts = {}
    stress = {}
    for label, model in models.items():
        set_global_seed(args.seed, deterministic=True)
        trained_model, record = train_model(
            model,
            label=label,
            cache=cache,
            config=config,
            args=args,
            seed=args.seed,
        )
        trained[label] = record
        rollouts[label] = evaluate_long_rollouts(trained_model, cache, device)
        stress[label] = evaluate_stress(trained_model, cache, config, device)
        models[label] = trained_model

    sample = cache["test_u0"][: args.n_sample].to(device)
    euler_a = models["a_euler_autonomous"]
    euler_b = models["b_euler_query_time"]
    direct = models["c_direct_map"]
    if not isinstance(euler_a, EulerBoundaryFamilyFlow):
        raise TypeError("A' must be an Euler flow")
    if not isinstance(euler_b, EulerBoundaryFamilyFlow):
        raise TypeError("B' must be an Euler flow")
    if not isinstance(direct, BoundaryFamilyDirectMap):
        raise TypeError("C must be a direct residual map")

    deployed = {
        "a_euler_autonomous": flow_cross_lag(euler_a, sample, MATCHED_NET_EVALUATIONS),
        "b_euler_query_time": flow_cross_lag(euler_b, sample, MATCHED_NET_EVALUATIONS),
        "c_direct_map": direct_map_cross_lag(direct, sample),
    }
    refinement = {
        label: {
            str(substeps): flow_cross_lag(model, sample, substeps)
            for substeps in REFINEMENT_SWEEP
        }
        for label, model in (
            ("a_euler_autonomous", euler_a),
            ("b_euler_query_time", euler_b),
        )
    }
    equal_substeps = {
        "a_euler_autonomous": flow_cross_lag_equal_substeps(euler_a, sample, 1),
        "b_euler_query_time": flow_cross_lag_equal_substeps(euler_b, sample, 1),
    }

    summary = {
        "experiment": "wave2_direct_map_work_matched",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": str(PROTOCOL_PATH.relative_to(REPOSITORY_ROOT)),
        "boundary_family": args.boundary_family,
        "enforcement_mode": "hard",
        "seed": int(args.seed),
        "smoke_only": bool(args.smoke),
        "n_sample": int(args.n_sample),
        "data_cache_sha256": digest,
        "parameter_count_each": EXPECTED_PARAMETERS,
        "net_evaluations_per_call": {
            label: net_evaluations_per_call(label) for label in MODEL_LABELS
        },
        "runtime": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "models": trained,
        "long_rollouts": rollouts,
        "boundary_stress": stress,
        "deployed_budget_cross_lag": deployed,
        "refinement_sweep_cross_lag": refinement,
        "equal_substep_cross_lag": equal_substeps,
        "comparison": {
            "mse_a_prime_over_c": mse_ratio(
                rollouts["a_euler_autonomous"], rollouts["c_direct_map"]
            ),
            "mse_b_prime_over_c": mse_ratio(
                rollouts["b_euler_query_time"], rollouts["c_direct_map"]
            ),
            "mse_a_rk4_over_c": mse_ratio(
                rollouts["a_rk4_autonomous"], rollouts["c_direct_map"]
            ),
            "mse_a_rk4_over_b_rk4": mse_ratio(
                rollouts["a_rk4_autonomous"], rollouts["b_rk4_query_time"]
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
                "boundary_family": args.boundary_family,
                "seed": args.seed,
                "mse_a_prime_over_c": summary["comparison"]["mse_a_prime_over_c"][
                    "geometric_mean"
                ],
                "mse_b_prime_over_c": summary["comparison"]["mse_b_prime_over_c"][
                    "geometric_mean"
                ],
                "mse_a_rk4_over_b_rk4": summary["comparison"]["mse_a_rk4_over_b_rk4"][
                    "geometric_mean"
                ],
            },
            indent=2,
        )
    )
    return summary


def prepare_data(args) -> dict[str, Any]:
    from boundary_family_semigroup import build_reference_cache

    config = wave_config(args.boundary_family, smoke_only=bool(args.smoke))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = build_reference_cache(config)
    cache_path = output_dir / f"{args.boundary_family}.pt"
    torch.save(cache, cache_path)
    digest = sha256_file(cache_path)
    payload = {
        "boundary_family": args.boundary_family,
        "smoke_only": bool(args.smoke),
        "path": str(cache_path.resolve()),
        "sha256": digest,
        "data_identity": config.data_identity(),
    }
    (output_dir / f"{args.boundary_family}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--boundary-family", choices=BOUNDARY_FAMILIES, required=True)
    prepare.add_argument("--smoke", action="store_true")

    cell = subparsers.add_parser("cell")
    cell.add_argument("--output-dir", required=True)
    cell.add_argument("--data-cache", required=True)
    cell.add_argument("--expect-cache-sha256", default="")
    cell.add_argument("--boundary-family", choices=BOUNDARY_FAMILIES, required=True)
    cell.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    cell.add_argument("--device", default="cuda")
    cell.add_argument("--n-sample", type=int, default=32)
    cell.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        return prepare_data(args)
    return run_cell(args)


if __name__ == "__main__":
    main()
