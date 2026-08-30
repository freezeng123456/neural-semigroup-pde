#!/usr/bin/env python3
"""Prepare data or run one Boundary-Compatible Semigroup Wave 1 cell."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.boundary_semigroup import (  # noqa: E402
    BOUNDARY_MODES,
    METRIC_FIELDS,
    TEMPORAL_MODES,
    TRAINING_SEEDS,
    BoundarySemigroupFlow,
    build_reference_cache,
    evaluate_model,
    parameter_count,
    parameter_fingerprint,
    time_key,
    validate_reference_cache,
    wave_config,
)
from experiments.experiment_artifacts import (  # noqa: E402
    atomic_torch_save,
    atomic_write_csv,
    atomic_write_json,
    device_provenance,
    git_commit,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
    torch_load_compat,
    verify_unchanged,
)


PROTOCOL_PATH = (
    REPOSITORY_ROOT
    / "docs"
    / "research"
    / "BOUNDARY_COMPATIBLE_SEMIGROUP_WAVE1_PROTOCOL.md"
)
SOURCE_PATHS = {
    "context": REPOSITORY_ROOT / "CONTEXT.md",
    "protocol": PROTOCOL_PATH,
    "boundary_module": EXPERIMENTS_DIR / "boundary_semigroup.py",
    "runner": Path(__file__).resolve(),
    "artifact_contract": EXPERIMENTS_DIR / "experiment_artifacts.py",
}
TRAINING_LOG_FIELDS = (
    "epoch",
    "train_mse",
    "validation_mse",
    "maximum_prediction_abs",
    "selected_best_so_far",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data", help="create and verify the one shared immutable cache"
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--smoke", action="store_true")

    cell = subparsers.add_parser("cell", help="train and evaluate one matrix cell")
    cell.add_argument("--output-dir", required=True)
    cell.add_argument("--data-cache", required=True)
    cell.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    cell.add_argument("--boundary-mode", choices=BOUNDARY_MODES, required=True)
    cell.add_argument("--temporal-mode", choices=TEMPORAL_MODES, required=True)
    cell.add_argument("--device", default="cuda")
    cell.add_argument("--smoke", action="store_true")
    return parser


def _actual_command() -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


def _run_identity(
    *, seed: int, boundary_mode: str, temporal_mode: str, smoke_only: bool
) -> dict[str, Any]:
    return {
        "experiment": "boundary_compatible_semigroup_wave1",
        "evidence_class": "exploratory_smoke" if smoke_only else "exploratory_full",
        "smoke_only": smoke_only,
        "training_seed": int(seed),
        "boundary_mode": boundary_mode,
        "temporal_mode": temporal_mode,
    }


def _write_manifest(
    root: Path,
    *,
    identity: dict[str, Any],
    artifact_names: tuple[str, ...],
    runtime: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    artifacts = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in artifact_names
    }
    payload = {
        "schema_version": 1,
        **identity,
        "created_at_unix": time.time(),
        "git_commit": git_commit(REPOSITORY_ROOT),
        "source_hashes": source_hashes(SOURCE_PATHS),
        "runtime": runtime,
        "artifacts": artifacts,
    }
    path = root / "exploratory_manifest.json"
    atomic_write_json(path, payload)
    return path, payload


def prepare_data(output_dir: str, *, smoke_only: bool) -> dict[str, Any]:
    config = wave_config(smoke_only=smoke_only)
    root = reserve_exploratory_root(output_dir)
    cache_path = root / "boundary_wave1_cache.pt"
    identity = {
        "experiment": config.experiment,
        "evidence_class": "exploratory_smoke" if smoke_only else "exploratory_full",
        "mode": "prepare_data",
        "smoke_only": smoke_only,
        "data_seed": config.data_seed,
    }
    run_card = {
        "schema_version": 1,
        **identity,
        "created_at_unix": time.time(),
        "actual_command": _actual_command(),
        "pid": os.getpid(),
        "config": config.as_dict(),
        "git_commit": git_commit(REPOSITORY_ROOT),
        "source_hashes": source_hashes(SOURCE_PATHS),
    }
    atomic_write_json(root / "run_card.json", run_card)
    atomic_write_json(root / "config.json", config.as_dict())

    cache = build_reference_cache(config)
    atomic_torch_save(cache, cache_path)
    cache_sha256 = sha256_file(cache_path)
    reloaded = torch_load_compat(cache_path, map_location="cpu")
    validation = validate_reference_cache(reloaded, config)
    results = {
        "schema_version": 1,
        **identity,
        "status": "passed",
        "cache": {
            "path": str(cache_path.resolve()),
            "bytes": cache_path.stat().st_size,
            "sha256": cache_sha256,
        },
        "validation": validation,
    }
    atomic_write_json(root / "results.json", results)
    runtime = device_provenance(torch.device("cpu"))
    manifest_path, _manifest = _write_manifest(
        root,
        identity=identity,
        artifact_names=(
            "run_card.json",
            "config.json",
            "boundary_wave1_cache.pt",
            "results.json",
        ),
        runtime=runtime,
    )
    receipt = {
        "schema_version": 1,
        **identity,
        "status": "passed",
        "normal_exit": True,
        "exit_code": 0,
        "actual_command": _actual_command(),
        "pid": os.getpid(),
        "cache_sha256": cache_sha256,
        "results_sha256": sha256_file(root / "results.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "source_hashes": source_hashes(SOURCE_PATHS),
    }
    atomic_write_json(root / "receipt.json", receipt)
    (root / "done").write_text("passed\n", encoding="utf-8")
    return results


def _set_training_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _build_model(config, *, boundary_mode: str, temporal_mode: str):
    return BoundarySemigroupFlow(
        n_grid=config.n_grid,
        hidden_width=config.hidden_width,
        boundary_mode=boundary_mode,
        temporal_mode=temporal_mode,
        query_time_scale=config.query_time_scale,
        rk4_steps=config.neural_rk4_steps,
    )


def _validation_mse(
    model: BoundarySemigroupFlow,
    val_u0: torch.Tensor,
    val_targets: dict[str, torch.Tensor],
    train_taus: tuple[float, ...],
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for tau in train_taus:
            target = val_targets[time_key(tau)]
            prediction = model(val_u0, tau)
            losses.append(F.mse_loss(prediction, target))
    return float(torch.stack(losses).mean().detach().cpu())


def _train(
    model: BoundarySemigroupFlow,
    cache: dict[str, Any],
    config,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]], int, float]:
    train_u0 = cache["train_u0"].to(device)
    train_tau = cache["train_tau"].to(device)
    train_target = cache["train_target"].to(device)
    val_u0 = cache["val_u0"].to(device)
    val_targets = {
        key: value.to(device) for key, value in cache["val_targets"].items()
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(seed + 7001)
    best_validation = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    training_log: list[dict[str, Any]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(
            config.n_train, generator=permutation_generator
        )
        epoch_loss = 0.0
        maximum_prediction_abs = 0.0
        for start in range(0, config.n_train, config.batch_size):
            indices = permutation[start : start + config.batch_size].to(device)
            prediction = model(train_u0[indices], train_tau[indices])
            loss = F.mse_loss(prediction, train_target[indices])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = int(indices.numel())
            epoch_loss += float(loss.detach().cpu()) * batch_size
            maximum_prediction_abs = max(
                maximum_prediction_abs,
                float(prediction.detach().abs().max().cpu()),
            )

        validation_mse: float | None = None
        selected = False
        if epoch % config.validation_interval == 0 or epoch == config.epochs:
            validation_mse = _validation_mse(
                model, val_u0, val_targets, config.train_taus
            )
            if not math.isfinite(validation_mse):
                raise RuntimeError(f"non-finite validation loss at epoch {epoch}")
            if validation_mse < best_validation:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                selected = True
        training_log.append(
            {
                "epoch": epoch,
                "train_mse": epoch_loss / config.n_train,
                "validation_mse": validation_mse,
                "maximum_prediction_abs": maximum_prediction_abs,
                "selected_best_so_far": selected,
            }
        )
    if best_state is None or best_epoch <= 0 or not math.isfinite(best_validation):
        raise RuntimeError("training did not produce a validation-selected checkpoint")
    return best_state, training_log, best_epoch, best_validation


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    config = wave_config(smoke_only=bool(args.smoke))
    cache_path = Path(args.data_cache).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"shared data cache does not exist: {cache_path}")
    cache_before = sha256_file(cache_path)
    cache = torch_load_compat(cache_path, map_location="cpu")
    cache_validation = validate_reference_cache(cache, config)
    root = reserve_exploratory_root(args.output_dir, input_paths=(cache_path,))
    identity = _run_identity(
        seed=args.seed,
        boundary_mode=args.boundary_mode,
        temporal_mode=args.temporal_mode,
        smoke_only=config.smoke_only,
    )
    selected_device = torch.device(args.device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    runtime = device_provenance(selected_device)
    actual_config = {
        **config.as_dict(),
        "training_seed": args.seed,
        "boundary_mode": args.boundary_mode,
        "temporal_mode": args.temporal_mode,
        "device": str(selected_device),
    }
    run_card = {
        "schema_version": 1,
        **identity,
        "created_at_unix": time.time(),
        "actual_command": _actual_command(),
        "pid": os.getpid(),
        "config": actual_config,
        "data_cache": {"path": str(cache_path), "sha256": cache_before},
        "checkpoint_selection": "minimum validation one-step MSE only",
        "test_metrics_used_for_selection": False,
        "git_commit": git_commit(REPOSITORY_ROOT),
        "source_hashes": source_hashes(SOURCE_PATHS),
        "runtime": runtime,
    }
    atomic_write_json(root / "run_card.json", run_card)
    atomic_write_json(root / "config.json", actual_config)

    _set_training_seed(args.seed)
    model = _build_model(
        config,
        boundary_mode=args.boundary_mode,
        temporal_mode=args.temporal_mode,
    ).to(selected_device)
    initial_fingerprint = parameter_fingerprint(model)
    model_parameter_count = parameter_count(model)
    best_state, training_log, best_epoch, best_validation = _train(
        model, cache, config, seed=args.seed, device=selected_device
    )
    atomic_write_csv(
        root / "training_log.csv",
        training_log,
        fieldnames=TRAINING_LOG_FIELDS,
    )
    checkpoint = {
        "schema_version": 1,
        **identity,
        "selected_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "checkpoint_rule": "minimum validation one-step MSE only",
        "test_metrics_used_for_selection": False,
        "parameter_count": model_parameter_count,
        "initial_parameter_fingerprint": initial_fingerprint,
        "data_cache_sha256": cache_before,
        "config": actual_config,
        "model_state_dict": best_state,
    }
    checkpoint_path = root / "checkpoint_best.pt"
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    model.load_state_dict(best_state)
    rows, evaluation_summary = evaluate_model(
        model, cache, config, device=selected_device
    )
    atomic_write_csv(root / "metrics.csv", rows, fieldnames=METRIC_FIELDS)
    cache_after = verify_unchanged(cache_path, cache_before, "shared data cache")
    results = {
        "schema_version": 1,
        **identity,
        "status": "passed",
        "normal_exit": True,
        "parameter_count": model_parameter_count,
        "initial_parameter_fingerprint": initial_fingerprint,
        "selected_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "checkpoint_sha256": checkpoint_sha256,
        "cache_validation": cache_validation,
        "evaluation_summary": evaluation_summary,
        "metrics": rows,
        "hard_boundary_threshold": config.boundary_threshold,
        "hard_boundary_pass": (
            args.boundary_mode != "hard_dirichlet"
            or evaluation_summary["maximum_hard_boundary_evidence"]
            <= config.boundary_threshold
        ),
    }
    atomic_write_json(root / "results.json", results)
    manifest_path, _manifest = _write_manifest(
        root,
        identity=identity,
        artifact_names=(
            "run_card.json",
            "config.json",
            "training_log.csv",
            "checkpoint_best.pt",
            "metrics.csv",
            "results.json",
        ),
        runtime=runtime,
    )
    receipt = {
        "schema_version": 1,
        **identity,
        "status": "passed",
        "normal_exit": True,
        "exit_code": 0,
        "actual_command": _actual_command(),
        "pid": os.getpid(),
        "runtime": runtime,
        "git_commit": git_commit(REPOSITORY_ROOT),
        "source_hashes": source_hashes(SOURCE_PATHS),
        "parameter_count": model_parameter_count,
        "initial_parameter_fingerprint": initial_fingerprint,
        "selected_epoch": best_epoch,
        "best_validation_mse": best_validation,
        "test_metrics_used_for_selection": False,
        "input_hashes": {
            "data_cache": {"before": cache_before, "after": cache_after}
        },
        "checkpoint_sha256": checkpoint_sha256,
        "training_log_sha256": sha256_file(root / "training_log.csv"),
        "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
        "results_sha256": sha256_file(root / "results.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "hard_boundary_pass": results["hard_boundary_pass"],
    }
    atomic_write_json(root / "receipt.json", receipt)
    (root / "done").write_text("passed\n", encoding="utf-8")
    return results


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = prepare_data(args.output_dir, smoke_only=bool(args.smoke))
    else:
        result = run_cell(args)
    print(
        {
            "status": result["status"],
            "experiment": result["experiment"],
            "smoke_only": result["smoke_only"],
        }
    )


if __name__ == "__main__":
    main()
