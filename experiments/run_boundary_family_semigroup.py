#!/usr/bin/env python3
"""Prepare one cache or run one Boundary-Family Semigroup Wave 2 cell."""

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

from experiments.boundary_family_semigroup import (  # noqa: E402
    BOUNDARY_FAMILIES,
    ENFORCEMENT_MODES,
    METRIC_FIELDS,
    TEMPORAL_MODES,
    TRAINING_SEEDS,
    BoundaryFamilyFlow,
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
    / "BOUNDARY_FAMILY_SEMIGROUP_WAVE2_PROTOCOL.md"
)
SOURCE_PATHS = {
    "context": REPOSITORY_ROOT / "CONTEXT.md",
    "protocol": PROTOCOL_PATH,
    "boundary_family_module": EXPERIMENTS_DIR / "boundary_family_semigroup.py",
    "runner": Path(__file__).resolve(),
    "artifact_contract": EXPERIMENTS_DIR / "experiment_artifacts.py",
}
TRAINING_LOG_FIELDS = (
    "epoch",
    "train_prediction_mse",
    "train_boundary_penalty",
    "train_objective",
    "validation_prediction_mse",
    "maximum_prediction_abs",
    "selected_best_so_far",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data", help="create one immutable boundary-family cache"
    )
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--boundary-family", choices=BOUNDARY_FAMILIES, required=True)
    prepare.add_argument("--smoke", action="store_true")

    cell = subparsers.add_parser("cell", help="train and evaluate one matrix cell")
    cell.add_argument("--output-dir", required=True)
    cell.add_argument("--data-cache", required=True)
    cell.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    cell.add_argument("--boundary-family", choices=BOUNDARY_FAMILIES, required=True)
    cell.add_argument("--enforcement-mode", choices=ENFORCEMENT_MODES, required=True)
    cell.add_argument("--temporal-mode", choices=TEMPORAL_MODES, required=True)
    cell.add_argument("--device", default="cuda")
    cell.add_argument("--smoke", action="store_true")
    return parser


def _actual_command() -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


def _source_provenance() -> dict[str, Any]:
    commit = git_commit(REPOSITORY_ROOT) or os.environ.get("SEMIGROUP_SOURCE_COMMIT")
    archive_sha256 = os.environ.get("SEMIGROUP_SOURCE_ARCHIVE_SHA256")
    if commit is not None and (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("source commit must be a lowercase 40-character Git SHA")
    if archive_sha256 is not None and (
        len(archive_sha256) != 64
        or any(character not in "0123456789abcdef" for character in archive_sha256)
    ):
        raise ValueError(
            "source archive SHA-256 must be a lowercase 64-character digest"
        )
    return {
        "git_commit": commit,
        "source_archive_sha256": archive_sha256,
        "source_hashes": source_hashes(SOURCE_PATHS),
    }


def _run_identity(
    *,
    boundary_family: str,
    seed: int,
    enforcement_mode: str,
    temporal_mode: str,
    smoke_only: bool,
) -> dict[str, Any]:
    return {
        "experiment": "boundary_family_semigroup_wave2",
        "evidence_class": "exploratory_smoke" if smoke_only else "exploratory_full",
        "smoke_only": smoke_only,
        "boundary_family": boundary_family,
        "training_seed": int(seed),
        "enforcement_mode": enforcement_mode,
        "temporal_mode": temporal_mode,
    }


def _write_manifest(
    root: Path,
    *,
    identity: dict[str, Any],
    artifact_names: tuple[str, ...],
    runtime: dict[str, Any],
) -> Path:
    artifacts = {
        name: {
            "bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in artifact_names
    }
    path = root / "exploratory_manifest.json"
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            **identity,
            "created_at_unix": time.time(),
            **_source_provenance(),
            "runtime": runtime,
            "artifacts": artifacts,
        },
    )
    return path


def cache_filename(boundary_family: str) -> str:
    if boundary_family not in BOUNDARY_FAMILIES:
        raise ValueError(f"unsupported boundary family: {boundary_family}")
    return f"boundary_family_wave2_{boundary_family}_cache.pt"


def prepare_data(
    output_dir: str, *, boundary_family: str, smoke_only: bool
) -> dict[str, Any]:
    config = wave_config(boundary_family, smoke_only=smoke_only)
    root = reserve_exploratory_root(output_dir)
    cache_path = root / cache_filename(boundary_family)
    identity = {
        "experiment": config.experiment,
        "evidence_class": "exploratory_smoke" if smoke_only else "exploratory_full",
        "mode": "prepare_data",
        "smoke_only": smoke_only,
        "boundary_family": boundary_family,
        "data_seed": config.data_seed,
    }
    atomic_write_json(
        root / "run_card.json",
        {
            "schema_version": 1,
            **identity,
            "created_at_unix": time.time(),
            "actual_command": _actual_command(),
            "pid": os.getpid(),
            "config": config.as_dict(),
            **_source_provenance(),
        },
    )
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
    manifest_path = _write_manifest(
        root,
        identity=identity,
        artifact_names=(
            "run_card.json",
            "config.json",
            cache_filename(boundary_family),
            "results.json",
        ),
        runtime=runtime,
    )
    atomic_write_json(
        root / "receipt.json",
        {
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
            **_source_provenance(),
        },
    )
    (root / "done").write_text("passed\n", encoding="utf-8")
    return results


def _set_training_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _build_model(
    config,
    *,
    enforcement_mode: str,
    temporal_mode: str,
) -> BoundaryFamilyFlow:
    return BoundaryFamilyFlow(
        n_grid=config.n_grid,
        hidden_width=config.hidden_width,
        boundary_spec=config.boundary_spec,
        enforcement_mode=enforcement_mode,
        temporal_mode=temporal_mode,
        query_time_scale=config.query_time_scale,
        rk4_steps=config.neural_rk4_steps,
    )


def _validation_mse(
    model: BoundaryFamilyFlow,
    val_u0: torch.Tensor,
    val_targets: dict[str, torch.Tensor],
    train_taus: tuple[float, ...],
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for tau in train_taus:
            losses.append(F.mse_loss(model(val_u0, tau), val_targets[time_key(tau)]))
    return float(torch.stack(losses).mean().detach().cpu())


def _train(
    model: BoundaryFamilyFlow,
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
    val_targets = {key: value.to(device) for key, value in cache["val_targets"].items()}
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
        permutation = torch.randperm(config.n_train, generator=permutation_generator)
        prediction_sum = 0.0
        penalty_sum = 0.0
        objective_sum = 0.0
        maximum_prediction_abs = 0.0
        for start in range(0, config.n_train, config.batch_size):
            indices = permutation[start : start + config.batch_size].to(device)
            prediction = model(train_u0[indices], train_tau[indices])
            prediction_mse = F.mse_loss(prediction, train_target[indices])
            boundary_penalty = (
                model.boundary_spec.state_residual(prediction).square().mean()
            )
            objective = prediction_mse
            if model.enforcement_mode == "penalty":
                objective = objective + config.penalty_lambda * boundary_penalty
            if not bool(torch.isfinite(objective)):
                raise RuntimeError(f"non-finite training objective at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            batch_size = int(indices.numel())
            prediction_sum += float(prediction_mse.detach().cpu()) * batch_size
            penalty_sum += float(boundary_penalty.detach().cpu()) * batch_size
            objective_sum += float(objective.detach().cpu()) * batch_size
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
                raise RuntimeError(f"non-finite validation MSE at epoch {epoch}")
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
                "train_prediction_mse": prediction_sum / config.n_train,
                "train_boundary_penalty": penalty_sum / config.n_train,
                "train_objective": objective_sum / config.n_train,
                "validation_prediction_mse": validation_mse,
                "maximum_prediction_abs": maximum_prediction_abs,
                "selected_best_so_far": selected,
            }
        )
    if best_state is None or best_epoch <= 0 or not math.isfinite(best_validation):
        raise RuntimeError("training did not produce a validation-selected checkpoint")
    return best_state, training_log, best_epoch, best_validation


def run_cell(args: argparse.Namespace) -> dict[str, Any]:
    config = wave_config(args.boundary_family, smoke_only=bool(args.smoke))
    cache_path = Path(args.data_cache).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"data cache does not exist: {cache_path}")
    cache_before = sha256_file(cache_path)
    cache = torch_load_compat(cache_path, map_location="cpu")
    cache_validation = validate_reference_cache(cache, config)
    root = reserve_exploratory_root(args.output_dir, input_paths=(cache_path,))
    identity = _run_identity(
        boundary_family=args.boundary_family,
        seed=args.seed,
        enforcement_mode=args.enforcement_mode,
        temporal_mode=args.temporal_mode,
        smoke_only=config.smoke_only,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    runtime = device_provenance(device)
    actual_config = {
        **config.as_dict(),
        "training_seed": args.seed,
        "enforcement_mode": args.enforcement_mode,
        "temporal_mode": args.temporal_mode,
        "device": str(device),
    }
    atomic_write_json(
        root / "run_card.json",
        {
            "schema_version": 1,
            **identity,
            "created_at_unix": time.time(),
            "actual_command": _actual_command(),
            "pid": os.getpid(),
            "config": actual_config,
            "data_cache": {"path": str(cache_path), "sha256": cache_before},
            "checkpoint_selection": "minimum validation prediction MSE only",
            "test_metrics_used_for_selection": False,
            **_source_provenance(),
            "runtime": runtime,
        },
    )
    atomic_write_json(root / "config.json", actual_config)

    _set_training_seed(args.seed)
    model = _build_model(
        config,
        enforcement_mode=args.enforcement_mode,
        temporal_mode=args.temporal_mode,
    ).to(device)
    initial_fingerprint = parameter_fingerprint(model)
    model_parameter_count = parameter_count(model)
    best_state, training_log, best_epoch, best_validation = _train(
        model, cache, config, seed=args.seed, device=device
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
        "checkpoint_rule": "minimum validation prediction MSE only",
        "test_metrics_used_for_selection": False,
        "parameter_count": model_parameter_count,
        "initial_parameter_fingerprint": initial_fingerprint,
        "data_cache_sha256": cache_before,
        **_source_provenance(),
        "config": actual_config,
        "model_state_dict": best_state,
    }
    checkpoint_path = root / "checkpoint_best.pt"
    atomic_torch_save(checkpoint, checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    model.load_state_dict(best_state)
    rows, evaluation_summary = evaluate_model(model, cache, config, device=device)
    atomic_write_csv(root / "metrics.csv", rows, fieldnames=METRIC_FIELDS)
    cache_after = verify_unchanged(cache_path, cache_before, "boundary-family cache")
    hard_boundary_pass = (
        args.enforcement_mode != "hard"
        or evaluation_summary["maximum_hard_boundary_evidence"]
        <= config.boundary_threshold
    )
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
        "hard_boundary_pass": hard_boundary_pass,
        "penalty_contract": {
            "lambda": config.penalty_lambda,
            "applied": args.enforcement_mode == "penalty",
            "location": "final training prediction only",
            "checkpoint_selection_uses_prediction_mse_only": True,
        },
    }
    atomic_write_json(root / "results.json", results)
    manifest_path = _write_manifest(
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
    atomic_write_json(
        root / "receipt.json",
        {
            "schema_version": 1,
            **identity,
            "status": "passed",
            "normal_exit": True,
            "exit_code": 0,
            "actual_command": _actual_command(),
            "pid": os.getpid(),
            "runtime": runtime,
            **_source_provenance(),
            "parameter_count": model_parameter_count,
            "initial_parameter_fingerprint": initial_fingerprint,
            "selected_epoch": best_epoch,
            "best_validation_mse": best_validation,
            "test_metrics_used_for_selection": False,
            "data_cache_path": str(cache_path),
            "input_hashes": {
                "data_cache": {"before": cache_before, "after": cache_after}
            },
            "checkpoint_sha256": checkpoint_sha256,
            "training_log_sha256": sha256_file(root / "training_log.csv"),
            "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
            "results_sha256": sha256_file(root / "results.json"),
            "manifest_sha256": sha256_file(manifest_path),
            "hard_boundary_pass": hard_boundary_pass,
        },
    )
    (root / "done").write_text("passed\n", encoding="utf-8")
    return results


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        result = prepare_data(
            args.output_dir,
            boundary_family=args.boundary_family,
            smoke_only=bool(args.smoke),
        )
    else:
        result = run_cell(args)
    print(
        {
            "status": result["status"],
            "experiment": result["experiment"],
            "boundary_family": result["boundary_family"],
            "smoke_only": result["smoke_only"],
        }
    )


if __name__ == "__main__":
    main()
