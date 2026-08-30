#!/usr/bin/env python3
"""Checkpoint-only Fisher cross-integrator and lag--horizon evaluator.

One invocation evaluates one already selected formal A/B checkpoint with one
integrator.  It never trains, tunes, selects a checkpoint, or generates a
cache.  The two supported preregistered modes are:

``integrator_control``
    The immutable 500-sample formal cache, lags 0.075/0.15, horizons
    1.2/2.4/4.8, and one of Euler/RK2/RK4.

``phase_diagram``
    The separately prepared 128-sample exploratory cache, seven lags, four
    horizons, and RK4.

Both modes report production fixed-substep and equal-RHS-work semantics.  The
equal-work path asserts direct/composed RHS equality for every grid cell.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    device_provenance,
    git_commit,
    require_input_file,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
    verify_unchanged,
)
from fisher_frozen_inputs import (  # noqa: E402
    FORMAL_CACHE_PROFILE,
    FORMAL_LOCKED_CACHE_SHA256,
    FORMAL_SOURCE_ARCHIVE_SHA256,
    FORMAL_SOURCE_COMMIT,
    PHASE_CACHE_PROFILE,
    load_frozen_fisher_cache,
    load_frozen_fisher_checkpoint,
    reference_states_at,
    runtime_training_source_hashes,
)
from latent_integrators import (  # noqa: E402
    evolve_latent_flow,
    integer_depth,
    integration_work,
    normalize_integrator,
    substeps_for_rhs_budget,
)
from seed_utils import set_global_seed  # noqa: E402


SCHEMA_VERSION = 1
TRACK = "fisher_semigroup_novelty_grid"
MODE_INTEGRATOR = "integrator_control"
MODE_PHASE = "phase_diagram"
WORK_FIXED = "production_fixed_substeps"
WORK_EQUAL = "equal_rhs_work"
FORMAL_TAUS = (0.075, 0.15)
FORMAL_HORIZONS = (1.2, 2.4, 4.8)
PHASE_TAUS = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
PHASE_HORIZONS = (0.6, 1.2, 2.4, 4.8)
FIXED_SUBSTEPS = 30
EQUAL_RHS_PER_BASE = 120
CSV_FIELDS = (
    "mode",
    "seed",
    "model",
    "integrator",
    "work_semantics",
    "lag",
    "horizon",
    "composition_depth",
    "direct_substeps",
    "composed_substeps",
    "direct_rhs_evaluations",
    "composed_rhs_evaluations",
    "rhs_work_equal",
    "prediction_mse_mean",
    "prediction_relative_l2_mean",
    "composition_mse_mean",
    "composition_relative_l2_mean",
    "prediction_nonfinite_samples",
    "composition_nonfinite_samples",
)


def _parse_positive_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive floats"
        ) from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be finite and positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _distribution(values: torch.Tensor) -> dict[str, Any]:
    detached = values.detach().reshape(-1).to(torch.float64).cpu()
    finite_mask = torch.isfinite(detached)
    finite = detached[finite_mask]
    total = int(detached.numel())
    if not finite.numel():
        return {
            "count": total,
            "finite_count": 0,
            "nonfinite_count": total,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.5, 0.9, 0.95], dtype=finite.dtype),
    )
    return {
        "count": total,
        "finite_count": int(finite.numel()),
        "nonfinite_count": total - int(finite.numel()),
        "mean": float(finite.mean().item()),
        "median": float(quantiles[0].item()),
        "p90": float(quantiles[1].item()),
        "p95": float(quantiles[2].item()),
        "max": float(finite.max().item()),
    }


def _pair_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    relative_eps: float = 1e-12,
) -> dict[str, Any]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"metric tensors must have equal shape: {prediction.shape} != {target.shape}"
        )
    difference = prediction - target
    batch = int(difference.shape[0])
    flattened_difference = difference.reshape(batch, -1)
    flattened_target = target.reshape(batch, -1)
    mse = flattened_difference.square().mean(dim=1)
    absolute_l2 = torch.linalg.vector_norm(flattened_difference, dim=1)
    target_l2 = torch.linalg.vector_norm(flattened_target, dim=1)
    relative_l2 = absolute_l2 / torch.clamp(target_l2, min=float(relative_eps))
    finite_prediction = torch.isfinite(prediction.reshape(batch, -1)).all(dim=1)
    finite_target = torch.isfinite(target.reshape(batch, -1)).all(dim=1)
    finite_pair = finite_prediction & finite_target
    return {
        "mse": _distribution(mse),
        "absolute_l2": _distribution(absolute_l2),
        "relative_l2": _distribution(relative_l2),
        "finite_samples": int(finite_pair.sum().item()),
        "nonfinite_samples": batch - int(finite_pair.sum().item()),
    }


def _runtime_source_hashes() -> dict[str, str]:
    return source_hashes(
        {
            "evaluate_fisher_semigroup_grid.py": Path(__file__).resolve(),
            "experiment_artifacts.py": EXPERIMENTS_DIR / "experiment_artifacts.py",
            "fisher_frozen_inputs.py": EXPERIMENTS_DIR / "fisher_frozen_inputs.py",
            "latent_integrators.py": EXPERIMENTS_DIR / "latent_integrators.py",
            "models.py": EXPERIMENTS_DIR / "models.py",
            "seed_utils.py": EXPERIMENTS_DIR / "seed_utils.py",
        }
    )


def _grid_for_mode(mode: str) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    if mode == MODE_INTEGRATOR:
        return FORMAL_CACHE_PROFILE, FORMAL_TAUS, FORMAL_HORIZONS
    if mode == MODE_PHASE:
        return PHASE_CACHE_PROFILE, PHASE_TAUS, PHASE_HORIZONS
    raise ValueError(f"unsupported evaluation mode: {mode!r}")


def _validate_protocol_args(args: argparse.Namespace) -> None:
    cache_profile, expected_taus, expected_horizons = _grid_for_mode(args.mode)
    if tuple(args.taus) != expected_taus or tuple(args.horizons) != expected_horizons:
        raise ValueError(
            "lag/horizon grid differs from the preregistered mode: "
            f"taus={tuple(args.taus)}, horizons={tuple(args.horizons)}"
        )
    args.cache_profile = cache_profile
    args.integrator = normalize_integrator(args.integrator)
    if args.mode == MODE_PHASE and args.integrator != "rk4":
        raise ValueError("phase_diagram is preregistered with RK4 only")
    if int(args.fixed_substeps) != FIXED_SUBSTEPS:
        raise ValueError(f"fixed-substeps is frozen at {FIXED_SUBSTEPS}")
    if int(args.equal_rhs_per_base) != EQUAL_RHS_PER_BASE:
        raise ValueError(f"equal RHS budget is frozen at {EQUAL_RHS_PER_BASE}")
    if int(args.seed) not in {31415, 271828, 161803}:
        raise ValueError("seed must be one of the three frozen formal seeds")
    if args.model not in {"latent", "latent_query_time"}:
        raise ValueError("model must be latent or latent_query_time")
    for digest_name in (
        "checkpoint_sha256",
        "cache_sha256",
        "source_archive_sha256",
    ):
        digest = str(getattr(args, digest_name)).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{digest_name} must be a SHA-256 digest")
        setattr(args, digest_name, digest)
    if args.mode == MODE_INTEGRATOR and args.cache_sha256 != FORMAL_LOCKED_CACHE_SHA256:
        raise ValueError(
            "integrator control must use the immutable formal cache SHA-256"
        )
    if args.source_archive_sha256 != FORMAL_SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 differs from the preregistered source")
    if args.expected_source_commit != FORMAL_SOURCE_COMMIT:
        raise ValueError("source commit differs from the preregistered source")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("max-samples must be positive")
    for lag in args.taus:
        for horizon in args.horizons:
            integer_depth(horizon, lag)
    substeps_for_rhs_budget(args.integrator, args.equal_rhs_per_base)


@torch.no_grad()
def evaluate_grid(
    model: torch.nn.Module,
    initial_states: torch.Tensor,
    references: Mapping[float, torch.Tensor],
    *,
    mode: str,
    seed: int,
    model_name: str,
    integrator: str,
    taus: Sequence[float],
    horizons: Sequence[float],
    fixed_substeps: int = FIXED_SUBSTEPS,
    equal_rhs_per_base: int = EQUAL_RHS_PER_BASE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate one checkpoint/integrator cell with shared composed rollouts."""

    method = normalize_integrator(integrator)
    equal_base_substeps = substeps_for_rhs_budget(method, equal_rhs_per_base)
    ordered_horizons = tuple(sorted(float(value) for value in horizons))
    cells: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for lag in tuple(float(value) for value in taus):
        depth_to_horizon = {
            integer_depth(horizon, lag): horizon for horizon in ordered_horizons
        }
        if len(depth_to_horizon) != len(ordered_horizons):
            raise ValueError(f"duplicate composition depths for lag={lag}")
        max_depth = max(depth_to_horizon)
        # RK4's preregistered 120-RHS equal-work call is also 30 substeps, so
        # its composed trajectory is byte-for-byte the production trajectory.
        # Cache by executable substep count instead of evaluating it twice.
        composed_snapshots: dict[int, dict[int, torch.Tensor]] = {}
        for semantics, base_substeps in (
            (WORK_FIXED, int(fixed_substeps)),
            (WORK_EQUAL, int(equal_base_substeps)),
        ):
            if base_substeps not in composed_snapshots:
                composed = initial_states
                snapshots: dict[int, torch.Tensor] = {}
                for depth in range(1, max_depth + 1):
                    composed = evolve_latent_flow(
                        model,
                        composed,
                        lag,
                        method=method,
                        substeps=base_substeps,
                        conditioning_time=lag,
                    )
                    if depth in depth_to_horizon:
                        snapshots[depth] = composed
                composed_snapshots[base_substeps] = snapshots
            snapshots = composed_snapshots[base_substeps]

            for depth, horizon in sorted(depth_to_horizon.items()):
                direct_substeps = (
                    int(fixed_substeps)
                    if semantics == WORK_FIXED
                    else depth * int(equal_base_substeps)
                )
                direct = evolve_latent_flow(
                    model,
                    initial_states,
                    horizon,
                    method=method,
                    substeps=direct_substeps,
                    conditioning_time=horizon,
                )
                composed_state = snapshots[depth]
                prediction = _pair_metrics(composed_state, references[horizon])
                composition = _pair_metrics(composed_state, direct)
                direct_work = integration_work(method, direct_substeps)
                one_call_work = integration_work(method, base_substeps)
                composed_substeps = depth * base_substeps
                composed_rhs = depth * int(one_call_work["rhs_evaluations"])
                direct_rhs = int(direct_work["rhs_evaluations"])
                rhs_equal = direct_rhs == composed_rhs
                if semantics == WORK_EQUAL and not rhs_equal:
                    raise RuntimeError(
                        "equal-work accounting mismatch: "
                        f"direct={direct_rhs}, composed={composed_rhs}"
                    )
                cell = {
                    "mode": mode,
                    "seed": int(seed),
                    "model": model_name,
                    "integrator": method,
                    "work_semantics": semantics,
                    "lag": lag,
                    "horizon": horizon,
                    "composition_depth": depth,
                    "work": {
                        "rhs_evaluations_per_substep": one_call_work[
                            "rhs_evaluations_per_substep"
                        ],
                        "base_call_substeps": base_substeps,
                        "direct_substeps": direct_substeps,
                        "composed_substeps": composed_substeps,
                        "direct_rhs_evaluations": direct_rhs,
                        "composed_rhs_evaluations": composed_rhs,
                        "rhs_work_equal": rhs_equal,
                    },
                    "prediction_vs_reference": prediction,
                    "composition_composed_vs_direct": composition,
                }
                cells.append(cell)
                rows.append(
                    {
                        "mode": mode,
                        "seed": int(seed),
                        "model": model_name,
                        "integrator": method,
                        "work_semantics": semantics,
                        "lag": lag,
                        "horizon": horizon,
                        "composition_depth": depth,
                        "direct_substeps": direct_substeps,
                        "composed_substeps": composed_substeps,
                        "direct_rhs_evaluations": direct_rhs,
                        "composed_rhs_evaluations": composed_rhs,
                        "rhs_work_equal": rhs_equal,
                        "prediction_mse_mean": prediction["mse"]["mean"],
                        "prediction_relative_l2_mean": prediction["relative_l2"][
                            "mean"
                        ],
                        "composition_mse_mean": composition["mse"]["mean"],
                        "composition_relative_l2_mean": composition["relative_l2"][
                            "mean"
                        ],
                        "prediction_nonfinite_samples": prediction["nonfinite_samples"],
                        "composition_nonfinite_samples": composition[
                            "nonfinite_samples"
                        ],
                    }
                )
    return cells, rows


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    _validate_protocol_args(args)
    checkpoint_path = require_input_file(args.checkpoint, "checkpoint")
    cache_path = require_input_file(args.test_cache, "Fisher test cache")
    source_archive_path = require_input_file(args.source_archive, "source archive")
    root = reserve_exploratory_root(
        args.output_dir,
        input_paths=(checkpoint_path, cache_path, source_archive_path),
    )
    started_at = time.time()
    evaluator_sources = _runtime_source_hashes()
    source_archive_sha_before = sha256_file(source_archive_path)
    if source_archive_sha_before != args.source_archive_sha256:
        raise ValueError(
            "source archive SHA-256 mismatch: "
            f"{source_archive_sha_before} != {args.source_archive_sha256}"
        )
    launch_manifest = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "running",
        "created_at_unix": started_at,
        "git_commit": git_commit(REPOSITORY_ROOT),
        "config": vars(args),
        "source_commit": args.expected_source_commit,
        "source_archive_sha256": args.source_archive_sha256,
        "evaluator_source_hashes": evaluator_sources,
        "training_source_hashes": runtime_training_source_hashes(),
        "protocol": {
            "checkpoint_only": True,
            "training": False,
            "fine_tuning": False,
            "optimization": False,
            "checkpoint_selection": False,
            "cache_generation": False,
            "cache_overwrite": False,
            "input_writes": False,
        },
    }
    atomic_write_json(root / "exploratory_manifest.json", launch_manifest)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    set_global_seed(int(args.seed), deterministic=bool(args.deterministic))
    frozen_checkpoint = load_frozen_fisher_checkpoint(
        checkpoint_path,
        expected_sha256=args.checkpoint_sha256,
        model_name=args.model,
        seed=int(args.seed),
        expected_source_commit=args.expected_source_commit,
    )
    cache, cache_validation, cache_sha_before = load_frozen_fisher_cache(
        cache_path,
        expected_sha256=args.cache_sha256,
        profile=args.cache_profile,
        taus=args.taus,
        horizons=args.horizons,
        max_samples=args.max_samples,
    )
    sample_count = int(cache_validation["evaluated_samples"])
    initial_states = cache["test_u0"][:sample_count].to(device)
    references = {
        float(horizon): reference_states_at(
            cache,
            horizon=float(horizon),
            reference_dt=float(cache_validation["reference_dt"]),
            sample_count=sample_count,
        ).to(device)
        for horizon in args.horizons
    }
    model = frozen_checkpoint.model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    cells, rows = evaluate_grid(
        model,
        initial_states,
        references,
        mode=args.mode,
        seed=int(args.seed),
        model_name=args.model,
        integrator=args.integrator,
        taus=args.taus,
        horizons=args.horizons,
        fixed_substeps=int(args.fixed_substeps),
        equal_rhs_per_base=int(args.equal_rhs_per_base),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    checkpoint_sha_after = verify_unchanged(
        checkpoint_path, frozen_checkpoint.sha256, "checkpoint"
    )
    cache_sha_after = verify_unchanged(cache_path, cache_sha_before, "test cache")
    source_archive_sha_after = verify_unchanged(
        source_archive_path, source_archive_sha_before, "source archive"
    )
    equal_cells = [cell for cell in cells if cell["work_semantics"] == WORK_EQUAL]
    equal_work_all_matched = all(cell["work"]["rhs_work_equal"] for cell in equal_cells)
    nonfinite_samples = sum(
        int(cell["prediction_vs_reference"]["nonfinite_samples"])
        + int(cell["composition_composed_vs_direct"]["nonfinite_samples"])
        for cell in cells
    )
    smoke_only = args.max_samples is not None and sample_count < int(
        cache_validation["cached_samples"]
    )
    result = {
        **launch_manifest,
        "status": "passed_smoke" if smoke_only else "passed",
        "normal_exit": True,
        "smoke_only": smoke_only,
        "sample_count": sample_count,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_memory,
        "environment": device_provenance(device),
        "inputs": {
            "checkpoint": {
                **frozen_checkpoint.provenance(),
                "sha256_before": frozen_checkpoint.sha256,
                "sha256_after": checkpoint_sha_after,
                "immutable_during_evaluation": True,
            },
            "test_cache": {
                "path": str(cache_path),
                "sha256_before": cache_sha_before,
                "sha256_after": cache_sha_after,
                "immutable_during_evaluation": True,
                "validation": cache_validation,
            },
            "source_archive": {
                "path": str(source_archive_path),
                "sha256_before": source_archive_sha_before,
                "sha256_after": source_archive_sha_after,
                "immutable_during_evaluation": True,
            },
        },
        "summary": {
            "cell_count": len(cells),
            "equal_work_cell_count": len(equal_cells),
            "equal_work_all_rhs_matched": equal_work_all_matched,
            "nonfinite_sample_events": nonfinite_samples,
        },
        "cells": cells,
    }
    atomic_write_json(root / "results.json", result)
    atomic_write_csv(root / "metrics.csv", rows, fieldnames=CSV_FIELDS)
    completed_manifest = dict(launch_manifest)
    completed_manifest.update(
        {
            "status": result["status"],
            "normal_exit": True,
            "smoke_only": smoke_only,
            "sample_count": sample_count,
            "cell_count": len(cells),
        }
    )
    atomic_write_json(root / "exploratory_manifest.json", completed_manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": result["status"],
        "normal_exit": True,
        "smoke_only": smoke_only,
        "mode": args.mode,
        "seed": int(args.seed),
        "model": args.model,
        "integrator": args.integrator,
        "sample_count": sample_count,
        "cell_count": len(cells),
        "equal_work_all_rhs_matched": equal_work_all_matched,
        "nonfinite_sample_events": nonfinite_samples,
        "input_hashes": {
            "checkpoint": {
                "before": frozen_checkpoint.sha256,
                "after": checkpoint_sha_after,
                "unchanged": frozen_checkpoint.sha256 == checkpoint_sha_after,
            },
            "test_cache": {
                "before": cache_sha_before,
                "after": cache_sha_after,
                "unchanged": cache_sha_before == cache_sha_after,
            },
            "source_archive": {
                "before": source_archive_sha_before,
                "after": source_archive_sha_after,
                "unchanged": source_archive_sha_before == source_archive_sha_after,
            },
        },
        "results_sha256": sha256_file(root / "results.json"),
        "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
        "manifest_sha256": sha256_file(root / "exploratory_manifest.json"),
        "evaluator_source_hashes": evaluator_sources,
    }
    if not equal_work_all_matched:
        raise RuntimeError("one or more equal-work cells have mismatched RHS counts")
    atomic_write_json(root / "receipt.json", receipt)
    (root / "done").write_text(f"{result['status']}\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(MODE_INTEGRATOR, MODE_PHASE), required=True)
    parser.add_argument(
        "--model", choices=("latent", "latent_query_time"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument(
        "--source-archive-sha256",
        default=FORMAL_SOURCE_ARCHIVE_SHA256,
    )
    parser.add_argument("--expected-source-commit", default=FORMAL_SOURCE_COMMIT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--integrator", choices=("euler", "rk2", "rk4"), required=True)
    parser.add_argument("--fixed-substeps", type=int, default=FIXED_SUBSTEPS)
    parser.add_argument("--equal-rhs-per-base", type=int, default=EQUAL_RHS_PER_BASE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--taus", type=_parse_positive_float_list, default=None)
    parser.add_argument("--horizons", type=_parse_positive_float_list, default=None)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.taus is None or args.horizons is None:
        _cache_profile, default_taus, default_horizons = _grid_for_mode(args.mode)
        if args.taus is None:
            args.taus = default_taus
        if args.horizons is None:
            args.horizons = default_horizons
    return args


def main(argv: list[str] | None = None) -> None:
    args = normalize_args(build_parser().parse_args(argv))
    result = run_evaluation(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "mode": args.mode,
                "seed": args.seed,
                "model": args.model,
                "integrator": args.integrator,
                "sample_count": result["sample_count"],
                "summary": result["summary"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
