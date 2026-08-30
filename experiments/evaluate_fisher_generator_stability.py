#!/usr/bin/env python3
"""Checkpoint-only Fisher generator, stability, and RK4 refinement evaluator.

One invocation evaluates one already selected formal Fisher A or B checkpoint.
It reads the immutable locked cache and source archive, creates one brand-new
exploratory root, and never trains, tunes, selects, or modifies an input.
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
    load_frozen_fisher_cache,
    load_frozen_fisher_checkpoint,
    runtime_training_source_hashes,
)
from fisher_generator_metrics import (  # noqa: E402
    autonomy_sensitivity_metrics,
    generator_residual_metrics,
    refinement_metrics,
    repeated_lag_snapshots,
    snapshot_states,
    stability_pair_metrics,
)
from seed_utils import set_global_seed  # noqa: E402


SCHEMA_VERSION = 1
TRACK = "fisher_generator_stability"
TAUS = (0.075, 0.15)
REFERENCE_TIMES = (0.0, 0.6, 1.2, 2.4, 4.8)
PAIR_FAMILIES = (
    "cyclic_cross_sample",
    "sinusoidal_0.001",
    "sinusoidal_0.01",
)
LEARNED_SAMPLES = 64
REFINEMENT_SAMPLES = 32
REFINEMENT_HORIZON = 1.2
REFINEMENT_SUBSTEPS = (8, 16, 30, 60)
FINEST_SUBSTEPS = 120
PRODUCTION_SUBSTEPS = 30
REFERENCE_OSL_TOLERANCE = 1e-4
AUTONOMY_TOLERANCE = 1e-7

GENERATOR_CSV_FIELDS = (
    "seed",
    "model",
    "state_source",
    "snapshot_time",
    "conditioning_time",
    "sample_count",
    "rms_residual_l2",
    "rms_reference_l2",
    "rms_relative_residual",
    "residual_mean",
    "residual_median",
    "residual_p90",
    "residual_p95",
    "residual_max",
    "relative_mean",
    "relative_median",
    "relative_p90",
    "relative_p95",
    "relative_max",
    "cosine_mean",
    "representation_error_max",
    "latent_abs_max",
    "raw_state_min",
    "raw_state_max",
    "raw_bound_violation_count",
    "represented_state_min",
    "represented_state_max",
    "saturation_count",
    "nonfinite_events",
)

STABILITY_CSV_FIELDS = (
    "seed",
    "model",
    "snapshot_time",
    "conditioning_time",
    "pair_family",
    "sample_count",
    "reference_mean",
    "reference_median",
    "reference_p95",
    "reference_max",
    "learned_mean",
    "learned_median",
    "learned_p95",
    "learned_max",
    "pair_distance_min",
    "pair_distance_median",
    "pair_distance_max",
    "reference_bound",
    "reference_bound_pass",
    "nonfinite_events",
)

REFINEMENT_CSV_FIELDS = (
    "seed",
    "model",
    "conditioning_time",
    "horizon",
    "composition_depth",
    "substeps_per_call",
    "finest_substeps_per_call",
    "total_substeps",
    "total_rhs_evaluations",
    "rms_learned_flow_difference_vs_finest_l2",
    "relative_error_mean",
    "relative_error_p95",
    "relative_error_max",
    "rms_learned_prediction_discrepancy_vs_cache_reference_l2",
    "observed_order_to_next",
    "next_substeps_per_call",
    "saturation_count",
    "saturation_fraction",
    "max_abs_latent",
    "encoder_clamp_activation_count",
    "encoder_clamp_activation_fraction",
    "reencoding_abs_max",
    "reencoding_rms",
    "learned_flow_numerical_to_cache_discrepancy_ratio",
    "learned_flow_numerical_confounding",
)


def _parse_float_list(value: str, *, allow_zero: bool = False) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated finite floats"
        ) from exc
    lower_ok = (lambda item: item >= 0) if allow_zero else (lambda item: item > 0)
    if not values or any(
        not math.isfinite(item) or not lower_ok(item) for item in values
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise argparse.ArgumentTypeError(f"all values must be finite and {qualifier}")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def _parse_nonnegative_float_list(value: str) -> tuple[float, ...]:
    return _parse_float_list(value, allow_zero=True)


def _parse_positive_float_list(value: str) -> tuple[float, ...]:
    return _parse_float_list(value, allow_zero=False)


def _parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        raw = tuple(item.strip() for item in value.split(",") if item.strip())
        parsed = tuple(int(item) for item in raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers"
        ) from exc
    if not parsed or any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError("all integer values must be positive")
    if tuple(str(value) for value in parsed) != raw:
        raise argparse.ArgumentTypeError(
            "integer values must use canonical decimal syntax"
        )
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("integer values must not contain duplicates")
    return parsed


def _runtime_source_hashes() -> dict[str, str]:
    return source_hashes(
        {
            "evaluate_fisher_generator_stability.py": Path(__file__).resolve(),
            "fisher_generator_metrics.py": EXPERIMENTS_DIR
            / "fisher_generator_metrics.py",
            "experiment_artifacts.py": EXPERIMENTS_DIR / "experiment_artifacts.py",
            "fisher_frozen_inputs.py": EXPERIMENTS_DIR / "fisher_frozen_inputs.py",
            "latent_integrators.py": EXPERIMENTS_DIR / "latent_integrators.py",
            "models.py": EXPERIMENTS_DIR / "models.py",
            "seed_utils.py": EXPERIMENTS_DIR / "seed_utils.py",
            "FISHER_GENERATOR_STABILITY_PROTOCOL.md": REPOSITORY_ROOT
            / "docs/research/FISHER_GENERATOR_STABILITY_PROTOCOL.md",
            "FISHER_KPP_THEOREM_CARD.md": REPOSITORY_ROOT
            / "docs/research/FISHER_KPP_THEOREM_CARD.md",
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("latent", "latent_query_time"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--cache-sha256", default=FORMAL_LOCKED_CACHE_SHA256)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--source-archive-sha256", default=FORMAL_SOURCE_ARCHIVE_SHA256)
    parser.add_argument("--expected-source-commit", default=FORMAL_SOURCE_COMMIT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--taus", type=_parse_positive_float_list, default=TAUS)
    parser.add_argument(
        "--reference-times",
        type=_parse_nonnegative_float_list,
        default=REFERENCE_TIMES,
    )
    parser.add_argument("--learned-samples", type=int, default=LEARNED_SAMPLES)
    parser.add_argument("--refinement-samples", type=int, default=REFINEMENT_SAMPLES)
    parser.add_argument("--refinement-horizon", type=float, default=REFINEMENT_HORIZON)
    parser.add_argument(
        "--refinement-substeps",
        type=_parse_positive_int_list,
        default=REFINEMENT_SUBSTEPS,
    )
    parser.add_argument("--finest-substeps", type=int, default=FINEST_SUBSTEPS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if int(args.seed) not in {31415, 271828, 161803}:
        raise ValueError("seed must be one of the three frozen formal seeds")
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive")
    if int(args.learned_samples) < 2 or int(args.refinement_samples) < 1:
        raise ValueError("learned-samples must be >=2 and refinement-samples >=1")
    if args.max_samples is not None and int(args.max_samples) < 2:
        raise ValueError("max-samples must be at least two")
    if float(args.refinement_horizon) <= 0:
        raise ValueError("refinement-horizon must be positive")
    if int(args.finest_substeps) <= max(
        int(value) for value in args.refinement_substeps
    ):
        raise ValueError("finest-substeps must exceed every refinement level")
    if len(tuple(args.taus)) != 2:
        raise ValueError("exactly two query lags are required")
    if 0.0 not in tuple(float(value) for value in args.reference_times):
        raise ValueError("reference-times must include time zero")
    divisibility_horizons = tuple(
        float(value) for value in args.reference_times if float(value) > 0
    ) + (float(args.refinement_horizon),)
    for horizon in divisibility_horizons:
        for lag in args.taus:
            ratio = horizon / float(lag)
            if not math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"horizon {horizon} must be divisible by query lag {lag}"
                )
    if PRODUCTION_SUBSTEPS not in tuple(
        int(value) for value in args.refinement_substeps
    ):
        if not args.smoke:
            raise ValueError(
                "the full refinement grid must contain production value 30"
            )
    for name in ("checkpoint_sha256", "cache_sha256", "source_archive_sha256"):
        digest = str(getattr(args, name)).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{name} must be a 64-character SHA-256 digest")
        setattr(args, name, digest)
    if args.cache_sha256 != FORMAL_LOCKED_CACHE_SHA256:
        raise ValueError("this protocol requires the immutable formal Fisher cache")
    if args.source_archive_sha256 != FORMAL_SOURCE_ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 differs from the frozen source")
    if args.expected_source_commit != FORMAL_SOURCE_COMMIT:
        raise ValueError("source commit differs from the frozen source")

    if not args.smoke:
        frozen = {
            "taus": (tuple(args.taus), TAUS),
            "reference_times": (tuple(args.reference_times), REFERENCE_TIMES),
            "learned_samples": (int(args.learned_samples), LEARNED_SAMPLES),
            "refinement_samples": (
                int(args.refinement_samples),
                REFINEMENT_SAMPLES,
            ),
            "refinement_horizon": (
                float(args.refinement_horizon),
                REFINEMENT_HORIZON,
            ),
            "refinement_substeps": (
                tuple(args.refinement_substeps),
                REFINEMENT_SUBSTEPS,
            ),
            "finest_substeps": (int(args.finest_substeps), FINEST_SUBSTEPS),
            "max_samples": (args.max_samples, None),
        }
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in frozen.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(
                "full diagnostic settings differ from the frozen protocol: "
                + json.dumps(mismatches, sort_keys=True)
            )


def _flatten_generator_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    residual = metrics["residual_l2"]
    relative = metrics["relative_residual_l2"]
    cosine = metrics["cosine_alignment"]
    return {
        "seed": row["seed"],
        "model": row["model"],
        "state_source": row["state_source"],
        "snapshot_time": row["snapshot_time"],
        "conditioning_time": row["conditioning_time"],
        "sample_count": metrics["sample_count"],
        "rms_residual_l2": metrics["rms_residual_l2"],
        "rms_reference_l2": metrics["rms_reference_l2"],
        "rms_relative_residual": metrics["rms_relative_residual"],
        "residual_mean": residual["mean"],
        "residual_median": residual["median"],
        "residual_p90": residual["p90"],
        "residual_p95": residual["p95"],
        "residual_max": residual["max"],
        "relative_mean": relative["mean"],
        "relative_median": relative["median"],
        "relative_p90": relative["p90"],
        "relative_p95": relative["p95"],
        "relative_max": relative["max"],
        "cosine_mean": cosine["mean"],
        "representation_error_max": metrics["representation_error_l2"]["max"],
        "latent_abs_max": metrics["latent_abs_max"]["max"],
        "raw_state_min": metrics["raw_state_min"],
        "raw_state_max": metrics["raw_state_max"],
        "raw_bound_violation_count": metrics["raw_bound_violation_count"],
        "represented_state_min": metrics["represented_state_min"],
        "represented_state_max": metrics["represented_state_max"],
        "saturation_count": metrics["saturation_count"],
        "nonfinite_events": metrics["nonfinite_events"],
    }


def _flatten_stability_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row["metrics"]
    reference = metrics["reference_quotient"]
    learned = metrics["learned_quotient"]
    distance = metrics["pair_distance_l2"]
    return {
        "seed": row["seed"],
        "model": row["model"],
        "snapshot_time": row["snapshot_time"],
        "conditioning_time": row["conditioning_time"],
        "pair_family": row["pair_family"],
        "sample_count": metrics["sample_count"],
        "reference_mean": reference["mean"],
        "reference_median": reference["median"],
        "reference_p95": reference["p95"],
        "reference_max": reference["max"],
        "learned_mean": learned["mean"],
        "learned_median": learned["median"],
        "learned_p95": learned["p95"],
        "learned_max": learned["max"],
        "pair_distance_min": distance["min"],
        "pair_distance_median": distance["median"],
        "pair_distance_max": distance["max"],
        "reference_bound": row["reference_bound"],
        "reference_bound_pass": row["reference_bound_pass"],
        "nonfinite_events": metrics["nonfinite_events"],
    }


def _flatten_refinement_row(
    row: Mapping[str, Any], summary: Mapping[str, Any], *, seed: int, model: str
) -> dict[str, Any]:
    relative = row["relative_learned_flow_difference_vs_finest_l2"]
    is_production = int(row["substeps_per_call"]) == PRODUCTION_SUBSTEPS
    return {
        "seed": seed,
        "model": model,
        "conditioning_time": row["lag"],
        "horizon": row["horizon"],
        "composition_depth": row["composition_depth"],
        "substeps_per_call": row["substeps_per_call"],
        "finest_substeps_per_call": row["finest_substeps_per_call"],
        "total_substeps": row["total_substeps"],
        "total_rhs_evaluations": row["total_rhs_evaluations"],
        "rms_learned_flow_difference_vs_finest_l2": row[
            "rms_learned_flow_difference_vs_finest_l2"
        ],
        "relative_error_mean": relative["mean"],
        "relative_error_p95": relative["p95"],
        "relative_error_max": relative["max"],
        "rms_learned_prediction_discrepancy_vs_cache_reference_l2": row[
            "rms_learned_prediction_discrepancy_vs_cache_reference_l2"
        ],
        "observed_order_to_next": row["observed_order_to_next"],
        "next_substeps_per_call": row["next_substeps_per_call"],
        "saturation_count": row["saturation_count"],
        "saturation_fraction": row["saturation_fraction"],
        "max_abs_latent": row["max_abs_latent"],
        "encoder_clamp_activation_count": row["encoder_clamp_activation_count"],
        "encoder_clamp_activation_fraction": row["encoder_clamp_activation_fraction"],
        "reencoding_abs_max": row["reencoding_abs_max"],
        "reencoding_rms": row["reencoding_rms"],
        "learned_flow_numerical_to_cache_discrepancy_ratio": (
            summary["learned_flow_numerical_to_cache_discrepancy_ratio"]
            if is_production
            else None
        ),
        "learned_flow_numerical_confounding": (
            summary["learned_flow_numerical_confounding"] if is_production else None
        ),
    }


def _positive_reference_times(
    values: Sequence[float], refinement_horizon: float
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {float(value) for value in values if float(value) > 0}
            | {float(refinement_horizon)}
        )
    )


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    checkpoint_path = require_input_file(args.checkpoint, "checkpoint")
    cache_path = require_input_file(args.test_cache, "Fisher locked test cache")
    source_archive_path = require_input_file(args.source_archive, "source archive")
    root = reserve_exploratory_root(
        args.output_dir,
        input_paths=(checkpoint_path, cache_path, source_archive_path),
    )
    started_at = time.time()
    reservation_manifest = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "running",
        "created_at_unix": started_at,
        "config": vars(args),
    }
    atomic_write_json(root / "exploratory_manifest.json", reservation_manifest)
    evaluator_sources = _runtime_source_hashes()
    source_archive_sha_before = sha256_file(source_archive_path)

    launch_manifest = {
        **reservation_manifest,
        "git_commit": git_commit(REPOSITORY_ROOT),
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
            "finite_sample_diagnostic_not_uniform_bound": True,
        },
    }
    atomic_write_json(root / "exploratory_manifest.json", launch_manifest)
    if source_archive_sha_before != args.source_archive_sha256:
        raise ValueError("source archive SHA-256 differs from the frozen launch input")

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
    positive_times = _positive_reference_times(
        args.reference_times, args.refinement_horizon
    )
    cache, cache_validation, cache_sha_before = load_frozen_fisher_cache(
        cache_path,
        expected_sha256=args.cache_sha256,
        profile=FORMAL_CACHE_PROFILE,
        taus=args.taus,
        horizons=positive_times,
        max_samples=args.max_samples,
    )
    sample_count = int(cache_validation["evaluated_samples"])
    learned_samples = min(int(args.learned_samples), sample_count)
    refinement_samples = min(int(args.refinement_samples), sample_count)
    if learned_samples < 2:
        raise ValueError(
            "evaluated cache subset must contain at least two learned samples"
        )
    model = frozen_checkpoint.model.to(device).eval()
    length = float(cache_validation["config"]["L"])
    diffusivity = float(cache_validation["config"]["nu"])
    reaction_rate = float(cache_validation["config"]["reaction_rate"])
    reference_dt = float(cache_validation["reference_dt"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    evaluation_started = time.perf_counter()
    reference_snapshots = {
        float(snapshot_time): snapshot_states(
            cache,
            time_value=float(snapshot_time),
            reference_dt=reference_dt,
            sample_count=sample_count,
        )
        for snapshot_time in args.reference_times
    }

    generator_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    learned_path_execution: list[dict[str, Any]] = []
    nonfinite_events = 0
    saturation_count = 0
    encoder_clamp_activation_count = 0
    reference_bound = reaction_rate + REFERENCE_OSL_TOLERANCE

    for snapshot_time, cpu_states in reference_snapshots.items():
        states = cpu_states.to(device)
        for conditioning_time in args.taus:
            metrics = generator_residual_metrics(
                model,
                states,
                conditioning_time=float(conditioning_time),
                length=length,
                diffusivity=diffusivity,
                reaction_rate=reaction_rate,
                batch_size=int(args.batch_size),
            )
            generator_rows.append(
                {
                    "seed": int(args.seed),
                    "model": args.model,
                    "state_source": "reference_trajectory",
                    "snapshot_time": snapshot_time,
                    "conditioning_time": float(conditioning_time),
                    "metrics": metrics,
                }
            )
            nonfinite_events += int(metrics["nonfinite_events"])
            saturation_count += int(metrics["saturation_count"])
            for pair_family in PAIR_FAMILIES:
                pair_metrics = stability_pair_metrics(
                    model,
                    states,
                    pair_family=pair_family,
                    conditioning_time=float(conditioning_time),
                    length=length,
                    diffusivity=diffusivity,
                    reaction_rate=reaction_rate,
                    batch_size=int(args.batch_size),
                )
                reference_max = pair_metrics["reference_quotient"]["max"]
                bound_pass = reference_max is not None and float(
                    reference_max
                ) <= float(reference_bound)
                stability_rows.append(
                    {
                        "seed": int(args.seed),
                        "model": args.model,
                        "snapshot_time": snapshot_time,
                        "conditioning_time": float(conditioning_time),
                        "pair_family": pair_family,
                        "reference_bound": reference_bound,
                        "reference_bound_pass": bound_pass,
                        "metrics": pair_metrics,
                    }
                )
                nonfinite_events += int(pair_metrics["nonfinite_events"])

    initial_learned = reference_snapshots[0.0][:learned_samples].to(device)
    learned_horizons = tuple(
        float(value) for value in args.reference_times if float(value) > 0
    )
    for conditioning_time in args.taus:
        learned_snapshots, execution = repeated_lag_snapshots(
            model,
            initial_learned,
            lag=float(conditioning_time),
            horizons=learned_horizons,
            substeps_per_call=PRODUCTION_SUBSTEPS,
        )
        learned_path_execution.append(execution)
        saturation_count += int(execution["saturation_count"])
        encoder_clamp_activation_count += int(
            execution["encoder_clamp_activation_count"]
        )
        all_learned_snapshots = {0.0: initial_learned, **learned_snapshots}
        for snapshot_time, states in sorted(all_learned_snapshots.items()):
            metrics = generator_residual_metrics(
                model,
                states,
                conditioning_time=float(conditioning_time),
                length=length,
                diffusivity=diffusivity,
                reaction_rate=reaction_rate,
                batch_size=int(args.batch_size),
            )
            generator_rows.append(
                {
                    "seed": int(args.seed),
                    "model": args.model,
                    "state_source": "learned_repeated_lag_trajectory",
                    "snapshot_time": snapshot_time,
                    "conditioning_time": float(conditioning_time),
                    "metrics": metrics,
                }
            )
            nonfinite_events += int(metrics["nonfinite_events"])
            saturation_count += int(metrics["saturation_count"])

    autonomy_states = reference_snapshots[0.0][:learned_samples].to(device)
    autonomy = autonomy_sensitivity_metrics(
        model,
        autonomy_states,
        conditioning_times=args.taus,
        length=length,
    )
    autonomy_required = args.model == "latent"
    autonomy_pass = (
        not autonomy_required
        or float(autonomy["max_difference_l2"]) <= AUTONOMY_TOLERANCE
    )

    refinement_summaries: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []
    refinement_initial = reference_snapshots[0.0][:refinement_samples].to(device)
    refinement_reference = snapshot_states(
        cache,
        time_value=float(args.refinement_horizon),
        reference_dt=reference_dt,
        sample_count=refinement_samples,
    ).to(device)
    production_substeps = (
        PRODUCTION_SUBSTEPS
        if PRODUCTION_SUBSTEPS in tuple(args.refinement_substeps)
        else int(args.refinement_substeps[-1])
    )
    for conditioning_time in args.taus:
        refinement_summary, rows = refinement_metrics(
            model,
            refinement_initial,
            refinement_reference,
            lag=float(conditioning_time),
            horizon=float(args.refinement_horizon),
            substeps=args.refinement_substeps,
            finest_substeps=int(args.finest_substeps),
            production_substeps=production_substeps,
            length=length,
        )
        refinement_summaries.append(refinement_summary)
        refinement_rows.extend(rows)
        saturation_count += int(refinement_summary["finest_saturation_count"])
        encoder_clamp_activation_count += int(
            refinement_summary["finest_encoder_clamp_activation_count"]
        )
        saturation_count += sum(int(row["saturation_count"]) for row in rows)
        encoder_clamp_activation_count += sum(
            int(row["encoder_clamp_activation_count"]) for row in rows
        )
        nonfinite_events += sum(
            int(row["learned_flow_difference_vs_finest_l2"]["nonfinite_count"])
            + int(
                row["relative_learned_flow_difference_vs_finest_l2"]["nonfinite_count"]
            )
            + int(
                row["learned_prediction_discrepancy_vs_cache_reference_l2"][
                    "nonfinite_count"
                ]
            )
            for row in rows
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - evaluation_started
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )

    checkpoint_sha_after = verify_unchanged(
        checkpoint_path, frozen_checkpoint.sha256, "checkpoint"
    )
    cache_sha_after = verify_unchanged(cache_path, cache_sha_before, "test cache")
    source_archive_sha_after = verify_unchanged(
        source_archive_path, source_archive_sha_before, "source archive"
    )
    reference_osl_pass = all(
        bool(row["reference_bound_pass"]) for row in stability_rows
    )
    saturation_pass = saturation_count == 0
    encoder_clamp_pass = encoder_clamp_activation_count == 0
    finite_pass = nonfinite_events == 0
    engineering_pass = (
        reference_osl_pass
        and autonomy_pass
        and saturation_pass
        and encoder_clamp_pass
        and finite_pass
    )
    smoke_only = bool(args.smoke)
    status = (
        "passed_smoke"
        if smoke_only and engineering_pass
        else "passed"
        if engineering_pass
        else "failed_engineering"
    )

    result = {
        **launch_manifest,
        "status": status,
        "normal_exit": engineering_pass,
        "smoke_only": smoke_only,
        "sample_count": sample_count,
        "learned_sample_count": learned_samples,
        "refinement_sample_count": refinement_samples,
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_memory_bytes,
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
        "diagnostics": {
            "generator_residual": generator_rows,
            "one_sided_stability": stability_rows,
            "autonomy_sensitivity": autonomy,
            "learned_path_execution": learned_path_execution,
            "rk4_refinement": {
                "summaries": refinement_summaries,
                "rows": refinement_rows,
                "finest_is_surrogate_not_exact": True,
            },
        },
        "summary": {
            "generator_row_count": len(generator_rows),
            "stability_row_count": len(stability_rows),
            "refinement_row_count": len(refinement_rows),
            "reference_osl_bound": reference_bound,
            "reference_osl_pass": reference_osl_pass,
            "autonomy_required": autonomy_required,
            "autonomy_tolerance": AUTONOMY_TOLERANCE,
            "autonomy_pass": autonomy_pass,
            "saturation_count": saturation_count,
            "saturation_pass": saturation_pass,
            "encoder_clamp_activation_count": encoder_clamp_activation_count,
            "encoder_clamp_pass": encoder_clamp_pass,
            "nonfinite_events": nonfinite_events,
            "finite_pass": finite_pass,
            "engineering_pass": engineering_pass,
            "learned_flow_numerical_non_dominant": all(
                not bool(item["learned_flow_numerical_confounding"])
                for item in refinement_summaries
            ),
        },
    }

    generator_csv_rows = [_flatten_generator_row(row) for row in generator_rows]
    stability_csv_rows = [_flatten_stability_row(row) for row in stability_rows]
    refinement_csv_rows: list[dict[str, Any]] = []
    summary_by_lag = {float(item["lag"]): item for item in refinement_summaries}
    for row in refinement_rows:
        refinement_csv_rows.append(
            _flatten_refinement_row(
                row,
                summary_by_lag[float(row["lag"])],
                seed=int(args.seed),
                model=args.model,
            )
        )

    atomic_write_json(root / "results.json", result)
    atomic_write_csv(
        root / "generator_metrics.csv",
        generator_csv_rows,
        fieldnames=GENERATOR_CSV_FIELDS,
    )
    atomic_write_csv(
        root / "stability_metrics.csv",
        stability_csv_rows,
        fieldnames=STABILITY_CSV_FIELDS,
    )
    atomic_write_csv(
        root / "refinement_metrics.csv",
        refinement_csv_rows,
        fieldnames=REFINEMENT_CSV_FIELDS,
    )
    completed_manifest = dict(launch_manifest)
    completed_manifest.update(
        {
            "status": status,
            "normal_exit": engineering_pass,
            "smoke_only": smoke_only,
            "sample_count": sample_count,
            "engineering_pass": engineering_pass,
        }
    )
    atomic_write_json(root / "exploratory_manifest.json", completed_manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": status,
        "normal_exit": engineering_pass,
        "smoke_only": smoke_only,
        "seed": int(args.seed),
        "model": args.model,
        "sample_count": sample_count,
        "engineering_gates": result["summary"],
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
        "artifact_hashes": {
            "results.json": sha256_file(root / "results.json"),
            "generator_metrics.csv": sha256_file(root / "generator_metrics.csv"),
            "stability_metrics.csv": sha256_file(root / "stability_metrics.csv"),
            "refinement_metrics.csv": sha256_file(root / "refinement_metrics.csv"),
            "exploratory_manifest.json": sha256_file(
                root / "exploratory_manifest.json"
            ),
        },
        "evaluator_source_hashes": evaluator_sources,
    }
    atomic_write_json(root / "receipt.json", receipt)
    if not engineering_pass:
        (root / "failed").write_text("failed_engineering\n", encoding="utf-8")
        raise RuntimeError(
            "Fisher generator/stability evaluator failed one or more engineering gates"
        )
    (root / "done").write_text(f"{status}\n", encoding="utf-8")
    return result


def _record_failed_exception(args: argparse.Namespace, exc: Exception) -> None:
    """Close a root that this evaluator reserved before an unexpected failure."""

    root = Path(args.output_dir).expanduser().resolve()
    manifest_path = root / "exploratory_manifest.json"
    receipt_path = root / "receipt.json"
    if not root.is_dir() or receipt_path.exists() or not manifest_path.is_file():
        return
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(manifest, dict) or manifest.get("track") != TRACK:
        return
    error = {"type": type(exc).__name__, "message": str(exc)}
    failed_manifest = dict(manifest)
    failed_manifest.update(
        {"status": "failed_exception", "normal_exit": False, "error": error}
    )
    atomic_write_json(manifest_path, failed_manifest)
    atomic_write_json(
        receipt_path,
        {
            "schema_version": SCHEMA_VERSION,
            "exploratory": True,
            "do_not_use_for_formal": True,
            "track": TRACK,
            "status": "failed_exception",
            "normal_exit": False,
            "smoke_only": bool(args.smoke),
            "seed": int(args.seed),
            "model": args.model,
            "error": error,
            "artifact_hashes": {
                "exploratory_manifest.json": sha256_file(manifest_path)
            },
        },
    )
    (root / "failed").write_text("failed_exception\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        result = run_evaluation(args)
    except Exception as exc:
        _record_failed_exception(args, exc)
        raise
    print(
        json.dumps(
            {
                "status": result["status"],
                "seed": args.seed,
                "model": args.model,
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
