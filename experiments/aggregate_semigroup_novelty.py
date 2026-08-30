#!/usr/bin/env python3
"""Independently aggregate the preregistered semigroup novelty experiments.

The aggregator is deliberately read-only with respect to experiment roots.  It
accepts only completed, non-smoke cells, re-hashes every JSON/CSV/manifest
artifact named by a receipt, verifies immutable-input evidence, and then
applies the frozen structural rules.  A scientific hypothesis may fail while
the experiment remains a valid, complete execution.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    git_commit,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
)


SCHEMA_VERSION = 1
TRACK = "semigroup_novelty_independent_aggregation"
SEEDS = (31415, 271828, 161803)
MODELS = ("latent", "latent_query_time")
INTEGRATORS = ("euler", "rk2", "rk4")
FORMAL_TAUS = (0.075, 0.15)
FORMAL_HORIZONS = (1.2, 2.4, 4.8)
PHASE_TAUS = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
PHASE_HORIZONS = (0.6, 1.2, 2.4, 4.8)
EQUAL_WORK = "equal_rhs_work"
GEOMETRIC_MEAN_FLOOR = 1e-30
EXPECTED_EVALUATOR_COMMIT = "3c46763a5b001dfae7faf3e28a0d1c832c8281cb"

INTEGRATOR_FIELDS = (
    "integrator",
    "seed",
    "a_defect_geometric_mean",
    "b_defect_geometric_mean",
    "a_over_b_defect",
    "a_lower_defect",
    "a_mse_geometric_mean",
    "b_mse_geometric_mean",
    "a_over_b_mse",
)
PHASE_FIELDS = (
    "seed",
    "lag",
    "horizon",
    "composition_depth",
    "a_defect",
    "b_defect",
    "a_over_b_defect",
    "a_lower_defect",
    "a_mse",
    "b_mse",
    "a_over_b_mse",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    _assert_finite_json(payload, label=str(path))
    return payload


def _assert_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number in {label}: {value!r}")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite_json(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_finite_json(item, label=label)


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_named_hash(
    run_dir: Path,
    receipt: Mapping[str, Any],
    *,
    file_name: str,
    receipt_key: str,
) -> str:
    path = _require_file(run_dir / file_name)
    observed = sha256_file(path)
    expected = str(receipt.get(receipt_key, ""))
    if observed != expected:
        raise ValueError(
            f"artifact hash mismatch for {path}: observed={observed}, expected={expected}"
        )
    return observed


def _verify_completion_files(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _load_json(_require_file(run_dir / "receipt.json"))
    result = _load_json(_require_file(run_dir / "results.json"))
    manifest = _load_json(_require_file(run_dir / "exploratory_manifest.json"))
    _require_file(run_dir / "metrics.csv")
    done = _require_file(run_dir / "done").read_text(encoding="utf-8").strip()
    if done != "passed" or result.get("status") != "passed":
        raise ValueError(f"run is not a completed full evaluation: {run_dir}")
    if receipt.get("status") != "passed" or manifest.get("status") != "passed":
        raise ValueError(f"receipt/manifest status mismatch: {run_dir}")
    if not result.get("normal_exit") or not receipt.get("normal_exit"):
        raise ValueError(f"run lacks normal-exit evidence: {run_dir}")
    if bool(result.get("smoke_only")) or bool(receipt.get("smoke_only")):
        raise ValueError(f"smoke output is not accepted by aggregation: {run_dir}")
    _verify_named_hash(
        run_dir, receipt, file_name="results.json", receipt_key="results_sha256"
    )
    _verify_named_hash(
        run_dir, receipt, file_name="metrics.csv", receipt_key="metrics_csv_sha256"
    )
    _verify_named_hash(
        run_dir,
        receipt,
        file_name="exploratory_manifest.json",
        receipt_key="manifest_sha256",
    )
    input_hashes = receipt.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError(f"missing immutable-input receipt: {run_dir}")
    for label in ("checkpoint", "test_cache", "source_archive"):
        record = input_hashes.get(label)
        if not isinstance(record, Mapping):
            raise ValueError(f"missing {label} hash record: {run_dir}")
        if record.get("before") != record.get("after") or not record.get("unchanged"):
            raise ValueError(f"input changed during evaluation ({label}): {run_dir}")
    if not receipt.get("equal_work_all_rhs_matched"):
        raise ValueError(f"equal-work accounting failed: {run_dir}")
    if int(receipt.get("nonfinite_sample_events", -1)) != 0:
        raise ValueError(f"non-finite sample event in {run_dir}")
    return result, receipt


def _verify_support_artifact(
    run_dir: Path,
    *,
    required_flags: Sequence[str],
) -> dict[str, Any]:
    receipt = _load_json(_require_file(run_dir / "receipt.json"))
    if receipt.get("status") != "passed" or not receipt.get("normal_exit"):
        raise ValueError(f"support artifact did not pass: {run_dir}")
    for flag in required_flags:
        if not receipt.get(flag):
            raise ValueError(f"support artifact is missing {flag}: {run_dir}")
    _verify_named_hash(
        run_dir, receipt, file_name="results.json", receipt_key="results_sha256"
    )
    _verify_named_hash(
        run_dir,
        receipt,
        file_name="exploratory_manifest.json",
        receipt_key="manifest_sha256",
    )
    if "metrics_csv_sha256" in receipt:
        _verify_named_hash(
            run_dir,
            receipt,
            file_name="metrics.csv",
            receipt_key="metrics_csv_sha256",
        )
    return receipt


def _cell_identity(cell: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        float(cell["lag"]),
        float(cell["horizon"]),
        str(cell["work_semantics"]),
    )


def _expected_cell_identities(
    taus: Sequence[float], horizons: Sequence[float]
) -> set[tuple[float, float, str]]:
    return {
        (float(lag), float(horizon), semantics)
        for lag in taus
        for horizon in horizons
        for semantics in ("production_fixed_substeps", EQUAL_WORK)
    }


def _verify_grid(
    result: Mapping[str, Any],
    *,
    taus: Sequence[float],
    horizons: Sequence[float],
) -> list[dict[str, Any]]:
    cells = result.get("cells")
    if not isinstance(cells, list):
        raise ValueError("result does not contain a cell list")
    identities = [_cell_identity(cell) for cell in cells]
    expected = _expected_cell_identities(taus, horizons)
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError(
            f"incomplete or duplicate grid: observed={len(identities)}, expected={len(expected)}"
        )
    equal_cells = [cell for cell in cells if cell["work_semantics"] == EQUAL_WORK]
    for cell in equal_cells:
        work = cell.get("work", {})
        if not work.get("rhs_work_equal"):
            raise ValueError("equal-work cell has unequal RHS counts")
        if int(work["direct_rhs_evaluations"]) != int(work["composed_rhs_evaluations"]):
            raise ValueError("equal-work count values disagree")
    return equal_cells


def _positive_metric(cell: Mapping[str, Any], *keys: str) -> float:
    value: Any = cell
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"missing metric path: {keys}")
        value = value[key]
    metric = float(value)
    if not math.isfinite(metric) or metric < 0:
        raise ValueError(f"invalid non-negative metric at {keys}: {metric}")
    return metric


def _geometric_mean(values: Iterable[float]) -> float:
    sequence = [float(value) for value in values]
    if not sequence:
        raise ValueError("geometric mean requires at least one value")
    if any(not math.isfinite(value) or value < 0 for value in sequence):
        raise ValueError("geometric mean received an invalid metric")
    return math.exp(
        statistics.fmean(
            math.log(max(value, GEOMETRIC_MEAN_FLOOR)) for value in sequence
        )
    )


def _descriptive_comparison(
    a_values: Sequence[float],
    b_values: Sequence[float],
) -> dict[str, Any]:
    if not a_values or len(a_values) != len(b_values):
        raise ValueError("descriptive comparison requires aligned non-empty metrics")
    a_geometric_mean = _geometric_mean(a_values)
    b_geometric_mean = _geometric_mean(b_values)
    a_lower_count = sum(
        a_value < b_value for a_value, b_value in zip(a_values, b_values, strict=True)
    )
    return {
        "a_geometric_mean": a_geometric_mean,
        "b_geometric_mean": b_geometric_mean,
        "a_over_b": a_geometric_mean / max(b_geometric_mean, GEOMETRIC_MEAN_FLOOR),
        "a_lower_count": a_lower_count,
        "a_lower_fraction": a_lower_count / len(a_values),
        "comparison_count": len(a_values),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for position in range(start, stop):
            ranks[order[position]] = average
        start = stop
    return ranks


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("Spearman correlation requires equal nontrivial samples")
    x_ranks = _average_ranks(x_values)
    y_ranks = _average_ranks(y_values)
    x_mean = statistics.fmean(x_ranks)
    y_mean = statistics.fmean(y_ranks)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_ranks, y_ranks, strict=True)
    )
    denominator = math.sqrt(
        sum((value - x_mean) ** 2 for value in x_ranks)
        * sum((value - y_mean) ** 2 for value in y_ranks)
    )
    return numerator / denominator if denominator else 0.0


def _discover_runs(root: Path, *, expected_count: int) -> list[Path]:
    resolved = root.expanduser().resolve()
    runs = sorted(path.parent for path in (resolved / "cells").glob("*/results.json"))
    if len(runs) != expected_count:
        raise ValueError(
            f"unexpected completed run count under {resolved}: "
            f"observed={len(runs)}, expected={expected_count}"
        )
    return runs


def _collect_integrator_runs(
    root: Path,
) -> dict[tuple[int, str, str], list[dict[str, Any]]]:
    collected: dict[tuple[int, str, str], list[dict[str, Any]]] = {}
    for run_dir in _discover_runs(root, expected_count=18):
        result, receipt = _verify_completion_files(run_dir)
        key = (
            int(receipt["seed"]),
            str(receipt["model"]),
            str(receipt["integrator"]),
        )
        if key in collected:
            raise ValueError(f"duplicate integrator cell: {key}")
        if int(result.get("sample_count", 0)) != 500:
            raise ValueError(f"integrator cell did not use all 500 samples: {run_dir}")
        collected[key] = _verify_grid(
            result, taus=FORMAL_TAUS, horizons=FORMAL_HORIZONS
        )
    expected = {
        (seed, model, integrator)
        for seed in SEEDS
        for model in MODELS
        for integrator in INTEGRATORS
    }
    if set(collected) != expected:
        raise ValueError("integrator matrix identity differs from preregistration")
    return collected


def _collect_phase_runs(root: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    collected: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for run_dir in _discover_runs(root, expected_count=6):
        result, receipt = _verify_completion_files(run_dir)
        key = (int(receipt["seed"]), str(receipt["model"]))
        if key in collected:
            raise ValueError(f"duplicate phase cell: {key}")
        if (
            receipt.get("integrator") != "rk4"
            or int(result.get("sample_count", 0)) != 128
        ):
            raise ValueError(f"phase cell violates RK4/128-sample protocol: {run_dir}")
        collected[key] = _verify_grid(result, taus=PHASE_TAUS, horizons=PHASE_HORIZONS)
    expected = {(seed, model) for seed in SEEDS for model in MODELS}
    if set(collected) != expected:
        raise ValueError("phase matrix identity differs from preregistration")
    return collected


def _metric_by_grid(
    cells: Sequence[Mapping[str, Any]],
) -> dict[tuple[float, float], Mapping[str, Any]]:
    return {(float(cell["lag"]), float(cell["horizon"])): cell for cell in cells}


def _aggregate_integrators(
    runs: Mapping[tuple[int, str, str], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_integrator: dict[str, Any] = {}
    passing_integrators = 0
    for integrator in INTEGRATORS:
        seed_directions: list[bool] = []
        mse_seed_directions: list[bool] = []
        all_a_defects: list[float] = []
        all_b_defects: list[float] = []
        all_a_mse: list[float] = []
        all_b_mse: list[float] = []
        for seed in SEEDS:
            a_cells = runs[(seed, "latent", integrator)]
            b_cells = runs[(seed, "latent_query_time", integrator)]
            a_defects = [
                _positive_metric(
                    cell,
                    "composition_composed_vs_direct",
                    "relative_l2",
                    "mean",
                )
                for cell in a_cells
            ]
            b_defects = [
                _positive_metric(
                    cell,
                    "composition_composed_vs_direct",
                    "relative_l2",
                    "mean",
                )
                for cell in b_cells
            ]
            a_mse = [
                _positive_metric(cell, "prediction_vs_reference", "mse", "mean")
                for cell in a_cells
            ]
            b_mse = [
                _positive_metric(cell, "prediction_vs_reference", "mse", "mean")
                for cell in b_cells
            ]
            a_defect_gm = _geometric_mean(a_defects)
            b_defect_gm = _geometric_mean(b_defects)
            a_mse_gm = _geometric_mean(a_mse)
            b_mse_gm = _geometric_mean(b_mse)
            direction = a_defect_gm < b_defect_gm
            seed_directions.append(direction)
            mse_seed_directions.append(a_mse_gm < b_mse_gm)
            all_a_defects.extend(a_defects)
            all_b_defects.extend(b_defects)
            all_a_mse.extend(a_mse)
            all_b_mse.extend(b_mse)
            rows.append(
                {
                    "integrator": integrator,
                    "seed": seed,
                    "a_defect_geometric_mean": a_defect_gm,
                    "b_defect_geometric_mean": b_defect_gm,
                    "a_over_b_defect": a_defect_gm
                    / max(b_defect_gm, GEOMETRIC_MEAN_FLOOR),
                    "a_lower_defect": direction,
                    "a_mse_geometric_mean": a_mse_gm,
                    "b_mse_geometric_mean": b_mse_gm,
                    "a_over_b_mse": a_mse_gm / max(b_mse_gm, GEOMETRIC_MEAN_FLOOR),
                }
            )
        aggregate_a = _geometric_mean(all_a_defects)
        aggregate_b = _geometric_mean(all_b_defects)
        aggregate_direction = aggregate_a < aggregate_b
        seed_pass_count = sum(seed_directions)
        integrator_pass = aggregate_direction and seed_pass_count >= 2
        passing_integrators += int(integrator_pass)
        descriptive_mse = _descriptive_comparison(all_a_mse, all_b_mse)
        descriptive_mse.update(
            {
                "a_lower_seed_count": sum(mse_seed_directions),
                "excluded_from_structural_rule": True,
            }
        )
        per_integrator[integrator] = {
            "a_defect_geometric_mean": aggregate_a,
            "b_defect_geometric_mean": aggregate_b,
            "a_over_b_defect": aggregate_a / max(aggregate_b, GEOMETRIC_MEAN_FLOOR),
            "a_lower_defect": aggregate_direction,
            "a_lower_seed_count": seed_pass_count,
            "passes_integrator_rule": integrator_pass,
            "descriptive_prediction_mse": descriptive_mse,
        }
    return rows, {
        "geometric_mean_floor": GEOMETRIC_MEAN_FLOOR,
        "per_integrator": per_integrator,
        "passing_integrator_count": passing_integrators,
        "cross_integrator_structural_pass": passing_integrators >= 2,
        "all_equal_work_rhs_matched": True,
    }


def _aggregate_phase(
    runs: Mapping[tuple[int, str], Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    fraction_pass_count = 0
    association_pass_count = 0
    for seed in SEEDS:
        seed_row_start = len(rows)
        a_grid = _metric_by_grid(runs[(seed, "latent")])
        b_grid = _metric_by_grid(runs[(seed, "latent_query_time")])
        if set(a_grid) != set(b_grid):
            raise ValueError(f"A/B phase grids differ for seed {seed}")
        lower_count = 0
        depths: list[float] = []
        b_defects: list[float] = []
        for lag, horizon in sorted(a_grid):
            a_cell = a_grid[(lag, horizon)]
            b_cell = b_grid[(lag, horizon)]
            a_defect = _positive_metric(
                a_cell,
                "composition_composed_vs_direct",
                "relative_l2",
                "mean",
            )
            b_defect = _positive_metric(
                b_cell,
                "composition_composed_vs_direct",
                "relative_l2",
                "mean",
            )
            a_mse = _positive_metric(a_cell, "prediction_vs_reference", "mse", "mean")
            b_mse = _positive_metric(b_cell, "prediction_vs_reference", "mse", "mean")
            depth = int(a_cell["composition_depth"])
            if depth != int(b_cell["composition_depth"]):
                raise ValueError(
                    f"A/B depth mismatch at seed={seed}, lag={lag}, H={horizon}"
                )
            direction = a_defect < b_defect
            lower_count += int(direction)
            depths.append(float(depth))
            b_defects.append(b_defect)
            rows.append(
                {
                    "seed": seed,
                    "lag": lag,
                    "horizon": horizon,
                    "composition_depth": depth,
                    "a_defect": a_defect,
                    "b_defect": b_defect,
                    "a_over_b_defect": a_defect / max(b_defect, GEOMETRIC_MEAN_FLOOR),
                    "a_lower_defect": direction,
                    "a_mse": a_mse,
                    "b_mse": b_mse,
                    "a_over_b_mse": a_mse / max(b_mse, GEOMETRIC_MEAN_FLOOR),
                }
            )
        fraction = lower_count / len(a_grid)
        rho = _spearman(depths, b_defects)
        fraction_pass = fraction >= 0.8
        association_pass = rho > 0.0
        fraction_pass_count += int(fraction_pass)
        association_pass_count += int(association_pass)
        seed_rows = rows[seed_row_start:]
        defect_summary = _descriptive_comparison(
            [row["a_defect"] for row in seed_rows],
            [row["b_defect"] for row in seed_rows],
        )
        mse_summary = _descriptive_comparison(
            [row["a_mse"] for row in seed_rows],
            [row["b_mse"] for row in seed_rows],
        )
        mse_summary["excluded_from_structural_rule"] = True
        per_seed[str(seed)] = {
            "valid_grid_cells": len(a_grid),
            "a_lower_defect_cells": lower_count,
            "a_lower_defect_fraction": fraction,
            "a_fraction_at_least_0p8": fraction_pass,
            "b_depth_defect_spearman": rho,
            "b_positive_depth_association": association_pass,
            "a_over_b_defect_geometric_mean": defect_summary["a_over_b"],
            "descriptive_prediction_mse": mse_summary,
        }
    overall_defect = _descriptive_comparison(
        [row["a_defect"] for row in rows],
        [row["b_defect"] for row in rows],
    )
    overall_mse = _descriptive_comparison(
        [row["a_mse"] for row in rows],
        [row["b_mse"] for row in rows],
    )
    overall_mse["excluded_from_structural_rule"] = True
    return rows, {
        "per_seed": per_seed,
        "a_fraction_pass_seed_count": fraction_pass_count,
        "b_association_pass_seed_count": association_pass_count,
        "composition_depth_hypothesis_pass": (
            fraction_pass_count >= 2 and association_pass_count >= 2
        ),
        "descriptive_overall": {
            "defect": overall_defect,
            "prediction_mse": overall_mse,
        },
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    if args.evaluator_commit != EXPECTED_EVALUATOR_COMMIT:
        raise ValueError(
            "evaluator commit differs from the frozen implementation: "
            f"{args.evaluator_commit} != {EXPECTED_EVALUATOR_COMMIT}"
        )
    integrator_root = Path(args.integrator_root).expanduser().resolve()
    phase_root = Path(args.phase_root).expanduser().resolve()
    calibration_root = Path(args.calibration_root).expanduser().resolve()
    phase_cache_root = Path(args.phase_cache_root).expanduser().resolve()
    output_root = reserve_exploratory_root(
        args.output_dir,
        input_paths=(integrator_root, phase_root, calibration_root, phase_cache_root),
    )
    source_digest_map = source_hashes(
        {
            "aggregate_semigroup_novelty.py": Path(__file__).resolve(),
            "experiment_artifacts.py": EXPERIMENTS_DIR / "experiment_artifacts.py",
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "aggregating",
        "created_at_unix": time.time(),
        "git_commit": git_commit(REPOSITORY_ROOT),
        "evaluator_commit": args.evaluator_commit,
        "aggregation_semantics": {
            "defect_aggregation": "geometric mean of equal-work relative-L2 means",
            "geometric_mean_floor": GEOMETRIC_MEAN_FLOOR,
            "depth_association": "Spearman rank correlation with average tie ranks",
        },
        "input_roots": {
            "integrator": str(integrator_root),
            "phase": str(phase_root),
            "calibration": str(calibration_root),
            "phase_cache": str(phase_cache_root),
        },
        "source_hashes": source_digest_map,
    }
    atomic_write_json(output_root / "exploratory_manifest.json", manifest)

    calibration_receipt = _verify_support_artifact(
        calibration_root, required_flags=("calibration_pass",)
    )
    phase_cache_receipt = _verify_support_artifact(
        phase_cache_root, required_flags=("reload_validation_passed",)
    )
    integrator_runs = _collect_integrator_runs(integrator_root)
    phase_runs = _collect_phase_runs(phase_root)
    integrator_rows, integrator_summary = _aggregate_integrators(integrator_runs)
    phase_rows, phase_summary = _aggregate_phase(phase_runs)

    completion = {
        "calibration_cells": 1,
        "integrator_evaluator_cells": len(integrator_runs),
        "phase_evaluator_cells": len(phase_runs),
        "integrator_grid_rows": len(integrator_rows),
        "phase_grid_rows": len(phase_rows),
        "all_receipts_parseable": True,
        "all_inputs_unchanged": True,
        "all_equal_work_rhs_matched": True,
        "all_nonfinite_event_counts_zero": True,
        "artifact_completion_pass": True,
    }
    result = {
        **manifest,
        "status": "passed",
        "normal_exit": True,
        "completion": completion,
        "calibration": {
            "status": calibration_receipt["status"],
            "calibration_pass": calibration_receipt["calibration_pass"],
            "exact_semigroup_pass": calibration_receipt.get("exact_semigroup_pass"),
            "all_order_windows_pass": calibration_receipt.get("all_order_windows_pass"),
        },
        "phase_cache": {
            "status": phase_cache_receipt["status"],
            "cache_sha256": phase_cache_receipt["cache_sha256"],
            "reload_validation_passed": phase_cache_receipt["reload_validation_passed"],
        },
        "integrator_control": integrator_summary,
        "phase_diagram": phase_summary,
    }
    atomic_write_json(output_root / "results.json", result)
    atomic_write_csv(
        output_root / "integrator_seed_summary.csv",
        integrator_rows,
        fieldnames=INTEGRATOR_FIELDS,
    )
    atomic_write_csv(
        output_root / "phase_grid.csv", phase_rows, fieldnames=PHASE_FIELDS
    )
    completed_manifest = dict(manifest)
    completed_manifest.update({"status": "passed", "normal_exit": True})
    atomic_write_json(output_root / "exploratory_manifest.json", completed_manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "passed",
        "normal_exit": True,
        "artifact_completion_pass": True,
        "cross_integrator_structural_pass": integrator_summary[
            "cross_integrator_structural_pass"
        ],
        "composition_depth_hypothesis_pass": phase_summary[
            "composition_depth_hypothesis_pass"
        ],
        "results_sha256": sha256_file(output_root / "results.json"),
        "integrator_csv_sha256": sha256_file(
            output_root / "integrator_seed_summary.csv"
        ),
        "phase_csv_sha256": sha256_file(output_root / "phase_grid.csv"),
        "manifest_sha256": sha256_file(output_root / "exploratory_manifest.json"),
        "source_hashes": source_digest_map,
    }
    atomic_write_json(output_root / "receipt.json", receipt)
    (output_root / "done").write_text("passed\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrator-root", required=True)
    parser.add_argument("--phase-root", required=True)
    parser.add_argument("--calibration-root", required=True)
    parser.add_argument("--phase-cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evaluator-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = aggregate(args)
    print(
        json.dumps(
            {
                "status": result["status"],
                "completion": result["completion"],
                "integrator_control": result["integrator_control"],
                "phase_diagram": result["phase_diagram"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
