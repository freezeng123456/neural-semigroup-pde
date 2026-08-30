#!/usr/bin/env python3
"""Verify and aggregate all 12 Boundary-Compatible Semigroup Wave 1 cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Mapping


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.boundary_semigroup import (  # noqa: E402
    BOUNDARY_MODES,
    METRIC_FLOOR,
    TEMPORAL_MODES,
    TRAINING_SEEDS,
    expected_metric_identities,
    geometric_mean,
    wave_config,
)
from experiments.experiment_artifacts import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    git_commit,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
)


SOURCE_PATHS = {
    "context": REPOSITORY_ROOT / "CONTEXT.md",
    "protocol": REPOSITORY_ROOT
    / "docs"
    / "research"
    / "BOUNDARY_COMPATIBLE_SEMIGROUP_WAVE1_PROTOCOL.md",
    "boundary_module": EXPERIMENTS_DIR / "boundary_semigroup.py",
    "runner": EXPERIMENTS_DIR / "run_boundary_semigroup.py",
    "aggregator": Path(__file__).resolve(),
}
CELL_SUMMARY_FIELDS = (
    "seed",
    "boundary_mode",
    "temporal_mode",
    "parameter_count",
    "selected_epoch",
    "best_validation_mse",
    "primary_equal_work_defect_geometric_mean",
    "long_rollout_mse_geometric_mean",
    "maximum_normal_boundary_residual",
    "boundary_stress_final_endpoint_max",
    "hard_boundary_pass",
    "run_dir",
)
SEED_SUMMARY_FIELDS = (
    "seed",
    "hard_auto_primary_defect",
    "hard_query_primary_defect",
    "hard_auto_over_query_defect",
    "hard_auto_lower_defect",
    "hard_auto_long_mse",
    "hard_query_long_mse",
    "hard_auto_over_query_long_mse",
    "hard_auto_lower_long_mse",
    "hard_stress_endpoint",
    "unconstrained_stress_endpoint",
    "hard_lower_stress_endpoint",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    _assert_finite(payload, label=str(path))
    return payload


def _assert_finite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite(item, label=label)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item, label=label)


def _require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _verify_hash(run_dir: Path, receipt: Mapping[str, Any], name: str, key: str) -> str:
    path = _require_file(run_dir / name)
    observed = sha256_file(path)
    expected = str(receipt.get(key, ""))
    if observed != expected:
        raise ValueError(
            f"artifact hash mismatch for {path}: observed={observed}, expected={expected}"
        )
    return observed


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"invalid CSV boolean: {value!r}")


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = {
                "evaluation_kind": raw["evaluation_kind"],
                "tau": float(raw["tau"]),
                "horizon": float(raw["horizon"]),
                "composition_depth": int(raw["composition_depth"]),
                "sample_count": int(raw["sample_count"]),
                "equal_rhs_work": _parse_bool(raw["equal_rhs_work"]),
            }
            for name in (
                "equal_work_composition_defect",
                "production_work_composition_defect",
                "rollout_mse",
                "rollout_interior_mse",
                "rollout_relative_l2",
                "direct_equal_work_mse",
                "composed_boundary_max",
                "composed_boundary_rms",
                "max_stage_boundary",
                "rhs_boundary_max",
            ):
                row[name] = float(raw[name])
                if not math.isfinite(row[name]) or row[name] < 0:
                    raise ValueError(f"invalid metric {name} in {path}")
            for name in (
                "composed_rhs_evaluations",
                "direct_equal_rhs_evaluations",
                "direct_production_rhs_evaluations",
            ):
                row[name] = int(raw[name])
            rows.append(row)
    identities = {
        (row["evaluation_kind"], row["tau"], row["horizon"]) for row in rows
    }
    if len(rows) != len(expected_metric_identities()) or identities != expected_metric_identities():
        raise ValueError(f"metric grid is incomplete or duplicated: {path}")
    for row in rows:
        if not row["equal_rhs_work"]:
            raise ValueError(f"equal-work flag failed in {path}")
        if row["composed_rhs_evaluations"] != row["direct_equal_rhs_evaluations"]:
            raise ValueError(f"equal-work RHS counts differ in {path}")
    return rows


def _verify_cell(run_dir: Path) -> dict[str, Any]:
    receipt = _load_json(_require_file(run_dir / "receipt.json"))
    results = _load_json(_require_file(run_dir / "results.json"))
    actual_config = _load_json(_require_file(run_dir / "config.json"))
    expected_config = wave_config(smoke_only=False).as_dict()
    for name, expected_value in expected_config.items():
        if actual_config.get(name) != expected_value:
            raise ValueError(
                f"cell config differs from frozen {name}: "
                f"observed={actual_config.get(name)!r}, expected={expected_value!r}"
            )
    done = _require_file(run_dir / "done").read_text(encoding="utf-8").strip()
    if done != "passed" or receipt.get("status") != "passed":
        raise ValueError(f"cell is not complete: {run_dir}")
    if not receipt.get("normal_exit") or int(receipt.get("exit_code", -1)) != 0:
        raise ValueError(f"cell lacks normal-exit evidence: {run_dir}")
    if receipt.get("smoke_only") or results.get("smoke_only"):
        raise ValueError(f"smoke cell cannot enter full aggregation: {run_dir}")
    _verify_hash(run_dir, receipt, "checkpoint_best.pt", "checkpoint_sha256")
    _verify_hash(run_dir, receipt, "training_log.csv", "training_log_sha256")
    _verify_hash(run_dir, receipt, "metrics.csv", "metrics_csv_sha256")
    _verify_hash(run_dir, receipt, "results.json", "results_sha256")
    _verify_hash(
        run_dir, receipt, "exploratory_manifest.json", "manifest_sha256"
    )
    input_hashes = receipt.get("input_hashes", {}).get("data_cache", {})
    if input_hashes.get("before") != input_hashes.get("after"):
        raise ValueError(f"shared cache changed during cell: {run_dir}")
    if receipt.get("test_metrics_used_for_selection") is not False:
        raise ValueError(f"test-time checkpoint selection detected: {run_dir}")
    for name in ("training_seed", "boundary_mode", "temporal_mode"):
        if actual_config.get(name) != receipt.get(name):
            raise ValueError(f"config/receipt identity mismatch for {name}: {run_dir}")
    rows = _load_metrics(run_dir / "metrics.csv")
    primary = [row for row in rows if row["evaluation_kind"] == "primary_semigroup"]
    long_rollout = [row for row in rows if row["evaluation_kind"] == "long_rollout"]
    recomputed_primary = geometric_mean(
        row["equal_work_composition_defect"] for row in primary
    )
    recomputed_mse = geometric_mean(row["rollout_mse"] for row in long_rollout)
    summary = results.get("evaluation_summary", {})
    if not math.isclose(
        recomputed_primary,
        float(summary["primary_equal_work_defect_geometric_mean"]),
        rel_tol=1e-10,
        abs_tol=1e-18,
    ):
        raise ValueError(f"primary defect summary is not reproducible: {run_dir}")
    if not math.isclose(
        recomputed_mse,
        float(summary["long_rollout_mse_geometric_mean"]),
        rel_tol=1e-10,
        abs_tol=1e-18,
    ):
        raise ValueError(f"long-rollout MSE summary is not reproducible: {run_dir}")
    stress = summary.get("boundary_stress", {})
    return {
        "seed": int(receipt["training_seed"]),
        "boundary_mode": str(receipt["boundary_mode"]),
        "temporal_mode": str(receipt["temporal_mode"]),
        "parameter_count": int(receipt["parameter_count"]),
        "initial_parameter_fingerprint": str(
            receipt["initial_parameter_fingerprint"]
        ),
        "selected_epoch": int(receipt["selected_epoch"]),
        "best_validation_mse": float(receipt["best_validation_mse"]),
        "primary_equal_work_defect_geometric_mean": recomputed_primary,
        "long_rollout_mse_geometric_mean": recomputed_mse,
        "maximum_normal_boundary_residual": float(
            summary["maximum_normal_boundary_residual"]
        ),
        "maximum_hard_boundary_evidence": float(
            summary["maximum_hard_boundary_evidence"]
        ),
        "boundary_stress_final_endpoint_max": float(stress["final_endpoint_max"]),
        "hard_boundary_pass": bool(results["hard_boundary_pass"]),
        "data_cache_sha256": str(input_hashes["before"]),
        "git_commit": str(receipt.get("git_commit")),
        "source_hashes": receipt.get("source_hashes"),
        "run_dir": str(run_dir.resolve()),
        "metrics": rows,
    }


def _discover_cells(wave_root: Path) -> list[Path]:
    root = wave_root.expanduser().resolve()
    runs = sorted(path.parent for path in (root / "cells").glob("*/receipt.json"))
    if len(runs) != 12:
        raise ValueError(
            f"expected 12 completed cell receipts under {root / 'cells'}, found {len(runs)}"
        )
    return runs


def aggregate(wave_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = wave_config(smoke_only=False)
    run_dirs = _discover_cells(Path(wave_root))
    root = reserve_exploratory_root(
        output_dir,
        input_paths=tuple(run_dir / "receipt.json" for run_dir in run_dirs),
    )
    cells = [_verify_cell(run_dir) for run_dir in run_dirs]
    by_identity: dict[tuple[int, str, str], dict[str, Any]] = {}
    for cell in cells:
        identity = (cell["seed"], cell["boundary_mode"], cell["temporal_mode"])
        if identity in by_identity:
            raise ValueError(f"duplicate cell identity: {identity}")
        by_identity[identity] = cell
    expected = {
        (seed, boundary_mode, temporal_mode)
        for seed in TRAINING_SEEDS
        for boundary_mode in BOUNDARY_MODES
        for temporal_mode in TEMPORAL_MODES
    }
    if set(by_identity) != expected:
        raise ValueError("completed cell identities differ from preregistration")

    parameter_counts = {cell["parameter_count"] for cell in cells}
    cache_hashes = {cell["data_cache_sha256"] for cell in cells}
    git_commits = {cell["git_commit"] for cell in cells}
    serialized_source_hashes = {
        json.dumps(cell["source_hashes"], sort_keys=True) for cell in cells
    }
    paired_initialization = all(
        len(
            {
                by_identity[(seed, boundary_mode, temporal_mode)][
                    "initial_parameter_fingerprint"
                ]
                for boundary_mode in BOUNDARY_MODES
                for temporal_mode in TEMPORAL_MODES
            }
        )
        == 1
        for seed in TRAINING_SEEDS
    )
    if len(parameter_counts) != 1:
        raise ValueError("parameter counts are not identical across the 2 x 2 matrix")
    if len(cache_hashes) != 1:
        raise ValueError("cells did not share one unchanged data cache")
    if len(git_commits) != 1 or len(serialized_source_hashes) != 1:
        raise ValueError("cells were not run from one identical source snapshot")
    if not paired_initialization:
        raise ValueError("the four cells do not share initialization within each seed")

    hard_cells = [cell for cell in cells if cell["boundary_mode"] == "hard_dirichlet"]
    boundary_pass = all(
        cell["hard_boundary_pass"]
        and cell["maximum_hard_boundary_evidence"] <= config.boundary_threshold
        for cell in hard_cells
    )
    seed_rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        hard_auto = by_identity[(seed, "hard_dirichlet", "autonomous")]
        hard_query = by_identity[(seed, "hard_dirichlet", "query_time")]
        unconstrained_auto = by_identity[(seed, "unconstrained", "autonomous")]
        auto_defect = hard_auto["primary_equal_work_defect_geometric_mean"]
        query_defect = hard_query["primary_equal_work_defect_geometric_mean"]
        auto_mse = hard_auto["long_rollout_mse_geometric_mean"]
        query_mse = hard_query["long_rollout_mse_geometric_mean"]
        hard_stress = hard_auto["boundary_stress_final_endpoint_max"]
        unconstrained_stress = unconstrained_auto[
            "boundary_stress_final_endpoint_max"
        ]
        seed_rows.append(
            {
                "seed": seed,
                "hard_auto_primary_defect": auto_defect,
                "hard_query_primary_defect": query_defect,
                "hard_auto_over_query_defect": auto_defect
                / max(query_defect, METRIC_FLOOR),
                "hard_auto_lower_defect": auto_defect < query_defect,
                "hard_auto_long_mse": auto_mse,
                "hard_query_long_mse": query_mse,
                "hard_auto_over_query_long_mse": auto_mse
                / max(query_mse, METRIC_FLOOR),
                "hard_auto_lower_long_mse": auto_mse < query_mse,
                "hard_stress_endpoint": hard_stress,
                "unconstrained_stress_endpoint": unconstrained_stress,
                "hard_lower_stress_endpoint": hard_stress < unconstrained_stress,
            }
        )

    direction_count = sum(row["hard_auto_lower_defect"] for row in seed_rows)
    aggregate_auto_defect = geometric_mean(
        row["hard_auto_primary_defect"] for row in seed_rows
    )
    aggregate_query_defect = geometric_mean(
        row["hard_query_primary_defect"] for row in seed_rows
    )
    temporal_pass = (
        direction_count >= 2 and aggregate_auto_defect < aggregate_query_defect
    )
    mse_direction_count = sum(row["hard_auto_lower_long_mse"] for row in seed_rows)
    aggregate_auto_mse = geometric_mean(row["hard_auto_long_mse"] for row in seed_rows)
    aggregate_query_mse = geometric_mean(
        row["hard_query_long_mse"] for row in seed_rows
    )

    cell_rows = [
        {
            name: cell[name]
            for name in CELL_SUMMARY_FIELDS
        }
        for cell in sorted(
            cells,
            key=lambda item: (
                item["seed"], item["boundary_mode"], item["temporal_mode"]
            ),
        )
    ]
    result = {
        "schema_version": 1,
        "experiment": config.experiment,
        "evidence_class": "exploratory_full_aggregate",
        "status": "passed",
        "execution_complete": True,
        "cell_count": len(cells),
        "controls": {
            "single_parameter_count": next(iter(parameter_counts)),
            "single_data_cache_sha256": next(iter(cache_hashes)),
            "single_git_commit": next(iter(git_commits)),
            "single_source_snapshot": True,
            "paired_initialization_within_seed": paired_initialization,
        },
        "boundary_rule": {
            "threshold": config.boundary_threshold,
            "hard_cell_count": len(hard_cells),
            "pass": boundary_pass,
            "maximum_evidence": max(
                cell["maximum_hard_boundary_evidence"] for cell in hard_cells
            ),
        },
        "temporal_rule": {
            "paired_seed_direction_count": direction_count,
            "required_direction_count": 2,
            "hard_autonomous_defect_geometric_mean": aggregate_auto_defect,
            "hard_query_time_defect_geometric_mean": aggregate_query_defect,
            "autonomous_over_query_time": aggregate_auto_defect
            / max(aggregate_query_defect, METRIC_FLOOR),
            "pass": temporal_pass,
        },
        "prediction_support": {
            "descriptive_only": True,
            "paired_seed_autonomous_lower_mse_count": mse_direction_count,
            "hard_autonomous_long_mse_geometric_mean": aggregate_auto_mse,
            "hard_query_time_long_mse_geometric_mean": aggregate_query_mse,
            "autonomous_over_query_time": aggregate_auto_mse
            / max(aggregate_query_mse, METRIC_FLOOR),
        },
        "seed_summaries": seed_rows,
        "scientific_hypotheses": {
            "hard_boundary_compatibility": boundary_pass,
            "temporal_semigroup_comparison": temporal_pass,
            "prediction_gain_claimed": False,
        },
    }
    atomic_write_csv(root / "cell_summary.csv", cell_rows, fieldnames=CELL_SUMMARY_FIELDS)
    atomic_write_csv(root / "seed_summary.csv", seed_rows, fieldnames=SEED_SUMMARY_FIELDS)
    atomic_write_json(root / "aggregate.json", result)
    artifact_names = ("cell_summary.csv", "seed_summary.csv", "aggregate.json")
    manifest = {
        "schema_version": 1,
        "experiment": config.experiment,
        "evidence_class": "exploratory_full_aggregate",
        "created_at_unix": time.time(),
        "git_commit": git_commit(REPOSITORY_ROOT),
        "source_hashes": source_hashes(SOURCE_PATHS),
        "input_cell_receipts": {
            str(run_dir.resolve()): sha256_file(run_dir / "receipt.json")
            for run_dir in run_dirs
        },
        "artifacts": {
            name: {"bytes": (root / name).stat().st_size, "sha256": sha256_file(root / name)}
            for name in artifact_names
        },
    }
    atomic_write_json(root / "exploratory_manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "experiment": config.experiment,
        "evidence_class": "exploratory_full_aggregate",
        "status": "passed",
        "normal_exit": True,
        "exit_code": 0,
        "actual_command": shlex.join(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
        ),
        "cell_count": len(cells),
        "execution_complete": True,
        "boundary_rule_pass": boundary_pass,
        "temporal_rule_pass": temporal_pass,
        "aggregate_sha256": sha256_file(root / "aggregate.json"),
        "cell_summary_sha256": sha256_file(root / "cell_summary.csv"),
        "seed_summary_sha256": sha256_file(root / "seed_summary.csv"),
        "manifest_sha256": sha256_file(root / "exploratory_manifest.json"),
    }
    atomic_write_json(root / "receipt.json", receipt)
    (root / "done").write_text("passed\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = aggregate(args.wave_root, args.output_dir)
    print(
        {
            "status": result["status"],
            "cell_count": result["cell_count"],
            "boundary_rule_pass": result["boundary_rule"]["pass"],
            "temporal_rule_pass": result["temporal_rule"]["pass"],
        }
    )


if __name__ == "__main__":
    main()
