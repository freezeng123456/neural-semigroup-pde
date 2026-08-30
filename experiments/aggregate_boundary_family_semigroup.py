#!/usr/bin/env python3
"""Aggregate the complete 54-cell Boundary-Family Semigroup Wave 2 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.boundary_family_semigroup import (  # noqa: E402
    BOUNDARY_FAMILIES,
    ENFORCEMENT_MODES,
    METRIC_FLOOR,
    REFINEMENT_STEPS,
    TEMPORAL_MODES,
    TRAINING_SEEDS,
    expected_metric_identities,
    geometric_mean,
    wave_config,
)
from experiments.experiment_artifacts import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    device_provenance,
    git_commit,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
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
    "runner": EXPERIMENTS_DIR / "run_boundary_family_semigroup.py",
    "aggregator": Path(__file__).resolve(),
    "artifact_contract": EXPERIMENTS_DIR / "experiment_artifacts.py",
}
CELL_SUMMARY_FIELDS = (
    "boundary_family",
    "seed",
    "enforcement_mode",
    "temporal_mode",
    "parameter_count",
    "selected_epoch",
    "best_validation_mse",
    "uniform_primary_defect_geometric_mean",
    "primary_nonuniform_defect_geometric_mean",
    "long_rollout_mse_geometric_mean",
    "maximum_hard_boundary_evidence",
    "boundary_stress_final_state_max",
    "hard_boundary_pass",
    "data_cache_sha256",
    "checkpoint_sha256",
    "run_dir",
)
FAMILY_SEED_FIELDS = (
    "boundary_family",
    "seed",
    "hard_auto_nonuniform_defect",
    "hard_query_nonuniform_defect",
    "hard_auto_over_query_defect",
    "hard_auto_lower_defect",
    "hard_auto_long_mse",
    "hard_query_long_mse",
    "hard_auto_over_query_long_mse",
    "hard_auto_lower_long_mse",
    "hard_auto_stress_residual",
    "penalty_auto_stress_residual",
    "unconstrained_auto_stress_residual",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _verify_hash(root: Path, receipt: dict[str, Any], name: str, field: str) -> str:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256_file(path)
    if observed != receipt.get(field):
        raise ValueError(f"artifact hash mismatch for {path}")
    return observed


def _verify_manifest(run_dir: Path, receipt: dict[str, Any]) -> None:
    manifest = _load_json(run_dir / "exploratory_manifest.json")
    for name in (
        "experiment",
        "boundary_family",
        "training_seed",
        "enforcement_mode",
        "temporal_mode",
    ):
        if manifest.get(name) != receipt.get(name):
            raise ValueError(f"manifest identity mismatch for {name}: {run_dir}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"manifest has no artifact map: {run_dir}")
    for name, expected in artifacts.items():
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(expected.get("bytes", -1)):
            raise ValueError(f"manifest byte count mismatch for {path}")
        if sha256_file(path) != expected.get("sha256"):
            raise ValueError(f"manifest artifact hash mismatch for {path}")


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
                "partition_id": raw["partition_id"],
                "partition_json": raw["partition_json"],
                "horizon": float(raw["horizon"]),
                "segment_count": int(raw["segment_count"]),
                "steps_per_segment": int(raw["steps_per_segment"]),
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
                "composed_boundary_state_max",
                "composed_boundary_state_rms",
                "composed_boundary_equation_max",
                "max_stage_boundary_state",
                "max_stage_boundary_equation",
                "rhs_boundary_tangent_max",
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
        (row["evaluation_kind"], row["partition_id"], row["steps_per_segment"])
        for row in rows
    }
    if identities != expected_metric_identities(8) or len(rows) != len(identities):
        raise ValueError(f"metric grid is incomplete or duplicated: {path}")
    for row in rows:
        if not row["equal_rhs_work"]:
            raise ValueError(f"equal-work flag failed in {path}")
        if row["composed_rhs_evaluations"] != row["direct_equal_rhs_evaluations"]:
            raise ValueError(f"equal-work RHS counts differ in {path}")
    return rows


def _verify_cell(run_dir: Path) -> dict[str, Any]:
    receipt = _load_json(run_dir / "receipt.json")
    results = _load_json(run_dir / "results.json")
    actual_config = _load_json(run_dir / "config.json")
    family = str(receipt.get("boundary_family"))
    expected_config = wave_config(family, smoke_only=False).as_dict()
    for name, expected_value in expected_config.items():
        if actual_config.get(name) != expected_value:
            raise ValueError(f"cell config differs from frozen {name}: {run_dir}")
    for name in (
        "training_seed",
        "boundary_family",
        "enforcement_mode",
        "temporal_mode",
    ):
        if actual_config.get(name) != receipt.get(name):
            raise ValueError(f"config/receipt identity mismatch for {name}: {run_dir}")
    if (run_dir / "done").read_text(encoding="utf-8").strip() != "passed":
        raise ValueError(f"cell lacks passing done marker: {run_dir}")
    if receipt.get("status") != "passed" or not receipt.get("normal_exit"):
        raise ValueError(f"cell is not normally complete: {run_dir}")
    if int(receipt.get("exit_code", -1)) != 0 or receipt.get("smoke_only"):
        raise ValueError(f"invalid full-cell exit/smoke state: {run_dir}")
    _verify_hash(run_dir, receipt, "checkpoint_best.pt", "checkpoint_sha256")
    _verify_hash(run_dir, receipt, "training_log.csv", "training_log_sha256")
    _verify_hash(run_dir, receipt, "metrics.csv", "metrics_csv_sha256")
    _verify_hash(run_dir, receipt, "results.json", "results_sha256")
    _verify_hash(run_dir, receipt, "exploratory_manifest.json", "manifest_sha256")
    _verify_manifest(run_dir, receipt)
    input_hashes = receipt.get("input_hashes", {}).get("data_cache", {})
    if input_hashes.get("before") != input_hashes.get("after"):
        raise ValueError(f"cache changed during cell: {run_dir}")
    cache_path = Path(str(receipt.get("data_cache_path", ""))).expanduser().resolve()
    if not cache_path.is_file() or sha256_file(cache_path) != input_hashes.get(
        "before"
    ):
        raise ValueError(f"current cache hash differs from cell receipt: {run_dir}")
    if receipt.get("test_metrics_used_for_selection") is not False:
        raise ValueError(f"test-time checkpoint selection detected: {run_dir}")
    rows = _load_metrics(run_dir / "metrics.csv")
    uniform = [row for row in rows if row["evaluation_kind"] == "uniform_primary"]
    nonuniform = [
        row
        for row in rows
        if row["evaluation_kind"] == "nonuniform_refinement"
        and row["steps_per_segment"] == 8
    ]
    long_rollout = [row for row in rows if row["evaluation_kind"] == "long_rollout"]
    recomputed = {
        "uniform_primary_defect_geometric_mean": geometric_mean(
            row["equal_work_composition_defect"] for row in uniform
        ),
        "primary_nonuniform_defect_geometric_mean": geometric_mean(
            row["equal_work_composition_defect"] for row in nonuniform
        ),
        "long_rollout_mse_geometric_mean": geometric_mean(
            row["rollout_mse"] for row in long_rollout
        ),
    }
    summary = results.get("evaluation_summary", {})
    for name, value in recomputed.items():
        if not math.isclose(value, float(summary[name]), rel_tol=1e-10, abs_tol=1e-18):
            raise ValueError(f"summary {name} is not reproducible: {run_dir}")
    result_metrics = results.get("metrics")
    if not isinstance(result_metrics, list) or len(result_metrics) != len(rows):
        raise ValueError(f"results/CSV metric counts differ: {run_dir}")
    result_identities = {
        (
            row.get("evaluation_kind"),
            row.get("partition_id"),
            int(row.get("steps_per_segment", -1)),
        )
        for row in result_metrics
    }
    csv_identities = {
        (row["evaluation_kind"], row["partition_id"], row["steps_per_segment"])
        for row in rows
    }
    if result_identities != csv_identities:
        raise ValueError(f"results/CSV metric identities differ: {run_dir}")
    stress = summary.get("boundary_stress", {})
    return {
        "boundary_family": family,
        "seed": int(receipt["training_seed"]),
        "enforcement_mode": str(receipt["enforcement_mode"]),
        "temporal_mode": str(receipt["temporal_mode"]),
        "parameter_count": int(receipt["parameter_count"]),
        "initial_parameter_fingerprint": str(receipt["initial_parameter_fingerprint"]),
        "selected_epoch": int(receipt["selected_epoch"]),
        "best_validation_mse": float(receipt["best_validation_mse"]),
        **recomputed,
        "refinement": summary["nonuniform_refinement_defect_geometric_means"],
        "maximum_hard_boundary_evidence": float(
            summary["maximum_hard_boundary_evidence"]
        ),
        "boundary_stress_final_state_max": float(stress["final_state_max"]),
        "hard_boundary_pass": bool(results["hard_boundary_pass"]),
        "data_cache_path": str(cache_path),
        "data_cache_sha256": str(input_hashes["before"]),
        "checkpoint_sha256": str(receipt["checkpoint_sha256"]),
        "git_commit": str(receipt.get("git_commit")),
        "source_archive_sha256": receipt.get("source_archive_sha256"),
        "source_hashes": receipt.get("source_hashes"),
        "run_dir": str(run_dir.resolve()),
    }


def _source_provenance() -> dict[str, Any]:
    return {
        "git_commit": git_commit(REPOSITORY_ROOT)
        or os.environ.get("SEMIGROUP_SOURCE_COMMIT"),
        "source_archive_sha256": os.environ.get("SEMIGROUP_SOURCE_ARCHIVE_SHA256"),
        "source_hashes": source_hashes(SOURCE_PATHS),
    }


def _actual_command() -> str:
    return shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])


def aggregate(wave_root: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(wave_root).expanduser().resolve()
    run_dirs = sorted(path.parent for path in (root / "cells").glob("*/receipt.json"))
    if len(run_dirs) != 54:
        raise ValueError(f"expected 54 completed cells, found {len(run_dirs)}")
    output = reserve_exploratory_root(
        output_dir, input_paths=tuple(run_dir / "receipt.json" for run_dir in run_dirs)
    )
    cells = [_verify_cell(run_dir) for run_dir in run_dirs]
    by_identity: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for cell in cells:
        identity = (
            cell["boundary_family"],
            cell["seed"],
            cell["enforcement_mode"],
            cell["temporal_mode"],
        )
        if identity in by_identity:
            raise ValueError(f"duplicate cell identity: {identity}")
        by_identity[identity] = cell
    expected = {
        (family, seed, enforcement, temporal)
        for family in BOUNDARY_FAMILIES
        for seed in TRAINING_SEEDS
        for enforcement in ENFORCEMENT_MODES
        for temporal in TEMPORAL_MODES
    }
    if set(by_identity) != expected:
        raise ValueError("completed identities differ from the frozen matrix")

    parameter_counts = {cell["parameter_count"] for cell in cells}
    commits = {cell["git_commit"] for cell in cells}
    archives = {cell["source_archive_sha256"] for cell in cells}
    serialized_hashes = {
        json.dumps(cell["source_hashes"], sort_keys=True) for cell in cells
    }
    if len(parameter_counts) != 1:
        raise ValueError("parameter counts differ across the matrix")
    if len(commits) != 1 or len(serialized_hashes) != 1:
        raise ValueError("cells do not share one source snapshot")
    if len(archives) != 1 or next(iter(archives)) is None:
        raise ValueError("cells lack one source archive SHA-256")
    source_commit = next(iter(commits))
    source_archive_sha256 = next(iter(archives))
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or len(source_archive_sha256) != 64
        or any(
            character not in "0123456789abcdef" for character in source_archive_sha256
        )
    ):
        raise ValueError("source commit or archive digest is malformed")
    recorded_source_hashes = cells[0]["source_hashes"]
    current_source_hashes = source_hashes(
        {
            name: SOURCE_PATHS[name]
            for name in recorded_source_hashes
            if name in SOURCE_PATHS
        }
    )
    if current_source_hashes != recorded_source_hashes:
        raise ValueError("current frozen source files differ from cell provenance")
    paired_initialization = all(
        len(
            {
                by_identity[(family, seed, enforcement, temporal)][
                    "initial_parameter_fingerprint"
                ]
                for family in BOUNDARY_FAMILIES
                for enforcement in ENFORCEMENT_MODES
                for temporal in TEMPORAL_MODES
            }
        )
        == 1
        for seed in TRAINING_SEEDS
    )
    if not paired_initialization:
        raise ValueError("18 cells do not share initialization within each seed")
    cache_hashes_by_family = {
        family: {
            cell["data_cache_sha256"]
            for cell in cells
            if cell["boundary_family"] == family
        }
        for family in BOUNDARY_FAMILIES
    }
    if any(len(values) != 1 for values in cache_hashes_by_family.values()):
        raise ValueError("a boundary family did not use one unchanged cache")
    if len({next(iter(values)) for values in cache_hashes_by_family.values()}) != 3:
        raise ValueError("the three boundary families did not use distinct caches")

    family_seed_rows: list[dict[str, Any]] = []
    family_summaries: dict[str, Any] = {}
    for family in BOUNDARY_FAMILIES:
        hard_cells = [
            cell
            for cell in cells
            if cell["boundary_family"] == family and cell["enforcement_mode"] == "hard"
        ]
        boundary_pass = all(
            cell["hard_boundary_pass"]
            and cell["maximum_hard_boundary_evidence"]
            <= wave_config(family).boundary_threshold
            for cell in hard_cells
        )
        seed_rows = []
        for seed in TRAINING_SEEDS:
            hard_auto = by_identity[(family, seed, "hard", "autonomous")]
            hard_query = by_identity[(family, seed, "hard", "query_time")]
            penalty_auto = by_identity[(family, seed, "penalty", "autonomous")]
            unconstrained_auto = by_identity[
                (family, seed, "unconstrained", "autonomous")
            ]
            auto_defect = hard_auto["primary_nonuniform_defect_geometric_mean"]
            query_defect = hard_query["primary_nonuniform_defect_geometric_mean"]
            auto_mse = hard_auto["long_rollout_mse_geometric_mean"]
            query_mse = hard_query["long_rollout_mse_geometric_mean"]
            row = {
                "boundary_family": family,
                "seed": seed,
                "hard_auto_nonuniform_defect": auto_defect,
                "hard_query_nonuniform_defect": query_defect,
                "hard_auto_over_query_defect": auto_defect
                / max(query_defect, METRIC_FLOOR),
                "hard_auto_lower_defect": auto_defect < query_defect,
                "hard_auto_long_mse": auto_mse,
                "hard_query_long_mse": query_mse,
                "hard_auto_over_query_long_mse": auto_mse
                / max(query_mse, METRIC_FLOOR),
                "hard_auto_lower_long_mse": auto_mse < query_mse,
                "hard_auto_stress_residual": hard_auto[
                    "boundary_stress_final_state_max"
                ],
                "penalty_auto_stress_residual": penalty_auto[
                    "boundary_stress_final_state_max"
                ],
                "unconstrained_auto_stress_residual": unconstrained_auto[
                    "boundary_stress_final_state_max"
                ],
            }
            family_seed_rows.append(row)
            seed_rows.append(row)
        direction_count = sum(row["hard_auto_lower_defect"] for row in seed_rows)
        auto_defect_gm = geometric_mean(
            row["hard_auto_nonuniform_defect"] for row in seed_rows
        )
        query_defect_gm = geometric_mean(
            row["hard_query_nonuniform_defect"] for row in seed_rows
        )
        family_summaries[family] = {
            "boundary_rule": {
                "pass": boundary_pass,
                "hard_cell_count": len(hard_cells),
                "threshold": wave_config(family).boundary_threshold,
                "maximum_evidence": max(
                    cell["maximum_hard_boundary_evidence"] for cell in hard_cells
                ),
            },
            "temporal_rule": {
                "pass": direction_count >= 2 and auto_defect_gm < query_defect_gm,
                "paired_seed_direction_count": direction_count,
                "hard_auto_defect_geometric_mean": auto_defect_gm,
                "hard_query_defect_geometric_mean": query_defect_gm,
                "hard_auto_over_query": auto_defect_gm
                / max(query_defect_gm, METRIC_FLOOR),
            },
            "prediction_descriptive": {
                "hard_auto_lower_long_mse_seed_count": sum(
                    row["hard_auto_lower_long_mse"] for row in seed_rows
                ),
                "hard_auto_long_mse_geometric_mean": geometric_mean(
                    row["hard_auto_long_mse"] for row in seed_rows
                ),
                "hard_query_long_mse_geometric_mean": geometric_mean(
                    row["hard_query_long_mse"] for row in seed_rows
                ),
            },
            "hard_refinement_descriptive": {
                temporal: {
                    str(steps): geometric_mean(
                        float(
                            by_identity[(family, seed, "hard", temporal)]["refinement"][
                                str(steps)
                            ]
                        )
                        for seed in TRAINING_SEEDS
                    )
                    for steps in REFINEMENT_STEPS
                }
                for temporal in TEMPORAL_MODES
            },
        }

    boundary_implementation_pass = all(
        summary["boundary_rule"]["pass"] for summary in family_summaries.values()
    )
    if not boundary_implementation_pass:
        raise ValueError("hard boundary implementation gate failed")
    controls = {
        "single_parameter_count": next(iter(parameter_counts)),
        "single_git_commit": source_commit,
        "single_source_archive_sha256": source_archive_sha256,
        "single_source_snapshot": True,
        "paired_initialization_within_seed": paired_initialization,
        "cache_sha256_by_family": {
            family: next(iter(values))
            for family, values in cache_hashes_by_family.items()
        },
    }
    result = {
        "schema_version": 1,
        "experiment": "boundary_family_semigroup_wave2",
        "evidence_class": "exploratory_full_aggregate",
        "status": "passed",
        "execution_complete": True,
        "cell_count": len(cells),
        "boundary_implementation_pass": boundary_implementation_pass,
        "controls": controls,
        "family_summaries": family_summaries,
    }
    cell_rows = [
        {name: cell[name] for name in CELL_SUMMARY_FIELDS}
        for cell in sorted(
            cells,
            key=lambda value: (
                value["boundary_family"],
                value["seed"],
                value["enforcement_mode"],
                value["temporal_mode"],
            ),
        )
    ]
    atomic_write_csv(
        output / "cell_summary.csv", cell_rows, fieldnames=CELL_SUMMARY_FIELDS
    )
    atomic_write_csv(
        output / "family_seed_summary.csv",
        family_seed_rows,
        fieldnames=FAMILY_SEED_FIELDS,
    )
    atomic_write_json(output / "aggregate.json", result)
    runtime = device_provenance(torch.device("cpu"))
    provenance = _source_provenance()
    manifest = {
        "schema_version": 1,
        "experiment": result["experiment"],
        "evidence_class": result["evidence_class"],
        "created_at_unix": time.time(),
        **provenance,
        "runtime": runtime,
        "inputs": {
            str(run_dir.resolve()): sha256_file(run_dir / "receipt.json")
            for run_dir in run_dirs
        },
        "artifacts": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256_file(output / name),
            }
            for name in (
                "aggregate.json",
                "cell_summary.csv",
                "family_seed_summary.csv",
            )
        },
    }
    atomic_write_json(output / "exploratory_manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "experiment": result["experiment"],
        "evidence_class": result["evidence_class"],
        "status": "passed",
        "normal_exit": True,
        "exit_code": 0,
        "actual_command": _actual_command(),
        "pid": os.getpid(),
        "cell_count": len(cells),
        "boundary_implementation_pass": boundary_implementation_pass,
        **provenance,
        "aggregate_sha256": sha256_file(output / "aggregate.json"),
        "cell_summary_sha256": sha256_file(output / "cell_summary.csv"),
        "family_seed_summary_sha256": sha256_file(output / "family_seed_summary.csv"),
        "manifest_sha256": sha256_file(output / "exploratory_manifest.json"),
    }
    atomic_write_json(output / "receipt.json", receipt)
    (output / "done").write_text("passed\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = aggregate(args.wave_root, args.output_dir)
    print(
        {
            "status": result["status"],
            "cell_count": result["cell_count"],
            "boundary_implementation_pass": result["boundary_implementation_pass"],
        }
    )


if __name__ == "__main__":
    main()
