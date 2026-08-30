#!/usr/bin/env python3
"""Independently aggregate the six Fisher generator/stability evaluator cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from collections.abc import Mapping, Sequence
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
from evaluate_fisher_generator_stability import (  # noqa: E402
    GENERATOR_CSV_FIELDS,
    PAIR_FAMILIES,
    REFERENCE_TIMES,
    REFINEMENT_CSV_FIELDS,
    REFINEMENT_SUBSTEPS,
    STABILITY_CSV_FIELDS,
    TAUS,
    TRACK,
)
from fisher_frozen_inputs import (  # noqa: E402
    FORMAL_LOCKED_CACHE_SHA256,
    FORMAL_SOURCE_ARCHIVE_SHA256,
)


SCHEMA_VERSION = 1
AGGREGATE_TRACK = "fisher_generator_stability_aggregate"
SEEDS = (31415, 271828, 161803)
MODELS = ("latent", "latent_query_time")
CHECKPOINT_SHA256 = {
    (
        31415,
        "latent",
    ): "4330c5760e5ec28d5150db8d7876282f83a474d3a62836988c92919f55441adb",
    (
        31415,
        "latent_query_time",
    ): "e57f1eccb1714357a832f5f9fa1e805b2e26869078e7429fdd81a7c22f991cd6",
    (
        271828,
        "latent",
    ): "0fa5f3067ef58999e575705db516cab9b94481a4a768eae7ced58034bd8ac1cb",
    (
        271828,
        "latent_query_time",
    ): "c87e71f8a11c5c848802880b8f2f685d8f6778d9534ce41746ba20b75ae51f42",
    (
        161803,
        "latent",
    ): "84dcd009982fbc2167aa0f08233ceb2af1ca8cc059623bf29078f64799bf3f9e",
    (
        161803,
        "latent_query_time",
    ): "f580c9c91441565932a706de4044ab1d62992673ba1e649eb67fd8ed3a9d673a",
}

CELL_CSV_FIELDS = (
    "seed",
    "model",
    "conditioning_time",
    "reference_generator_residual_gm",
    "empirical_learned_osl_max",
    "learned_flow_numerical_to_cache_discrepancy_ratio",
    "learned_flow_numerical_confounding",
    "checkpoint_sha256",
    "cell_root",
)

PAIR_CSV_FIELDS = (
    "seed",
    "conditioning_time",
    "generator_residual_a",
    "generator_residual_b",
    "generator_residual_a_over_b",
    "learned_osl_max_a",
    "learned_osl_max_b",
    "a_empirical_stability_no_worse",
    "numerical_to_cache_ratio_a",
    "numerical_to_cache_ratio_b",
)

CELL_ARTIFACT_NAMES = {
    "results.json",
    "generator_metrics.csv",
    "stability_metrics.csv",
    "refinement_metrics.csv",
    "exploratory_manifest.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ValueError(f"CSV header differs from the frozen schema: {path}")
        rows = list(reader)
    if any(None in row or set(row) != set(expected_fields) for row in rows):
        raise ValueError(f"CSV row differs from the frozen schema: {path}")
    return rows


def _geometric_mean(values: Sequence[float]) -> float:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(
        not math.isfinite(value) or value <= 0 for value in normalized
    ):
        raise ValueError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(value) for value in normalized) / len(normalized))


def _finite_float(value: Any, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _validate_csv_values(
    rows: Sequence[Mapping[str, str]],
    *,
    numeric_fields: Sequence[str],
    optional_numeric_fields: Sequence[str] = (),
    label: str,
) -> None:
    optional = set(optional_numeric_fields)
    for row in rows:
        for field in numeric_fields:
            value = row[field]
            if field in optional and value == "":
                continue
            _finite_float(value, f"{label}.{field}")


def _validate_cell_root(root: Path) -> dict[str, Any]:
    required = (
        "results.json",
        "generator_metrics.csv",
        "stability_metrics.csv",
        "refinement_metrics.csv",
        "exploratory_manifest.json",
        "receipt.json",
        "done",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"cell root is incomplete {root}: missing={missing}")
    result = _read_json(root / "results.json")
    receipt = _read_json(root / "receipt.json")
    manifest = _read_json(root / "exploratory_manifest.json")
    if result.get("track") != TRACK or receipt.get("track") != TRACK:
        raise ValueError(f"cell track mismatch: {root}")
    if result.get("status") != "passed" or receipt.get("status") != "passed":
        raise ValueError(f"cell is not a full passing evaluation: {root}")
    if result.get("normal_exit") is not True or receipt.get("normal_exit") is not True:
        raise ValueError(f"cell did not record normal exit: {root}")
    if result.get("smoke_only") is not False or receipt.get("smoke_only") is not False:
        raise ValueError(f"smoke cell cannot enter the full aggregate: {root}")
    if (
        manifest.get("exploratory") is not True
        or manifest.get("do_not_use_for_formal") is not True
        or manifest.get("status") != "passed"
        or manifest.get("normal_exit") is not True
    ):
        raise ValueError(f"cell manifest lacks exploratory scope: {root}")
    if result.get("summary", {}).get("engineering_pass") is not True:
        raise ValueError(f"cell engineering gate did not pass: {root}")

    seed = int(receipt.get("seed", -1))
    model = str(receipt.get("model"))
    identity = (seed, model)
    if identity not in CHECKPOINT_SHA256:
        raise ValueError(f"unexpected cell identity {identity}: {root}")
    expected_checkpoint = CHECKPOINT_SHA256[identity]
    input_hashes = receipt.get("input_hashes")
    if not isinstance(input_hashes, Mapping):
        raise ValueError(f"cell has no input hash contract: {root}")
    expected_inputs = {
        "checkpoint": expected_checkpoint,
        "test_cache": FORMAL_LOCKED_CACHE_SHA256,
        "source_archive": FORMAL_SOURCE_ARCHIVE_SHA256,
    }
    for name, expected_digest in expected_inputs.items():
        record = input_hashes.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"cell input hash record is missing {name}: {root}")
        if (
            record.get("before") != expected_digest
            or record.get("after") != expected_digest
            or record.get("unchanged") is not True
        ):
            raise ValueError(f"cell input hash mismatch for {name}: {root}")

    artifact_hashes = receipt.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError(f"cell has no artifact hashes: {root}")
    if set(artifact_hashes) != CELL_ARTIFACT_NAMES:
        raise ValueError(f"cell artifact hash keys differ from the contract: {root}")
    for name, expected_digest in artifact_hashes.items():
        digest = str(expected_digest).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"cell artifact digest is malformed for {name}: {root}")
        if Path(str(name)).name != str(name):
            raise ValueError(f"cell artifact name is not a basename: {name!r}")
        artifact = root / str(name)
        if not artifact.is_file() or sha256_file(artifact) != digest:
            raise ValueError(f"cell artifact hash mismatch for {name}: {root}")
    evaluator_source_hashes = receipt.get("evaluator_source_hashes")
    if not isinstance(evaluator_source_hashes, Mapping) or not evaluator_source_hashes:
        raise ValueError(f"cell has no evaluator source hashes: {root}")

    generator_rows = _read_csv(root / "generator_metrics.csv", GENERATOR_CSV_FIELDS)
    stability_rows = _read_csv(root / "stability_metrics.csv", STABILITY_CSV_FIELDS)
    refinement_rows = _read_csv(root / "refinement_metrics.csv", REFINEMENT_CSV_FIELDS)
    _validate_csv_values(
        generator_rows,
        numeric_fields=tuple(
            field
            for field in GENERATOR_CSV_FIELDS
            if field not in {"model", "state_source"}
        ),
        optional_numeric_fields=("cosine_mean",),
        label="generator",
    )
    _validate_csv_values(
        stability_rows,
        numeric_fields=tuple(
            field
            for field in STABILITY_CSV_FIELDS
            if field not in {"model", "pair_family", "reference_bound_pass"}
        ),
        label="stability",
    )
    _validate_csv_values(
        refinement_rows,
        numeric_fields=tuple(
            field
            for field in REFINEMENT_CSV_FIELDS
            if field not in {"model", "learned_flow_numerical_confounding"}
        ),
        optional_numeric_fields=(
            "observed_order_to_next",
            "next_substeps_per_call",
            "learned_flow_numerical_to_cache_discrepancy_ratio",
        ),
        label="refinement",
    )
    for label, rows in (
        ("generator", generator_rows),
        ("stability", stability_rows),
        ("refinement", refinement_rows),
    ):
        if any(int(row["seed"]) != seed or row["model"] != model for row in rows):
            raise ValueError(f"cell {label} rows have a mismatched identity: {root}")
    if any(
        row["reference_bound_pass"] != "True" or int(row["nonfinite_events"]) != 0
        for row in stability_rows
    ):
        raise ValueError(f"cell stability gates are inconsistent: {root}")
    if any(int(row["nonfinite_events"]) != 0 for row in generator_rows):
        raise ValueError(f"cell generator finite gate is inconsistent: {root}")
    for row in refinement_rows:
        production = int(row["substeps_per_call"]) == 30
        ratio = row["learned_flow_numerical_to_cache_discrepancy_ratio"]
        flag = row["learned_flow_numerical_confounding"]
        if production and (ratio == "" or flag not in {"True", "False"}):
            raise ValueError(f"cell production refinement row is incomplete: {root}")
        if not production and (ratio != "" or flag != ""):
            raise ValueError(f"cell non-production refinement row is malformed: {root}")
    generator_keys = [
        (
            row["state_source"],
            _finite_float(row["conditioning_time"], "generator conditioning time"),
            _finite_float(row["snapshot_time"], "generator snapshot time"),
        )
        for row in generator_rows
    ]
    expected_generator_keys = {
        (source, tau, snapshot_time)
        for source in ("reference_trajectory", "learned_repeated_lag_trajectory")
        for tau in TAUS
        for snapshot_time in REFERENCE_TIMES
    }
    stability_keys = [
        (
            _finite_float(row["conditioning_time"], "stability conditioning time"),
            _finite_float(row["snapshot_time"], "stability snapshot time"),
            row["pair_family"],
        )
        for row in stability_rows
    ]
    expected_stability_keys = {
        (tau, snapshot_time, family)
        for tau in TAUS
        for snapshot_time in REFERENCE_TIMES
        for family in PAIR_FAMILIES
    }
    refinement_keys = [
        (
            _finite_float(row["conditioning_time"], "refinement conditioning time"),
            int(row["substeps_per_call"]),
        )
        for row in refinement_rows
    ]
    expected_refinement_keys = {
        (tau, substeps) for tau in TAUS for substeps in REFINEMENT_SUBSTEPS
    }
    for label, actual, expected in (
        ("generator", generator_keys, expected_generator_keys),
        ("stability", stability_keys, expected_stability_keys),
        ("refinement", refinement_keys, expected_refinement_keys),
    ):
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f"cell {label} grid differs from the frozen keys: {root}")

    summaries: list[dict[str, Any]] = []
    for conditioning_time in TAUS:
        reference_generator = [
            row
            for row in generator_rows
            if row["state_source"] == "reference_trajectory"
            and math.isclose(float(row["conditioning_time"]), conditioning_time)
        ]
        if len(reference_generator) != len(REFERENCE_TIMES):
            raise ValueError(f"cell reference generator grid is incomplete: {root}")
        generator_gm = _geometric_mean(
            [
                _finite_float(
                    row["rms_relative_residual"], "generator relative residual"
                )
                for row in reference_generator
            ]
        )
        learned_stability = [
            row
            for row in stability_rows
            if math.isclose(float(row["conditioning_time"]), conditioning_time)
        ]
        if len(learned_stability) != len(REFERENCE_TIMES) * 3:
            raise ValueError(f"cell learned stability grid is incomplete: {root}")
        learned_osl_max = max(
            _finite_float(row["learned_max"], "learned OSL maximum")
            for row in learned_stability
        )
        production_refinement = [
            row
            for row in refinement_rows
            if math.isclose(float(row["conditioning_time"]), conditioning_time)
            and int(row["substeps_per_call"]) == 30
        ]
        if len(production_refinement) != 1:
            raise ValueError(f"cell has no unique production refinement row: {root}")
        numerical_ratio = _finite_float(
            production_refinement[0][
                "learned_flow_numerical_to_cache_discrepancy_ratio"
            ],
            "learned-flow numerical/cache discrepancy ratio",
        )
        confounding_value = production_refinement[0][
            "learned_flow_numerical_confounding"
        ]
        if confounding_value not in {"True", "False"}:
            raise ValueError(f"invalid numerical confounding flag: {root}")
        numerical_confounding = confounding_value == "True"
        summaries.append(
            {
                "seed": seed,
                "model": model,
                "conditioning_time": conditioning_time,
                "reference_generator_residual_gm": generator_gm,
                "empirical_learned_osl_max": learned_osl_max,
                "learned_flow_numerical_to_cache_discrepancy_ratio": numerical_ratio,
                "learned_flow_numerical_confounding": numerical_confounding,
                "checkpoint_sha256": expected_checkpoint,
                "cell_root": str(root),
            }
        )
    return {
        "identity": identity,
        "root": str(root),
        "receipt_sha256": sha256_file(root / "receipt.json"),
        "evaluator_source_hashes": dict(evaluator_source_hashes),
        "summaries": summaries,
    }


def aggregate(
    cell_roots: Sequence[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    roots = tuple(Path(value).expanduser().resolve() for value in cell_roots)
    if len(roots) != len(SEEDS) * len(MODELS):
        raise ValueError("exactly six full cell roots are required")
    cells = [_validate_cell_root(root) for root in roots]
    identities = [tuple(cell["identity"]) for cell in cells]
    expected_identities = {(seed, model) for seed in SEEDS for model in MODELS}
    if set(identities) != expected_identities or len(set(identities)) != len(
        identities
    ):
        raise ValueError(
            f"cell identities do not match the frozen matrix: {identities}"
        )
    source_snapshots = {
        json.dumps(cell["evaluator_source_hashes"], sort_keys=True) for cell in cells
    }
    if len(source_snapshots) != 1:
        raise ValueError("the six cells do not share one evaluator source snapshot")
    evaluator_source_hashes = dict(cells[0]["evaluator_source_hashes"])

    output_root = reserve_exploratory_root(output_dir, input_paths=roots)
    started_at = time.time()
    cell_rows = [row for cell in cells for row in cell["summaries"]]
    by_identity_tau = {
        (int(row["seed"]), str(row["model"]), float(row["conditioning_time"])): row
        for row in cell_rows
    }
    pair_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for conditioning_time in TAUS:
            a = by_identity_tau[(seed, "latent", conditioning_time)]
            b = by_identity_tau[(seed, "latent_query_time", conditioning_time)]
            pair_rows.append(
                {
                    "seed": seed,
                    "conditioning_time": conditioning_time,
                    "generator_residual_a": a["reference_generator_residual_gm"],
                    "generator_residual_b": b["reference_generator_residual_gm"],
                    "generator_residual_a_over_b": a["reference_generator_residual_gm"]
                    / b["reference_generator_residual_gm"],
                    "learned_osl_max_a": a["empirical_learned_osl_max"],
                    "learned_osl_max_b": b["empirical_learned_osl_max"],
                    "a_empirical_stability_no_worse": a["empirical_learned_osl_max"]
                    <= b["empirical_learned_osl_max"],
                    "numerical_to_cache_ratio_a": a[
                        "learned_flow_numerical_to_cache_discrepancy_ratio"
                    ],
                    "numerical_to_cache_ratio_b": b[
                        "learned_flow_numerical_to_cache_discrepancy_ratio"
                    ],
                }
            )

    generator_by_tau: dict[str, Any] = {}
    stability_by_tau: dict[str, Any] = {}
    for conditioning_time in TAUS:
        tau_rows = [
            row
            for row in pair_rows
            if math.isclose(float(row["conditioning_time"]), conditioning_time)
        ]
        ratio_gm = _geometric_mean(
            [float(row["generator_residual_a_over_b"]) for row in tau_rows]
        )
        stability_wins = sum(
            bool(row["a_empirical_stability_no_worse"]) for row in tau_rows
        )
        generator_by_tau[str(conditioning_time)] = {
            "geometric_mean_a_over_b": ratio_gm,
            "material_advantage_threshold": 0.90,
            "material_advantage": ratio_gm <= 0.90,
        }
        stability_by_tau[str(conditioning_time)] = {
            "seeds_a_no_worse": stability_wins,
            "required_seeds": 2,
            "empirical_advantage": stability_wins >= 2,
        }

    generator_advantage = all(
        bool(item["material_advantage"]) for item in generator_by_tau.values()
    )
    stability_advantage = all(
        bool(item["empirical_advantage"]) for item in stability_by_tau.values()
    )
    numerical_non_dominant = all(
        not bool(row["learned_flow_numerical_confounding"]) for row in cell_rows
    )
    if not numerical_non_dominant:
        explanation_branch = "learned_flow_numerics_confounding_in_at_least_one_cell"
    elif not generator_advantage or not stability_advantage:
        explanation_branch = "missing_transfer_term_not_improved"
    else:
        explanation_branch = "advance_to_spatial_and_reference_consistency"

    result = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": AGGREGATE_TRACK,
        "status": "passed",
        "normal_exit": True,
        "created_at_unix": started_at,
        "git_commit": git_commit(REPOSITORY_ROOT),
        "source_hashes": source_hashes(
            {
                "aggregate_fisher_generator_stability.py": Path(__file__).resolve(),
                "evaluate_fisher_generator_stability.py": EXPERIMENTS_DIR
                / "evaluate_fisher_generator_stability.py",
                "experiment_artifacts.py": EXPERIMENTS_DIR / "experiment_artifacts.py",
                "fisher_generator_metrics.py": EXPERIMENTS_DIR
                / "fisher_generator_metrics.py",
                "FISHER_GENERATOR_STABILITY_PROTOCOL.md": REPOSITORY_ROOT
                / "docs/research/FISHER_GENERATOR_STABILITY_PROTOCOL.md",
                "FISHER_KPP_THEOREM_CARD.md": REPOSITORY_ROOT
                / "docs/research/FISHER_KPP_THEOREM_CARD.md",
            }
        ),
        "frozen_provenance": {
            "source_commit": "637345584dc2db8ddccf9116a995615c3c036104",
            "source_archive_sha256": FORMAL_SOURCE_ARCHIVE_SHA256,
            "locked_test_cache_sha256": FORMAL_LOCKED_CACHE_SHA256,
            "checkpoint_sha256": {
                f"{seed}:{model}": digest
                for (seed, model), digest in CHECKPOINT_SHA256.items()
            },
            "evaluator_source_hashes": evaluator_source_hashes,
        },
        "matrix": {
            "seeds": list(SEEDS),
            "models": list(MODELS),
            "conditioning_times": list(TAUS),
            "cell_count": len(cells),
            "paired_row_count": len(pair_rows),
        },
        "cells": cells,
        "decision": {
            "generator_matching": generator_by_tau,
            "empirical_stability": stability_by_tau,
            "learned_flow_numerical_non_dominant": numerical_non_dominant,
            "generator_material_advantage": generator_advantage,
            "empirical_stability_advantage": stability_advantage,
            "explanation_branch": explanation_branch,
            "finite_sample_results_are_not_uniform_theorem_constants": True,
        },
    }
    atomic_write_json(output_root / "aggregate.json", result)
    atomic_write_csv(
        output_root / "cell_summary.csv",
        cell_rows,
        fieldnames=CELL_CSV_FIELDS,
    )
    atomic_write_csv(
        output_root / "paired_summary.csv",
        pair_rows,
        fieldnames=PAIR_CSV_FIELDS,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": AGGREGATE_TRACK,
        "status": "passed",
        "cell_roots": [str(root) for root in roots],
        "frozen_provenance": result["frozen_provenance"],
        "source_hashes": result["source_hashes"],
        "cell_receipt_sha256": {
            str(cell["identity"]): cell["receipt_sha256"] for cell in cells
        },
    }
    atomic_write_json(output_root / "exploratory_manifest.json", manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": AGGREGATE_TRACK,
        "status": "passed",
        "normal_exit": True,
        "cell_count": len(cells),
        "decision": result["decision"],
        "frozen_provenance": result["frozen_provenance"],
        "source_hashes": result["source_hashes"],
        "artifact_hashes": {
            "aggregate.json": sha256_file(output_root / "aggregate.json"),
            "cell_summary.csv": sha256_file(output_root / "cell_summary.csv"),
            "paired_summary.csv": sha256_file(output_root / "paired_summary.csv"),
            "exploratory_manifest.json": sha256_file(
                output_root / "exploratory_manifest.json"
            ),
        },
    }
    atomic_write_json(output_root / "receipt.json", receipt)
    (output_root / "done").write_text("passed\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = aggregate(args.cell_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "matrix": result["matrix"],
                "decision": result["decision"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
