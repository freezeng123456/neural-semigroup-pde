#!/usr/bin/env python3
"""Validate the canonical artifact contract of an Allen--Cahn fair run.

This verifier is intentionally independent of training and checkpoint
selection.  A launcher calls it only after the trainer exits successfully;
it checks that the run contains the portable cell-level records required for
recovery and later checkpoint-only evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_json(path: Path):
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _require_nonempty_file(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty artifact: {path}")


def validate_artifacts(output_dir, expected_models):
    """Validate one completed output directory and return a compact receipt."""
    output_dir = Path(output_dir).resolve()
    expected_models = tuple(dict.fromkeys(expected_models))
    if not expected_models:
        raise ValueError("at least one expected model is required")

    summary_path = output_dir / "summary.json"
    provenance_path = output_dir / "provenance.json"
    metrics_path = output_dir / "metrics.csv"
    for path in (summary_path, provenance_path, metrics_path):
        _require_nonempty_file(path)

    summary = _load_json(summary_path)
    provenance = _load_json(provenance_path)
    summary_models = tuple(summary.get("results", {}))
    if summary_models != expected_models:
        raise ValueError(
            "summary model order does not match the launcher contract: "
            f"expected {expected_models}, got {summary_models}"
        )
    provenance_models = tuple(provenance.get("config", {}).get("models", ()))
    if provenance_models != expected_models:
        raise ValueError(
            "provenance model order does not match the launcher contract: "
            f"expected {expected_models}, got {provenance_models}"
        )

    try:
        with metrics_path.open(encoding="utf-8", newline="") as handle:
            metric_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ValueError(f"invalid metrics artifact: {metrics_path}") from exc
    if not metric_rows or not all(row.get("model") and row.get("tau") for row in metric_rows):
        raise ValueError(f"metrics artifact has no complete model/tau rows: {metrics_path}")
    metric_models = {row["model"] for row in metric_rows}
    if metric_models != set(expected_models):
        raise ValueError(
            "metrics models do not match the launcher contract: "
            f"expected {set(expected_models)}, got {metric_models}"
        )

    model_receipts = {}
    for model_name in expected_models:
        model_dir = output_dir / model_name
        result_path = model_dir / "result.json"
        _require_nonempty_file(result_path)
        result = _load_json(result_path)
        if result.get("model") != model_name:
            raise ValueError(f"result model mismatch in {result_path}")
        if summary["results"][model_name].get("parameter_count") != result.get(
            "parameter_count"
        ):
            raise ValueError(f"summary/result parameter mismatch for {model_name}")
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_names = (
            f"{model_name}_best.pt",
            f"{model_name}_final.pt",
            f"{model_name}_history.pt",
        )
        for checkpoint_name in checkpoint_names:
            _require_nonempty_file(checkpoint_dir / checkpoint_name)
        model_receipts[model_name] = {
            "parameter_count": result["parameter_count"],
            "checkpoint_epoch": result["checkpoint_epoch"],
            "metrics_rows": sum(
                row["model"] == model_name for row in metric_rows
            ),
        }

    return {
        "artifact_contract": "allen_cahn_fair_v1",
        "output_dir": str(output_dir),
        "models": model_receipts,
        "metrics_rows": len(metric_rows),
        "status": "passed",
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    receipt = validate_artifacts(args.output_dir, args.models)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
