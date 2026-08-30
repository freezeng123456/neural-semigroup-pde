from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import pytest


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from aggregate_semigroup_novelty import (  # noqa: E402
    FORMAL_HORIZONS,
    FORMAL_TAUS,
    INTEGRATORS,
    MODELS,
    PHASE_HORIZONS,
    PHASE_TAUS,
    SEEDS,
    _descriptive_comparison,
    _geometric_mean,
    _spearman,
    aggregate,
)
from experiment_artifacts import sha256_file  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerow({"metric": "finite", "value": 1})


def _write_support_artifact(
    root: Path,
    *,
    calibration: bool,
) -> None:
    root.mkdir(parents=True)
    result = {"status": "passed", "normal_exit": True}
    manifest = {"status": "passed", "normal_exit": True}
    _write_json(root / "results.json", result)
    _write_json(root / "exploratory_manifest.json", manifest)
    receipt = {
        "status": "passed",
        "normal_exit": True,
        "results_sha256": sha256_file(root / "results.json"),
        "manifest_sha256": sha256_file(root / "exploratory_manifest.json"),
    }
    if calibration:
        _write_csv(root / "metrics.csv")
        receipt.update(
            {
                "calibration_pass": True,
                "exact_semigroup_pass": True,
                "all_order_windows_pass": True,
                "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
            }
        )
    else:
        receipt.update(
            {
                "cache_sha256": "1" * 64,
                "reload_validation_passed": True,
            }
        )
    _write_json(root / "receipt.json", receipt)


def _cell(
    *,
    lag: float,
    horizon: float,
    semantics: str,
    model: str,
    phase: bool,
) -> dict:
    depth = int(round(horizon / lag))
    b_defect = depth * 1e-4 if phase else 1e-3
    defect = b_defect * 0.1 if model == "latent" else b_defect
    prediction_mse = 2e-3 if model == "latent" else 1e-3
    work_equal = semantics == "equal_rhs_work"
    direct_rhs = depth * 120 if work_equal else 120
    composed_rhs = depth * 120
    return {
        "lag": lag,
        "horizon": horizon,
        "composition_depth": depth,
        "work_semantics": semantics,
        "work": {
            "rhs_work_equal": work_equal,
            "direct_rhs_evaluations": direct_rhs,
            "composed_rhs_evaluations": composed_rhs,
        },
        "prediction_vs_reference": {
            "mse": {"mean": prediction_mse},
            "relative_l2": {"mean": prediction_mse**0.5},
        },
        "composition_composed_vs_direct": {
            "mse": {"mean": defect**2},
            "relative_l2": {"mean": defect},
        },
    }


def _write_evaluator_run(
    run_dir: Path,
    *,
    seed: int,
    model: str,
    integrator: str,
    phase: bool,
) -> None:
    run_dir.mkdir(parents=True)
    taus = PHASE_TAUS if phase else FORMAL_TAUS
    horizons = PHASE_HORIZONS if phase else FORMAL_HORIZONS
    cells = [
        _cell(
            lag=lag,
            horizon=horizon,
            semantics=semantics,
            model=model,
            phase=phase,
        )
        for lag in taus
        for horizon in horizons
        for semantics in ("production_fixed_substeps", "equal_rhs_work")
    ]
    result = {
        "status": "passed",
        "normal_exit": True,
        "smoke_only": False,
        "sample_count": 128 if phase else 500,
        "cells": cells,
    }
    manifest = {"status": "passed", "normal_exit": True}
    _write_json(run_dir / "results.json", result)
    _write_json(run_dir / "exploratory_manifest.json", manifest)
    _write_csv(run_dir / "metrics.csv")
    receipt = {
        "status": "passed",
        "normal_exit": True,
        "smoke_only": False,
        "seed": seed,
        "model": model,
        "integrator": integrator,
        "equal_work_all_rhs_matched": True,
        "nonfinite_sample_events": 0,
        "input_hashes": {
            label: {"before": label, "after": label, "unchanged": True}
            for label in ("checkpoint", "test_cache", "source_archive")
        },
        "results_sha256": sha256_file(run_dir / "results.json"),
        "metrics_csv_sha256": sha256_file(run_dir / "metrics.csv"),
        "manifest_sha256": sha256_file(run_dir / "exploratory_manifest.json"),
    }
    _write_json(run_dir / "receipt.json", receipt)
    (run_dir / "done").write_text("passed\n", encoding="utf-8")


def test_geometric_mean_and_spearman_tie_handling() -> None:
    assert _geometric_mean([1.0, 4.0]) == pytest.approx(2.0)
    assert _descriptive_comparison([1.0, 4.0], [2.0, 2.0]) == {
        "a_geometric_mean": pytest.approx(2.0),
        "b_geometric_mean": pytest.approx(2.0),
        "a_over_b": pytest.approx(1.0),
        "a_lower_count": 1,
        "a_lower_fraction": pytest.approx(0.5),
        "comparison_count": 2,
    }
    assert _spearman([1.0, 2.0, 2.0, 3.0], [2.0, 4.0, 4.0, 8.0]) == pytest.approx(1.0)
    assert _spearman([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 0.0


def test_full_aggregation_separates_completion_from_scientific_rules(
    tmp_path: Path,
) -> None:
    integrator_root = tmp_path / "integrator"
    phase_root = tmp_path / "phase"
    for seed in SEEDS:
        for model in MODELS:
            for integrator in INTEGRATORS:
                _write_evaluator_run(
                    integrator_root / "cells" / f"{seed}-{model}-{integrator}",
                    seed=seed,
                    model=model,
                    integrator=integrator,
                    phase=False,
                )
            _write_evaluator_run(
                phase_root / "cells" / f"{seed}-{model}-rk4",
                seed=seed,
                model=model,
                integrator="rk4",
                phase=True,
            )
    calibration_root = tmp_path / "calibration"
    phase_cache_root = tmp_path / "phase-cache"
    _write_support_artifact(calibration_root, calibration=True)
    _write_support_artifact(phase_cache_root, calibration=False)
    output = tmp_path / "aggregate"
    result = aggregate(
        argparse.Namespace(
            integrator_root=str(integrator_root),
            phase_root=str(phase_root),
            calibration_root=str(calibration_root),
            phase_cache_root=str(phase_cache_root),
            output_dir=str(output),
            evaluator_commit="test-evaluator-commit",
        )
    )
    assert result["status"] == "passed"
    assert result["completion"]["artifact_completion_pass"] is True
    assert result["integrator_control"]["cross_integrator_structural_pass"] is True
    assert result["phase_diagram"]["composition_depth_hypothesis_pass"] is True
    rk4_mse = result["integrator_control"]["per_integrator"]["rk4"][
        "descriptive_prediction_mse"
    ]
    assert rk4_mse["a_over_b"] == pytest.approx(2.0)
    assert rk4_mse["a_lower_seed_count"] == 0
    phase_overall = result["phase_diagram"]["descriptive_overall"]
    assert phase_overall["defect"]["a_over_b"] == pytest.approx(0.1)
    assert phase_overall["prediction_mse"]["a_over_b"] == pytest.approx(2.0)
    assert phase_overall["prediction_mse"]["a_lower_count"] == 0
    assert phase_overall["prediction_mse"]["excluded_from_structural_rule"] is True
    assert (output / "receipt.json").is_file()
    assert (output / "integrator_seed_summary.csv").is_file()
    assert (output / "phase_grid.csv").is_file()


def test_aggregation_rejects_smoke_cell(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_evaluator_run(
        run_dir,
        seed=31415,
        model="latent",
        integrator="rk4",
        phase=False,
    )
    result = _load_json(run_dir / "results.json")
    result["smoke_only"] = True
    _write_json(run_dir / "results.json", result)
    receipt = _load_json(run_dir / "receipt.json")
    receipt["smoke_only"] = True
    receipt["results_sha256"] = sha256_file(run_dir / "results.json")
    _write_json(run_dir / "receipt.json", receipt)
    from aggregate_semigroup_novelty import _verify_completion_files

    with pytest.raises(ValueError, match="smoke output"):
        _verify_completion_files(run_dir)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
