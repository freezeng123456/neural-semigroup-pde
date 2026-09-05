"""Acceptance checks for the matched Burgers screen aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aggregate_burgers_semigroup_screen import EXPECTED_CELLS, main


def build_summary(seed, *, mse_ratio=0.95, cache="cache-digest", parameters=1313):
    def evaluation(defect, drift):
        return {
            "rollouts": {
                cell: {
                    "rollout_mse_mean": 1e-3,
                    "mean_drift_max": drift,
                    "energy_change_mean": -1e-3,
                    "energy_increase_fraction": 0.0,
                }
                for cell in EXPECTED_CELLS
            },
            "composition": {
                cell: {
                    "equal_work_rms_defect_mean": defect,
                    "composition_depth": 10,
                    "rhs_evaluations_each_path": 480,
                }
                for cell in EXPECTED_CELLS
            },
        }

    return {
        "experiment": "burgers_matched_semigroup_screen",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": seed,
        "data_cache_sha256": cache,
        "parameter_count_each": parameters,
        "models": {
            "a_autonomous": {
                "query_conditioned": False,
                "parameter_count": parameters,
                "evaluation": evaluation(1e-9, 5e-8),
            },
            "b_query_time": {
                "query_conditioned": True,
                "parameter_count": parameters,
                "evaluation": evaluation(1e-4, 6e-8),
            },
        },
        "comparison": {
            "mse_a_over_b": {cell: mse_ratio for cell in EXPECTED_CELLS}
        },
    }


def write_matrix(tmp_path: Path, summaries) -> list[Path]:
    paths = []
    for summary in summaries:
        path = tmp_path / f"s{summary['seed']}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)
    return paths


def run(tmp_path: Path, summaries):
    paths = write_matrix(tmp_path, summaries)
    output = tmp_path / "aggregate.json"
    main([str(path) for path in paths] + ["--output", str(output)])
    return json.loads(output.read_text(encoding="utf-8"))


def test_material_accuracy_advantage_is_reported(tmp_path):
    result = run(
        tmp_path,
        [build_summary(seed, mse_ratio=0.8) for seed in (31415, 271828, 161803)],
    )
    assert result["conclusion"] == "autonomous_materially_more_accurate"
    assert result["seeds_favoring_autonomous"] == 3
    assert result["cells"] == 12


def test_tied_accuracy_keeps_the_composition_only_conclusion(tmp_path):
    result = run(
        tmp_path,
        [build_summary(seed, mse_ratio=1.001) for seed in (31415, 271828, 161803)],
    )
    assert (
        result["conclusion"]
        == "composition_advantage_without_material_accuracy_advantage"
    )
    assert result["seeds_favoring_autonomous"] == 0
    assert result["equal_work_defect"]["cells_with_lower_autonomous_defect"] == 12


def test_mixed_cache_digests_are_rejected(tmp_path):
    summaries = [build_summary(seed) for seed in (31415, 271828, 161803)]
    summaries[1]["data_cache_sha256"] = "other-digest"
    with pytest.raises(ValueError, match="one immutable data cache"):
        run(tmp_path, summaries)


def test_unmatched_rhs_work_is_rejected(tmp_path):
    summaries = [build_summary(seed) for seed in (31415, 271828, 161803)]
    cell = EXPECTED_CELLS[0]
    summaries[0]["models"]["b_query_time"]["evaluation"]["composition"][cell][
        "rhs_evaluations_each_path"
    ] = 960
    with pytest.raises(ValueError, match="unmatched RHS work"):
        run(tmp_path, summaries)


def test_formal_classification_is_rejected(tmp_path):
    summaries = [build_summary(seed) for seed in (31415, 271828, 161803)]
    summaries[2]["do_not_use_for_formal"] = False
    with pytest.raises(ValueError, match="classification mismatch"):
        run(tmp_path, summaries)
