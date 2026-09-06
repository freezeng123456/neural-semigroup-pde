"""Acceptance checks for the Burgers query-conditioning aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aggregate_burgers_query_conditioning import main

CELLS = (
    "tau=0.04:horizon=0.4",
    "tau=0.04:horizon=0.8",
    "tau=0.08:horizon=0.4",
    "tau=0.08:horizon=0.8",
)
CROSS = ("horizon=0.4", "horizon=0.8")


def build_result(
    seed,
    *,
    path3_query=5e-3,
    path3_autonomous=0.0,
    path2_verdict="collapsed",
    path3_verdict="structural",
    cache="cache-digest",
):
    def rows(defect):
        return {
            cell: {"equal_work_rms_defect_mean": defect} for cell in CELLS
        }

    def cross(defect):
        return {
            cell: {
                "equal_work_rms_defect_mean": defect,
                "composed_state_rms": 0.46,
            }
            for cell in CROSS
        }

    return {
        "experiment": "burgers_query_conditioning_attribution",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": seed,
        "n_sample": 16,
        "inputs": {"data_cache_sha256": cache},
        "measurements": {
            "a_autonomous": {
                "path1_frozen_semantics": rows(4e-8),
                "path2_matched_conditioning": rows(4e-8),
                "path3_in_range_cross_lag": cross(path3_autonomous),
            },
            "b_query_time": {
                "path1_frozen_semantics": rows(0.16),
                "path2_matched_conditioning": rows(4.4e-8),
                "path3_in_range_cross_lag": cross(path3_query),
            },
        },
        "query_time_classification": {
            "path1_frozen_semantics": {cell: "structural" for cell in CELLS},
            "path2_matched_conditioning": {cell: path2_verdict for cell in CELLS},
            "path3_in_range_cross_lag": {cell: path3_verdict for cell in CROSS},
        },
    }


def run(tmp_path: Path, results):
    paths = []
    for result in results:
        path = tmp_path / f"s{result['seed']}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "aggregate.json"
    main([str(path) for path in paths] + ["--output", str(output)])
    return json.loads(output.read_text(encoding="utf-8"))


def test_artifact_plus_in_range_structure_is_the_pooled_conclusion(tmp_path):
    result = run(
        tmp_path, [build_result(seed) for seed in (31415, 271828, 161803)]
    )
    assert result["conclusion"] == (
        "headline_defect_is_conditioning_artifact_in_range_defect_is_structural"
    )
    assert result["unanimity"] == {
        "path1_structural": "12/12",
        "path2_collapsed": "12/12",
        "path3_structural": "6/6",
    }
    assert result["in_range_defect_relative_to_state_rms"] == pytest.approx(
        5e-3 / 0.46
    )


def test_collapsed_in_range_defect_withdraws_the_structural_claim(tmp_path):
    result = run(
        tmp_path,
        [
            build_result(seed, path3_verdict="collapsed")
            for seed in (31415, 271828, 161803)
        ],
    )
    assert result["conclusion"] == (
        "headline_defect_is_conditioning_artifact_no_in_range_defect"
    )


def test_surviving_headline_defect_is_reported(tmp_path):
    result = run(
        tmp_path,
        [
            build_result(seed, path2_verdict="structural")
            for seed in (31415, 271828, 161803)
        ],
    )
    assert result["conclusion"] == "headline_defect_survives_matched_conditioning"


def test_nonzero_autonomous_cross_lag_defect_is_rejected(tmp_path):
    results = [build_result(seed) for seed in (31415, 271828, 161803)]
    results[1]["measurements"]["a_autonomous"]["path3_in_range_cross_lag"][
        "horizon=0.4"
    ]["equal_work_rms_defect_mean"] = 1e-12
    with pytest.raises(ValueError, match="must agree exactly"):
        run(tmp_path, results)


def test_mixed_cache_digests_are_rejected(tmp_path):
    results = [build_result(seed) for seed in (31415, 271828, 161803)]
    results[0]["inputs"]["data_cache_sha256"] = "other"
    with pytest.raises(ValueError, match="one immutable data cache"):
        run(tmp_path, results)
