"""Checks for the minimal-generator Burgers A/B lane."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from experiments.aggregate_burgers_minimal_ab import ARCHITECTURE, main
from experiments.run_burgers_minimal_ab import (
    EXPECTED_PARAMETERS,
    build_pair,
)

CELLS = (
    "tau=0.04:horizon=0.4",
    "tau=0.04:horizon=0.8",
    "tau=0.08:horizon=0.4",
    "tau=0.08:horizon=0.8",
)
CROSS = ("horizon=0.4", "horizon=0.8")


def test_paired_minimal_models_match_and_are_the_recommended_variant():
    torch.manual_seed(3)
    autonomous, query = build_pair(
        {"N": 16, "L": 2.0 * math.pi, "nu": 0.01}, 2
    )
    assert autonomous.radius == 0
    assert autonomous.query_conditioned is False
    assert query.query_conditioned is True
    for model in (autonomous, query):
        assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMETERS
    for left, right in zip(autonomous.parameters(), query.parameters()):
        assert torch.equal(left, right)


def test_autonomous_minimal_model_ignores_the_requested_lag():
    torch.manual_seed(3)
    autonomous, query = build_pair(
        {"N": 16, "L": 2.0 * math.pi, "nu": 0.01}, 2
    )
    state = torch.randn(2, 16)
    with torch.no_grad():
        assert torch.allclose(autonomous.rhs(state, 0.04), autonomous.rhs(state, 0.8))
        # Before training the pair shares weights, so B must agree with A only
        # where the control input vanishes.
        assert not torch.allclose(query.rhs(state, 0.04), query.rhs(state, 0.8))


def build_summary(seed, *, mse_ratio, verdict="structural", architecture=ARCHITECTURE):
    def cross(defect):
        return {
            cell: {
                "equal_work_rms_defect_mean": defect,
                "composed_state_rms": 0.46,
            }
            for cell in CROSS
        }

    return {
        "experiment": "burgers_minimal_generator_ab",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "architecture": architecture,
        "seed": seed,
        "data_cache_sha256": "cache-digest",
        "parameter_count_each": EXPECTED_PARAMETERS,
        "models": {
            "a_autonomous": {"parameter_count": EXPECTED_PARAMETERS},
            "b_query_time": {"parameter_count": EXPECTED_PARAMETERS},
        },
        "composition_paths": {
            "a_autonomous": {
                "path2_matched_conditioning": {
                    cell: {"equal_work_rms_defect_mean": 4e-8} for cell in CELLS
                },
                "path3_in_range_cross_lag": cross(0.0),
            },
            "b_query_time": {
                "path2_matched_conditioning": {
                    cell: {"equal_work_rms_defect_mean": 4e-8} for cell in CELLS
                },
                "path3_in_range_cross_lag": cross(5e-3),
            },
        },
        "in_range_cross_lag_verdict": {cell: verdict for cell in CROSS},
        "comparison": {
            "mse_a_over_b": {cell: mse_ratio for cell in CELLS},
            "mse_a_over_b_geometric_mean": mse_ratio,
        },
    }


def run(tmp_path: Path, summaries):
    paths = []
    for summary in summaries:
        path = tmp_path / f"s{summary['seed']}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "aggregate.json"
    main([str(path) for path in paths] + ["--output", str(output)])
    return json.loads(output.read_text(encoding="utf-8"))


def test_both_conclusions_unchanged_adopts_the_reduction(tmp_path):
    result = run(
        tmp_path,
        [build_summary(seed, mse_ratio=0.99) for seed in (31415, 271828, 161803)],
    )
    assert result["accuracy_conclusion"] == "accuracy_conclusion_unchanged"
    assert result["structure_conclusion"] == "structure_conclusion_unchanged"
    assert result["adopt_reduction"] is True
    assert result["in_range_structural_cells"] == "6/6"


def test_material_accuracy_gain_flips_the_accuracy_conclusion(tmp_path):
    result = run(
        tmp_path,
        [build_summary(seed, mse_ratio=0.8) for seed in (31415, 271828, 161803)],
    )
    assert result["accuracy_conclusion"] == "accuracy_conclusion_changed"
    assert result["adopt_reduction"] is False


def test_collapsed_in_range_defect_flips_the_structure_conclusion(tmp_path):
    result = run(
        tmp_path,
        [
            build_summary(seed, mse_ratio=0.99, verdict="collapsed")
            for seed in (31415, 271828, 161803)
        ],
    )
    assert result["structure_conclusion"] == "structure_conclusion_changed"
    assert result["adopt_reduction"] is False


def test_one_nonstructural_seed_flips_the_structure_conclusion(tmp_path):
    summaries = [
        build_summary(seed, mse_ratio=0.99) for seed in (31415, 271828, 161803)
    ]
    summaries[1]["in_range_cross_lag_verdict"]["horizon=0.4"] = "intermediate"
    result = run(tmp_path, summaries)
    assert result["structure_conclusion"] == "structure_conclusion_changed"


def test_foreign_architecture_is_rejected(tmp_path):
    summaries = [
        build_summary(seed, mse_ratio=0.99) for seed in (31415, 271828, 161803)
    ]
    summaries[0]["architecture"] = "r2_h32_l2_anchor"
    with pytest.raises(ValueError, match="only accepts architecture"):
        run(tmp_path, summaries)
