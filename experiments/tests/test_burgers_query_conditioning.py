"""Checks for the Burgers query-conditioning attribution evaluator."""

from __future__ import annotations

import math

import pytest
import torch

from experiments.evaluate_burgers_query_conditioning import (
    classify,
    main,
    measure_model,
)
from experiments.run_burgers_semigroup_screen import PeriodicBurgersFluxFlow


def build_model(query_conditioned):
    torch.manual_seed(11)
    return PeriodicBurgersFluxFlow(
        n_grid=16,
        length=2.0 * math.pi,
        viscosity=0.01,
        hidden=8,
        ode_steps=2,
        query_conditioned=query_conditioned,
    ).eval()


def test_autonomous_model_ignores_the_direct_path_conditioning():
    """Zeroing the control channel must make paths 1 and 2 identical."""

    model = build_model(False)
    sample = torch.randn(3, 16)
    measured = measure_model(model, sample)
    for key, row in measured["path1_frozen_semantics"].items():
        matched = measured["path2_matched_conditioning"][key]
        assert row["equal_work_rms_defect_mean"] == pytest.approx(
            matched["equal_work_rms_defect_mean"], rel=0, abs=0
        )


def test_query_model_separates_the_two_direct_path_semantics():
    model = build_model(True)
    sample = torch.randn(3, 16)
    measured = measure_model(model, sample)
    differences = [
        abs(
            measured["path1_frozen_semantics"][key]["equal_work_rms_defect_mean"]
            - measured["path2_matched_conditioning"][key][
                "equal_work_rms_defect_mean"
            ]
        )
        for key in measured["path1_frozen_semantics"]
    ]
    assert max(differences) > 0.0


def test_cross_lag_paths_receive_equal_rhs_work():
    model = build_model(True)
    measured = measure_model(model, torch.randn(2, 16))
    for row in measured["path3_in_range_cross_lag"].values():
        fine_work = 4 * row["fine_depth"] * model.ode_steps
        coarse_work = 4 * row["coarse_depth"] * row["coarse_substeps_per_call"]
        assert fine_work == coarse_work == row["rhs_evaluations_each_path"]


def test_classification_bands():
    assert classify(5e-8, 1e-8) == "collapsed"
    assert classify(1e-6, 1e-8) == "structural"
    assert classify(5e-7, 1e-8) == "intermediate"
    with pytest.raises(ValueError, match="integrator floor"):
        classify(1.0, 0.0)


def test_cache_digest_mismatch_aborts(tmp_path):
    cache = tmp_path / "cache.pt"
    torch.save({"config": {"N": 16, "L": 1.0, "nu": 0.01}, "val_u0": torch.zeros(2, 16)}, cache)
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": build_model(False).state_dict()}, checkpoint)
    with pytest.raises(RuntimeError, match="does not match the frozen screen cache"):
        main(
            [
                "--autonomous-checkpoint",
                str(checkpoint),
                "--query-checkpoint",
                str(checkpoint),
                "--data-cache",
                str(cache),
                "--expect-cache-sha256",
                "0" * 64,
                "--output",
                str(tmp_path / "out.json"),
                "--seed",
                "1",
            ]
        )
