"""Acceptance and artifact-integrity checks using synthetic completed runs."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from experiments.aggregate_burgers_validated import (
    ARCHITECTURES, CELLS, EXPECTED_SEEDS, HORIZONS, LEARNED_MODELS, MODELS,
    ORACLE, PARAMETERS, SCHEMA, SOURCE_FILES, aggregate, configuration_digest,
    main, sha256_file,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def fake_evaluation(mses, query=False):
    evaluation = {key: {} for key in ("one_step", "rollouts", "composition_matched_conditioning",
                  "composition_historical_conditioning", "cross_lag", "time_integration_refinement")}
    raw = {**copy.deepcopy(evaluation), "sample_indices": list(range(128))}
    for tau in ("0.04", "0.08"):
        evaluation["one_step"][tau] = {"mse_mean": .001, "relative_l2_mean": .1, "undefined_relative_l2_count": 0}
        raw["one_step"][tau] = {"mse": [.001] * 128, "relative_l2": [.1] * 128}
    for i, condition in enumerate(CELLS):
        tau, horizon = [float(part.split("=")[1]) for part in condition.split(":")]
        substeps, depth = round(12 * tau / .04), round(horizon / tau)
        mse = mses[i]
        evaluation["rollouts"][condition] = {
            "mse_mean": mse, "relative_l2_mean": math.sqrt(mse), "undefined_relative_l2_count": 0,
            "mean_drift_max": 2e-8, "energy_monotone_fraction": .5 if query else 1.,
            "energy_positive_increment_sum_mean": 1e-4 if query else 0.,
            "energy_max_positive_increment": 2e-4 if query else 0.,
            "energy_horizon_delta_mean": -.01, "energy_increase_step_count_total": 64 if query else 0,
            "energy_tolerance": 1e-8, "rk4_substeps_per_call": substeps, "rhs_evaluations": 4 * substeps * depth,
        }
        raw["rollouts"][condition] = {
            "mse": [mse] * 128, "relative_l2": [math.sqrt(mse)] * 128,
            "mean_drift": [2e-8] * 128,
            "energy_monotone": [True] * 64 + [not query] * 64,
        }
        for section in ("composition_matched_conditioning", "composition_historical_conditioning"):
            defect = .02 if query and section == "composition_historical_conditioning" else 0.
            evaluation[section][condition] = {
                "rms_defect_mean": defect, "rms_defect_max": defect,
                "direct_conditioning_tau": horizon if section == "composition_historical_conditioning" else tau,
                "rhs_evaluations_each_path": 4 * substeps * depth,
            }
            raw[section][condition] = [defect] * 128
        evaluation["time_integration_refinement"][condition] = {
            "mse_vs_production_mean": 1e-12, "rms_vs_production_mean": 1e-6,
            "rms_vs_production_max": 1e-6, "production_substeps_per_call": substeps,
            "refined_substeps_per_call": 2 * substeps,
        }
        raw["time_integration_refinement"][condition] = {"mse_vs_production": [1e-12] * 128, "rms_vs_production": [1e-6] * 128}
    for horizon in HORIZONS:
        defect = .003 if query else 0.
        evaluation["cross_lag"][horizon] = {"rms_defect_mean": defect, "rms_defect_max": defect,
            "fine_substeps": 12, "coarse_substeps": 24, "rhs_evaluations_each_path": round(48 * float(horizon) / .04)}
        raw["cross_lag"][horizon] = [defect] * 128
    return evaluation, raw


def refresh(root, summary=None):
    if summary is None:
        summary = json.loads((root / "summary.json").read_text())
    summary["config_sha256"] = configuration_digest(summary["configuration"])
    for label, cell in summary["cells"].items():
        write_json(root / label / "summary.json", cell)
    selection = {label: {key: value for key, value in summary["cells"][label].items()
                         if key not in ("evaluation", "eval_seconds")} for label in LEARNED_MODELS if label in summary["cells"]}
    write_json(root / "selection_complete.json", selection)
    write_json(root / "summary.json", summary)
    write_json(root / "COMPLETE.json", {"schema_version": SCHEMA,
        "summary_sha256": sha256_file(root / "summary.json"),
        "cache_sha256": summary["cache_sha256_after"], "config_sha256": summary["config_sha256"]})


def build_matrix(tmp_path, ratios=(.8, .8, .8)):
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"synthetic immutable cache; aggregator must not unpickle")
    cache_hash = sha256_file(cache)
    roots = []
    for seed, ratio in zip(EXPECTED_SEEDS, ratios):
        root = tmp_path / f"s{seed}"
        root.mkdir()
        roots.append(root)
        config = {
            "cache_config": {"N": 64, "L": 2 * math.pi, "nu": .01, "reference_N": 512,
                "reference_dt": .0005, "n_train": 1000, "n_val": 50, "n_test": 128,
                "train_taus": [.025, .05, .075, .1], "test_taus": [.04, .08], "horizons": [.4, .8],
                "data_seeds": {"train": 2026100301, "val": 2026100302, "test": 2026100303}},
            "settings": {"seed": seed, "output_dir": str(root), "data_cache": str(cache),
                "expect_cache_sha256": cache_hash, "epochs": 100, "batch_size": 64, "eval_batch_size": 32,
                "validation_interval": 10, "ode_steps": 12, "lr": .001, "weight_decay": 1e-5,
                "threads": 1, "device": "cpu", "architectures": list(ARCHITECTURES)},
            "architectures": ARCHITECTURES, "energy_tolerance": 1e-8,
        }
        cells = {}
        for label in MODELS:
            (root / label).mkdir()
            if label == ORACLE:
                multiplier, query = .25, False
                cell = {"parameter_count": 0, "description": "fixed exact flux"}
            else:
                architecture, temporal = label.split("_")
                query = temporal == "query"
                multiplier = (1. if architecture == "small" else 2.) * (1. if query else ratio if architecture == "small" else 1.1)
                cell = {"architecture": architecture, "query_conditioned": query,
                    "parameter_count": PARAMETERS[architecture],
                    "initial_state_sha256": hashlib.sha256(f"{seed}:{architecture}".encode()).hexdigest(),
                    "selected_epoch": 100, "best_val_mse": .001,
                    "train_and_validation_seconds": 10., "validation_seconds": 2., "train_seconds_excluding_validation": 8.}
                for stage in ("selected", "final"):
                    path = root / label / f"{stage}.pt"
                    path.write_bytes(f"synthetic {seed} {label} {stage}".encode())
                    cell[f"{stage}_sha256"] = sha256_file(path)
            cell["evaluation"], raw = fake_evaluation([.001 * (i + 1) * multiplier for i in range(4)], query)
            cell["eval_seconds"] = 1.
            cells[label] = cell
            write_json(root / label / "per_sample.json", raw)
        summary = {"schema_version": SCHEMA, "scope": "full", "configuration": config,
            "exploratory": True, "do_not_use_for_formal": True,
            "sources": {name: hashlib.sha256(name.encode()).hexdigest() for name in SOURCE_FILES},
            "cache_sha256_before": cache_hash, "cache_sha256_after": cache_hash, "cells": cells}
        refresh(root, summary)
    return roots, cache


@pytest.fixture
def matrix(tmp_path):
    return build_matrix(tmp_path)


def test_complete_study_has_paired_accuracy_size_physics_and_absolute_defects(matrix, tmp_path):
    roots, _ = matrix
    result = main([*(str(path) for path in roots), "--output-dir", str(tmp_path / "aggregate")])
    assert result["accuracy_by_architecture"]["small"]["geometric_mean_mse_ratio"] == pytest.approx(.8)
    assert result["accuracy_by_architecture"]["small"]["material_accuracy_advantage"] is True
    assert result["accuracy_by_architecture"]["large"]["material_accuracy_advantage"] is False
    assert result["small_over_large_by_temporal_rule"]["autonomous"]["geometric_mean_mse_ratio"] == pytest.approx(4 / 11)
    assert result["small_over_large_by_temporal_rule"]["query"]["geometric_mean_mse_ratio"] == pytest.approx(.5)
    assert result["model_metrics"]["small_autonomous"]["cross_lag"]["rms_defect_mean"] == 0
    assert result["model_metrics"]["small_query"]["cross_lag"]["rms_defect_mean"] == pytest.approx(.003)
    assert result["model_metrics"]["small_query"]["energy_monotone_fraction_min"] == .5
    assert result["model_metrics"]["small_autonomous"]["spatial_mean_drift_max"] == 2e-8
    assert result["model_metrics"]["small_autonomous"]["time_integration_refinement"]["rms_vs_production_mean"] == pytest.approx(1e-6)
    assert result["reference_oracle_mse"][CELLS[0]]["mean_over_seeds"] == pytest.approx(.00025)
    assert len(result["accuracy_by_architecture"]["small"]["cell_mse_ratios"]) == 12
    assert result["exploratory"] is True and result["do_not_use_for_formal"] is True
    with (tmp_path / "aggregate/cells.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 60


def test_material_decision_requires_two_seeds_even_when_pooled_ratio_passes(tmp_path):
    roots, _ = build_matrix(tmp_path, ratios=(.5, 1.05, 1.05))
    result, _, _ = aggregate(roots)
    small = result["accuracy_by_architecture"]["small"]
    assert small["geometric_mean_mse_ratio"] == pytest.approx((.5 * 1.05 ** 2) ** (1 / 3))
    assert small["geometric_mean_mse_ratio"] < .9
    assert small["seeds_favoring_numerator"] == 1
    assert small["material_accuracy_advantage"] is False


@pytest.mark.parametrize("fault", ["missing_marker", "wrong_summary_hash", "false_scope", "missing_model",
    "missing_condition", "cache_changed", "before_after_mismatch", "checkpoint_changed", "pair_initial_mismatch",
    "wrong_parameters", "different_source", "different_config", "duplicate_seed", "wrong_refinement_work",
    "sample_count", "sample_summary_mismatch", "nonfinite"])
def test_invalid_or_incomplete_matrix_is_rejected(matrix, fault):
    roots, cache = matrix
    root = roots[0]
    summary = json.loads((root / "summary.json").read_text())
    if fault == "missing_marker":
        (root / "COMPLETE.json").unlink()
    elif fault == "wrong_summary_hash":
        with (root / "summary.json").open("a") as handle:
            handle.write(" ")
    elif fault == "cache_changed":
        cache.write_bytes(b"changed")
    elif fault == "checkpoint_changed":
        (root / "small_autonomous/selected.pt").write_bytes(b"changed")
    elif fault in ("sample_count", "sample_summary_mismatch", "nonfinite"):
        path = root / "small_autonomous/per_sample.json"
        raw = json.loads(path.read_text())
        if fault == "sample_count":
            raw["rollouts"][CELLS[0]]["mse"].pop()
        elif fault == "sample_summary_mismatch":
            raw["rollouts"][CELLS[0]]["mse"][0] += .5
        else:
            raw["rollouts"][CELLS[0]]["mse"][0] = float("nan")
        path.write_text(json.dumps(raw))
    else:
        if fault == "false_scope": summary["scope"] = "pilot"
        elif fault == "missing_model": del summary["cells"]["large_query"]
        elif fault == "missing_condition": del summary["cells"]["small_autonomous"]["evaluation"]["rollouts"][CELLS[0]]
        elif fault == "before_after_mismatch": summary["cache_sha256_before"] = "a" * 64
        elif fault == "pair_initial_mismatch": summary["cells"]["small_query"]["initial_state_sha256"] = "a" * 64
        elif fault == "wrong_parameters": summary["cells"]["small_query"]["parameter_count"] = 34
        elif fault == "different_source": summary["sources"]["run_burgers_validated.py"] = "a" * 64
        elif fault == "different_config": summary["configuration"]["settings"]["eval_batch_size"] = 16
        elif fault == "duplicate_seed": summary["configuration"]["settings"]["seed"] = EXPECTED_SEEDS[1]
        elif fault == "wrong_refinement_work": summary["cells"]["small_autonomous"]["evaluation"]["time_integration_refinement"][CELLS[0]]["refined_substeps_per_call"] = 12
        refresh(root, summary)
    with pytest.raises(ValueError):
        aggregate(roots)


def test_portable_cache_override_is_accepted_only_for_identical_bytes(matrix, tmp_path):
    roots, cache = matrix
    for root in roots:
        summary = json.loads((root / "summary.json").read_text())
        summary["configuration"]["settings"]["data_cache"] = "missing-relative-path/cache.pt"
        refresh(root, summary)
    with pytest.raises(ValueError, match="live cache"):
        aggregate(roots)
    result, _, _ = aggregate(roots, data_cache=cache)
    assert result["trained_model_count"] == 12
    other = tmp_path / "wrong-cache.pt"
    other.write_bytes(b"different cache")
    with pytest.raises(ValueError, match="live cache"):
        aggregate(roots, data_cache=other)


def test_duplicate_directories_and_existing_output_are_rejected(matrix, tmp_path):
    roots, _ = matrix
    with pytest.raises(ValueError, match="distinct seed directories"):
        aggregate([roots[0], roots[0], roots[2]])
    with pytest.raises(ValueError, match="exactly three"):
        aggregate(roots[:2])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        main([*(str(path) for path in roots), "--output-dir", str(tmp_path)])


def test_duplicate_json_keys_are_rejected_even_with_a_valid_marker(matrix):
    roots, _ = matrix
    path = roots[0] / "COMPLETE.json"
    text = path.read_text()
    path.write_text(text[:-1] + ', "schema_version": "burgers-paired-v2"}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        aggregate(roots)
