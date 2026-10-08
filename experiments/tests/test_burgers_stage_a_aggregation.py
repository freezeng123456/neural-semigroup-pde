"""Reject incomplete, unpaired, altered, or numerically inconsistent studies."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math

import pytest

from experiments import aggregate_burgers_stage_a as aggregator


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def refresh(root, summary=None):
    if summary is None:
        summary = json.loads((root / "summary.json").read_text())
    summary["config_sha256"] = aggregator.configuration_digest(summary["configuration"])
    write_json(root / "manifest.json", {key: value for key, value in summary.items() if key != "cells"})
    for label, cell in summary["cells"].items():
        write_json(root / label / "summary.json", cell)
    write_json(root / "selection_complete.json", {
        label: {key: value for key, value in summary["cells"][label].items() if key not in ("evaluation", "eval_seconds")}
        for label in aggregator.TRAINED if label in summary["cells"]})
    write_json(root / "summary.json", summary)
    write_json(root / "COMPLETE.json", {"schema_version": aggregator.SCHEMA,
        "summary_sha256": aggregator.sha256_file(root / "summary.json"),
        "cache_sha256": summary["cache_sha256"], "protocol_sha256": summary["protocol_sha256"]})


def build_matrix(tmp_path, small_ratios=(.8, .8, .8), large_ratios=(1.1, 1.1, 1.1)):
    cache = tmp_path / "cache.pt"
    cache.write_bytes(b"deliberately not a pickle; aggregation must not load checkpoints or cache")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen synthetic protocol")
    sources = tmp_path / "source"
    sources.mkdir()
    (sources / "runner.py").write_text("# synthetic runner\n")
    hashes = {"runner.py": aggregator.sha256_file(sources / "runner.py")}
    roots = []
    for index, seed in enumerate(aggregator.EXPECTED_SEEDS):
        root = tmp_path / f"s{seed}"
        root.mkdir()
        roots.append(root)
        cells = {}
        for label in aggregator.MODELS:
            cell_dir = root / label
            cell_dir.mkdir()
            cell = {"parameter_count": aggregator.PARAMETERS.get(label, 0), "eval_seconds": .1}
            if label in aggregator.TRAINED:
                pair = "small" if label.startswith("small_") else label.split("_")[0]
                training = {"selected_epoch": 90, "epochs_completed": 100, "optimizer_updates": 1600,
                    "batch_order_sha256": digest(f"order:{seed}"), "train_seconds": 2., "validation_seconds": .2,
                    "network_evaluations_per_training_sample": aggregator.CALLS_PER_TRAINING_SAMPLE[label],
                    "network_evaluations_training": 100000 * aggregator.CALLS_PER_TRAINING_SAMPLE[label],
                    "peak_cuda_memory_bytes": 2048}
                for stage in ("selected", "final"):
                    path = cell_dir / f"{stage}.pt"
                    path.write_bytes(f"synthetic checkpoint {seed} {label} {stage}".encode())
                    training[f"{stage}_sha256"] = aggregator.sha256_file(path)
                cell.update(initial_state_sha256=digest(f"initial:{seed}:{pair}"), training=training)
            multiplier = .95 if label == "small_autonomous" else small_ratios[index] if label == "osg33_gdsg" else large_ratios[index] if label == "osg1313_gdsg" else 1.
            evaluation = {"rollouts": {}, "cross_lag": {}, "composition_in_range": {}}
            raw = {"sample_indices": list(range(128)), "rollouts": {}, "cross_lag": {}, "composition_in_range": {}}
            for condition_index, key in enumerate(aggregator.CONDITIONS):
                mse = .001 * (condition_index + 1) * multiplier
                relative = math.sqrt(mse)
                evaluation["rollouts"][key] = {"mse_mean": mse, "relative_l2_mean": relative,
                    "mean_drift_max": 1e-8, "energy_monotone_fraction": 1.,
                    "spectral_error_mean": {band: mse for band in aggregator.SPECTRAL_BANDS}}
                raw["rollouts"][key] = {"mse": [mse] * 128, "relative_l2": [relative] * 128,
                    "mean_drift": [1e-8] * 128, "energy_monotone": [True] * 128,
                    "spectral_error": {band: [mse] * 128 for band in aggregator.SPECTRAL_BANDS}}
            for section, keys in (("cross_lag", ("0.4", "0.8")), ("composition_in_range", ("0.04+0.04=0.08",))):
                for key in keys:
                    evaluation[section][key] = {"rms_defect_mean": 0.}
                    raw[section][key] = {"rms_defect": [0.] * 128}
            cell["evaluation"] = evaluation
            cells[label] = cell
            write_json(cell_dir / "per_sample.json", raw)
        summary = {"schema_version": aggregator.SCHEMA, "seed": seed, "run_kind": "full",
            "exploratory": True, "do_not_use_for_formal": True, "source_hashes": hashes,
            "cache_sha256": aggregator.sha256_file(cache), "protocol_sha256": aggregator.sha256_file(protocol),
            "configuration": {"epochs": 100, "batch_size": 64, "n_train": 1000, "n_val": 50, "n_test": 128,
                "validation_interval": 10, "ode_steps": 12, "lr": .001, "weight_decay": 1e-5,
                "seed": seed, "output_dir": str(root), "device": "cuda"}, "cells": cells}
        refresh(root, summary)
    return roots, cache, protocol, sources


@pytest.fixture
def matrix(tmp_path):
    return build_matrix(tmp_path)


def test_full_matrix_has_two_separate_primary_comparisons_and_auditable_raw_tables(matrix, tmp_path):
    roots, cache, protocol, sources = matrix
    result = aggregator.main([*(str(root) for root in roots), "--output-dir", str(tmp_path / "aggregate"),
        "--data-cache", str(cache), "--protocol-file", str(protocol), "--source-dir", str(sources)])
    assert result["trained_model_count"] == 18
    assert result["rollout_cell_count"] == 108
    assert result["primary_osg_comparisons"]["osg33"]["geometric_mean_mse_ratio"] == pytest.approx(.8)
    assert result["primary_osg_comparisons"]["osg33"]["material_accuracy_advantage"] is True
    assert result["primary_osg_comparisons"]["osg1313"]["material_accuracy_advantage"] is False
    assert result["secondary_small_autonomous_over_query"]["geometric_mean_mse_ratio"] == pytest.approx(.95)
    assert result["exploratory"] is True and result["do_not_use_for_formal"] is True
    assert all(result["provenance"]["external_input_verification"].values())
    assert len(result["primary_osg_comparisons"]["osg33"]["cell_mse_ratios"]) == 12
    for table, count in (("cells", 108), ("costs", 27), ("per_sample", 13824)):
        with (tmp_path / "aggregate" / f"{table}.csv").open() as handle:
            assert len(list(csv.DictReader(handle))) == count
    receipt = json.loads((tmp_path / "aggregate/COMPLETE.json").read_text())
    for name, sha in receipt["artifact_sha256"].items():
        assert sha == aggregator.sha256_file(tmp_path / "aggregate" / name)


def test_seed_direction_gate_cannot_be_replaced_by_pooled_accuracy(tmp_path):
    roots, *_ = build_matrix(tmp_path, small_ratios=(.5, 1.05, 1.05))
    result, _, _ = aggregator.aggregate(roots)
    comparison = result["primary_osg_comparisons"]["osg33"]
    assert comparison["geometric_mean_mse_ratio"] == pytest.approx((.5 * 1.05 ** 2) ** (1 / 3))
    assert comparison["geometric_mean_mse_ratio"] < .9
    assert comparison["seeds_favoring_numerator"] == 1
    assert comparison["material_accuracy_advantage"] is False
    assert not any(result["provenance"]["external_input_verification"].values())


@pytest.mark.parametrize("fault", ["missing_marker", "summary_hash", "pilot", "missing_model", "missing_condition",
    "checkpoint", "initial_pair", "batch_order_pair", "source_mismatch", "cache_mismatch", "protocol_mismatch",
    "configuration", "duplicate_seed", "sample_count", "sample_mean", "relative_mean", "drift_max", "energy_fraction",
    "nonfinite", "duplicate_json_key", "selection_changed", "manifest_changed", "short_training", "updates",
    "selected_epoch", "parameter_count", "zero_denominator", "network_calls", "spectral_mean", "composition_mean"])
def test_incomplete_corrupted_or_unpaired_inputs_are_rejected(matrix, fault):
    roots, *_ = matrix
    root = roots[0]
    summary = json.loads((root / "summary.json").read_text())
    if fault == "missing_marker":
        (root / "COMPLETE.json").unlink()
    elif fault == "summary_hash":
        with (root / "summary.json").open("a") as handle:
            handle.write(" ")
    elif fault == "checkpoint":
        (root / "osg33_free/selected.pt").write_bytes(b"changed")
    elif fault in ("sample_count", "sample_mean", "relative_mean", "drift_max", "energy_fraction", "nonfinite", "spectral_mean", "composition_mean"):
        path = root / "osg33_free/per_sample.json"
        raw = json.loads(path.read_text())
        row = raw["rollouts"][aggregator.CONDITIONS[0]]
        if fault == "sample_count": row["mse"].pop()
        elif fault == "sample_mean": row["mse"][0] += 1
        elif fault == "relative_mean": row["relative_l2"][0] += 1
        elif fault == "drift_max": row["mean_drift"][0] += 1
        elif fault == "energy_fraction": row["energy_monotone"][0] = False
        elif fault == "nonfinite": row["mse"][0] = float("nan")
        elif fault == "spectral_mean": row["spectral_error"][aggregator.SPECTRAL_BANDS[0]][0] += 1.
        elif fault == "composition_mean": raw["composition_in_range"]["0.04+0.04=0.08"]["rms_defect"][0] += .01
        path.write_text(json.dumps(raw))
    elif fault == "duplicate_json_key":
        path = root / "COMPLETE.json"
        path.write_text(path.read_text()[:-1] + ', "schema_version": "duplicate"}')
    elif fault == "selection_changed":
        path = root / "selection_complete.json"
        selected = json.loads(path.read_text())
        selected["osg33_free"]["training"]["selected_epoch"] = 80
        write_json(path, selected)
    elif fault == "manifest_changed":
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["cache_sha256"] = "a" * 64
        write_json(path, manifest)
    else:
        if fault == "pilot": summary["run_kind"] = "pilot"
        elif fault == "missing_model": del summary["cells"]["zero"]
        elif fault == "missing_condition": del summary["cells"]["osg33_free"]["evaluation"]["rollouts"][aggregator.CONDITIONS[0]]
        elif fault == "initial_pair": summary["cells"]["osg33_gdsg"]["initial_state_sha256"] = "b" * 64
        elif fault == "batch_order_pair": summary["cells"]["osg33_gdsg"]["training"]["batch_order_sha256"] = "b" * 64
        elif fault == "source_mismatch": summary["source_hashes"]["runner.py"] = "b" * 64
        elif fault == "cache_mismatch": summary["cache_sha256"] = "b" * 64
        elif fault == "protocol_mismatch": summary["protocol_sha256"] = "b" * 64
        elif fault == "configuration": summary["configuration"]["weight_decay"] = 1e-4
        elif fault == "duplicate_seed": summary["seed"] = aggregator.EXPECTED_SEEDS[1]
        elif fault == "short_training": summary["cells"]["osg33_free"]["training"]["epochs_completed"] = 90
        elif fault == "updates": summary["cells"]["osg33_free"]["training"]["optimizer_updates"] = 1500
        elif fault == "selected_epoch": summary["cells"]["osg33_free"]["training"]["selected_epoch"] = 93
        elif fault == "parameter_count": summary["cells"]["osg33_free"]["parameter_count"] = 34
        elif fault == "network_calls": summary["cells"]["osg33_gdsg"]["training"]["network_evaluations_training"] = 100000
        elif fault == "zero_denominator":
            summary["cells"]["osg33_free"]["evaluation"]["rollouts"][aggregator.CONDITIONS[0]]["mse_mean"] = 0.
            path = root / "osg33_free/per_sample.json"
            raw = json.loads(path.read_text())
            raw["rollouts"][aggregator.CONDITIONS[0]]["mse"] = [0.] * 128
            write_json(path, raw)
        refresh(root, summary)
    with pytest.raises(ValueError):
        aggregator.aggregate(roots)


@pytest.mark.parametrize("which", ["cache", "protocol", "source"])
def test_optional_external_byte_verification_rejects_changed_artifacts(matrix, which):
    roots, cache, protocol, sources = matrix
    path = {"cache": cache, "protocol": protocol, "source": sources / "runner.py"}[which]
    path.write_bytes(b"altered external input")
    with pytest.raises(ValueError, match="hash mismatch"):
        aggregator.aggregate(roots, cache, protocol, sources)


def test_duplicate_roots_and_existing_output_cannot_be_used(matrix, tmp_path):
    roots, *_ = matrix
    with pytest.raises(ValueError, match="three distinct"):
        aggregator.aggregate([roots[0], roots[0], roots[2]])
    with pytest.raises(ValueError, match="three distinct"):
        aggregator.aggregate(roots[:2])
    with pytest.raises(ValueError, match="refusing to overwrite"):
        aggregator.main([*(str(root) for root in roots), "--output-dir", str(tmp_path)])


def test_zero_numerator_is_reported_without_an_epsilon(tmp_path):
    roots, *_ = build_matrix(tmp_path, small_ratios=(0., 0., 0.))
    result, _, _ = aggregator.aggregate(roots)
    comparison = result["primary_osg_comparisons"]["osg33"]
    assert comparison["geometric_mean_mse_ratio"] == 0.
    assert comparison["material_accuracy_advantage"] is True
    json.dumps(result, allow_nan=False)


def test_unavailable_cpu_peak_memory_stays_unavailable(matrix):
    roots, *_ = matrix
    for root in roots:
        summary = json.loads((root / "summary.json").read_text())
        for label in aggregator.TRAINED:
            summary["cells"][label]["training"]["peak_cuda_memory_bytes"] = None
        refresh(root, summary)
    result, _, _ = aggregator.aggregate(roots)
    assert all(row["peak_cuda_memory_bytes"] is None for row in result["costs"] if row["model"] in aggregator.TRAINED)


def test_actual_runner_evaluation_contract_matches_the_aggregator():
    import torch
    from experiments import run_burgers_stage_a as runner
    from experiments.tests.test_burgers_validated import make_cache

    torch.set_num_threads(1)
    test = make_cache(n_test=128)["splits"]["test"]
    summary, raw = runner.evaluate(runner.Persistence(), test, device="cpu", batch_size=32)
    aggregator._validate_rollouts(summary, raw)
    assert summary["cross_lag"]["0.8"]["rms_defect_mean"] == 0.


def test_incorrect_configuration_digest_is_rejected_even_with_matching_manifest(matrix):
    roots, *_ = matrix
    root = roots[0]
    for name in ("summary.json", "manifest.json"):
        value = json.loads((root / name).read_text())
        value["config_sha256"] = "a" * 64
        write_json(root / name, value)
    marker = json.loads((root / "COMPLETE.json").read_text())
    marker["summary_sha256"] = aggregator.sha256_file(root / "summary.json")
    write_json(root / "COMPLETE.json", marker)
    with pytest.raises(ValueError, match="configuration hash mismatch"):
        aggregator.aggregate(roots)
