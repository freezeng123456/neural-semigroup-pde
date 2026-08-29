"""Contract tests for the exploratory Fisher mechanism evaluator."""

import json
from pathlib import Path

import pytest
import torch

from experiments import evaluate_fisher_mechanism_diagnostics as stable_entry
from experiments import evaluate_fisher_mechanism as evaluator
from experiments.models import QueryTimeConditionedLatentFlow


def _tiny_args(tmp_path: Path):
    args = stable_entry.build_parser().parse_args(
        [
            "--checkpoint",
            str(tmp_path / "latent_query_time_best.pt"),
            "--test-cache",
            str(tmp_path / "frozen_test.pt"),
            "--output-dir",
            str(tmp_path / "exploratory-diagnostics"),
            "--source-archive",
            str(tmp_path / "source.tar.gz"),
            "--device",
            "cpu",
            "--deterministic",
            "--N",
            "8",
            "--reference-dt",
            "0.01",
            "--query-taus",
            "0.02",
            "--query-horizons",
            "0.04",
            "--total-horizons",
            "0.08",
            "--segment-counts",
            "2,4,8,16",
            "--n-unequal-partitions",
            "1",
            "--ode-steps",
            "1",
            "--max-samples",
            "2",
        ]
    )
    (tmp_path / "source.tar.gz").write_bytes(b"test archive")
    args.expected_source_archive_sha256 = evaluator._sha256_file(
        tmp_path / "source.tar.gz"
    )
    return args


def _write_fake_checkpoint(path: Path):
    kwargs = {
        "N": 8,
        "hidden_V": [4],
        "hidden_K": [4],
        "stencil_radius": 1,
        "interaction_radius": 1,
        "beta_V": 0.0,
        "beta_V_floor": 0.1,
    }
    model = QueryTimeConditionedLatentFlow(**kwargs)
    source_hashes = evaluator._runtime_source_hashes()
    checkpoint = {
        "epoch": 3,
        "best_val_mse": 0.25,
        "model_state_dict": model.state_dict(),
        "run_metadata": {
            "model": "latent_query_time",
            "provenance": {
                "git_commit": evaluator.FORMAL_SOURCE_COMMIT,
                "source_hashes": source_hashes,
            },
            "model_config": {
                "class": "QueryTimeConditionedLatentFlow",
                "kwargs": kwargs,
            },
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
    }
    torch.save(checkpoint, path)


def _write_fake_cache(path: Path):
    test_u0 = torch.full((2, 8), 0.4)
    times = torch.arange(9, dtype=torch.float32) * 0.01
    trajectories = [
        (times.clone(), test_u0[index].repeat(9, 1)) for index in range(2)
    ]
    cache = {
        "schema_version": 1,
        "locked_test_config": {
            "schema_version": 1,
            "pde": "fisher-kpp",
            "split": "locked_independent_test",
            "test_seed": 314163,
            "L": 10.0,
            "nu": 0.1,
            "reaction_rate": 1.0,
            "fixed_tau": 0.1,
            "reference_dt": 0.01,
            "eval_horizon": 0.08,
        },
        "test_u0": test_u0,
        "test_trajs": trajectories,
    }
    torch.save(cache, path)


def test_repeatable_spec_and_short_aliases_are_explicitly_supported(tmp_path):
    parser = stable_entry.build_parser()
    args = parser.parse_args(
        [
            "--checkpoint-spec",
            "42:latent_query_time:/tmp/a.pt",
            "--checkpoint-spec",
            "137:latent_query_time:/tmp/b.pt",
            "--test-cache",
            "/tmp/frozen.pt",
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
            "--taus",
            "0.02,0.04",
            "--horizons",
            "0.08",
            "--segment-counts",
            "2,4,8,16",
        ]
    )
    stable_entry.normalize_diagnostic_args(args)
    assert stable_entry.checkpoint_specs_from_args(args) == [
        {"seed": 42, "model": "latent_query_time", "path": "/tmp/a.pt"},
        {"seed": 137, "model": "latent_query_time", "path": "/tmp/b.pt"},
    ]
    assert args.query_taus == (0.02, 0.04)
    assert args.query_horizons == (0.08,)
    assert args.total_horizons == (0.08,)


def test_partitions_are_seeded_positive_and_reproducible():
    first = evaluator.deterministic_unequal_partition(1.2, 16, 2718, trial=2)
    second = evaluator.deterministic_unequal_partition(1.2, 16, 2718, trial=2)
    assert first == second
    assert len(first) == 16
    assert all(value > 0 for value in first)
    assert sum(first) == pytest.approx(1.2)
    assert tuple(reversed(first)) != first
    equal = evaluator.equal_partition(1.2, 4)
    assert equal == pytest.approx((0.3, 0.3, 0.3, 0.3))


def test_shuffled_query_schedule_is_deterministic_and_hashed():
    first, first_hash = evaluator.deterministic_shuffled_query_times(
        10, (0.025, 0.1, 0.2), 1729, stream=4
    )
    second, second_hash = evaluator.deterministic_shuffled_query_times(
        10, (0.025, 0.1, 0.2), 1729, stream=4
    )
    assert first.tolist() == second.tolist()
    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_checkpoint_only_diagnostics_keep_inputs_immutable_and_write_contract(tmp_path):
    args = _tiny_args(tmp_path)
    _write_fake_checkpoint(Path(args.checkpoint))
    _write_fake_cache(Path(args.test_cache))
    args.expected_cache_sha256 = evaluator._sha256_file(args.test_cache)
    checkpoint_before = Path(args.checkpoint).read_bytes()
    cache_before = Path(args.test_cache).read_bytes()

    result = stable_entry.evaluate_diagnostics(args)

    assert result["exploratory"] is True
    assert result["do_not_use_for_formal"] is True
    assert result["evaluation_mode"] == evaluator.EVALUATION_MODE
    assert result["protocol"]["checkpoint_only"] is True
    assert result["protocol"]["training"] is False
    assert result["protocol"]["test_cache_generation"] is False
    assert result["query_time_ablation"]["controls"] == [
        "normal",
        "fixed_0.1",
        "shuffled",
    ]
    assert result["composition_stress"]["segment_counts"] == [2, 4, 8, 16]
    assert set(result["composition_stress"]["work_modes"]) == {
        "fixed_per_call",
        "equal_total_rk4_work",
    }
    output_dir = Path(args.output_dir)
    for name in (
        "exploratory_manifest.json",
        "results.json",
        "metrics.csv",
        "receipt.json",
    ):
        assert (output_dir / name).is_file()
    manifest = json.loads((output_dir / "exploratory_manifest.json").read_text())
    receipt = json.loads((output_dir / "receipt.json").read_text())
    assert manifest["exploratory"] is True
    assert receipt["status"] == "passed"
    assert receipt["input_hashes"]["checkpoint"]["unchanged"] is True
    assert receipt["input_hashes"]["test_cache"]["unchanged"] is True
    assert Path(args.checkpoint).read_bytes() == checkpoint_before
    assert Path(args.test_cache).read_bytes() == cache_before


def test_output_guard_rejects_formal_or_unmarked_roots(tmp_path):
    with pytest.raises(ValueError, match="new exploratory root"):
        evaluator._assert_new_output_root(tmp_path / "results")
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="already exists|new exploratory root"):
        evaluator._assert_new_output_root(existing)


def test_missing_cache_is_never_generated(tmp_path):
    args = _tiny_args(tmp_path)
    _write_fake_checkpoint(Path(args.checkpoint))
    with pytest.raises(FileNotFoundError, match="refuses to generate"):
        stable_entry.evaluate_diagnostics(args)
    assert not Path(args.test_cache).exists()


def test_multi_checkpoint_specs_write_reserved_aggregate_root(tmp_path):
    args = _tiny_args(tmp_path)
    first_checkpoint = Path(args.checkpoint)
    second_checkpoint = tmp_path / "latent_query_time_second_best.pt"
    _write_fake_checkpoint(first_checkpoint)
    _write_fake_checkpoint(second_checkpoint)
    _write_fake_cache(Path(args.test_cache))
    args.expected_cache_sha256 = evaluator._sha256_file(args.test_cache)
    args.checkpoint = None
    args.checkpoint_specs = [
        {"seed": 42, "model": evaluator.MODEL_NAME, "path": str(first_checkpoint)},
        {"seed": 137, "model": evaluator.MODEL_NAME, "path": str(second_checkpoint)},
    ]

    stable_entry.main(
        [
            "--checkpoint-spec",
            f"42:{evaluator.MODEL_NAME}:{first_checkpoint}",
            "--checkpoint-spec",
            f"137:{evaluator.MODEL_NAME}:{second_checkpoint}",
            "--test-cache",
            str(args.test_cache),
            "--output-dir",
            str(args.output_dir),
            "--source-archive",
            str(args.source_archive),
            "--expected-source-archive-sha256",
            str(args.expected_source_archive_sha256),
            "--expected-cache-sha256",
            str(args.expected_cache_sha256),
            "--device",
            "cpu",
            "--deterministic",
            "--N",
            "8",
            "--reference-dt",
            "0.01",
            "--query-taus",
            "0.02",
            "--query-horizons",
            "0.04",
            "--total-horizons",
            "0.08",
            "--segment-counts",
            "2,4,8,16",
            "--n-unequal-partitions",
            "1",
            "--ode-steps",
            "1",
            "--max-samples",
            "2",
        ]
    )

    output_dir = Path(args.output_dir)
    assert (output_dir / "seed_42_latent_query_time" / "receipt.json").is_file()
    assert (output_dir / "seed_137_latent_query_time" / "receipt.json").is_file()
    aggregate_receipt = json.loads((output_dir / "receipt.json").read_text())
    assert aggregate_receipt["status"] == "passed"
    assert (output_dir / "metrics_index.json").is_file()
