from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.evaluate_fisher_kpp_checkpoint import (
    build_parser,
    cache_eval_horizon,
    evaluate_checkpoint,
    load_or_generate_locked_test_data,
    requested_eval_horizons,
    validate_args,
)
from experiments.run_fisher_fair import _source_hashes, build_model, model_config


def _tiny_args(tmp_path, *, prepare_test_data_only=False):
    return build_parser().parse_args(
        [
            "--model",
            "latent",
            "--output-dir",
            str(tmp_path / "out"),
            "--test-cache",
            str(tmp_path / "locked_test.pt"),
            "--device",
            "cpu",
            "--test-seed",
            "19",
            "--deterministic",
            "--N",
            "8",
            "--L",
            "10",
            "--nu",
            "0.1",
            "--reaction-rate",
            "1",
            "--reference-dt",
            "0.01",
            "--fixed-tau",
            "0.02",
            "--eval-horizons",
            "0.04,0.08,0.12",
            "--checkpoint-eval-horizon",
            "0.04",
            "--n-test",
            "2",
            "--beta-v-floor",
            "0.1",
            "--expected-train-taus",
            "0.01,0.02",
            "--expected-n-train",
            "4",
            "--expected-n-val",
            "2",
            "--expected-epochs",
            "2",
            "--expected-batch-size",
            "2",
            "--expected-lr",
            "0.001",
            "--expected-validation-interval",
            "1",
            *(
                ["--prepare-test-data-only"]
                if prepare_test_data_only
                else ["--checkpoint", str(tmp_path / "model_best.pt")]
            ),
        ]
    )


def _write_tiny_checkpoint(path):
    model_args = SimpleNamespace(N=8, beta_v_floor=0.1)
    model = build_model("latent", model_args)
    checkpoint = {
        "epoch": 2,
        "best_val_mse": 0.2,
        "model_state_dict": model.state_dict(),
        "run_metadata": {
            "regime": "variable",
            "data_seed": 7,
            "N": 8,
            "L": 10.0,
            "nu": 0.1,
            "r": 1.0,
            "reference_dt": 0.01,
            "fixed_tau": 0.02,
            "train_taus": [0.01, 0.02],
            "eval_horizon": 0.04,
            "n_train": 4,
            "n_val": 2,
            "training_seed": 7,
            "model": "latent",
            "model_config": model_config("latent", model_args),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-3,
            "validation_interval": 1,
            "architecture_only": True,
            "auxiliary_loss_weights": {
                "alpha_rollout": 0.0,
                "alpha_energy": 0.0,
                "alpha_bound": 0.0,
            },
            "provenance": {
                "source_hashes": _source_hashes(),
                "data_cache_sha256": "not-a-real-cache-in-a-unit-test",
            },
        },
    }
    torch.save(checkpoint, path)


def test_multi_horizon_cli_requires_increasing_aligned_horizons(tmp_path):
    args = _tiny_args(tmp_path)
    assert requested_eval_horizons(args) == pytest.approx((0.04, 0.08, 0.12))
    assert cache_eval_horizon(args) == pytest.approx(0.12)
    validate_args(args)

    duplicate = build_parser().parse_args(
        [
            "--model",
            "latent",
            "--output-dir",
            str(tmp_path / "out"),
            "--test-cache",
            str(tmp_path / "cache.pt"),
            "--eval-horizons",
            "0.04,0.04",
        ]
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_args(duplicate)


def test_locked_cache_is_immutable_and_evaluation_is_checkpoint_only(tmp_path):
    preparation_args = _tiny_args(tmp_path, prepare_test_data_only=True)
    data, status, validation = load_or_generate_locked_test_data(
        preparation_args, allow_generate=True
    )
    assert status == "created"
    assert validation["test_u0_shape"] == [2, 8]

    loaded, loaded_status, loaded_validation = load_or_generate_locked_test_data(
        preparation_args, allow_generate=True
    )
    assert loaded_status == "loaded"
    assert loaded_validation == validation
    assert torch.equal(data["test_u0"], loaded["test_u0"])

    checkpoint_path = tmp_path / "model_best.pt"
    _write_tiny_checkpoint(checkpoint_path)
    cache_path = Path(preparation_args.test_cache)
    checkpoint_before = checkpoint_path.read_bytes()
    cache_before = cache_path.read_bytes()

    result = evaluate_checkpoint(_tiny_args(tmp_path))
    assert result["evaluation_mode"] == "checkpoint_only_locked_test_multi_horizon"
    assert result["protocol"] == {
        "checkpoint_only": True,
        "training": False,
        "optimization": False,
        "checkpoint_selection": False,
        "test_cache_generation": False,
        "test_cache_overwrite": False,
        "multi_horizon_strategy": "frozen_cache_prefixes",
    }
    assert result["checkpoint"]["immutable_during_evaluation"] is True
    assert result["locked_test"]["immutable_during_evaluation"] is True
    assert [item["evaluation"]["rollout_steps"] for item in result["evaluations"]] == [
        2,
        4,
        6,
    ]
    assert all(item["metrics"]["n_valid"] == 2 for item in result["evaluations"])
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert cache_path.read_bytes() == cache_before
    assert (Path(_tiny_args(tmp_path).output_dir) / "summary.json").exists()
    assert (Path(_tiny_args(tmp_path).output_dir) / "metrics.csv").exists()


def test_evaluation_never_creates_a_missing_locked_cache(tmp_path):
    args = _tiny_args(tmp_path)
    _write_tiny_checkpoint(tmp_path / "model_best.pt")
    cache_path = Path(args.test_cache)
    assert not cache_path.exists()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluate_checkpoint(args)
    assert not cache_path.exists()


def test_evaluation_rejects_checkpoint_without_matching_source_hashes(tmp_path):
    preparation_args = _tiny_args(tmp_path, prepare_test_data_only=True)
    load_or_generate_locked_test_data(preparation_args, allow_generate=True)
    checkpoint_path = tmp_path / "model_best.pt"
    _write_tiny_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["run_metadata"]["provenance"]["source_hashes"]["models.py"] = "wrong"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="source hashes differ"):
        evaluate_checkpoint(_tiny_args(tmp_path))
