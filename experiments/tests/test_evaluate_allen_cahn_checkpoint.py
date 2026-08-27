import csv
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.evaluate_allen_cahn_checkpoint import (
    build_parser,
    cache_eval_horizon,
    evaluation_tau,
    evaluate_checkpoint,
    load_or_generate_locked_test_data,
    requested_eval_horizons,
    validate_args,
)
from experiments.run_allen_cahn_fair import (
    _source_hashes,
    build_model,
    model_config,
)


def _tiny_args(tmp_path, *, prepare_test_data_only=False):
    return build_parser().parse_args(
        [
            "--model",
            "latent_physics_anchored_periodic",
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
            str(2.0 * math.pi),
            "--reference-dt",
            "0.01",
            "--fixed-tau",
            "0.02",
            "--eval-horizon",
            "0.04",
            "--n-test",
            "2",
            "--beta-v-floor",
            "0",
            *(
                ["--prepare-test-data-only"]
                if prepare_test_data_only
                else ["--checkpoint", str(tmp_path / "model_best.pt")]
            ),
        ]
    )


def _tiny_multi_args(tmp_path, *, prepare_test_data_only=False):
    return build_parser().parse_args(
        [
            "--model",
            "latent_physics_anchored_periodic",
            "--output-dir",
            str(tmp_path / "multi-out"),
            "--test-cache",
            str(tmp_path / "multi-locked_test.pt"),
            "--device",
            "cpu",
            "--test-seed",
            "19",
            "--deterministic",
            "--N",
            "8",
            "--L",
            str(2.0 * math.pi),
            "--reference-dt",
            "0.01",
            "--fixed-tau",
            "0.02",
            "--eval-horizons",
            "0.04,0.08,0.12",
            "--n-test",
            "2",
            "--beta-v-floor",
            "0",
            *(
                ["--prepare-test-data-only"]
                if prepare_test_data_only
                else ["--checkpoint", str(tmp_path / "multi-model_best.pt")]
            ),
        ]
    )


def test_multi_horizon_cli_requires_increasing_aligned_horizons(tmp_path):
    args = _tiny_multi_args(tmp_path)
    assert requested_eval_horizons(args) == pytest.approx((0.04, 0.08, 0.12))
    assert cache_eval_horizon(args) == pytest.approx(0.12)
    validate_args(args)

    duplicate = build_parser().parse_args(
        [
            "--model",
            "latent_physics_anchored_periodic",
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


def test_single_horizon_cli_default_remains_backward_compatible(tmp_path):
    args = build_parser().parse_args(
        [
            "--model",
            "latent_physics_anchored_periodic",
            "--output-dir",
            str(tmp_path / "out"),
            "--test-cache",
            str(tmp_path / "cache.pt"),
        ]
    )
    assert args.eval_horizon == pytest.approx(1.2)
    assert requested_eval_horizons(args) == pytest.approx((1.2,))


def test_locked_cache_is_immutable_and_checkpoint_evaluation_is_training_free(tmp_path):
    preparation_args = _tiny_args(tmp_path, prepare_test_data_only=True)
    validate_args(preparation_args)
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

    model_args = SimpleNamespace(N=8, beta_v_floor=0.0)
    model = build_model("latent_physics_anchored_periodic", model_args)
    checkpoint = {
        "epoch": 1,
        "best_val_mse": 0.5,
        "model_state_dict": model.state_dict(),
        "run_metadata": {
            "model": "latent_physics_anchored_periodic",
            "model_config": model_config(
                "latent_physics_anchored_periodic", model_args
            ),
            "N": 8,
            "L": 2.0 * math.pi,
            "epsilon": 0.1,
            "reference_dt": 0.01,
            "fixed_tau": 0.02,
            "eval_horizon": 0.04,
            "data_seed": 7,
            "provenance": {"source_hashes": _source_hashes()},
        },
    }
    torch.save(checkpoint, tmp_path / "model_best.pt")

    evaluation_args = _tiny_args(tmp_path)
    result = evaluate_checkpoint(evaluation_args)
    assert result["evaluation_mode"] == "checkpoint_only_locked_test"
    assert result["checkpoint"]["epoch"] == 1
    assert result["locked_test"]["independent_from_checkpoint_training_data"] is True
    assert result["metrics"]["n_valid"] == 2
    assert (Path(evaluation_args.output_dir) / "summary.json").exists()
    assert (Path(evaluation_args.output_dir) / "metrics.csv").exists()


def test_multi_horizon_evaluation_reuses_long_cache_prefixes_and_preserves_inputs(
    tmp_path,
):
    preparation_args = _tiny_multi_args(tmp_path, prepare_test_data_only=True)
    data, status, validation = load_or_generate_locked_test_data(
        preparation_args, allow_generate=True
    )
    assert status == "created"
    assert validation["cache_horizon"] == pytest.approx(0.12)
    assert validation["requested_horizons"] == pytest.approx((0.04, 0.08, 0.12))

    model_args = SimpleNamespace(N=8, beta_v_floor=0.0)
    model = build_model("latent_physics_anchored_periodic", model_args)
    checkpoint = {
        "epoch": 3,
        "best_val_mse": 0.25,
        "model_state_dict": model.state_dict(),
        "run_metadata": {
            "model": "latent_physics_anchored_periodic",
            "model_config": model_config(
                "latent_physics_anchored_periodic", model_args
            ),
            "N": 8,
            "L": 2.0 * math.pi,
            "epsilon": 0.1,
            "reference_dt": 0.01,
            "fixed_tau": 0.02,
            # The checkpoint was selected under the first/shortest protocol
            # horizon; longer horizons are evaluation-only extensions.
            "eval_horizon": 0.04,
            "data_seed": 7,
            "provenance": {"source_hashes": _source_hashes()},
        },
    }
    checkpoint_path = tmp_path / "multi-model_best.pt"
    torch.save(checkpoint, checkpoint_path)
    cache_path = Path(preparation_args.test_cache)
    checkpoint_before = checkpoint_path.read_bytes()
    cache_before = cache_path.read_bytes()

    evaluation_args = _tiny_multi_args(tmp_path)
    result = evaluate_checkpoint(evaluation_args)

    assert result["schema_version"] == 2
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
    assert result["evaluation_horizons"] == pytest.approx((0.04, 0.08, 0.12))
    assert [item["evaluation"]["rollout_steps"] for item in result["evaluations"]] == [
        2,
        4,
        6,
    ]
    assert all(item["metrics"]["n_valid"] == 2 for item in result["evaluations"])
    assert set(result["metrics_by_horizon"]) == {"0.04", "0.08", "0.12"}
    assert result["checkpoint"]["sha256_before"] == result["checkpoint"]["sha256_after"]
    assert result["locked_test"]["sha256_before"] == result["locked_test"]["sha256_after"]
    assert result["checkpoint"]["immutable_during_evaluation"] is True
    assert result["locked_test"]["immutable_during_evaluation"] is True
    assert checkpoint_path.read_bytes() == checkpoint_before
    assert cache_path.read_bytes() == cache_before

    with (Path(evaluation_args.output_dir) / "metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) == {"metric", "value"}
    assert {
        row["metric"].split(":", maxsplit=1)[0].split("=", maxsplit=1)[1]
        for row in rows
    } == {
        "0.04",
        "0.08",
        "0.12",
    }


def test_multi_horizon_evaluation_never_creates_a_missing_cache(tmp_path):
    args = _tiny_multi_args(tmp_path)
    cache_path = Path(args.test_cache)
    assert not cache_path.exists()
    with pytest.raises(FileNotFoundError, match="does not exist"):
        evaluate_checkpoint(args)
    assert not cache_path.exists()


def test_variable_time_checkpoint_can_be_evaluated_at_an_unseen_aligned_tau(
    tmp_path,
):
    preparation_args = _tiny_multi_args(tmp_path, prepare_test_data_only=True)
    load_or_generate_locked_test_data(preparation_args, allow_generate=True)

    model_args = SimpleNamespace(N=8, beta_v_floor=0.0)
    model = build_model("latent_physics_anchored_periodic", model_args)
    checkpoint = {
        "epoch": 4,
        "best_val_mse": 0.2,
        "model_state_dict": model.state_dict(),
        "run_metadata": {
            "model": "latent_physics_anchored_periodic",
            "model_config": model_config(
                "latent_physics_anchored_periodic", model_args
            ),
            "N": 8,
            "L": 2.0 * math.pi,
            "epsilon": 0.1,
            "reference_dt": 0.01,
            # This remains the checkpoint's training-protocol lag.
            "fixed_tau": 0.02,
            "eval_horizon": 0.04,
            "data_seed": 7,
            "provenance": {"source_hashes": _source_hashes()},
        },
    }
    checkpoint_path = tmp_path / "multi-model_best.pt"
    torch.save(checkpoint, checkpoint_path)

    evaluation_args = build_parser().parse_args(
        [
            "--model", "latent_physics_anchored_periodic",
            "--checkpoint", str(checkpoint_path),
            "--output-dir", str(tmp_path / "unseen-tau-out"),
            "--test-cache", str(preparation_args.test_cache),
            "--device", "cpu", "--test-seed", "19", "--deterministic",
            "--N", "8", "--L", str(2.0 * math.pi),
            "--reference-dt", "0.01", "--fixed-tau", "0.02",
            "--eval-tau", "0.04", "--eval-horizons", "0.04,0.08,0.12",
            "--n-test", "2", "--beta-v-floor", "0",
        ]
    )
    validate_args(evaluation_args)
    assert evaluation_tau(evaluation_args) == pytest.approx(0.04)
    result = evaluate_checkpoint(evaluation_args)
    assert result["checkpoint"]["protocol_eval_horizon"] == pytest.approx(0.04)
    assert result["locked_test"]["config"]["fixed_tau"] == pytest.approx(0.02)
    assert result["evaluation"]["tau"] == pytest.approx(0.04)
    assert [item["evaluation"]["rollout_steps"] for item in result["evaluations"]] == [
        1,
        2,
        3,
    ]
