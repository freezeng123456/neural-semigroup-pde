import math
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.evaluate_allen_cahn_checkpoint import (
    build_parser,
    evaluate_checkpoint,
    load_or_generate_locked_test_data,
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
