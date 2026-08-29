"""Contract tests for the explicitly exploratory Fisher first-wave runner."""

import inspect
import json
import runpy
from pathlib import Path

import pytest
import torch

from experiments import run_fisher_exploratory as exploratory
from experiments import run_fisher_fair


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _args(tmp_path: Path, experiment: str = "capacity", **overrides):
    values = {
        "--experiment": experiment,
        "--output-dir": str(tmp_path / "exploratory-output"),
        "--data-cache": str(tmp_path / "exploratory-output" / "data.pt"),
        "--device": "cpu",
        "--N": "8",
        "--reference-dt": "0.01",
        "--fixed-tau": "0.02",
        "--eval-horizon": "0.04",
        "--n-train": "4",
        "--n-val": "2",
        "--epochs": "1",
        "--batch-size": "2",
        "--ode-steps": "2",
        "--deterministic": None,
        "--no-resume": None,
    }
    for key, value in overrides.items():
        flag = key if key.startswith("--") else f"--{key.replace('_', '-')}"
        values[flag] = str(value)
    argv = []
    for flag, value in values.items():
        argv.append(flag)
        if value is not None:
            argv.append(value)
    return exploratory.build_parser().parse_args(argv)


def test_formal_fisher_defaults_and_model_contract_remain_unchanged():
    args = run_fisher_fair.build_parser().parse_args(
        [
            "--regime",
            "variable",
            "--models",
            "latent",
            "latent_query_time",
            "--output-dir",
            "formal-output",
            "--data-cache",
            "formal-data.pt",
            "--N",
            "64",
        ]
    )
    assert args.train_taus == (0.025, 0.05, 0.1, 0.2)
    assert args.eval_taus == (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
    assert args.validation_interval == 5

    autonomous = run_fisher_fair.build_model("latent", args)
    query_time = run_fisher_fair.build_model("latent_query_time", args)
    assert sum(parameter.numel() for parameter in autonomous.parameters()) == 9603
    assert sum(parameter.numel() for parameter in query_time.parameters()) == 9667
    assert not hasattr(autonomous, "interaction_embedding_dim")


def test_source_hashes_support_relative_runpy_script_path(monkeypatch):
    """The SCNet compatibility shim invokes this runner through relative runpy."""
    monkeypatch.chdir(REPOSITORY_ROOT)
    namespace = runpy.run_path(
        "experiments/run_fisher_exploratory.py",
        run_name="fisher_exploratory_relative_runpy_test",
    )

    source_hashes = namespace["_source_hashes"]()

    assert "experiments/run_fisher_exploratory.py" in source_hashes
    assert len(source_hashes["experiments/run_fisher_exploratory.py"]) == 64


def test_json_outputs_are_strict_and_validation_gaps_become_null(tmp_path):
    history = {
        "epoch": [1, 2],
        "validation_performed": [False, True],
        "val_mse": [float("nan"), 0.25],
        "val_bound_viol": [float("nan"), 0.0],
        "val_energy_mono": [float("nan"), 1.0],
        "L_step": [0.5, 0.4],
    }
    normalized = exploratory._history_for_json(history)
    assert normalized["val_mse"] == [None, 0.25]
    assert normalized["val_bound_viol"] == [None, 0.0]
    assert normalized["val_energy_mono"] == [None, 1.0]

    output = tmp_path / "strict.json"
    exploratory._write_json(output, {"history": normalized})
    assert json.loads(output.read_text())["history"] == normalized
    assert "NaN" not in output.read_text()

    rejected = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="Out of range float values"):
        exploratory._write_json(rejected, {"metric": float("nan")})
    assert not rejected.exists()


def test_capacity_matched_a_wide_records_an_exact_9667_parameter_match():
    args = _args(Path("/tmp"), N=64)
    autonomous = exploratory.build_capacity_matched_autonomous_model(args)
    query_time = exploratory.build_query_time_model(args)
    metadata = exploratory.parameter_match_metadata(autonomous, query_time)

    assert metadata["target_parameter_count"] == 9667
    assert metadata["actual_parameter_count"] == 9667
    assert metadata["difference_actual_minus_target"] == 0
    assert metadata["exact_match"] is True
    assert metadata["autonomous_interaction_embedding_dim"] == 9

    # The widened model has no absolute-time argument and remains bounded.
    assert list(inspect.signature(autonomous.forward).parameters) == ["u", "tau"]
    output = autonomous(torch.rand(2, 64) * 0.8 + 0.1, 0.02)
    assert output.shape == (2, 64)
    assert torch.all(output > 0.0)
    assert torch.all(output < 1.0)


def test_b_soft_defaults_and_lambda_semantics_use_shared_rollout_contract(tmp_path, monkeypatch):
    args = _args(tmp_path, experiment="b-soft", N=8)
    exploratory.validate_args(args)
    assert args.lambdas == exploratory.SUPPORTED_B_SOFT_LAMBDAS

    observed = []

    def fake_train(model, **kwargs):
        observed.append(float(kwargs["alpha_rollout"]))
        metadata = {
            "loss_weights": {"alpha_rollout": float(kwargs["alpha_rollout"])},
            "composition_loss_source": "training.rollout_loss",
        }
        return (
            {
                "schema_version": 1,
                "exploratory": True,
                "do_not_use_for_formal": True,
                "model": kwargs["model_name"],
                "history": {"L_rollout": [0.0]},
            },
            metadata,
        )

    monkeypatch.setattr(exploratory, "_train_autonomous_model", fake_train)
    result = exploratory.run_b_soft_experiment(
        args,
        data={},
        data_cache=Path(args.data_cache),
    )

    assert observed == list(exploratory.SUPPORTED_B_SOFT_LAMBDAS)
    assert result["requested_lambdas"] == list(exploratory.SUPPORTED_B_SOFT_LAMBDAS)
    for lambda_key, item in result["models"].items():
        assert item["lambda"] == pytest.approx(float(lambda_key))
        assert item["lambda_semantics"]["parameter"] == "alpha_rollout"
        assert item["lambda_semantics"]["loss"] == "training.rollout_loss"
        assert item["lambda_semantics"]["total_objective"] == (
            "L_step + lambda * L_rollout"
        )


def test_non_autonomous_rate_and_absolute_time_input_are_explicit(tmp_path):
    args = _args(tmp_path, experiment="non-autonomous", omega=1.5)
    exploratory.validate_args(args)
    assert exploratory.non_autonomous_rate(0.0, omega=1.5) == pytest.approx(1.0)
    assert exploratory.non_autonomous_rate(
        torch.tensor([0.0, 1.0]), omega=1.5
    ).tolist() == pytest.approx([1.0, 1.0 + 0.5 * torch.sin(torch.tensor(1.5)).item()])

    autonomous = exploratory.build_capacity_matched_autonomous_model(args)
    non_autonomous = exploratory.build_non_autonomous_model(args)
    assert list(inspect.signature(autonomous.forward).parameters) == ["u", "tau"]
    assert "absolute_time" in inspect.signature(non_autonomous.forward).parameters
    assert non_autonomous.latent_dynamics_requires_absolute_time is True

    # Exercise the time feature directly with a one-hidden-unit stencil.  The
    # default copied initialization is neutral, so this test makes the
    # feature's dependency unambiguous without relying on a trained checkpoint.
    stencil = exploratory.AbsoluteTimeConditionedStencilMLP(radius=1, hidden_dims=[1])
    linear_layers = [layer for layer in stencil.net if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        for parameter in stencil.parameters():
            parameter.zero_()
        linear_layers[0].weight[0, -1] = 1.0
        linear_layers[-1].weight[0, 0] = 1.0
    z = torch.zeros(2, 8)
    assert not torch.allclose(stencil(z, 0.0), stencil(z, 0.2))


def test_non_autonomous_evaluation_disables_autograd_and_chunks_validation(tmp_path):
    args = _args(tmp_path, experiment="non-autonomous", omega=1.5, eval_batch_size=1)
    observed_grad_states = []
    observed_batch_sizes = []

    class IdentityModel(torch.nn.Module):
        def forward(self, u, tau, *, absolute_time=None):
            observed_grad_states.append(torch.is_grad_enabled())
            observed_batch_sizes.append(int(u.shape[0]))
            return u

    times = torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04])
    states = torch.zeros(5, args.N)
    data = {
        "val_u0": torch.zeros(args.n_val, args.N),
        "val_t0": torch.zeros(args.n_val),
        "val_trajs": [(times, states) for _ in range(args.n_val)],
    }
    result = exploratory._non_autonomous_evaluate(
        IdentityModel(), data, args, absolute_time_input=True
    )

    assert result["n_valid"] == args.n_val * 2
    assert observed_grad_states
    assert not any(observed_grad_states)
    assert observed_batch_sizes == [1, 1, 1, 1]
    assert result["eval_batch_size"] == 1


def test_non_autonomous_cache_is_reproducible_and_carries_shared_time_identity(tmp_path):
    args = _args(tmp_path, experiment="non-autonomous", omega=1.5, start_time_max=0.2)
    exploratory.validate_args(args)
    first = exploratory._generate_non_autonomous_data(args)
    second = exploratory._generate_non_autonomous_data(args)

    for key in ("train_u0", "train_t0", "train_ut", "val_u0", "val_t0"):
        assert torch.equal(first[key], second[key])
    for index, ((first_times, first_states), (second_times, second_states)) in enumerate(
        zip(first["val_trajs"], second["val_trajs"])
    ):
        assert torch.equal(first_times, second_times)
        assert torch.equal(first_states, second_states)
        assert float(first_times[0]) == pytest.approx(float(first["val_t0"][index]))

    validation = exploratory.validate_data_cache(
        first,
        args,
        exploratory.NON_AUTONOMOUS_DATA_KIND,
    )
    assert validation["train_start_time_range"] is not None
    assert first["data_generation_config"]["absolute_time_input"] is True
    assert first["data_generation_config"]["reaction_schedule"]["formula"] == (
        "r(t)=1+0.5*sin(omega*t)"
    )


def test_exploratory_cache_rejects_unmarked_formal_payload(tmp_path):
    args = _args(tmp_path, experiment="capacity")
    formal_payload = {
        "data_generation_config": exploratory._common_data_config(
            args, exploratory.AUTONOMOUS_DATA_KIND
        ),
        "train_u0": torch.zeros(args.n_train, args.N),
        "train_ut": torch.zeros(args.n_train, args.N),
        "val_u0": torch.zeros(args.n_val, args.N),
        "val_t0": torch.zeros(args.n_val),
        "train_t0": None,
        "val_trajs": [],
    }
    with pytest.raises(ValueError, match="not an exploratory schema"):
        exploratory.validate_data_cache(
            formal_payload,
            args,
            exploratory.AUTONOMOUS_DATA_KIND,
        )
