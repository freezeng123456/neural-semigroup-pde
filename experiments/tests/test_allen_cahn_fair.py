import math
from pathlib import Path

import pytest
import torch

from experiments.models import StencilMLP
from experiments.run_allen_cahn_fair import (
    LOWER_BOUND,
    UPPER_BOUND,
    allen_cahn_free_energy,
    build_model,
    build_parser,
    data_config,
    generate_data,
    load_or_generate_data,
    main,
    reference_steps_for_duration,
    rollout_steps_for_horizon,
    validate_args,
    validate_data_cache,
)


def _tiny_argv(tmp_path, *, regime="variable", prepare_data_only=False):
    return [
            "--regime",
            regime,
            "--models",
            "latent",
            "resnet",
            "fno",
            "--output-dir",
            str(tmp_path / "out"),
            "--data-cache",
            str(tmp_path / "allen_cahn_data.pt"),
            "--device",
            "cpu",
            "--N",
            "8",
            "--L",
            str(2.0 * math.pi),
            "--reference-dt",
            "0.01",
            "--fixed-tau",
            "0.02",
            "--train-taus",
            "0.01,0.02",
            "--eval-taus",
            "0.01,0.02",
            "--eval-horizon",
            "0.04",
            "--n-train",
            "4",
            "--n-val",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--deterministic",
            "--no-resume",
            *(["--prepare-data-only"] if prepare_data_only else []),
        ]


def _tiny_args(tmp_path, *, regime="variable", prepare_data_only=False):
    return build_parser().parse_args(
        _tiny_argv(
            tmp_path,
            regime=regime,
            prepare_data_only=prepare_data_only,
        )
    )


def test_parser_defaults_match_fair_protocol():
    args = build_parser().parse_args(
        [
            "--regime",
            "variable",
            "--output-dir",
            "out",
            "--data-cache",
            "data.pt",
        ]
    )
    assert args.train_taus == (0.025, 0.05, 0.1, 0.2)
    assert args.eval_taus == (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
    assert args.validation_interval == 5
    assert args.beta_v_floor > 0
    assert args.alpha_rollout == 0.0
    assert args.alpha_trajectory == 0.0
    assert args.trajectory_horizon == 0.0
    assert args.alpha_energy == 0.0
    assert args.alpha_bound == 0.0
    assert args.alpha_v == 0.0
    assert args.prepare_data_only is False


def test_beta_v_floor_allows_zero_diagnostic_but_rejects_negative_values():
    zero_floor = build_parser().parse_args(
        [
            "--regime", "fixed", "--output-dir", "out", "--data-cache", "data.pt",
            "--beta-v-floor", "0",
        ]
    )
    validate_args(zero_floor)
    assert zero_floor.beta_v_floor == 0.0

    negative_floor = build_parser().parse_args(
        [
            "--regime", "fixed", "--output-dir", "out", "--data-cache", "data.pt",
            "--beta-v-floor", "-0.01",
        ]
    )
    with pytest.raises(ValueError, match="non-negative"):
        validate_args(negative_floor)


def test_periodic_stencil_is_equivariant_to_cyclic_spatial_shifts():
    torch.manual_seed(7)
    stencil = StencilMLP(radius=3, hidden_dims=[8, 8])
    state = torch.randn(3, 64)
    shifted = torch.roll(state, shifts=1, dims=-1)
    assert torch.allclose(
        stencil(shifted),
        torch.roll(stencil(state), shifts=1, dims=-1),
        rtol=1e-6,
        atol=1e-6,
    )


def test_horizons_and_reference_grid_require_exact_alignment():
    assert [
        rollout_steps_for_horizon(1.2, tau)
        for tau in (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
    ] == [48, 24, 16, 12, 8, 6]
    assert reference_steps_for_duration(0.2, 0.005) == 40
    with pytest.raises(ValueError, match="integer multiple"):
        rollout_steps_for_horizon(1.2, 0.25)
    with pytest.raises(ValueError, match="integer multiple"):
        reference_steps_for_duration(0.023, 0.005)


def test_reference_trajectory_supervision_requires_fixed_aligned_multistep_horizon(
    tmp_path,
):
    valid = build_parser().parse_args(
        [
            "--regime", "fixed", "--output-dir", str(tmp_path / "out"),
            "--data-cache", str(tmp_path / "data.pt"), "--fixed-tau", "0.02",
            "--reference-dt", "0.01", "--eval-horizon", "0.04",
            "--alpha-trajectory", "0.1", "--trajectory-horizon", "0.04",
        ]
    )
    validate_args(valid)

    one_step = build_parser().parse_args(
        [
            "--regime", "fixed", "--output-dir", str(tmp_path / "out"),
            "--data-cache", str(tmp_path / "data.pt"), "--fixed-tau", "0.02",
            "--reference-dt", "0.01", "--eval-horizon", "0.04",
            "--alpha-trajectory", "0.1", "--trajectory-horizon", "0.02",
        ]
    )
    with pytest.raises(ValueError, match="at least two"):
        validate_args(one_step)

    variable = build_parser().parse_args(
        [
            "--regime", "variable", "--output-dir", str(tmp_path / "out"),
            "--data-cache", str(tmp_path / "data.pt"), "--reference-dt", "0.005",
            "--alpha-trajectory", "0.1", "--trajectory-horizon", "0.2",
        ]
    )
    with pytest.raises(ValueError, match="requires --regime fixed"):
        validate_args(variable)


def test_model_budget_and_bounded_latent_output():
    args = build_parser().parse_args(
        [
            "--regime",
            "fixed",
            "--output-dir",
            "out",
            "--data-cache",
            "data.pt",
            "--device",
            "cpu",
            "--N",
            "64",
        ]
    )
    latent = build_model("latent", args)
    decoded_interaction = build_model("latent_decoded_interaction", args)
    resnet = build_model("resnet", args)
    fno = build_model("fno", args)
    counts = {
        "latent": sum(parameter.numel() for parameter in latent.parameters()),
        "latent_decoded_interaction": sum(
            parameter.numel() for parameter in decoded_interaction.parameters()
        ),
        "resnet": sum(parameter.numel() for parameter in resnet.parameters()),
        "fno": sum(parameter.numel() for parameter in fno.parameters()),
    }
    assert counts["latent"] == 9603
    assert counts["latent_decoded_interaction"] == counts["latent"]
    assert abs(counts["resnet"] - counts["latent"]) <= 550
    assert abs(counts["fno"] - counts["latent"]) <= 300

    u = torch.linspace(-0.8, 0.8, 16).reshape(2, 8)
    bounded = build_model("latent", _tiny_args(Path("/tmp")))
    output = bounded(u, 0.02)
    assert float(output.detach().min()) >= LOWER_BOUND
    assert float(output.detach().max()) <= UPPER_BOUND
    decoded_output = build_model(
        "latent_decoded_interaction", _tiny_args(Path("/tmp"))
    )(u, 0.02)
    assert float(decoded_output.detach().min()) >= LOWER_BOUND
    assert float(decoded_output.detach().max()) <= UPPER_BOUND


def test_physical_free_energy_is_distinct_and_vectorized():
    zeros = torch.zeros(3, 16)
    ones = torch.ones(3, 16)
    zero_energy = allen_cahn_free_energy(zeros, L=2.0 * math.pi, epsilon=0.1)
    one_energy = allen_cahn_free_energy(ones, L=2.0 * math.pi, epsilon=0.1)
    assert zero_energy.shape == (3,)
    assert torch.allclose(zero_energy, torch.full((3,), math.pi / 2.0))
    assert torch.allclose(one_energy, torch.zeros(3))

    n_sites = 64
    domain_length = 2.0 * math.pi
    epsilon = 0.1
    x = torch.arange(n_sites, dtype=torch.float64) * domain_length / n_sites
    sine = torch.sin(x).unsqueeze(0)
    sine_energy = allen_cahn_free_energy(
        sine, L=domain_length, epsilon=epsilon
    )
    expected = epsilon**2 * math.pi / 2.0 + 3.0 * math.pi / 16.0
    assert float(sine_energy) == pytest.approx(expected, rel=1e-10, abs=1e-10)


def test_data_generation_is_frozen_balanced_and_strictly_aligned(tmp_path):
    args = _tiny_args(tmp_path)
    data = generate_data(args)
    validation = validate_data_cache(data, args)
    assert data["data_generation_config"] == data_config(args)
    assert validation["train_tau_counts"] == {"0.01": 2, "0.02": 2}
    assert validation["validation_trajectory_length"] == 5
    assert len(data["val_trajs"]) == 2
    assert len(data["val_trajs"][0][0]) == 5
    assert torch.all(data["val_u0"] >= LOWER_BOUND)
    assert torch.all(data["val_u0"] <= UPPER_BOUND)


def test_fixed_data_contains_a_frozen_reference_rollout_target(tmp_path):
    argv = _tiny_argv(tmp_path, regime="fixed") + [
        "--alpha-trajectory", "0.1", "--trajectory-horizon", "0.04",
    ]
    args = build_parser().parse_args(argv)
    validate_args(args)
    data = generate_data(args)
    validation = validate_data_cache(data, args)
    assert data_config(args)["trajectory_horizon"] == 0.04
    assert data["train_rollout_ut"].shape == data["train_u0"].shape
    assert validation["train_rollout_ut_shape"] == [4, 8]
    assert validation["train_rollout_ut_range"] is not None


def test_prepare_data_only_reuses_validated_cache_without_checkpoints(tmp_path):
    args = _tiny_args(tmp_path, prepare_data_only=True)
    data, status, validation = load_or_generate_data(args)
    assert status == "created"
    assert validation["validation_trajectory_length"] == 5
    data_again, status_again, validation_again = load_or_generate_data(args)
    assert status_again == "loaded"
    assert validation_again == validation
    assert torch.equal(data["train_u0"], data_again["train_u0"])
    main(_tiny_argv(tmp_path, prepare_data_only=True))
    assert (tmp_path / "out" / "data_provenance.json").exists()
    assert not (tmp_path / "out" / "provenance.json").exists()
    assert not (tmp_path / "out" / "latent" / "checkpoints").exists()
