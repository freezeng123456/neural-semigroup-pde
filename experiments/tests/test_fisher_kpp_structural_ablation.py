import math

import pytest
import torch

from experiments.run_fisher_kpp_structural_ablation import (
    VARIANTS,
    NoInteractionLatent,
    ConstantMobilityLatent,
    benchmark_batched_inference,
    build_model,
    build_parser,
    main,
    variant_configuration,
)


def _args(variant, **overrides):
    values = {
        "variant": variant,
        "output_dir": "out",
        "data_cache": "data.pt",
        "device": "cpu",
        "seed": 42,
        "data_seed": 42,
        "deterministic": True,
        "no_resume": True,
        "N": 8,
        "L": 10.0,
        "nu": 0.1,
        "reaction_rate": 1.0,
        "reference_dt": 0.005,
        "fixed_tau": 0.1,
        "eval_horizon": 0.2,
        "n_train": 4,
        "n_val": 2,
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "validation_interval": 5,
        "beta_v_floor": 0.1,
        "aux_alpha_rollout": 0.1,
        "aux_alpha_energy": 0.01,
        "aux_alpha_v": 0.0,
    }
    values.update(overrides)
    return type("Args", (), values)()


def test_parser_exposes_phase3_controls_and_defaults_to_five_epoch_validation():
    args = build_parser().parse_args(
        [
            "--variant",
            "full",
            "--output-dir",
            "out",
            "--data-cache",
            "data.pt",
            "--seed",
            "123",
            "--data-seed",
            "17",
            "--deterministic",
            "--no-resume",
        ]
    )
    assert args.seed == 123
    assert args.data_seed == 17
    assert args.deterministic is True
    assert args.no_resume is True
    assert args.prepare_data_only is False
    assert args.validation_interval == 5
    assert args.fixed_tau == pytest.approx(0.1)


def test_prepare_data_only_writes_validated_provenance_without_checkpoints(tmp_path):
    output_dir = tmp_path / "prepared"
    cache_path = tmp_path / "shared" / "fisher_data.pt"
    main(
        [
            "--output-dir",
            str(output_dir),
            "--data-cache",
            str(cache_path),
            "--device",
            "cpu",
            "--data-seed",
            "19",
            "--deterministic",
            "--prepare-data-only",
            "--N",
            "8",
            "--n-train",
            "4",
            "--n-val",
            "2",
            "--eval-horizon",
            "0.2",
        ]
    )

    provenance_path = output_dir / "data_provenance.json"
    assert cache_path.exists()
    assert provenance_path.exists()
    provenance = __import__("json").loads(provenance_path.read_text())
    assert provenance["mode"] == "prepare-data-only"
    assert provenance["status"] == "validated"
    assert provenance["data_seed"] == 19
    assert not list(output_dir.rglob("*checkpoint*"))


def test_training_mode_requires_prepared_shared_cache(tmp_path):
    with pytest.raises(FileNotFoundError, match="--prepare-data-only"):
        main(
            [
                "--variant",
                "full",
                "--output-dir",
                str(tmp_path / "training"),
                "--data-cache",
                str(tmp_path / "missing.pt"),
                "--device",
                "cpu",
                "--N",
                "8",
                "--n-train",
                "4",
                "--n-val",
                "2",
                "--eval-horizon",
                "0.2",
            ]
        )
    assert not list((tmp_path / "training").rglob("*checkpoint*"))


def test_all_variants_have_explicit_structural_and_loss_configuration():
    for variant in VARIANTS:
        config = variant_configuration(_args(variant))
        assert config["variant"] == variant
        assert set(config["loss_weights"]) == {
            "alpha_rollout",
            "alpha_energy",
            "alpha_bound",
            "alpha_V",
            "weight_decay",
        }
        if variant == "full":
            assert config["architecture_only"] is True
            assert config["beta_V_floor"] == pytest.approx(0.1)
            assert config["loss_weights"]["alpha_rollout"] == 0.0
            assert config["loss_weights"]["alpha_energy"] == 0.0
        elif variant == "no_coercive_floor":
            assert config["beta_V_floor"] == 0.0
        elif variant == "no_interaction":
            assert config["interaction_radius"] == 0
        elif variant == "constant_mobility":
            assert config["mobility"] == "constant_identity"
        elif variant == "auxiliary_losses":
            assert config["architecture_only"] is False
            assert config["loss_weights"]["alpha_rollout"] > 0.0
            assert config["loss_weights"]["alpha_energy"] > 0.0


def test_structural_models_remove_unreachable_parameter_groups_and_preserve_bounds():
    torch.manual_seed(3)
    no_interaction = build_model(_args("no_interaction"))
    constant_mobility = build_model(_args("constant_mobility"))
    full = build_model(_args("full"))

    assert isinstance(no_interaction, NoInteractionLatent)
    assert isinstance(constant_mobility, ConstantMobilityLatent)
    assert "emb" not in no_interaction.state_dict()
    assert "K_net" not in constant_mobility.state_dict()
    assert sum(p.numel() for p in no_interaction.parameters()) < sum(
        p.numel() for p in full.parameters()
    )
    assert sum(p.numel() for p in constant_mobility.parameters()) < sum(
        p.numel() for p in full.parameters()
    )

    u = torch.rand(3, 8) * 0.8 + 0.1
    for model in (no_interaction, constant_mobility, full):
        output = model(u, 0.1)
        assert output.shape == u.shape
        assert torch.all(output > 0.0)
        assert torch.all(output < 1.0)

    z = constant_mobility.encode(u)
    assert torch.allclose(
        constant_mobility._dynamics_and_grad(z, None)[0],
        torch.ones_like(z),
    )


def test_batched_inference_uses_strictly_aligned_reference_steps():
    class Identity(torch.nn.Module):
        def forward(self, state, tau):
            return state

    times = torch.arange(5, dtype=torch.float32) * 0.05
    state = torch.tensor([0.25, 0.75], dtype=torch.float32)
    trajectory = [(times, state.repeat(5, 1))]
    model = Identity()
    result = benchmark_batched_inference(
        model,
        state.unsqueeze(0),
        trajectory,
        tau=0.1,
        rollout_steps=2,
        device="cpu",
        reference_dt=0.05,
    )
    assert result["state_steps"] == 2
    assert result["model_forward_calls"] == 2
    assert result["estimated_rk4_vector_field_calls"] is None
    assert math.isfinite(result["seconds"])

    with pytest.raises(ValueError, match="not exactly alignable"):
        benchmark_batched_inference(
            model,
            state.unsqueeze(0),
            trajectory,
            tau=0.075,
            rollout_steps=1,
            device="cpu",
            reference_dt=0.05,
        )
