"""Tests for the Fisher generator/stability exploratory lane."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from experiments import aggregate_fisher_generator_stability as aggregator
from experiments import evaluate_fisher_generator_stability as evaluator
from experiments.experiment_artifacts import sha256_file
from experiments.fisher_frozen_inputs import (
    FORMAL_SOURCE_COMMIT,
    runtime_training_source_hashes,
)
from experiments.fisher_generator_metrics import (
    deterministic_sinusoidal_pair,
    fisher_spectral_generator,
    generator_mse_loss,
    generator_tube_mse_loss,
    learned_generator_tube_states,
    learned_physical_generator,
    one_sided_quotient,
    refinement_metrics,
    repeated_lag_snapshots,
    trapezoid_time_weights,
)
from experiments.models import LatentSemigroupNet


class _ToyPhysicalFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("eps", torch.tensor(1e-6))

    def encode(self, state):
        return torch.logit(state.clamp(self.eps, 1.0 - self.eps))

    def decode(self, latent):
        return torch.sigmoid(latent)

    def _interaction_matrix(self):
        return torch.zeros(2, 2)

    def latent_dynamics(self, latent, interactions):
        del interactions
        return -torch.ones_like(latent)

    def forward(self, state, tau):
        latent = self.encode(state)
        return self.decode(latent - float(tau))


def test_spectral_fisher_generator_and_reference_one_sided_bound():
    length = 2.0 * math.pi
    grid = torch.arange(32, dtype=torch.float64) * length / 32
    sine = torch.sin(3.0 * grid).reshape(1, -1)
    diffusion_only = fisher_spectral_generator(
        sine,
        length=length,
        diffusivity=0.2,
        reaction_rate=1e-12,
    )
    assert torch.allclose(diffusion_only, -1.8 * sine, rtol=1e-8, atol=1e-8)

    first = torch.stack((0.35 + 0.1 * torch.sin(grid), 0.45 + 0.1 * torch.cos(grid)))
    second = deterministic_sinusoidal_pair(first, amplitude=0.01)
    first_rhs = fisher_spectral_generator(
        first, length=length, diffusivity=0.1, reaction_rate=1.0
    )
    second_rhs = fisher_spectral_generator(
        second, length=length, diffusivity=0.1, reaction_rate=1.0
    )
    quotient, distance = one_sided_quotient(
        first_rhs, second_rhs, first, second, length=length
    )
    assert float(quotient.max().item()) <= 1.0 + 1e-10
    assert bool((distance > 0).all().item())


def test_physical_generator_uses_decoder_chain_rule():
    model = _ToyPhysicalFlow()
    states = torch.tensor([[0.2, 0.3], [0.7, 0.6]], dtype=torch.float64)
    physical, represented, latent = learned_physical_generator(
        model, states, conditioning_time=0.1
    )
    assert torch.allclose(represented, states)
    assert torch.allclose(physical, -states * (1.0 - states))
    assert torch.allclose(torch.sigmoid(latent), states)


def test_learned_tube_is_detached_and_uses_trapezoid_weights():
    model = _ToyPhysicalFlow()
    initial = torch.tensor([[0.2, 0.3], [0.7, 0.6]], requires_grad=True)
    times = (0.0, 0.25, 0.5, 1.0)
    tube = learned_generator_tube_states(model, initial, times)
    assert tube.shape == (2, 4, 2)
    assert tube.requires_grad is False
    assert torch.allclose(tube[:, 0], initial.detach())
    weights = trapezoid_time_weights(times)
    assert weights.tolist() == pytest.approx([0.125, 0.25, 0.375, 0.25])
    assert weights.sum().item() == pytest.approx(1.0)


def test_generator_tube_loss_matches_explicit_weighted_point_loss():
    model = _ToyPhysicalFlow()
    initial = torch.tensor([[0.2, 0.3], [0.7, 0.6]], dtype=torch.float64)
    times = (0.0, 0.5, 1.0)
    tube = learned_generator_tube_states(model, initial, times)
    actual = generator_tube_mse_loss(
        model,
        tube,
        times=times,
        conditioning_time=0.1,
        length=2.0,
        diffusivity=0.0,
        reaction_rate=1.0,
    )
    explicit_weights = trapezoid_time_weights(
        times, dtype=tube.dtype
    ).repeat(tube.shape[0])
    expected = generator_mse_loss(
        model,
        tube.reshape(-1, tube.shape[-1]),
        conditioning_time=0.1,
        length=2.0,
        diffusivity=0.0,
        reaction_rate=1.0,
        sample_weights=explicit_weights,
    )
    assert actual.item() == pytest.approx(expected.item())


def test_spectral_generator_supports_an_odd_periodic_grid():
    length = 2.0 * math.pi
    grid = torch.arange(31, dtype=torch.float64) * length / 31
    state = (0.4 + 0.1 * torch.sin(5.0 * grid)).reshape(1, -1)
    actual = fisher_spectral_generator(
        state,
        length=length,
        diffusivity=0.2,
        reaction_rate=1e-12,
    )
    expected = -0.5 * torch.sin(5.0 * grid).reshape(1, -1)
    assert torch.allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_deterministic_perturbation_is_reproducible_bounded_and_nonzero():
    states = torch.full((5, 16), 0.4)
    first = deterministic_sinusoidal_pair(states, amplitude=0.01)
    second = deterministic_sinusoidal_pair(states, amplitude=0.01)
    assert torch.equal(first, second)
    assert float(first.min().item()) >= 1e-5
    assert float(first.max().item()) <= 1.0 - 1e-5
    assert bool(((first - states).square().sum(dim=1) > 0).all().item())


def _small_model() -> LatentSemigroupNet:
    torch.manual_seed(7)
    return LatentSemigroupNet(
        N=8,
        hidden_V=[4],
        hidden_K=[4],
        stencil_radius=1,
        interaction_radius=1,
        beta_V=0.0,
        beta_V_floor=0.1,
    ).eval()


def test_refinement_is_side_effect_free_and_records_exact_rhs_work():
    model = _small_model()
    state = torch.stack(
        (
            torch.linspace(0.25, 0.55, 8),
            torch.linspace(0.35, 0.65, 8),
        )
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    original_steps = model.ode_steps
    summary, rows = refinement_metrics(
        model,
        state,
        state.clone(),
        lag=0.05,
        horizon=0.1,
        substeps=(1, 2, 3),
        finest_substeps=4,
        production_substeps=3,
        length=1.0,
    )
    assert summary["composition_depth"] == 2
    assert len(rows) == 3
    assert [row["total_rhs_evaluations"] for row in rows] == [8, 16, 24]
    assert model.ode_steps == original_steps
    assert summary["finest_encoder_clamp_activation_count"] == 0
    assert all(row["encoder_clamp_activation_count"] == 0 for row in rows)
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )


class _ClampActivatingFlow(_ToyPhysicalFlow):
    def latent_dynamics(self, latent, interactions):
        del interactions
        return torch.full_like(latent, 1000.0)


def test_repeated_lag_reports_encoder_clamp_activation():
    model = _ClampActivatingFlow()
    snapshots, execution = repeated_lag_snapshots(
        model,
        torch.full((2, 2), 0.5),
        lag=1.0,
        horizons=(1.0,),
        substeps_per_call=1,
    )
    assert 1.0 in snapshots
    assert execution["saturation_count"] > 0
    assert execution["encoder_clamp_activation_count"] > 0
    assert execution["reencoding_abs_max"] > 0


def _write_smoke_checkpoint(path: Path) -> None:
    model = LatentSemigroupNet(
        N=64,
        hidden_V=[2],
        hidden_K=[2],
        stencil_radius=1,
        interaction_radius=1,
        beta_V=0.0,
        beta_V_floor=0.1,
    )
    kwargs = {
        "N": 64,
        "hidden_V": [2],
        "hidden_K": [2],
        "stencil_radius": 1,
        "interaction_radius": 1,
        "beta_V": 0.0,
        "beta_V_floor": 0.1,
    }
    torch.save(
        {
            "epoch": 2,
            "best_val_mse": 0.1,
            "model_state_dict": model.state_dict(),
            "run_metadata": {
                "regime": "variable",
                "training_seed": 31415,
                "model": "latent",
                "model_config": {"class": "LatentSemigroupNet", "kwargs": kwargs},
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "architecture_only": True,
                "auxiliary_loss_weights": {
                    "alpha_rollout": 0.0,
                    "alpha_energy": 0.0,
                    "alpha_bound": 0.0,
                },
                "provenance": {
                    "git_commit": FORMAL_SOURCE_COMMIT,
                    "source_hashes": runtime_training_source_hashes(),
                    "data_cache_sha256": "smoke-cache",
                },
            },
        },
        path,
    )


def _write_smoke_cache(path: Path) -> None:
    n_sites = 64
    reference_dt = 0.005
    steps = round(4.8 / reference_dt)
    grid = 2.0 * math.pi * torch.arange(n_sites) / n_sites
    initial = torch.stack(
        (0.35 + 0.05 * torch.sin(grid), 0.45 + 0.04 * torch.cos(grid))
    )
    times = torch.arange(steps + 1, dtype=torch.float32) * reference_dt
    trajectories = [
        (times.clone(), initial[index].repeat(steps + 1, 1)) for index in range(2)
    ]
    torch.save(
        {
            "schema_version": 1,
            "locked_test_config": {
                "pde": "fisher-kpp",
                "split": "locked_independent_test",
                "test_seed": 314163,
                "N": n_sites,
                "L": 10.0,
                "nu": 0.1,
                "reaction_rate": 1.0,
                "reference_dt": reference_dt,
                "eval_horizon": 4.8,
            },
            "test_u0": initial,
            "test_trajs": trajectories,
        },
        path,
    )


def test_checkpoint_only_smoke_writes_complete_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "latent_best.pt"
    cache = tmp_path / "locked_test.pt"
    archive = tmp_path / "source.tar.gz"
    output = tmp_path / "fisher-generator-smoke"
    _write_smoke_checkpoint(checkpoint)
    _write_smoke_cache(cache)
    archive.write_bytes(b"frozen source archive")
    cache_digest = sha256_file(cache)
    archive_digest = sha256_file(archive)
    monkeypatch.setattr(evaluator, "FORMAL_LOCKED_CACHE_SHA256", cache_digest)
    monkeypatch.setattr(evaluator, "FORMAL_SOURCE_ARCHIVE_SHA256", archive_digest)
    args = evaluator.build_parser().parse_args(
        [
            "--model",
            "latent",
            "--seed",
            "31415",
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-sha256",
            sha256_file(checkpoint),
            "--test-cache",
            str(cache),
            "--cache-sha256",
            cache_digest,
            "--source-archive",
            str(archive),
            "--source-archive-sha256",
            archive_digest,
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--deterministic",
            "--batch-size",
            "2",
            "--reference-times",
            "0,0.15",
            "--learned-samples",
            "2",
            "--refinement-samples",
            "2",
            "--refinement-horizon",
            "0.15",
            "--refinement-substeps",
            "1,2",
            "--finest-substeps",
            "4",
            "--max-samples",
            "2",
            "--smoke",
        ]
    )
    checkpoint_before = checkpoint.read_bytes()
    cache_before = cache.read_bytes()
    result = evaluator.run_evaluation(args)
    assert result["status"] == "passed_smoke"
    assert result["summary"]["engineering_pass"] is True
    assert result["summary"]["reference_osl_pass"] is True
    assert result["summary"]["autonomy_pass"] is True
    assert checkpoint.read_bytes() == checkpoint_before
    assert cache.read_bytes() == cache_before
    for name in (
        "results.json",
        "generator_metrics.csv",
        "stability_metrics.csv",
        "refinement_metrics.csv",
        "exploratory_manifest.json",
        "receipt.json",
        "done",
    ):
        assert (output / name).is_file()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fake_full_cell(root: Path, seed: int, model: str) -> None:
    root.mkdir()
    residual = 0.5 if model == "latent" else 1.0
    learned_osl = 0.1 if model == "latent" else 0.2
    generator_rows: list[dict[str, object]] = []
    for state_source in ("reference_trajectory", "learned_repeated_lag_trajectory"):
        for conditioning_time in evaluator.TAUS:
            for snapshot_time in evaluator.REFERENCE_TIMES:
                row = {field: 0.0 for field in evaluator.GENERATOR_CSV_FIELDS}
                row.update(
                    {
                        "seed": seed,
                        "model": model,
                        "state_source": state_source,
                        "conditioning_time": conditioning_time,
                        "snapshot_time": snapshot_time,
                        "rms_relative_residual": residual,
                        "cosine_mean": 1.0,
                        "nonfinite_events": 0,
                    }
                )
                generator_rows.append(row)
    stability_rows: list[dict[str, object]] = []
    for conditioning_time in evaluator.TAUS:
        for snapshot_time in evaluator.REFERENCE_TIMES:
            for pair_family in evaluator.PAIR_FAMILIES:
                row = {field: 0.0 for field in evaluator.STABILITY_CSV_FIELDS}
                row.update(
                    {
                        "seed": seed,
                        "model": model,
                        "conditioning_time": conditioning_time,
                        "snapshot_time": snapshot_time,
                        "pair_family": pair_family,
                        "learned_max": learned_osl,
                        "reference_bound_pass": True,
                        "nonfinite_events": 0,
                    }
                )
                stability_rows.append(row)
    refinement_rows: list[dict[str, object]] = []
    for conditioning_time in evaluator.TAUS:
        for substeps in evaluator.REFINEMENT_SUBSTEPS:
            row = {field: 0.0 for field in evaluator.REFINEMENT_CSV_FIELDS}
            row.update(
                {
                    "seed": seed,
                    "model": model,
                    "conditioning_time": conditioning_time,
                    "substeps_per_call": substeps,
                    "observed_order_to_next": "",
                    "next_substeps_per_call": "",
                    "learned_flow_numerical_to_cache_discrepancy_ratio": (
                        0.01 if substeps == 30 else ""
                    ),
                    "learned_flow_numerical_confounding": (
                        False if substeps == 30 else ""
                    ),
                }
            )
            refinement_rows.append(row)
    _write_csv(root / "generator_metrics.csv", generator_rows)
    _write_csv(root / "stability_metrics.csv", stability_rows)
    _write_csv(root / "refinement_metrics.csv", refinement_rows)
    result = {
        "track": evaluator.TRACK,
        "status": "passed",
        "normal_exit": True,
        "smoke_only": False,
        "summary": {"engineering_pass": True},
    }
    manifest = {
        "exploratory": True,
        "do_not_use_for_formal": True,
        "status": "passed",
        "normal_exit": True,
    }
    (root / "results.json").write_text(json.dumps(result), encoding="utf-8")
    (root / "exploratory_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    artifact_names = (
        "results.json",
        "generator_metrics.csv",
        "stability_metrics.csv",
        "refinement_metrics.csv",
        "exploratory_manifest.json",
    )
    checkpoint_digest = aggregator.CHECKPOINT_SHA256[(seed, model)]
    receipt = {
        "track": evaluator.TRACK,
        "status": "passed",
        "normal_exit": True,
        "smoke_only": False,
        "seed": seed,
        "model": model,
        "input_hashes": {
            "checkpoint": {
                "before": checkpoint_digest,
                "after": checkpoint_digest,
                "unchanged": True,
            },
            "test_cache": {
                "before": aggregator.FORMAL_LOCKED_CACHE_SHA256,
                "after": aggregator.FORMAL_LOCKED_CACHE_SHA256,
                "unchanged": True,
            },
            "source_archive": {
                "before": aggregator.FORMAL_SOURCE_ARCHIVE_SHA256,
                "after": aggregator.FORMAL_SOURCE_ARCHIVE_SHA256,
                "unchanged": True,
            },
        },
        "artifact_hashes": {name: sha256_file(root / name) for name in artifact_names},
        "evaluator_source_hashes": {"evaluator.py": "f" * 64},
    }
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    (root / "done").write_text("passed\n", encoding="utf-8")


def test_six_cell_aggregate_recomputes_frozen_decision(tmp_path):
    roots = []
    for seed in aggregator.SEEDS:
        for model in aggregator.MODELS:
            root = tmp_path / f"cell-{seed}-{model}"
            _fake_full_cell(root, seed, model)
            roots.append(root)
    output = tmp_path / "aggregate"
    result = aggregator.aggregate(roots, output)
    assert result["status"] == "passed"
    assert result["decision"]["generator_material_advantage"] is True
    assert result["decision"]["empirical_stability_advantage"] is True
    assert result["decision"]["learned_flow_numerical_non_dominant"] is True
    assert (
        result["decision"]["explanation_branch"]
        == "advance_to_spatial_and_reference_consistency"
    )
    assert (output / "receipt.json").is_file()
    assert (output / "done").read_text(encoding="utf-8") == "passed\n"


def test_full_protocol_rejects_unfrozen_settings():
    args = evaluator.build_parser().parse_args(
        [
            "--model",
            "latent",
            "--seed",
            "31415",
            "--checkpoint",
            "/tmp/checkpoint.pt",
            "--checkpoint-sha256",
            "a" * 64,
            "--test-cache",
            "/tmp/cache.pt",
            "--source-archive",
            "/tmp/source.tar.gz",
            "--learned-samples",
            "63",
            "--output-dir",
            "/tmp/new-exploratory-root",
        ]
    )
    with pytest.raises(ValueError, match="frozen protocol"):
        evaluator.validate_args(args)


def _minimal_parser_args(output: Path) -> list[str]:
    return [
        "--model",
        "latent",
        "--seed",
        "31415",
        "--checkpoint",
        "/tmp/checkpoint.pt",
        "--checkpoint-sha256",
        "a" * 64,
        "--test-cache",
        "/tmp/cache.pt",
        "--source-archive",
        "/tmp/source.tar.gz",
        "--output-dir",
        str(output),
    ]


def test_smoke_still_requires_two_lags_zero_and_divisible_horizons(tmp_path):
    args = evaluator.build_parser().parse_args(
        _minimal_parser_args(tmp_path / "unused") + ["--smoke"]
    )
    args.taus = (0.075,)
    with pytest.raises(ValueError, match="exactly two"):
        evaluator.validate_args(args)

    args.taus = evaluator.TAUS
    args.reference_times = (0.15,)
    with pytest.raises(ValueError, match="time zero"):
        evaluator.validate_args(args)

    args.reference_times = (0.0, 0.1)
    with pytest.raises(ValueError, match="must be divisible"):
        evaluator.validate_args(args)


def test_main_records_failed_receipt_after_root_reservation(tmp_path, monkeypatch):
    output = tmp_path / "failed-root"

    def fail_after_reservation(args):
        output.mkdir()
        (output / "exploratory_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": evaluator.SCHEMA_VERSION,
                    "exploratory": True,
                    "do_not_use_for_formal": True,
                    "track": evaluator.TRACK,
                    "status": "running",
                }
            ),
            encoding="utf-8",
        )
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(evaluator, "run_evaluation", fail_after_reservation)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        evaluator.main(_minimal_parser_args(output))
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_exception"
    assert receipt["normal_exit"] is False
    assert (output / "failed").read_text(encoding="utf-8") == "failed_exception\n"
