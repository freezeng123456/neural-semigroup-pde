"""Analytic controls and source-provenance rejection for Fisher attribution."""
from __future__ import annotations

import copy
import hashlib
import io
from pathlib import Path
import tarfile

import pytest
import torch

from experiments import evaluate_fisher_inrange_attribution as evaluator
from experiments.experiment_artifacts import sha256_file
from experiments.models import LatentSemigroupNet, QueryTimeConditionedLatentFlow


class ConstantLatentFlow(torch.nn.Module):
    def __init__(self, *, conditioned=False, rate=0.5, eps=1e-6):
        super().__init__()
        self.latent_dynamics_requires_tau = conditioned
        self.rate = rate
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float64))

    def encode(self, state):
        return torch.logit(state.clamp(float(self.eps), 1 - float(self.eps)))

    def decode(self, latent):
        return torch.sigmoid(latent)

    def _interaction_matrix(self):
        return torch.zeros(2, 2)

    def latent_dynamics(self, latent, interactions, *, tau=None):
        value = self.rate if tau is None else self.rate * tau[:, None]
        return torch.ones_like(latent) * value


def small_evaluation(model):
    initial = torch.tensor([[0.3, 0.6], [0.4, 0.7]], dtype=torch.float64)
    horizons = (0.3, 0.6)
    references = {h: torch.sigmoid(torch.logit(initial) + model.rate * h) for h in horizons}
    return evaluator.evaluate_model(model, initial, references, horizons=horizons,
                                    substeps={0.075: 3, 0.15: 6})


def test_analytic_autonomous_flow_has_small_composition_and_exact_work():
    result = small_evaluation(ConstantLatentFlow())
    for lag, entry in result["by_lag"].items():
        for horizon, cell in entry["by_horizon"].items():
            assert cell["rollout_vs_reference"]["mse_mean"] < 1e-28
            assert cell["fixed_condition_composition"]["mse_mean"] < 1e-28
            assert cell["direct_conditioning_change"]["mse_mean"] < 1e-28
            assert cell["rhs_evaluations_each_path"] == round(float(horizon) / 0.025) * 4
            refinement = cell["time_integration_refinement"]
            assert refinement["refined_rhs_evaluations"] == 2 * refinement["production_rhs_evaluations"]
        assert entry["clamp_counterfactual"]["status"] == "available"
    assert all(v["mse_mean"] < 1e-28 for v in result["cross_lag"].values())


def test_conditioned_flow_separates_inrange_and_historical_conditioning():
    result = small_evaluation(ConstantLatentFlow(conditioned=True))
    assert result["cross_lag"]["0.6"]["mse_mean"] > 0
    for entry in result["by_lag"].values():
        cell = entry["by_horizon"]["0.6"]
        assert cell["fixed_condition_composition"]["mse_mean"] < 1e-28
        assert cell["historical_condition_composition"]["mse_mean"] > 0
        assert cell["direct_conditioning_change"]["mse_mean"] > 0


def test_roundtrip_representation_change_is_observed_at_call_boundaries():
    model = ConstantLatentFlow(rate=2, eps=0.01)
    initial = torch.full((1, 2), 0.99, dtype=torch.float64)
    composed, trace = evaluator.trajectory(model, initial, lag=0.075, horizons=(0.3,),
                                           substeps=3, conditioning_time=0.075)
    carried, _ = evaluator.trajectory(model, initial, lag=0.075, horizons=(0.3,),
                                      substeps=3, conditioning_time=0.075, reencode=False)
    assert trace["encoder_clipped_entries"] == 6
    assert trace["reencode_boundaries"] == 3
    assert trace["roundtrip_latent_abs_max"] == pytest.approx(0.15)
    assert evaluator.pair_metrics(composed[0.3], carried[0.3])["mse_mean"] > 0


def test_clamp_activations_are_counted_and_counterfactual_can_diverge():
    model = ConstantLatentFlow(rate=1000)
    initial = torch.full((1, 2), 0.5, dtype=torch.float64)
    _, trace = evaluator.trajectory(model, initial, lag=0.075, horizons=(0.15,),
                                    substeps=3, conditioning_time=0.075)
    assert trace["latent_clipped_entries"] > 0
    assert trace["roundtrip_latent_abs_max"] > 1


@pytest.mark.parametrize("model_class", [LatentSemigroupNet, QueryTimeConditionedLatentFlow])
def test_instrumented_rk4_matches_existing_model_without_mutating_weights(model_class):
    torch.manual_seed(17)
    model = model_class(N=4, hidden_V=[2], hidden_K=[2], stencil_radius=0,
                        interaction_radius=1).eval()
    initial = torch.tensor([[0.2, 0.4, 0.6, 0.8]], dtype=torch.float32)
    before = copy.deepcopy(model.state_dict())
    actual, _ = evaluator.trajectory(model, initial, lag=0.075, horizons=(0.075,),
                                     substeps=30, conditioning_time=0.075)
    with torch.no_grad():
        expected = model(initial, 0.075)
    torch.testing.assert_close(actual[0.075], expected, rtol=1e-6, atol=1e-7)
    assert model.ode_steps == 30
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())


def test_unequal_work_and_nonfinite_primary_states_are_rejected():
    model = ConstantLatentFlow()
    state = torch.full((1, 2), 0.5)
    with pytest.raises(ValueError, match="equal RK4 work"):
        evaluator.evaluate_model(model, state, {0.3: state}, horizons=(0.3,),
                                 substeps={0.075: 3, 0.15: 3})
    with pytest.raises(RuntimeError, match="non-finite"):
        evaluator.pair_metrics(state * float("nan"), state)


def write_archive(path, sources):
    with tarfile.open(path, "w:gz") as archive:
        for name, content in sources.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def checkpoint_fixture(tmp_path, role="baseline", n_sites=4):
    source_dir = Path(evaluator.__file__).parent
    sources = {name: (source_dir / name).read_bytes() for name in evaluator.REQUIRED_SOURCES}
    archive = tmp_path / "archive.tar.gz"
    write_archive(archive, {"experiments/" + name: data for name, data in sources.items()})
    cls = QueryTimeConditionedLatentFlow if role == "query" else LatentSemigroupNet
    kwargs = {"N": n_sites, "hidden_V": [2], "hidden_K": [2], "stencil_radius": 0,
              "interaction_radius": 1, "beta_V": 0.0, "beta_V_floor": 0.0}
    model = cls(**kwargs)
    metadata = {"regime": "variable", "training_seed": 31415, "train_taus": [0.025, 0.05, 0.1, 0.2],
        "model": "latent_query_time" if role == "query" else "latent",
        "model_config": {"class": cls.__name__, "kwargs": kwargs},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "architecture_only": role != "tube",
        "auxiliary_loss_weights": {"alpha_rollout": 0.0, "alpha_generator": 0.01 if role == "tube" else 0.0},
        "generator_supervision": {"sampling": "detached_learned_tube" if role == "tube" else "none"},
        "provenance": {"git_commit": "synthetic-test-only", "source_hashes": {
            name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}}}
    payload = {"model_state_dict": model.state_dict(), "run_metadata": metadata, "epoch": 20}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(payload, checkpoint)
    options = {"expected_sha256": sha256_file(checkpoint), "source_archive": archive,
               "archive_sha256": sha256_file(archive), "source_commit": "synthetic-test-only",
               "role": role, "seed": 31415}
    return checkpoint, options, payload


@pytest.mark.parametrize("role", ["baseline", "query", "tube"])
def test_reconstruction_uses_verified_archived_definition(tmp_path, role):
    checkpoint, options, payload = checkpoint_fixture(tmp_path, role)
    model, provenance = evaluator.load_archived_checkpoint(checkpoint, **options)
    assert type(model).__module__.startswith("_frozen_fisher_models_")
    assert provenance["training_source_hashes_verified"] == payload["run_metadata"]["provenance"]["source_hashes"]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in payload["model_state_dict"].items())


@pytest.mark.parametrize("corruption", ["digest", "source", "commit", "role", "seed", "parameter_count"])
def test_corrupted_checkpoint_contract_is_rejected(tmp_path, corruption):
    checkpoint, options, payload = checkpoint_fixture(tmp_path)
    metadata = payload["run_metadata"]
    if corruption == "digest":
        options["expected_sha256"] = "0" * 64
    else:
        if corruption == "source":
            metadata["provenance"]["source_hashes"]["models.py"] = "0" * 64
        elif corruption == "commit":
            metadata["provenance"]["git_commit"] = "wrong"
        elif corruption == "role":
            options["role"] = "tube"
        elif corruption == "seed":
            options["seed"] = 161803
        else:
            metadata["parameter_count"] += 1
        torch.save(payload, checkpoint)
        options["expected_sha256"] = sha256_file(checkpoint)
    with pytest.raises(ValueError):
        evaluator.load_archived_checkpoint(checkpoint, **options)


def test_archive_rejects_hash_mismatch_and_traversal_without_extraction(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    write_archive(archive, {"../escape.py": b"raise RuntimeError('must never execute')"})
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluator.archived_sources(archive, "0" * 64)
    with pytest.raises(ValueError, match="unsafe member"):
        evaluator.archived_sources(archive, sha256_file(archive))
    assert not (tmp_path.parent / "escape.py").exists()


def test_nonfinite_unclamped_counterfactual_is_marked_unavailable(monkeypatch):
    original = evaluator.trajectory
    def fail_only_unclamped(*args, **kwargs):
        if kwargs.get("clamp") is False:
            raise RuntimeError("non-finite latent state in RK4 trajectory")
        return original(*args, **kwargs)
    monkeypatch.setattr(evaluator, "trajectory", fail_only_unclamped)
    result = small_evaluation(ConstantLatentFlow())
    assert all(entry["clamp_counterfactual"]["status"] == "unavailable" for entry in result["by_lag"].values())


def test_complete_receipt_hashes_outputs_and_refuses_reuse(tmp_path, monkeypatch):
    # Exercise real archive/checkpoint/cache validation and durable completion.
    # Numerical path behavior is covered separately by analytic and model tests.
    from types import SimpleNamespace
    import json
    checkpoint, options, _ = checkpoint_fixture(tmp_path, n_sites=64)
    monkeypatch.setattr(evaluator, "FORMAL_SOURCE_ARCHIVE_SHA256", options["archive_sha256"])
    monkeypatch.setattr(evaluator, "FORMAL_SOURCE_COMMIT", "synthetic-test-only")
    cache_path = tmp_path / "cache.pt"
    initial = torch.full((2, 64), 0.5)
    times = torch.arange(961, dtype=torch.float64) * 0.005
    cache = {"test_u0": initial, "test_trajs": [(times, initial[i].repeat(961, 1)) for i in range(2)],
             "locked_test_config": {"pde": "fisher-kpp", "split": "locked_independent_test",
                 "test_seed": 314163, "N": 64, "L": 10.0, "nu": 0.1,
                 "reaction_rate": 1.0, "reference_dt": 0.005, "eval_horizon": 4.8}}
    torch.save(cache, cache_path)
    monkeypatch.setattr(evaluator, "FORMAL_LOCKED_CACHE_SHA256", sha256_file(cache_path))
    monkeypatch.setattr(evaluator, "evaluate_model", lambda model, initial, references: {"count": len(initial)})
    output = tmp_path / "new-output"
    args = SimpleNamespace(role="baseline", seed=31415, n_test=2, batch_size=1,
        checkpoint=str(checkpoint), checkpoint_sha256=options["expected_sha256"],
        source_archive=str(options["source_archive"]), test_cache=str(cache_path),
        device="cpu", output_dir=str(output))
    result = evaluator.run(args)
    assert result["smoke_only"] is True
    assert [batch["sample_start"] for batch in result["batches"]] == [0, 1]
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["results_sha256"] == sha256_file(output / "results.json")
    assert complete["manifest_sha256"] == sha256_file(output / "manifest.json")
    assert complete["inputs_unchanged"] and complete["evaluator_sources_unchanged"]
    with pytest.raises(ValueError, match="refusing to reuse"):
        evaluator.run(args)


def test_missing_checkpoint_does_not_create_an_output_directory(tmp_path):
    from types import SimpleNamespace
    output = tmp_path / "new-output"
    args = SimpleNamespace(role="baseline", seed=31415, n_test=2, batch_size=1,
                           checkpoint=str(tmp_path / "missing.pt"), output_dir=str(output))
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        evaluator.run(args)
    assert not output.exists()


def test_rk4_refinement_reduces_error_for_nonconstant_linear_latent_field():
    class LinearLatentFlow(ConstantLatentFlow):
        def latent_dynamics(self, latent, interactions, *, tau=None):
            return -4 * latent
    model = LinearLatentFlow()
    initial = torch.tensor([[0.2, 0.6]], dtype=torch.float64)
    horizon = 0.3
    # Use float64 in the exact expression so the convergence check is not
    # limited by the precision of an independently rounded reference.
    import math
    exact = torch.sigmoid(torch.logit(initial) * math.exp(-4 * horizon))
    production, _ = evaluator.trajectory(model, initial, lag=0.15, horizons=(horizon,),
                                         substeps=1, conditioning_time=0.15)
    refined, _ = evaluator.trajectory(model, initial, lag=0.15, horizons=(horizon,),
                                      substeps=2, conditioning_time=0.15)
    production_error = (production[horizon] - exact).norm().item()
    refined_error = (refined[horizon] - exact).norm().item()
    assert production_error > 10 * refined_error > 0
    result = evaluator.evaluate_model(model, initial, {horizon: exact}, horizons=(horizon,),
                                      substeps={0.075: 1, 0.15: 2})
    diagnostic = result["by_lag"]["0.15"]["by_horizon"]["0.3"]["time_integration_refinement"]
    assert diagnostic["production_vs_refined"]["mse_mean"] > 0
    assert diagnostic["refined_vs_reference"]["mse_mean"] < result["by_lag"]["0.15"]["by_horizon"]["0.3"]["rollout_vs_reference"]["mse_mean"]
