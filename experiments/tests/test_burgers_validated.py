"""Data isolation, conditioning semantics and artifacts for the v2 study."""
from __future__ import annotations

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch

from experiments import run_burgers_validated as runner


def diagnostic(n_grid=512, dt=0.0005):
    return {"version": "burgers-fourier-galerkin-v2", "N": n_grid,
            "L": 2 * math.pi, "nu": 0.01, "dt": dt,
            "max_step_energy_increase": 0.0, "max_mean_drift": 1e-16,
            "max_abs_state": 0.9, "initial_projection_rms": 1e-16}


def make_cache(n_train=4, n_val=2, n_test=2):
    """Distinct smooth samples with time-consistent synthetic trajectories."""
    times = torch.tensor(runner.TIMES, dtype=torch.float64)
    x = torch.arange(64, dtype=torch.float32) * (2 * math.pi / 64)
    splits = {}
    for offset, (name, count) in enumerate((("train", n_train), ("val", n_val), ("test", n_test))):
        scale = 0.1 + offset * 0.15 + torch.arange(count, dtype=torch.float32) * 0.005
        initial = scale[:, None] * torch.sin(x)[None]
        states = initial[:, None] * torch.exp(-times.float())[None, :, None]
        splits[name] = {"u0": initial, "times": times.clone(), "states": states,
                        "reference_diagnostics": [diagnostic() for _ in range(math.ceil(count / 50))]}
    train = splits["train"]
    train["tau"] = torch.tensor([runner.TRAIN_TAUS[i % 4] for i in range(n_train)], dtype=torch.float64)
    train["ut"] = torch.stack([train["states"][i, runner.time_index(times, tau)] for i, tau in enumerate(train["tau"].tolist())])
    metrics = {"max_relative_l2": 1e-7, "pooled_relative_l2": 1e-8, "per_sample_max_relative_l2": [1e-7, 1e-8]}
    audit = {"passed": True, "n_samples": 2, "audit_seed": 2026100304,
             "limits": dict(runner.REFERENCE_LIMITS), "times": list(runner.TIMES),
             "scope": runner.AUDIT_SCOPE,
             "solver_diagnostics": [diagnostic(), diagnostic(dt=0.00025), diagnostic(1024, 0.00025)],
             **{key: copy.deepcopy(metrics) for key in ("spatial", "temporal", "projected_spatial", "projected_restart_closure")}}
    return {"schema_version": runner.CACHE_SCHEMA, "reference_audit": audit,
            "projection": runner.PROJECTION, "target_scope": runner.TARGET_SCOPE,
            "exploratory": True, "do_not_use_for_formal": True,
            "config": {"N": 64, "L": 2 * math.pi, "nu": 0.01, "reference_N": 512, "reference_dt": 0.0005,
                       "train_taus": list(runner.TRAIN_TAUS), "test_taus": list(runner.TEST_TAUS),
                       "horizons": list(runner.HORIZONS), "data_seeds": dict(runner.DATA_SEEDS),
                       "n_train": n_train, "n_val": n_val, "n_test": n_test}, "splits": splits}


def test_cache_contract_rejects_wrong_time_and_mismatched_targets():
    cache = make_cache()
    assert runner.validate_cache(cache)["n_test"] == 2
    wrong = copy.deepcopy(cache)
    wrong["splits"]["test"]["times"][2] = 0.04001
    with pytest.raises(ValueError, match="times"):
        runner.validate_cache(wrong)
    wrong = copy.deepcopy(cache)
    wrong["splits"]["train"]["ut"][0, 0] += 1
    with pytest.raises(ValueError, match="train.ut"):
        runner.validate_cache(wrong)
    with pytest.raises(ValueError, match="exactly once"):
        runner.time_index(cache["splits"]["val"]["times"], 0.039)


@pytest.mark.parametrize("fault", ["overlap", "count", "dtype", "nonfinite", "schema", "data_seed"])
def test_cache_rejects_unusable_or_leaking_data(fault):
    cache = make_cache()
    if fault == "overlap":
        cache["splits"]["test"]["u0"][0] = cache["splits"]["val"]["u0"][0]
        cache["splits"]["test"]["states"][0, 0] = cache["splits"]["test"]["u0"][0]
    elif fault == "count":
        cache["config"]["n_test"] += 1
    elif fault == "dtype":
        cache["splits"]["val"]["times"] = cache["splits"]["val"]["times"].float()
    elif fault == "nonfinite":
        cache["splits"]["test"]["states"][0, -1, 0] = math.nan
    elif fault == "schema":
        cache["schema_version"] = 1
    else:
        cache["config"]["data_seeds"]["test"] = cache["config"]["data_seeds"]["train"]
    with pytest.raises(ValueError):
        runner.validate_cache(cache)


def test_energy_checks_detect_transient_growth_despite_lower_final_energy():
    energy = torch.tensor([[1.0, 0.8, 0.9, 0.7], [1.0, 0.9, 0.8, 0.7]], dtype=torch.float64)
    result = runner.energy_metrics(energy)
    assert result["energy_monotone"] == [False, True]
    assert result["energy_increase_step_count"] == [1, 0]
    assert result["energy_positive_increment_sum"] == pytest.approx([0.1, 0])
    assert result["energy_horizon_delta"] == pytest.approx([-0.3, -0.3])


def test_equal_work_paths_and_initial_state_identity():
    assert runner.fair_substeps(0.04) == 12
    assert runner.fair_substeps(0.08) == 24
    for horizon in runner.HORIZONS:
        assert 12 * runner.exact_depth(horizon, 0.04) == 24 * runner.exact_depth(horizon, 0.08)
    config = make_cache()["config"]
    for architecture, count in (("small", 33), ("large", 1313)):
        autonomous = runner.build_model(architecture, False, config)
        query = runner.build_model(architecture, True, config)
        query.load_state_dict(autonomous.state_dict())
        assert sum(p.numel() for p in autonomous.parameters()) == count
        assert runner.state_digest(query.state_dict()) == runner.state_digest(autonomous.state_dict())
        assert autonomous.query_conditioned is False and query.query_conditioned is True


class AnalyticConditionedFlow(torch.nn.Module):
    def __init__(self, query=True):
        super().__init__()
        self.query = query
        self.calls = []

    def integrate(self, state, duration, *, substeps, conditioning_tau):
        self.calls.append((duration, substeps, conditioning_tau))
        rate = conditioning_tau if self.query else 0.1
        return state * math.exp(-float(duration) * rate)


def test_conditioning_diagnostics_separate_structural_from_numerical_defect():
    cache = make_cache()
    model = AnalyticConditionedFlow()
    summary, per_sample = runner.evaluate_model(model, cache["splits"]["test"], device="cpu", batch_size=2)
    key = "tau=0.04:horizon=0.8"
    assert summary["composition_matched_conditioning"][key]["rms_defect_mean"] < 1e-6
    assert summary["composition_historical_conditioning"][key]["rms_defect_mean"] > 0.01
    assert summary["cross_lag"]["0.8"]["rms_defect_mean"] > 0.001
    assert len(per_sample["rollouts"][key]["energy_at_steps"][0]) == 21
    assert {step for duration, step, condition in model.calls if duration == 0.08 and condition == 0.08} == {24, 48}
    refinement = summary["time_integration_refinement"][key]
    assert refinement["production_substeps_per_call"] == 12
    assert refinement["refined_substeps_per_call"] == 24
    assert refinement["rms_vs_production_mean"] == 0.0
    json.dumps(summary, allow_nan=False)


def test_autonomous_zero_defect_needs_no_positive_floor_or_ratio():
    cache = make_cache()
    model = runner.build_model("small", False, cache["config"], ode_steps=1)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    summary, _ = runner.evaluate_model(model, cache["splits"]["test"], device="cpu", batch_size=2, base_steps=1)
    assert summary["cross_lag"]["0.8"]["rms_defect_mean"] == 0.0
    json.dumps(summary, allow_nan=False)


def test_train_uses_validation_only_and_always_checks_final_epoch(tmp_path):
    torch.set_num_threads(1)
    cache = make_cache()
    args = SimpleNamespace(seed=31415, batch_size=4, lr=1e-3, weight_decay=1e-5,
                           epochs=1, validation_interval=10, device="cpu", eval_batch_size=2)
    model = runner.build_model("small", False, cache["config"], ode_steps=1)
    root = tmp_path / "cell"
    result = runner.train_model(model, cache["splits"]["train"], cache["splits"]["val"], args, root, {})
    assert result["selected_epoch"] == 1
    assert (root / "selected.pt").is_file() and (root / "final.pt").is_file()
    history = json.loads((root / "history.json").read_text())
    assert history[-1]["val_mse"] is not None
    assert result["best_val_mse"] == runner.validation_mse(model, cache["splits"]["val"], device="cpu", batch_size=2)
    with pytest.raises(FileExistsError):
        runner.train_model(model, cache["splits"]["train"], cache["splits"]["val"], args, root, {})


def test_strict_json_refuses_nan_and_zero_relative_reference_is_explicit(tmp_path):
    path = tmp_path / "metrics.json"
    with pytest.raises(ValueError):
        runner.strict_json(path, {"mse": math.nan})
    assert not path.exists()
    errors = runner.sample_errors(torch.ones(2, 64), torch.zeros(2, 64))
    assert errors["relative_l2"] == [None, None]
    assert runner.error_summary(errors)["undefined_relative_l2_count"] == 2


def test_exact_flux_oracle_conserves_mean_and_has_no_parameters():
    config = make_cache()["config"]
    model = runner.ExactFluxOracle(config, 12)
    assert sum(p.numel() for p in model.parameters()) == 0
    state = make_cache()["splits"]["train"]["u0"]
    assert torch.allclose(model.rhs(state, 0.04).mean(dim=1), torch.zeros(4), atol=1e-8)


def setup_main_fixture(tmp_path, monkeypatch):
    # The artifact contract hashes source files independently of importing them.
    # Isolate those paths so this test also works before the cache-builder lands.
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    for name in ("run_burgers_validated.py", "burgers_reference.py", "prepare_burgers_validated_cache.py",
                 "run_burgers_flux_ablation.py", "run_burgers_semigroup_screen.py", "pde_solver.py",
                 "seed_utils.py", "experiment_artifacts.py"):
        (source_dir / name).write_text("# isolated provenance fixture\n")
    monkeypatch.setattr(runner, "EXPERIMENTS_DIR", source_dir)
    cache_path = tmp_path / "cache.pt"
    torch.save(make_cache(), cache_path)
    output = tmp_path / "study"
    argv = ["--data-cache", str(cache_path), "--output-dir", str(output), "--seed", "31415",
            "--epochs", "1", "--architectures", "small", "--ode-steps", "1"]
    return cache_path, output, argv


def test_main_completes_only_after_pair_selection_and_refuses_overwrite(tmp_path, monkeypatch):
    cache_path, output, argv = setup_main_fixture(tmp_path, monkeypatch)
    original_evaluate = runner.evaluate_model

    def checked_evaluate(*args, **kwargs):
        assert (output / "small_autonomous" / "selected.pt").is_file()
        assert (output / "small_query" / "selected.pt").is_file()
        assert (output / "selection_complete.json").is_file()
        assert not (output / "COMPLETE.json").exists()
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "evaluate_model", checked_evaluate)
    before = runner.sha256_file(cache_path)
    result = runner.main(argv)
    marker = json.loads((output / "COMPLETE.json").read_text())
    assert marker["summary_sha256"] == runner.sha256_file(output / "summary.json")
    assert result["cache_sha256_before"] == result["cache_sha256_after"] == before
    assert result["cells"]["small_autonomous"]["initial_state_sha256"] == result["cells"]["small_query"]["initial_state_sha256"]
    assert result["scope"] == "pilot"
    with pytest.raises(ValueError, match="refusing to reuse"):
        runner.main(argv)


def test_failed_test_evaluation_never_writes_completion_marker(tmp_path, monkeypatch):
    _, output, argv = setup_main_fixture(tmp_path, monkeypatch)

    def fail_evaluation(*args, **kwargs):
        raise FloatingPointError("synthetic non-finite rollout")

    monkeypatch.setattr(runner, "evaluate_model", fail_evaluation)
    with pytest.raises(FloatingPointError, match="synthetic"):
        runner.main(argv)
    assert (output / "selection_complete.json").exists()
    assert not (output / "COMPLETE.json").exists()


@pytest.mark.parametrize("fault", ["failed", "missing", "relaxed_limits", "spatial_gate", "temporal_gate",
                                   "physical_gate", "wrong_grid", "wrong_dt", "sample_count",
                                   "projection", "target_scope", "sampled_scope", "batch_gate"])
def test_reference_validation_cannot_be_bypassed_by_schema_or_passed_flag(fault):
    cache = make_cache()
    audit = cache["reference_audit"]
    if fault == "failed":
        audit["passed"] = False
    elif fault == "missing":
        del cache["reference_audit"]
    elif fault == "relaxed_limits":
        audit["limits"]["spatial_max_relative_l2"] = 0.1
    elif fault in {"spatial_gate", "temporal_gate"}:
        name, value = ("spatial", 0.002) if fault == "spatial_gate" else ("temporal", 0.00002)
        audit[name]["max_relative_l2"] = value
        audit[name]["per_sample_max_relative_l2"][0] = value
    elif fault == "physical_gate":
        audit["solver_diagnostics"][0]["max_step_energy_increase"] = 1e-9
    elif fault == "wrong_grid":
        audit["solver_diagnostics"][2]["N"] = 512
    elif fault == "wrong_dt":
        audit["solver_diagnostics"][1]["dt"] = 0.0005
    elif fault == "sample_count":
        audit["n_samples"] = 3
    elif fault in {"projection", "target_scope"}:
        del cache[fault]
    elif fault == "sampled_scope":
        audit["scope"] = "Certified for all initial states"
    else:
        cache["splits"]["test"]["reference_diagnostics"][0]["max_mean_drift"] = 1e-9
    with pytest.raises(ValueError):
        runner.validate_cache(cache)


def test_nominal_parameter_matching_records_inactive_control_weights():
    config = make_cache()["config"]
    for architecture, nominal, inactive in (("small", 33, 8), ("large", 1313, 32)):
        autonomous = runner.parameter_counts(runner.build_model(architecture, False, config))
        query = runner.parameter_counts(runner.build_model(architecture, True, config))
        assert autonomous["parameter_count"] == query["parameter_count"] == nominal
        assert autonomous["inactive_control_weight_count"] == inactive
        assert autonomous["active_parameter_count"] == nominal - inactive
        assert query["inactive_control_weight_count"] == 0
        assert query["active_parameter_count"] == nominal


def test_refinement_reports_per_sample_integration_sensitivity():
    cache = make_cache()
    model = runner.ExactFluxOracle(cache["config"], 1)
    summary, per_sample = runner.evaluate_model(model, cache["splits"]["test"], device="cpu", batch_size=2, base_steps=1)
    for key, record in summary["time_integration_refinement"].items():
        samples = per_sample["time_integration_refinement"][key]
        assert len(samples["mse_vs_production"]) == 2
        assert record["refined_substeps_per_call"] == 2 * record["production_substeps_per_call"]
        assert record["mse_vs_production_mean"] == pytest.approx(sum(samples["mse_vs_production"]) / 2)
        assert samples["rms_vs_production"] == pytest.approx([math.sqrt(value) for value in samples["mse_vs_production"]])
    assert summary["time_integration_refinement"]["tau=0.04:horizon=0.8"]["rms_vs_production_mean"] > 0.0
