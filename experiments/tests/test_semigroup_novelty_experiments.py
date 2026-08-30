"""Contracts for the refactored semigroup novelty experiment lane."""

import pytest
import torch

from experiments.calibrate_linear_advection_diffusion import run_calibration
from experiments.evaluate_fisher_semigroup_grid import WORK_EQUAL, evaluate_grid
from experiments.experiment_artifacts import (
    atomic_write_json,
    reserve_exploratory_root,
    sha256_file,
)
from experiments.fisher_frozen_inputs import (
    FORMAL_CACHE_PROFILE,
    FORMAL_SOURCE_COMMIT,
    load_frozen_fisher_cache,
    load_frozen_fisher_checkpoint,
    runtime_training_source_hashes,
)
from experiments.latent_integrators import (
    evolve_latent_flow,
    integration_work,
    substeps_for_rhs_budget,
)
from experiments.models import LatentSemigroupNet, QueryTimeConditionedLatentFlow
from experiments.prepare_fisher_phase_cache import FisherPhaseCacheConfig


def _small_kwargs():
    return {
        "N": 8,
        "hidden_V": [4],
        "hidden_K": [4],
        "stencil_radius": 1,
        "interaction_radius": 1,
        "beta_V": 0.0,
        "beta_V_floor": 0.1,
    }


@pytest.mark.parametrize(
    "model_class",
    [LatentSemigroupNet, QueryTimeConditionedLatentFlow],
)
def test_refactored_rk4_matches_historical_forward_without_model_mutation(model_class):
    torch.manual_seed(7)
    model = model_class(**_small_kwargs()).eval()
    model.ode_steps = 3
    state = torch.linspace(0.2, 0.8, 16).reshape(2, 8)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    expected = model(state, 0.1)
    actual = evolve_latent_flow(
        model,
        state,
        0.1,
        method="rk4",
        substeps=3,
    )

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert model.ode_steps == 3
    assert all(
        torch.equal(before[name], value) for name, value in model.state_dict().items()
    )


def test_rhs_budget_is_executable_and_exact():
    assert substeps_for_rhs_budget("euler", 120) == 120
    assert substeps_for_rhs_budget("rk2", 120) == 60
    assert substeps_for_rhs_budget("rk4", 120) == 30
    assert integration_work("rk4", 30)["rhs_evaluations"] == 120
    with pytest.raises(ValueError, match="not divisible"):
        substeps_for_rhs_budget("rk4", 119)


def test_grid_reuses_one_interface_and_asserts_equal_rhs_work():
    torch.manual_seed(11)
    model = LatentSemigroupNet(**_small_kwargs()).eval()
    initial = torch.full((2, 8), 0.4)
    references = {
        0.04: initial.clone(),
        0.08: initial.clone(),
    }
    cells, rows = evaluate_grid(
        model,
        initial,
        references,
        mode="unit_test",
        seed=31415,
        model_name="latent",
        integrator="rk2",
        taus=(0.02,),
        horizons=(0.04, 0.08),
        fixed_substeps=2,
        equal_rhs_per_base=4,
    )
    assert len(cells) == 4
    assert len(rows) == 4
    equal_cells = [cell for cell in cells if cell["work_semantics"] == WORK_EQUAL]
    assert len(equal_cells) == 2
    assert all(cell["work"]["rhs_work_equal"] for cell in equal_cells)
    assert [cell["composition_depth"] for cell in equal_cells] == [2, 4]


def test_artifact_root_is_unique_and_json_is_strict(tmp_path):
    root = reserve_exploratory_root(tmp_path / "exploratory-contract")
    atomic_write_json(root / "result.json", {"finite": 1.0, "nonfinite": float("nan")})
    assert '"nonfinite": null' in (root / "result.json").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to reuse"):
        reserve_exploratory_root(root)


def test_frozen_checkpoint_adapter_reconstructs_both_structure_and_provenance(tmp_path):
    kwargs = _small_kwargs()
    model = LatentSemigroupNet(**kwargs)
    checkpoint_path = tmp_path / "latent_best.pt"
    checkpoint = {
        "epoch": 4,
        "best_val_mse": 0.125,
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
                "data_cache_sha256": "unit-test-cache",
            },
        },
    }
    torch.save(checkpoint, checkpoint_path)
    loaded = load_frozen_fisher_checkpoint(
        checkpoint_path,
        expected_sha256=sha256_file(checkpoint_path),
        model_name="latent",
        seed=31415,
    )
    assert loaded.model_name == "latent"
    assert loaded.seed == 31415
    assert loaded.epoch == 4
    assert loaded.provenance()["parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )


def test_formal_cache_adapter_checks_identity_and_reference_grid(tmp_path):
    cache_path = tmp_path / "fisher_formal.pt"
    n_sites = 64
    reference_dt = 0.005
    horizon = 4.8
    steps = round(horizon / reference_dt)
    initial = torch.full((1, n_sites), 0.4)
    times = torch.arange(steps + 1, dtype=torch.float32) * reference_dt
    states = initial[0].repeat(steps + 1, 1)
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
                "eval_horizon": horizon,
            },
            "test_u0": initial,
            "test_trajs": [(times, states)],
        },
        cache_path,
    )
    _payload, validation, digest = load_frozen_fisher_cache(
        cache_path,
        expected_sha256=sha256_file(cache_path),
        profile=FORMAL_CACHE_PROFILE,
        taus=(0.075,),
        horizons=(1.2, 2.4, 4.8),
        max_samples=1,
    )
    assert digest == sha256_file(cache_path)
    assert validation["reference_steps"] == 960
    assert validation["evaluated_samples"] == 1


def test_phase_cache_cli_protocol_is_frozen():
    FisherPhaseCacheConfig().validate()
    with pytest.raises(ValueError, match="frozen"):
        FisherPhaseCacheConfig(n_test=127).validate()


def test_analytic_calibration_passes_fixed_order_windows(tmp_path):
    result = run_calibration(tmp_path / "linear-calibration")
    assert result["status"] == "passed"
    assert result["summary"] == {
        "exact_semigroup_pass": True,
        "all_errors_decrease": True,
        "all_order_windows_pass": True,
        "calibration_pass": True,
    }
    assert (tmp_path / "linear-calibration" / "receipt.json").is_file()
