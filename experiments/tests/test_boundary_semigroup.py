"""Contracts for the boundary-compatible semigroup exploratory lane."""

from argparse import Namespace

import pytest
import torch

from experiments.boundary_semigroup import (
    BoundarySemigroupFlow,
    build_reference_cache,
    composed_rollout,
    endpoint_max,
    evaluate_model,
    expected_metric_identities,
    parameter_count,
    parameter_fingerprint,
    project_homogeneous_dirichlet,
    reference_solve,
    sample_sine_initial_conditions,
    validate_reference_cache,
    wave_config,
)
from experiments.experiment_artifacts import sha256_file
from experiments.run_boundary_semigroup import prepare_data, run_cell


def _model(boundary_mode: str, temporal_mode: str) -> BoundarySemigroupFlow:
    return BoundarySemigroupFlow(
        n_grid=17,
        hidden_width=8,
        boundary_mode=boundary_mode,
        temporal_mode=temporal_mode,
        query_time_scale=0.08,
        rk4_steps=2,
    )


def test_dirichlet_projection_is_exact_and_idempotent():
    state = torch.randn(4, 17)
    projected = project_homogeneous_dirichlet(state)
    assert float(endpoint_max(projected)) == 0.0
    assert torch.equal(projected, project_homogeneous_dirichlet(projected))
    assert torch.equal(projected[:, 1:-1], state[:, 1:-1])


def test_all_four_cells_have_identical_parameters_and_initialization():
    counts = set()
    fingerprints = set()
    for boundary_mode in ("hard_dirichlet", "unconstrained"):
        for temporal_mode in ("autonomous", "query_time"):
            torch.manual_seed(31415)
            model = _model(boundary_mode, temporal_mode)
            counts.add(parameter_count(model))
            fingerprints.add(parameter_fingerprint(model))
    assert len(counts) == 1
    assert len(fingerprints) == 1


def test_hard_flow_projects_input_rhs_stages_and_output():
    torch.manual_seed(2)
    model = _model("hard_dirichlet", "query_time")
    state = torch.randn(5, 17)
    state[:, 0] = 0.4
    state[:, -1] = -0.3
    rhs = model.vector_field(state, 0.04)
    output, diagnostics = model.integrate(
        state, 0.16, map_duration=0.16, steps=8, return_diagnostics=True
    )
    assert float(endpoint_max(rhs)) == 0.0
    assert float(endpoint_max(output)) == 0.0
    assert diagnostics["max_stage_boundary"] == 0.0


def test_temporal_intervention_changes_only_duration_feature():
    torch.manual_seed(9)
    autonomous = _model("unconstrained", "autonomous")
    torch.manual_seed(9)
    query_time = _model("unconstrained", "query_time")
    assert parameter_fingerprint(autonomous) == parameter_fingerprint(query_time)
    state = torch.randn(3, 17)
    assert torch.equal(
        autonomous.vector_field(state, 0.02),
        autonomous.vector_field(state, 0.16),
    )
    assert not torch.allclose(
        query_time.vector_field(state, 0.02),
        query_time.vector_field(state, 0.16),
    )


def test_autonomous_equal_work_direct_and_composed_paths_match():
    torch.manual_seed(5)
    model = _model("hard_dirichlet", "autonomous")
    initial = project_homogeneous_dirichlet(torch.randn(6, 17))
    composed, work = composed_rollout(model, initial, tau=0.04, horizon=0.16)
    direct, direct_work = model.integrate(
        initial,
        0.16,
        map_duration=0.16,
        steps=model.rk4_steps * 4,
        return_diagnostics=True,
    )
    assert torch.allclose(direct, composed, rtol=1e-6, atol=1e-7)
    assert work["rhs_evaluations"] == direct_work["rhs_evaluations"]


def test_reference_solver_and_cache_have_exact_zero_endpoints():
    config = wave_config(smoke_only=True)
    initial = sample_sine_initial_conditions(
        4, n_grid=config.n_grid, modes=config.sine_modes, seed=7
    )
    reference = reference_solve(
        initial,
        0.16,
        dx=config.dx,
        diffusivity=config.diffusivity,
        reference_dt=config.reference_dt,
    )
    assert float(endpoint_max(reference)) == 0.0
    cache_a = build_reference_cache(config)
    cache_b = build_reference_cache(config)
    validation = validate_reference_cache(cache_a, config)
    assert validation["maximum_reference_endpoint"] == 0.0
    assert torch.equal(cache_a["train_u0"], cache_b["train_u0"])
    assert torch.equal(cache_a["train_target"], cache_b["train_target"])


def test_evaluator_has_frozen_grid_and_exact_work_counts():
    config = wave_config(smoke_only=True)
    cache = build_reference_cache(config)
    torch.manual_seed(11)
    model = _model("hard_dirichlet", "autonomous")
    rows, summary = evaluate_model(
        model, cache, config, device=torch.device("cpu")
    )
    identities = {
        (row["evaluation_kind"], row["tau"], row["horizon"]) for row in rows
    }
    assert identities == expected_metric_identities()
    assert all(row["equal_rhs_work"] for row in rows)
    assert all(
        row["composed_rhs_evaluations"] == row["direct_equal_rhs_evaluations"]
        for row in rows
    )
    assert summary["maximum_hard_boundary_evidence"] == 0.0


def test_smoke_runner_writes_parseable_validation_selected_artifacts(tmp_path):
    prep_root = tmp_path / "boundary-smoke-prep"
    prepare_result = prepare_data(str(prep_root), smoke_only=True)
    cache_path = prep_root / "boundary_wave1_cache.pt"
    cache_hash = sha256_file(cache_path)
    cell_root = tmp_path / "boundary-smoke-cell"
    result = run_cell(
        Namespace(
            output_dir=str(cell_root),
            data_cache=str(cache_path),
            seed=31415,
            boundary_mode="hard_dirichlet",
            temporal_mode="autonomous",
            device="cpu",
            smoke=True,
        )
    )
    assert prepare_result["status"] == "passed"
    assert result["status"] == "passed"
    assert result["hard_boundary_pass"]
    assert result["selected_epoch"] in {1, 2}
    assert sha256_file(cache_path) == cache_hash
    assert (cell_root / "receipt.json").is_file()
    assert (cell_root / "done").read_text(encoding="utf-8") == "passed\n"


def test_protocol_rejects_unknown_modes():
    with pytest.raises(ValueError, match="boundary_mode"):
        BoundarySemigroupFlow(
            n_grid=17,
            hidden_width=8,
            boundary_mode="periodic",
            temporal_mode="autonomous",
            query_time_scale=0.08,
            rk4_steps=2,
        )
