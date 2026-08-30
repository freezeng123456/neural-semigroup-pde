"""Contracts for the Boundary-Family Semigroup Wave 2 lane."""

from argparse import Namespace

import pytest
import torch

from experiments.boundary_family_semigroup import (
    BOUNDARY_FAMILIES,
    BOUNDARY_THRESHOLD,
    ENFORCEMENT_MODES,
    NONUNIFORM_PARTITIONS,
    REFINEMENT_STEPS,
    TEMPORAL_MODES,
    BoundaryFamilyFlow,
    BoundaryFamilySpec,
    build_reference_cache,
    composed_partition,
    evaluate_model,
    expected_metric_identities,
    parameter_count,
    parameter_fingerprint,
    residual_max,
    validate_reference_cache,
    wave_config,
)
from experiments.experiment_artifacts import sha256_file
from experiments.run_boundary_family_semigroup import (
    cache_filename,
    prepare_data,
    run_cell,
)


def _spec(family: str, n_grid: int = 17) -> BoundaryFamilySpec:
    return BoundaryFamilySpec(family=family, dx=1.0 / (n_grid - 1))


def _model(
    family: str,
    enforcement_mode: str,
    temporal_mode: str,
) -> BoundaryFamilyFlow:
    return BoundaryFamilyFlow(
        n_grid=17,
        hidden_width=8,
        boundary_spec=_spec(family),
        enforcement_mode=enforcement_mode,
        temporal_mode=temporal_mode,
        query_time_scale=0.08,
        rk4_steps=2,
    )


@pytest.mark.parametrize("family", BOUNDARY_FAMILIES)
def test_affine_state_and_tangent_reconstruction(family):
    spec = _spec(family)
    state = torch.randn(6, 17)
    tangent = torch.randn(6, 17)
    projected_state = spec.project_state(state)
    projected_tangent = spec.project_tangent(tangent)
    assert (
        float(residual_max(spec.state_residual(projected_state))) <= BOUNDARY_THRESHOLD
    )
    assert (
        float(residual_max(spec.equation_residual(projected_state)))
        <= BOUNDARY_THRESHOLD
    )
    assert (
        float(residual_max(spec.tangent_residual(projected_tangent)))
        <= BOUNDARY_THRESHOLD
    )
    assert torch.equal(projected_state[:, 1:-1], state[:, 1:-1])
    assert torch.equal(projected_tangent[:, 1:-1], tangent[:, 1:-1])


def test_frozen_boundary_values_and_outward_normal_signs():
    dirichlet = _spec("inhomogeneous_dirichlet")
    projected = dirichlet.project_state(torch.zeros(2, 17))
    assert torch.equal(projected[:, 0], torch.full((2,), 0.20))
    assert torch.equal(projected[:, -1], torch.full((2,), -0.10))

    neumann = _spec("homogeneous_neumann")
    linear = torch.arange(17, dtype=torch.float32).repeat(2, 1)
    projected = neumann.project_state(linear)
    residual = neumann.equation_residual(projected)
    assert torch.equal(residual, torch.zeros_like(residual))

    robin = _spec("robin")
    projected = robin.project_state(torch.linspace(-0.3, 0.4, 17).repeat(2, 1))
    assert float(residual_max(robin.equation_residual(projected))) <= BOUNDARY_THRESHOLD


def test_all_18_cells_per_seed_have_identical_parameters_and_initialization():
    counts = set()
    fingerprints = set()
    for family in BOUNDARY_FAMILIES:
        for enforcement_mode in ENFORCEMENT_MODES:
            for temporal_mode in TEMPORAL_MODES:
                torch.manual_seed(31415)
                model = _model(family, enforcement_mode, temporal_mode)
                counts.add(parameter_count(model))
                fingerprints.add(parameter_fingerprint(model))
    assert len(counts) == 1
    assert len(fingerprints) == 1


@pytest.mark.parametrize("family", BOUNDARY_FAMILIES)
def test_hard_flow_projects_input_rhs_stages_and_output(family):
    torch.manual_seed(2)
    model = _model(family, "hard", "query_time")
    state = torch.randn(5, 17)
    state[:, 0] += 0.4
    state[:, -1] -= 0.3
    rhs = model.vector_field(state, 0.04)
    output, diagnostics = model.integrate(
        state, 0.16, map_duration=0.16, steps=8, return_diagnostics=True
    )
    spec = model.boundary_spec
    assert float(residual_max(spec.tangent_residual(rhs))) <= BOUNDARY_THRESHOLD
    assert float(residual_max(spec.state_residual(output))) <= BOUNDARY_THRESHOLD
    assert diagnostics["max_stage_boundary_state"] <= BOUNDARY_THRESHOLD
    assert diagnostics["max_stage_boundary_equation"] <= BOUNDARY_THRESHOLD
    assert diagnostics["max_rhs_boundary_tangent"] <= BOUNDARY_THRESHOLD


def test_temporal_intervention_changes_only_duration_feature():
    torch.manual_seed(9)
    autonomous = _model("robin", "unconstrained", "autonomous")
    torch.manual_seed(9)
    query_time = _model("robin", "unconstrained", "query_time")
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


@pytest.mark.parametrize("family", BOUNDARY_FAMILIES)
def test_reference_cache_is_deterministic_and_boundary_compatible(family):
    config = wave_config(family, smoke_only=True)
    cache_a = build_reference_cache(config)
    cache_b = build_reference_cache(config)
    validation = validate_reference_cache(cache_a, config)
    assert validation["maximum_reference_state_residual"] <= config.boundary_threshold
    assert (
        validation["maximum_reference_equation_residual"] <= config.boundary_threshold
    )
    assert torch.equal(cache_a["train_u0"], cache_b["train_u0"])
    assert torch.equal(cache_a["train_target"], cache_b["train_target"])


def test_nonuniform_partitions_are_equal_work_without_identical_micro_steps():
    torch.manual_seed(5)
    model = _model("homogeneous_neumann", "hard", "autonomous")
    initial = model.boundary_spec.project_state(torch.randn(6, 17))
    for partition in NONUNIFORM_PARTITIONS:
        for steps in REFINEMENT_STEPS:
            composed, work = composed_partition(
                model,
                initial,
                partition=partition,
                steps_per_segment=steps,
            )
            direct, direct_work = model.integrate(
                initial,
                sum(partition),
                map_duration=sum(partition),
                steps=steps * len(partition),
                return_diagnostics=True,
            )
            assert composed.shape == direct.shape
            assert work["rhs_evaluations"] == direct_work["rhs_evaluations"]
            assert work["rk4_steps"] == steps * len(partition)


def test_evaluator_has_complete_grid_and_exact_work_counts():
    config = wave_config("robin", smoke_only=True)
    cache = build_reference_cache(config)
    torch.manual_seed(11)
    model = _model("robin", "hard", "autonomous")
    rows, summary = evaluate_model(model, cache, config, device=torch.device("cpu"))
    identities = {
        (row["evaluation_kind"], row["partition_id"], row["steps_per_segment"])
        for row in rows
    }
    assert identities == expected_metric_identities(config.neural_rk4_steps)
    assert all(row["equal_rhs_work"] for row in rows)
    assert all(
        row["composed_rhs_evaluations"] == row["direct_equal_rhs_evaluations"]
        for row in rows
    )
    assert summary["maximum_hard_boundary_evidence"] <= config.boundary_threshold


def test_smoke_runner_writes_penalty_cell_artifacts(tmp_path):
    prep_root = tmp_path / "boundary-family-smoke-prep"
    prepare_result = prepare_data(
        str(prep_root), boundary_family="inhomogeneous_dirichlet", smoke_only=True
    )
    cache_path = prep_root / cache_filename("inhomogeneous_dirichlet")
    cache_hash = sha256_file(cache_path)
    cell_root = tmp_path / "boundary-family-smoke-cell"
    result = run_cell(
        Namespace(
            output_dir=str(cell_root),
            data_cache=str(cache_path),
            seed=31415,
            boundary_family="inhomogeneous_dirichlet",
            enforcement_mode="penalty",
            temporal_mode="query_time",
            device="cpu",
            smoke=True,
        )
    )
    assert prepare_result["status"] == "passed"
    assert result["status"] == "passed"
    assert result["penalty_contract"]["applied"]
    assert result["selected_epoch"] in {1, 2}
    assert sha256_file(cache_path) == cache_hash
    assert (cell_root / "receipt.json").is_file()
    assert (cell_root / "done").read_text(encoding="utf-8") == "passed\n"


def test_unknown_family_and_mode_are_rejected():
    with pytest.raises(ValueError, match="boundary family"):
        wave_config("periodic")
    with pytest.raises(ValueError, match="enforcement mode"):
        BoundaryFamilyFlow(
            n_grid=17,
            hidden_width=8,
            boundary_spec=_spec("robin"),
            enforcement_mode="learned",
            temporal_mode="autonomous",
            query_time_scale=0.08,
            rk4_steps=2,
        )
