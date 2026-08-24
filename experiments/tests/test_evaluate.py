import math

import pytest
import torch

from experiments.evaluate import (
    compute_equal_work_semigroup_defect,
    evaluate_full,
    evaluate_ode_steps_sweep,
    evaluate_ode_substep_convergence,
    resolve_reference_stride,
)
from experiments.phase0_audit import infer_reference_dt, parse_ode_steps


class EulerDecayModel(torch.nn.Module):
    def __init__(self, ode_steps=4):
        super().__init__()
        self.ode_steps = ode_steps

    def forward(self, u, tau):
        if tau == 0:
            return u
        step = float(tau) / self.ode_steps
        value = u
        for _ in range(self.ode_steps):
            value = value * (1.0 - step)
        return value

    def energy(self, u):
        return torch.sum(u**2, dim=-1).mean()


class FailingSubstepModel(EulerDecayModel):
    def forward(self, u, tau):
        if self.ode_steps == 8:
            raise RuntimeError("intentional substep failure")
        return super().forward(u, tau)

    def encode(self, u):
        return u

    def latent_dynamics(self, z):
        return -z


class LatentDiagnosticModel(EulerDecayModel):
    def encode(self, u):
        return u

    def latent_dynamics(self, z):
        return -z


class BatchTrackingDecayModel(EulerDecayModel):
    def __init__(self, ode_steps=4):
        super().__init__(ode_steps=ode_steps)
        self.batch_sizes = []

    def forward(self, u, tau):
        self.batch_sizes.append(u.shape[0])
        return super().forward(u, tau)


class WorkTrackingDecayModel(EulerDecayModel):
    def __init__(self, ode_steps=4):
        super().__init__(ode_steps=ode_steps)
        self.calls = []

    def forward(self, u, tau):
        self.calls.append(
            {"tau": float(tau), "ode_steps": int(self.ode_steps), "batch": u.shape[0]}
        )
        return super().forward(u, tau)


class NoEnergyDecayModel(torch.nn.Module):
    def __init__(self, ode_steps=4):
        super().__init__()
        self.ode_steps = ode_steps

    def forward(self, u, tau):
        step = float(tau) / self.ode_steps
        value = u
        for _ in range(self.ode_steps):
            value = value * (1.0 - step)
        return value


def make_reference(n_snapshots=9, reference_dt=0.05):
    times = torch.arange(n_snapshots, dtype=torch.float64) * reference_dt
    states = torch.exp(-times).to(torch.float32).unsqueeze(1)
    return times, states


def test_reference_stride_rejects_silent_rounding():
    assert resolve_reference_stride(0.1, 0.05) == 2
    with pytest.raises(ValueError, match="not exactly alignable"):
        resolve_reference_stride(0.1, 0.03)


def test_phase0_legacy_data_time_spacing_is_inferred_without_silent_rounding():
    times, states = make_reference(n_snapshots=3, reference_dt=0.005)
    assert infer_reference_dt([(times, states)]) == pytest.approx(0.005)


def test_evaluation_uses_actual_aligned_step_counts():
    model = EulerDecayModel()
    times, states = make_reference(n_snapshots=5)
    metrics, details = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        device="cpu",
    )
    assert metrics["n_valid"] == 1
    assert metrics["min_evaluated_steps"] == 2
    assert details["per_step_counts"] == [1, 1, 0, 0]
    assert details["evaluated_steps"] == [2]


def test_full_evaluation_batches_rollout_and_semigroup_inference():
    model = BatchTrackingDecayModel()
    times, states = make_reference(n_snapshots=5)
    val_u0 = torch.tensor([[1.0], [0.8], [0.6]])
    trajectories = []
    for scale in (1.0, 0.8, 0.6):
        trajectories.append((times, states * scale))
    metrics, details = evaluate_full(
        model,
        val_u0,
        trajectories,
        tau=0.1,
        rollout_steps=2,
        reference_dt=0.05,
        device="cpu",
    )
    assert metrics["n_valid"] == 3
    assert details["per_step_counts"] == [3, 3]
    assert model.batch_sizes
    assert set(model.batch_sizes) == {3}


def test_batched_full_evaluation_matches_individual_evaluation():
    times, states = make_reference(n_snapshots=9)
    scales = (1.0, 0.8, 0.6)
    val_u0 = torch.tensor([[scale] for scale in scales])
    trajectories = [(times, states * scale) for scale in scales]
    batch_metrics, _ = evaluate_full(
        EulerDecayModel(),
        val_u0,
        trajectories,
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        device="cpu",
    )
    individual_metrics = []
    for sample_idx in range(len(scales)):
        metrics, _ = evaluate_full(
            EulerDecayModel(),
            val_u0[sample_idx : sample_idx + 1],
            [trajectories[sample_idx]],
            tau=0.1,
            rollout_steps=4,
            reference_dt=0.05,
            device="cpu",
        )
        individual_metrics.append(metrics)

    for key in (
        "rollout_mse_mean",
        "rollout_rel_l2_mean",
        "bound_viol_mean",
        "numerical_semigroup_defect_mse_mean",
        "numerical_semigroup_defect_abs_l2_mean",
        "numerical_semigroup_defect_rel_l2_mean",
        "learned_energy_mono_frac_mean",
    ):
        expected = sum(metrics[key] for metrics in individual_metrics) / len(
            individual_metrics
        )
        assert batch_metrics[key] == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_relative_l2_and_max_bound_metrics_are_reported():
    times, states = make_reference(n_snapshots=5)
    metrics, details = evaluate_full(
        EulerDecayModel(),
        torch.tensor([[1.2]]),
        [(times, states * 1.2)],
        tau=0.1,
        rollout_steps=2,
        reference_dt=0.05,
        lower_bound=0.0,
        upper_bound=1.0,
        device="cpu",
    )
    assert metrics["rollout_rel_l2_mean"] >= 0.0
    assert metrics["bound_viol_max"] > 0.0
    assert len(details["per_step_rel_l2"]) == 2
    assert len(details["per_step_bound_max"]) == 2


def test_single_aligned_step_is_retained_without_semigroup_defect():
    model = EulerDecayModel()
    times, states = make_reference(n_snapshots=3)
    metrics, details = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        device="cpu",
    )
    assert metrics["n_valid"] == 1
    assert metrics["min_evaluated_steps"] == 1
    assert metrics["numerical_semigroup_defect_status"] == "not_evaluated"
    assert math.isnan(metrics["numerical_semigroup_defect_mse_mean"])
    assert details["per_step_counts"] == [1, 0, 0, 0]


def test_reference_length_and_timestamp_must_be_strictly_validated():
    model = EulerDecayModel()
    times, states = make_reference(n_snapshots=5)
    with pytest.raises(ValueError, match="same length"):
        evaluate_full(
            model,
            torch.tensor([[1.0]]),
            [(times[:-1], states)],
            tau=0.1,
            rollout_steps=2,
            reference_dt=0.05,
            device="cpu",
        )

    times = times.clone()
    times[2] += 0.01
    with pytest.raises(ValueError, match="does not match"):
        evaluate_full(
            model,
            torch.tensor([[1.0]]),
            [(times, states)],
            tau=0.1,
            rollout_steps=2,
            reference_dt=0.05,
            device="cpu",
        )


def test_learned_and_physical_energy_are_separate():
    model = EulerDecayModel()
    times, states = make_reference()

    def increasing_physical_energy(u):
        return -torch.sum(u**2, dim=-1).mean()

    metrics, _ = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        physical_energy_fn=increasing_physical_energy,
        device="cpu",
    )
    assert metrics["learned_energy_status"] == "evaluated"
    assert metrics["physical_energy_status"] == "evaluated"
    assert metrics["learned_energy_mono_frac_mean"] == pytest.approx(1.0)
    assert metrics["energy_mono_frac_mean"] == pytest.approx(1.0)
    assert metrics["physical_energy_mono_frac_mean"] == pytest.approx(0.0)

    no_physical_metrics, _ = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        device="cpu",
    )
    assert no_physical_metrics["physical_energy_status"] == "not_evaluated"
    assert "physical_energy_mono_frac_mean" not in no_physical_metrics


def test_numerical_semigroup_fields_and_substep_restoration():
    model = EulerDecayModel(ode_steps=4)
    times, states = make_reference()
    metrics, _ = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        device="cpu",
    )
    assert math.isfinite(metrics["numerical_semigroup_defect_abs_l2_mean"])
    assert math.isfinite(metrics["numerical_semigroup_defect_rel_l2_mean"])
    assert metrics["semigroup_defect_mean"] == pytest.approx(
        metrics["numerical_semigroup_defect_mse_mean"]
    )

    diagnostic = evaluate_ode_substep_convergence(
        model,
        torch.tensor([[1.0]]),
        tau=0.2,
        substep_counts=(2, 8),
        device="cpu",
    )
    assert diagnostic["status"] == "evaluated"
    assert diagnostic["comparisons"][0]["absolute_l2"] > 0
    assert model.ode_steps == 4

    failing = FailingSubstepModel(ode_steps=4)
    with pytest.raises(RuntimeError, match="intentional"):
        evaluate_ode_substep_convergence(
            failing,
            torch.tensor([[1.0]]),
            tau=0.2,
            substep_counts=(2, 8),
            device="cpu",
        )
    assert failing.ode_steps == 4


def test_equal_work_defect_matches_direct_and_composed_work():
    model = WorkTrackingDecayModel(ode_steps=7)
    defect = compute_equal_work_semigroup_defect(
        model,
        torch.tensor([[1.0]]),
        tau=0.1,
        steps=3,
        base_ode_steps=4,
        max_pairs=1,
        device="cpu",
    )

    assert defect["status"] == "evaluated"
    assert defect["semantics"] == "equal_work_base_step"
    assert defect["pair_work"] == [
        {
            "i": 1,
            "j": 1,
            "direct_ode_steps": 8,
            "composed_ode_steps": [4, 4],
            "direct_total_ode_steps": 8,
            "composed_total_ode_steps": 8,
        }
    ]
    assert [call["ode_steps"] for call in model.calls] == [8, 4, 4]
    assert model.ode_steps == 7


def test_equal_work_restores_original_ode_steps_after_failure():
    failing = FailingSubstepModel(ode_steps=4)
    with pytest.raises(RuntimeError, match="intentional"):
        compute_equal_work_semigroup_defect(
            failing,
            torch.tensor([[1.0]]),
            tau=0.1,
            steps=3,
            base_ode_steps=8,
            max_pairs=1,
            device="cpu",
        )
    assert failing.ode_steps == 4


def test_phase0_ode_steps_sweep_supports_default_counts_and_restores_model():
    model = EulerDecayModel(ode_steps=30)
    audit = evaluate_ode_steps_sweep(
        model,
        torch.tensor([[1.0]]),
        tau=0.2,
        substep_counts=parse_ode_steps("5,10,15,30,60,120"),
        semigroup_steps=3,
        max_pairs=1,
        include_defects=True,
        device="cpu",
    )

    assert audit["status"] == "evaluated"
    assert audit["ode_steps"] == [5, 10, 15, 30, 60, 120]
    assert len(audit["sweep"]) == 6
    assert audit["sweep"][0]["production_fixed_ode_steps_defect"]["semantics"] == (
        "production_fixed_ode_steps"
    )
    assert audit["sweep"][0]["equal_work_base_step_defect"]["base_ode_steps"] == 5
    assert model.ode_steps == 30


def test_optional_latent_norm_diagnostics_are_explicit():
    times, states = make_reference()
    metrics, _ = evaluate_full(
        LatentDiagnosticModel(),
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=2,
        reference_dt=0.05,
        collect_latent_diagnostics=True,
        device="cpu",
    )
    assert metrics["latent_diagnostics_status"] == "evaluated"
    assert metrics["latent_l2_norm_max"] == pytest.approx(1.0)
    assert metrics["latent_vector_field_l2_norm_max"] == pytest.approx(1.0)


def test_energy_positive_increments_are_reported_per_step_and_at_maximum():
    model = EulerDecayModel()
    times, states = make_reference()

    def increasing_physical_energy(u):
        return -torch.sum(u**2, dim=-1).mean()

    metrics, details = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=4,
        reference_dt=0.05,
        physical_energy_fn=increasing_physical_energy,
        device="cpu",
    )

    assert metrics["learned_energy_positive_increment_mean"] == pytest.approx(0.0)
    assert metrics["learned_energy_positive_increment_max"] == pytest.approx(0.0)
    assert metrics["physical_energy_positive_increment_mean"] > 0.0
    assert metrics["physical_energy_positive_increment_max"] > 0.0
    assert len(details["learned_energy_positive_increment_per_step"]) == 4
    assert len(details["physical_energy_positive_increment_per_step"]) == 4
    assert max(details["physical_energy_max_positive_increment_per_step"]) > 0.0


def test_no_energy_model_has_explicit_unavailable_energy_diagnostics():
    model = NoEnergyDecayModel()
    times, states = make_reference()
    metrics, details = evaluate_full(
        model,
        torch.tensor([[1.0]]),
        [(times, states)],
        tau=0.1,
        rollout_steps=2,
        reference_dt=0.05,
        device="cpu",
    )

    assert metrics["learned_energy_status"] == "not_evaluated"
    assert "learned_energy_positive_increment_mean" not in metrics
    assert "learned_energy_positive_increment_max" not in metrics
    assert "learned_energy_positive_increment_per_step" not in details
