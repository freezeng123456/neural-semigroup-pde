import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pde_solver import FisherKPPSolver
from reference_audit import run_reference_audit, sample_initial_condition_specs


def test_fisher_solver_supports_float64_and_rejects_silent_time_truncation():
    solver = FisherKPPSolver(N=16, dt=0.01, dtype=torch.float64)
    initial = torch.full((16,), 0.4, dtype=torch.float64)
    times, states = solver.solve(initial, 0.02)
    assert times.dtype == torch.float64
    assert states.dtype == torch.float64
    with pytest.raises(ValueError, match="integer multiple"):
        solver.solve(initial, 0.025)


def test_resolution_independent_initial_condition_is_nested():
    spec = sample_initial_condition_specs(1, seed=7)[0]
    coarse = spec.evaluate(16, 10.0)
    fine = spec.evaluate(32, 10.0)
    assert torch.allclose(coarse, fine[::2], atol=1e-12, rtol=1e-12)


def test_reference_audit_returns_finite_temporal_and_spatial_rows():
    result = run_reference_audit(
        n_samples=1,
        seed=3,
        resolutions=[16, 32],
        dts=[0.02, 0.01],
        final_time=0.04,
        L=10.0,
        nu=0.1,
        r=1.0,
    )
    assert {row["kind"] for row in result["rows"]} == {
        "temporal",
        "spatial",
        "float32_vs_float64",
        "production_vs_fine",
    }
    assert all(torch.isfinite(torch.tensor(row["relative_l2"])) for row in result["rows"])
    production = next(
        row for row in result["rows"] if row["kind"] == "production_vs_fine"
    )
    assert production["candidate_dtype"] == "float32"
    assert production["reference_dtype"] == "float64"


def test_reference_audit_requires_nested_grids_and_aligned_times():
    with pytest.raises(ValueError, match="nested"):
        run_reference_audit(
            n_samples=1,
            seed=1,
            resolutions=[24, 32],
            dts=[0.02, 0.01],
            final_time=0.04,
            L=10.0,
            nu=0.1,
            r=1.0,
        )
    with pytest.raises(ValueError, match="integer multiple"):
        run_reference_audit(
            n_samples=1,
            seed=1,
            resolutions=[16, 32],
            dts=[0.03, 0.01],
            final_time=0.04,
            L=10.0,
            nu=0.1,
            r=1.0,
        )
