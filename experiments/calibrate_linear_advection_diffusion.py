#!/usr/bin/env python3
"""Analytic periodic advection--diffusion calibration for ODE accounting.

The equation is ``u_t + c u_x = nu u_xx``.  Its Fourier flow is exact, so this
calibration separates exact semigroup composition from Euler/RK2/RK4 temporal
discretization error.  It trains no model and makes no cross-PDE performance
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from collections.abc import Sequence
from typing import Any

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    device_provenance,
    git_commit,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
)
from latent_integrators import integrate_explicit, integration_work  # noqa: E402


SCHEMA_VERSION = 1
TRACK = "linear_advection_diffusion_semigroup_calibration"
METHODS = ("euler", "rk2", "rk4")
DT_VALUES = (0.1, 0.05, 0.025, 0.0125)
ORDER_WINDOWS = {
    "euler": (0.8, 1.2),
    "rk2": (1.7, 2.3),
    "rk4": (3.4, 4.6),
}
EXACT_DEFECT_TOLERANCE = 5e-12
CSV_FIELDS = (
    "kind",
    "integrator",
    "dt",
    "substeps",
    "rhs_evaluations",
    "relative_l2_error",
    "observed_order_to_next",
)


def _initial_conditions(
    *,
    samples: int,
    n_sites: int,
    length: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    x = torch.arange(n_sites, dtype=torch.float64) * (length / n_sites)
    values = torch.zeros(samples, n_sites, dtype=torch.float64)
    for mode in range(1, 6):
        amplitudes = torch.randn(samples, 1, generator=generator, dtype=torch.float64)
        amplitudes *= 0.2 / mode
        phases = (
            2.0
            * math.pi
            * torch.rand(samples, 1, generator=generator, dtype=torch.float64)
        )
        values += amplitudes * torch.sin(2.0 * math.pi * mode * x / length + phases)
    return values


def _relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((prediction - target).reshape(-1))
    scale = torch.linalg.vector_norm(target.reshape(-1))
    return float((difference / torch.clamp(scale, min=1e-15)).item())


def run_calibration(output_dir: str | Path) -> dict[str, Any]:
    root = reserve_exploratory_root(output_dir)
    config = {
        "N": 32,
        "L": 10.0,
        "nu": 0.05,
        "advection_speed": 0.7,
        "horizon": 0.8,
        "samples": 32,
        "seed": 314165,
        "dt_values": list(DT_VALUES),
        "integrators": list(METHODS),
        "composition_partition": [0.1, 0.25, 0.45],
        "dtype": "torch.float64",
        "exact_defect_tolerance": EXACT_DEFECT_TOLERANCE,
        "order_windows": ORDER_WINDOWS,
    }
    source_digest_map = source_hashes(
        {
            "calibrate_linear_advection_diffusion.py": Path(__file__).resolve(),
            "experiment_artifacts.py": EXPERIMENTS_DIR / "experiment_artifacts.py",
            "latent_integrators.py": EXPERIMENTS_DIR / "latent_integrators.py",
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "running",
        "created_at_unix": time.time(),
        "git_commit": git_commit(REPOSITORY_ROOT),
        "config": config,
        "source_hashes": source_digest_map,
        "protocol": {
            "training": False,
            "checkpoint_access": False,
            "cache_access": False,
            "analytic_reference": True,
        },
    }
    atomic_write_json(root / "exploratory_manifest.json", manifest)

    device = torch.device("cpu")
    initial = _initial_conditions(
        samples=config["samples"],
        n_sites=config["N"],
        length=config["L"],
        seed=config["seed"],
    )
    wave_numbers = (
        2.0
        * math.pi
        * torch.fft.fftfreq(
            config["N"], d=config["L"] / config["N"], dtype=torch.float64
        )
    )
    eigenvalues = (
        -1j * config["advection_speed"] * wave_numbers
        - config["nu"] * wave_numbers.square()
    )

    def exact_flow(state: torch.Tensor, duration: float) -> torch.Tensor:
        spectrum = torch.fft.fft(state, dim=-1)
        multiplier = torch.exp(eigenvalues * float(duration))
        return torch.fft.ifft(spectrum * multiplier, dim=-1).real

    def rhs(state: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(state, dim=-1)
        return torch.fft.ifft(eigenvalues * spectrum, dim=-1).real

    horizon = float(config["horizon"])
    exact_direct = exact_flow(initial, horizon)
    exact_composed = initial
    for segment in config["composition_partition"]:
        exact_composed = exact_flow(exact_composed, float(segment))
    exact_defect = _relative_l2(exact_composed, exact_direct)

    method_results: dict[str, Any] = {}
    rows: list[dict[str, Any]] = [
        {
            "kind": "exact_composition",
            "integrator": "exact_fourier",
            "dt": None,
            "substeps": None,
            "rhs_evaluations": 0,
            "relative_l2_error": exact_defect,
            "observed_order_to_next": None,
        }
    ]
    all_orders_pass = True
    for method in METHODS:
        errors: list[float] = []
        refinements: list[dict[str, Any]] = []
        for dt in DT_VALUES:
            ratio = horizon / dt
            substeps = int(round(ratio))
            if not math.isclose(ratio, substeps, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError("calibration dt does not divide the horizon")
            numerical = integrate_explicit(
                rhs,
                initial,
                horizon,
                substeps=substeps,
                method=method,
            )
            error = _relative_l2(numerical, exact_direct)
            errors.append(error)
            work = integration_work(method, substeps)
            refinements.append(
                {
                    "dt": dt,
                    "substeps": substeps,
                    "rhs_evaluations": work["rhs_evaluations"],
                    "relative_l2_error": error,
                }
            )
        orders = [
            math.log(errors[index] / errors[index + 1], 2.0)
            for index in range(len(errors) - 1)
        ]
        lower, upper = ORDER_WINDOWS[method]
        method_pass = all(lower <= order <= upper for order in orders)
        all_orders_pass = all_orders_pass and method_pass
        for index, refinement in enumerate(refinements):
            refinement["observed_order_to_next"] = (
                orders[index] if index < len(orders) else None
            )
            rows.append(
                {
                    "kind": "temporal_convergence",
                    "integrator": method,
                    **refinement,
                }
            )
        method_results[method] = {
            "refinements": refinements,
            "observed_orders": orders,
            "accepted_order_window": [lower, upper],
            "order_pass": method_pass,
            "error_strictly_decreases": all(
                errors[index + 1] < errors[index] for index in range(len(errors) - 1)
            ),
        }

    exact_pass = exact_defect <= EXACT_DEFECT_TOLERANCE
    decreasing_pass = all(
        result["error_strictly_decreases"] for result in method_results.values()
    )
    passed = exact_pass and all_orders_pass and decreasing_pass
    result = {
        **manifest,
        "status": "passed" if passed else "failed_calibration",
        "normal_exit": True,
        "environment": device_provenance(device),
        "exact_semigroup": {
            "partition": config["composition_partition"],
            "relative_l2_defect": exact_defect,
            "tolerance": EXACT_DEFECT_TOLERANCE,
            "pass": exact_pass,
        },
        "integrators": method_results,
        "summary": {
            "exact_semigroup_pass": exact_pass,
            "all_errors_decrease": decreasing_pass,
            "all_order_windows_pass": all_orders_pass,
            "calibration_pass": passed,
        },
    }
    atomic_write_json(root / "results.json", result)
    atomic_write_csv(root / "metrics.csv", rows, fieldnames=CSV_FIELDS)
    completed_manifest = dict(manifest)
    completed_manifest["status"] = result["status"]
    completed_manifest["normal_exit"] = True
    atomic_write_json(root / "exploratory_manifest.json", completed_manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": result["status"],
        "normal_exit": True,
        **result["summary"],
        "results_sha256": sha256_file(root / "results.json"),
        "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
        "manifest_sha256": sha256_file(root / "exploratory_manifest.json"),
        "source_hashes": source_digest_map,
    }
    atomic_write_json(root / "receipt.json", receipt)
    (root / "done").write_text(f"{result['status']}\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_calibration(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": result["summary"],
                "output_dir": str(Path(args.output_dir).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
