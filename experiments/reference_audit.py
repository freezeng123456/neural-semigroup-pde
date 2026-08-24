#!/usr/bin/env python3
"""Convergence audit for the one-dimensional Fisher--KPP reference solver.

The training benchmark historically used a 64-point, float32 reference solve.
This audit evaluates the same continuous random Fourier initial conditions on
nested grids and separates temporal refinement from spatial refinement.  It is
diagnostic only: it does not train a neural model or overwrite cached data.
"""

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pde_solver import FisherKPPSolver


@dataclass(frozen=True)
class FourierInitialCondition:
    """Resolution-independent specification of a smooth periodic state."""

    modes: tuple
    phases: tuple
    amplitudes: tuple
    center: float = 0.4
    half_range: float = 0.4

    def evaluate(self, N, L, *, dtype=torch.float64):
        x = torch.arange(int(N), dtype=dtype) * (float(L) / int(N))
        raw = torch.zeros(int(N), dtype=dtype)
        for mode, phase, amplitude in zip(self.modes, self.phases, self.amplitudes):
            raw = raw + float(amplitude) * torch.sin(
                2.0 * math.pi * int(mode) * x / float(L) + float(phase)
            )
        # Normalize by the Fourier coefficients, not by a sampled grid maximum.
        # The same continuous initial condition then agrees exactly on nested
        # grids at their shared points.
        scale = max(
            sum(abs(float(value)) for value in self.amplitudes),
            torch.finfo(dtype).eps,
        )
        return (self.center + self.half_range * raw / scale).clamp(0.05, 0.95)


def sample_initial_condition_specs(n_samples, seed):
    rng = np.random.default_rng(int(seed))
    specs = []
    for _ in range(int(n_samples)):
        n_modes = int(rng.integers(2, 6))
        specs.append(
            FourierInitialCondition(
                modes=tuple(int(x) for x in rng.integers(1, 4, size=n_modes)),
                phases=tuple(float(x) for x in rng.uniform(0.0, 2.0 * np.pi, size=n_modes)),
                amplitudes=tuple(float(x) for x in rng.uniform(0.05, 0.3, size=n_modes)),
            )
        )
    return specs


def relative_l2(candidate, reference, eps=1e-15):
    diff = torch.linalg.vector_norm(candidate - reference)
    denom = torch.linalg.vector_norm(reference).clamp_min(float(eps))
    return float((diff / denom).item())


def solve_final(spec, *, N, L, nu, r, dt, final_time, dtype=torch.float64):
    solver = FisherKPPSolver(
        N=N,
        L=L,
        nu=nu,
        r=r,
        dt=dt,
        dtype=dtype,
    )
    u0 = spec.evaluate(N, L, dtype=dtype)
    _, trajectory = solver.solve(u0, final_time, save_every=round(final_time / dt))
    return trajectory[-1]


def run_reference_audit(
    *,
    n_samples,
    seed,
    resolutions,
    dts,
    final_time,
    L,
    nu,
    r,
):
    resolutions = tuple(sorted({int(value) for value in resolutions}))
    dts = tuple(sorted({float(value) for value in dts}, reverse=True))
    if len(resolutions) < 2:
        raise ValueError("at least two resolutions are required")
    if len(dts) < 2:
        raise ValueError("at least two time steps are required")
    finest_resolution = resolutions[-1]
    finest_dt = dts[-1]
    production_resolution = resolutions[0]
    production_dt = dts[0]
    for resolution in resolutions[:-1]:
        if finest_resolution % resolution != 0:
            raise ValueError("resolutions must be nested divisors of the finest grid")
    for dt in dts:
        ratio = float(final_time) / dt
        if not np.isclose(ratio, round(ratio), rtol=1e-10, atol=1e-12):
            raise ValueError("final_time must be an integer multiple of every dt")

    specs = sample_initial_condition_specs(n_samples, seed)
    temporal_rows = []
    spatial_rows = []
    precision_rows = []
    production_rows = []
    for sample_index, spec in enumerate(specs):
        temporal_outputs = {
            dt: solve_final(
                spec,
                N=finest_resolution,
                L=L,
                nu=nu,
                r=r,
                dt=dt,
                final_time=final_time,
            )
            for dt in dts
        }
        temporal_reference = temporal_outputs[finest_dt]
        for dt in dts[:-1]:
            temporal_rows.append(
                {
                    "sample": sample_index,
                    "kind": "temporal",
                    "resolution": finest_resolution,
                    "dt": dt,
                    "reference_dt": finest_dt,
                    "relative_l2": relative_l2(temporal_outputs[dt], temporal_reference),
                }
            )

        finest_output = temporal_reference
        for resolution in resolutions[:-1]:
            coarse = solve_final(
                spec,
                N=resolution,
                L=L,
                nu=nu,
                r=r,
                dt=finest_dt,
                final_time=final_time,
            )
            stride = finest_resolution // resolution
            restricted_reference = finest_output[::stride]
            spatial_rows.append(
                {
                    "sample": sample_index,
                    "kind": "spatial",
                    "resolution": resolution,
                    "reference_resolution": finest_resolution,
                    "dt": finest_dt,
                    "relative_l2": relative_l2(coarse, restricted_reference),
                }
            )

        production_float64 = solve_final(
            spec,
            N=production_resolution,
            L=L,
            nu=nu,
            r=r,
            dt=production_dt,
            final_time=final_time,
            dtype=torch.float64,
        )
        production_float32 = solve_final(
            spec,
            N=production_resolution,
            L=L,
            nu=nu,
            r=r,
            dt=production_dt,
            final_time=final_time,
            dtype=torch.float32,
        ).to(torch.float64)
        precision_rows.append(
            {
                "sample": sample_index,
                "kind": "float32_vs_float64",
                "resolution": production_resolution,
                "dt": production_dt,
                "candidate_dtype": "float32",
                "reference_dtype": "float64",
                "relative_l2": relative_l2(
                    production_float32,
                    production_float64,
                ),
            }
        )
        production_stride = finest_resolution // production_resolution
        production_rows.append(
            {
                "sample": sample_index,
                "kind": "production_vs_fine",
                "resolution": production_resolution,
                "dt": production_dt,
                "candidate_dtype": "float32",
                "reference_resolution": finest_resolution,
                "reference_dt": finest_dt,
                "reference_dtype": "float64",
                "relative_l2": relative_l2(
                    production_float32,
                    finest_output[::production_stride],
                ),
            }
        )

    rows = temporal_rows + spatial_rows + precision_rows + production_rows
    grouped = {}
    for row in rows:
        key = (row["kind"], row["resolution"], row["dt"])
        grouped.setdefault(key, []).append(row["relative_l2"])
    aggregates = []
    for key, values in sorted(grouped.items()):
        aggregates.append(
            {
                "kind": key[0],
                "resolution": key[1],
                "dt": key[2],
                "mean_relative_l2": float(np.mean(values)),
                "max_relative_l2": float(np.max(values)),
                "std_relative_l2": float(np.std(values)),
                "n_samples": len(values),
            }
        )
    return {
        "config": {
            "n_samples": int(n_samples),
            "seed": int(seed),
            "resolutions": list(resolutions),
            "dts": list(dts),
            "final_time": float(final_time),
            "L": float(L),
            "nu": float(nu),
            "r": float(r),
            "dtype": "float64",
            "production_label": {
                "resolution": production_resolution,
                "dt": production_dt,
                "dtype": "float32",
            },
        },
        "initial_condition_specs": [asdict(spec) for spec in specs],
        "aggregates": aggregates,
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument(
        "--dts", type=float, nargs="+", default=[0.005, 0.002, 0.001, 0.0005]
    )
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--r", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_reference_audit(
        n_samples=args.n_samples,
        seed=args.seed,
        resolutions=args.resolutions,
        dts=args.dts,
        final_time=args.final_time,
        L=args.L,
        nu=args.nu,
        r=args.r,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "reference_audit.json")
    csv_path = os.path.join(args.output_dir, "reference_audit.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    fieldnames = sorted({key for row in result["rows"] for key in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])
    print(json.dumps({"json": json_path, "csv": csv_path, "aggregates": result["aggregates"]}, indent=2))


if __name__ == "__main__":
    main()
