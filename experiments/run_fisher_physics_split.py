#!/usr/bin/env python3
"""Zero-parameter Fisher--KPP physics-split evaluation.

Implements ``docs/research/FISHER_PHYSICS_SPLIT_PROTOCOL.md``.
The protocol commit predates every number this runner writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch import nn

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate import evaluate_full  # noqa: E402
from evaluate_fisher_kpp_checkpoint import (  # noqa: E402
    LOWER_BOUND,
    UPPER_BOUND,
    cache_prefix_trajectories,
)
from run_fisher_fair import (  # noqa: E402
    fisher_energy,
    reference_steps_for_duration,
    rollout_steps_for_horizon,
)

EXPECTED_CACHE_SHA256 = "29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e"
FORMAL_DECISION = EXPERIMENTS_DIR / "results" / "fisher_formal_18_cell_decision.json"
DEFAULT_CACHE = Path(
    "/jizhicfs/yuyechen/neural_semigroup/"
    "20260828-fisher-kpp-formal-locked-cache-s314163-6373455-h20-r1/"
    "locked/fisher_kpp_n64_test_s314163_h4p8.pt"
)
TEST_TAUS = (0.075, 0.15)
HORIZONS = (1.2, 2.4, 4.8)
MATERIAL_THRESHOLD = 0.90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tau_column(tau, state: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(tau):
        duration = tau.to(device=state.device, dtype=state.dtype)
    else:
        duration = torch.as_tensor(float(tau), device=state.device, dtype=state.dtype)
    while duration.ndim < state.ndim:
        duration = duration.unsqueeze(-1)
    return duration


class FisherPhysicsSplit(nn.Module):
    """Exact heat plus exact logistic reaction, then a (0, 1) projection."""

    def __init__(self, *, n_grid: int = 64, length: float = 10.0, nu: float = 0.1, reaction_rate: float = 1.0):
        super().__init__()
        self.N = int(n_grid)
        self.length = float(length)
        self.nu = float(nu)
        self.reaction_rate = float(reaction_rate)
        self.m = LOWER_BOUND
        self.M = UPPER_BOUND
        self.eps = 1e-6
        spacing = self.length / self.N
        wavenumbers = 2.0 * math.pi * torch.fft.rfftfreq(self.N, d=spacing)
        self.register_buffer("diffusion_eigenvalues", -self.nu * (wavenumbers ** 2))

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        clipped = state.clamp(self.m + self.eps, self.M - self.eps)
        scaled = (clipped - self.m) / (self.M - self.m)
        return torch.log(scaled / (1.0 - scaled))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.m + (self.M - self.m) * torch.sigmoid(latent)

    def project(self, state: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(state))

    def heat_semigroup(self, state: torch.Tensor, tau) -> torch.Tensor:
        duration = tau_column(tau, state)
        decay = torch.exp(duration * self.diffusion_eigenvalues)
        transformed = torch.fft.rfft(state.to(dtype=torch.float64), dim=-1)
        evolved = torch.fft.irfft(
            transformed * decay.to(dtype=transformed.dtype),
            n=self.N,
            dim=-1,
        )
        return evolved.to(dtype=state.dtype)

    def increment(self, state: torch.Tensor) -> torch.Tensor:
        return self.reaction_rate * state * (1.0 - state)

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        linear = self.heat_semigroup(state, tau)
        step = tau_column(tau, linear)
        return self.project(linear + step * self.increment(linear))


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def geometric_mean(values) -> float:
    logs = [math.log(max(float(value), 1e-30)) for value in values]
    return math.exp(sum(logs) / len(logs))


def load_published_a_cells(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return list(payload["cells"])


def compare_to_formal_a(physics_cells: dict, published: list[dict]) -> dict:
    ratios = {}
    rows = []
    for cell in published:
        key = f"tau={cell['tau']},T={cell['horizon']}"
        physics_mse = float(physics_cells[key]["rollout_mse_mean"])
        a_mse = float(cell["a_mse"])
        ratio = physics_mse / a_mse
        ratios[f"seed={cell['seed']},{key}"] = ratio
        rows.append(
            {
                "seed": int(cell["seed"]),
                "tau": float(cell["tau"]),
                "horizon": float(cell["horizon"]),
                "c_phys_mse": physics_mse,
                "a_mse": a_mse,
                "c_phys_over_a": ratio,
            }
        )
    pooled = geometric_mean(ratios.values())
    seed_gms = {}
    for seed in sorted({int(cell["seed"]) for cell in published}):
        seed_values = [
            row["c_phys_over_a"] for row in rows if int(row["seed"]) == seed
        ]
        seed_gms[str(seed)] = geometric_mean(seed_values)
    dominating_seeds = sum(value <= MATERIAL_THRESHOLD for value in seed_gms.values())
    decision = (
        "fisher_physics_split_dominates_formal_A"
        if pooled <= MATERIAL_THRESHOLD
        else "fisher_formal_A_survives_the_clean_split"
    )
    return {
        "cells": rows,
        "ratios": ratios,
        "geometric_mean_c_phys_over_a": pooled,
        "seed_geometric_mean_c_phys_over_a": seed_gms,
        "seeds_with_c_phys_at_most_0.90": dominating_seeds,
        "material_threshold": MATERIAL_THRESHOLD,
        "decision": decision,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--formal-decision", type=Path, default=FORMAL_DECISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=1.0)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    return parser.parse_args(argv)


@torch.no_grad()
def evaluate_physics_split(args) -> dict:
    cache_path = Path(args.test_cache).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"locked test cache does not exist: {cache_path}")
    cache_sha = sha256_file(cache_path)
    paired = cache_sha == EXPECTED_CACHE_SHA256
    test_data = torch.load(cache_path, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = FisherPhysicsSplit(
        n_grid=args.N,
        length=args.L,
        nu=args.nu,
        reaction_rate=args.reaction_rate,
    ).to(device)
    model.eval()
    if parameter_count(model) != 0:
        raise RuntimeError("FisherPhysicsSplit must have zero trainable parameters")

    class _Args:
        reference_dt = float(args.reference_dt)

    helper = _Args()
    energy = fisher_energy(args)
    cells = {}
    started = time.perf_counter()
    for tau in TEST_TAUS:
        for horizon in HORIZONS:
            depth = rollout_steps_for_horizon(horizon, tau)
            prefixes = cache_prefix_trajectories(test_data, horizon, helper)
            metrics, _details = evaluate_full(
                model,
                test_data["test_u0"],
                prefixes,
                tau=tau,
                rollout_steps=depth,
                device=device,
                model_name="c_physics_split",
                lower_bound=LOWER_BOUND,
                upper_bound=UPPER_BOUND,
                reference_dt=args.reference_dt,
                physical_energy_fn=energy,
            )
            cells[f"tau={tau},T={horizon}"] = {
                "tau": float(tau),
                "horizon": float(horizon),
                "rollout_steps": depth,
                "rollout_mse_mean": float(metrics["rollout_mse_mean"]),
                "rollout_rel_l2_mean": float(metrics["rollout_rel_l2_mean"]),
                "bound_viol_mean": float(metrics["bound_viol_mean"]),
                "physical_energy_mono_frac": float(
                    metrics.get("physical_energy_mono_frac_mean", float("nan"))
                ),
                "semigroup_defect_mean": float(metrics["semigroup_defect_mean"]),
            }
    comparison = compare_to_formal_a(cells, load_published_a_cells(args.formal_decision))
    return {
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/FISHER_PHYSICS_SPLIT_PROTOCOL.md",
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha,
        "expected_cache_sha256": EXPECTED_CACHE_SHA256,
        "comparison_is_paired": paired,
        "parameter_count": 0,
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "c_physics_split": cells,
        "comparison": comparison,
        "reference_steps_T4.8": reference_steps_for_duration(4.8, args.reference_dt),
    }


def main(argv=None):
    args = parse_args(argv)
    payload = evaluate_physics_split(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    decision = payload["comparison"]["decision"]
    pooled = payload["comparison"]["geometric_mean_c_phys_over_a"]
    print(
        json.dumps(
            {
                "output": str(output),
                "paired": payload["comparison_is_paired"],
                "geometric_mean_c_phys_over_a": pooled,
                "decision": decision,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
