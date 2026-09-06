#!/usr/bin/env python3
"""Allen--Cahn reaction-only direct-map follow-up.

Implements ``docs/research/ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_PROTOCOL.md``.
The protocol was committed before this runner produced a result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import torch_load_compat  # noqa: E402
from models import ScalarMLP, TimeConditionedStencilMLP  # noqa: E402
from run_allen_cahn_work_matched_direct_map import (  # noqa: E402
    HIDDEN,
    HORIZONS,
    LOWER_BOUND,
    SELECTION_TAU,
    STENCIL_RADIUS,
    TEST_TAUS,
    UPPER_BOUND,
    AllenCahnDirectHeatMap,
    EulerAutonomousFlow,
    EulerQueryTimeFlow,
    architecture_kwargs,
    compose,
    cross_lag_defect,
    evaluate_rollouts,
    generate_locked_test,
    load_or_build_cache,
    parameter_count,
    physical_energy,
    ratio_table,
    sha256_file,
    tau_column,
    train_one,
)
from seed_utils import set_global_seed  # noqa: E402


class AllenCahnReactionOnlyDirectMap(nn.Module):
    """Exact heat plus exact reaction plus a learned residual, no interaction."""

    def __init__(self, *, n_grid: int, epsilon: float, length: float):
        super().__init__()
        self.N = int(n_grid)
        self.m = LOWER_BOUND
        self.M = UPPER_BOUND
        self.epsilon = float(epsilon)
        self.length = float(length)
        self.ode_steps = 1
        self.eps = 1e-6
        self.V_net = ScalarMLP(HIDDEN, beta=0.0, beta_floor=0.0)
        self.R_net = TimeConditionedStencilMLP(radius=STENCIL_RADIUS, hidden_dims=HIDDEN)
        spacing = self.length / self.N
        wavenumbers = 2.0 * torch.pi * torch.fft.fftfreq(self.N, d=spacing)
        self.register_buffer("laplacian_eigenvalues", -(wavenumbers ** 2))

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
        decay = torch.exp(duration * (self.epsilon ** 2) * self.laplacian_eigenvalues)
        transformed = torch.fft.fft(state.to(dtype=torch.float64), dim=-1)
        evolved = torch.fft.ifft(transformed * decay.to(dtype=transformed.dtype), dim=-1)
        return evolved.real.to(dtype=state.dtype)

    def residual(self, state: torch.Tensor, tau) -> torch.Tensor:
        pointwise = self.V_net(state.unsqueeze(-1)).squeeze(-1)
        return pointwise + self.R_net(state, tau)

    def increment(self, state: torch.Tensor, tau) -> torch.Tensor:
        reaction = state * (1.0 - state.square())
        return reaction + self.residual(state, tau)

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        linear = self.heat_semigroup(state, tau)
        step = tau_column(tau, linear)
        return self.project(linear + step * self.increment(linear, tau))


class AllenCahnPhysicsSplit(nn.Module):
    """Zero-parameter heat plus exact reaction. Diagnostic only."""

    def __init__(self, *, n_grid: int, epsilon: float, length: float):
        super().__init__()
        self.template = AllenCahnReactionOnlyDirectMap(
            n_grid=n_grid, epsilon=epsilon, length=length
        )
        for parameter in self.template.parameters():
            parameter.requires_grad_(False)

    def forward(self, state: torch.Tensor, tau) -> torch.Tensor:
        linear = self.template.heat_semigroup(state, tau)
        step = tau_column(tau, linear)
        reaction = linear * (1.0 - linear.square())
        return self.template.project(linear + step * reaction)


def load_frozen_flow(label: str, prior_root: Path, args):
    kwargs = architecture_kwargs(args.N)
    if label == "a_euler_autonomous":
        model = EulerAutonomousFlow(**kwargs, ode_steps=1)
    elif label == "b_euler_query_time":
        model = EulerQueryTimeFlow(**kwargs, ode_steps=1)
    elif label == "c_direct_heat_map":
        model = AllenCahnDirectHeatMap(**kwargs, epsilon=args.epsilon, length=args.L)
    else:
        raise ValueError(label)
    path = (
        prior_root
        / f"s{args.seed}"
        / "checkpoints"
        / label
        / f"{label}_best.pt"
    )
    checkpoint = torch_load_compat(path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()
    return model, {
        "label": label,
        "frozen": True,
        "parameter_count": parameter_count(model),
        "best_checkpoint": str(path),
        "best_checkpoint_sha256": sha256_file(path),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--prior-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * torch.pi)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=5)
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prior_root = Path(args.prior_root)
    data = load_or_build_cache(args, Path(args.data_cache))
    locked_path = prior_root / f"s{args.seed}" / "locked_test.pt"
    if locked_path.exists():
        locked = torch_load_compat(locked_path, map_location="cpu")
        locked_source = str(locked_path)
    else:
        locked = generate_locked_test(args)
        locked_source = "regenerated"
    torch.save(locked, output / "locked_test.pt")

    models = {}
    trained = {}
    for label in ("a_euler_autonomous", "b_euler_query_time", "c_direct_heat_map"):
        models[label], trained[label] = load_frozen_flow(label, prior_root, args)

    set_global_seed(args.seed, deterministic=True)
    reaction = AllenCahnReactionOnlyDirectMap(
        n_grid=args.N, epsilon=args.epsilon, length=args.L
    )
    trained["c_reaction_only"] = train_one(reaction, "c_reaction_only", data, args)
    models["c_reaction_only"] = reaction.to(args.device)

    physics = AllenCahnPhysicsSplit(
        n_grid=args.N, epsilon=args.epsilon, length=args.L
    ).to(args.device)
    models["c_physics_split"] = physics
    trained["c_physics_split"] = {
        "label": "c_physics_split",
        "frozen": True,
        "parameter_count": 0,
        "training_seconds": 0.0,
    }

    rollouts = {
        label: evaluate_rollouts(model, locked, args) for label, model in models.items()
    }
    sample = locked["u0"][:64].to(args.device)
    defects = {
        label: cross_lag_defect(model, sample, 0.075, 0.15)
        for label, model in models.items()
    }
    summary = {
        "experiment": "allen_cahn_reaction_only_direct_map",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "cache_sha256": sha256_file(Path(args.data_cache)),
        "locked_test_source": locked_source,
        "protocol": "docs/research/ALLEN_CAHN_REACTION_ONLY_DIRECT_MAP_PROTOCOL.md",
        "models": trained,
        "rollouts": rollouts,
        "deployed_budget_cross_lag": defects,
        "ratios": {
            "mse_a_prime_over_c_rxn": ratio_table(
                rollouts, "a_euler_autonomous", "c_reaction_only"
            ),
            "mse_old_c_over_c_rxn": ratio_table(
                rollouts, "c_direct_heat_map", "c_reaction_only"
            ),
            "mse_phys_over_c_rxn": ratio_table(
                rollouts, "c_physics_split", "c_reaction_only"
            ),
            "mse_a_prime_over_old_c": ratio_table(
                rollouts, "a_euler_autonomous", "c_direct_heat_map"
            ),
        },
        "learned_evaluations_per_call": {
            "a_euler_autonomous": 1,
            "b_euler_query_time": 1,
            "c_direct_heat_map": 1,
            "c_reaction_only": 1,
            "c_physics_split": 0,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
