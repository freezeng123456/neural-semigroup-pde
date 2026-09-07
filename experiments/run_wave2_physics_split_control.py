#!/usr/bin/env python3
"""Wave 2 physics-split direct-map control.

Implements ``docs/research/WAVE2_PHYSICS_SPLIT_CONTROL_PROTOCOL.md``.
Loads frozen A', B', and residual C; trains only the physics-split map.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from boundary_family_direct_map import (  # noqa: E402
    EXPECTED_PARAMETERS,
    MATCHED_NET_EVALUATIONS,
    BoundaryFamilyDirectMap,
    EulerBoundaryFamilyFlow,
    compose,
    direct_map_cross_lag,
    parameter_count,
)
from boundary_family_physics_split import build_physics_split_map  # noqa: E402
from boundary_family_semigroup import (  # noqa: E402
    BOUNDARY_FAMILIES,
    TRAINING_SEEDS,
    BoundaryFamilyFlow,
    validate_reference_cache,
    wave_config,
)
from experiment_artifacts import torch_load_compat  # noqa: E402
from run_wave2_direct_map_control import (  # noqa: E402
    evaluate_long_rollouts,
    evaluate_stress,
    mse_ratio,
    sha256_file,
    train_model,
)
from seed_utils import set_global_seed  # noqa: E402


FROZEN_LABELS = (
    "a_euler_autonomous",
    "b_euler_query_time",
    "c_direct_map",
)
PREVIOUS_ROOT_NAME = "20260907-wave2-direct-map-work-matched-3seed-h20-r1"


def _load_frozen_models(config, *, previous_cell: Path, device: torch.device):
    autonomous = EulerBoundaryFamilyFlow(
        BoundaryFamilyFlow(
            n_grid=config.n_grid,
            hidden_width=config.hidden_width,
            boundary_spec=config.boundary_spec,
            enforcement_mode="hard",
            temporal_mode="autonomous",
            query_time_scale=config.query_time_scale,
            rk4_steps=1,
        )
    )
    query = EulerBoundaryFamilyFlow(
        BoundaryFamilyFlow(
            n_grid=config.n_grid,
            hidden_width=config.hidden_width,
            boundary_spec=config.boundary_spec,
            enforcement_mode="hard",
            temporal_mode="query_time",
            query_time_scale=config.query_time_scale,
            rk4_steps=1,
        )
    )
    residual = BoundaryFamilyDirectMap(
        n_grid=config.n_grid,
        hidden_width=config.hidden_width,
        boundary_spec=config.boundary_spec,
        query_time_scale=config.query_time_scale,
    )
    paths = {
        "a_euler_autonomous": previous_cell / "checkpoints/a_euler_autonomous_best.pt",
        "b_euler_query_time": previous_cell / "checkpoints/b_euler_query_time_best.pt",
        "c_direct_map": previous_cell / "checkpoints/c_direct_map_best.pt",
    }
    autonomous.flow.load_state_dict(
        torch_load_compat(paths["a_euler_autonomous"], map_location="cpu")[
            "model_state_dict"
        ],
        strict=True,
    )
    query.flow.load_state_dict(
        torch_load_compat(paths["b_euler_query_time"], map_location="cpu")[
            "model_state_dict"
        ],
        strict=True,
    )
    residual.load_state_dict(
        torch_load_compat(paths["c_direct_map"], map_location="cpu")["model_state_dict"],
        strict=True,
    )
    models = {
        "a_euler_autonomous": autonomous.to(device).eval(),
        "b_euler_query_time": query.to(device).eval(),
        "c_direct_map": residual.to(device).eval(),
    }
    for model in models.values():
        if parameter_count(model) != EXPECTED_PARAMETERS:
            raise RuntimeError("frozen model parameter count mismatch")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    digests = {label: sha256_file(path) for label, path in paths.items()}
    return models, digests


def run_cell(args) -> dict[str, Any]:
    config = wave_config(args.boundary_family, smoke_only=bool(args.smoke))
    cache_path = Path(args.data_cache).expanduser().resolve()
    digest = sha256_file(cache_path)
    if args.expect_cache_sha256 and digest != args.expect_cache_sha256:
        raise RuntimeError(
            f"data cache digest mismatch: {digest} != {args.expect_cache_sha256}"
        )
    cache = torch_load_compat(cache_path, map_location="cpu")
    validate_reference_cache(cache, config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    previous_cell = Path(args.previous_cell).expanduser().resolve()
    frozen, frozen_digests = _load_frozen_models(
        config, previous_cell=previous_cell, device=device
    )

    set_global_seed(args.seed, deterministic=True)
    physics = build_physics_split_map(
        n_grid=config.n_grid,
        hidden_width=config.hidden_width,
        boundary_spec=config.boundary_spec,
        query_time_scale=config.query_time_scale,
        diffusivity=config.diffusivity,
    )
    if parameter_count(physics) != EXPECTED_PARAMETERS:
        raise RuntimeError("physics-split parameter count mismatch")
    physics, trained = train_model(
        physics,
        label="c_physics_split",
        cache=cache,
        config=config,
        args=args,
        seed=args.seed,
    )

    models = {**frozen, "c_physics_split": physics}
    rollouts = {
        label: evaluate_long_rollouts(model, cache, device)
        for label, model in models.items()
    }
    stress = {
        label: evaluate_stress(model, cache, config, device)
        for label, model in models.items()
    }
    sample = cache["test_u0"][: args.n_sample].to(device)
    deployed = {
        "c_direct_map": direct_map_cross_lag(frozen["c_direct_map"], sample),
        "c_physics_split": direct_map_cross_lag(physics, sample),
    }

    summary = {
        "experiment": "wave2_physics_split_control",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": "docs/research/WAVE2_PHYSICS_SPLIT_CONTROL_PROTOCOL.md",
        "boundary_family": args.boundary_family,
        "enforcement_mode": "hard",
        "seed": int(args.seed),
        "smoke_only": bool(args.smoke),
        "n_sample": int(args.n_sample),
        "data_cache_sha256": digest,
        "previous_cell": str(previous_cell),
        "frozen_checkpoint_sha256": frozen_digests,
        "parameter_count_each": EXPECTED_PARAMETERS,
        "net_evaluations_per_call": {
            label: MATCHED_NET_EVALUATIONS
            for label in (*FROZEN_LABELS, "c_physics_split")
        },
        "runtime": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
        },
        "models": {"c_physics_split": trained},
        "long_rollouts": rollouts,
        "boundary_stress": stress,
        "deployed_budget_cross_lag": deployed,
        "comparison": {
            "mse_a_prime_over_c_star": mse_ratio(
                rollouts["a_euler_autonomous"], rollouts["c_physics_split"]
            ),
            "mse_b_prime_over_c_star": mse_ratio(
                rollouts["b_euler_query_time"], rollouts["c_physics_split"]
            ),
            "mse_c_star_over_c_residual": mse_ratio(
                rollouts["c_physics_split"], rollouts["c_direct_map"]
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "boundary_family": args.boundary_family,
                "seed": args.seed,
                "mse_a_prime_over_c_star": summary["comparison"][
                    "mse_a_prime_over_c_star"
                ]["geometric_mean"],
                "mse_c_star_over_c_residual": summary["comparison"][
                    "mse_c_star_over_c_residual"
                ]["geometric_mean"],
            },
            indent=2,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--expect-cache-sha256", default="")
    parser.add_argument("--previous-cell", required=True)
    parser.add_argument("--boundary-family", choices=BOUNDARY_FAMILIES, required=True)
    parser.add_argument("--seed", type=int, choices=TRAINING_SEEDS, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-sample", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run_cell(args)


if __name__ == "__main__":
    main()
