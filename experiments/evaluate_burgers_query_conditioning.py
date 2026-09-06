#!/usr/bin/env python3
"""Checkpoint-only decomposition of the Burgers query-time composition defect.

Implements the three paths frozen in
``docs/research/BURGERS_QUERY_CONDITIONING_ATTRIBUTION_PROTOCOL.md``.  The
evaluator trains nothing, selects nothing, and writes no cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import torch_load_compat  # noqa: E402
from run_burgers_semigroup_screen import (  # noqa: E402
    HORIZONS,
    TEST_TAUS,
    PeriodicBurgersFluxFlow,
)

FLOOR_COLLAPSE_FACTOR = 10.0
FLOOR_STRUCTURAL_FACTOR = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms_defect(left: torch.Tensor, right: torch.Tensor) -> float:
    """Per-sample RMS difference averaged over samples.

    This is the convention used by the screen's composition endpoint, so path
    1 below reproduces the committed screen values.
    """

    return float(
        torch.sqrt((left - right).square().mean(dim=1)).mean().item()
    )


def build_model(
    checkpoint: Path,
    *,
    query_conditioned: bool,
    config: dict,
    device: str,
) -> PeriodicBurgersFluxFlow:
    model = PeriodicBurgersFluxFlow(
        n_grid=int(config["N"]),
        length=float(config["L"]),
        viscosity=float(config["nu"]),
        hidden=int(config["hidden"]),
        ode_steps=int(config["ode_steps"]),
        query_conditioned=query_conditioned,
    )
    payload = torch_load_compat(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def compose(
    model: PeriodicBurgersFluxFlow,
    state: torch.Tensor,
    *,
    tau: float,
    depth: int,
    substeps: int,
) -> torch.Tensor:
    for _ in range(depth):
        state = model.integrate(
            state, tau, substeps=substeps, conditioning_tau=tau
        )
    return state


@torch.no_grad()
def measure_model(
    model: PeriodicBurgersFluxFlow,
    sample: torch.Tensor,
) -> dict[str, object]:
    steps = int(model.ode_steps)
    frozen_semantics = {}
    matched_conditioning = {}
    for tau in TEST_TAUS:
        for horizon in HORIZONS:
            depth = int(round(horizon / tau))
            composed = compose(
                model, sample, tau=tau, depth=depth, substeps=steps
            )
            direct_horizon = model.integrate(
                sample,
                horizon,
                substeps=depth * steps,
                conditioning_tau=horizon,
            )
            direct_matched = model.integrate(
                sample,
                horizon,
                substeps=depth * steps,
                conditioning_tau=tau,
            )
            key = f"tau={tau}:horizon={horizon}"
            work = 4 * depth * steps
            frozen_semantics[key] = {
                "equal_work_rms_defect_mean": rms_defect(
                    direct_horizon, composed
                ),
                "composition_depth": depth,
                "rhs_evaluations_each_path": work,
                "direct_path_control_input": horizon / max(TEST_TAUS),
            }
            matched_conditioning[key] = {
                "equal_work_rms_defect_mean": rms_defect(
                    direct_matched, composed
                ),
                "composition_depth": depth,
                "rhs_evaluations_each_path": work,
                "direct_path_control_input": tau / max(TEST_TAUS),
            }

    fine_tau, coarse_tau = min(TEST_TAUS), max(TEST_TAUS)
    cross_lag = {}
    for horizon in HORIZONS:
        fine_depth = int(round(horizon / fine_tau))
        coarse_depth = int(round(horizon / coarse_tau))
        work = 4 * fine_depth * steps
        coarse_substeps, remainder = divmod(work, 4 * coarse_depth)
        if remainder:
            raise ValueError("cross-lag paths cannot be given equal RHS work")
        fine = compose(
            model, sample, tau=fine_tau, depth=fine_depth, substeps=steps
        )
        coarse = compose(
            model,
            sample,
            tau=coarse_tau,
            depth=coarse_depth,
            substeps=coarse_substeps,
        )
        cross_lag[f"horizon={horizon}"] = {
            "equal_work_rms_defect_mean": rms_defect(fine, coarse),
            "fine_lag": fine_tau,
            "coarse_lag": coarse_tau,
            "fine_depth": fine_depth,
            "coarse_depth": coarse_depth,
            "coarse_substeps_per_call": coarse_substeps,
            "rhs_evaluations_each_path": work,
            "composed_state_rms": float(
                torch.sqrt(fine.square().mean(dim=1)).mean().item()
            ),
        }
    return {
        "path1_frozen_semantics": frozen_semantics,
        "path2_matched_conditioning": matched_conditioning,
        "path3_in_range_cross_lag": cross_lag,
    }


def classify(value: float, floor: float) -> str:
    if floor <= 0:
        raise ValueError("integrator floor must be positive")
    if value <= FLOOR_COLLAPSE_FACTOR * floor:
        return "collapsed"
    if value >= FLOOR_STRUCTURAL_FACTOR * floor:
        return "structural"
    return "intermediate"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autonomous-checkpoint", type=Path, required=True)
    parser.add_argument("--query-checkpoint", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--expect-cache-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-sample", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--ode-steps", type=int, default=12)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cache_digest = sha256_file(args.data_cache)
    if cache_digest != args.expect_cache_sha256:
        raise RuntimeError(
            "data cache digest does not match the frozen screen cache: "
            f"{cache_digest} != {args.expect_cache_sha256}"
        )
    cache = torch_load_compat(args.data_cache, map_location="cpu")
    config = dict(cache["config"])
    config["hidden"] = args.hidden
    config["ode_steps"] = args.ode_steps
    sample = cache["val_u0"][: args.n_sample].to(args.device)

    models = {
        "a_autonomous": build_model(
            args.autonomous_checkpoint,
            query_conditioned=False,
            config=config,
            device=args.device,
        ),
        "b_query_time": build_model(
            args.query_checkpoint,
            query_conditioned=True,
            config=config,
            device=args.device,
        ),
    }
    measurements = {
        label: measure_model(model, sample) for label, model in models.items()
    }

    floors = {
        key: row["equal_work_rms_defect_mean"]
        for key, row in measurements["a_autonomous"][
            "path2_matched_conditioning"
        ].items()
    }
    classification = {
        "path1_frozen_semantics": {
            key: classify(
                measurements["b_query_time"]["path1_frozen_semantics"][key][
                    "equal_work_rms_defect_mean"
                ],
                floors[key],
            )
            for key in floors
        },
        "path2_matched_conditioning": {
            key: classify(
                measurements["b_query_time"]["path2_matched_conditioning"][key][
                    "equal_work_rms_defect_mean"
                ],
                floors[key],
            )
            for key in floors
        },
    }
    cross_floors = {
        key: row["equal_work_rms_defect_mean"]
        for key, row in measurements["a_autonomous"][
            "path3_in_range_cross_lag"
        ].items()
    }
    classification["path3_in_range_cross_lag"] = {
        key: classify(
            measurements["b_query_time"]["path3_in_range_cross_lag"][key][
                "equal_work_rms_defect_mean"
            ],
            max(
                floors[f"tau={min(TEST_TAUS)}:{key}"],
                floors[f"tau={max(TEST_TAUS)}:{key}"],
            ),
        )
        for key in cross_floors
    }

    result = {
        "experiment": "burgers_query_conditioning_attribution",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "protocol": (
            "docs/research/"
            "BURGERS_QUERY_CONDITIONING_ATTRIBUTION_PROTOCOL.md"
        ),
        "seed": int(args.seed),
        "n_sample": int(args.n_sample),
        "device": str(args.device),
        "collapse_factor": FLOOR_COLLAPSE_FACTOR,
        "structural_factor": FLOOR_STRUCTURAL_FACTOR,
        "inputs": {
            "data_cache_sha256": cache_digest,
            "autonomous_checkpoint_sha256": sha256_file(
                args.autonomous_checkpoint
            ),
            "query_checkpoint_sha256": sha256_file(args.query_checkpoint),
        },
        "measurements": measurements,
        "autonomous_integrator_floor": floors,
        "query_time_classification": classification,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()
