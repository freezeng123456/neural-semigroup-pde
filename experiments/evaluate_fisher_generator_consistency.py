#!/usr/bin/env python3
"""Compare a baseline autonomous Fisher model with one generator-supervised model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from fisher_generator_metrics import generator_residual_metrics  # noqa: E402
from models import LatentSemigroupNet  # noqa: E402


TAUS = (0.075, 0.15)
HORIZONS = (1.2, 2.4, 4.8)
SNAPSHOT_TIMES = (0.0, 0.6, 1.2, 2.4, 4.8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_latent_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata = checkpoint["run_metadata"]
    if metadata["model"] != "latent":
        raise ValueError(f"expected latent checkpoint, got {metadata['model']!r}")
    model = LatentSemigroupNet(**metadata["model_config"]["kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), metadata


@torch.no_grad()
def rollout_metrics(
    model,
    initial_states,
    trajectories,
    *,
    tau: float,
    horizons,
    reference_dt: float,
    batch_size: int,
    device: torch.device,
):
    stride = int(round(tau / reference_dt))
    if not math.isclose(stride * reference_dt, tau, abs_tol=1e-12):
        raise ValueError("tau is not aligned to the reference grid")
    horizon_steps = {horizon: int(round(horizon / tau)) for horizon in horizons}
    if any(
        not math.isclose(steps * tau, horizon, abs_tol=1e-12)
        for horizon, steps in horizon_steps.items()
    ):
        raise ValueError("a horizon is not an integer multiple of tau")
    max_steps = max(horizon_steps.values())
    step_horizons = {steps: horizon for horizon, steps in horizon_steps.items()}
    references = torch.stack(
        [
            states[stride : max_steps * stride + 1 : stride]
            for _times, states in trajectories
        ]
    )
    per_sample_mse = {horizon: [] for horizon in horizons}
    per_sample_bound = {horizon: [] for horizon in horizons}
    for start in range(0, len(initial_states), batch_size):
        prediction = initial_states[start : start + batch_size].to(device)
        target = references[start : start + batch_size].to(device)
        mse = torch.zeros(len(prediction), dtype=torch.float64, device=device)
        bound = torch.zeros_like(mse)
        for step in range(max_steps):
            prediction = model(prediction, tau)
            mse += (prediction - target[:, step]).square().mean(dim=1)
            bound += (
                torch.relu(-prediction).mean(dim=1)
                + torch.relu(prediction - 1.0).mean(dim=1)
            )
            completed_steps = step + 1
            if completed_steps in step_horizons:
                horizon = step_horizons[completed_steps]
                per_sample_mse[horizon].append((mse / completed_steps).cpu())
                per_sample_bound[horizon].append((bound / completed_steps).cpu())
    return {
        horizon: {
            "rollout_mse_mean": float(
                torch.cat(per_sample_mse[horizon]).mean().item()
            ),
            "bound_viol_mean": float(
                torch.cat(per_sample_bound[horizon]).mean().item()
            ),
            "n_test": int(len(initial_states)),
            "rollout_steps": horizon_steps[horizon],
        }
        for horizon in horizons
    }


def snapshot_states(trajectories, reference_dt: float):
    indices = [int(round(time / reference_dt)) for time in SNAPSHOT_TIMES]
    return torch.cat(
        [torch.stack([states[index] for _times, states in trajectories]) for index in indices]
    )


def evaluate_model(model, initial_states, trajectories, cache_config, args, device):
    rollouts = {}
    for tau in TAUS:
        tau_metrics = rollout_metrics(
            model,
            initial_states,
            trajectories,
            tau=tau,
            horizons=HORIZONS,
            reference_dt=float(cache_config["reference_dt"]),
            batch_size=args.batch_size,
            device=device,
        )
        for horizon, metrics in tau_metrics.items():
            key = f"tau={tau}:horizon={horizon}"
            rollouts[key] = metrics
    states = snapshot_states(trajectories, float(cache_config["reference_dt"]))
    generator = generator_residual_metrics(
        model,
        states.to(device),
        conditioning_time=TAUS[0],
        length=float(cache_config["L"]),
        diffusivity=float(cache_config["nu"]),
        reaction_rate=float(cache_config["reaction_rate"]),
        batch_size=args.batch_size,
    )
    return {"rollouts": rollouts, "generator": generator}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--regularized-checkpoint", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    cache = torch.load(args.test_cache, map_location="cpu", weights_only=False)
    initial_states = cache["test_u0"][: args.n_test]
    trajectories = cache["test_trajs"][: args.n_test]
    cache_config = cache["locked_test_config"]

    baseline, baseline_metadata = load_latent_checkpoint(
        args.baseline_checkpoint, device
    )
    regularized, regularized_metadata = load_latent_checkpoint(
        args.regularized_checkpoint, device
    )
    baseline_result = evaluate_model(
        baseline, initial_states, trajectories, cache_config, args, device
    )
    regularized_result = evaluate_model(
        regularized, initial_states, trajectories, cache_config, args, device
    )

    mse_ratios = {
        key: regularized_result["rollouts"][key]["rollout_mse_mean"]
        / baseline_result["rollouts"][key]["rollout_mse_mean"]
        for key in baseline_result["rollouts"]
    }
    result = {
        "experiment": "fisher_generator_consistency_pair",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": args.seed,
        "n_test": args.n_test,
        "taus": list(TAUS),
        "horizons": list(HORIZONS),
        "input_sha256": {
            "baseline_checkpoint": sha256_file(args.baseline_checkpoint),
            "regularized_checkpoint": sha256_file(args.regularized_checkpoint),
            "test_cache": sha256_file(args.test_cache),
        },
        "training": {
            "baseline_alpha_generator": baseline_metadata.get(
                "auxiliary_loss_weights", {}
            ).get("alpha_generator", 0.0),
            "regularized_alpha_generator": regularized_metadata.get(
                "auxiliary_loss_weights", {}
            ).get("alpha_generator"),
        },
        "baseline": baseline_result,
        "regularized": regularized_result,
        "comparison": {
            "generator_residual_ratio": regularized_result["generator"][
                "rms_relative_residual"
            ]
            / baseline_result["generator"]["rms_relative_residual"],
            "rollout_mse_ratios": mse_ratios,
            "mse_ratio_geometric_mean": math.exp(
                sum(math.log(value) for value in mse_ratios.values())
                / len(mse_ratios)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
