#!/usr/bin/env python3
"""Locate where Fisher generator matching stops transferring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
import sys

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate_fisher_generator_consistency import (  # noqa: E402
    load_latent_checkpoint,
    sha256_file,
)
from experiment_artifacts import torch_load_compat  # noqa: E402
from fisher_generator_metrics import generator_residual_metrics  # noqa: E402


SNAPSHOT_TIMES = (0.0, 0.6, 1.2, 2.4, 4.8)


def periodic_gradient_rms(states: torch.Tensor, length: float) -> torch.Tensor:
    dx = float(length) / states.shape[1]
    gradient = (torch.roll(states, shifts=-1, dims=1) - states) / dx
    return gradient.square().mean(dim=1).sqrt()


def metric_pair(baseline, regularized, states, args, device):
    states = states.to(device)
    common = {
        "conditioning_time": 0.075,
        "length": args.length,
        "diffusivity": args.nu,
        "reaction_rate": args.reaction_rate,
        "batch_size": args.batch_size,
    }
    base = generator_residual_metrics(baseline, states, **common)
    reg = generator_residual_metrics(regularized, states, **common)
    return {
        "state_count": int(states.shape[0]),
        "baseline": base,
        "regularized": reg,
        "ratio": {
            "raw_rms_residual": reg["rms_residual_l2"] / base["rms_residual_l2"],
            "relative_rms_residual": reg["rms_relative_residual"]
            / base["rms_relative_residual"],
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--regularized-checkpoint", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-states", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    baseline, _ = load_latent_checkpoint(args.baseline_checkpoint, device)
    regularized, _ = load_latent_checkpoint(args.regularized_checkpoint, device)
    train = torch_load_compat(args.training_cache, map_location="cpu")
    test = torch_load_compat(args.test_cache, map_location="cpu")
    sets = {
        "train_initial": train["train_u0"][: args.n_states],
        "train_one_step_target": train["train_ut"][: args.n_states],
        "validation_initial": train["val_u0"][: args.n_states],
    }
    reference_dt = float(test["locked_test_config"]["reference_dt"])
    trajectories = test["test_trajs"][: args.n_states]
    for time in SNAPSHOT_TIMES:
        index = int(round(time / reference_dt))
        sets[f"locked_time={time}"] = torch.stack(
            [states[index] for _times, states in trajectories]
        )

    distributions = {}
    for name, states in sets.items():
        entry = metric_pair(baseline, regularized, states, args, device)
        gradient = periodic_gradient_rms(states, args.length)
        ordered = torch.argsort(gradient)
        entry["gradient_rms"] = {
            "mean": float(gradient.mean().item()),
            "min": float(gradient.min().item()),
            "max": float(gradient.max().item()),
        }
        entry["gradient_tertiles"] = {
            label: metric_pair(
                baseline, regularized, states[indices], args, device
            )
            for label, indices in zip(
                ("low", "middle", "high"), torch.tensor_split(ordered, 3)
            )
        }
        distributions[name] = entry

    result = {
        "experiment": "fisher_generator_distribution_diagnostic",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": args.seed,
        "input_sha256": {
            "baseline_checkpoint": sha256_file(args.baseline_checkpoint),
            "regularized_checkpoint": sha256_file(args.regularized_checkpoint),
            "training_cache": sha256_file(args.training_cache),
            "test_cache": sha256_file(args.test_cache),
        },
        "distributions": distributions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: value["ratio"] for name, value in distributions.items()}, indent=2))


if __name__ == "__main__":
    main()
