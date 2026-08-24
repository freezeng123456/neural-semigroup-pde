#!/usr/bin/env python3
"""Benchmark validation and formal inference on a fixed trained checkpoint."""

import argparse
import json
import os
import socket
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation-root", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed_call(function, repeats, device):
    durations = []
    value = None
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        value = function()
        synchronize(device)
        durations.append(time.perf_counter() - start)
    return value, durations


def fisher_kpp_discrete_energy(u, *, domain_length, nu, reaction_rate):
    dx = float(domain_length) / u.shape[-1]
    grad = (torch.roll(u, shifts=-1, dims=-1) - u) / dx
    density = 0.5 * float(nu) * grad.square()
    density -= float(reaction_rate) * (0.5 * u.square() - u.pow(3) / 3.0)
    return dx * density.sum(dim=-1)


def main():
    args = parse_args()
    if args.n_samples <= 0 or args.rollout_steps <= 0 or args.repeats <= 0:
        raise ValueError("sample, rollout, and repeat counts must be positive")

    experiments_dir = os.path.join(
        os.path.abspath(args.implementation_root), "experiments"
    )
    sys.path.insert(0, experiments_dir)
    from evaluate import evaluate_full
    from models import LatentSemigroupNet
    from seed_utils import set_global_seed
    from training import evaluate_on_trajectories

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("run_metadata") or {}
    set_global_seed(int(metadata.get("seed", 42)), deterministic=True)
    model = LatentSemigroupNet(
        N=int(metadata.get("N", 64)),
        hidden_V=metadata.get("hidden_V", [64, 64]),
        hidden_K=metadata.get("hidden_K", [64, 64]),
        stencil_radius=int(metadata.get("stencil_radius", 3)),
        beta_V=float(metadata.get("beta_V", 0.0)),
        beta_V_floor=float(metadata.get("beta_V_floor", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(args.device).eval()

    data = torch.load(args.data, map_location="cpu", weights_only=False)
    n_samples = min(args.n_samples, len(data["val_u0"]))
    val_u0 = data["val_u0"][:n_samples]
    val_trajs = data["val_trajs"][:n_samples]
    tau = float(metadata.get("tau", 0.1))
    reference_dt = float(metadata.get("dt_pde", 0.005))
    with torch.no_grad():
        model(val_u0[: min(2, n_samples)].to(args.device), tau)
    synchronize(args.device)

    validation_metrics, validation_seconds = timed_call(
        lambda: evaluate_on_trajectories(
            model,
            val_u0,
            val_trajs,
            tau=tau,
            rollout_steps=args.rollout_steps,
            device=args.device,
            reference_dt=reference_dt,
        ),
        args.repeats,
        args.device,
    )
    physical_energy = lambda u: fisher_kpp_discrete_energy(
        u,
        domain_length=float(metadata.get("L", 10.0)),
        nu=float(metadata.get("nu", 0.1)),
        reaction_rate=float(metadata.get("r", 1.0)),
    )
    (formal_metrics, _details), formal_seconds = timed_call(
        lambda: evaluate_full(
            model,
            val_u0,
            val_trajs,
            tau=tau,
            rollout_steps=args.rollout_steps,
            device=args.device,
            reference_dt=reference_dt,
            physical_energy_fn=physical_energy,
            ode_substep_counts=(15, 30, 60),
            collect_latent_diagnostics=True,
        ),
        args.repeats,
        args.device,
    )
    result = {
        "implementation_root": os.path.abspath(args.implementation_root),
        "hostname": socket.gethostname(),
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "n_samples": n_samples,
        "rollout_steps": args.rollout_steps,
        "repeats": args.repeats,
        "validation_seconds": validation_seconds,
        "formal_evaluation_seconds": formal_seconds,
        "validation_metrics": validation_metrics,
        "formal_metrics": formal_metrics,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
