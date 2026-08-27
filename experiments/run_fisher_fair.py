#!/usr/bin/env python3
"""Run parameter-matched fixed- or variable-time Fisher--KPP experiments."""

import argparse
import json
import math
import os
import socket
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import evaluate_full
from models import (
    LatentSemigroupNet,
    QueryTimeConditionedLatentFlow,
    TimeConditionedFNO,
    TimeConditionedResNet,
)
from pde_solver import (
    FisherKPPSolver,
    generate_initial_conditions,
    generate_variable_tau_training_data,
)
from seed_utils import set_global_seed
from training import train_model


MODEL_NAMES = ("latent", "latent_query_time", "resnet", "fno")


def parse_float_list(value):
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated float list") from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError("times must be finite and strictly positive")
    return values


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("fixed", "variable"), required=True)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=1.0)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--fixed-tau", type=float, default=0.1)
    parser.add_argument(
        "--train-taus",
        type=parse_float_list,
        default=(0.025, 0.05, 0.1, 0.2),
    )
    parser.add_argument(
        "--eval-taus",
        type=parse_float_list,
        default=(0.025, 0.05, 0.075, 0.1, 0.15, 0.2),
    )
    parser.add_argument("--eval-horizon", type=float, default=1.2)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--beta-v-floor", type=float, default=0.1)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def rollout_steps_for_horizon(horizon, tau):
    ratio = float(horizon) / float(tau)
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(ratio, steps, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(f"eval_horizon={horizon} is not an integer multiple of tau={tau}")
    return steps


def data_config(args):
    return {
        "regime": args.regime,
        "data_seed": args.data_seed,
        "N": args.N,
        "L": args.L,
        "nu": args.nu,
        "r": args.reaction_rate,
        "reference_dt": args.reference_dt,
        "fixed_tau": args.fixed_tau,
        "train_taus": list(args.train_taus),
        "eval_horizon": args.eval_horizon,
        "n_train": args.n_train,
        "n_val": args.n_val,
    }


def generate_data(args):
    set_global_seed(args.data_seed, deterministic=args.deterministic)
    if args.regime == "fixed":
        train_u0 = generate_initial_conditions(args.N, args.n_train, args.L)
        solver = FisherKPPSolver(
            N=args.N,
            L=args.L,
            nu=args.nu,
            r=args.reaction_rate,
            dt=args.reference_dt,
        )
        train_ut = solver.solve_batch_final(train_u0, args.fixed_tau)
        train_tau = None
    else:
        train_u0, train_tau, train_ut = generate_variable_tau_training_data(
            N=args.N,
            n_train=args.n_train,
            taus=args.train_taus,
            L=args.L,
            nu=args.nu,
            r=args.reaction_rate,
            dt=args.reference_dt,
        )

    val_u0 = generate_initial_conditions(args.N, args.n_val, args.L)
    solver = FisherKPPSolver(
        N=args.N,
        L=args.L,
        nu=args.nu,
        r=args.reaction_rate,
        dt=args.reference_dt,
    )
    val_trajs = []
    for initial in val_u0:
        val_trajs.append(
            solver.solve(initial, args.eval_horizon, save_every=1)
        )
    return {
        "train_u0": train_u0,
        "train_tau": train_tau,
        "train_ut": train_ut,
        "val_u0": val_u0,
        "val_trajs": val_trajs,
        "data_generation_config": data_config(args),
    }


def load_or_generate_data(args):
    cache_path = os.path.abspath(args.data_cache)
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if data.get("data_generation_config") != data_config(args):
            raise ValueError("data cache configuration does not match this run")
        return data
    data = generate_data(args)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(data, cache_path)
    return data


def build_model(name, args):
    if name == "latent":
        return LatentSemigroupNet(
            N=args.N,
            hidden_V=[64, 64],
            hidden_K=[64, 64],
            stencil_radius=3,
            interaction_radius=2,
            beta_V=0.0,
            beta_V_floor=args.beta_v_floor,
        )
    if name == "latent_query_time":
        return QueryTimeConditionedLatentFlow(
            N=args.N,
            hidden_V=[64, 64],
            hidden_K=[64, 64],
            stencil_radius=3,
            interaction_radius=2,
            beta_V=0.0,
            beta_V_floor=args.beta_v_floor,
        )
    if name == "resnet":
        return TimeConditionedResNet(N=args.N, width=18, blocks=3)
    if name == "fno":
        return TimeConditionedFNO(N=args.N, width=16, modes=8, layers=4)
    raise ValueError(f"unsupported model: {name}")


def temporal_structure_metadata(name):
    """Describe whether a model is a cross-query-time semigroup candidate."""
    if name == "latent":
        return {
            "kind": "autonomous_time_homogeneous_gradient_flow",
            "continuous_cross_tau_semigroup": True,
            "query_time_conditioning": False,
        }
    if name == "latent_query_time":
        return {
            "kind": "query_time_conditioned_gradient_flow",
            "continuous_cross_tau_semigroup": False,
            "query_time_conditioning": "mobility_stencil",
            "energy_dissipation": "holds for each fixed positive query time",
        }
    return {
        "kind": "direct_time_conditioned_operator",
        "continuous_cross_tau_semigroup": False,
        "query_time_conditioning": True,
    }


def fisher_energy(args):
    def energy(u):
        dx = float(args.L) / u.shape[-1]
        grad = (torch.roll(u, shifts=-1, dims=-1) - u) / dx
        density = 0.5 * float(args.nu) * grad.square()
        density -= float(args.reaction_rate) * (
            0.5 * u.square() - u.pow(3) / 3.0
        )
        return dx * density.sum(dim=-1)

    return energy


def run_model(name, args, data, device):
    set_global_seed(args.seed, deterministic=args.deterministic)
    model = build_model(name, args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model_dir = os.path.join(args.output_dir, name)
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    selection_tau = args.fixed_tau
    run_metadata = {
        **data_config(args),
        "training_seed": args.seed,
        "model": name,
        "parameter_count": parameter_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "validation_interval": args.validation_interval,
        "architecture_only": True,
        "temporal_structure": temporal_structure_metadata(name),
        "auxiliary_loss_weights": {
            "alpha_rollout": 0.0,
            "alpha_energy": 0.0,
            "alpha_bound": 0.0,
        },
    }

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    history = train_model(
        model,
        data["train_u0"],
        data["train_ut"],
        data["val_u0"],
        data["val_trajs"],
        train_tau=data["train_tau"],
        tau=selection_tau,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_rollout=0.0,
        alpha_energy=0.0,
        # This screen attributes temporal structure, so no model receives an
        # extra training loss that its matched competitor does not receive.
        alpha_bound=0.0,
        weight_decay=1e-5,
        checkpoint_dir=checkpoint_dir,
        model_name=name,
        device=device,
        resume_from=None if args.no_resume else os.path.join(checkpoint_dir, f"{name}_best.pt"),
        reference_dt=args.reference_dt,
        run_metadata=run_metadata,
        validation_interval=args.validation_interval,
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - start
    training_peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if str(device).startswith("cuda")
        else None
    )

    best_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    eval_taus = (args.fixed_tau,) if args.regime == "fixed" else args.eval_taus
    evaluations = {}
    for eval_tau in eval_taus:
        rollout_steps = rollout_steps_for_horizon(args.eval_horizon, eval_tau)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        evaluation_start = time.perf_counter()
        metrics, details = evaluate_full(
            model,
            data["val_u0"],
            data["val_trajs"],
            tau=eval_tau,
            rollout_steps=rollout_steps,
            device=device,
            model_name=name,
            reference_dt=args.reference_dt,
            physical_energy_fn=fisher_energy(args),
            collect_latent_diagnostics=name in ("latent", "latent_query_time"),
            equal_work_base_ode_steps=(
                30 if name in ("latent", "latent_query_time") else None
            ),
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
        evaluation_seconds = time.perf_counter() - evaluation_start
        evaluation_peak_memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if str(device).startswith("cuda")
            else None
        )
        evaluations[str(eval_tau)] = {
            "metrics": metrics,
            "details": details,
            "evaluation_seconds": evaluation_seconds,
            "peak_memory_bytes": evaluation_peak_memory_bytes,
        }

    result = {
        "model": name,
        "parameter_count": parameter_count,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_seconds": training_seconds,
        "training_peak_memory_bytes": training_peak_memory_bytes,
        "training_examples_per_second": (
            args.epochs * args.n_train / training_seconds
        ),
        "optimizer_updates": args.epochs * math.ceil(args.n_train / args.batch_size),
        "examples_seen": args.epochs * args.n_train,
        "temporal_structure": temporal_structure_metadata(name),
        "history": history,
        "evaluations": evaluations,
    }
    with open(os.path.join(model_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_train <= 0 or args.n_val <= 0 or args.epochs <= 0:
        raise ValueError("sample and epoch counts must be positive")
    if args.validation_interval <= 0:
        raise ValueError("validation interval must be positive")
    for eval_tau in ((args.fixed_tau,) if args.regime == "fixed" else args.eval_taus):
        rollout_steps_for_horizon(args.eval_horizon, eval_tau)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    data = load_or_generate_data(args)
    results = {}
    for name in dict.fromkeys(args.models):
        results[name] = run_model(name, args, data, device)
    summary = {
        "schema_version": 1,
        "experiment": "fisher_parameter_matched_fair",
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if str(device).startswith("cuda") else None,
        "torch_version": torch.__version__,
        "config": vars(args),
        "data_generation_config": data["data_generation_config"],
        "results": results,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print(json.dumps({
        "output_dir": os.path.abspath(args.output_dir),
        "models": {name: result["parameter_count"] for name, result in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
