#!/usr/bin/env python3
"""Run the no-retraining Phase-0 numerical audit on an existing checkpoint.

The audit keeps the production fixed-``ode_steps`` defect separate from an
equal-work base-step defect, reports learned/physical energy increments, and
evaluates an RK4 ``ode_steps`` sweep.  It only loads a checkpoint and data;
it never enters the training loop.

Example (Fisher--KPP checkpoint)::

    python experiments/phase0_audit.py \
        --data experiments/checkpoints/data.pt \
        --checkpoint experiments/checkpoints/latent_best.pt \
        --physical-energy fisher-kpp \
        --output experiments/results/phase0_audit.json \
        --device cuda

For old checkpoints without embedded run metadata, model architecture values
can be supplied explicitly with the corresponding command-line options.
"""

import argparse
import json
import os
import socket
import sys
import time

import torch

try:
    from .evaluate import DEFAULT_PHASE0_ODE_STEPS
except ImportError:  # direct ``python experiments/phase0_audit.py`` execution
    from evaluate import DEFAULT_PHASE0_ODE_STEPS


def parse_ode_steps(value):
    """Parse a comma-separated positive integer RK4 sweep."""
    try:
        values = [item.strip() for item in str(value).split(",") if item.strip()]
        if not values:
            raise ValueError
        return tuple(int(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "ode steps must be a comma-separated list of positive integers"
        ) from exc


def parse_hidden_dims(value):
    """Parse a comma-separated MLP hidden-dimension list."""
    try:
        values = [item.strip() for item in str(value).split(",") if item.strip()]
        if not values:
            raise ValueError
        dims = tuple(int(item) for item in values)
        if any(dim <= 0 for dim in dims):
            raise ValueError
        return list(dims)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "hidden dimensions must be a comma-separated list of positive integers"
        ) from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Phase-0 numerical semigroup/RK4 audit; no retraining"
    )
    parser.add_argument("--data", required=True, help="checkpoint data .pt file")
    parser.add_argument(
        "--checkpoint", required=True, help="trained model checkpoint .pt file"
    )
    parser.add_argument("--output", required=True, help="JSON audit output path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--sweep-samples",
        type=int,
        default=1,
        help="number of initial states for the RK4 sweep (default: 1)",
    )
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--reference-dt", type=float, default=None)
    parser.add_argument(
        "--ode-steps",
        type=parse_ode_steps,
        default=DEFAULT_PHASE0_ODE_STEPS,
        help="comma-separated sweep; default: 5,10,15,30,60,120",
    )
    parser.add_argument(
        "--equal-work-base-ode-steps",
        type=int,
        default=None,
        help="base RK4 steps per tau; defaults to the checkpoint model value",
    )
    parser.add_argument(
        "--model-kind",
        choices=("latent", "latent-bounded"),
        default="latent",
        help="latent model class used by the checkpoint",
    )
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--hidden-v", type=parse_hidden_dims, default=None)
    parser.add_argument("--hidden-k", type=parse_hidden_dims, default=None)
    parser.add_argument("--stencil-radius", type=int, default=None)
    parser.add_argument("--interaction-radius", type=int, default=None)
    parser.add_argument("--beta-v", type=float, default=None)
    parser.add_argument("--beta-v-floor", type=float, default=None)
    parser.add_argument(
        "--lower-bound",
        type=float,
        default=None,
        help="state lower bound; defaults to checkpoint metadata, then 0",
    )
    parser.add_argument(
        "--upper-bound",
        type=float,
        default=None,
        help="state upper bound; defaults to checkpoint metadata, then 1",
    )
    parser.add_argument(
        "--physical-energy",
        choices=("none", "fisher-kpp", "allen-cahn", "burgers-l2"),
        default="none",
    )
    parser.add_argument("--domain-length", type=float, default=None)
    parser.add_argument("--nu", type=float, default=None)
    parser.add_argument("--reaction-rate", type=float, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    return parser


def _import_experiment_modules():
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    from evaluate import evaluate_full, evaluate_ode_steps_sweep, normalize_ode_steps
    from models import LatentSemigroupNet, LatentSemigroupNetBounded

    return evaluate_full, evaluate_ode_steps_sweep, normalize_ode_steps, (
        LatentSemigroupNet,
        LatentSemigroupNetBounded,
    )


def _infer_n_from_state_dict(state_dict):
    embedding = state_dict.get("emb")
    if embedding is None:
        raise ValueError("cannot infer N: checkpoint state_dict has no emb tensor")
    return int(embedding.shape[0])


def _metadata_or(metadata, args_value, key, default):
    if args_value is not None:
        return args_value
    return metadata.get(key, default)


def infer_reference_dt(val_trajs):
    """Infer the stored reference spacing for legacy data without metadata."""
    if not val_trajs:
        return None
    timestamps = val_trajs[0][0]
    if len(timestamps) < 2:
        return None
    reference_dt = float(timestamps[1] - timestamps[0])
    if not torch.isfinite(torch.as_tensor(reference_dt)) or reference_dt <= 0:
        raise ValueError("cannot infer a positive reference_dt from validation timestamps")
    return reference_dt


def build_model(args, checkpoint, model_classes):
    """Instantiate the checkpoint architecture without changing its weights."""
    latent_class, bounded_class = model_classes
    metadata = checkpoint.get("run_metadata") or {}
    state_dict = checkpoint["model_state_dict"]
    n = int(_metadata_or(metadata, args.N, "N", _infer_n_from_state_dict(state_dict)))
    hidden_v = _metadata_or(metadata, args.hidden_v, "hidden_V", [64, 64])
    hidden_k = _metadata_or(metadata, args.hidden_k, "hidden_K", [64, 64])
    stencil_radius = int(
        _metadata_or(metadata, args.stencil_radius, "stencil_radius", 3)
    )
    interaction_radius = int(
        _metadata_or(metadata, args.interaction_radius, "interaction_radius", 2)
    )
    beta_v = float(_metadata_or(metadata, args.beta_v, "beta_V", 0.0))
    beta_v_floor = float(
        _metadata_or(metadata, args.beta_v_floor, "beta_V_floor", 0.0)
    )
    common = {
        "N": n,
        "hidden_V": hidden_v,
        "hidden_K": hidden_k,
        "stencil_radius": stencil_radius,
        "beta_V": beta_v,
        "beta_V_floor": beta_v_floor,
        "interaction_radius": interaction_radius,
    }
    if args.model_kind == "latent-bounded":
        model = bounded_class(
            m=args.lower_bound,
            M=args.upper_bound,
            **common,
        )
    else:
        model = latent_class(**common)
    model.load_state_dict(state_dict, strict=True)
    return model, {
        "model_kind": args.model_kind,
        "N": n,
        "hidden_V": list(hidden_v),
        "hidden_K": list(hidden_k),
        "stencil_radius": stencil_radius,
        "interaction_radius": interaction_radius,
        "beta_V": beta_v,
        "beta_V_floor": beta_v_floor,
        "lower_bound": args.lower_bound,
        "upper_bound": args.upper_bound,
    }


def make_physical_energy(args):
    """Create an optional vectorized physical energy functional."""
    if args.physical_energy == "none":
        return None

    if args.physical_energy == "fisher-kpp":
        def fisher_kpp_energy(u):
            dx = float(args.domain_length) / u.shape[-1]
            grad = (torch.roll(u, shifts=-1, dims=-1) - u) / dx
            density = 0.5 * float(args.nu) * grad.square()
            density -= float(args.reaction_rate) * (
                0.5 * u.square() - u.pow(3) / 3.0
            )
            return dx * density.sum(dim=-1)

        return fisher_kpp_energy

    if args.physical_energy == "allen-cahn":
        def allen_cahn_energy(u):
            dx = float(args.domain_length) / u.shape[-1]
            grad = (torch.roll(u, shifts=-1, dims=-1) - u) / dx
            density = 0.5 * float(args.epsilon) ** 2 * grad.square()
            density += 0.25 * (1.0 - u.square()).square()
            return dx * density.sum(dim=-1)

        return allen_cahn_energy

    if args.physical_energy == "burgers-l2":
        def burgers_l2_energy(u):
            dx = float(args.domain_length) / u.shape[-1]
            return 0.5 * dx * u.square().sum(dim=-1)

        return burgers_l2_energy

    raise ValueError(f"unsupported physical energy: {args.physical_energy}")


def _synchronize(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def run_audit(args):
    (
        evaluate_full,
        evaluate_ode_steps_sweep,
        normalize_ode_steps,
        model_classes,
    ) = _import_experiment_modules()
    counts = normalize_ode_steps(args.ode_steps)
    if len(counts) < 2:
        raise ValueError("--ode-steps must contain at least two distinct counts")
    if args.rollout_steps <= 0 or args.max_pairs <= 0:
        raise ValueError("rollout and pair counts must be positive")
    if args.sweep_samples <= 0:
        raise ValueError("--sweep-samples must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("run_metadata") or {}
    args.lower_bound = float(
        _metadata_or(
            metadata,
            args.lower_bound,
            "lower_bound",
            metadata.get("m", 0.0),
        )
    )
    args.upper_bound = float(
        _metadata_or(
            metadata,
            args.upper_bound,
            "upper_bound",
            metadata.get("M", 1.0),
        )
    )
    args.domain_length = float(
        _metadata_or(metadata, args.domain_length, "L", 10.0)
    )
    args.nu = float(_metadata_or(metadata, args.nu, "nu", 0.1))
    args.reaction_rate = float(
        _metadata_or(metadata, args.reaction_rate, "r", 1.0)
    )
    args.epsilon = float(
        _metadata_or(metadata, args.epsilon, "epsilon", 0.1)
    )
    if args.lower_bound >= args.upper_bound:
        raise ValueError("lower bound must be strictly smaller than upper bound")
    if args.domain_length <= 0 or args.nu < 0 or args.epsilon <= 0:
        raise ValueError(
            "domain length and epsilon must be positive, and nu must be non-negative"
        )
    model, model_config = build_model(args, checkpoint, model_classes)
    tau = float(_metadata_or(metadata, args.tau, "tau", 0.1))
    reference_dt = _metadata_or(metadata, args.reference_dt, "dt_pde", None)
    reference_dt_source = "argument_or_checkpoint"
    if reference_dt is None:
        reference_dt = infer_reference_dt(data["val_trajs"])
        reference_dt_source = "first_validation_trajectory"
    elif args.reference_dt is None:
        reference_dt_source = "checkpoint_metadata"
    else:
        reference_dt_source = "command_line"
    if reference_dt is not None:
        reference_dt = float(reference_dt)
    n_available = len(data["val_u0"])
    n_samples = n_available if args.n_samples is None else min(args.n_samples, n_available)
    if n_samples <= 0:
        raise ValueError("data contains no validation samples")
    n_sweep = min(args.sweep_samples, n_samples)
    val_u0 = data["val_u0"][:n_samples]
    val_trajs = data["val_trajs"][:n_samples]
    device = torch.device(args.device)
    model.to(device).eval()
    physical_energy = make_physical_energy(args)
    equal_base = args.equal_work_base_ode_steps
    if equal_base is None:
        equal_base = int(model.ode_steps) if hasattr(model, "ode_steps") else None

    with torch.no_grad():
        model(val_u0[: min(2, n_samples)].to(device), tau)
    _synchronize(device)

    start = time.perf_counter()
    formal_metrics, formal_details = evaluate_full(
        model,
        val_u0,
        val_trajs,
        tau=tau,
        rollout_steps=args.rollout_steps,
        device=device,
        model_name=os.path.basename(args.checkpoint),
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        reference_dt=reference_dt,
        physical_energy_fn=physical_energy,
        equal_work_base_ode_steps=equal_base,
    )
    _synchronize(device)
    formal_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sweep = evaluate_ode_steps_sweep(
        model,
        val_u0[:n_sweep],
        tau,
        counts,
        device=device,
        semigroup_steps=args.rollout_steps,
        max_pairs=args.max_pairs,
        include_defects=True,
    )
    _synchronize(device)
    sweep_seconds = time.perf_counter() - start

    result = {
        "audit_schema_version": 1,
        "audit_kind": "phase0_numerical_semigroup_rk4",
        "semantics": {
            "production_defect": (
                "fixed ode_steps per model call; direct and composed paths do not "
                "receive equal total RK4 work"
            ),
            "equal_work_defect": (
                "base-step matched; direct and composed paths receive the same "
                "total RK4 substep count"
            ),
            "sweep": "one checkpoint reused for all ode_steps; no retraining",
        },
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and str(device).startswith("cuda")
            else None
        ),
        "torch_version": torch.__version__,
        "checkpoint": os.path.abspath(args.checkpoint),
        "data": os.path.abspath(args.data),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model": model_config,
        "tau": tau,
        "reference_dt": reference_dt,
        "reference_dt_source": reference_dt_source,
        "n_samples": n_samples,
        "sweep_samples": n_sweep,
        "rollout_steps": args.rollout_steps,
        "max_pairs": args.max_pairs,
        "ode_steps": list(counts),
        "equal_work_base_ode_steps": equal_base,
        "physical_energy": args.physical_energy,
        "physical_energy_config": {
            "domain_length": args.domain_length,
            "nu": args.nu,
            "reaction_rate": args.reaction_rate,
            "epsilon": args.epsilon,
        },
        "formal_evaluation_seconds": formal_seconds,
        "ode_steps_sweep_seconds": sweep_seconds,
        "formal_metrics": formal_metrics,
        "formal_details": formal_details,
        "ode_steps_sweep": sweep,
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    print(json.dumps(result, indent=2, allow_nan=True))
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    run_audit(args)


if __name__ == "__main__":
    main()
