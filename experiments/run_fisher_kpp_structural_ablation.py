#!/usr/bin/env python3
"""Run the standardized Phase 3 Fisher--KPP structural ablation.

Each invocation trains one variant for one seed and writes a self-contained
result directory.  Data generation, strict reference-time alignment, sparse
validation, and rollout metrics are deliberately delegated to
``run_fisher_fair`` and the shared training/evaluation modules so that Phase 3
uses exactly the same frozen-data controls as the fair Fisher--KPP runner.

The ``full`` variant is the primary architecture-only configuration.  The
``auxiliary_losses`` variant uses the same architecture with the optional
semigroup and learned-energy losses enabled; the remaining variants remove one
structural ingredient while keeping auxiliary losses disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch
import torch.nn.functional as F


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import run_fisher_fair as fair  # noqa: E402
from evaluate import (  # noqa: E402
    evaluate_full,
    prepare_aligned_rollout_batches,
)
from models import (  # noqa: E402
    LatentSemigroupNet,
    _broadcast_positive_tau,
)
from seed_utils import set_global_seed  # noqa: E402
from training import train_model  # noqa: E402


VARIANTS = (
    "full",
    "no_coercive_floor",
    "no_interaction",
    "constant_mobility",
    "auxiliary_losses",
)
DEFAULT_TRAIN_TAUS = (0.025, 0.05, 0.1, 0.2)
DEFAULT_AUXILIARY_LOSS_WEIGHTS = {
    "alpha_rollout": 0.1,
    "alpha_energy": 0.01,
    "alpha_bound": 0.0,
    "alpha_V": 0.0,
}


class NoInteractionLatent(LatentSemigroupNet):
    """Latent model with the spatial interaction term and its parameters removed."""

    def __init__(self, **kwargs):
        kwargs["interaction_radius"] = 0
        super().__init__(**kwargs)
        # The base class always constructs these objects.  Removing them here
        # makes the ablation's parameter count represent the actual model,
        # rather than counting parameters that can never affect its dynamics.
        del self.emb
        del self.interaction_mask

    def psi(self, z):
        return self.V_net(z.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    def grad_psi(self, z, a_ij=None):
        _v_value, d_v_dz = self.V_net.value_and_grad(z.unsqueeze(-1))
        return d_v_dz.squeeze(-1)

    def latent_dynamics(self, z, a_ij=None):
        mobility = F.softplus(self.K_net(z)) + 5e-3
        return -mobility * self.grad_psi(z)

    def forward(self, u, tau):
        z = self.encode(u)
        dt = _broadcast_positive_tau(tau, z)[:, 0, :1] / self.ode_steps
        for _ in range(self.ode_steps):
            z = self.rk4_step(z, dt)
        return self.decode(z)


class ConstantMobilityLatent(LatentSemigroupNet):
    """Latent model with an identity, state-independent mobility."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # There is no reason to retain an unreachable K network in this
        # structural variant; removing it makes the reported count explicit.
        del self.K_net

    def _dynamics_and_grad(self, z, a_ij):
        return torch.ones_like(z), self.grad_psi(z, a_ij=a_ij)


def build_parser():
    """Build the one-variant, one-seed Phase 3 command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=VARIANTS,
        default="full",
        help="one structural variant to train; ignored by data-only preparation",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help=(
            "generate or validate the frozen data cache, write data provenance, "
            "and exit without creating training checkpoints"
        ),
    )

    # Fisher--KPP and frozen-data controls.  These mirror the common fair
    # runner, with fixed tau because Phase 3 is a fixed-step ablation.
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=1.0)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--fixed-tau", "--tau", dest="fixed_tau", type=float, default=0.1)
    parser.add_argument("--eval-horizon", type=float, default=1.2)
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=50)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--beta-v-floor", type=float, default=0.1)
    parser.add_argument(
        "--aux-alpha-rollout",
        type=float,
        default=DEFAULT_AUXILIARY_LOSS_WEIGHTS["alpha_rollout"],
        help="rollout-loss weight used only by auxiliary_losses",
    )
    parser.add_argument(
        "--aux-alpha-energy",
        type=float,
        default=DEFAULT_AUXILIARY_LOSS_WEIGHTS["alpha_energy"],
        help="learned-energy-loss weight used only by auxiliary_losses",
    )
    parser.add_argument(
        "--aux-alpha-v",
        type=float,
        default=DEFAULT_AUXILIARY_LOSS_WEIGHTS["alpha_V"],
        help="direct V-value auxiliary-loss weight used only by auxiliary_losses",
    )
    return parser


def _validate_args(args):
    positive_fields = (
        "N",
        "L",
        "nu",
        "reaction_rate",
        "reference_dt",
        "fixed_tau",
        "eval_horizon",
        "n_train",
        "n_val",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
    )
    for field in positive_fields:
        value = getattr(args, field)
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{field} must be finite and strictly positive")
    if args.validation_interval <= 0:
        raise ValueError("validation_interval must be positive")
    if args.beta_v_floor < 0:
        raise ValueError("beta_v_floor must be non-negative")
    for field in ("aux_alpha_rollout", "aux_alpha_energy", "aux_alpha_v"):
        value = getattr(args, field)
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{field} must be finite and non-negative")
    fair.rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)


def _fair_data_args(args):
    """Adapt the Phase 3 namespace to the shared fair-runner data API."""

    values = vars(args).copy()
    values.update(
        {
            "regime": "fixed",
            "train_taus": DEFAULT_TRAIN_TAUS,
        }
    )
    return argparse.Namespace(**values)


def load_or_generate_frozen_data(args, *, allow_generate=True):
    """Load or atomically create the exact fixed-time fair-runner cache.

    Normal training invocations deliberately require an existing cache.  The
    separate preparation mode is the only path that may generate it, which
    makes parallel per-seed jobs read-only with respect to the shared cache.
    """

    cache_path = Path(args.data_cache).absolute()
    fair_args = _fair_data_args(args)
    if cache_path.exists():
        return fair.load_or_generate_data(fair_args)
    if not allow_generate:
        raise FileNotFoundError(
            "frozen data cache does not exist; run this runner once with "
            "--prepare-data-only before launching training jobs: "
            f"{cache_path}"
        )

    data = fair.generate_data(fair_args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f"{cache_path.name}.tmp-{os.getpid()}"
    )
    try:
        torch.save(data, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    # Re-open through the shared loader so preparation validates the same
    # configuration check used by every subsequent training process.
    return fair.load_or_generate_data(fair_args)


def validate_frozen_data(args, data):
    """Validate cache structure, finiteness, and all Phase 3 time alignment."""

    expected_config = fair.data_config(_fair_data_args(args))
    if data.get("data_generation_config") != expected_config:
        raise ValueError("data cache configuration does not match this run")
    required_keys = {
        "train_u0",
        "train_tau",
        "train_ut",
        "val_u0",
        "val_trajs",
        "data_generation_config",
    }
    missing = sorted(required_keys.difference(data))
    if missing:
        raise ValueError(f"data cache is missing required keys: {missing}")
    if data["train_tau"] is not None:
        raise ValueError("fixed-time Fisher--KPP cache must have train_tau=None")
    train_u0 = data["train_u0"]
    train_ut = data["train_ut"]
    val_u0 = data["val_u0"]
    val_trajs = data["val_trajs"]
    if not torch.is_tensor(train_u0) or not torch.is_tensor(train_ut):
        raise ValueError("training states in data cache must be tensors")
    if train_u0.shape != train_ut.shape:
        raise ValueError("train_u0 and train_ut must have identical shapes")
    if tuple(train_u0.shape) != (args.n_train, args.N):
        raise ValueError(
            f"unexpected training shape {tuple(train_u0.shape)}, "
            f"expected {(args.n_train, args.N)}"
        )
    if not torch.isfinite(train_u0).all() or not torch.isfinite(train_ut).all():
        raise ValueError("training cache contains non-finite values")
    if len(val_u0) != args.n_val or len(val_trajs) != args.n_val:
        raise ValueError("validation cache length does not match n_val")
    if not torch.is_tensor(val_u0) or tuple(val_u0.shape) != (args.n_val, args.N):
        raise ValueError("validation initial states have an unexpected shape")
    if not torch.isfinite(val_u0).all():
        raise ValueError("validation initial states contain non-finite values")
    for trajectory_index, trajectory in enumerate(val_trajs):
        if not isinstance(trajectory, (tuple, list)) or len(trajectory) != 2:
            raise ValueError(
                f"validation trajectory {trajectory_index} must be a (time, state) pair"
            )
        times, states = trajectory
        if not torch.is_tensor(times) or not torch.is_tensor(states):
            raise ValueError(
                f"validation trajectory {trajectory_index} must contain tensors"
            )
        if not torch.isfinite(times).all() or not torch.isfinite(states).all():
            raise ValueError(
                f"validation trajectory {trajectory_index} contains non-finite values"
            )

    rollout_steps = fair.rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)
    reference_stride, rollout_batches, evaluated_steps = prepare_aligned_rollout_batches(
        val_u0,
        val_trajs,
        args.fixed_tau,
        rollout_steps,
        args.reference_dt,
    )
    if len(evaluated_steps) != args.n_val or len(rollout_batches) == 0:
        raise ValueError("validation cache does not contain a complete aligned horizon")
    return {
        "reference_stride": reference_stride,
        "rollout_steps": rollout_steps,
        "aligned_validation_trajectories": len(evaluated_steps),
        "train_shape": list(train_u0.shape),
        "validation_shape": list(val_u0.shape),
        "train_state_range": [float(train_u0.min()), float(train_u0.max())],
        "validation_state_range": [float(val_u0.min()), float(val_u0.max())],
    }


def write_data_provenance(args, data, validation):
    """Write the compact provenance artifact for the one-time prep job."""

    cache_path = Path(args.data_cache).absolute()
    provenance = {
        "schema_version": 1,
        "experiment": "fisher_kpp_structural_ablation",
        "phase": 3,
        "mode": "prepare-data-only",
        "status": "validated",
        "created_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "data_seed": args.data_seed,
        "deterministic": bool(args.deterministic),
        "data_generation_config": data["data_generation_config"],
        "validation": validation,
        "data_cache": {
            "path": str(cache_path),
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256_file(cache_path),
        },
        "git": _git_provenance(),
        "runner_sha256": _sha256_file(Path(__file__)),
    }
    output_path = Path(args.output_dir).absolute() / "data_provenance.json"
    _write_json(output_path, provenance)
    return output_path, provenance


def variant_configuration(args):
    """Return JSON-safe architecture and loss settings for one variant."""

    if args.variant == "no_coercive_floor":
        beta_v_floor = 0.0
    else:
        beta_v_floor = float(args.beta_v_floor)

    losses = {
        "alpha_rollout": 0.0,
        "alpha_energy": 0.0,
        "alpha_bound": 0.0,
        "alpha_V": 0.0,
        "weight_decay": float(args.weight_decay),
    }
    if args.variant == "auxiliary_losses":
        losses.update(
            {
                "alpha_rollout": float(args.aux_alpha_rollout),
                "alpha_energy": float(args.aux_alpha_energy),
                "alpha_V": float(args.aux_alpha_v),
            }
        )

    interaction_radius = 0 if args.variant == "no_interaction" else 2
    mobility = "constant_identity" if args.variant == "constant_mobility" else "learned_diagonal"
    return {
        "variant": args.variant,
        "architecture_only": args.variant != "auxiliary_losses",
        "beta_V": 0.0,
        "beta_V_floor": beta_v_floor,
        "hidden_V": [64, 64],
        "hidden_K": [64, 64],
        "stencil_radius": 3,
        "interaction_radius": interaction_radius,
        "mobility": mobility,
        "loss_weights": losses,
    }


def build_model(args):
    """Construct a structural variant without changing shared model files."""

    config = variant_configuration(args)
    common = {
        "N": args.N,
        "hidden_V": config["hidden_V"],
        "hidden_K": config["hidden_K"],
        "stencil_radius": config["stencil_radius"],
        "interaction_radius": config["interaction_radius"],
        "beta_V": config["beta_V"],
        "beta_V_floor": config["beta_V_floor"],
    }
    if args.variant == "no_interaction":
        return NoInteractionLatent(**common)
    if args.variant == "constant_mobility":
        return ConstantMobilityLatent(**common)
    return LatentSemigroupNet(**common)


def _count_parameters(model):
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def _sync_if_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark_batched_inference(
    model,
    val_u0,
    val_trajs,
    tau,
    rollout_steps,
    device,
    reference_dt,
):
    """Measure plain batched rollout latency after strict alignment checks."""

    _stride, rollout_batches, _evaluated_steps = prepare_aligned_rollout_batches(
        val_u0,
        val_trajs,
        tau,
        rollout_steps,
        reference_dt,
    )
    model.eval()
    _sync_if_cuda(device)
    started = time.perf_counter()
    state_steps = 0
    forward_calls = 0
    for rollout_batch in rollout_batches:
        state = rollout_batch["initial_states"].to(device)
        for _ in range(rollout_batch["steps"]):
            state = model(state, tau)
            state_steps += int(state.shape[0])
            forward_calls += 1
    _sync_if_cuda(device)
    seconds = time.perf_counter() - started
    ode_steps = int(getattr(model, "ode_steps", 0))
    return {
        "seconds": seconds,
        "state_steps": state_steps,
        "model_forward_calls": forward_calls,
        "inference_seconds_per_state_step": seconds / state_steps if state_steps else float("nan"),
        "inference_seconds_per_model_forward": seconds / forward_calls if forward_calls else float("nan"),
        "estimated_rk4_vector_field_calls": (
            forward_calls * ode_steps * 4 if ode_steps else None
        ),
        "ode_steps": ode_steps if ode_steps else None,
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance():
    result = {"commit": None, "worktree_status": None}
    try:
        result["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        result["worktree_status"] = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPOSITORY_DIR,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


def _source_provenance():
    source_paths = (
        Path(__file__),
        EXPERIMENTS_DIR / "run_fisher_fair.py",
        EXPERIMENTS_DIR / "training.py",
        EXPERIMENTS_DIR / "evaluate.py",
        EXPERIMENTS_DIR / "models.py",
        EXPERIMENTS_DIR / "pde_solver.py",
        EXPERIMENTS_DIR / "seed_utils.py",
    )
    return {
        str(path.relative_to(REPOSITORY_DIR)): _sha256_file(path)
        for path in source_paths
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def _device_provenance(device):
    return {
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if str(device).startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def run_variant(args, data, device):
    """Train, checkpoint, evaluate, and write one Phase 3 variant."""

    set_global_seed(args.seed, deterministic=args.deterministic)
    model = build_model(args)
    counts = _count_parameters(model)
    variant_config = variant_configuration(args)
    output_dir = Path(args.output_dir).absolute()
    model_dir = output_dir / args.variant
    checkpoint_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    run_metadata = {
        "experiment": "fisher_kpp_structural_ablation",
        "phase": 3,
        "variant_config": variant_config,
        "parameter_counts": counts,
        "training_seed": args.seed,
        "data_seed": args.data_seed,
        "deterministic": bool(args.deterministic),
        "fixed_tau": args.fixed_tau,
        "reference_dt": args.reference_dt,
        "validation_interval": args.validation_interval,
    }
    loss_weights = variant_config["loss_weights"]

    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        _sync_if_cuda(device)
    training_started = time.perf_counter()
    history = train_model(
        model,
        data["train_u0"],
        data["train_ut"],
        data["val_u0"],
        data["val_trajs"],
        tau=args.fixed_tau,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha_rollout=loss_weights["alpha_rollout"],
        alpha_energy=loss_weights["alpha_energy"],
        alpha_bound=loss_weights["alpha_bound"],
        alpha_V=loss_weights["alpha_V"],
        weight_decay=loss_weights["weight_decay"],
        checkpoint_dir=str(checkpoint_dir),
        model_name=args.variant,
        device=device,
        resume_from=(
            None
            if args.no_resume
            else str(checkpoint_dir / f"{args.variant}_best.pt")
        ),
        reference_dt=args.reference_dt,
        run_metadata=run_metadata,
        validation_interval=args.validation_interval,
    )
    _sync_if_cuda(device)
    training_seconds = time.perf_counter() - training_started
    training_peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if str(device).startswith("cuda")
        else None
    )

    best_path = checkpoint_dir / f"{args.variant}_best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"best checkpoint was not written: {best_path}")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    rollout_steps = fair.rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    metrics, details = evaluate_full(
        model,
        data["val_u0"],
        data["val_trajs"],
        tau=args.fixed_tau,
        rollout_steps=rollout_steps,
        device=device,
        model_name=args.variant,
        reference_dt=args.reference_dt,
        physical_energy_fn=fair.fisher_energy(_fair_data_args(args)),
        collect_latent_diagnostics=True,
        equal_work_base_ode_steps=30,
    )
    _sync_if_cuda(device)
    evaluation_seconds = time.perf_counter() - evaluation_started
    evaluation_peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if str(device).startswith("cuda")
        else None
    )
    inference = benchmark_batched_inference(
        model,
        data["val_u0"],
        data["val_trajs"],
        args.fixed_tau,
        rollout_steps,
        device,
        args.reference_dt,
    )

    final_variant_config = dict(variant_config)
    final_variant_config["parameter_counts"] = counts
    final_variant_config["effective_beta_after_training"] = float(
        model.V_net.effective_beta.detach().cpu().item()
    )
    result = {
        "schema_version": 1,
        "experiment": "fisher_kpp_structural_ablation",
        "phase": 3,
        "variant": args.variant,
        "variant_config": final_variant_config,
        "parameter_count": counts["parameter_count"],
        "trainable_parameter_count": counts["trainable_parameter_count"],
        "checkpoint_epoch": checkpoint.get("epoch"),
        "training_seconds": training_seconds,
        "training_peak_memory_bytes": training_peak_memory_bytes,
        "training_examples_per_second": (
            args.epochs * args.n_train / training_seconds
            if training_seconds > 0
            else float("nan")
        ),
        "optimizer_updates": args.epochs * math.ceil(args.n_train / args.batch_size),
        "examples_seen": args.epochs * args.n_train,
        "loss_weights": loss_weights,
        "beta_V_floor": variant_config["beta_V_floor"],
        "history": history,
        "evaluations": {
            str(args.fixed_tau): {
                "metrics": metrics,
                "details": details,
                "evaluation_seconds": evaluation_seconds,
                "evaluation_peak_memory_bytes": evaluation_peak_memory_bytes,
                "batched_inference": inference,
            }
        },
    }
    _write_json(model_dir / "result.json", result)
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if (
        not args.prepare_data_only
        and str(args.device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but no CUDA device is available")

    output_dir = Path(args.output_dir).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_data_only:
        data = load_or_generate_frozen_data(args, allow_generate=True)
        validation = validate_frozen_data(args, data)
        provenance_path, provenance = write_data_provenance(
            args, data, validation
        )
        print(
            json.dumps(
                {
                    "mode": provenance["mode"],
                    "data_cache": provenance["data_cache"],
                    "provenance": str(provenance_path),
                },
                indent=2,
            )
        )
        return

    # Training jobs are read-only consumers of the shared cache.  This is the
    # guard that prevents three per-seed jobs from racing to create it.
    data = load_or_generate_frozen_data(args, allow_generate=False)
    validate_frozen_data(args, data)
    device = torch.device(args.device)
    data_cache = Path(args.data_cache).absolute()

    provenance = {
        "schema_version": 1,
        "experiment": "fisher_kpp_structural_ablation",
        "phase": 3,
        "variant": args.variant,
        "status": "started",
        "started_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "command": [sys.executable, str(Path(__file__).absolute()), *sys.argv[1:]]
        if argv is None
        else [sys.executable, str(Path(__file__).absolute()), *list(argv)],
        "config": vars(args),
        "data_generation_config": data["data_generation_config"],
        "data_cache": {
            "path": str(data_cache),
            "sha256": _sha256_file(data_cache),
        },
        "git": _git_provenance(),
        "source_sha256": _source_provenance(),
        "device": _device_provenance(device),
        "variant_config": variant_configuration(args),
    }
    _write_json(output_dir / "provenance.json", provenance)

    result = run_variant(args, data, device)
    provenance.update(
        {
            "status": "completed",
            "completed_at_unix": time.time(),
            "result_path": str(output_dir / args.variant / "result.json"),
            "parameter_count": result["parameter_count"],
            "metrics": result["evaluations"][str(args.fixed_tau)]["metrics"],
        }
    )
    _write_json(output_dir / "provenance.json", provenance)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment": "fisher_kpp_structural_ablation",
            "phase": 3,
            "variant": args.variant,
            "architecture_only_primary": args.variant == "full",
            "provenance_path": str(output_dir / "provenance.json"),
            "result": result,
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "variant": args.variant,
                "parameter_count": result["parameter_count"],
                "rollout_mse": result["evaluations"][str(args.fixed_tau)]["metrics"].get(
                    "rollout_mse_mean"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
