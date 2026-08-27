#!/usr/bin/env python3
"""Evaluate frozen Fisher--KPP checkpoints on an independent locked test set.

Evaluation mode deliberately has no optimizer, training, checkpoint-selection,
or test-cache-generation path.  It reads a selected checkpoint and an already
frozen independently seeded cache, then evaluates requested horizons as
prefixes of that one cache.  The explicit ``--prepare-test-data-only`` mode is
the only path allowed to create the test cache.

The companion fair runner records model reconstruction, training protocol, and
runtime-source hashes in every newly trained checkpoint.  This entry point
checks those fields before inference so that a locked result cannot quietly
mix an incompatible Fisher--KPP model, PDE, or source implementation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from evaluate import evaluate_full  # noqa: E402
from pde_solver import FisherKPPSolver, generate_initial_conditions  # noqa: E402
from run_fisher_fair import (  # noqa: E402
    MODEL_NAMES,
    _device_provenance,
    _git_commit,
    _sha256_file,
    _source_hashes,
    build_model,
    fisher_energy,
    model_config,
    parse_float_list,
    reference_steps_for_duration,
    rollout_steps_for_horizon,
)
from seed_utils import set_global_seed  # noqa: E402


LOWER_BOUND = 0.0
UPPER_BOUND = 1.0
LOCKED_TEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
MULTI_HORIZON_EVALUATION_SCHEMA_VERSION = 2
DEFAULT_TEST_SEED = 314163
DEFAULT_N_TEST = 500
DEFAULT_EVAL_HORIZON = 1.2
DEFAULT_TRAIN_TAUS = (0.025, 0.05, 0.1, 0.2)
LATENT_MODELS = {"latent", "latent_query_time"}


def parse_positive_float_list(value):
    """Parse a comma-separated list of finite, strictly positive horizons."""
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated float list"
        ) from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "horizons must be finite and strictly positive"
        )
    return values


def requested_eval_horizons(args):
    """Return the ordered physical horizons requested by the evaluator."""
    if args.eval_horizons is not None:
        return tuple(float(horizon) for horizon in args.eval_horizons)
    return (float(args.eval_horizon),)


def cache_eval_horizon(args):
    """Return the longest requested horizon that the frozen cache must cover."""
    return max(requested_eval_horizons(args))


def checkpoint_eval_horizon(args):
    """Return the training-selection horizon recorded in the checkpoint."""
    if args.checkpoint_eval_horizon is not None:
        return float(args.checkpoint_eval_horizon)
    return requested_eval_horizons(args)[0]


def evaluation_tau(args):
    """Return the requested locked-test model lag without changing training tau."""
    return float(args.fixed_tau if args.eval_tau is None else args.eval_tau)


def horizon_key(horizon):
    """Return a stable JSON/CSV key for one physical horizon."""
    return format(float(horizon), ".15g")


def _evaluation_schema_version(args):
    return (
        MULTI_HORIZON_EVALUATION_SCHEMA_VERSION
        if len(requested_eval_horizons(args)) > 1
        else EVALUATION_SCHEMA_VERSION
    )


def build_parser():
    """Build the strict Fisher--KPP checkpoint-only evaluator CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="selected *_best.pt checkpoint; required unless only preparing test data",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--reaction-rate", type=float, default=1.0)
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument(
        "--fixed-tau",
        type=float,
        default=0.1,
        help="the selected checkpoint's training-protocol lag",
    )
    parser.add_argument(
        "--eval-tau",
        type=float,
        default=None,
        help=(
            "optional aligned unseen rollout lag; fixed-tau remains the "
            "checkpoint's recorded training-protocol lag"
        ),
    )
    horizon_group = parser.add_mutually_exclusive_group()
    horizon_group.add_argument(
        "--eval-horizon",
        type=float,
        default=DEFAULT_EVAL_HORIZON,
        help="evaluate one physical horizon (default: 1.2)",
    )
    horizon_group.add_argument(
        "--eval-horizons",
        type=parse_positive_float_list,
        help="comma-separated, strictly increasing horizons",
    )
    parser.add_argument(
        "--checkpoint-eval-horizon",
        type=float,
        default=None,
        help=(
            "checkpoint training-selection horizon; defaults to the first "
            "requested evaluation horizon"
        ),
    )
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument(
        "--beta-v-floor",
        type=float,
        default=0.1,
        help="must exactly match the latent potential floor in the checkpoint",
    )
    parser.add_argument(
        "--expected-train-taus",
        type=parse_float_list,
        default=DEFAULT_TRAIN_TAUS,
        help="frozen variable-lag Fisher training protocol",
    )
    parser.add_argument("--expected-n-train", type=int, default=1000)
    parser.add_argument("--expected-n-val", type=int, default=50)
    parser.add_argument("--expected-epochs", type=int, default=100)
    parser.add_argument("--expected-batch-size", type=int, default=64)
    parser.add_argument("--expected-lr", type=float, default=1e-3)
    parser.add_argument("--expected-validation-interval", type=int, default=5)
    parser.add_argument(
        "--prepare-test-data-only",
        action="store_true",
        help="create or validate a locked cache without loading a checkpoint",
    )
    return parser


def validate_args(args):
    """Validate locked-test identity, alignment, and reconstruction inputs."""
    if args.test_seed < 0:
        raise ValueError("test-seed must be non-negative")
    if args.N <= 1:
        raise ValueError("N must be greater than one")
    if args.n_test <= 0:
        raise ValueError("n-test must be positive")
    if args.expected_n_train <= 0 or args.expected_n_val <= 0:
        raise ValueError("expected training and validation counts must be positive")
    if args.expected_epochs <= 0 or args.expected_batch_size <= 0:
        raise ValueError("expected epochs and batch size must be positive")
    if args.expected_validation_interval <= 0:
        raise ValueError("expected validation interval must be positive")
    for name in (
        "L",
        "nu",
        "reaction_rate",
        "reference_dt",
        "fixed_tau",
        "expected_lr",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not math.isfinite(args.beta_v_floor) or args.beta_v_floor < 0:
        raise ValueError("beta-v-floor must be finite and non-negative")
    if not args.expected_train_taus:
        raise ValueError("expected-train-taus must not be empty")
    for tau in args.expected_train_taus:
        reference_steps_for_duration(tau, args.reference_dt)
    rollout_tau = evaluation_tau(args)
    if not math.isfinite(rollout_tau) or rollout_tau <= 0:
        raise ValueError("eval-tau must be finite and strictly positive")
    horizons = requested_eval_horizons(args)
    if any(not math.isfinite(horizon) or horizon <= 0 for horizon in horizons):
        raise ValueError("evaluation horizons must be finite and strictly positive")
    if any(left >= right for left, right in zip(horizons, horizons[1:])):
        raise ValueError("eval-horizons must be strictly increasing")
    for horizon in horizons:
        reference_steps_for_duration(horizon, args.reference_dt)
        rollout_steps_for_horizon(horizon, rollout_tau)
    reference_steps_for_duration(checkpoint_eval_horizon(args), args.reference_dt)
    reference_steps_for_duration(args.fixed_tau, args.reference_dt)
    if not args.prepare_test_data_only and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --prepare-test-data-only is used")


def locked_test_config(args):
    """Return the complete identity of the independent Fisher--KPP split."""
    return {
        "schema_version": LOCKED_TEST_SCHEMA_VERSION,
        "pde": "fisher-kpp",
        "equation": "u_t=nu*u_xx+r*u*(1-u)",
        "split": "locked_independent_test",
        "split_policy": "independent_seeded_stream_distinct_from_training_data",
        "test_seed": int(args.test_seed),
        "deterministic": bool(args.deterministic),
        "initial_condition_policy": "low_frequency_fourier_modes_clamped_to_[0.05,0.95]",
        "N": int(args.N),
        "L": float(args.L),
        "nu": float(args.nu),
        "reaction_rate": float(args.reaction_rate),
        "reference_dt": float(args.reference_dt),
        "fixed_tau": float(args.fixed_tau),
        "eval_horizon": cache_eval_horizon(args),
        "n_test": int(args.n_test),
        "bounds": [LOWER_BOUND, UPPER_BOUND],
    }


def _atomic_torch_save(payload, path):
    """Serialize a cache atomically so partial artifacts cannot be reused."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json(path, payload):
    """Write one JSON artifact atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _render_metric_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True, allow_nan=True)


def _write_metrics_csv(path, evaluations):
    """Write a stable two-column metrics table across requested horizons."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        for evaluation in evaluations:
            prefix = f"horizon={horizon_key(evaluation['horizon'])}:"
            for name in sorted(evaluation["metrics"]):
                writer.writerow(
                    {
                        "metric": prefix + name,
                        "value": _render_metric_value(evaluation["metrics"][name]),
                    }
                )


def _expected_times(n_steps, reference_dt):
    return torch.arange(n_steps + 1, dtype=torch.float64) * float(reference_dt)


def validate_locked_test_cache(data, args):
    """Validate cache identity, complete reference trajectories, and bounds."""
    if not isinstance(data, Mapping):
        raise ValueError("locked test cache must be a mapping")
    if data.get("schema_version") != LOCKED_TEST_SCHEMA_VERSION:
        raise ValueError("locked test cache schema version does not match")
    expected_config = locked_test_config(args)
    actual_config = data.get("locked_test_config")
    if actual_config != expected_config:
        actual_horizon = (
            actual_config.get("eval_horizon")
            if isinstance(actual_config, Mapping)
            else None
        )
        if (
            isinstance(actual_horizon, (int, float))
            and float(actual_horizon) < cache_eval_horizon(args)
            and isinstance(actual_config, Mapping)
            and all(
                actual_config.get(name) == value
                for name, value in expected_config.items()
                if name != "eval_horizon"
            )
        ):
            raise ValueError(
                "locked test cache horizon is too short for this evaluation: "
                f"cache={actual_horizon}, required={cache_eval_horizon(args)}"
            )
        raise ValueError("locked test cache configuration differs from the protocol")

    test_u0 = data.get("test_u0")
    test_trajs = data.get("test_trajs")
    if not torch.is_tensor(test_u0) or tuple(test_u0.shape) != (args.n_test, args.N):
        raise ValueError(
            "locked test cache must contain test_u0 with shape "
            f"({args.n_test}, {args.N})"
        )
    if not torch.isfinite(test_u0).all():
        raise ValueError("locked test cache contains non-finite initial states")
    if float(test_u0.min()) < LOWER_BOUND - 1e-4 or float(test_u0.max()) > UPPER_BOUND + 1e-4:
        raise ValueError("locked test initial states leave the admissible bounds")
    if not isinstance(test_trajs, (list, tuple)) or len(test_trajs) != args.n_test:
        raise ValueError("locked test cache must contain one trajectory per state")

    n_reference_steps = reference_steps_for_duration(
        cache_eval_horizon(args), args.reference_dt
    )
    expected_times = _expected_times(n_reference_steps, args.reference_dt)
    all_states = []
    for index, trajectory in enumerate(test_trajs):
        if not isinstance(trajectory, (tuple, list)) or len(trajectory) != 2:
            raise ValueError(f"locked test trajectory {index} must be a (times, states) pair")
        times, states = trajectory
        if not torch.is_tensor(times) or not torch.is_tensor(states):
            raise ValueError(f"locked test trajectory {index} must contain tensors")
        if tuple(times.shape) != (n_reference_steps + 1,):
            raise ValueError(f"locked test trajectory {index} has an invalid time shape")
        if tuple(states.shape) != (n_reference_steps + 1, args.N):
            raise ValueError(f"locked test trajectory {index} has an invalid state shape")
        if not torch.isfinite(times).all() or not torch.isfinite(states).all():
            raise ValueError(f"locked test trajectory {index} contains non-finite values")
        if not torch.allclose(
            times.to(torch.float64), expected_times, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"locked test trajectory {index} has a mismatched time grid")
        if float(states.min()) < LOWER_BOUND - 1e-4 or float(states.max()) > UPPER_BOUND + 1e-4:
            raise ValueError(f"locked test trajectory {index} leaves the admissible bounds")
        if not torch.allclose(states[0], test_u0[index], rtol=1e-6, atol=1e-6):
            raise ValueError(f"locked test trajectory {index} does not start at test_u0")
        all_states.append(states)

    stacked_states = torch.stack(all_states)
    return {
        "test_u0_shape": list(test_u0.shape),
        "test_trajectory_length": n_reference_steps + 1,
        "cache_horizon": cache_eval_horizon(args),
        "requested_horizons": list(requested_eval_horizons(args)),
        "test_u0_range": [float(test_u0.min()), float(test_u0.max())],
        "test_trajectory_range": [
            float(stacked_states.min()),
            float(stacked_states.max()),
        ],
    }


def generate_locked_test_data(args):
    """Generate one deterministic, standalone Fisher--KPP test split."""
    set_global_seed(args.test_seed, deterministic=args.deterministic)
    solver = FisherKPPSolver(
        N=args.N,
        L=args.L,
        nu=args.nu,
        r=args.reaction_rate,
        dt=args.reference_dt,
    )
    test_u0 = generate_initial_conditions(args.N, args.n_test, args.L)
    horizon = cache_eval_horizon(args)
    test_trajs = [solver.solve(initial, horizon, save_every=1) for initial in test_u0]
    return {
        "schema_version": LOCKED_TEST_SCHEMA_VERSION,
        "locked_test_config": locked_test_config(args),
        "test_u0": test_u0,
        "test_trajs": test_trajs,
    }


def load_or_generate_locked_test_data(args, *, allow_generate):
    """Load a valid cache or create it only in explicit preparation mode."""
    cache_path = Path(args.test_cache).resolve()
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        return data, "loaded", validate_locked_test_cache(data, args)
    if not allow_generate:
        raise FileNotFoundError(
            "locked test cache does not exist; freeze it first with "
            f"--prepare-test-data-only: {cache_path}"
        )
    data = generate_locked_test_data(args)
    validate_locked_test_cache(data, args)
    _atomic_torch_save(data, cache_path)
    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
    return saved, "created", validate_locked_test_cache(saved, args)


def cache_prefix_trajectories(test_data, horizon, args):
    """Return read-only prefixes from one frozen cache without resimulation."""
    cached_horizon = float(test_data["locked_test_config"]["eval_horizon"])
    if horizon > cached_horizon:
        raise ValueError(
            "requested horizon exceeds the frozen cache: "
            f"requested={horizon}, cache={cached_horizon}"
        )
    n_reference_steps = reference_steps_for_duration(horizon, args.reference_dt)
    return [
        (times[: n_reference_steps + 1], states[: n_reference_steps + 1])
        for times, states in test_data["test_trajs"]
    ]


def _model_args(args):
    return SimpleNamespace(N=args.N, beta_v_floor=args.beta_v_floor)


def _float_matches(actual, expected):
    return actual is not None and math.isclose(
        float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
    )


def _taus_match(actual, expected):
    if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
        return False
    return all(_float_matches(left, right) for left, right in zip(actual, expected))


def _assert_checkpoint_compatible(checkpoint, args):
    """Reject model, PDE, protocol, or source incompatibilities before inference."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")
    metadata = checkpoint.get("run_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint has no run_metadata for protocol validation")
    if metadata.get("model") != args.model:
        raise ValueError(
            "checkpoint model does not match --model: "
            f"{metadata.get('model')!r} != {args.model!r}"
        )
    if metadata.get("regime") != "variable":
        raise ValueError("formal Fisher transfer checkpoints must use variable-lag training")
    expected_model_config = model_config(args.model, _model_args(args))
    if metadata.get("model_config") != expected_model_config:
        raise ValueError("checkpoint model configuration does not match evaluation args")

    expected_numeric = {
        "L": args.L,
        "nu": args.nu,
        "r": args.reaction_rate,
        "reference_dt": args.reference_dt,
        "fixed_tau": args.fixed_tau,
        "eval_horizon": checkpoint_eval_horizon(args),
        "lr": args.expected_lr,
    }
    for name, expected in expected_numeric.items():
        if not _float_matches(metadata.get(name), expected):
            raise ValueError(
                f"checkpoint {name}={metadata.get(name)!r} does not match requested {expected!r}"
            )
    expected_integer = {
        "N": args.N,
        "n_train": args.expected_n_train,
        "n_val": args.expected_n_val,
        "epochs": args.expected_epochs,
        "batch_size": args.expected_batch_size,
        "validation_interval": args.expected_validation_interval,
    }
    for name, expected in expected_integer.items():
        if int(metadata.get(name, -1)) != int(expected):
            raise ValueError(
                f"checkpoint {name}={metadata.get(name)!r} does not match requested {expected!r}"
            )
    if not _taus_match(metadata.get("train_taus"), args.expected_train_taus):
        raise ValueError("checkpoint train_taus do not match the locked Fisher protocol")
    if metadata.get("architecture_only") is not True:
        raise ValueError("formal Fisher checkpoint must be trained architecture-only")
    losses = metadata.get("auxiliary_loss_weights")
    if not isinstance(losses, Mapping) or any(float(value) != 0.0 for value in losses.values()):
        raise ValueError("formal Fisher checkpoint has non-zero auxiliary loss weights")
    if int(metadata.get("data_seed", -1)) == int(args.test_seed):
        raise ValueError("locked test seed must differ from training-data seed")

    expected_parameter_count = sum(
        parameter.numel() for parameter in build_model(args.model, _model_args(args)).parameters()
    )
    if int(metadata.get("parameter_count", -1)) != expected_parameter_count:
        raise ValueError("checkpoint parameter count does not match reconstructed model")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no provenance for compatibility validation")
    recorded_source_hashes = provenance.get("source_hashes")
    expected_source_hashes = _source_hashes()
    if not isinstance(recorded_source_hashes, Mapping):
        raise ValueError("checkpoint provenance has no source hashes")
    mismatches = {
        name: {"checkpoint": recorded_source_hashes.get(name), "evaluator": expected}
        for name, expected in expected_source_hashes.items()
        if recorded_source_hashes.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "checkpoint source hashes differ from evaluator runtime: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return metadata, expected_model_config


def _evaluate_one_horizon(model, test_data, args, horizon, device):
    """Evaluate one trajectory prefix without changing inputs or model weights."""
    tau = evaluation_tau(args)
    rollout_steps = rollout_steps_for_horizon(horizon, tau)
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    metrics, details = evaluate_full(
        model,
        test_data["test_u0"],
        cache_prefix_trajectories(test_data, horizon, args),
        tau=tau,
        rollout_steps=rollout_steps,
        device=device,
        model_name=args.model,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        reference_dt=args.reference_dt,
        physical_energy_fn=fisher_energy(args),
        collect_latent_diagnostics=args.model in LATENT_MODELS,
        equal_work_base_ode_steps=30 if args.model in LATENT_MODELS else None,
    )
    if is_cuda:
        torch.cuda.synchronize(device)
    return {
        "horizon": float(horizon),
        "evaluation": {
            "tau": tau,
            "horizon": float(horizon),
            "rollout_steps": rollout_steps,
            "seconds": time.perf_counter() - start,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if is_cuda else None
            ),
        },
        "metrics": metrics,
        "details": details,
    }


def evaluate_checkpoint(args):
    """Evaluate a selected checkpoint without any training or selection path."""
    validate_args(args)
    cache_path = Path(args.test_cache).resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"locked test cache does not exist: {cache_path}")
    cache_sha256_before = _sha256_file(cache_path)
    test_data, cache_status, cache_validation = load_or_generate_locked_test_data(
        args, allow_generate=False
    )

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    checkpoint_sha256_before = _sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_metadata, expected_model_config = _assert_checkpoint_compatible(
        checkpoint, args
    )

    set_global_seed(args.test_seed, deterministic=args.deterministic)
    model = build_model(args.model, _model_args(args))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    evaluations = [
        _evaluate_one_horizon(model, test_data, args, horizon, device)
        for horizon in requested_eval_horizons(args)
    ]

    checkpoint_sha256_after = _sha256_file(checkpoint_path)
    cache_sha256_after = _sha256_file(cache_path)
    if checkpoint_sha256_before != checkpoint_sha256_after:
        raise RuntimeError(
            "checkpoint changed during checkpoint-only evaluation: "
            f"before={checkpoint_sha256_before}, after={checkpoint_sha256_after}"
        )
    if cache_sha256_before != cache_sha256_after:
        raise RuntimeError(
            "locked test cache changed during checkpoint-only evaluation: "
            f"before={cache_sha256_before}, after={cache_sha256_after}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = requested_eval_horizons(args)
    common_result = {
        "experiment": "fisher_kpp_locked_checkpoint_evaluation",
        "git_commit": _git_commit(),
        "protocol": {
            "checkpoint_only": True,
            "training": False,
            "optimization": False,
            "checkpoint_selection": False,
            "test_cache_generation": False,
            "test_cache_overwrite": False,
            "multi_horizon_strategy": "frozen_cache_prefixes",
        },
        "model": args.model,
        "model_config": expected_model_config,
        "parameter_count": parameter_count,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256_before,
            "sha256_before": checkpoint_sha256_before,
            "sha256_after": checkpoint_sha256_after,
            "immutable_during_evaluation": True,
            "epoch": checkpoint.get("epoch"),
            "best_val_mse": checkpoint.get("best_val_mse"),
            "training_data_seed": checkpoint_metadata.get("data_seed"),
            "training_cache_sha256": checkpoint_metadata.get("provenance", {}).get(
                "data_cache_sha256"
            ),
            "protocol_eval_horizon": checkpoint_eval_horizon(args),
        },
        "locked_test": {
            "cache": str(cache_path),
            "sha256": cache_sha256_before,
            "sha256_before": cache_sha256_before,
            "sha256_after": cache_sha256_after,
            "immutable_during_evaluation": True,
            "cache_status": cache_status,
            "config": locked_test_config(args),
            "validation": cache_validation,
            "independent_from_checkpoint_training_data": (
                int(checkpoint_metadata["data_seed"]) != int(args.test_seed)
            ),
        },
        "environment": _device_provenance(device),
        "source_hashes": _source_hashes(),
    }
    if len(evaluations) == 1:
        evaluation = evaluations[0]
        result = {
            **common_result,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_mode": "checkpoint_only_locked_test",
            "evaluation": evaluation["evaluation"],
            "metrics": evaluation["metrics"],
            "details": evaluation["details"],
        }
    else:
        total_seconds = sum(item["evaluation"]["seconds"] for item in evaluations)
        peak_memories = [
            item["evaluation"]["peak_memory_bytes"]
            for item in evaluations
            if item["evaluation"]["peak_memory_bytes"] is not None
        ]
        result = {
            **common_result,
            "schema_version": MULTI_HORIZON_EVALUATION_SCHEMA_VERSION,
            "evaluation_mode": "checkpoint_only_locked_test_multi_horizon",
            "evaluation_horizons": list(horizons),
            "evaluation": {
                "tau": evaluation_tau(args),
                "horizons": list(horizons),
                "seconds": total_seconds,
                "peak_memory_bytes": max(peak_memories) if peak_memories else None,
            },
            "evaluations": evaluations,
            "metrics_by_horizon": {
                horizon_key(item["horizon"]): item["metrics"] for item in evaluations
            },
            "details_by_horizon": {
                horizon_key(item["horizon"]): item["details"] for item in evaluations
            },
        }
    _write_json(output_dir / "summary.json", result)
    _write_metrics_csv(output_dir / "metrics.csv", evaluations)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_sha256": checkpoint_sha256_before,
                "locked_test_cache_sha256": cache_sha256_before,
                "horizons": [
                    {
                        "horizon": item["horizon"],
                        "rollout_mse_mean": item["metrics"].get("rollout_mse_mean"),
                        "physical_energy_mono_frac_mean": item["metrics"].get(
                            "physical_energy_mono_frac_mean"
                        ),
                    }
                    for item in evaluations
                ],
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )
    return result


def prepare_locked_test_data(args):
    """Freeze or validate a test cache without reading any checkpoint."""
    validate_args(args)
    _, cache_status, validation = load_or_generate_locked_test_data(
        args, allow_generate=True
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": _evaluation_schema_version(args),
        "experiment": "fisher_kpp_locked_checkpoint_evaluation",
        "mode": "prepare_locked_test_data_only",
        "git_commit": _git_commit(),
        "locked_test": {
            "cache": str(Path(args.test_cache).resolve()),
            "sha256": _sha256_file(args.test_cache),
            "cache_status": cache_status,
            "config": locked_test_config(args),
            "validation": validation,
            "requested_horizons": list(requested_eval_horizons(args)),
        },
        "source_hashes": _source_hashes(),
    }
    _write_json(output_dir / "test_data_provenance.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True))
    return payload


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.prepare_test_data_only:
        prepare_locked_test_data(args)
    else:
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
