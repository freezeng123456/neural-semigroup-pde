#!/usr/bin/env python3
"""Evaluate one frozen Allen--Cahn checkpoint on an independent test set.

This entry point deliberately contains no optimizer or training path. It
creates a separately seeded, immutable test cache in an explicit preparation
stage, then evaluates an already-selected checkpoint against that cache. The
protocol prevents the training/validation data used for checkpoint selection
from silently becoming the reported held-out result.
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
from pde_solver import AllenCahnSolver, generate_allen_cahn_initial_conditions  # noqa: E402
from run_allen_cahn_fair import (  # noqa: E402
    LOWER_BOUND,
    MODEL_NAMES,
    UPPER_BOUND,
    _device_provenance,
    _git_commit,
    _sha256_file,
    _solve_batch_trajectory,
    _source_hashes,
    allen_cahn_free_energy,
    build_model,
    model_config,
    reference_steps_for_duration,
    rollout_steps_for_horizon,
)
from seed_utils import set_global_seed  # noqa: E402


LOCKED_TEST_SCHEMA_VERSION = 1
EVALUATION_SCHEMA_VERSION = 1
DEFAULT_TEST_SEED = 314159
DEFAULT_N_TEST = 500
LATENT_MODELS = {
    "latent",
    "latent_decoded_interaction",
    "latent_decoded_interaction_jacobian_mobility",
    "latent_decoded_energy",
    "latent_periodic_decoded_interaction",
    "latent_physics_anchored_periodic",
}


def build_parser():
    """Build the checkpoint-only locked-test CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        help="selected *_best.pt checkpoint; required unless only preparing the test cache",
    )
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-seed", type=int, default=DEFAULT_TEST_SEED)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--L", type=float, default=2.0 * math.pi)
    parser.add_argument(
        "--eps", "--epsilon", dest="epsilon", type=float, default=0.1
    )
    parser.add_argument("--reference-dt", type=float, default=0.005)
    parser.add_argument("--fixed-tau", type=float, default=0.1)
    parser.add_argument("--eval-horizon", type=float, default=1.2)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument(
        "--beta-v-floor",
        type=float,
        default=0.0,
        help="must exactly match the frozen checkpoint latent potential floor",
    )
    parser.add_argument(
        "--prepare-test-data-only",
        action="store_true",
        help="create or validate the locked test cache, but do not load a checkpoint",
    )
    return parser


def validate_args(args):
    """Validate test-set and model reconstruction invariants."""
    if args.test_seed < 0:
        raise ValueError("test-seed must be non-negative")
    if args.N <= 1:
        raise ValueError("N must be greater than one")
    if args.n_test <= 0:
        raise ValueError("n-test must be positive")
    for name in ("L", "epsilon", "reference_dt", "fixed_tau", "eval_horizon"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if not math.isfinite(args.beta_v_floor) or args.beta_v_floor < 0:
        raise ValueError("beta-v-floor must be finite and non-negative")
    reference_steps_for_duration(args.eval_horizon, args.reference_dt)
    reference_steps_for_duration(args.fixed_tau, args.reference_dt)
    rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)
    if not args.prepare_test_data_only and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --prepare-test-data-only is used")


def locked_test_config(args):
    """Return the complete identity of the independently seeded test split."""
    return {
        "schema_version": LOCKED_TEST_SCHEMA_VERSION,
        "pde": "allen-cahn",
        "split": "locked_independent_test",
        "test_seed": int(args.test_seed),
        "deterministic": bool(args.deterministic),
        "split_policy": "independent_seeded_stream_distinct_from_training_data",
        "initial_condition_policy": "low_frequency_fourier_modes_scaled_to_0.9",
        "N": int(args.N),
        "L": float(args.L),
        "epsilon": float(args.epsilon),
        "reference_dt": float(args.reference_dt),
        "fixed_tau": float(args.fixed_tau),
        "eval_horizon": float(args.eval_horizon),
        "n_test": int(args.n_test),
        "bounds": [LOWER_BOUND, UPPER_BOUND],
    }


def _atomic_torch_save(payload, path):
    """Serialize a cache atomically so partial artifacts are never reused."""
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
    """Write a JSON artifact atomically."""
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


def _write_metrics_csv(path, metrics):
    """Store all scalar and structured metrics in a simple stable table."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("metric", "value"))
        writer.writeheader()
        for name in sorted(metrics):
            value = metrics[name]
            if isinstance(value, (str, int, float, bool)) or value is None:
                rendered = value
            else:
                rendered = json.dumps(value, sort_keys=True, allow_nan=True)
            writer.writerow({"metric": name, "value": rendered})


def _expected_times(n_steps, reference_dt):
    return torch.arange(n_steps + 1, dtype=torch.float64) * float(reference_dt)


def validate_locked_test_cache(data, args):
    """Validate the exact test-cache schema before inference can start."""
    if not isinstance(data, Mapping):
        raise ValueError("locked test cache must be a mapping")
    if data.get("schema_version") != LOCKED_TEST_SCHEMA_VERSION:
        raise ValueError("locked test cache schema version does not match")
    expected_config = locked_test_config(args)
    if data.get("locked_test_config") != expected_config:
        raise ValueError(
            "locked test cache configuration differs from the requested protocol"
        )

    test_u0 = data.get("test_u0")
    test_trajs = data.get("test_trajs")
    if not torch.is_tensor(test_u0) or tuple(test_u0.shape) != (args.n_test, args.N):
        raise ValueError(
            "locked test cache must contain test_u0 with shape "
            f"({args.n_test}, {args.N})"
        )
    if not torch.isfinite(test_u0).all():
        raise ValueError("locked test cache contains non-finite initial states")
    if (
        float(test_u0.min()) < LOWER_BOUND - 1e-4
        or float(test_u0.max()) > UPPER_BOUND + 1e-4
    ):
        raise ValueError("locked test initial states leave the admissible bounds")
    if not isinstance(test_trajs, (list, tuple)) or len(test_trajs) != args.n_test:
        raise ValueError("locked test cache must contain one trajectory per test state")

    n_reference_steps = reference_steps_for_duration(
        args.eval_horizon, args.reference_dt
    )
    expected_times = _expected_times(n_reference_steps, args.reference_dt)
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
        if not torch.allclose(times.to(torch.float64), expected_times, rtol=1e-10, atol=1e-12):
            raise ValueError(f"locked test trajectory {index} has a mismatched time grid")
        if (
            float(states.min()) < LOWER_BOUND - 1e-4
            or float(states.max()) > UPPER_BOUND + 1e-4
        ):
            raise ValueError(f"locked test trajectory {index} leaves the admissible bounds")

    return {
        "test_u0_shape": list(test_u0.shape),
        "test_trajectory_length": n_reference_steps + 1,
        "test_u0_range": [float(test_u0.min()), float(test_u0.max())],
        "test_trajectory_range": [
            float(torch.stack([trajectory[1] for trajectory in test_trajs]).min()),
            float(torch.stack([trajectory[1] for trajectory in test_trajs]).max()),
        ],
    }


def generate_locked_test_data(args):
    """Generate one deterministic, standalone Allen--Cahn test split."""
    set_global_seed(args.test_seed, deterministic=args.deterministic)
    solver = AllenCahnSolver(
        N=args.N,
        L=args.L,
        eps=args.epsilon,
        dt=args.reference_dt,
    )
    test_u0 = generate_allen_cahn_initial_conditions(args.N, args.n_test, args.L)
    n_reference_steps = reference_steps_for_duration(
        args.eval_horizon, args.reference_dt
    )
    times, trajectories = _solve_batch_trajectory(solver, test_u0, n_reference_steps)
    test_trajs = [
        (times.clone(), trajectories[index].clone()) for index in range(args.n_test)
    ]
    return {
        "schema_version": LOCKED_TEST_SCHEMA_VERSION,
        "locked_test_config": locked_test_config(args),
        "test_u0": test_u0,
        "test_trajs": test_trajs,
    }


def load_or_generate_locked_test_data(args, *, allow_generate):
    """Load a validated test cache, creating it only in the explicit prep stage."""
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


def _model_args(args):
    """Return the minimal namespace required to reconstruct a fair-runner model."""
    return SimpleNamespace(N=args.N, beta_v_floor=args.beta_v_floor)


def _assert_checkpoint_compatible(checkpoint, args):
    """Reject a checkpoint whose architecture or PDE protocol does not match."""
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

    expected_model_config = model_config(args.model, _model_args(args))
    if metadata.get("model_config") != expected_model_config:
        raise ValueError("checkpoint model configuration does not match evaluation args")

    expected_numeric = {
        "L": args.L,
        "epsilon": args.epsilon,
        "reference_dt": args.reference_dt,
        "fixed_tau": args.fixed_tau,
        "eval_horizon": args.eval_horizon,
    }
    for name, expected in expected_numeric.items():
        actual = metadata.get(name)
        if actual is None or not math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"checkpoint {name}={actual!r} does not match requested {expected!r}"
            )
    if int(metadata.get("N", -1)) != args.N:
        raise ValueError("checkpoint spatial grid N does not match evaluation args")
    if int(metadata.get("data_seed", -1)) == args.test_seed:
        raise ValueError(
            "locked test seed must differ from the checkpoint training-data seed"
        )

    expected_source_hashes = _source_hashes()
    recorded_source_hashes = metadata.get("provenance", {}).get("source_hashes")
    if not isinstance(recorded_source_hashes, Mapping):
        raise ValueError("checkpoint has no source hashes for compatibility validation")
    source_mismatches = {
        name: {
            "checkpoint": recorded_source_hashes.get(name),
            "evaluator": expected_hash,
        }
        for name, expected_hash in expected_source_hashes.items()
        if recorded_source_hashes.get(name) != expected_hash
    }
    if source_mismatches:
        raise ValueError(
            "checkpoint source hashes differ from the evaluator source: "
            + json.dumps(source_mismatches, sort_keys=True)
        )
    return metadata, expected_model_config


def evaluate_checkpoint(args):
    """Evaluate a frozen checkpoint without modifying its weights or optimizer."""
    validate_args(args)
    test_data, cache_status, cache_validation = load_or_generate_locked_test_data(
        args, allow_generate=False
    )
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_metadata, expected_model_config = _assert_checkpoint_compatible(
        checkpoint, args
    )

    set_global_seed(args.test_seed, deterministic=args.deterministic)
    model = build_model(args.model, _model_args(args))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    physical_energy = lambda state: allen_cahn_free_energy(  # noqa: E731
        state, L=args.L, epsilon=args.epsilon
    )
    rollout_steps = rollout_steps_for_horizon(args.eval_horizon, args.fixed_tau)

    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    evaluation_start = time.perf_counter()
    metrics, details = evaluate_full(
        model,
        test_data["test_u0"],
        test_data["test_trajs"],
        tau=args.fixed_tau,
        rollout_steps=rollout_steps,
        device=device,
        model_name=args.model,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        reference_dt=args.reference_dt,
        physical_energy_fn=physical_energy,
        collect_latent_diagnostics=args.model in LATENT_MODELS,
        equal_work_base_ode_steps=30 if args.model in LATENT_MODELS else None,
    )
    if is_cuda:
        torch.cuda.synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "experiment": "allen_cahn_locked_checkpoint_evaluation",
        "evaluation_mode": "checkpoint_only_locked_test",
        "git_commit": _git_commit(),
        "model": args.model,
        "model_config": expected_model_config,
        "parameter_count": parameter_count,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "epoch": checkpoint.get("epoch"),
            "best_val_mse": checkpoint.get("best_val_mse"),
            "training_data_seed": checkpoint_metadata.get("data_seed"),
            "training_cache_sha256": checkpoint_metadata.get("provenance", {}).get(
                "data_cache_sha256"
            ),
        },
        "locked_test": {
            "cache": str(Path(args.test_cache).resolve()),
            "sha256": _sha256_file(args.test_cache),
            "cache_status": cache_status,
            "config": locked_test_config(args),
            "validation": cache_validation,
            "independent_from_checkpoint_training_data": (
                int(checkpoint_metadata["data_seed"]) != args.test_seed
            ),
        },
        "evaluation": {
            "tau": args.fixed_tau,
            "horizon": args.eval_horizon,
            "rollout_steps": rollout_steps,
            "seconds": evaluation_seconds,
            "peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if is_cuda else None
            ),
        },
        "metrics": metrics,
        "details": details,
        "environment": _device_provenance(device),
        "source_hashes": _source_hashes(),
    }
    _write_json(output_dir / "summary.json", result)
    _write_metrics_csv(output_dir / "metrics.csv", metrics)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "checkpoint_sha256": checkpoint_sha256,
                "locked_test_cache_sha256": result["locked_test"]["sha256"],
                "rollout_mse_mean": metrics.get("rollout_mse_mean"),
                "physical_energy_mono_frac_mean": metrics.get(
                    "physical_energy_mono_frac_mean"
                ),
                "evaluation_seconds": evaluation_seconds,
            },
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )
    return result


def prepare_locked_test_data(args):
    """Freeze the test cache and write a provenance-only preparation artifact."""
    validate_args(args)
    _, cache_status, validation = load_or_generate_locked_test_data(
        args, allow_generate=True
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "experiment": "allen_cahn_locked_checkpoint_evaluation",
        "mode": "prepare_locked_test_data_only",
        "git_commit": _git_commit(),
        "locked_test": {
            "cache": str(Path(args.test_cache).resolve()),
            "sha256": _sha256_file(args.test_cache),
            "cache_status": cache_status,
            "config": locked_test_config(args),
            "validation": validation,
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
