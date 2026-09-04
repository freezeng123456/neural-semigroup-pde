#!/usr/bin/env python3
"""Checkpoint-only Fisher query-time and composition diagnostics.

This evaluator is deliberately separate from the formal Fisher runner and
the locked-test evaluator.  It consumes an already selected
``latent_query_time`` checkpoint and an already frozen Fisher test cache.  It
never trains, selects a checkpoint, generates a cache, or writes to an input
checkpoint/cache.

The diagnostic has two parts:

* ``query_time_ablation`` evaluates the same physical rollout with the
  normal query-time feature, a feature fixed to ``0.1``, and a deterministic
  shuffled feature.  The latter assigns a reproducibly shuffled pool of
  positive query times to test samples while the physical evolution lag and
  references remain unchanged.
* ``composition_stress`` compares one total-horizon call with compositions
  made from equal partitions and seeded Dirichlet random unequal partitions.
  The required segment counts are 2, 4, 8, and 16.  Both the ordinary
  fixed-per-call RK4 budget and an equal-total-work direct path are recorded.

Every generated artifact is explicitly exploratory and carries
``do_not_use_for_formal: true``.  The output root must be separate from
formal/locked/results/checkpoint roots.  Existing inputs are hashed before
and after evaluation and are never opened for writing.

Example (all inputs must already exist)::

    python experiments/evaluate_fisher_mechanism.py \
        --checkpoint /path/to/latent_query_time_best.pt \
        --test-cache /path/to/frozen_fisher_test.pt \
        --output-dir /tmp/fisher-mechanism-diagnostics \
        --device cpu --deterministic

The default horizons match the formal Fisher cache horizon prefixes, but this
entry point is exploratory and its outputs must not be pooled with formal
decision artifacts.
"""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from models import QueryTimeConditionedLatentFlow, _broadcast_positive_tau  # noqa: E402


EXPLORATORY_SCHEMA_VERSION = 1
EXPLORATORY_TRACK = "fisher_query_time_mechanism_diagnostics"
EVALUATION_MODE = "checkpoint_only_fisher_mechanism_diagnostics"
MODEL_NAME = "latent_query_time"
FORMAL_LOCKED_TEST_SCHEMA_VERSION = 1
FORMAL_SOURCE_COMMIT = "637345584dc2db8ddccf9116a995615c3c036104"
FORMAL_SOURCE_ARCHIVE_SHA256 = (
    "a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5"
)
FORMAL_LOCKED_CACHE_SHA256 = (
    "29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e"
)
REQUIRED_SEGMENT_COUNTS = (2, 4, 8, 16)
DEFAULT_QUERY_TAUS = (0.075, 0.15)
DEFAULT_QUERY_HORIZONS = (1.2, 2.4, 4.8)
DEFAULT_TOTAL_HORIZONS = (1.2, 2.4, 4.8)
DEFAULT_SHUFFLE_POOL = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
DEFAULT_FIXED_QUERY_TIME = 0.1
DEFAULT_SHUFFLE_SEED = 1729
DEFAULT_PARTITION_SEED = 2718
DEFAULT_UNEQUAL_PARTITIONS = 3


def _parse_positive_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive finite floats"
        ) from exc
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "all values must be finite and strictly positive"
        )
    return values


def _parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        raw = tuple(item.strip() for item in value.split(",") if item.strip())
        values = tuple(int(item) for item in raw)
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError(
            "all values must be strictly positive integers"
        )
    return values


def parse_checkpoint_spec(value: str) -> dict[str, Any]:
    """Parse the repeatable ``seed:model:path`` checkpoint identity."""

    if not isinstance(value, str):
        raise argparse.ArgumentTypeError(
            "checkpoint spec must have the form seed:model:path"
        )
    seed_text, separator, remainder = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError(
            "checkpoint spec must have the form seed:model:path"
        )
    model_text, separator, path_text = remainder.partition(":")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError(
            "checkpoint spec must have the form seed:model:path"
        )
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("checkpoint spec seed must be an integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("checkpoint spec seed must be non-negative")
    if model_text != MODEL_NAME:
        raise argparse.ArgumentTypeError(
            f"mechanism diagnostics currently require model={MODEL_NAME}"
        )
    return {"seed": seed, "model": model_text, "path": path_text}


def build_parser() -> argparse.ArgumentParser:
    """Build the explicitly exploratory, checkpoint-only CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--checkpoint",
        help="one already selected latent_query_time *_best.pt checkpoint; never modified",
    )
    checkpoint_group.add_argument(
        "--checkpoint-spec",
        action="append",
        type=parse_checkpoint_spec,
        dest="checkpoint_specs",
        help="repeatable seed:latent_query_time:path identity; all checkpoints are read-only",
    )
    parser.add_argument(
        "--test-cache",
        required=True,
        help="already frozen Fisher test cache; it is read-only and never generated",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="new exploratory output root, separate from formal/results/checkpoint roots",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--beta-v-floor", type=float, default=None)
    parser.add_argument("--reference-dt", type=float, default=None)
    parser.add_argument("--expected-source-commit", default=FORMAL_SOURCE_COMMIT)
    parser.add_argument(
        "--expected-source-archive-sha256",
        default=FORMAL_SOURCE_ARCHIVE_SHA256,
    )
    parser.add_argument(
        "--source-archive",
        default=None,
        help=(
            "frozen source.tar.gz used by the checkpoint run; its SHA-256 is "
            "recomputed before and after diagnostics"
        ),
    )
    parser.add_argument("--expected-cache-sha256", default=FORMAL_LOCKED_CACHE_SHA256)
    parser.add_argument(
        "--query-taus",
        "--taus",
        dest="query_taus",
        type=_parse_positive_float_list,
        default=DEFAULT_QUERY_TAUS,
        help="physical rollout lags for the query-time ablation",
    )
    parser.add_argument(
        "--query-horizons",
        dest="query_horizons",
        type=_parse_positive_float_list,
        default=None,
        help="reference-aligned horizons for the query-time ablation",
    )
    parser.add_argument(
        "--total-horizons",
        dest="total_horizons",
        type=_parse_positive_float_list,
        default=None,
        help="total physical horizons for partition composition stress",
    )
    parser.add_argument(
        "--horizons",
        dest="diagnostic_horizons",
        type=_parse_positive_float_list,
        default=None,
        help="set both query-ablation and composition horizons",
    )
    parser.add_argument(
        "--segment-counts",
        type=_parse_positive_int_list,
        default=REQUIRED_SEGMENT_COUNTS,
        help="must include the required 2,4,8,16 segment stress cases",
    )
    parser.add_argument(
        "--fixed-query-time",
        type=float,
        default=DEFAULT_FIXED_QUERY_TIME,
        help="query-time feature used by the fixed control (default: 0.1)",
    )
    parser.add_argument(
        "--shuffle-pool",
        type=_parse_positive_float_list,
        default=DEFAULT_SHUFFLE_POOL,
        help="positive query-time values shuffled across samples",
    )
    parser.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    parser.add_argument("--partition-seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument(
        "--n-unequal-partitions",
        type=int,
        default=DEFAULT_UNEQUAL_PARTITIONS,
        help="seeded random unequal partitions per horizon/segment count",
    )
    parser.add_argument(
        "--ode-steps",
        type=int,
        default=None,
        help="optional temporary base RK4 count; default is checkpoint model.ode_steps",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="optional prefix for a CPU smoke; default uses every cached test sample",
    )
    return parser


def checkpoint_specs_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return normalized checkpoint identities from either CLI spelling."""

    specs = getattr(args, "checkpoint_specs", None)
    if specs:
        return [dict(spec) for spec in specs]
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        return [{"seed": None, "model": MODEL_NAME, "path": checkpoint}]
    raise ValueError("one --checkpoint or at least one --checkpoint-spec is required")


def normalize_diagnostic_args(args: argparse.Namespace) -> None:
    """Fill the long-form horizon aliases used by the implementation."""

    alias_horizons = getattr(args, "diagnostic_horizons", None)
    explicit_query = getattr(args, "query_horizons", None)
    explicit_total = getattr(args, "total_horizons", None)
    if alias_horizons is not None and (explicit_query is not None or explicit_total is not None):
        if (
            explicit_query is None
            or explicit_total is None
            or tuple(explicit_query) != tuple(alias_horizons)
            or tuple(explicit_total) != tuple(alias_horizons)
        ):
            raise ValueError(
                "--horizons cannot be combined with --query-horizons or --total-horizons"
            )
    if alias_horizons is not None:
        args.query_horizons = tuple(alias_horizons)
        args.total_horizons = tuple(alias_horizons)
        # Make normalization idempotent for per-checkpoint runs in the
        # repeatable --checkpoint-spec path.
        args.diagnostic_horizons = None
    else:
        if explicit_query is None:
            args.query_horizons = DEFAULT_QUERY_HORIZONS
        if explicit_total is None:
            args.total_horizons = DEFAULT_TOTAL_HORIZONS


_FORBIDDEN_OUTPUT_COMPONENTS = {
    "formal",
    "locked",
    "results",
    "results-recovery",
    "results_recovery",
    "checkpoints",
    "source.tar.gz",
}


def _resolve_path(path: os.PathLike[str] | str, label: str) -> Path:
    # Source archives are valid read-only inputs; the output-root guard below
    # separately rejects an output path named ``source.tar.gz``.
    return Path(path).expanduser().resolve()


def _assert_new_output_root(
    output_dir: os.PathLike[str] | str,
    *,
    input_paths: Sequence[os.PathLike[str] | str] = (),
) -> Path:
    """Reject formal roots and roots that could overlap an input artifact."""

    raw_output = Path(output_dir).expanduser()
    if raw_output.is_symlink():
        raise ValueError(
            "output directory must be a new ordinary directory, not a symlink: "
            f"{raw_output}"
        )
    resolved = _resolve_path(output_dir, "output directory")
    parts = {part.lower() for part in resolved.parts}
    if parts.intersection(_FORBIDDEN_OUTPUT_COMPONENTS):
        raise ValueError(
            "output directory must be a new exploratory root, not a "
            f"formal/locked/results/checkpoint path: {resolved}"
        )
    for raw_input in input_paths:
        input_path = _resolve_path(raw_input, "input")
        if resolved == input_path:
            raise ValueError("output directory may not equal an input artifact")
        if resolved in input_path.parents or input_path in resolved.parents:
            raise ValueError(
                "output directory must not be an ancestor/descendant of an input "
                f"artifact: output={resolved}, input={input_path}"
            )
    if resolved.exists():
        # Every invocation gets a fresh canonical root.  Refusing reuse also
        # prevents an old receipt/result symlink from redirecting an atomic
        # output write into a checkpoint, cache, or another run.
        raise ValueError(
            "output directory must be a new exploratory root; refusing to reuse "
            f"existing path: {resolved}"
        )
    return resolved


def _sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _runtime_source_hashes() -> dict[str, str]:
    paths = {
        "evaluate_fisher_mechanism.py": Path(__file__).resolve(),
        "models.py": EXPERIMENTS_DIR / "models.py",
        "evaluate.py": EXPERIMENTS_DIR / "evaluate.py",
        "run_fisher_fair.py": EXPERIMENTS_DIR / "run_fisher_fair.py",
        "pde_solver.py": EXPERIMENTS_DIR / "pde_solver.py",
        "seed_utils.py": EXPERIMENTS_DIR / "seed_utils.py",
    }
    return {
        name: _sha256_file(path)
        for name, path in paths.items()
        if path.is_file()
    }


def _device_provenance(device: torch.device) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device": str(device),
        "gpu": gpu,
    }


def _json_safe(value: Any) -> Any:
    """Convert tensors/numpy values and non-finite floats to strict JSON."""

    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _write_json(path: os.PathLike[str] | str, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_metrics_csv(path: os.PathLike[str] | str, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "diagnostic",
        "control",
        "query_tau",
        "horizon",
        "segment_count",
        "partition_type",
        "ordering",
        "partition_id",
        "work_mode",
        "metric",
        "value",
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: _json_safe(row.get(key))
                        for key in fieldnames
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _duration_steps(duration: float, reference_dt: float) -> int:
    ratio = float(duration) / float(reference_dt)
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(ratio, steps, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError(
            "duration must be a positive integer multiple of reference_dt; "
            f"duration={duration}, reference_dt={reference_dt}, ratio={ratio:.17g}"
        )
    return steps


def validate_args(args: argparse.Namespace) -> None:
    normalize_diagnostic_args(args)
    if args.N is not None and int(args.N) <= 1:
        raise ValueError("N must be greater than one when supplied")
    if args.beta_v_floor is not None and (
        not math.isfinite(float(args.beta_v_floor)) or float(args.beta_v_floor) < 0
    ):
        raise ValueError("beta-v-floor must be finite and non-negative")
    if args.reference_dt is not None and (
        not math.isfinite(float(args.reference_dt)) or float(args.reference_dt) <= 0
    ):
        raise ValueError("reference-dt must be finite and strictly positive")
    if not math.isfinite(float(args.fixed_query_time)) or args.fixed_query_time <= 0:
        raise ValueError("fixed-query-time must be finite and strictly positive")
    if int(args.shuffle_seed) < 0 or int(args.partition_seed) < 0:
        raise ValueError("shuffle-seed and partition-seed must be non-negative")
    if int(args.n_unequal_partitions) <= 0:
        raise ValueError("n-unequal-partitions must be positive")
    for name in (
        "expected_source_commit",
        "expected_source_archive_sha256",
        "expected_cache_sha256",
    ):
        value = getattr(args, name, None)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    for name in ("expected_source_archive_sha256", "expected_cache_sha256"):
        value = getattr(args, name)
        if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
            raise ValueError(f"{name} must be a 64-character SHA-256 digest")
    if args.ode_steps is not None and int(args.ode_steps) <= 0:
        raise ValueError("ode-steps must be positive when supplied")
    if args.max_samples is not None and int(args.max_samples) <= 0:
        raise ValueError("max-samples must be positive when supplied")
    if len(tuple(args.shuffle_pool)) < 2:
        raise ValueError("shuffle-pool must contain at least two values")
    segment_counts = tuple(int(item) for item in args.segment_counts)
    if len(set(segment_counts)) != len(segment_counts):
        raise ValueError("segment-counts must not contain duplicates")
    missing = sorted(set(REQUIRED_SEGMENT_COUNTS).difference(segment_counts))
    if missing:
        raise ValueError(
            "composition stress must include segment counts 2, 4, 8, and 16; "
            f"missing={missing}"
        )
    for name in ("query_taus", "query_horizons", "total_horizons"):
        values = tuple(float(item) for item in getattr(args, name))
        if not values or any(not math.isfinite(item) or item <= 0 for item in values):
            raise ValueError(f"{name} must contain finite positive values")
    checkpoint_paths = [spec["path"] for spec in checkpoint_specs_from_args(args)]
    if not args.source_archive:
        raise ValueError(
            "--source-archive is required so the frozen source archive hash can be verified"
        )
    source_archive = Path(args.source_archive).expanduser().resolve()
    if not source_archive.is_file():
        raise FileNotFoundError(f"source archive does not exist: {source_archive}")
    _assert_new_output_root(
        args.output_dir,
        input_paths=(*checkpoint_paths, args.test_cache, source_archive),
    )


def _cache_reference_dt(cache_config: Mapping[str, Any], args: argparse.Namespace) -> float:
    configured = cache_config.get("reference_dt")
    if args.reference_dt is not None:
        requested = float(args.reference_dt)
        if configured is not None and not math.isclose(
            requested, float(configured), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "--reference-dt does not match the frozen cache: "
                f"{requested} != {configured}"
            )
        return requested
    if configured is None:
        raise ValueError("reference_dt is missing from the frozen cache")
    reference_dt = float(configured)
    if not math.isfinite(reference_dt) or reference_dt <= 0:
        raise ValueError("frozen cache reference_dt must be finite and positive")
    return reference_dt


def validate_frozen_test_cache(
    data: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Validate an existing formal Fisher cache without creating or changing it."""

    if not isinstance(data, Mapping):
        raise ValueError("frozen Fisher test cache must be a mapping")
    if data.get("schema_version") != FORMAL_LOCKED_TEST_SCHEMA_VERSION:
        raise ValueError("frozen Fisher test cache schema version does not match")
    config = data.get("locked_test_config")
    if not isinstance(config, Mapping):
        raise ValueError("frozen Fisher test cache has no locked_test_config")
    if config.get("pde") != "fisher-kpp" or config.get("split") != "locked_independent_test":
        raise ValueError("input cache is not the frozen Fisher locked test split")
    expected_identity = {
        "test_seed": 314163,
        "L": 10.0,
        "nu": 0.1,
        "reaction_rate": 1.0,
        "fixed_tau": 0.1,
    }
    for name, expected in expected_identity.items():
        actual = config.get(name)
        if isinstance(expected, float):
            if actual is None or not math.isclose(
                float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"frozen cache {name} does not match the locked Fisher identity: "
                    f"{actual!r} != {expected!r}"
                )
        elif actual != expected:
            raise ValueError(
                f"frozen cache {name} does not match the locked Fisher identity: "
                f"{actual!r} != {expected!r}"
            )
    reference_dt = _cache_reference_dt(config, args)
    cache_horizon = float(config.get("eval_horizon", float("nan")))
    if not math.isfinite(cache_horizon) or cache_horizon <= 0:
        raise ValueError("frozen cache eval_horizon must be finite and positive")
    test_u0 = data.get("test_u0")
    trajectories = data.get("test_trajs")
    if not torch.is_tensor(test_u0) or test_u0.ndim != 2:
        raise ValueError("frozen cache must contain a rank-2 test_u0 tensor")
    cached_samples, n_sites = (int(test_u0.shape[0]), int(test_u0.shape[1]))
    if cached_samples <= 0 or n_sites <= 1:
        raise ValueError("frozen cache has no usable test samples/sites")
    if args.N is not None and int(args.N) != n_sites:
        raise ValueError(f"--N does not match frozen cache: {args.N} != {n_sites}")
    if not torch.isfinite(test_u0).all():
        raise ValueError("frozen cache test_u0 contains non-finite values")
    if not isinstance(trajectories, (list, tuple)) or len(trajectories) != cached_samples:
        raise ValueError("frozen cache trajectory count does not match test_u0")
    reference_steps = _duration_steps(cache_horizon, reference_dt)
    expected_times = torch.arange(reference_steps + 1, dtype=torch.float64) * reference_dt
    for index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, (list, tuple)) or len(trajectory) != 2:
            raise ValueError(f"frozen trajectory {index} must be a (times, states) pair")
        times, states = trajectory
        if not torch.is_tensor(times) or not torch.is_tensor(states):
            raise ValueError(f"frozen trajectory {index} must contain tensors")
        if tuple(times.shape) != (reference_steps + 1,):
            raise ValueError(f"frozen trajectory {index} has an invalid time shape")
        if tuple(states.shape) != (reference_steps + 1, n_sites):
            raise ValueError(f"frozen trajectory {index} has an invalid state shape")
        if not torch.isfinite(times).all() or not torch.isfinite(states).all():
            raise ValueError(f"frozen trajectory {index} contains non-finite values")
        if not torch.allclose(times.to(torch.float64), expected_times, rtol=1e-6, atol=1e-7):
            raise ValueError(f"frozen trajectory {index} has a mismatched time grid")
        if not torch.allclose(states[0], test_u0[index], rtol=1e-6, atol=1e-6):
            raise ValueError(f"frozen trajectory {index} does not start at test_u0")
    max_samples = cached_samples if args.max_samples is None else int(args.max_samples)
    if max_samples > cached_samples:
        raise ValueError(
            f"max-samples={max_samples} exceeds frozen cache samples={cached_samples}"
        )
    for horizon in tuple(args.query_horizons) + tuple(args.total_horizons):
        if float(horizon) > cache_horizon + 1e-10:
            raise ValueError(
                f"requested horizon {horizon} exceeds frozen cache horizon {cache_horizon}"
            )
        _duration_steps(float(horizon), reference_dt)
    for tau in args.query_taus:
        _duration_steps(float(tau), reference_dt)
        for horizon in args.query_horizons:
            _duration_steps(float(horizon), float(tau))
    return {
        "schema_version": FORMAL_LOCKED_TEST_SCHEMA_VERSION,
        "cache_horizon": cache_horizon,
        "reference_dt": reference_dt,
        "reference_steps": reference_steps,
        "cached_samples": cached_samples,
        "evaluated_samples": max_samples,
        "N": n_sites,
        "test_seed": config.get("test_seed"),
        "config": dict(config),
    }


def load_frozen_test_cache(
    path: os.PathLike[str] | str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    cache_path = _resolve_path(path, "test cache")
    if not cache_path.is_file():
        raise FileNotFoundError(
            "checkpoint-only mechanism evaluator refuses to generate a missing "
            f"test cache: {cache_path}"
        )
    before = _sha256_file(cache_path)
    try:
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        # SCNet's pinned torch 1.12 runtime does not yet expose the keyword.
        if "weights_only" not in str(exc):
            raise
        data = torch.load(cache_path, map_location="cpu")
    expected_sha = getattr(args, "expected_cache_sha256", None)
    if expected_sha is not None and before != expected_sha:
        raise ValueError(
            "frozen Fisher test cache SHA-256 differs from the expected input: "
            f"actual={before}, expected={expected_sha}"
        )
    validation = validate_frozen_test_cache(data, args)
    return dict(data), validation, before


def _checkpoint_kwargs(
    metadata: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    model_config = metadata.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint has no model_config for reconstruction")
    if model_config.get("class") != "QueryTimeConditionedLatentFlow":
        raise ValueError("checkpoint is not a QueryTimeConditionedLatentFlow checkpoint")
    kwargs = model_config.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise ValueError("checkpoint model_config has no kwargs")
    kwargs = dict(kwargs)
    required = {
        "N",
        "hidden_V",
        "hidden_K",
        "stencil_radius",
        "interaction_radius",
        "beta_V",
        "beta_V_floor",
    }
    missing = sorted(required.difference(kwargs))
    if missing:
        raise ValueError(f"checkpoint model_config is missing fields: {missing}")
    if args.N is not None and int(kwargs["N"]) != int(args.N):
        raise ValueError(f"--N does not match checkpoint model_config: {args.N}")
    if args.beta_v_floor is not None and not math.isclose(
        float(kwargs["beta_V_floor"]), float(args.beta_v_floor), rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("--beta-v-floor does not match checkpoint model_config")
    return kwargs


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
) -> tuple[Mapping[str, Any], dict[str, Any], dict[str, str]]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint has no model_state_dict")
    metadata = checkpoint.get("run_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint has no run_metadata for reconstruction")
    if metadata.get("model") != MODEL_NAME:
        raise ValueError(
            f"checkpoint model must be {MODEL_NAME!r}, got {metadata.get('model')!r}"
        )
    kwargs = _checkpoint_kwargs(metadata, args)
    expected_source_hashes = _runtime_source_hashes()
    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no provenance/source hashes")
    expected_commit = getattr(args, "expected_source_commit", None)
    recorded_commit = provenance.get("git_commit")
    if expected_commit is not None and recorded_commit != expected_commit:
        raise ValueError(
            "checkpoint source commit differs from the frozen input: "
            f"actual={recorded_commit!r}, expected={expected_commit!r}"
        )
    recorded_source_hashes = provenance.get("source_hashes")
    if not isinstance(recorded_source_hashes, Mapping):
        raise ValueError("checkpoint provenance has no source hashes")
    required_runtime_sources = {
        "models.py",
        "evaluate.py",
        "run_fisher_fair.py",
        "pde_solver.py",
        "seed_utils.py",
    }
    missing_sources = sorted(required_runtime_sources.difference(recorded_source_hashes))
    if missing_sources:
        raise ValueError(
            "checkpoint provenance is incomplete for evaluator reconstruction: "
            f"{missing_sources}"
        )
    mismatches = {
        name: {
            "checkpoint": recorded_source_hashes.get(name),
            "evaluator": expected_source_hashes.get(name),
        }
        for name in required_runtime_sources
        if recorded_source_hashes.get(name) != expected_source_hashes.get(name)
    }
    if mismatches:
        raise ValueError(
            "checkpoint source hashes differ from evaluator runtime: "
            + json.dumps(mismatches, sort_keys=True)
        )
    expected_count = sum(
        parameter.numel()
        for parameter in QueryTimeConditionedLatentFlow(**kwargs).parameters()
    )
    if int(metadata.get("parameter_count", -1)) != expected_count:
        raise ValueError(
            "checkpoint parameter_count does not match reconstructed model: "
            f"{metadata.get('parameter_count')} != {expected_count}"
        )
    return metadata, kwargs, dict(expected_source_hashes)


def load_checkpoint(
    path: os.PathLike[str] | str,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, Mapping[str, Any], dict[str, Any], str]:
    checkpoint_path = _resolve_path(path, "checkpoint")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    before = _sha256_file(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata, kwargs, source_hashes = _validate_checkpoint(checkpoint, args)
    model = QueryTimeConditionedLatentFlow(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, metadata, {
        "model_config": {"class": "QueryTimeConditionedLatentFlow", "kwargs": kwargs},
        "source_hashes": source_hashes,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_mse": checkpoint.get("best_val_mse"),
        "recorded_source_hashes": dict(metadata["provenance"]["source_hashes"]),
    }, before


def _interaction_matrix(model: torch.nn.Module) -> torch.Tensor:
    if hasattr(model, "_interaction_matrix"):
        return model._interaction_matrix()  # type: ignore[attr-defined]
    if not hasattr(model, "emb") or not hasattr(model, "interaction_mask"):
        raise TypeError("query-time latent model lacks interaction parameters")
    a_full = F.softplus(torch.mm(model.emb, model.emb.t()))  # type: ignore[attr-defined]
    return a_full * model.interaction_mask  # type: ignore[attr-defined]


@torch.no_grad()
def forward_with_query_condition(
    model: torch.nn.Module,
    u: torch.Tensor,
    evolution_tau: float | torch.Tensor,
    conditioning_tau: float | torch.Tensor,
    *,
    ode_steps: int,
) -> torch.Tensor:
    """Integrate for ``evolution_tau`` while exposing another query-time value.

    ``QueryTimeConditionedLatentFlow.forward`` couples these two values.  The
    evaluator intentionally separates them without changing the model or
    checkpoint, so fixed/shuffled controls can keep physical duration and
    reference alignment unchanged.
    """

    if int(ode_steps) <= 0:
        raise ValueError("ode_steps must be positive")
    z = model.encode(u)
    a_ij = _interaction_matrix(model)
    evolution = _broadcast_positive_tau(evolution_tau, z)[:, 0, 0]
    condition = _broadcast_positive_tau(conditioning_tau, z)[:, 0, 0]
    dt = (evolution / int(ode_steps)).reshape(-1, 1)
    for _ in range(int(ode_steps)):
        z = model.rk4_step(z, dt, condition, a_ij)  # type: ignore[attr-defined]
    return model.decode(z)


def deterministic_shuffled_query_times(
    count: int,
    pool: Sequence[float],
    seed: int,
    *,
    stream: int = 0,
) -> tuple[np.ndarray, str]:
    """Return a deterministic repeated-pool permutation and its content hash."""

    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    pool_array = np.asarray(tuple(float(item) for item in pool), dtype="<f8")
    if pool_array.ndim != 1 or pool_array.size < 2:
        raise ValueError("pool must contain at least two values")
    if not np.isfinite(pool_array).all() or (pool_array <= 0).any():
        raise ValueError("pool must contain finite positive values")
    repeated = np.resize(pool_array, count)
    generator = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(stream)])
    )
    values = repeated[generator.permutation(count)]
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    return values, digest


def _control_condition(
    control: str,
    *,
    evolution_tau: float,
    count: int,
    fixed_query_time: float,
    shuffle_pool: Sequence[float],
    shuffle_seed: int,
    stream: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float | torch.Tensor, str | None]:
    if control == "normal":
        return evolution_tau, None
    if control == "fixed_0.1":
        return fixed_query_time, None
    if control != "shuffled":
        raise ValueError(f"unknown query-time control: {control}")
    values, digest = deterministic_shuffled_query_times(
        count, shuffle_pool, shuffle_seed, stream=stream
    )
    return torch.as_tensor(values, device=device, dtype=dtype), digest


@torch.no_grad()
def _apply_control(
    model: torch.nn.Module,
    state: torch.Tensor,
    evolution_tau: float,
    *,
    control: str,
    fixed_query_time: float,
    shuffle_pool: Sequence[float],
    shuffle_seed: int,
    stream: int,
    ode_steps: int,
) -> tuple[torch.Tensor, str | None]:
    conditioning, digest = _control_condition(
        control,
        evolution_tau=evolution_tau,
        count=int(state.shape[0]),
        fixed_query_time=fixed_query_time,
        shuffle_pool=shuffle_pool,
        shuffle_seed=shuffle_seed,
        stream=stream,
        device=state.device,
        dtype=state.dtype,
    )
    output = forward_with_query_condition(
        model,
        state,
        evolution_tau,
        conditioning,
        ode_steps=ode_steps,
    )
    return output, digest


def _distribution(values: torch.Tensor | np.ndarray | Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("diagnostic metric contains non-finite values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _difference_metrics(composed: torch.Tensor, direct: torch.Tensor) -> dict[str, Any]:
    difference = (composed - direct).reshape(composed.shape[0], -1)
    direct_flat = direct.reshape(direct.shape[0], -1)
    mse = difference.square().mean(dim=1)
    absolute_l2 = torch.linalg.vector_norm(difference, dim=1)
    relative_l2 = absolute_l2 / torch.clamp(
        torch.linalg.vector_norm(direct_flat, dim=1), min=1e-12
    )
    return {
        "mse": _distribution(mse.detach().cpu()),
        "absolute_l2": _distribution(absolute_l2.detach().cpu()),
        "relative_l2": _distribution(relative_l2.detach().cpu()),
    }


@torch.no_grad()
def evaluate_query_time_ablation(
    model: torch.nn.Module,
    test_u0: torch.Tensor,
    reference_states: torch.Tensor,
    *,
    reference_dt: float,
    query_taus: Sequence[float],
    query_horizons: Sequence[float],
    fixed_query_time: float,
    shuffle_pool: Sequence[float],
    shuffle_seed: int,
    ode_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controls = ("normal", "fixed_0.1", "shuffled")
    rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    for tau_index, tau in enumerate(query_taus):
        tau = float(tau)
        reference_stride = _duration_steps(tau, reference_dt)
        for horizon_index, horizon in enumerate(query_horizons):
            horizon = float(horizon)
            rollout_steps = _duration_steps(horizon, tau)
            for control_index, control in enumerate(controls):
                state = test_u0
                per_step_mse: list[torch.Tensor] = []
                per_step_relative: list[torch.Tensor] = []
                shuffle_digests: list[str] = []
                for step_index in range(rollout_steps):
                    stream = (
                        1000003 * (tau_index + 1)
                        + 10007 * (horizon_index + 1)
                        + 101 * (control_index + 1)
                        + step_index
                    )
                    state, digest = _apply_control(
                        model,
                        state,
                        tau,
                        control=control,
                        fixed_query_time=fixed_query_time,
                        shuffle_pool=shuffle_pool,
                        shuffle_seed=shuffle_seed,
                        stream=stream,
                        ode_steps=ode_steps,
                    )
                    if digest is not None:
                        shuffle_digests.append(digest)
                    reference = reference_states[:, (step_index + 1) * reference_stride]
                    diff = (state - reference).reshape(state.shape[0], -1)
                    ref_flat = reference.reshape(reference.shape[0], -1)
                    per_step_mse.append(diff.square().mean(dim=1))
                    per_step_relative.append(
                        torch.linalg.vector_norm(diff, dim=1)
                        / torch.clamp(torch.linalg.vector_norm(ref_flat, dim=1), min=1e-12)
                    )
                sample_mse = torch.stack(per_step_mse, dim=0).mean(dim=0)
                sample_relative = torch.stack(per_step_relative, dim=0).mean(dim=0)
                metrics = {
                    "rollout_mse": _distribution(sample_mse.detach().cpu()),
                    "rollout_relative_l2": _distribution(sample_relative.detach().cpu()),
                    "per_step_mse_mean": [
                        float(item.mean().item()) for item in per_step_mse
                    ],
                    "per_step_relative_l2_mean": [
                        float(item.mean().item()) for item in per_step_relative
                    ],
                }
                evaluation = {
                    "control": control,
                    "query_tau": tau,
                    "horizon": horizon,
                    "rollout_steps": rollout_steps,
                    "reference_stride": reference_stride,
                    "ode_steps_per_call": int(ode_steps),
                    "fixed_query_time": fixed_query_time if control == "fixed_0.1" else None,
                    "shuffle_pool": list(shuffle_pool) if control == "shuffled" else None,
                    "shuffle_seed": int(shuffle_seed) if control == "shuffled" else None,
                    "shuffle_schedule_sha256_by_step": shuffle_digests,
                    "metrics": metrics,
                }
                evaluations.append(evaluation)
                for metric_name, distribution in (
                    ("rollout_mse", metrics["rollout_mse"]),
                    ("rollout_relative_l2", metrics["rollout_relative_l2"]),
                ):
                    for statistic, value in distribution.items():
                        rows.append(
                            {
                                "diagnostic": "query_time_ablation",
                                "control": control,
                                "query_tau": tau,
                                "horizon": horizon,
                                "metric": f"{metric_name}.{statistic}",
                                "value": value,
                            }
                        )
    return {
        "controls": list(controls),
        "fixed_query_time": fixed_query_time,
        "shuffle_pool": list(shuffle_pool),
        "shuffle_seed": int(shuffle_seed),
        "semantics": (
            "evolution_tau determines physical integration and reference alignment; "
            "fixed/shuffled controls change only the query-time feature supplied to "
            "the mobility stencil"
        ),
        "evaluations": evaluations,
    }, rows


def equal_partition(total_horizon: float, segment_count: int) -> tuple[float, ...]:
    total_horizon = float(total_horizon)
    segment_count = int(segment_count)
    if total_horizon <= 0 or segment_count <= 0:
        raise ValueError("total_horizon and segment_count must be positive")
    return tuple(total_horizon / segment_count for _ in range(segment_count))


def deterministic_unequal_partition(
    total_horizon: float,
    segment_count: int,
    seed: int,
    *,
    trial: int,
) -> tuple[float, ...]:
    if total_horizon <= 0 or segment_count <= 0 or trial < 0:
        raise ValueError("invalid random partition arguments")
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [int(seed), int(trial), int(segment_count), int(round(total_horizon * 1e9))]
        )
    )
    weights = generator.dirichlet(np.ones(int(segment_count), dtype=np.float64))
    partition = tuple(float(total_horizon * weight) for weight in weights)
    if not all(value > 0 for value in partition):
        raise RuntimeError("Dirichlet partition unexpectedly contained a non-positive segment")
    return partition


def _partition_entry(
    partition: Sequence[float],
    *,
    partition_type: str,
    ordering: str,
    partition_id: str,
    total_horizon: float,
    segment_count: int,
) -> dict[str, Any]:
    values = tuple(float(item) for item in partition)
    if len(values) != int(segment_count):
        raise ValueError("partition length does not match segment_count")
    if not math.isclose(sum(values), float(total_horizon), rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("partition segments do not sum to total_horizon")
    return {
        "partition_id": partition_id,
        "partition_type": partition_type,
        "ordering": ordering,
        "total_horizon": float(total_horizon),
        "segment_count": int(segment_count),
        "segments": list(values),
    }


@torch.no_grad()
def _composition_case(
    model: torch.nn.Module,
    test_u0: torch.Tensor,
    partition_entry: Mapping[str, Any],
    *,
    control: str,
    work_mode: str,
    base_ode_steps: int,
    fixed_query_time: float,
    shuffle_pool: Sequence[float],
    shuffle_seed: int,
    stream_seed: int,
) -> dict[str, Any]:
    total_horizon = float(partition_entry["total_horizon"])
    segment_count = int(partition_entry["segment_count"])
    segments = tuple(float(item) for item in partition_entry["segments"])
    direct_ode_steps = (
        int(base_ode_steps)
        if work_mode == "fixed_per_call"
        else int(base_ode_steps) * segment_count
    )
    direct, direct_digest = _apply_control(
        model,
        test_u0,
        total_horizon,
        control=control,
        fixed_query_time=fixed_query_time,
        shuffle_pool=shuffle_pool,
        shuffle_seed=shuffle_seed,
        stream=stream_seed,
        ode_steps=direct_ode_steps,
    )
    state = test_u0
    composed_digests: list[str] = []
    for index, segment in enumerate(segments):
        state, digest = _apply_control(
            model,
            state,
            segment,
            control=control,
            fixed_query_time=fixed_query_time,
            shuffle_pool=shuffle_pool,
            shuffle_seed=shuffle_seed,
            stream=stream_seed + index + 1,
            ode_steps=int(base_ode_steps),
        )
        if digest is not None:
            composed_digests.append(digest)
    result = {
        **dict(partition_entry),
        "control": control,
        "work_mode": work_mode,
        "base_ode_steps": int(base_ode_steps),
        "direct_ode_steps": direct_ode_steps,
        "composed_ode_steps": [int(base_ode_steps)] * segment_count,
        "direct_total_ode_steps": direct_ode_steps,
        "composed_total_ode_steps": int(base_ode_steps) * segment_count,
        "fixed_query_time": fixed_query_time if control == "fixed_0.1" else None,
        "shuffle_pool": list(shuffle_pool) if control == "shuffled" else None,
        "shuffle_seed": int(shuffle_seed) if control == "shuffled" else None,
        "direct_shuffle_schedule_sha256": direct_digest,
        "composed_shuffle_schedule_sha256_by_segment": composed_digests,
        "metrics": _difference_metrics(state, direct),
    }
    return result


def evaluate_composition_stress(
    model: torch.nn.Module,
    test_u0: torch.Tensor,
    *,
    total_horizons: Sequence[float],
    segment_counts: Sequence[int],
    n_unequal_partitions: int,
    partition_seed: int,
    fixed_query_time: float,
    shuffle_pool: Sequence[float],
    shuffle_seed: int,
    base_ode_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controls = ("normal", "fixed_0.1", "shuffled")
    work_modes = ("fixed_per_call", "equal_total_rk4_work")
    rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    partition_manifest: list[dict[str, Any]] = []
    for horizon_index, total_horizon in enumerate(total_horizons):
        for count_index, segment_count in enumerate(segment_counts):
            equal = _partition_entry(
                equal_partition(total_horizon, segment_count),
                partition_type="equal",
                ordering="forward",
                partition_id=f"equal_h{horizon_index}_s{count_index}",
                total_horizon=total_horizon,
                segment_count=segment_count,
            )
            entries = [equal]
            for trial in range(int(n_unequal_partitions)):
                random_partition = deterministic_unequal_partition(
                    total_horizon,
                    segment_count,
                    partition_seed,
                    trial=trial,
                )
                random_entry = _partition_entry(
                    random_partition,
                    partition_type="unequal_random",
                    ordering="forward",
                    partition_id=f"unequal_h{horizon_index}_s{count_index}_r{trial}",
                    total_horizon=total_horizon,
                    segment_count=segment_count,
                )
                entries.append(random_entry)
                entries.append(
                    _partition_entry(
                        tuple(reversed(random_partition)),
                        partition_type="unequal_random",
                        ordering="reversed",
                        partition_id=f"unequal_h{horizon_index}_s{count_index}_r{trial}_reversed",
                        total_horizon=total_horizon,
                        segment_count=segment_count,
                    )
                )
            partition_manifest.extend(entries)
            for partition_index, partition_entry in enumerate(entries):
                for control_index, control in enumerate(controls):
                    for work_index, work_mode in enumerate(work_modes):
                        case = _composition_case(
                            model,
                            test_u0,
                            partition_entry,
                            control=control,
                            work_mode=work_mode,
                            base_ode_steps=base_ode_steps,
                            fixed_query_time=fixed_query_time,
                            shuffle_pool=shuffle_pool,
                            shuffle_seed=shuffle_seed,
                            stream_seed=(
                                1000003 * (horizon_index + 1)
                                + 10007 * (count_index + 1)
                                + 101 * (partition_index + 1)
                                + 17 * (control_index + 1)
                                + work_index
                            ),
                        )
                        evaluations.append(case)
                        for metric_name, distribution in case["metrics"].items():
                            for statistic, value in distribution.items():
                                rows.append(
                                    {
                                        "diagnostic": "composition_stress",
                                        "control": control,
                                        "horizon": total_horizon,
                                        "segment_count": segment_count,
                                        "partition_type": case["partition_type"],
                                        "ordering": case["ordering"],
                                        "partition_id": case["partition_id"],
                                        "work_mode": work_mode,
                                        "metric": f"{metric_name}.{statistic}",
                                        "value": value,
                                    }
                                )
    return {
        "controls": list(controls),
        "work_modes": list(work_modes),
        "required_segment_counts": list(REQUIRED_SEGMENT_COUNTS),
        "segment_counts": [int(item) for item in segment_counts],
        "total_horizons": [float(item) for item in total_horizons],
        "n_unequal_partitions_per_case": int(n_unequal_partitions),
        "partition_seed": int(partition_seed),
        "partitions": partition_manifest,
        "semantics": {
            "fixed_per_call": (
                "direct uses base_ode_steps for one total-horizon call; each "
                "composed segment also uses base_ode_steps"
            ),
            "equal_total_rk4_work": (
                "direct uses segment_count*base_ode_steps while each composed "
                "segment uses base_ode_steps; total RK4 work is equal"
            ),
            "query_time_control": (
                "normal uses each physical duration; fixed_0.1 holds the feature "
                "at 0.1; shuffled uses independent deterministic sample schedules"
            ),
        },
        "evaluations": evaluations,
    }, rows


def _flatten_input_rows(
    query_rows: Sequence[Mapping[str, Any]],
    composition_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [dict(row) for row in (*query_rows, *composition_rows)]


def _validate_input_unchanged(
    path: Path,
    before: str,
    label: str,
) -> str:
    after = _sha256_file(path)
    if before != after:
        raise RuntimeError(
            f"{label} changed during checkpoint-only evaluation: "
            f"before={before}, after={after}"
        )
    return after


def evaluate_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    """Run both diagnostics and write only exploratory outputs."""

    validate_args(args)
    output_dir = _assert_new_output_root(
        args.output_dir,
        input_paths=(args.checkpoint, args.test_cache),
    )
    # Reserve the fresh root before loading or evaluating any input.  This
    # makes the canonical-root guarantee explicit and prevents a later write
    # from following a path that appeared after validation.
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = _resolve_path(args.checkpoint, "checkpoint")
    cache_path = _resolve_path(args.test_cache, "test cache")
    source_archive_path = _resolve_path(args.source_archive, "source archive")
    source_archive_sha_before = _sha256_file(source_archive_path)
    if source_archive_sha_before != args.expected_source_archive_sha256:
        raise ValueError(
            "frozen source archive SHA-256 differs from the expected input: "
            f"actual={source_archive_sha_before}, expected={args.expected_source_archive_sha256}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    test_data, cache_validation, cache_sha_before = load_frozen_test_cache(
        cache_path, args
    )
    model, checkpoint_metadata, checkpoint_info, checkpoint_sha_before = load_checkpoint(
        checkpoint_path, args
    )
    model = model.to(device).eval()
    evaluated_samples = int(cache_validation["evaluated_samples"])
    test_u0 = test_data["test_u0"][:evaluated_samples].to(device)
    reference_states = torch.stack(
        [trajectory[1] for trajectory in test_data["test_trajs"][:evaluated_samples]]
    ).to(device)
    base_ode_steps = int(args.ode_steps if args.ode_steps is not None else model.ode_steps)
    if base_ode_steps <= 0:
        raise ValueError("checkpoint model.ode_steps must be positive")

    query_result, query_rows = evaluate_query_time_ablation(
        model,
        test_u0,
        reference_states,
        reference_dt=float(cache_validation["reference_dt"]),
        query_taus=args.query_taus,
        query_horizons=args.query_horizons,
        fixed_query_time=float(args.fixed_query_time),
        shuffle_pool=args.shuffle_pool,
        shuffle_seed=int(args.shuffle_seed),
        ode_steps=base_ode_steps,
    )
    composition_result, composition_rows = evaluate_composition_stress(
        model,
        test_u0,
        total_horizons=args.total_horizons,
        segment_counts=args.segment_counts,
        n_unequal_partitions=int(args.n_unequal_partitions),
        partition_seed=int(args.partition_seed),
        fixed_query_time=float(args.fixed_query_time),
        shuffle_pool=args.shuffle_pool,
        shuffle_seed=int(args.shuffle_seed),
        base_ode_steps=base_ode_steps,
    )

    checkpoint_sha_after = _validate_input_unchanged(
        checkpoint_path, checkpoint_sha_before, "checkpoint"
    )
    cache_sha_after = _validate_input_unchanged(
        cache_path, cache_sha_before, "test cache"
    )
    source_archive_sha_after = _validate_input_unchanged(
        source_archive_path, source_archive_sha_before, "source archive"
    )
    source_hashes = _runtime_source_hashes()
    protocol = {
        "checkpoint_only": True,
        "training": False,
        "optimization": False,
        "checkpoint_selection": False,
        "test_cache_generation": False,
        "test_cache_overwrite": False,
        "input_checkpoint_write": False,
        "input_cache_write": False,
        "gradients": False,
    }
    common = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "evaluation_mode": EVALUATION_MODE,
        "checkpoint_spec": getattr(args, "checkpoint_spec", None),
        "protocol": protocol,
        "git_commit": _git_commit(),
        "source_commit": getattr(args, "expected_source_commit", None),
        "source_archive_sha256": getattr(args, "expected_source_archive_sha256", None),
        "source_archive_sha256_before": source_archive_sha_before,
        "source_archive_sha256_after": source_archive_sha_after,
        "source_hashes": source_hashes,
        "environment": _device_provenance(device),
        "config": vars(args),
        "inputs": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256_before": checkpoint_sha_before,
                "sha256_after": checkpoint_sha_after,
                "immutable_during_evaluation": True,
                "epoch": checkpoint_info["checkpoint_epoch"],
                "best_val_mse": checkpoint_info["best_val_mse"],
                "model": MODEL_NAME,
                "model_config": checkpoint_info["model_config"],
                "checkpoint_source_hashes": checkpoint_info["recorded_source_hashes"],
                "training_data_cache_sha256": checkpoint_metadata.get(
                    "provenance", {}
                ).get("data_cache_sha256"),
            },
            "test_cache": {
                "path": str(cache_path),
                "sha256_before": cache_sha_before,
                "expected_sha256": getattr(args, "expected_cache_sha256", None),
                "sha256_after": cache_sha_after,
                "immutable_during_evaluation": True,
                "validation": cache_validation,
            },
            "source_archive": {
                "path": str(source_archive_path),
                "sha256_before": source_archive_sha_before,
                "sha256_after": source_archive_sha_after,
                "immutable_during_evaluation": True,
            },
        },
        "scope_guard": {
            "formal_protocol_modified": False,
            "formal_cache_modified": False,
            "formal_checkpoint_modified": False,
            "locked_cache_generated_or_modified": False,
            "source_archive_modified": False,
            "results_recovery_modified": False,
            "scnet_or_t4_login": False,
        },
        "sample_count": evaluated_samples,
        "base_ode_steps": base_ode_steps,
        "query_time_ablation": query_result,
        "composition_stress": composition_result,
    }
    manifest = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "evaluation_mode": EVALUATION_MODE,
        "checkpoint_spec": getattr(args, "checkpoint_spec", None),
        "created_at_unix": time.time(),
        "protocol": protocol,
        "config": vars(args),
        "input_hashes_before": {
            "checkpoint": checkpoint_sha_before,
            "test_cache": cache_sha_before,
            "source_archive": source_archive_sha_before,
        },
        "source_hashes": source_hashes,
        "source_commit": getattr(args, "expected_source_commit", None),
        "source_archive_sha256": getattr(args, "expected_source_archive_sha256", None),
        "expected_cache_sha256": getattr(args, "expected_cache_sha256", None),
        "source_archive_sha256_before": source_archive_sha_before,
        "source_archive_sha256_after": source_archive_sha_after,
        "scope_guard": common["scope_guard"],
    }
    _write_json(output_dir / "exploratory_manifest.json", manifest)
    _write_json(output_dir / "results.json", common)
    _write_metrics_csv(
        output_dir / "metrics.csv",
        _flatten_input_rows(query_rows, composition_rows),
    )
    results_sha = _sha256_file(output_dir / "results.json")
    receipt = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "evaluation_mode": EVALUATION_MODE,
        "status": "passed",
        "normal_exit": True,
        "protocol": protocol,
        "results_sha256": results_sha,
        "metrics_csv_sha256": _sha256_file(output_dir / "metrics.csv"),
        "input_hashes": {
            "checkpoint": {
                "before": checkpoint_sha_before,
                "after": checkpoint_sha_after,
                "unchanged": checkpoint_sha_before == checkpoint_sha_after,
            },
            "test_cache": {
                "before": cache_sha_before,
                "after": cache_sha_after,
                "unchanged": cache_sha_before == cache_sha_after,
            },
            "source_archive": {
                "before": source_archive_sha_before,
                "after": source_archive_sha_after,
                "unchanged": source_archive_sha_before == source_archive_sha_after,
            },
        },
        "expected_controls": ["normal", "fixed_0.1", "shuffled"],
        "expected_segment_counts": list(REQUIRED_SEGMENT_COUNTS),
        "source_commit": getattr(args, "expected_source_commit", None),
        "source_archive_sha256": getattr(args, "expected_source_archive_sha256", None),
        "expected_cache_sha256": getattr(args, "expected_cache_sha256", None),
        "source_hashes": source_hashes,
        "scope_guard": common["scope_guard"],
    }
    _write_json(output_dir / "receipt.json", receipt)
    (output_dir / "done").write_text("passed\n", encoding="utf-8")
    return common


def _args_for_checkpoint(
    base_args: argparse.Namespace,
    spec: Mapping[str, Any],
    output_dir: Path,
) -> argparse.Namespace:
    child = copy.deepcopy(base_args)
    child.checkpoint = str(spec["path"])
    child.checkpoint_specs = None
    child.checkpoint_spec = dict(spec)
    child.output_dir = str(output_dir)
    return child


def _write_multi_checkpoint_root(
    output_dir: Path,
    base_args: argparse.Namespace,
    specs: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write a marked aggregate root for repeatable checkpoint specs."""

    # ``main`` reserves the aggregate root before running child evaluators so
    # a failure leaves an inspectable, unique partial root.  The children
    # create their own directories below it; this writer must therefore
    # validate the reserved root rather than attempting to create it again.
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValueError(
            "aggregate output directory must be a reserved ordinary directory: "
            f"{output_dir}"
        )
    expected_children = {
        f"seed_{spec['seed']}_{spec['model']}" for spec in specs
    }
    actual_children = {
        child.name
        for child in output_dir.iterdir()
        if child.is_dir() and not child.is_symlink()
    }
    unexpected_children = sorted(actual_children.difference(expected_children))
    missing_children = sorted(expected_children.difference(actual_children))
    if unexpected_children or missing_children:
        raise ValueError(
            "aggregate output directory does not contain exactly the completed "
            f"checkpoint roots: unexpected={unexpected_children}, "
            f"missing={missing_children}"
        )
    source_hashes = _runtime_source_hashes()
    protocol = {
        "checkpoint_only": True,
        "training": False,
        "optimization": False,
        "checkpoint_selection": False,
        "test_cache_generation": False,
        "test_cache_overwrite": False,
        "input_checkpoint_write": False,
        "input_cache_write": False,
        "gradients": False,
    }
    scope_guard = {
        "formal_protocol_modified": False,
        "formal_cache_modified": False,
        "formal_checkpoint_modified": False,
        "locked_cache_generated_or_modified": False,
        "source_archive_modified": False,
        "results_recovery_modified": False,
        "scnet_or_t4_login": False,
    }
    spec_keys = [
        f"seed_{spec['seed']}_{spec['model']}"
        for spec in specs
    ]
    aggregate = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "evaluation_mode": EVALUATION_MODE,
        "source_commit": getattr(base_args, "expected_source_commit", None),
        "source_archive_sha256": getattr(base_args, "expected_source_archive_sha256", None),
        "expected_cache_sha256": getattr(base_args, "expected_cache_sha256", None),
        "checkpoint_specs": [dict(spec) for spec in specs],
        "protocol": protocol,
        "config": vars(base_args),
        "source_hashes": source_hashes,
        "scope_guard": scope_guard,
        "results": {
            key: result for key, result in zip(spec_keys, results)
        },
    }
    all_inputs = {
        key: {
            "checkpoint": result["inputs"]["checkpoint"],
            "test_cache": result["inputs"]["test_cache"],
        }
        for key, result in zip(spec_keys, results)
    }
    aggregate["input_hashes"] = all_inputs
    manifest = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "evaluation_mode": EVALUATION_MODE,
        "source_commit": getattr(base_args, "expected_source_commit", None),
        "source_archive_sha256": getattr(base_args, "expected_source_archive_sha256", None),
        "expected_cache_sha256": getattr(base_args, "expected_cache_sha256", None),
        "created_at_unix": time.time(),
        "protocol": protocol,
        "config": vars(base_args),
        "checkpoint_specs": [dict(spec) for spec in specs],
        "input_hashes_before": all_inputs,
        "source_hashes": source_hashes,
        "scope_guard": scope_guard,
    }
    _write_json(output_dir / "exploratory_manifest.json", manifest)
    _write_json(output_dir / "results.json", aggregate)
    # Keep an aggregate, machine-readable metrics table alongside the
    # per-checkpoint roots.  The nested results remain the authoritative
    # detail; this table is only a convenience index and never feeds formal
    # selection.
    aggregate_rows: list[dict[str, Any]] = []
    for spec, result in zip(specs, results):
        child_key = f"seed_{spec['seed']}_{spec['model']}"
        for section_name, section in (
            ("query_time_ablation", result["query_time_ablation"]),
            ("composition_stress", result["composition_stress"]),
        ):
            for evaluation in section["evaluations"]:
                aggregate_rows.append(
                    {
                        "diagnostic": section_name,
                        "checkpoint": child_key,
                        "control": evaluation.get("control"),
                        "query_tau": evaluation.get("query_tau"),
                        "horizon": evaluation.get("horizon", evaluation.get("total_horizon")),
                        "segment_count": evaluation.get("segment_count"),
                        "partition_type": evaluation.get("partition_type"),
                        "ordering": evaluation.get("ordering"),
                        "partition_id": evaluation.get("partition_id"),
                        "work_mode": evaluation.get("work_mode"),
                        "metrics": evaluation.get("metrics"),
                    }
                )
    _write_json(output_dir / "metrics_index.json", {"rows": aggregate_rows})
    result_sha = _sha256_file(output_dir / "results.json")
    receipt = {
        "schema_version": EXPLORATORY_SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": EXPLORATORY_TRACK,
        "evaluation_mode": EVALUATION_MODE,
        "source_commit": getattr(base_args, "expected_source_commit", None),
        "source_archive_sha256": getattr(base_args, "expected_source_archive_sha256", None),
        "expected_cache_sha256": getattr(base_args, "expected_cache_sha256", None),
        "status": "passed",
        "normal_exit": True,
        "protocol": protocol,
        "results_sha256": result_sha,
        "checkpoint_specs": [dict(spec) for spec in specs],
        "scope_guard": scope_guard,
        "source_hashes": source_hashes,
    }
    _write_json(output_dir / "receipt.json", receipt)
    (output_dir / "done").write_text("passed\n", encoding="utf-8")
    return aggregate


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    normalize_diagnostic_args(args)
    specs = checkpoint_specs_from_args(args)
    input_paths = [spec["path"] for spec in specs] + [args.test_cache]
    output_dir = _assert_new_output_root(args.output_dir, input_paths=input_paths)
    if len(specs) == 1:
        result = evaluate_diagnostics(
            _args_for_checkpoint(args, specs[0], output_dir)
        )
        payload = {
            "exploratory": True,
            "evaluation_mode": EVALUATION_MODE,
            "output_dir": str(output_dir),
            "checkpoint_sha256": result["inputs"]["checkpoint"]["sha256_before"],
            "test_cache_sha256": result["inputs"]["test_cache"]["sha256_before"],
            "sample_count": result["sample_count"],
            "controls": result["query_time_ablation"]["controls"],
            "segment_counts": result["composition_stress"]["segment_counts"],
        }
    else:
        # Reserve the aggregate root before child evaluators create their
        # independent roots.  This keeps the canonical-root guarantee and
        # makes the aggregation step compatible with the child write order.
        output_dir.mkdir(parents=True, exist_ok=False)
        child_results = []
        for spec in specs:
            child_name = f"seed_{spec['seed']}_{spec['model']}"
            child_results.append(
                evaluate_diagnostics(
                    _args_for_checkpoint(args, spec, output_dir / child_name)
                )
            )
        result = _write_multi_checkpoint_root(
            output_dir,
            args,
            specs,
            child_results,
        )
        payload = {
            "exploratory": True,
            "evaluation_mode": EVALUATION_MODE,
            "output_dir": str(output_dir),
            "checkpoint_specs": [dict(spec) for spec in specs],
            "sample_count": [item["sample_count"] for item in child_results],
            "controls": child_results[0]["query_time_ablation"]["controls"],
            "segment_counts": child_results[0]["composition_stress"]["segment_counts"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
