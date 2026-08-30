"""Read-only reconstruction of frozen Fisher checkpoints and test caches.

This adapter is the only new-experiment module that knows the historical
checkpoint/cache schemas.  Numerical evaluators consume the smaller
``FrozenFisherCheckpoint`` and validated cache interfaces instead of copying
schema checks into every launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import torch

try:  # Package import in pytest / module execution.
    from .experiment_artifacts import (
        require_input_file,
        sha256_file,
        source_hashes,
        torch_load_compat,
    )
    from .models import LatentSemigroupNet, QueryTimeConditionedLatentFlow
except ImportError:  # Direct ``python experiments/<script>.py`` execution.
    from experiment_artifacts import (  # type: ignore
        require_input_file,
        sha256_file,
        source_hashes,
        torch_load_compat,
    )
    from models import LatentSemigroupNet, QueryTimeConditionedLatentFlow  # type: ignore


FORMAL_SOURCE_COMMIT = "637345584dc2db8ddccf9116a995615c3c036104"
FORMAL_SOURCE_ARCHIVE_SHA256 = (
    "a370efa4af9bbb11fbcd72ef422410f651f8f2ad28eee561c45e766c88fcadf5"
)
FORMAL_LOCKED_CACHE_SHA256 = (
    "29d0e4b36e9d758f87264ae1d555d865c9e037776903e1cab448a0a8782b490e"
)
FORMAL_CACHE_PROFILE = "formal_locked"
PHASE_CACHE_PROFILE = "exploratory_phase"
MODEL_CLASSES = {
    "latent": ("LatentSemigroupNet", LatentSemigroupNet),
    "latent_query_time": (
        "QueryTimeConditionedLatentFlow",
        QueryTimeConditionedLatentFlow,
    ),
}

EXPERIMENTS_DIR = Path(__file__).resolve().parent
TRAINING_SOURCE_FILES = {
    "run_fisher_fair.py": EXPERIMENTS_DIR / "run_fisher_fair.py",
    "models.py": EXPERIMENTS_DIR / "models.py",
    "training.py": EXPERIMENTS_DIR / "training.py",
    "evaluate.py": EXPERIMENTS_DIR / "evaluate.py",
    "pde_solver.py": EXPERIMENTS_DIR / "pde_solver.py",
    "seed_utils.py": EXPERIMENTS_DIR / "seed_utils.py",
}


def runtime_training_source_hashes() -> dict[str, str]:
    """Hash exactly the sources frozen into formal Fisher checkpoints."""

    return source_hashes(TRAINING_SOURCE_FILES)


def _validate_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


@dataclass(frozen=True)
class FrozenFisherCheckpoint:
    """One reconstructed, validation-selected, read-only Fisher checkpoint."""

    path: Path
    sha256: str
    model_name: str
    seed: int
    model: torch.nn.Module
    metadata: Mapping[str, Any]
    model_config: Mapping[str, Any]
    epoch: int | None
    best_val_mse: float | None
    recorded_source_hashes: Mapping[str, str]

    def provenance(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "model": self.model_name,
            "seed": self.seed,
            "epoch": self.epoch,
            "best_val_mse": self.best_val_mse,
            "parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "model_config": dict(self.model_config),
            "recorded_source_hashes": dict(self.recorded_source_hashes),
            "training_data_cache_sha256": self.metadata.get("provenance", {}).get(
                "data_cache_sha256"
            ),
        }


def load_frozen_fisher_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    model_name: str,
    seed: int,
    expected_source_commit: str = FORMAL_SOURCE_COMMIT,
) -> FrozenFisherCheckpoint:
    """Reconstruct a formal A/B checkpoint and verify its frozen provenance."""

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"unsupported frozen Fisher model: {model_name!r}")
    checkpoint_path = require_input_file(path, "checkpoint")
    expected_digest = _validate_sha256(expected_sha256, "checkpoint SHA-256")
    actual_digest = sha256_file(checkpoint_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "checkpoint SHA-256 differs from the frozen launch input: "
            f"actual={actual_digest}, expected={expected_digest}"
        )
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    state_dict = checkpoint.get("model_state_dict")
    metadata = checkpoint.get("run_metadata")
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint has no model_state_dict")
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint has no run_metadata")
    if metadata.get("model") != model_name:
        raise ValueError(
            f"checkpoint model mismatch: {metadata.get('model')!r} != {model_name!r}"
        )
    if metadata.get("regime") != "variable":
        raise ValueError("frozen Fisher checkpoint must use variable-lag training")
    if metadata.get("architecture_only") is not True:
        raise ValueError("frozen Fisher checkpoint must be architecture-only")
    losses = metadata.get("auxiliary_loss_weights")
    if not isinstance(losses, Mapping) or any(
        float(value) != 0.0 for value in losses.values()
    ):
        raise ValueError("frozen Fisher checkpoint has non-zero auxiliary losses")
    recorded_seed = metadata.get("training_seed", metadata.get("seed"))
    if int(recorded_seed if recorded_seed is not None else -1) != int(seed):
        raise ValueError(
            f"checkpoint training seed mismatch: {recorded_seed!r} != {seed}"
        )

    config = metadata.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint has no model_config")
    expected_class_name, model_class = MODEL_CLASSES[model_name]
    if config.get("class") != expected_class_name:
        raise ValueError(
            f"checkpoint class mismatch: {config.get('class')!r} != {expected_class_name!r}"
        )
    kwargs = config.get("kwargs")
    if not isinstance(kwargs, Mapping):
        raise ValueError("checkpoint model_config has no kwargs")
    required_kwargs = {
        "N",
        "hidden_V",
        "hidden_K",
        "stencil_radius",
        "interaction_radius",
        "beta_V",
        "beta_V_floor",
    }
    missing_kwargs = sorted(required_kwargs.difference(kwargs))
    if missing_kwargs:
        raise ValueError(f"checkpoint model kwargs are incomplete: {missing_kwargs}")

    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint has no provenance")
    if provenance.get("git_commit") != expected_source_commit:
        raise ValueError(
            "checkpoint source commit differs from the frozen protocol: "
            f"{provenance.get('git_commit')!r} != {expected_source_commit!r}"
        )
    recorded_hashes = provenance.get("source_hashes")
    if not isinstance(recorded_hashes, Mapping):
        raise ValueError("checkpoint provenance has no source_hashes")
    runtime_hashes = runtime_training_source_hashes()
    missing_sources = sorted(set(runtime_hashes).difference(recorded_hashes))
    if missing_sources:
        raise ValueError(f"checkpoint provenance is missing sources: {missing_sources}")
    mismatches = {
        name: {"checkpoint": recorded_hashes.get(name), "runtime": digest}
        for name, digest in runtime_hashes.items()
        if recorded_hashes.get(name) != digest
    }
    if mismatches:
        raise ValueError(
            "checkpoint source hashes differ from evaluator runtime: "
            + json.dumps(mismatches, sort_keys=True)
        )

    model = model_class(**dict(kwargs))
    expected_parameters = sum(parameter.numel() for parameter in model.parameters())
    if int(metadata.get("parameter_count", -1)) != expected_parameters:
        raise ValueError(
            "checkpoint parameter_count does not match reconstruction: "
            f"{metadata.get('parameter_count')} != {expected_parameters}"
        )
    model.load_state_dict(state_dict, strict=True)
    epoch = checkpoint.get("epoch")
    best_val_mse = checkpoint.get("best_val_mse")
    return FrozenFisherCheckpoint(
        path=checkpoint_path,
        sha256=actual_digest,
        model_name=model_name,
        seed=int(seed),
        model=model,
        metadata=metadata,
        model_config=config,
        epoch=int(epoch) if epoch is not None else None,
        best_val_mse=float(best_val_mse) if best_val_mse is not None else None,
        recorded_source_hashes={
            str(key): str(value) for key, value in recorded_hashes.items()
        },
    )


def _aligned_steps(duration: float, reference_dt: float, label: str) -> int:
    ratio = float(duration) / float(reference_dt)
    steps = int(round(ratio))
    if (
        not math.isfinite(ratio)
        or steps <= 0
        or not math.isclose(ratio, steps, rel_tol=1e-9, abs_tol=1e-10)
    ):
        raise ValueError(
            f"{label} must be a positive integer multiple of reference_dt: "
            f"ratio={ratio:.17g}"
        )
    return steps


def _cache_contract(profile: str) -> tuple[str, dict[str, Any]]:
    if profile == FORMAL_CACHE_PROFILE:
        return "locked_test_config", {
            "pde": "fisher-kpp",
            "split": "locked_independent_test",
            "test_seed": 314163,
            "N": 64,
            "L": 10.0,
            "nu": 0.1,
            "reaction_rate": 1.0,
            "reference_dt": 0.005,
            "eval_horizon": 4.8,
        }
    if profile == PHASE_CACHE_PROFILE:
        return "exploratory_test_config", {
            "pde": "fisher-kpp",
            "split": "exploratory_semigroup_phase",
            "test_seed": 314164,
            "N": 64,
            "L": 10.0,
            "nu": 0.1,
            "reaction_rate": 1.0,
            "reference_dt": 0.005,
            "eval_horizon": 4.8,
            "n_test": 128,
        }
    raise ValueError(f"unsupported Fisher cache profile: {profile!r}")


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def load_frozen_fisher_cache(
    path: str | Path,
    *,
    expected_sha256: str,
    profile: str,
    taus: Sequence[float],
    horizons: Sequence[float],
    max_samples: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Load and fully validate a formal or preregistered exploratory cache."""

    cache_path = require_input_file(path, "Fisher test cache")
    expected_digest = _validate_sha256(expected_sha256, "cache SHA-256")
    actual_digest = sha256_file(cache_path)
    if actual_digest != expected_digest:
        raise ValueError(
            "Fisher cache SHA-256 differs from the frozen launch input: "
            f"actual={actual_digest}, expected={expected_digest}"
        )
    payload = torch_load_compat(cache_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("Fisher cache payload must be a mapping")
    config_key, identity = _cache_contract(profile)
    config = payload.get(config_key)
    if not isinstance(config, Mapping):
        raise ValueError(f"Fisher cache has no {config_key}")
    for name, expected in identity.items():
        if name == "N" and config.get(name) is None:
            continue
        if not _matches(config.get(name), expected):
            raise ValueError(
                f"Fisher cache {name} mismatch: {config.get(name)!r} != {expected!r}"
            )
    if profile == PHASE_CACHE_PROFILE:
        if (
            payload.get("exploratory") is not True
            or payload.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("phase cache lacks explicit exploratory markers")

    test_u0 = payload.get("test_u0")
    trajectories = payload.get("test_trajs")
    if not torch.is_tensor(test_u0) or test_u0.ndim != 2:
        raise ValueError("Fisher cache test_u0 must be a rank-2 tensor")
    cached_samples, n_sites = map(int, test_u0.shape)
    if n_sites != int(identity["N"]):
        raise ValueError(f"Fisher cache grid mismatch: {n_sites} != {identity['N']}")
    if profile == PHASE_CACHE_PROFILE and cached_samples != int(identity["n_test"]):
        raise ValueError(
            f"phase cache sample count mismatch: {cached_samples} != {identity['n_test']}"
        )
    if (
        not isinstance(trajectories, (list, tuple))
        or len(trajectories) != cached_samples
    ):
        raise ValueError("Fisher cache trajectories do not match test_u0")
    reference_dt = float(identity["reference_dt"])
    cache_horizon = float(identity["eval_horizon"])
    reference_steps = _aligned_steps(cache_horizon, reference_dt, "cache horizon")
    expected_times = (
        torch.arange(reference_steps + 1, dtype=torch.float64) * reference_dt
    )
    finite_min = float(test_u0.min().item())
    finite_max = float(test_u0.max().item())
    if not torch.isfinite(test_u0).all():
        raise ValueError("Fisher cache test_u0 contains non-finite values")
    for index, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, (list, tuple)) or len(trajectory) != 2:
            raise ValueError(f"Fisher trajectory {index} is not a (times, states) pair")
        times, states = trajectory
        if not torch.is_tensor(times) or not torch.is_tensor(states):
            raise ValueError(f"Fisher trajectory {index} does not contain tensors")
        if tuple(times.shape) != (reference_steps + 1,):
            raise ValueError(f"Fisher trajectory {index} has invalid time shape")
        if tuple(states.shape) != (reference_steps + 1, n_sites):
            raise ValueError(f"Fisher trajectory {index} has invalid state shape")
        if not torch.isfinite(times).all() or not torch.isfinite(states).all():
            raise ValueError(f"Fisher trajectory {index} contains non-finite values")
        if not torch.allclose(
            times.to(torch.float64), expected_times, rtol=1e-6, atol=1e-7
        ):
            raise ValueError(f"Fisher trajectory {index} has a mismatched time grid")
        if not torch.allclose(states[0], test_u0[index], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Fisher trajectory {index} does not start at test_u0")
        finite_min = min(finite_min, float(states.min().item()))
        finite_max = max(finite_max, float(states.max().item()))

    evaluated_samples = cached_samples if max_samples is None else int(max_samples)
    if evaluated_samples <= 0 or evaluated_samples > cached_samples:
        raise ValueError(
            f"max_samples must be in [1, {cached_samples}], got {evaluated_samples}"
        )
    normalized_taus = tuple(float(value) for value in taus)
    normalized_horizons = tuple(float(value) for value in horizons)
    for tau in normalized_taus:
        _aligned_steps(tau, reference_dt, "lag")
    for horizon in normalized_horizons:
        if horizon > cache_horizon + 1e-10:
            raise ValueError(
                f"requested horizon {horizon} exceeds cache horizon {cache_horizon}"
            )
        _aligned_steps(horizon, reference_dt, "horizon")

    return (
        dict(payload),
        {
            "profile": profile,
            "config_key": config_key,
            "config": dict(config),
            "cached_samples": cached_samples,
            "evaluated_samples": evaluated_samples,
            "N": n_sites,
            "reference_dt": reference_dt,
            "cache_horizon": cache_horizon,
            "reference_steps": reference_steps,
            "state_min": finite_min,
            "state_max": finite_max,
            "test_u0_dtype": str(test_u0.dtype),
        },
        actual_digest,
    )


def reference_states_at(
    cache: Mapping[str, Any],
    *,
    horizon: float,
    reference_dt: float,
    sample_count: int,
) -> torch.Tensor:
    """Stack exactly one validated trajectory prefix endpoint."""

    index = _aligned_steps(horizon, reference_dt, "horizon")
    trajectories = cache["test_trajs"]
    return torch.stack(
        [trajectories[sample_index][1][index] for sample_index in range(sample_count)]
    )


__all__ = [
    "FORMAL_CACHE_PROFILE",
    "FORMAL_LOCKED_CACHE_SHA256",
    "FORMAL_SOURCE_ARCHIVE_SHA256",
    "FORMAL_SOURCE_COMMIT",
    "FrozenFisherCheckpoint",
    "PHASE_CACHE_PROFILE",
    "load_frozen_fisher_cache",
    "load_frozen_fisher_checkpoint",
    "reference_states_at",
    "runtime_training_source_hashes",
]
