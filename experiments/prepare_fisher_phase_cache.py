#!/usr/bin/env python3
"""Prepare the preregistered exploratory Fisher lag--horizon cache.

This is the only command in the novelty experiment lane that creates input
data.  It always creates one new exploratory root and the fixed seed-314164,
128-sample, horizon-4.8 cache.  Checkpoint evaluators have no cache-generation
path and consume the resulting SHA-256 explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch


EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENTS_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import (  # noqa: E402
    atomic_torch_save,
    atomic_write_json,
    git_commit,
    reserve_exploratory_root,
    sha256_file,
    source_hashes,
)
from fisher_frozen_inputs import (  # noqa: E402
    PHASE_CACHE_PROFILE,
    load_frozen_fisher_cache,
)
from pde_solver import FisherKPPSolver, generate_initial_conditions  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402


SCHEMA_VERSION = 1
TRACK = "fisher_semigroup_phase_cache"


@dataclass(frozen=True)
class FisherPhaseCacheConfig:
    """Frozen data-generation contract from the mechanism preregistration."""

    pde: str = "fisher-kpp"
    split: str = "exploratory_semigroup_phase"
    test_seed: int = 314164
    N: int = 64
    L: float = 10.0
    nu: float = 0.1
    reaction_rate: float = 1.0
    reference_dt: float = 0.005
    eval_horizon: float = 4.8
    n_test: int = 128

    def validate(self) -> None:
        if self.pde != "fisher-kpp" or self.split != "exploratory_semigroup_phase":
            raise ValueError(
                "cache identity must remain the preregistered Fisher phase split"
            )
        if (
            int(self.test_seed) != 314164
            or int(self.N) != 64
            or int(self.n_test) != 128
        ):
            raise ValueError("cache seed, grid, and sample count are frozen")
        expected = {
            "L": (float(self.L), 10.0),
            "nu": (float(self.nu), 0.1),
            "reaction_rate": (float(self.reaction_rate), 1.0),
            "reference_dt": (float(self.reference_dt), 0.005),
            "eval_horizon": (float(self.eval_horizon), 4.8),
        }
        mismatches = {
            name: actual
            for name, (actual, target) in expected.items()
            if not math.isclose(actual, target, rel_tol=1e-12, abs_tol=1e-12)
        }
        if mismatches:
            raise ValueError(f"cache numerical protocol is frozen: {mismatches}")


def _runtime_source_hashes() -> dict[str, str]:
    return source_hashes(
        {
            "prepare_fisher_phase_cache.py": Path(__file__).resolve(),
            "experiment_artifacts.py": EXPERIMENTS_DIR / "experiment_artifacts.py",
            "fisher_frozen_inputs.py": EXPERIMENTS_DIR / "fisher_frozen_inputs.py",
            "pde_solver.py": EXPERIMENTS_DIR / "pde_solver.py",
            "seed_utils.py": EXPERIMENTS_DIR / "seed_utils.py",
        }
    )


def generate_cache(config: FisherPhaseCacheConfig) -> dict[str, Any]:
    """Generate all trajectories in one vectorized spectral time loop."""

    config.validate()
    set_global_seed(config.test_seed, deterministic=True)
    initial_states = generate_initial_conditions(config.N, config.n_test, config.L)
    solver = FisherKPPSolver(
        N=config.N,
        L=config.L,
        nu=config.nu,
        r=config.reaction_rate,
        dt=config.reference_dt,
        dtype=torch.float32,
    )
    ratio = config.eval_horizon / config.reference_dt
    steps = int(round(ratio))
    if not math.isclose(ratio, steps, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("eval_horizon must align with reference_dt")

    times = torch.arange(steps + 1, dtype=torch.float32) * config.reference_dt
    states = torch.empty(
        steps + 1,
        config.n_test,
        config.N,
        dtype=torch.float32,
    )
    states[0] = initial_states
    spectral_state = torch.fft.rfft(initial_states, dim=-1)
    for step in range(1, steps + 1):
        spectral_state = solver.step(spectral_state)
        states[step] = torch.fft.irfft(spectral_state, n=config.N, dim=-1)
        if step % 120 == 0 or step == steps:
            print(
                json.dumps(
                    {
                        "event": "reference_progress",
                        "step": step,
                        "steps": steps,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    trajectories = [
        (times.clone(), states[:, sample_index, :].contiguous())
        for sample_index in range(config.n_test)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "cache_profile": PHASE_CACHE_PROFILE,
        "created_at_unix": time.time(),
        "exploratory_test_config": asdict(config),
        "test_u0": initial_states,
        "test_trajs": trajectories,
    }


def prepare_cache(output_dir: str | Path) -> dict[str, Any]:
    """Create, hash, reload, and validate one immutable exploratory cache."""

    config = FisherPhaseCacheConfig()
    config.validate()
    root = reserve_exploratory_root(output_dir)
    source_digest_map = _runtime_source_hashes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "preparing",
        "created_at_unix": time.time(),
        "git_commit": git_commit(REPOSITORY_ROOT),
        "config": asdict(config),
        "source_hashes": source_digest_map,
        "protocol": {
            "prepare_only": True,
            "training": False,
            "checkpoint_access": False,
            "formal_cache_access": False,
            "overwrite": False,
        },
    }
    atomic_write_json(root / "exploratory_manifest.json", manifest)
    cache_path = root / "inputs" / "fisher_phase_s314164_n128_h4p8.pt"
    started = time.perf_counter()
    cache = generate_cache(config)
    atomic_torch_save(cache, cache_path)
    cache_sha256 = sha256_file(cache_path)
    reloaded, validation, validated_sha256 = load_frozen_fisher_cache(
        cache_path,
        expected_sha256=cache_sha256,
        profile=PHASE_CACHE_PROFILE,
        taus=(0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3),
        horizons=(0.6, 1.2, 2.4, 4.8),
    )
    del reloaded
    elapsed = time.perf_counter() - started
    result = {
        **manifest,
        "status": "passed",
        "normal_exit": True,
        "elapsed_seconds": elapsed,
        "cache": {
            "path": str(cache_path),
            "sha256": cache_sha256,
            "validated_sha256": validated_sha256,
            "byte_size": cache_path.stat().st_size,
            "validation": validation,
        },
    }
    atomic_write_json(root / "results.json", result)
    manifest["status"] = "passed"
    manifest["cache_sha256"] = cache_sha256
    atomic_write_json(root / "exploratory_manifest.json", manifest)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "exploratory": True,
        "do_not_use_for_formal": True,
        "track": TRACK,
        "status": "passed",
        "normal_exit": True,
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "results_sha256": sha256_file(root / "results.json"),
        "manifest_sha256": sha256_file(root / "exploratory_manifest.json"),
        "reload_validation_passed": True,
        "source_hashes": source_digest_map,
    }
    atomic_write_json(root / "receipt.json", receipt)
    (root / "done").write_text("passed\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="brand-new exploratory root; an existing path is always rejected",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = prepare_cache(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cache": result["cache"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
