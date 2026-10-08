#!/usr/bin/env python3
"""Frozen-checkpoint Fisher conditioning and representation controls.

One invocation evaluates one selected A, B, or A+Tube checkpoint. See
``docs/research/FISHER_INRANGE_ATTRIBUTION_USAGE.md`` for frozen input hashes
and commands. This exploratory evaluator does not replace the common-path
vector-field/stability experiment in FISHER_TUBE_ATTRIBUTION_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import types

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.experiment_artifacts import (
    atomic_write_json, device_provenance, require_input_file,
    reserve_exploratory_root, sha256_file, source_hashes,
    torch_load_compat, verify_unchanged,
)
from experiments.fisher_frozen_inputs import (
    FORMAL_CACHE_PROFILE, FORMAL_LOCKED_CACHE_SHA256,
    FORMAL_SOURCE_ARCHIVE_SHA256, FORMAL_SOURCE_COMMIT,
    load_frozen_fisher_cache, reference_states_at,
)
from experiments.latent_integrators import explicit_step, integer_depth, interaction_matrix

LAGS = (0.075, 0.15)
HORIZONS = (1.2, 2.4, 4.8)
SUBSTEPS = {0.075: 30, 0.15: 60}
SEEDS = (31415, 271828, 161803)
TUBE_SOURCE_COMMIT = "abd1c2f2879315cf0321173dc20e53c7fdce0019"
TUBE_ARCHIVE_SHA256 = "62106584600cc6a9caf8c46255f43f59ce066c9c1e5f1270f9f827923de1ab2e"
REQUIRED_SOURCES = {"models.py", "training.py", "run_fisher_fair.py", "evaluate.py", "pde_solver.py", "seed_utils.py"}


def checked_digest(path: Path, expected: str, label: str) -> str:
    expected = str(expected).lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"{label} expected hash must be a SHA-256 digest")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")
    return actual


def archived_sources(path: Path, expected_sha256: str) -> dict[str, bytes]:
    """Read regular source members without extracting any archive path."""
    checked_digest(path, expected_sha256, "source archive")
    sources = {}
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if ".." in parts or Path(member.name).is_absolute():
                raise ValueError("source archive contains an unsafe member path")
            if not member.name.endswith(".py"):
                continue
            if not member.isfile():
                raise ValueError("Python source archive members must be ordinary files")
            if member.size > 10 * 1024 * 1024:
                raise ValueError("source archive member exceeds 10 MiB")
            # Training records source basenames; ignore tests and nested packages.
            if len(parts) > 1 and parts[-2] != "experiments":
                continue
            name = parts[-1]
            if name in sources:
                raise ValueError(f"ambiguous source archive basename: {name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read archive source: {member.name}")
            sources[name] = handle.read()
    if not REQUIRED_SOURCES.issubset(sources):
        raise ValueError(f"source archive lacks required files: {sorted(REQUIRED_SOURCES - sources.keys())}")
    return sources


def load_archived_checkpoint(checkpoint_path: Path, *, expected_sha256: str,
                             source_archive: Path, archive_sha256: str,
                             source_commit: str, role: str, seed: int):
    """Verify frozen training sources before executing the archived model definition."""
    if role not in {"baseline", "query", "tube"}:
        raise ValueError("unknown Fisher checkpoint role")
    checked_digest(checkpoint_path, expected_sha256, "checkpoint")
    sources = archived_sources(source_archive, archive_sha256)
    payload = torch_load_compat(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("run_metadata"), Mapping):
        raise ValueError("checkpoint lacks run_metadata")
    metadata = payload["run_metadata"]
    expected_name = "latent_query_time" if role == "query" else "latent"
    expected_class = "QueryTimeConditionedLatentFlow" if role == "query" else "LatentSemigroupNet"
    if metadata.get("model") != expected_name or metadata.get("regime") != "variable":
        raise ValueError("checkpoint model/regime differs from the frozen role")
    if metadata.get("training_seed", metadata.get("seed")) != seed:
        raise ValueError("checkpoint seed differs from requested seed")
    training_lags = tuple(float(value) for value in metadata.get("train_taus", ()))
    if training_lags != (0.025, 0.05, 0.1, 0.2):
        raise ValueError("checkpoint training lags differ from frozen Fisher support")
    losses = metadata.get("auxiliary_loss_weights")
    if not isinstance(losses, Mapping) or not losses:
        raise ValueError("checkpoint lacks auxiliary loss metadata")
    expected_generator = 0.01 if role == "tube" else 0.0
    if metadata.get("architecture_only") is not (role != "tube"):
        raise ValueError("checkpoint architecture_only differs from role")
    if float(losses.get("alpha_generator", 0.0)) != expected_generator or any(
        not math.isfinite(float(v)) or (k != "alpha_generator" and float(v) != 0.0)
        for k, v in losses.items()
    ):
        raise ValueError("checkpoint auxiliary losses differ from frozen role")
    if role == "tube" and metadata.get("generator_supervision", {}).get("sampling") != "detached_learned_tube":
        raise ValueError("Tube checkpoint lacks detached learned-tube supervision")
    provenance = metadata.get("provenance", {})
    if provenance.get("git_commit") != source_commit:
        raise ValueError("checkpoint source commit mismatch")
    recorded = provenance.get("source_hashes", {})
    if not isinstance(recorded, Mapping) or not REQUIRED_SOURCES.issubset(recorded):
        raise ValueError("checkpoint lacks required source hashes")
    for name, digest in recorded.items():
        if name not in sources or hashlib.sha256(sources[name]).hexdigest() != digest:
            raise ValueError(f"checkpoint source hash mismatch or absent archived file: {name}")
    config = metadata.get("model_config", {})
    if config.get("class") != expected_class or not isinstance(config.get("kwargs"), Mapping):
        raise ValueError("checkpoint model_config differs from frozen role")
    # No ambient models.py import: inference uses exactly the archived bytes.
    module = types.ModuleType("_frozen_fisher_models_" + archive_sha256[:16])
    module.__file__ = str(source_archive) + "::experiments/models.py"
    exec(compile(sources["models.py"], module.__file__, "exec"), module.__dict__)
    model = getattr(module, expected_class)(**dict(config["kwargs"]))
    if sum(p.numel() for p in model.parameters()) != metadata.get("parameter_count"):
        raise ValueError("checkpoint parameter count mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("checkpoint contains non-finite parameters or buffers")
    return model.eval(), {
        "role": role, "seed": seed, "model": expected_name, "training_lags": training_lags,
        "epoch": payload.get("epoch"), "best_val_mse": payload.get("best_val_mse"),
        "model_config": dict(config), "training_source_commit": source_commit,
        "training_source_hashes_verified": dict(recorded),
        "model_definition": module.__file__, "training": False,
    }


def pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict:
    if first.shape != second.shape or first.ndim != 2 or not first.numel():
        raise ValueError("metrics require equal nonempty [sample, site] tensors")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise RuntimeError("non-finite state in evaluation")
    error = (first - second).double()
    mse = error.square().mean(dim=1)
    relative = error.norm(dim=1) / second.double().norm(dim=1).clamp_min(1e-12)
    return {"mse_mean": mse.mean().item(), "relative_l2_mean": relative.mean().item(),
            "per_sample_mse": mse.cpu().tolist(), "per_sample_relative_l2": relative.cpu().tolist()}


@torch.no_grad()
def trajectory(model, initial: torch.Tensor, *, lag: float, horizons: Sequence[float],
               substeps: int, conditioning_time: float, reencode: bool = True,
               clamp: bool = True) -> tuple[dict[float, torch.Tensor], dict]:
    """Trace RK4, optionally exposing call-boundary representation changes."""
    if substeps <= 0 or not math.isfinite(conditioning_time) or conditioning_time <= 0:
        raise ValueError("positive substeps and conditioning time are required")
    horizon_depths = {integer_depth(float(h), lag): float(h) for h in horizons}
    if len(horizon_depths) != len(horizons) or not horizon_depths:
        raise ValueError("horizons must be nonempty and unique")
    for name in ("encode", "decode", "latent_dynamics"):
        if not callable(getattr(model, name, None)):
            raise TypeError(f"latent instrumentation unavailable: model lacks {name}")
    latent = model.encode(initial)
    initial_projection = pair_metrics(model.decode(latent), initial)
    interactions = interaction_matrix(model)
    condition = latent.new_full((len(initial),), conditioning_time)
    if getattr(model, "latent_dynamics_requires_tau", False):
        rhs = lambda z: model.latent_dynamics(z, interactions, tau=condition)
    else:
        rhs = lambda z: model.latent_dynamics(z, interactions)
    dt = latent.new_tensor(lag / substeps)
    counters = {"latent_clipped_entries": 0, "latent_step_entries": 0,
                "reencode_boundaries": 0, "roundtrip_latent_abs_max": 0.0,
                "encoder_clipped_entries": 0 if hasattr(model, "eps") else None,
                "initial_projection": initial_projection}
    snapshots = {}
    for depth in range(1, max(horizon_depths) + 1):
        if depth > 1 and reencode:
            physical = model.decode(latent)
            encoded = model.encode(physical)
            counters["reencode_boundaries"] += 1
            counters["roundtrip_latent_abs_max"] = max(counters["roundtrip_latent_abs_max"],
                                                       (encoded - latent).abs().max().item())
            if counters["encoder_clipped_entries"] is not None:
                eps = float(model.eps)
                counters["encoder_clipped_entries"] += int(((physical < eps) | (physical > 1 - eps)).sum())
            latent = encoded
        for _ in range(substeps):
            latent = explicit_step(rhs, latent, dt, "rk4")
            if not torch.isfinite(latent).all():
                raise RuntimeError("non-finite latent state in RK4 trajectory")
            counters["latent_step_entries"] += latent.numel()
            if clamp:
                counters["latent_clipped_entries"] += int(((latent < -20) | (latent > 20)).sum())
                latent = latent.clamp(-20, 20)
        if depth in horizon_depths:
            snapshots[horizon_depths[depth]] = model.decode(latent).clone()
    counters["encoder_clip_diagnostic"] = "available" if hasattr(model, "eps") else "unavailable: no model.eps"
    return snapshots, counters


@torch.no_grad()
def evaluate_model(model, initial, references, *, lags=LAGS, horizons=HORIZONS,
                   substeps=None) -> dict:
    substeps = SUBSTEPS if substeps is None else substeps
    if len(lags) != 2 or len(set(lags)) != 2:
        raise ValueError("two distinct lags are required")
    if not math.isclose(lags[0] / substeps[lags[0]], lags[1] / substeps[lags[1]], rel_tol=1e-12):
        raise ValueError("cross-lag evaluation must use equal RK4 work")
    results, composed_states = {}, {}
    for lag in lags:
        composed, trace = trajectory(model, initial, lag=lag, horizons=horizons,
                                     substeps=substeps[lag], conditioning_time=lag)
        refined, _ = trajectory(model, initial, lag=lag, horizons=horizons,
                                 substeps=2 * substeps[lag], conditioning_time=lag)
        carried, carried_trace = trajectory(model, initial, lag=lag, horizons=horizons,
                                           substeps=substeps[lag], conditioning_time=lag, reencode=False)
        try:
            unclamped, _ = trajectory(model, initial, lag=lag, horizons=horizons,
                                      substeps=substeps[lag], conditioning_time=lag, reencode=False, clamp=False)
            clamp_control = {"status": "available", "by_horizon": {
                str(h): pair_metrics(carried[h], unclamped[h]) for h in horizons}}
        except RuntimeError as exc:
            if "non-finite" not in str(exc):
                raise
            clamp_control = {"status": "unavailable", "reason": str(exc)}
        by_horizon = {}
        for horizon in horizons:
            depth = integer_depth(horizon, lag)
            historical, _ = trajectory(model, initial, lag=horizon, horizons=(horizon,),
                                        substeps=depth * substeps[lag], conditioning_time=horizon,
                                        reencode=False)
            by_horizon[str(horizon)] = {
                "rhs_evaluations_each_path": 4 * depth * substeps[lag],
                "composition_depth": depth,
                "rollout_vs_reference": pair_metrics(composed[horizon], references[horizon]),
                "time_integration_refinement": {
                    "production_rhs_evaluations": 4 * depth * substeps[lag],
                    "refined_rhs_evaluations": 8 * depth * substeps[lag],
                    "production_vs_refined": pair_metrics(composed[horizon], refined[horizon]),
                    "refined_vs_reference": pair_metrics(refined[horizon], references[horizon]),
                },
                "fixed_condition_composition": pair_metrics(composed[horizon], carried[horizon]),
                "historical_condition_composition": pair_metrics(composed[horizon], historical[horizon]),
                "direct_conditioning_change": pair_metrics(historical[horizon], carried[horizon]),
            }
        composed_states[lag] = composed
        results[str(lag)] = {"substeps_per_call": substeps[lag], "conditioning_time": lag,
                             "trace_composed": trace, "trace_continuous_latent": carried_trace,
                             "clamp_counterfactual": clamp_control, "by_horizon": by_horizon}
    return {"by_lag": results, "cross_lag": {
        str(h): pair_metrics(composed_states[lags[0]][h], composed_states[lags[1]][h]) for h in horizons}}


def run(args) -> dict:
    archive_expected = TUBE_ARCHIVE_SHA256 if args.role == "tube" else FORMAL_SOURCE_ARCHIVE_SHA256
    commit = TUBE_SOURCE_COMMIT if args.role == "tube" else FORMAL_SOURCE_COMMIT
    if args.seed not in SEEDS or args.n_test <= 0 or args.n_test > 128 or args.batch_size <= 0:
        raise ValueError("use a frozen seed, 1..128 test samples, and a positive batch size")
    checkpoint = require_input_file(args.checkpoint, "checkpoint")
    archive = require_input_file(args.source_archive, "source archive")
    cache_path = require_input_file(args.test_cache, "locked Fisher cache")
    source_paths = {name: Path(__file__).parent / name for name in (
        Path(__file__).name, "fisher_frozen_inputs.py", "experiment_artifacts.py", "latent_integrators.py")}
    before_sources = source_hashes(source_paths)
    model, provenance = load_archived_checkpoint(checkpoint, expected_sha256=args.checkpoint_sha256,
        source_archive=archive, archive_sha256=archive_expected, source_commit=commit,
        role=args.role, seed=args.seed)
    cache, validation, cache_hash = load_frozen_fisher_cache(cache_path,
        expected_sha256=FORMAL_LOCKED_CACHE_SHA256, profile=FORMAL_CACHE_PROFILE,
        taus=LAGS, horizons=HORIZONS, max_samples=args.n_test)
    if int(provenance["model_config"]["kwargs"].get("N", -1)) != int(validation["N"]):
        raise ValueError("checkpoint grid differs from locked cache")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root = reserve_exploratory_root(args.output_dir, input_paths=(checkpoint, archive, cache_path))
    input_hashes = {str(checkpoint): args.checkpoint_sha256.lower(), str(archive): archive_expected,
                    str(cache_path): cache_hash}
    manifest = {"schema": "fisher-inrange-attribution-v1", "exploratory": True,
                "do_not_use_for_formal": True, "training": False, "checkpoint_selection": False,
                "lags": LAGS, "horizons": HORIZONS, "substeps": SUBSTEPS,
                "n_test": args.n_test, "smoke_only": args.n_test < 128,
                "checkpoint": provenance, "inputs_sha256": input_hashes,
                "evaluator_sources": before_sources, "cache_validation": validation,
                "environment": device_provenance(device),
                "interpretation": {
                    "fixed_condition_composition": "Same vector field and RK4 timestep; representation and rounding control, not an integration-error estimate.",
                    "cross_lag": "Both conditioning values are in training range and RHS work is equal.",
                    "time_integration_refinement": "Double RK4 substeps per call while retaining conditioning, clamp, and reencoding semantics; measures numerical sensitivity, not a certified error bound.",
                    "historical_condition_composition": "Direct conditioning 1.2/2.4/4.8 is outside training range; report separately.",
                    "clamp_counterfactual": "Diagnostic intervention, not an alternative prediction or additive error decomposition.",
                    "common_path_generator_stability": "unavailable: use evaluate_fisher_tube_attribution.py for the separate registered question."}}
    atomic_write_json(root / "manifest.json", manifest)
    model = model.to(device).eval()
    # Keep every sample result. Batches preserve deterministic original cache order.
    batches = []
    for start in range(0, args.n_test, args.batch_size):
        end = min(args.n_test, start + args.batch_size)
        references = {h: reference_states_at(cache, horizon=h, reference_dt=validation["reference_dt"],
                       sample_count=end)[start:end].to(device) for h in HORIZONS}
        batches.append({"sample_start": start, "sample_stop": end, "metrics": evaluate_model(
            model, cache["test_u0"][start:end].to(device), references)})
    for path, digest in input_hashes.items():
        verify_unchanged(path, digest, "frozen input")
    if source_hashes(source_paths) != before_sources:
        raise RuntimeError("evaluator source changed during evaluation")
    result = {**manifest, "status": "completed", "batches": batches}
    atomic_write_json(root / "results.json", result)
    atomic_write_json(root / "COMPLETE.json", {"status": "completed", "smoke_only": args.n_test < 128,
        "results_sha256": sha256_file(root / "results.json"), "manifest_sha256": sha256_file(root / "manifest.json"),
        "inputs_unchanged": True, "evaluator_sources_unchanged": True})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("baseline", "query", "tube"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--test-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "output_dir": str(Path(args.output_dir).resolve()),
                      "smoke_only": result["smoke_only"]}, allow_nan=False))


if __name__ == "__main__":
    main()
