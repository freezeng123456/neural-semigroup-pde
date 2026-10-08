#!/usr/bin/env python3
"""Paired Burgers study against the independent, versioned reference cache.

Selection sees train/validation only. The locked test split is evaluated after
all selected checkpoints have been fixed. Existing screen sources stay frozen.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiment_artifacts import (  # noqa: E402
    atomic_torch_save, atomic_write_json, device_provenance, git_commit,
    reserve_exploratory_root, sha256_file, torch_load_compat, verify_unchanged,
)
from run_burgers_flux_ablation import AblatedBurgersFluxFlow  # noqa: E402
from run_burgers_semigroup_screen import PeriodicBurgersFluxFlow  # noqa: E402
from seed_utils import set_global_seed  # noqa: E402

SCHEMA = "burgers-paired-v2"
CACHE_SCHEMA = "burgers-reference-v2"
TRAIN_TAUS = (0.025, 0.05, 0.075, 0.1)
TEST_TAUS = (0.04, 0.08)
HORIZONS = (0.4, 0.8)
TIMES = (0.0, 0.025, 0.04, 0.05, 0.075, 0.08, 0.1, 0.4, 0.8)
DATA_SEEDS = {"train": 2026100301, "val": 2026100302, "test": 2026100303}
ARCHITECTURES = {"small": (0, 8, 1), "large": (2, 32, 2)}
ENERGY_TOLERANCE = 1e-8
REFERENCE_LIMITS = {"spatial_max_relative_l2": 0.001, "time_max_relative_l2": 0.00001,
                    "max_step_energy_increase": 1e-12, "max_mean_drift": 1e-12}
PROJECTION = "Fourier low pass to N64; coarse Nyquist omitted"
TARGET_SCOPE = "Projected high-resolution PDE; coarse Markov closure not assumed"
AUDIT_SCOPE = "Sampled convergence and physical checks; no uniform PDE error certificate."


def finite_nonnegative(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def validate_reference_diagnostics(diagnostic: dict, config: dict, n_grid: int, dt: float) -> None:
    expected = {"version": "burgers-fourier-galerkin-v2", "N": n_grid,
                "L": config["L"], "nu": config["nu"], "dt": dt}
    if not isinstance(diagnostic, dict) or any(diagnostic.get(key) != value for key, value in expected.items()):
        raise ValueError("reference solver diagnostic does not match frozen solver configuration")
    for key in ("max_step_energy_increase", "max_mean_drift"):
        if finite_nonnegative(diagnostic.get(key), key) > REFERENCE_LIMITS[key]:
            raise ValueError(f"reference physical gate failed: {key}")
    for key in ("max_abs_state", "initial_projection_rms"):
        finite_nonnegative(diagnostic.get(key), key)


def validate_reference_audit(cache: dict, config: dict) -> None:
    if cache.get("projection") != PROJECTION or cache.get("target_scope") != TARGET_SCOPE:
        raise ValueError("cache projection and target_scope must match the projected-PDE contract")
    if cache.get("exploratory") is not True or cache.get("do_not_use_for_formal") is not True:
        raise ValueError("reference cache must retain its exploratory evidence flags")
    audit = cache.get("reference_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError("cache requires a passed reference_audit")
    if audit.get("limits") != REFERENCE_LIMITS:
        raise ValueError("reference_audit limits do not match frozen gates")
    if audit.get("audit_seed") != 2026100304 or list(audit.get("times", ())) != list(TIMES):
        raise ValueError("reference_audit seed or times do not match the independent audit")
    if audit.get("scope") != AUDIT_SCOPE:
        raise ValueError("reference_audit must retain the sampled convergence disclaimer")
    count = audit.get("n_samples")
    if type(count) is not int or count < 1:
        raise ValueError("reference_audit n_samples must be positive")
    for key in ("spatial", "temporal", "projected_spatial", "projected_restart_closure"):
        metric = audit.get(key, {})
        maximum = finite_nonnegative(metric.get("max_relative_l2"), f"audit {key} maximum")
        finite_nonnegative(metric.get("pooled_relative_l2"), f"audit {key} pooled error")
        per_sample = metric.get("per_sample_max_relative_l2")
        if not isinstance(per_sample, list) or len(per_sample) != count:
            raise ValueError(f"audit {key} must retain every sampled error")
        values = [finite_nonnegative(value, f"audit {key} sample error") for value in per_sample]
        if not math.isclose(max(values), maximum, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(f"audit {key} maximum disagrees with sampled errors")
    if audit["spatial"]["max_relative_l2"] > REFERENCE_LIMITS["spatial_max_relative_l2"]:
        raise ValueError("reference spatial convergence gate failed")
    if audit["temporal"]["max_relative_l2"] > REFERENCE_LIMITS["time_max_relative_l2"]:
        raise ValueError("reference temporal convergence gate failed")
    diagnostics = audit.get("solver_diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 3:
        raise ValueError("reference audit requires three solver diagnostic records")
    for diagnostic, (size, dt) in zip(diagnostics, ((512, 0.0005), (512, 0.00025), (1024, 0.00025))):
        validate_reference_diagnostics(diagnostic, config, size, dt)


def time_index(times: torch.Tensor, value: float) -> int:
    """Find the recorded time exactly within roundoff, never by truncation."""
    if not isinstance(times, torch.Tensor) or times.ndim != 1:
        raise ValueError("times must be a one-dimensional tensor")
    matches = torch.nonzero(torch.isclose(
        times.to(torch.float64), torch.tensor(value, dtype=torch.float64),
        rtol=0.0, atol=1e-12,
    )).flatten()
    if len(matches) != 1:
        raise ValueError(f"time {value} must occur exactly once in cache")
    return int(matches.item())


def validate_cache(cache: dict) -> dict:
    """Validate the complete cache before reserving any experiment output."""
    if cache.get("schema_version") != CACHE_SCHEMA:
        raise ValueError(f"requires cache schema {CACHE_SCHEMA}")
    config = cache.get("config", {})
    fixed = {"N": 64, "L": 2 * math.pi, "nu": 0.01,
             "reference_N": 512, "reference_dt": 0.0005}
    for key, expected in fixed.items():
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cache config {key} must equal {expected}")
    for key, expected in (("train_taus", TRAIN_TAUS), ("test_taus", TEST_TAUS), ("horizons", HORIZONS)):
        if config.get(key) != list(expected):
            raise ValueError(f"cache config {key} does not match protocol")
    if config.get("data_seeds") != DATA_SEEDS:
        raise ValueError("cache data_seeds do not match the frozen split seeds")
    validate_reference_audit(cache, config)
    splits = cache.get("splits", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("cache must contain separate train, val, test splits")
    seen = set()
    for name, split in splits.items():
        count = config.get(f"n_{name}")
        if type(count) is not int or count <= 0:
            raise ValueError(f"n_{name} must be a positive integer")
        diagnostics = split.get("reference_diagnostics")
        if not isinstance(diagnostics, list) or len(diagnostics) != math.ceil(count / 50):
            raise ValueError(f"{name} must retain every reference batch diagnostic")
        for diagnostic in diagnostics:
            validate_reference_diagnostics(diagnostic, config, config["reference_N"], config["reference_dt"])
        expected_shapes = {"u0": (count, 64), "times": (len(TIMES),), "states": (count, len(TIMES), 64)}
        for key, shape in expected_shapes.items():
            tensor = split.get(key)
            dtype = torch.float64 if key == "times" else torch.float32
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"{name}.{key} must have shape {shape} and dtype {dtype}")
            if tensor.device.type != "cpu" or not torch.isfinite(tensor).all():
                raise ValueError(f"{name}.{key} must be finite CPU data")
        if not torch.allclose(split["times"], torch.tensor(TIMES, dtype=torch.float64), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name}.times must match sorted protocol times")
        if not torch.equal(split["states"][:, 0], split["u0"]):
            raise ValueError(f"{name} time-zero states must equal u0")
        hashes = {hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest() for row in split["u0"]}
        if seen.intersection(hashes):
            raise ValueError("initial states overlap between data splits")
        seen.update(hashes)
    train = splits["train"]
    count = config["n_train"]
    tau, target = train.get("tau"), train.get("ut")
    if not isinstance(tau, torch.Tensor) or tau.dtype != torch.float64 or tuple(tau.shape) != (count,):
        raise ValueError("train.tau must be float64 with one value per sample")
    if not isinstance(target, torch.Tensor) or target.dtype != torch.float32 or tuple(target.shape) != (count, 64):
        raise ValueError("train.ut must be float32 with shape [n_train,64]")
    for lag in tau.unique().tolist():
        if lag not in TRAIN_TAUS:
            raise ValueError("train.tau contains a lag outside training support")
        mask = tau == lag
        if not torch.equal(target[mask], train["states"][mask, time_index(train["times"], lag)]):
            raise ValueError("train.ut does not match the recorded reference time")
    return config


def build_model(architecture: str, query_conditioned: bool, config: dict, ode_steps: int = 12):
    radius, hidden, layers = ARCHITECTURES[architecture]
    return AblatedBurgersFluxFlow(
        n_grid=config["N"], length=config["L"], viscosity=config["nu"],
        radius=radius, hidden=hidden, hidden_layers=layers,
        ode_steps=ode_steps, query_conditioned=query_conditioned,
    )


def state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_counts(model) -> dict:
    nominal = sum(parameter.numel() for parameter in model.parameters())
    inactive = 0 if model.query_conditioned else model.flux[0].out_features
    return {"parameter_count": nominal, "active_parameter_count": nominal - inactive,
            "inactive_control_weight_count": inactive,
            "parameter_count_scope": "Nominal architectures match; autonomous zero conditioning leaves first-layer control weights inactive"}


def strict_json(path: Path, payload: dict | list) -> None:
    # Fail closed: NaN metrics must never be silently rewritten as null.
    json.dumps(payload, allow_nan=False)
    atomic_write_json(path, payload)


def scalar_mean(values: torch.Tensor) -> float:
    return float(values.to(torch.float64).mean().item())


def sample_errors(prediction: torch.Tensor, reference: torch.Tensor) -> dict:
    error2 = (prediction.to(torch.float64) - reference.to(torch.float64)).square().mean(dim=1)
    reference2 = reference.to(torch.float64).square().mean(dim=1)
    # A zero reference has no meaningful relative error; absolute MSE remains.
    relative = [math.sqrt(e / r) if r > 0 else None for e, r in zip(error2.tolist(), reference2.tolist())]
    if not torch.isfinite(error2).all() or any(v is not None and not math.isfinite(v) for v in relative):
        raise FloatingPointError("non-finite prediction error")
    return {"mse": error2.cpu().tolist(), "relative_l2": relative}


def error_summary(errors: dict) -> dict:
    relative = [v for v in errors["relative_l2"] if v is not None]
    return {"mse_mean": sum(errors["mse"]) / len(errors["mse"]),
            "relative_l2_mean": sum(relative) / len(relative) if relative else None,
            "undefined_relative_l2_count": len(errors["relative_l2"]) - len(relative)}


def energy_metrics(energies: torch.Tensor, tolerance: float = ENERGY_TOLERANCE) -> dict:
    """Per-sample diagnostics include every observed rollout step."""
    energies = energies.to(torch.float64)
    if energies.ndim != 2 or energies.shape[1] < 2 or not torch.isfinite(energies).all():
        raise ValueError("energies must be finite [sample, step] with at least two steps")
    increments = energies[:, 1:] - energies[:, :-1]
    positive = increments.clamp_min(0)
    return {
        "energy_monotone": (increments <= tolerance).all(dim=1).tolist(),
        "energy_increase_step_count": (increments > tolerance).sum(dim=1).tolist(),
        "energy_positive_increment_sum": positive.sum(dim=1).tolist(),
        "energy_max_positive_increment": positive.max(dim=1).values.tolist(),
        "energy_horizon_delta": (energies[:, -1] - energies[:, 0]).tolist(),
        "energy_at_steps": energies.tolist(),
        "energy_tolerance": tolerance,
    }


def exact_depth(horizon: float, tau: float) -> int:
    depth = round(horizon / tau)
    if depth <= 0 or not math.isclose(depth * tau, horizon, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("horizon must be an integer multiple of rollout lag")
    return depth


def fair_substeps(tau: float, base_steps: int = 12) -> int:
    count = round(base_steps * tau / min(TEST_TAUS))
    if count <= 0 or not math.isclose(count * min(TEST_TAUS), base_steps * tau, abs_tol=1e-12):
        raise ValueError("cannot assign equal RK4 work at this lag")
    return count


@torch.no_grad()
def validation_mse(model, validation: dict, *, device: str, batch_size: int) -> float:
    model.eval()
    total, count = 0.0, 0
    for lag in TRAIN_TAUS:
        index = time_index(validation["times"], lag)
        for start in range(0, len(validation["u0"]), batch_size):
            initial = validation["u0"][start:start + batch_size].to(device)
            reference = validation["states"][start:start + batch_size, index].to(device)
            errors = (model(initial, lag).to(torch.float64) - reference).square().mean(dim=1)
            total += errors.sum().item()
            count += len(initial)
    score = total / count
    if not math.isfinite(score):
        raise FloatingPointError("non-finite validation MSE")
    return score


def synchronize(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def train_model(model, train: dict, validation: dict, args, cell_dir: Path, provenance: dict) -> dict:
    """Only train/validation are accepted; no test object reaches selection."""
    set_global_seed(args.seed, deterministic=True)
    loader = DataLoader(TensorDataset(train["u0"], train["tau"], train["ut"]),
                        batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cell_dir.mkdir(parents=True, exist_ok=False)
    best, selected_epoch = math.inf, None
    history, validation_seconds = [], 0.0
    synchronize(args.device)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0.0, 0
        for initial, tau, target in loader:
            initial, tau, target = initial.to(args.device), tau.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(initial, tau)
            loss = (prediction - target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(initial)
            count += len(initial)
        scheduler.step()
        score = None
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            synchronize(args.device)
            validation_started = time.perf_counter()
            score = validation_mse(model, validation, device=args.device, batch_size=args.eval_batch_size)
            synchronize(args.device)
            validation_seconds += time.perf_counter() - validation_started
        history.append({"epoch": epoch, "train_mse": total / count, "val_mse": score,
                        "learning_rate_next": optimizer.param_groups[0]["lr"]})
        if score is not None and score < best:
            best, selected_epoch = score, epoch
            atomic_torch_save({"model_state_dict": copy.deepcopy(model.state_dict()),
                               "epoch": epoch, "val_mse": score, "provenance": provenance},
                              cell_dir / "selected.pt")
        strict_json(cell_dir / "history.json", history)
        if score is not None:
            print(f"{cell_dir.name} epoch={epoch} train_mse={total / count:.8g} val_mse={score:.8g}", flush=True)
    atomic_torch_save({"model_state_dict": model.state_dict(), "epoch": args.epochs,
                       "val_mse": history[-1]["val_mse"], "optimizer_state_dict": optimizer.state_dict(),
                       "scheduler_state_dict": scheduler.state_dict(), "provenance": provenance},
                      cell_dir / "final.pt")
    synchronize(args.device)
    elapsed = time.perf_counter() - started
    checkpoint = torch_load_compat(cell_dir / "selected.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {"selected_epoch": selected_epoch, "best_val_mse": best,
            "train_and_validation_seconds": elapsed, "validation_seconds": validation_seconds,
            "train_seconds_excluding_validation": elapsed - validation_seconds,
            "selected_sha256": sha256_file(cell_dir / "selected.pt"),
            "final_sha256": sha256_file(cell_dir / "final.pt")}


class ExactFluxOracle(PeriodicBurgersFluxFlow):
    """Known quadratic flux with the same centered finite differences.

    This measures a model-class spatial baseline; it is not the reference PDE
    solver and centered differencing does not guarantee energy dissipation.
    """
    def __init__(self, config: dict, ode_steps: int):
        super().__init__(n_grid=config["N"], length=config["L"], viscosity=config["nu"],
                         radius=0, hidden=1, ode_steps=ode_steps, query_conditioned=False)
        del self.flux

    def local_flux(self, state, conditioning_tau):
        return 0.5 * state.square()


@torch.no_grad()
def evaluate_model(model, test: dict, *, device: str, batch_size: int, base_steps: int = 12) -> tuple[dict, dict]:
    model.eval()
    n = len(test["u0"])
    initial_cpu = test["u0"]
    one_step, rollout_states, rollout_energy, refinement_states = {}, {}, {}, {}
    for tau in TEST_TAUS:
        substeps = fair_substeps(tau, base_steps)
        predictions, states_by_horizon, energies_by_horizon = [], {h: [] for h in HORIZONS}, {h: [] for h in HORIZONS}
        refined_by_horizon = {h: [] for h in HORIZONS}
        for start in range(0, n, batch_size):
            state = initial_cpu[start:start + batch_size].to(device)
            refined = state
            energies = [0.5 * state.to(torch.float64).square().mean(dim=1)]
            for depth in range(1, exact_depth(max(HORIZONS), tau) + 1):
                state = model.integrate(state, tau, substeps=substeps, conditioning_tau=tau)
                refined = model.integrate(refined, tau, substeps=2 * substeps, conditioning_tau=tau)
                energies.append(0.5 * state.to(torch.float64).square().mean(dim=1))
                if depth == 1:
                    predictions.append(state.cpu())
                for horizon in HORIZONS:
                    if depth == exact_depth(horizon, tau):
                        states_by_horizon[horizon].append(state.cpu())
                        refined_by_horizon[horizon].append(refined.cpu())
                        energies_by_horizon[horizon].append(torch.stack(energies, dim=1).cpu())
        prediction = torch.cat(predictions)
        one_step[str(tau)] = sample_errors(prediction, test["states"][:, time_index(test["times"], tau)])
        for horizon in HORIZONS:
            key = f"tau={tau}:horizon={horizon}"
            rollout_states[key] = torch.cat(states_by_horizon[horizon])
            refinement_states[key] = torch.cat(refined_by_horizon[horizon])
            rollout_energy[key] = torch.cat(energies_by_horizon[horizon])
    per_sample = {"sample_indices": list(range(n)), "one_step": one_step, "rollouts": {},
                  "composition_matched_conditioning": {}, "composition_historical_conditioning": {}, "cross_lag": {},
                  "time_integration_refinement": {}}
    summary = {"one_step": {key: error_summary(value) for key, value in one_step.items()},
               "rollouts": {}, "composition_matched_conditioning": {},
               "composition_historical_conditioning": {}, "cross_lag": {}, "time_integration_refinement": {}}
    for tau in TEST_TAUS:
        substeps = fair_substeps(tau, base_steps)
        for horizon in HORIZONS:
            key = f"tau={tau}:horizon={horizon}"
            state = rollout_states[key]
            refinement_mse = (state.to(torch.float64) - refinement_states[key].to(torch.float64)).square().mean(dim=1)
            refinement_rms = refinement_mse.sqrt()
            per_sample["time_integration_refinement"][key] = {
                "mse_vs_production": refinement_mse.tolist(), "rms_vs_production": refinement_rms.tolist()}
            summary["time_integration_refinement"][key] = {
                "mse_vs_production_mean": scalar_mean(refinement_mse),
                "rms_vs_production_mean": scalar_mean(refinement_rms),
                "rms_vs_production_max": refinement_rms.max().item(),
                "production_substeps_per_call": substeps, "refined_substeps_per_call": 2 * substeps,
                "interpretation": "Learned-flow time-step sensitivity; independent of PDE-reference accuracy and model selection"}
            errors = sample_errors(state, test["states"][:, time_index(test["times"], horizon)])
            energy = energy_metrics(rollout_energy[key])
            drift = (state.to(torch.float64).mean(dim=1) - initial_cpu.to(torch.float64).mean(dim=1)).abs()
            per_sample["rollouts"][key] = {**errors, **energy, "mean_drift": drift.tolist()}
            summary["rollouts"][key] = {**error_summary(errors), "mean_drift_max": drift.max().item(),
                "energy_monotone_fraction": sum(energy["energy_monotone"]) / n,
                "energy_positive_increment_sum_mean": sum(energy["energy_positive_increment_sum"]) / n,
                "energy_max_positive_increment": max(energy["energy_max_positive_increment"]),
                "energy_horizon_delta_mean": sum(energy["energy_horizon_delta"]) / n,
                "energy_increase_step_count_total": sum(energy["energy_increase_step_count"]),
                "energy_tolerance": ENERGY_TOLERANCE,
                "rk4_substeps_per_call": substeps,
                "rhs_evaluations": 4 * substeps * exact_depth(horizon, tau)}
            for name, conditioning in (("composition_matched_conditioning", tau), ("composition_historical_conditioning", horizon)):
                differences = []
                for start in range(0, n, batch_size):
                    direct = model.integrate(initial_cpu[start:start + batch_size].to(device), horizon,
                                             substeps=exact_depth(horizon, tau) * substeps,
                                             conditioning_tau=conditioning)
                    difference = (direct.cpu().to(torch.float64) - state[start:start + batch_size]).square().mean(dim=1).sqrt()
                    differences.extend(difference.tolist())
                per_sample[name][key] = differences
                summary[name][key] = {"rms_defect_mean": sum(differences) / n,
                                     "rms_defect_max": max(differences),
                                     "direct_conditioning_tau": conditioning,
                                     "rhs_evaluations_each_path": 4 * substeps * exact_depth(horizon, tau)}
                if name == "composition_matched_conditioning":
                    summary[name][key]["interpretation"] = "Same-step identity control; not an estimate of general integration error"
    for horizon in HORIZONS:
        fine = rollout_states[f"tau={TEST_TAUS[0]}:horizon={horizon}"]
        coarse = rollout_states[f"tau={TEST_TAUS[1]}:horizon={horizon}"]
        defects = (fine.to(torch.float64) - coarse.to(torch.float64)).square().mean(dim=1).sqrt()
        key = str(horizon)
        per_sample["cross_lag"][key] = defects.tolist()
        summary["cross_lag"][key] = {"rms_defect_mean": scalar_mean(defects), "rms_defect_max": defects.max().item(),
                                    "fine_substeps": fair_substeps(TEST_TAUS[0], base_steps),
                                    "coarse_substeps": fair_substeps(TEST_TAUS[1], base_steps),
                                    "rhs_evaluations_each_path": 4 * base_steps * exact_depth(horizon, TEST_TAUS[0])}
    # No structural-defect ratio is formed: autonomous equal-step paths can be
    # bitwise identical, making their numerical floor exactly zero.
    json.dumps(summary, allow_nan=False)
    json.dumps(per_sample, allow_nan=False)
    return summary, per_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--architectures", nargs="+", choices=tuple(ARCHITECTURES), default=list(ARCHITECTURES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--ode-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--expect-cache-sha256")
    return parser


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    for key in ("epochs", "threads", "batch_size", "eval_batch_size", "validation_interval", "ode_steps"):
        if getattr(args, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if args.seed < 0 or args.lr <= 0 or args.weight_decay < 0 or len(set(args.architectures)) != len(args.architectures):
        raise ValueError("invalid optimizer, seed, or duplicate architecture settings")
    torch.set_num_threads(args.threads)
    cache_path = args.data_cache.expanduser().resolve()
    cache_hash = sha256_file(cache_path)
    if args.expect_cache_sha256 and args.expect_cache_sha256 != cache_hash:
        raise ValueError("cache SHA-256 does not match expected digest")
    cache = torch_load_compat(cache_path, map_location="cpu")
    config = validate_cache(cache)
    root = reserve_exploratory_root(args.output_dir, input_paths=(cache_path,))
    source_names = ("run_burgers_validated.py", "burgers_reference.py", "prepare_burgers_validated_cache.py",
                    "run_burgers_flux_ablation.py", "run_burgers_semigroup_screen.py", "pde_solver.py",
                    "seed_utils.py", "experiment_artifacts.py")
    sources = {name: sha256_file(EXPERIMENTS_DIR / name) for name in source_names}
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration = {"cache_config": config, "settings": settings, "architectures": ARCHITECTURES,
                     "energy_tolerance": ENERGY_TOLERANCE,
                     "projection": cache["projection"], "target_scope": cache["target_scope"],
                     "reference_audit_scope": cache["reference_audit"]["scope"],
                     "selection": "validation MSE averaged over all four training lags; earliest epoch wins ties",
                     "test_policy": "all selections complete before any locked test evaluation",
                     "evaluation_work": "0.04 uses base_steps, 0.08 uses 2*base_steps; identical total RK4 work",
                     "matched_conditioning_scope": "Same-step identity control; use 2x-substep refinement to measure time-integration sensitivity",
                     "parameter_count_scope": "Autonomous small/large models have 8/32 inactive control weights; nominal parameter counts match paired models"}
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    provenance = {"schema_version": SCHEMA, "configuration": configuration, "config_sha256": config_hash,
                  "sources": sources, "git_commit": git_commit(EXPERIMENTS_DIR.parent),
                  "cache_sha256_before": cache_hash, "runtime": device_provenance(torch.device(args.device)),
                  "exploratory": True, "do_not_use_for_formal": True,
                  "reference_audit": cache["reference_audit"],
                  "scope": "full" if [config[f"n_{name}"] for name in ("train", "val", "test")] == [1000, 50, 128] and args.epochs == 100 else "pilot"}
    strict_json(root / "manifest.json", provenance)
    cells, models = {}, {}
    for architecture in args.architectures:
        set_global_seed(args.seed, deterministic=True)
        template = build_model(architecture, False, config, args.ode_steps)
        initial_state = copy.deepcopy(template.state_dict())
        for query in (False, True):
            label = f"{architecture}_{'query' if query else 'autonomous'}"
            model = build_model(architecture, query, config, args.ode_steps).to(args.device)
            model.load_state_dict(initial_state, strict=True)
            cell_provenance = {**provenance, "architecture": architecture, "query_conditioned": query,
                               "initial_state_sha256": state_digest(initial_state), **parameter_counts(model)}
            training = train_model(model, cache["splits"]["train"], cache["splits"]["val"], args, root / label, cell_provenance)
            cells[label] = {"architecture": architecture, "query_conditioned": query,
                            **parameter_counts(model),
                            "initial_state_sha256": cell_provenance["initial_state_sha256"], **training}
            models[label] = model
    strict_json(root / "selection_complete.json", cells)
    models["exact_flux_fd_oracle"] = ExactFluxOracle(config, args.ode_steps).to(args.device)
    cells["exact_flux_fd_oracle"] = {"parameter_count": 0, "description": "known quadratic flux with centered finite differences"}
    for label, model in models.items():
        synchronize(args.device)
        started = time.perf_counter()
        summary, per_sample = evaluate_model(model, cache["splits"]["test"], device=args.device,
                                            batch_size=args.eval_batch_size, base_steps=args.ode_steps)
        synchronize(args.device)
        cells[label]["eval_seconds"] = time.perf_counter() - started
        cells[label]["evaluation"] = summary
        strict_json(root / label / "per_sample.json", per_sample)
        strict_json(root / label / "summary.json", cells[label])
    after = verify_unchanged(cache_path, cache_hash, "reference cache")
    sources_after = {name: sha256_file(EXPERIMENTS_DIR / name) for name in source_names}
    if sources_after != sources:
        raise RuntimeError("source files changed while experiment was running")
    result = {**provenance, "cache_sha256_after": after, "cells": cells}
    strict_json(root / "summary.json", result)
    strict_json(root / "COMPLETE.json", {"schema_version": SCHEMA, "summary_sha256": sha256_file(root / "summary.json"),
                                         "cache_sha256": after, "config_sha256": config_hash})
    print(f"Completed {root}", flush=True)
    return result


if __name__ == "__main__":
    main()
