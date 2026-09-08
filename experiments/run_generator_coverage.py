#!/usr/bin/env python3
"""Matched snapshot-budget experiments for generator identification.

True reaction values enter diagnostics and the explicitly privileged oracle only.
All ordinary training objectives consume state snapshots exclusively.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
import run_unknown_reaction_matched as base

MODES = ("initial", "teacher", "detached", "unroll")
SEEDS = (31415, 271828, 161803)
LAGS = (0.05, 0.1, 0.2)


class Flow(base.ReactionFlow):
    def __init__(self, conditioned=False, budget=4, scheme="euler"):
        if scheme not in ("euler", "midpoint") or budget % (2 if scheme == "midpoint" else 1):
            raise ValueError("budget must be divisible by reaction stages")
        super().__init__(conditioned, budget // (2 if scheme == "midpoint" else 1), True)
        self.scheme, self.budget = scheme, budget

    def forward(self, u, duration, substeps=None, conditioning=None):
        steps = self.substeps if substeps is None else substeps
        condition = duration if conditioning is None else conditioning
        dt = base.lag_column(duration, u) / steps
        for _ in range(steps):
            u = base.heat(u, dt / 2)
            first = self.reaction(u, condition)
            rate = self.reaction(u + dt * first / 2, condition) if self.scheme == "midpoint" else first
            u = base.heat(u + dt * rate, dt / 2)
        return u


def weights_hash(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()


def make_sequences(u, lags, length):
    # Requested times are rounded before lookup to avoid floating-point key drift.
    result = torch.empty(u.shape[0], length + 1, u.shape[-1], dtype=torch.float64)
    result[:, 0] = u
    for lag in LAGS:
        mask = (lags - lag).abs() < 1e-7
        times = [round(lag * j, 8) for j in range(1, length + 1)]
        ref = base.reference(u[mask], times)
        for j, t in enumerate(times, 1):
            result[mask, j] = ref[t]
    return result.float()


def prepare(root, smoke, length=4, data_offset=0):
    count, nv, nt = (8, 4, 4) if smoke else (128, 32, 64)
    train = {}
    for seed in SEEDS[:1] if smoke else SEEDS:
        u = base.initial_states(count, seed + 910000 + data_offset, device="cpu")
        rng = torch.Generator().manual_seed(seed + 810000 + data_offset)
        tau = torch.tensor(LAGS)[torch.randint(3, (count,), generator=rng)]
        train[seed] = {"u": u.float(), "tau": tau, "states": make_sequences(u, tau, length)}
    vu = base.initial_states(nv, 2026090801 + data_offset, device="cpu")
    val = {"u": vu.float(), "targets": base.reference(vu, (*LAGS, 0.4, 0.8))}
    splits = {}
    for name, seed, stress in (("test", 2026090802, False), ("stress", 2026090803, True)):
        u = base.initial_states(nt if name == "test" else max(4, nt // 2), seed + data_offset, stress, "cpu")
        splits[name] = {"u": u, "targets": base.reference(u, base.HORIZONS)}
    fine = base.reference(splits["test"]["u"][:4], base.HORIZONS, dt=0.001)
    errors = {str(t): float((fine[t] - splits["test"]["targets"][t][:4]).norm() / fine[t].norm()) for t in base.HORIZONS}
    audit = {"relative_l2": errors, "passed": max(errors.values()) < 1e-5}
    if not audit["passed"]:
        raise RuntimeError("reference accuracy failed")
    data = {"train": train, "val": val, **splits}
    torch.save(data, root / "cache.pt")
    base.dump(root / "cache_metadata.json", {"sha256": base.digest(root / "cache.pt"), "length": length,
        "count": count, "data_offset": data_offset, "validation_seed": 2026090801 + data_offset,
        "test_seed": 2026090802 + data_offset, "stress_seed": 2026090803 + data_offset,
        "reference_audit": audit})
    return data


def training_loss(model, data, mode, length):
    if mode == "initial":
        return (model(data["u"], data["tau"]) - data["states"][:, 1]).square().mean()
    count = data["u"].shape[0] // length
    states, tau = data["states"][:count], data["tau"][:count]
    if mode == "initial_repeat":
        u = states[:, :1].expand(-1, length, -1).reshape(-1, base.N)
        target = states[:, 1:2].expand(-1, length, -1).reshape(-1, base.N)
        return (model(u, tau[:, None].expand(-1, length).reshape(-1)) - target).square().mean()
    if mode == "teacher":
        u = states[:, :-1].reshape(-1, base.N)
        return (model(u, tau[:, None].expand(-1, length).reshape(-1)) - states[:, 1:].reshape(-1, base.N)).square().mean()
    if mode not in ("detached", "unroll"):
        raise ValueError(mode)
    u, losses = states[:, 0], []
    for j in range(length):
        if mode == "detached":
            u = u.detach()
        u = model(u, tau)
        losses.append((u - states[:, j + 1]).square().mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def validate(model, data):
    one = base.validation(model, data)
    errors = []
    for lag in (0.1, 0.2):
        u = data["u"]
        for j in range(1, round(0.8 / lag) + 1):
            u = model(u, lag)
            if j in (round(0.4 / lag), round(0.8 / lag)):
                errors.append((u - data["targets"][round(j * lag, 8)].float()).square().mean())
    return {"one_step": one, "rollout": float(torch.stack(errors).mean())}


@torch.no_grad()
def generator_diagnostics(model, train, test, length):
    grid = torch.linspace(-1.2, 1.2, 241, device=test["u"].device).reshape(1, -1)
    distributions = {"initial": train["u"], "reference_short_trajectory": train["states"][:, :-1].reshape(-1, base.N),
        "reference_long": torch.cat([test["targets"][t].float() for t in base.HORIZONS])}
    u, deployed = test["u"].float(), []
    for _ in range(16):
        u = model(u, 0.15)
        deployed.append(u)
    distributions["learned_long"] = torch.cat(deployed)
    curves, errors, occupancy = {}, {}, {}
    for lag in (0.0, 0.075, 0.15, 0.3):
        prediction = model.reaction(grid, lag)
        curves[str(lag)] = prediction.flatten().cpu().tolist()
        errors[str(lag)] = {name: float((model.reaction(values, lag) - base.truth_reaction(values)).square().mean())
            for name, values in {"uniform_grid": grid, **distributions}.items()}
    low, high = float(train["u"].min()), float(train["u"].max())
    for name, values in distributions.items():
        occupancy[name] = {"minimum": float(values.min()), "maximum": float(values.max()),
            "outside_initial_range_fraction": float(((values < low) | (values > high)).float().mean()),
            "abs_above_075_fraction": float((values.abs() > 0.75).float().mean()),
            "histogram_48_bins_minus12_to12": torch.histc(values.float().cpu(), bins=48, min=-1.2, max=1.2).tolist()}
    return {"states": grid.flatten().cpu().tolist(), "truth": base.truth_reaction(grid).flatten().cpu().tolist(),
        "curves": curves, "mse": errors, "occupancy": occupancy, "trajectory_length": length}


def checkpoint_evaluation(model, train, data, length):
    result = {"test": base.evaluate(model, data["test"]), "stress": base.evaluate(model, data["stress"]),
        "generator": generator_diagnostics(model, train, data["test"], length)}
    result["primary_mse"] = base.gm([result["test"]["endpoints"][f"tau={tau}:T={h}"]["mse"]
        for tau in (0.075, 0.15) for h in (1.2, 2.4)])
    return result


def train_cell(root, data, seed, mode, scheme, budget, epochs, length, device):
    name = f"{scheme}-b{budget}-L{length}-{mode}-s{seed}"
    outputs = {}
    for conditioned in (False, True):
        label = "B" if conditioned else "A"
        cell = root / "cells" / (name + "-" + label)
        cell.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(seed)
        model = Flow(conditioned, budget, scheme).to(device)
        train = data["train"][seed]
        initial_hash = weights_hash(model)
        count = train["u"].shape[0]
        oracle = mode == "oracle"
        config = {"seed": seed, "mode": mode, "scheme": scheme, "budget": budget,
            "length": length, "conditioned": conditioned, "epochs": epochs, "lr": 0.01,
            "prediction_pairs_per_update": count if not oracle else 0,
            "reaction_scalar_evaluations_per_update": count * base.N * budget if not oracle else 3 * 241,
            "unique_training_initial_states": count if mode == "initial" else count // length,
            "unique_snapshot_pairs": count // length if mode == "initial_repeat" else count,
            "initial_weights_sha256": initial_hash, "nominal_parameters": 65,
            "effective_parameters": 65 if conditioned else 49,
            "source_commit": json.loads((root / "provenance.json").read_text())["commit"],
            "cache_sha256": base.digest(root / "cache.pt"), "privileged_diagnostic": oracle,
            "training_uses_true_reaction": oracle, "dtype": "float32", "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
        base.dump(cell / "config.json", config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        best_scores = {"one_step": float("inf"), "rollout": float("inf")}
        best_states, best_epochs, rows, milestone = {}, {}, [], None
        start = time.perf_counter()
        training_seconds = 0.0
        grid = torch.linspace(-1.2, 1.2, 241, device=device).reshape(1, -1).expand(3, -1)
        grid_lags = torch.tensor(LAGS, device=device)
        for epoch in range(1, epochs + 1):
            if device == "cuda": torch.cuda.synchronize()
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = ((model.reaction(grid, grid_lags) - base.truth_reaction(grid)).square().mean()
                if oracle else training_loss(model, train, mode, length))
            if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss: {cell.name} epoch {epoch}")
            loss.backward()
            gradient_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())))
            optimizer.step()
            if device == "cuda": torch.cuda.synchronize()
            training_seconds += time.perf_counter() - tick
            if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
                scores = validate(model, data["val"])
                for key, score in scores.items():
                    if score < best_scores[key]:
                        best_scores[key], best_epochs[key] = score, epoch
                        best_states[key] = copy.deepcopy(model.state_dict())
                row = {"epoch": epoch, "train_mse": float(loss), "gradient_norm": gradient_norm,
                    **{f"validation_{k}": v for k, v in scores.items()}, "train_seconds": training_seconds,
                    "elapsed_seconds": time.perf_counter() - start}
                rows.append(row)
                with (cell / "metrics.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
                if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
                    message = json.dumps({"cell": cell.name, **row})
                    print(message, flush=True)
                    with (cell / "run.log").open("a") as f: f.write(message + "\n")
            if epoch == 120:
                milestone = copy.deepcopy(model.state_dict())
        states = {"final": copy.deepcopy(model.state_dict()), **{f"best_{k}": v for k, v in best_states.items()}}
        if milestone is not None: states["epoch120"] = milestone
        final_hash = weights_hash(model)
        if final_hash == initial_hash or not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("weights did not change or became nonfinite")
        evaluations = {}
        diagnostic_train = train if mode in ("initial", "oracle") else {k: v[:count // length] for k, v in train.items()}
        for key, state in states.items():
            torch.save(base.cpu_tree(state), cell / f"{key}.pt")
            model.load_state_dict(state)
            evaluations[key] = checkpoint_evaluation(model, diagnostic_train, data, length)
        model.load_state_dict(best_states["rollout"])
        refined = base.evaluate(model, data["test"], substeps=model.substeps * 4)
        structural = base.structural(model, data["test"]["u"])
        result = {"config": config, "best_validation": best_scores, "best_epochs": best_epochs,
            "training_seconds": training_seconds, "elapsed_seconds": time.perf_counter() - start,
            "final_weights_sha256": final_hash, "weights_changed": True, "weights_finite": True,
            "evaluations": evaluations, "refined_best_rollout": refined, "structural_best_rollout": structural,
            "structural_budget_note": "substeps count split steps; midpoint uses two reaction calls per split step"}
        base.dump(cell / "summary.json", result)
        (cell / "done").write_text("complete\n")
        outputs[cell.name] = result
        progress = sorted(p.parent.name for p in (root / "cells").glob("*/done"))
        base.dump(root / "progress.json", {"completed_cells": len(progress), "cells": progress})
        print(f"CELL_COMPLETE {cell.name}", flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--modes", nargs="+", default=list(MODES))
    parser.add_argument("--schemes", nargs="+", default=["euler"])
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--data-offset", type=int, default=0)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    try:
        expected = len(args.modes) * len(args.schemes) * (1 if args.smoke else 3) * 2
        base.dump(root / "provenance.json", {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": base.digest(__file__), "hostname": platform.node(), "python": os.sys.executable,
            "torch": torch.__version__, "cuda": torch.version.cuda, "expected_cells": expected,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "exploratory": True, "formal_confirmation": False})
        (root / "status").write_text("RUNNING\n")
        if args.cache:
            metadata = json.loads((args.cache.parent / "cache_metadata.json").read_text())
            if base.digest(args.cache) != metadata["sha256"] or metadata["length"] != args.length:
                raise RuntimeError("cache mismatch")
            import shutil
            for file in ("cache.pt", "cache_metadata.json"):
                shutil.copy2(args.cache.parent / file, root / file)
            data = torch.load(args.cache, map_location="cpu", weights_only=False)
        else:
            data = prepare(root, args.smoke, args.length, args.data_offset)
        data = base.device_tree(data, args.device)
        results = {}
        for scheme in args.schemes:
            for mode in args.modes:
                for seed in SEEDS[:1] if args.smoke else SEEDS:
                    results.update(train_cell(root, data, seed, mode, scheme, args.budget,
                        2 if args.smoke else args.epochs, args.length, args.device))
        base.dump(root / "completion.json", {"expected_cells": expected, "completed_cells": len(results),
            "all_weights_finite_changed": all(r["weights_finite"] and r["weights_changed"] for r in results.values())})
        if len(results) != expected: raise RuntimeError("missing cells")
        (root / "done").write_text("complete\n"); (root / "status").write_text("COMPLETE\n")
    except Exception:
        (root / "failed").write_text(traceback.format_exc()); (root / "status").write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
