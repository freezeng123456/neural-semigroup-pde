#!/usr/bin/env python3
"""Analyze complete generator-coverage roots without importing training code."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def gm(values):
    return math.exp(statistics.mean(math.log(max(v, 1e-300)) for v in values))


def analyze(roots, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in roots:
        for path in sorted((root / "cells").glob("*/summary.json")):
            if not (path.parent / "done").exists(): continue
            r = json.loads(path.read_text()); c = r["config"]
            for checkpoint, ev in r["evaluations"].items():
                rows.append({"root": root.name, "cell": path.parent.name, "scheme": c["scheme"],
                    "budget": c["budget"], "length": c["length"], "mode": c["mode"], "seed": c["seed"],
                    "model": "B" if c["conditioned"] else "A", "checkpoint": checkpoint,
                    "primary_mse": ev["primary_mse"],
                    "generator_grid_mse": gm([ev["generator"]["mse"][str(t)]["uniform_grid"] for t in (.075, .15)]),
                    "generator_reference_mse": gm([ev["generator"]["mse"][str(t)]["reference_long"] for t in (.075, .15)]),
                    "bound_violation": ev["test"]["bound_violation_fraction"],
                    "energy_monotone": ev["test"]["energy_monotone_fraction"],
                    "stress_mse": gm([ev["stress"]["endpoints"][f"tau={t}:T={h}"]["mse"] for t in (.075, .15) for h in (1.2, 2.4)]),
                    "training_seconds": r["training_seconds"], "epochs": c["epochs"],
                    "selected_epoch": r["best_epochs"].get(checkpoint[5:] if checkpoint.startswith("best_") else checkpoint,
                        int(checkpoint[5:]) if checkpoint.startswith("epoch") else c["epochs"]),
                    "privileged": c["privileged_diagnostic"]})
    if not rows: raise ValueError("no completed cells")
    with (out / "all_checkpoints.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    primary = [r for r in rows if r["checkpoint"] == "best_rollout"]
    groups = {}
    for r in primary:
        key = (r["root"], r["scheme"], r["budget"], r["length"], r["mode"])
        groups.setdefault(key, []).append(r)
    aggregates = []
    for key, group in groups.items():
        pairs = []
        for a in group:
            if a["model"] != "A": continue
            b = next(b for b in group if b["model"] == "B" and b["seed"] == a["seed"])
            pairs.append({"seed": a["seed"], "ratio": a["primary_mse"] / b["primary_mse"],
                "generator_ratio": a["generator_grid_mse"] / b["generator_grid_mse"],
                "energy_delta": a["energy_monotone"] - b["energy_monotone"],
                "bound_delta": a["bound_violation"] - b["bound_violation"]})
        ratio = gm([p["ratio"] for p in pairs])
        passed = (len(pairs) == 3 and ratio <= .9 and sum(p["ratio"] <= .9 for p in pairs) >= 2
            and max(p["ratio"] for p in pairs) <= 1.05 and min(p["energy_delta"] for p in pairs) >= -.02
            and max(p["bound_delta"] for p in pairs) <= .02)
        aggregate = dict(zip(("root", "scheme", "budget", "length", "mode"), key))
        aggregate.update({"pairs": pairs, "ratio_a_over_b": ratio,
            "accuracy_gate_passed": passed and key[-1] != "oracle", "privileged": key[-1] == "oracle"})
        for label in ("A", "B"):
            subset = [r for r in group if r["model"] == label]
            aggregate[label] = {"primary_mse": gm([r["primary_mse"] for r in subset]),
                "generator_grid_mse": gm([r["generator_grid_mse"] for r in subset]),
                "generator_reference_mse": gm([r["generator_reference_mse"] for r in subset]),
                "stress_mse": gm([r["stress_mse"] for r in subset]),
                "bound_max": max(r["bound_violation"] for r in subset),
                "energy_min": min(r["energy_monotone"] for r in subset),
                "mean_training_seconds": statistics.mean(r["training_seconds"] for r in subset)}
        aggregates.append(aggregate)
    comparisons = []
    for a in primary:
        if a["mode"] in ("initial", "oracle"): continue
        baseline = [b for b in primary if b["mode"] == "initial" and all(a[k] == b[k] for k in
            ("root", "scheme", "budget", "length", "seed", "model"))]
        if baseline:
            comparisons.append({"root": a["root"], "scheme": a["scheme"], "mode": a["mode"],
                "seed": a["seed"], "model": a["model"], "mse_over_initial": a["primary_mse"] / baseline[0]["primary_mse"],
                "generator_over_initial": a["generator_grid_mse"] / baseline[0]["generator_grid_mse"]})
    summary = {"completed_cells": len(primary), "checkpoint_evaluations": len(rows), "aggregates": aggregates,
        "comparisons_to_initial": comparisons, "exploratory": True, "checkpoint": "best_rollout"}
    (out / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# H20 generator coverage results", "", "Exploratory, best short-rollout validation checkpoint. Ratios below one favor A.", "",
        "| Run / scheme / mode | A MSE | B MSE | A/B | Seed ratios | A/B grid source MSE | Accuracy gate |",
        "|---|---:|---:|---:|---|---:|---|"]
    for a in aggregates:
        lines.append(f"| {a['root']} / {a['scheme']} / {a['mode']} | {a['A']['primary_mse']:.6g} | {a['B']['primary_mse']:.6g} | {a['ratio_a_over_b']:.4f} | "
            + ", ".join(f"{p['ratio']:.3f}" for p in a["pairs"]) + f" | {a['A']['generator_grid_mse']:.6g} / {a['B']['generator_grid_mse']:.6g} | "
            + ("privileged diagnostic" if a["privileged"] else "pass (exploratory)" if a["accuracy_gate_passed"] else "fail") + " |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"completed_cells": len(primary), "aggregates": aggregates}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.roots, args.output)
