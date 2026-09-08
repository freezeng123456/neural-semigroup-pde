#!/usr/bin/env python3
"""Paired mechanism contrasts and explicit structural-work units."""
import argparse
import json
import math
from pathlib import Path
import statistics


def gm(values):
    return math.exp(statistics.mean(math.log(max(v, 1e-300)) for v in values))


def primary(evaluation):
    return gm([evaluation["endpoints"][f"tau={t}:T={h}"]["mse"] for t in (.075, .15) for h in (1.2, 2.4)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    roots = ("wave1-euler", "wave2-midpoint", "oracle", "long-L4", "rate-L4", "long-L8")
    records = {}
    structure = {}
    checkpoint_contrasts = []
    for root in roots:
        for path in (args.root / root / "cells").glob("*/summary.json"):
            r = json.loads(path.read_text()); c = r["config"]
            key = (root, c["mode"], c["seed"], "B" if c["conditioned"] else "A")
            records[key] = r
            e = r["evaluations"]
            checkpoint_contrasts.append({"root": root, "mode": c["mode"], "seed": c["seed"], "model": key[-1],
                "one_step_selected_over_rollout_selected": e["best_one_step"]["primary_mse"] / e["best_rollout"]["primary_mse"],
                "final_over_rollout_selected": e["final"]["primary_mse"] / e["best_rollout"]["primary_mse"],
                "refined_over_primary": primary(r["refined_best_rollout"]) / e["best_rollout"]["primary_mse"],
                **({"final2000_over_epoch400": e["final"]["primary_mse"] / e["epoch400"]["primary_mse"]} if "epoch400" in e else {})})
            stages = 2 if c["scheme"] == "midpoint" else 1
            structure[str(path.relative_to(args.root))] = {split: {k: {
                "relative_rms_defect": value["relative_rms_defect"], "split_substeps": int(k),
                "actual_direct_reaction_calls": stages * int(k),
                "actual_composed_reaction_calls": stages * int(k) * len(split.split("+"))}
                for k, value in curve.items()} for split, curve in r["structural_best_rollout"].items()}
    contrasts = []
    for numerator, denominator in (("rate-L4", "long-L4"), ("long-L8", "long-L4"), ("wave2-midpoint", "wave1-euler")):
        for key, r in records.items():
            root, mode, seed, label = key
            other = records.get((denominator, mode, seed, label))
            if root != numerator or other is None: continue
            a, b = r["evaluations"]["best_rollout"], other["evaluations"]["best_rollout"]
            contrasts.append({"numerator": numerator, "denominator": denominator, "mode": mode, "seed": seed, "model": label,
                "primary_mse_ratio": a["primary_mse"] / b["primary_mse"],
                "generator_grid_ratio": gm([a["generator"]["mse"][str(t)]["uniform_grid"] / b["generator"]["mse"][str(t)]["uniform_grid"] for t in (.075, .15)]),
                "training_time_ratio": r["training_seconds"] / other["training_seconds"]})
    pooled = []
    for numerator, denominator, mode, label in sorted({(c["numerator"], c["denominator"], c["mode"], c["model"]) for c in contrasts}):
        subset = [c for c in contrasts if (c["numerator"], c["denominator"], c["mode"], c["model"]) == (numerator, denominator, mode, label)]
        pooled.append({"numerator": numerator, "denominator": denominator, "mode": mode, "model": label,
            **{field: gm([c[field] for c in subset]) for field in ("primary_mse_ratio", "generator_grid_ratio", "training_time_ratio")}})
    optimization = []
    for root, mode, label in sorted({(c["root"], c["mode"], c["model"]) for c in checkpoint_contrasts if "final2000_over_epoch400" in c}):
        subset = [c for c in checkpoint_contrasts if (c["root"], c["mode"], c["model"]) == (root, mode, label)]
        vals = [c["final2000_over_epoch400"] for c in subset]
        optimization.append({"root": root, "mode": mode, "model": label, "final2000_over_epoch400": gm(vals),
            "per_seed": [{"seed": c["seed"], "ratio": c["final2000_over_epoch400"]} for c in subset]})
    result = {"paired_contrasts": contrasts, "pooled_contrasts": pooled, "optimization": optimization,
        "checkpoint_contrasts": checkpoint_contrasts,
        "cautions": ["L8 changes independent initial-state diversity as well as trajectory depth.",
            "Rate and state losses share the exact cache and update/reaction budget.",
            "Refinement uses four times the split substeps and is a separate unequal-work diagnostic.",
            "Original structural summary call-count fields are split-step counts; the companion file explicitly converts them to actual reaction calls."]}
    (args.output / "mechanisms.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output / "STRUCTURAL_WORK_COUNTS.json").write_text(json.dumps({"unit_note": "Explicit derived actual call counts; original numerical defects are unchanged.", "cells": structure}, indent=2) + "\n")
    print(json.dumps({"pooled_contrasts": pooled, "optimization": optimization}, indent=2))


if __name__ == "__main__":
    main()
