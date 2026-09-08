#!/usr/bin/env python3
"""Standalone research figures from audited JSON/CSV (no GPU needed)."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"A": "#186a9d", "B": "#d16b36"}
MODES = ["initial", "teacher", "detached", "unroll", "initial_repeat"]
LABELS = {"initial": "Initial", "teacher": "Reference\nstates", "detached": "Detached\nrollout", "unroll": "Full\nunroll", "initial_repeat": "Repeated\ninitial"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--legacy", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((args.analysis / "analysis.json").read_text())
    rows = list(csv.DictReader((args.analysis / "all_checkpoints.csv").open()))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for i, scheme in enumerate(("euler", "midpoint")):
        group = [g for g in analysis["aggregates"] if g["scheme"] == scheme and g["mode"] != "oracle" and g["length"] == 4 and "wave" in g["root"]]
        group.sort(key=lambda g: MODES.index(g["mode"]))
        for j, metric in enumerate(("primary_mse", "generator_grid_mse")):
            ax = axes[i, j]
            for label, offset in (("A", -.10), ("B", .10)):
                values = [g[label][metric] for g in group]
                ax.plot(np.arange(len(group)) + offset, values, "o-", color=COLORS[label], label=label)
                for k, g in enumerate(group):
                    field = "primary_mse" if j == 0 else "generator_grid_mse"
                    seeds = [float(r[field]) for r in rows if r["root"] == g["root"] and r["mode"] == g["mode"] and r["model"] == label and r["checkpoint"] == "best_rollout"]
                    ax.scatter(np.full(len(seeds), k + offset), seeds, color=COLORS[label], alpha=.3, s=14)
            ax.set_yscale("log"); ax.set_xticks(range(len(group)), [LABELS[g["mode"]] for g in group])
            ax.set_title(scheme.capitalize() + (": long-horizon prediction" if j == 0 else ": source identification"))
            ax.set_ylabel("MSE (geometric mean over seeds)"); ax.grid(axis="y", alpha=.2); ax.legend()
        ax = axes[i, 2]
        for k, g in enumerate(group):
            ax.scatter([k] * len(g["pairs"]), [v["ratio"] for v in g["pairs"]], color="#6d7580", alpha=.6, s=24)
            ax.scatter(k, g["ratio_a_over_b"], marker="D", color="#182b42", s=40)
        ax.axhline(1, color="black", linewidth=.7)
        ax.axhline(.9, color="#39815e", linestyle="--", linewidth=1, label="10% improvement")
        ax.set_yscale("log"); ax.set_xticks(range(len(group)), [LABELS[g["mode"]] for g in group])
        ax.set_title(scheme.capitalize() + ": A / B prediction MSE")
        ax.set_ylabel("Ratio (below 1 favors A)"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Generator coverage screen | 3 seeds | equal reaction-call budget\nExploratory; checkpoints selected only by independent short-rollout validation", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / ("coverage_comparison." + extension))
    plt.close(fig)
    extension_roots = [name for name in ("long-L4", "rate-L4", "long-L8") if any(g["root"] == name for g in analysis["aggregates"])]
    if extension_roots:
        fig, axes = plt.subplots(len(extension_roots), 3, figsize=(14, 3.6 * len(extension_roots)), squeeze=False, layout="constrained")
        for i, name in enumerate(extension_roots):
            group = [g for g in analysis["aggregates"] if g["root"] == name]
            group.sort(key=lambda g: MODES.index(g["mode"]))
            for j, metric in enumerate(("primary_mse", "generator_grid_mse")):
                ax = axes[i, j]
                for label, offset in (("A", -.10), ("B", .10)):
                    ax.plot(np.arange(len(group)) + offset, [g[label][metric] for g in group], "o-", color=COLORS[label], label=label)
                    for k, g in enumerate(group):
                        values = [float(r[metric]) for r in rows if r["root"] == name and r["mode"] == g["mode"] and r["model"] == label and r["checkpoint"] == "best_rollout"]
                        ax.scatter(np.full(len(values), k + offset), values, color=COLORS[label], alpha=.3, s=14)
                ax.set_yscale("log"); ax.set_xticks(range(len(group)), [LABELS[g["mode"]] for g in group])
                ax.set_title(name + (": prediction" if j == 0 else ": source identification")); ax.set_ylabel("MSE")
                ax.grid(axis="y", alpha=.2); ax.legend()
            ax = axes[i, 2]
            for k, g in enumerate(group):
                ax.scatter([k] * 3, [v["ratio"] for v in g["pairs"]], color="#6d7580", alpha=.6, s=24)
                ax.scatter(k, g["ratio_a_over_b"], marker="D", color="#182b42", s=40)
            ax.axhline(1, color="black", linewidth=.7); ax.axhline(.9, color="#39815e", linestyle="--", linewidth=1)
            ax.set_yscale("log"); ax.set_xticks(range(len(group)), [LABELS[g["mode"]] for g in group])
            ax.set_title(name + ": A / B prediction MSE"); ax.set_ylabel("Ratio (below 1 favors A)"); ax.grid(axis="y", alpha=.2)
        fig.suptitle("Longer optimization, lag weighting, and trajectory depth | 2000 updates | 3 seeds\nFresh data relative to the 400-update screen; exploratory comparisons", fontsize=13)
        for extension in ("png", "pdf"):
            fig.savefig(args.output / ("extension_comparison." + extension))
        plt.close(fig)
    # Compare the selected checkpoint with the early and final weights.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, scheme in zip(axes, ("euler", "midpoint")):
        subset = [r for r in rows if r["scheme"] == scheme and r["mode"] in MODES[:4] and r["length"] == "4" and "wave" in r["root"]]
        for label in ("A", "B"):
            for check, style in (("epoch120", ":"), ("final", "--"), ("best_rollout", "-")):
                values = []
                for mode in MODES[:4]:
                    nums = [float(r["primary_mse"]) for r in subset if r["mode"] == mode and r["model"] == label and r["checkpoint"] == check]
                    values.append(np.exp(np.mean(np.log(nums))) if nums else np.nan)
                ax.plot(range(4), values, marker="o", linestyle=style, color=COLORS[label], label=label + " " + check)
        ax.set_xticks(range(4), [LABELS[m] for m in MODES[:4]]); ax.set_yscale("log")
        ax.set_title(scheme.capitalize() + ": optimization and selection"); ax.set_ylabel("Primary prediction MSE")
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8, ncol=2)
    for extension in ("png", "pdf"):
        fig.savefig(args.output / ("checkpoint_comparison." + extension))
    plt.close(fig)
    if args.legacy:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        legacy = [json.loads(f.read_text()) for f in args.legacy.glob("*-best.json")]
        for label in ("A", "B"):
            subset = [r for r in legacy if ("B" if r["legacy_config"]["conditioned"] else "A") == label]
            for k, r in enumerate(subset):
                g = r["generator"]
                axes[0].plot(g["states"], g["curves"]["0.15"], color=COLORS[label], alpha=.5, label=label if k == 0 else None)
            if subset:
                occupancy = subset[0]["generator"]["occupancy"]
                for name, linestyle in (("initial", "-"), ("reference_long", "--")):
                    values = np.array(occupancy[name]["histogram_48_bins_minus12_to12"])
                    axes[1].plot(np.linspace(-1.175, 1.175, 48), values / values.sum(), color=COLORS[label], linestyle=linestyle, label=label + " " + name)
        if legacy:
            axes[0].plot(legacy[0]["generator"]["states"], legacy[0]["generator"]["truth"], color="black", label="True reaction")
        axes[0].set_title("Previous 120-update models: reaction curves"); axes[0].set_ylabel("Reaction rate")
        axes[1].set_title("Previous experiment: visited-state distributions"); axes[1].set_ylabel("Fraction per bin")
        for ax in axes:
            ax.set_xlabel("State u"); ax.legend(fontsize=8); ax.grid(alpha=.2)
        for extension in ("png", "pdf"):
            fig.savefig(args.output / ("legacy_generator_diagnosis." + extension))
        plt.close(fig)


if __name__ == "__main__":
    main()
