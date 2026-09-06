#!/usr/bin/env python3
"""Aggregate the three frozen Fisher generator-consistency pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    pairs = [json.loads(path.read_text(encoding="utf-8")) for path in args.pairs]
    if {pair.get("seed") for pair in pairs} != {31415, 271828, 161803}:
        raise ValueError("expected the frozen three-seed matrix")
    if any(
        pair.get("experiment") != "fisher_generator_consistency_pair"
        or pair.get("exploratory") is not True
        or pair.get("do_not_use_for_formal") is not True
        for pair in pairs
    ):
        raise ValueError("pair classification mismatch")

    generator_ratio_key = (
        "learned_path_generator_residual_ratio"
        if all(
            "learned_path_generator_residual_ratio" in pair["comparison"]
            for pair in pairs
        )
        else "generator_residual_ratio"
    )
    generator_ratios = [pair["comparison"][generator_ratio_key] for pair in pairs]
    mse_ratios = [
        ratio
        for pair in pairs
        for ratio in pair["comparison"]["rollout_mse_ratios"].values()
    ]
    generator_gm = geometric_mean(generator_ratios)
    mse_gm = geometric_mean(mse_ratios)
    if generator_gm > 0.90:
        conclusion = "generator_term_not_materially_improved"
    elif mse_gm < 1.0:
        conclusion = "generator_matching_helped_prediction"
    else:
        conclusion = "generator_matching_not_sufficient"

    result = {
        "experiment": "fisher_generator_consistency_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seeds": sorted(pair["seed"] for pair in pairs),
        "generator_residual_ratio_geometric_mean": generator_gm,
        "primary_generator_ratio_key": generator_ratio_key,
        "rollout_mse_ratio_geometric_mean": mse_gm,
        "generator_residual_ratios": {
            str(pair["seed"]): pair["comparison"][generator_ratio_key]
            for pair in pairs
        },
        "per_seed_mse_ratio_geometric_mean": {
            str(pair["seed"]): pair["comparison"]["mse_ratio_geometric_mean"]
            for pair in pairs
        },
        "conclusion": conclusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
