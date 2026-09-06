#!/usr/bin/env python3
"""Pool the Wave 2 physics-split control and apply its frozen rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from boundary_family_direct_map import EXPECTED_PARAMETERS, MATCHED_NET_EVALUATIONS
from boundary_family_semigroup import BOUNDARY_FAMILIES, TRAINING_SEEDS

MATERIAL_RATIO = 0.90
RESIDUAL_GAP_RATIO = 0.50


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _validate(summaries):
    families = {summary["boundary_family"] for summary in summaries}
    seeds = {summary["seed"] for summary in summaries}
    if families != set(BOUNDARY_FAMILIES):
        raise ValueError(f"expected all three families, got {sorted(families)}")
    if seeds != set(TRAINING_SEEDS):
        raise ValueError(f"expected the frozen three-seed matrix, got {sorted(seeds)}")
    if len(summaries) != len(BOUNDARY_FAMILIES) * len(TRAINING_SEEDS):
        raise ValueError("expected one summary per family-seed cell")
    for summary in summaries:
        if (
            summary.get("experiment") != "wave2_physics_split_control"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
            or summary.get("smoke_only") is True
            or summary.get("parameter_count_each") != EXPECTED_PARAMETERS
        ):
            raise ValueError("summary classification mismatch")
        if set(summary["net_evaluations_per_call"].values()) != {MATCHED_NET_EVALUATIONS}:
            raise ValueError("work is not matched")
        if "c_physics_split" not in summary["models"]:
            raise ValueError("missing trained physics-split record")
        if set(summary["frozen_checkpoint_sha256"]) != {
            "a_euler_autonomous",
            "b_euler_query_time",
            "c_direct_map",
        }:
            raise ValueError("frozen checkpoint set mismatch")
    return {
        family: {row["data_cache_sha256"] for row in summaries if row["boundary_family"] == family}.pop()
        for family in BOUNDARY_FAMILIES
    }


def aggregate(paths):
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    caches = _validate(summaries)
    report = {}
    for family in BOUNDARY_FAMILIES:
        rows = [row for row in summaries if row["boundary_family"] == family]
        a_over = [
            row["comparison"]["mse_a_prime_over_c_star"]["geometric_mean"]
            for row in rows
        ]
        b_over = [
            row["comparison"]["mse_b_prime_over_c_star"]["geometric_mean"]
            for row in rows
        ]
        star_over_residual = [
            row["comparison"]["mse_c_star_over_c_residual"]["geometric_mean"]
            for row in rows
        ]
        pooled_a = geometric_mean(a_over)
        pooled_gap = geometric_mean(star_over_residual)
        seeds_meeting = sum(value <= MATERIAL_RATIO for value in a_over)
        report[family] = {
            "mse_a_prime_over_c_star": {
                "per_seed": a_over,
                "pooled": pooled_a,
                "seeds_meeting_0.90": seeds_meeting,
            },
            "mse_b_prime_over_c_star": {
                "per_seed": b_over,
                "pooled": geometric_mean(b_over),
            },
            "mse_c_star_over_c_residual": {
                "per_seed": star_over_residual,
                "pooled": pooled_gap,
            },
            "accuracy_verdict": (
                "flow_keeps_material_advantage_over_physics_split_map"
                if pooled_a <= MATERIAL_RATIO and seeds_meeting >= 2
                else "flow_has_no_material_advantage_over_physics_split_map"
            ),
            "residual_gap_verdict": (
                "physics_split_closes_most_of_the_residual_gap"
                if pooled_gap <= RESIDUAL_GAP_RATIO
                else "physics_split_does_not_close_the_residual_gap"
            ),
        }
    return {
        "experiment": "wave2_physics_split_control",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "n_summaries": len(summaries),
        "cache_sha256_by_family": caches,
        "families": report,
        "accuracy_verdicts": {
            family: report[family]["accuracy_verdict"] for family in BOUNDARY_FAMILIES
        },
        "residual_gap_verdicts": {
            family: report[family]["residual_gap_verdict"] for family in BOUNDARY_FAMILIES
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = aggregate(args.summaries)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["accuracy_verdicts"], indent=2))
    print(json.dumps(payload["residual_gap_verdicts"], indent=2))


if __name__ == "__main__":
    main()
