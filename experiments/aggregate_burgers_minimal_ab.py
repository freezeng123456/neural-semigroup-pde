#!/usr/bin/env python3
"""Pool the minimal-generator Burgers A/B run and apply its frozen rules.

Read-only.  Refuses any input not recorded at the recommended architecture, so
these numbers can never be pooled with the oversized-architecture screen or
attribution results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_SEEDS = {31415, 271828, 161803}
ARCHITECTURE = "r0_h8_l1_anchor"
MATERIAL_RATIO = 0.90

# Values measured on the oversized screen architecture, for comparison only.
OVERSIZED_MSE_RATIO = 0.9933175551831502
OVERSIZED_CROSS_LAG_DEFECT = 0.0057517971601788375
OVERSIZED_CROSS_LAG_RELATIVE = 0.012586344440964764


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summaries
    ]
    if {summary["seed"] for summary in summaries} != EXPECTED_SEEDS:
        raise ValueError("expected the frozen three-seed matrix")
    for summary in summaries:
        if (
            summary.get("experiment") != "burgers_minimal_generator_ab"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("summary classification mismatch")
        if summary.get("architecture") != ARCHITECTURE:
            raise ValueError(
                f"this lane only accepts architecture {ARCHITECTURE}"
            )
        models = summary["models"]
        if (
            models["a_autonomous"]["parameter_count"]
            != models["b_query_time"]["parameter_count"]
        ):
            raise ValueError("paired models must have identical parameter counts")
    digests = {summary["data_cache_sha256"] for summary in summaries}
    if len(digests) != 1:
        raise ValueError("all seeds must read one immutable data cache")

    per_seed_mse = {
        summary["seed"]: float(
            summary["comparison"]["mse_a_over_b_geometric_mean"]
        )
        for summary in summaries
    }
    cell_ratios = {
        f"s{summary['seed']}:{cell}": float(value)
        for summary in summaries
        for cell, value in summary["comparison"]["mse_a_over_b"].items()
    }
    mse_gm = geometric_mean(list(cell_ratios.values()))
    seeds_favoring_a = sum(1 for value in per_seed_mse.values() if value <= MATERIAL_RATIO)
    accuracy_conclusion = (
        "accuracy_conclusion_changed"
        if mse_gm <= MATERIAL_RATIO and seeds_favoring_a >= 2
        else "accuracy_conclusion_unchanged"
    )

    verdicts = [
        value
        for summary in summaries
        for value in summary["in_range_cross_lag_verdict"].values()
    ]
    structural = sum(1 for value in verdicts if value == "structural")
    per_seed_structural = {
        summary["seed"]: all(
            value == "structural"
            for value in summary["in_range_cross_lag_verdict"].values()
        )
        for summary in summaries
    }
    structure_conclusion = (
        "structure_conclusion_unchanged"
        if all(per_seed_structural.values())
        else "structure_conclusion_changed"
    )

    floor = geometric_mean(
        [
            row["equal_work_rms_defect_mean"]
            for summary in summaries
            for row in summary["composition_paths"]["a_autonomous"][
                "path2_matched_conditioning"
            ].values()
        ]
    )
    cross_lag = geometric_mean(
        [
            row["equal_work_rms_defect_mean"]
            for summary in summaries
            for row in summary["composition_paths"]["b_query_time"][
                "path3_in_range_cross_lag"
            ].values()
        ]
    )
    state_rms = geometric_mean(
        [
            row["composed_state_rms"]
            for summary in summaries
            for row in summary["composition_paths"]["b_query_time"][
                "path3_in_range_cross_lag"
            ].values()
        ]
    )
    relative = cross_lag / state_rms

    adopt = (
        accuracy_conclusion == "accuracy_conclusion_unchanged"
        and structure_conclusion == "structure_conclusion_unchanged"
    )
    aggregate = {
        "experiment": "burgers_minimal_generator_ab_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "architecture": ARCHITECTURE,
        "seeds": sorted(summary["seed"] for summary in summaries),
        "data_cache_sha256": digests.pop(),
        "parameter_count_each": summaries[0]["parameter_count_each"],
        "material_ratio_threshold": MATERIAL_RATIO,
        "rollout_mse_ratio_geometric_mean": mse_gm,
        "per_seed_rollout_mse_ratio": {
            str(seed): value for seed, value in sorted(per_seed_mse.items())
        },
        "cell_rollout_mse_ratios": cell_ratios,
        "cells_favoring_autonomous": sum(
            1 for value in cell_ratios.values() if value < 1.0
        ),
        "seeds_meeting_material_threshold": seeds_favoring_a,
        "autonomous_integrator_floor_geometric_mean": floor,
        "autonomous_cross_lag_defect": 0.0,
        "query_time_in_range_cross_lag_defect": cross_lag,
        "query_time_in_range_defect_over_floor": cross_lag / floor,
        "query_time_in_range_defect_relative_to_state_rms": relative,
        "in_range_structural_cells": f"{structural}/{len(verdicts)}",
        "oversized_architecture_reference": {
            "rollout_mse_ratio_geometric_mean": OVERSIZED_MSE_RATIO,
            "in_range_cross_lag_defect": OVERSIZED_CROSS_LAG_DEFECT,
            "in_range_defect_relative_to_state_rms": OVERSIZED_CROSS_LAG_RELATIVE,
        },
        "accuracy_conclusion": accuracy_conclusion,
        "structure_conclusion": structure_conclusion,
        "adopt_reduction": adopt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
