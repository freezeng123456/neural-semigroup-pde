#!/usr/bin/env python3
"""Pool the Burgers direct-map control and apply its frozen rules.

Read-only.  Refuses inputs recorded at any architecture other than the adopted
minimal generator, so these numbers cannot be pooled with the oversized
architecture's results.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_SEEDS = {31415, 271828, 161803}
ARCHITECTURE = "r0_h8_l1_anchor"
MATERIAL_RATIO = 0.90

# Adopted minimal-generator flow values, for comparison only.
FLOW_AUTONOMOUS_CROSS_LAG = 0.0
FLOW_QUERY_CROSS_LAG_RELATIVE = 0.007455667181173939


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
            summary.get("experiment") != "burgers_direct_map_control"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("summary classification mismatch")
        if summary.get("architecture") != ARCHITECTURE:
            raise ValueError(f"this lane only accepts architecture {ARCHITECTURE}")
        if summary["models"]["c_direct_map"]["parameter_count"] != summary[
            "parameter_count_each"
        ]:
            raise ValueError("the direct map must match the shared parameter budget")
    digests = {summary["data_cache_sha256"] for summary in summaries}
    if len(digests) != 1:
        raise ValueError("all seeds must read one immutable data cache")

    def pooled_ratio(key: str):
        return geometric_mean(
            [
                value
                for summary in summaries
                for value in summary["comparison"][key]["cells"].values()
            ]
        )

    a_over_c = pooled_ratio("mse_a_over_c")
    b_over_c = pooled_ratio("mse_b_over_c")
    per_seed_a_over_c = {
        summary["seed"]: summary["comparison"]["mse_a_over_c"]["geometric_mean"]
        for summary in summaries
    }
    seeds_meeting = sum(
        1 for value in per_seed_a_over_c.values() if value <= MATERIAL_RATIO
    )
    accuracy_conclusion = (
        "autonomy_has_material_accuracy_advantage_over_direct_map"
        if a_over_c <= MATERIAL_RATIO and seeds_meeting >= 2
        else "autonomy_has_no_accuracy_advantage_over_direct_map"
    )

    verdicts = [
        value
        for summary in summaries
        for value in summary["in_range_cross_lag_verdict"].values()
    ]
    structural = sum(1 for value in verdicts if value == "structural")
    specificity = (
        "structural_property_is_specific_to_autonomy"
        if structural == len(verdicts)
        else "structural_property_is_not_specific_to_autonomy"
    )

    cross_lag = geometric_mean(
        [
            row["rms_defect_mean"]
            for summary in summaries
            for row in summary["direct_map_composition"]["in_range_cross_lag"].values()
        ]
    )
    state_rms = geometric_mean(
        [
            row["composed_state_rms"]
            for summary in summaries
            for row in summary["direct_map_composition"]["in_range_cross_lag"].values()
        ]
    )
    floor = geometric_mean(
        [
            value
            for summary in summaries
            for value in summary["autonomous_integrator_floor"].values()
        ]
    )
    spreads = {
        label: max(summary["unseen_lag_mse_spread_max"][label] for summary in summaries)
        for label in ("a_autonomous", "b_query_time", "c_direct_map")
    }
    drift = {
        label: max(
            row["mean_drift_max"]
            for summary in summaries
            for row in summary["rollouts"][label].values()
        )
        for label in ("a_autonomous", "b_query_time", "c_direct_map")
    }
    energy = {
        label: max(
            row["energy_change_mean"]
            for summary in summaries
            for row in summary["rollouts"][label].values()
        )
        for label in ("a_autonomous", "b_query_time", "c_direct_map")
    }

    aggregate = {
        "experiment": "burgers_direct_map_control_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "architecture": ARCHITECTURE,
        "seeds": sorted(summary["seed"] for summary in summaries),
        "data_cache_sha256": digests.pop(),
        "parameter_count_each": summaries[0]["parameter_count_each"],
        "flux_evaluations_per_call": summaries[0]["flux_evaluations_per_call"],
        "material_ratio_threshold": MATERIAL_RATIO,
        "rollout_mse_ratio_a_over_c": a_over_c,
        "rollout_mse_ratio_b_over_c": b_over_c,
        "per_seed_a_over_c": {
            str(seed): value for seed, value in sorted(per_seed_a_over_c.items())
        },
        "seeds_meeting_material_threshold": seeds_meeting,
        "autonomous_integrator_floor_geometric_mean": floor,
        "direct_map_in_range_cross_lag_defect": cross_lag,
        "direct_map_in_range_defect_over_floor": cross_lag / floor,
        "direct_map_in_range_defect_relative_to_state_rms": cross_lag / state_rms,
        "in_range_structural_cells": f"{structural}/{len(verdicts)}",
        "unseen_lag_mse_spread_max": spreads,
        "spatial_mean_drift_max": drift,
        "quadratic_energy_change_max": energy,
        "flow_reference": {
            "autonomous_in_range_cross_lag_defect": FLOW_AUTONOMOUS_CROSS_LAG,
            "query_time_in_range_defect_relative_to_state_rms": (
                FLOW_QUERY_CROSS_LAG_RELATIVE
            ),
        },
        "accuracy_conclusion": accuracy_conclusion,
        "structural_specificity_conclusion": specificity,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
