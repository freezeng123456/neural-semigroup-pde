#!/usr/bin/env python3
"""Pool the work-matched Burgers control and apply its frozen rules.

Read-only.  Refuses inputs recorded at another architecture or at an unmatched
flux-evaluation budget, so these numbers cannot be pooled with the unmatched
direct-map control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_SEEDS = {31415, 271828, 161803}
ARCHITECTURE = "r0_h8_l1_anchor"
MATERIAL_RATIO = 0.90
MATCHED_FLUX_EVALUATIONS_PER_CALL = 1
REFINEMENT_SWEEP = ("1", "2", "4", "8", "16")
REFINEMENT_GAIN = 10.0
SEPARATION_FACTOR = 100.0
FLOW_LABELS = ("a_euler_autonomous", "b_euler_query_time")

# Unmatched-work reference, for comparison only.
UNMATCHED_MSE_A_OVER_C = 0.6992492034001871
UNMATCHED_MSE_B_OVER_C = 0.6779596387932305


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _validate(summaries):
    if {summary["seed"] for summary in summaries} != EXPECTED_SEEDS:
        raise ValueError("expected the frozen three-seed matrix")
    for summary in summaries:
        if (
            summary.get("experiment") != "burgers_work_matched_control"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("summary classification mismatch")
        if summary.get("architecture") != ARCHITECTURE:
            raise ValueError(f"this lane only accepts architecture {ARCHITECTURE}")
        budgets = set(summary["flux_evaluations_per_call"].values())
        if budgets != {MATCHED_FLUX_EVALUATIONS_PER_CALL}:
            raise ValueError(f"work is not matched: {budgets}")
        for label in FLOW_LABELS:
            for row in summary["equal_substep_cross_lag"][label].values():
                if label == "a_euler_autonomous" and row["rms_defect_mean"] != 0.0:
                    raise ValueError(
                        "the autonomous field must compose exactly at equal substeps"
                    )
    digests = {summary["data_cache_sha256"] for summary in summaries}
    if len(digests) != 1:
        raise ValueError("all seeds must read one immutable data cache")
    return digests.pop()


def _refinement_report(summaries):
    report = {}
    monotone = True
    gain_met = True
    separated = True
    for label in FLOW_LABELS:
        pooled = {}
        for substeps in REFINEMENT_SWEEP:
            pooled[substeps] = geometric_mean(
                [
                    row["rms_defect_mean"]
                    for summary in summaries
                    for row in summary["refinement_sweep_cross_lag"][label][
                        substeps
                    ].values()
                ]
            )
        report[label] = {
            "pooled_defect_by_substeps": pooled,
            "gain_from_one_to_sixteen": pooled["1"] / pooled["16"],
        }
        for summary in summaries:
            sweep = summary["refinement_sweep_cross_lag"][label]
            horizons = list(sweep[REFINEMENT_SWEEP[0]])
            for horizon in horizons:
                series = [
                    sweep[substeps][horizon]["rms_defect_mean"]
                    for substeps in REFINEMENT_SWEEP
                ]
                monotone &= all(
                    later < earlier for earlier, later in zip(series, series[1:])
                )
                gain_met &= series[0] >= REFINEMENT_GAIN * series[-1]
                direct = summary["deployed_budget_cross_lag"]["c_direct_map"][horizon][
                    "rms_defect_mean"
                ]
                separated &= direct >= SEPARATION_FACTOR * series[-1]
    report["monotone_in_every_cell"] = monotone
    report["refinement_gain_met_in_every_cell"] = gain_met
    report["direct_map_separated_in_every_cell"] = separated
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summaries
    ]
    cache_digest = _validate(summaries)

    def pooled(key: str):
        return geometric_mean(
            [
                value
                for summary in summaries
                for value in summary["comparison"][key]["cells"].values()
            ]
        )

    a_over_c = pooled("mse_a_over_c")
    b_over_c = pooled("mse_b_over_c")
    per_seed = {
        summary["seed"]: summary["comparison"]["mse_a_over_c"]["geometric_mean"]
        for summary in summaries
    }
    seeds_meeting = sum(1 for value in per_seed.values() if value <= MATERIAL_RATIO)
    accuracy_conclusion = (
        "flow_retains_accuracy_advantage_at_matched_work"
        if a_over_c <= MATERIAL_RATIO and seeds_meeting >= 2
        else "flow_accuracy_advantage_does_not_survive_work_matching"
    )

    refinement = _refinement_report(summaries)
    refinability_conclusion = (
        "flow_defect_refines_direct_map_defect_does_not"
        if refinement["monotone_in_every_cell"]
        and refinement["refinement_gain_met_in_every_cell"]
        and refinement["direct_map_separated_in_every_cell"]
        else "refinement_does_not_separate_the_formulations"
    )

    deployed = {
        label: geometric_mean(
            [
                row["rms_defect_mean"]
                for summary in summaries
                for row in summary["deployed_budget_cross_lag"][label].values()
            ]
        )
        for label in ("a_euler_autonomous", "b_euler_query_time", "c_direct_map")
    }
    selection = {
        label: geometric_mean(
            [summary["models"][label]["best_val_mse"] for summary in summaries]
        )
        for label in FLOW_LABELS
    }
    training_seconds = {
        label: sum(summary["models"][label]["training_seconds"] for summary in summaries)
        / len(summaries)
        for label in FLOW_LABELS
    }

    aggregate = {
        "experiment": "burgers_work_matched_control_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "architecture": ARCHITECTURE,
        "seeds": sorted(summary["seed"] for summary in summaries),
        "data_cache_sha256": cache_digest,
        "parameter_count_each": summaries[0]["parameter_count_each"],
        "flux_evaluations_per_call": MATCHED_FLUX_EVALUATIONS_PER_CALL,
        "material_ratio_threshold": MATERIAL_RATIO,
        "rollout_mse_ratio_a_over_c": a_over_c,
        "rollout_mse_ratio_b_over_c": b_over_c,
        "per_seed_a_over_c": {
            str(seed): value for seed, value in sorted(per_seed.items())
        },
        "seeds_meeting_material_threshold": seeds_meeting,
        "deployed_budget_cross_lag_geometric_mean": deployed,
        "one_step_selection_metric_geometric_mean": selection,
        "training_seconds_mean": training_seconds,
        "refinement": refinement,
        "unmatched_work_reference": {
            "rollout_mse_ratio_a_over_c": UNMATCHED_MSE_A_OVER_C,
            "rollout_mse_ratio_b_over_c": UNMATCHED_MSE_B_OVER_C,
            "flow_flux_evaluations_per_call": 48,
        },
        "accuracy_conclusion": accuracy_conclusion,
        "refinability_conclusion": refinability_conclusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
