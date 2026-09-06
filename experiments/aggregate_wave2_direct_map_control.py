#!/usr/bin/env python3
"""Pool the Wave 2 work-matched direct-map control and apply its frozen rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from boundary_family_semigroup import BOUNDARY_FAMILIES, TRAINING_SEEDS
from boundary_family_direct_map import (
    EXPECTED_PARAMETERS,
    MATCHED_NET_EVALUATIONS,
    REFINEMENT_SWEEP,
)

MATERIAL_RATIO = 0.90
AUTONOMOUS_REFINEMENT_GAIN = 64.0
QUERY_IRREDUCIBLE_GAIN = 8.0
STRUCTURAL_SEPARATION = 16.0
SWEEP = tuple(str(value) for value in REFINEMENT_SWEEP)


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
    cache_by_family = {}
    for summary in summaries:
        if (
            summary.get("experiment") != "wave2_direct_map_work_matched"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
            or summary.get("enforcement_mode") != "hard"
            or summary.get("smoke_only") is True
        ):
            raise ValueError("summary classification mismatch")
        if summary.get("parameter_count_each") != EXPECTED_PARAMETERS:
            raise ValueError("architecture mismatch")
        matched = {
            summary["net_evaluations_per_call"][label]
            for label in ("a_euler_autonomous", "b_euler_query_time", "c_direct_map")
        }
        if matched != {MATCHED_NET_EVALUATIONS}:
            raise ValueError(f"matched work is not 1: {matched}")
        if set(summary["refinement_sweep_cross_lag"]["a_euler_autonomous"]) != set(SWEEP):
            raise ValueError("autonomous refinement sweep is incomplete")
        equal = summary["equal_substep_cross_lag"]["a_euler_autonomous"]
        if float(equal["rms_defect_mean"]) != 0.0:
            raise ValueError("autonomous equal-substep defect must be exactly zero")
        family = summary["boundary_family"]
        digest = summary["data_cache_sha256"]
        cache_by_family.setdefault(family, digest)
        if cache_by_family[family] != digest:
            raise ValueError(f"family {family} used more than one cache")
    return cache_by_family


def _family_summaries(summaries, family: str):
    return [summary for summary in summaries if summary["boundary_family"] == family]


def _accuracy_report(summaries):
    report = {}
    for family in BOUNDARY_FAMILIES:
        rows = _family_summaries(summaries, family)
        a_over_c = [
            summary["comparison"]["mse_a_prime_over_c"]["geometric_mean"]
            for summary in rows
        ]
        b_over_c = [
            summary["comparison"]["mse_b_prime_over_c"]["geometric_mean"]
            for summary in rows
        ]
        unmatched = [
            summary["comparison"]["mse_a_rk4_over_c"]["geometric_mean"]
            for summary in rows
        ]
        wave2_ab = [
            summary["comparison"]["mse_a_rk4_over_b_rk4"]["geometric_mean"]
            for summary in rows
        ]
        pooled = geometric_mean(a_over_c)
        seeds_meeting = sum(value <= MATERIAL_RATIO for value in a_over_c)
        material = pooled <= MATERIAL_RATIO and seeds_meeting >= 2
        report[family] = {
            "mse_a_prime_over_c": {
                "per_seed": a_over_c,
                "pooled": pooled,
                "seeds_meeting_0.90": seeds_meeting,
            },
            "mse_b_prime_over_c": {
                "per_seed": b_over_c,
                "pooled": geometric_mean(b_over_c),
            },
            "mse_a_rk4_over_c_descriptive": {
                "per_seed": unmatched,
                "pooled": geometric_mean(unmatched),
            },
            "mse_a_rk4_over_b_rk4_descriptive": {
                "per_seed": wave2_ab,
                "pooled": geometric_mean(wave2_ab),
            },
            "verdict": (
                "flow_has_material_accuracy_advantage_over_direct_map"
                if material
                else "flow_has_no_accuracy_advantage_over_direct_map"
            ),
        }
    return report


def _mean_defect(summaries, label: str, substeps: str | None = None):
    values = []
    for summary in summaries:
        if substeps is None:
            values.append(
                summary["deployed_budget_cross_lag"][label]["rms_defect_mean"]
            )
        else:
            values.append(
                summary["refinement_sweep_cross_lag"][label][substeps][
                    "rms_defect_mean"
                ]
            )
    return geometric_mean(values)


def _refinability_report(summaries):
    report = {}
    autonomous_ok = True
    query_irreducible = True
    structural = True
    for family in BOUNDARY_FAMILIES:
        rows = _family_summaries(summaries, family)
        a_series = {
            substeps: _mean_defect(rows, "a_euler_autonomous", substeps)
            for substeps in SWEEP
        }
        b_series = {
            substeps: _mean_defect(rows, "b_euler_query_time", substeps)
            for substeps in SWEEP
        }
        c_deployed = _mean_defect(rows, "c_direct_map")
        a_gain = a_series["1"] / max(a_series["128"], 1e-16)
        b_gain = b_series["1"] / max(b_series["128"], 1e-16)
        separation = c_deployed / max(a_series["128"], 1e-16)
        family_autonomous = a_gain >= AUTONOMOUS_REFINEMENT_GAIN
        family_query = b_gain < QUERY_IRREDUCIBLE_GAIN
        family_structural = separation >= STRUCTURAL_SEPARATION
        autonomous_ok = autonomous_ok and family_autonomous
        query_irreducible = query_irreducible and family_query
        structural = structural and family_structural
        report[family] = {
            "a_prime_refinement": a_series,
            "b_prime_refinement": b_series,
            "c_deployed_defect": c_deployed,
            "a_prime_gain_1_to_128": a_gain,
            "b_prime_gain_1_to_128": b_gain,
            "c_over_a_prime_at_128": separation,
            "autonomous_defect_is_refinable": family_autonomous,
            "query_time_has_irreducible_defect": family_query,
            "structural_ordering_is_specific_to_autonomy": family_structural,
        }
    report["pooled_verdicts"] = {
        "autonomous_defect_is_refinable": autonomous_ok,
        "query_time_has_irreducible_defect": query_irreducible,
        "structural_ordering_is_specific_to_autonomy": structural,
    }
    return report


def aggregate(paths):
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    caches = _validate(summaries)
    accuracy = _accuracy_report(summaries)
    refinability = _refinability_report(summaries)
    payload = {
        "experiment": "wave2_direct_map_work_matched",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "n_summaries": len(summaries),
        "cache_sha256_by_family": caches,
        "accuracy": accuracy,
        "refinability": refinability,
        "accuracy_verdicts": {
            family: accuracy[family]["verdict"] for family in BOUNDARY_FAMILIES
        },
        "any_family_has_material_accuracy_advantage": any(
            accuracy[family]["verdict"]
            == "flow_has_material_accuracy_advantage_over_direct_map"
            for family in BOUNDARY_FAMILIES
        ),
    }
    return payload


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
    print(json.dumps(payload["refinability"]["pooled_verdicts"], indent=2))


if __name__ == "__main__":
    main()
