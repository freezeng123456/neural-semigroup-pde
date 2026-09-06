#!/usr/bin/env python3
"""Aggregate the frozen Allen--Cahn work-matched direct-map lane."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SEEDS = (42, 137, 2718)
MATERIAL = 0.90


def geometric_mean(values) -> float:
    logs = [math.log(max(float(value), 1e-30)) for value in values]
    return math.exp(sum(logs) / len(logs))


def load_summaries(root: Path) -> list[dict]:
    summaries = []
    for seed in SEEDS:
        path = root / f"s{seed}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("experiment") != "allen_cahn_work_matched_direct_map":
            raise ValueError(f"{path} is not this experiment")
        if int(summary["seed"]) != seed:
            raise ValueError(f"{path} has seed {summary['seed']}, expected {seed}")
        summaries.append(summary)
    return summaries


def seed_ratio(summary: dict, key: str) -> float:
    payload = summary["ratios"][key]
    if payload is None:
        raise ValueError(f"missing ratio {key}")
    return float(payload["geometric_mean"])


def decide(summaries: list[dict]) -> dict:
    a_over_c = [seed_ratio(row, "mse_a_prime_over_c") for row in summaries]
    b_over_c = [seed_ratio(row, "mse_b_prime_over_c") for row in summaries]
    rk4_over_c = [seed_ratio(row, "mse_a_rk4_over_c") for row in summaries]
    pooled_a = geometric_mean(a_over_c)
    seeds_agree = sum(value <= MATERIAL for value in a_over_c)
    if pooled_a <= MATERIAL and seeds_agree >= 2:
        accuracy = "flow_has_material_accuracy_advantage_over_direct_map"
    else:
        accuracy = "flow_accuracy_advantage_does_not_appear_at_matched_work"

    structure_ok = True
    refinable = True
    per_seed = []
    for summary in summaries:
        equal = float(summary["equal_substep_autonomous"]["rms_defect_mean"])
        defects = summary["deployed_budget_cross_lag"]
        a_def = float(defects["a_euler_autonomous"]["rms_defect_mean"])
        b_def = float(defects["b_euler_query_time"]["rms_defect_mean"])
        c_def = float(defects["c_direct_heat_map"]["rms_defect_mean"])
        if equal > 1e-6 or not (a_def < b_def and a_def < c_def):
            structure_ok = False
        series = [
            float(row["rms_defect_mean"])
            for row in summary["refinement"]["a_euler_autonomous"]
        ]
        if any(later >= earlier for earlier, later in zip(series, series[1:])):
            refinable = False
        per_seed.append(
            {
                "seed": summary["seed"],
                "mse_a_prime_over_c": seed_ratio(summary, "mse_a_prime_over_c"),
                "mse_b_prime_over_c": seed_ratio(summary, "mse_b_prime_over_c"),
                "mse_a_rk4_over_c": seed_ratio(summary, "mse_a_rk4_over_c"),
                "equal_substep_defect": equal,
                "deployed_defects": {
                    "a_euler_autonomous": a_def,
                    "b_euler_query_time": b_def,
                    "c_direct_heat_map": c_def,
                },
                "a_refinement": series,
            }
        )

    unmatched = geometric_mean(rk4_over_c)
    return {
        "experiment": "allen_cahn_work_matched_direct_map_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "pooled": {
            "mse_a_prime_over_c": pooled_a,
            "mse_b_prime_over_c": geometric_mean(b_over_c),
            "mse_a_rk4_over_c": unmatched,
            "seeds_a_prime_at_or_below_0.90": seeds_agree,
        },
        "per_seed": per_seed,
        "decisions": {
            "accuracy_matched_work": accuracy,
            "autonomy_orders_composition_defect": structure_ok,
            "flow_defect_refines_direct_map_defect_does_not": refinable,
            "unmatched_rk4_is_compute_effect_only": unmatched < MATERIAL
            and accuracy != "flow_has_material_accuracy_advantage_over_direct_map",
        },
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = decide(load_summaries(Path(args.root)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
