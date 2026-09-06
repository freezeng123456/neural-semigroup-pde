#!/usr/bin/env python3
"""Aggregate the frozen Allen--Cahn reaction-only direct-map lane."""

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
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("experiment") != "allen_cahn_reaction_only_direct_map":
            raise ValueError(f"{path} is not this experiment")
        summaries.append(summary)
    return summaries


def seed_ratio(summary: dict, key: str) -> float:
    return float(summary["ratios"][key]["geometric_mean"])


def decide(summaries: list[dict]) -> dict:
    a_over = [seed_ratio(row, "mse_a_prime_over_c_rxn") for row in summaries]
    old_over = [seed_ratio(row, "mse_old_c_over_c_rxn") for row in summaries]
    phys_over = [seed_ratio(row, "mse_phys_over_c_rxn") for row in summaries]
    pooled = geometric_mean(a_over)
    seeds_agree = sum(value <= MATERIAL for value in a_over)
    if pooled <= MATERIAL and seeds_agree >= 2:
        accuracy = "field_increment_beats_clean_split"
    else:
        accuracy = "accuracy_gap_was_the_diffusion_double_count"
    return {
        "experiment": "allen_cahn_reaction_only_direct_map_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "pooled": {
            "mse_a_prime_over_c_rxn": pooled,
            "mse_old_c_over_c_rxn": geometric_mean(old_over),
            "mse_phys_over_c_rxn": geometric_mean(phys_over),
            "seeds_a_prime_at_or_below_0.90": seeds_agree,
        },
        "per_seed": [
            {
                "seed": row["seed"],
                "mse_a_prime_over_c_rxn": seed_ratio(row, "mse_a_prime_over_c_rxn"),
                "mse_old_c_over_c_rxn": seed_ratio(row, "mse_old_c_over_c_rxn"),
                "mse_phys_over_c_rxn": seed_ratio(row, "mse_phys_over_c_rxn"),
                "parameter_count_c_rxn": row["models"]["c_reaction_only"][
                    "parameter_count"
                ],
            }
            for row in summaries
        ],
        "decisions": {"accuracy": accuracy},
    }


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = decide(load_summaries(Path(args.root)))
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
