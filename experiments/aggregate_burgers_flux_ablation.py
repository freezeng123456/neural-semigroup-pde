#!/usr/bin/env python3
"""Pool the three-seed Burgers flux-generator ablation and apply its rule.

Read-only.  The decision rule is the one frozen in
``docs/research/BURGERS_FLUX_ABLATION_PROTOCOL.md``: a variant is equivalent
inside the project's 10% materiality band, a component is removable only when
the variant removing it is equivalent or improved and passes the structural
check in every seed, and the recommended generator is the lowest-parameter
variant meeting that condition.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_SEEDS = {31415, 271828, 161803}
MATERIAL_BAND = 1.1
DRIFT_LIMIT = 1e-6
DEFECT_FACTOR = 10.0
REFERENCE_VARIANT = "r2_h32_l2_anchor"


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def classify(ratio: float) -> str:
    if ratio > MATERIAL_BAND:
        return "degraded"
    if ratio < 1.0 / MATERIAL_BAND:
        return "improved"
    return "equivalent"


def structural_failures(summaries, label) -> list[str]:
    failures = []
    for summary in summaries:
        variant = summary["variants"][label]
        reference = summary["variants"][REFERENCE_VARIANT]
        for cell, row in variant["evaluation"]["rollouts"].items():
            if float(row["mean_drift_max"]) > DRIFT_LIMIT:
                failures.append(
                    f"s{summary['seed']}:{cell}:mean_drift="
                    f"{row['mean_drift_max']:.3e}"
                )
        for cell, row in variant["evaluation"]["composition"].items():
            limit = DEFECT_FACTOR * float(
                reference["evaluation"]["composition"][cell][
                    "equal_work_rms_defect_mean"
                ]
            )
            if float(row["equal_work_rms_defect_mean"]) > limit:
                failures.append(
                    f"s{summary['seed']}:{cell}:defect="
                    f"{row['equal_work_rms_defect_mean']:.3e}>{limit:.3e}"
                )
    return failures


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
            summary.get("experiment") != "burgers_flux_ablation"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("summary classification mismatch")
    digests = {summary["data_cache_sha256"] for summary in summaries}
    if len(digests) != 1:
        raise ValueError("all seeds must read one immutable data cache")
    labels = {tuple(sorted(summary["variants"])) for summary in summaries}
    if len(labels) != 1:
        raise ValueError("all seeds must cover the same variant grid")
    grid = list(labels.pop())
    if REFERENCE_VARIANT not in grid:
        raise ValueError("the reference variant is missing from the grid")

    variants = {}
    for label in grid:
        parameters = {
            summary["variants"][label]["parameter_count"] for summary in summaries
        }
        if len(parameters) != 1:
            raise ValueError(f"variant {label} has inconsistent parameter counts")
        per_seed = {
            summary["seed"]: summary["rollout_mse_ratio_to_reference"][label][
                "geometric_mean"
            ]
            for summary in summaries
        }
        pooled = geometric_mean(list(per_seed.values()))
        failures = structural_failures(summaries, label)
        verdict = classify(pooled)
        seed_verdicts = {
            str(seed): classify(value) for seed, value in sorted(per_seed.items())
        }
        variants[label] = {
            "parameter_count": parameters.pop(),
            "patch_radius": summaries[0]["variants"][label]["patch_radius"],
            "hidden_width": summaries[0]["variants"][label]["hidden_width"],
            "hidden_layers": summaries[0]["variants"][label]["hidden_layers"],
            "viscous_anchor": summaries[0]["variants"][label]["viscous_anchor"],
            "per_seed_ratio": {str(k): v for k, v in sorted(per_seed.items())},
            "per_seed_verdict": seed_verdicts,
            "pooled_ratio": pooled,
            "pooled_verdict": verdict,
            "structural_failures": failures,
            "removable": (
                verdict in ("equivalent", "improved")
                and not failures
                and all(
                    value in ("equivalent", "improved")
                    for value in seed_verdicts.values()
                )
            ),
        }

    qualifying = {
        label: row
        for label, row in variants.items()
        if row["removable"] and label != REFERENCE_VARIANT
    }
    if qualifying:
        recommended = min(
            qualifying, key=lambda label: qualifying[label]["parameter_count"]
        )
        reduction = (
            variants[REFERENCE_VARIANT]["parameter_count"]
            / variants[recommended]["parameter_count"]
        )
    else:
        recommended = REFERENCE_VARIANT
        reduction = 1.0

    aggregate = {
        "experiment": "burgers_flux_ablation_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seeds": sorted(summary["seed"] for summary in summaries),
        "data_cache_sha256": digests.pop(),
        "reference_variant": REFERENCE_VARIANT,
        "material_band": MATERIAL_BAND,
        "drift_limit": DRIFT_LIMIT,
        "defect_factor": DEFECT_FACTOR,
        "variants": variants,
        "recommended_variant": recommended,
        "parameter_reduction_vs_reference": reduction,
        "conclusion": (
            "screen_architecture_is_oversized"
            if recommended != REFERENCE_VARIANT
            else "screen_architecture_size_is_justified"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
