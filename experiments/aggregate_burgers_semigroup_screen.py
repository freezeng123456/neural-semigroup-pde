#!/usr/bin/env python3
"""Aggregate the frozen three-seed matched Burgers semigroup screen.

The aggregation is read-only.  It accepts the three per-seed summaries only
when they share one immutable data cache, keep the paired parameter counts
identical, cover the complete frozen lag/horizon grid, and give both paths of
every composition cell the same number of right-hand-side evaluations.

The 10% material threshold is not introduced here.  It is the project-wide
convention already fixed in ``AUTO_RESEARCH_PROTOCOL.md`` and Section 4 of
``docs/research/SEMIGROUP_ATTRIBUTION_AND_TRANSFER_PROTOCOL.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_SEEDS = {31415, 271828, 161803}
EXPECTED_CELLS = (
    "tau=0.04:horizon=0.4",
    "tau=0.04:horizon=0.8",
    "tau=0.08:horizon=0.4",
    "tau=0.08:horizon=0.8",
)
MATERIAL_RATIO = 0.90


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _check_classification(summaries):
    if {summary.get("seed") for summary in summaries} != EXPECTED_SEEDS:
        raise ValueError("expected the frozen three-seed matrix")
    for summary in summaries:
        if (
            summary.get("experiment") != "burgers_matched_semigroup_screen"
            or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("summary classification mismatch")


def _check_immutable_inputs(summaries):
    digests = {summary["data_cache_sha256"] for summary in summaries}
    if len(digests) != 1:
        raise ValueError("all seeds must read one immutable data cache")
    counts = {summary["parameter_count_each"] for summary in summaries}
    if len(counts) != 1:
        raise ValueError("all seeds must use one parameter budget")
    for summary in summaries:
        models = summary["models"]
        if (
            models["a_autonomous"]["parameter_count"]
            != models["b_query_time"]["parameter_count"]
        ):
            raise ValueError("paired models must have identical parameter counts")
        if models["a_autonomous"]["query_conditioned"] is not False:
            raise ValueError("model A must be the autonomous variant")
        if models["b_query_time"]["query_conditioned"] is not True:
            raise ValueError("model B must be the query-time variant")
    return digests.pop(), counts.pop()


def _check_grid_and_work(summaries):
    for summary in summaries:
        for model in summary["models"].values():
            evaluation = model["evaluation"]
            if tuple(evaluation["rollouts"]) != EXPECTED_CELLS:
                raise ValueError("incomplete rollout grid")
            if tuple(evaluation["composition"]) != EXPECTED_CELLS:
                raise ValueError("incomplete composition grid")
            for row in evaluation["rollouts"].values():
                if not all(
                    math.isfinite(float(value)) for value in row.values()
                ):
                    raise ValueError("non-finite rollout statistic")
        for cell in EXPECTED_CELLS:
            work = {
                model["evaluation"]["composition"][cell]["rhs_evaluations_each_path"]
                for model in summary["models"].values()
            }
            if len(work) != 1:
                raise ValueError(f"unmatched RHS work in cell {cell}")


def _unseen_lag_spread(summaries, label):
    """Largest relative spread of rollout MSE across lags at a fixed horizon.

    One generator applied with matched work per unit time should predict the
    same state whatever intermediate lag is requested.  This is a post-hoc
    descriptive diagnostic and takes no part in the decision rule.
    """

    spreads = []
    for summary in summaries:
        rollouts = summary["models"][label]["evaluation"]["rollouts"]
        for horizon in ("0.4", "0.8"):
            values = [
                float(row["rollout_mse_mean"])
                for cell, row in rollouts.items()
                if cell.endswith(f"horizon={horizon}")
            ]
            spreads.append((max(values) - min(values)) / min(values))
    return max(spreads)


def _model_rows(summaries, label, field, cell_field="rollouts"):
    return {
        (summary["seed"], cell): float(
            summary["models"][label]["evaluation"][cell_field][cell][field]
        )
        for summary in summaries
        for cell in EXPECTED_CELLS
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summaries
    ]
    _check_classification(summaries)
    cache_digest, parameter_count = _check_immutable_inputs(summaries)
    _check_grid_and_work(summaries)

    cell_ratios = {
        (summary["seed"], cell): float(summary["comparison"]["mse_a_over_b"][cell])
        for summary in summaries
        for cell in EXPECTED_CELLS
    }
    per_seed_gm = {
        summary["seed"]: geometric_mean(
            [summary["comparison"]["mse_a_over_b"][cell] for cell in EXPECTED_CELLS]
        )
        for summary in summaries
    }
    mse_gm = geometric_mean(list(cell_ratios.values()))

    defect_a = _model_rows(
        summaries, "a_autonomous", "equal_work_rms_defect_mean", "composition"
    )
    defect_b = _model_rows(
        summaries, "b_query_time", "equal_work_rms_defect_mean", "composition"
    )
    drift_a = _model_rows(summaries, "a_autonomous", "mean_drift_max")
    drift_b = _model_rows(summaries, "b_query_time", "mean_drift_max")
    energy_a = _model_rows(summaries, "a_autonomous", "energy_change_mean")
    energy_b = _model_rows(summaries, "b_query_time", "energy_change_mean")
    energy_up_a = _model_rows(summaries, "a_autonomous", "energy_increase_fraction")
    energy_up_b = _model_rows(summaries, "b_query_time", "energy_increase_fraction")

    defect_gm_a = geometric_mean(list(defect_a.values()))
    defect_gm_b = geometric_mean(list(defect_b.values()))
    seeds_favoring_a = sum(1 for value in per_seed_gm.values() if value < 1.0)
    cells_favoring_a = sum(1 for value in cell_ratios.values() if value < 1.0)

    if mse_gm <= MATERIAL_RATIO and seeds_favoring_a >= 2:
        conclusion = "autonomous_materially_more_accurate"
    elif defect_gm_a < defect_gm_b:
        conclusion = "composition_advantage_without_material_accuracy_advantage"
    else:
        conclusion = "no_structural_or_accuracy_advantage"

    result = {
        "experiment": "burgers_matched_semigroup_screen_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "pde": "viscous_burgers_periodic",
        "seeds": sorted(summary["seed"] for summary in summaries),
        "data_cache_sha256": cache_digest,
        "parameter_count_each": parameter_count,
        "material_ratio_threshold": MATERIAL_RATIO,
        "cells": len(cell_ratios),
        "rollout_mse_ratio_geometric_mean": mse_gm,
        "per_seed_rollout_mse_ratio_geometric_mean": {
            str(seed): value for seed, value in sorted(per_seed_gm.items())
        },
        "cell_rollout_mse_ratios": {
            f"s{seed}:{cell}": value for (seed, cell), value in cell_ratios.items()
        },
        "seeds_favoring_autonomous": seeds_favoring_a,
        "cells_favoring_autonomous": cells_favoring_a,
        "equal_work_defect": {
            "geometric_mean_autonomous": defect_gm_a,
            "geometric_mean_query_time": defect_gm_b,
            "ratio_autonomous_over_query_time": defect_gm_a / defect_gm_b,
            "cells_with_lower_autonomous_defect": sum(
                1 for key in defect_a if defect_a[key] < defect_b[key]
            ),
        },
        "spatial_mean_drift_max": {
            "autonomous": max(drift_a.values()),
            "query_time": max(drift_b.values()),
        },
        "post_hoc_diagnostic_unseen_lag_mse_spread_max": {
            "autonomous": _unseen_lag_spread(summaries, "a_autonomous"),
            "query_time": _unseen_lag_spread(summaries, "b_query_time"),
            "note": "descriptive only; not part of the decision rule",
        },
        "quadratic_energy_change_mean": {
            "autonomous_max": max(energy_a.values()),
            "query_time_max": max(energy_b.values()),
            "autonomous_energy_increase_fraction_max": max(energy_up_a.values()),
            "query_time_energy_increase_fraction_max": max(energy_up_b.values()),
        },
        "conclusion": conclusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
