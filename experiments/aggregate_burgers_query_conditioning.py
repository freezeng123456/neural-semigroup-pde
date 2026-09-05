#!/usr/bin/env python3
"""Pool the per-seed Burgers query-conditioning attribution results.

Read-only.  The frozen collapse and structural bands live in
``evaluate_burgers_query_conditioning.py``; this script only pools the
per-seed classifications and reports the geometric means behind them.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EXPECTED_SEEDS = {31415, 271828, 161803}


def geometric_mean(values):
    values = [float(value) for value in values]
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("geometric means require positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _defects(results, label, path):
    return [
        float(row["equal_work_rms_defect_mean"])
        for result in results
        for row in result["measurements"][label][path].values()
    ]


def _unanimous(results, path, verdict):
    verdicts = [
        value
        for result in results
        for value in result["query_time_classification"][path].values()
    ]
    return sum(1 for value in verdicts if value == verdict), len(verdicts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.results
    ]
    if {result["seed"] for result in results} != EXPECTED_SEEDS:
        raise ValueError("expected the frozen three-seed matrix")
    for result in results:
        if (
            result.get("experiment") != "burgers_query_conditioning_attribution"
            or result.get("exploratory") is not True
            or result.get("do_not_use_for_formal") is not True
        ):
            raise ValueError("result classification mismatch")
    samples = {result["n_sample"] for result in results}
    if len(samples) != 1:
        raise ValueError("all seeds must use one sample set")
    digests = {result["inputs"]["data_cache_sha256"] for result in results}
    if len(digests) != 1:
        raise ValueError("all seeds must read one immutable data cache")

    autonomous_cross_lag = _defects(
        results, "a_autonomous", "path3_in_range_cross_lag"
    )
    if any(value != 0.0 for value in autonomous_cross_lag):
        # Equal right-hand-side work forces an identical substep size on both
        # cross-lag paths, so the zeroed control channel must make them the
        # same Runge-Kutta trajectory.  A nonzero value means the evaluator or
        # the model changed.
        raise ValueError("autonomous cross-lag paths must agree exactly")

    floor = geometric_mean(
        _defects(results, "a_autonomous", "path2_matched_conditioning")
    )
    query = {
        path: geometric_mean(_defects(results, "b_query_time", path))
        for path in ("path1_frozen_semantics", "path2_matched_conditioning")
    }
    query["path3_in_range_cross_lag"] = geometric_mean(
        _defects(results, "b_query_time", "path3_in_range_cross_lag")
    )
    state_rms = geometric_mean(
        [
            float(row["composed_state_rms"])
            for result in results
            for row in result["measurements"]["b_query_time"][
                "path3_in_range_cross_lag"
            ].values()
        ]
    )

    collapsed = _unanimous(results, "path2_matched_conditioning", "collapsed")
    structural_path1 = _unanimous(results, "path1_frozen_semantics", "structural")
    structural_path3 = _unanimous(results, "path3_in_range_cross_lag", "structural")
    if collapsed[0] == collapsed[1] and structural_path3[0] == structural_path3[1]:
        conclusion = "headline_defect_is_conditioning_artifact_in_range_defect_is_structural"
    elif collapsed[0] == collapsed[1]:
        conclusion = "headline_defect_is_conditioning_artifact_no_in_range_defect"
    else:
        conclusion = "headline_defect_survives_matched_conditioning"

    aggregate = {
        "experiment": "burgers_query_conditioning_attribution_aggregate",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seeds": sorted(result["seed"] for result in results),
        "n_sample": samples.pop(),
        "data_cache_sha256": digests.pop(),
        "autonomous_integrator_floor_geometric_mean": floor,
        "autonomous_cross_lag_defect": 0.0,
        "query_time_defect_geometric_mean": query,
        "query_time_defect_over_floor": {
            path: value / floor for path, value in query.items()
        },
        "matched_conditioning_over_frozen_semantics": (
            query["path2_matched_conditioning"] / query["path1_frozen_semantics"]
        ),
        "in_range_defect_relative_to_state_rms": (
            query["path3_in_range_cross_lag"] / state_rms
        ),
        "composed_state_rms_geometric_mean": state_rms,
        "unanimity": {
            "path1_structural": f"{structural_path1[0]}/{structural_path1[1]}",
            "path2_collapsed": f"{collapsed[0]}/{collapsed[1]}",
            "path3_structural": f"{structural_path3[0]}/{structural_path3[1]}",
        },
        "conclusion": conclusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
