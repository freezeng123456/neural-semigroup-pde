"""Acceptance checks for the Burgers flux-ablation aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.aggregate_burgers_flux_ablation import REFERENCE_VARIANT, main

CELLS = (
    "tau=0.04:horizon=0.4",
    "tau=0.04:horizon=0.8",
    "tau=0.08:horizon=0.4",
    "tau=0.08:horizon=0.8",
)


def variant(
    label,
    *,
    parameters,
    ratio,
    drift=5e-8,
    defect=4e-8,
):
    return {
        "label": label,
        "patch_radius": 0,
        "hidden_width": 8,
        "hidden_layers": 1,
        "viscous_anchor": True,
        "parameter_count": parameters,
        "evaluation": {
            "rollouts": {
                cell: {
                    "rollout_mse_mean": 1e-3 * ratio,
                    "mean_drift_max": drift,
                    "energy_change_mean": -1e-3,
                }
                for cell in CELLS
            },
            "composition": {
                cell: {
                    "equal_work_rms_defect_mean": defect,
                    "rhs_evaluations_each_path": 480,
                }
                for cell in CELLS
            },
        },
    }


def build_summary(seed, *, small_ratio=1.0, small_drift=5e-8, small_defect=4e-8):
    variants = {
        REFERENCE_VARIANT: variant(REFERENCE_VARIANT, parameters=1313, ratio=1.0),
        "r0_h8_l1_anchor": variant(
            "r0_h8_l1_anchor",
            parameters=33,
            ratio=small_ratio,
            drift=small_drift,
            defect=small_defect,
        ),
    }
    return {
        "experiment": "burgers_flux_ablation",
        "exploratory": True,
        "do_not_use_for_formal": True,
        "seed": seed,
        "data_cache_sha256": "cache-digest",
        "variants": variants,
        "rollout_mse_ratio_to_reference": {
            label: {
                "geometric_mean": (
                    1.0 if label == REFERENCE_VARIANT else small_ratio
                )
            }
            for label in variants
        },
    }


def run(tmp_path: Path, summaries):
    paths = []
    for summary in summaries:
        path = tmp_path / f"s{summary['seed']}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        paths.append(path)
    output = tmp_path / "aggregate.json"
    main([str(path) for path in paths] + ["--output", str(output)])
    return json.loads(output.read_text(encoding="utf-8"))


def test_equivalent_reduced_variant_is_recommended(tmp_path):
    result = run(
        tmp_path,
        [build_summary(seed, small_ratio=1.05) for seed in (31415, 271828, 161803)],
    )
    assert result["recommended_variant"] == "r0_h8_l1_anchor"
    assert result["conclusion"] == "screen_architecture_is_oversized"
    assert result["parameter_reduction_vs_reference"] == pytest.approx(1313 / 33)
    assert result["variants"]["r0_h8_l1_anchor"]["removable"] is True


def test_degraded_variant_keeps_the_reference(tmp_path):
    result = run(
        tmp_path,
        [build_summary(seed, small_ratio=1.4) for seed in (31415, 271828, 161803)],
    )
    assert result["recommended_variant"] == REFERENCE_VARIANT
    assert result["conclusion"] == "screen_architecture_size_is_justified"
    assert result["variants"]["r0_h8_l1_anchor"]["pooled_verdict"] == "degraded"


def test_drift_failure_blocks_removal(tmp_path):
    result = run(
        tmp_path,
        [
            build_summary(seed, small_ratio=1.0, small_drift=1e-3)
            for seed in (31415, 271828, 161803)
        ],
    )
    assert result["variants"]["r0_h8_l1_anchor"]["removable"] is False
    assert result["variants"]["r0_h8_l1_anchor"]["structural_failures"]
    assert result["recommended_variant"] == REFERENCE_VARIANT


def test_defect_blowup_blocks_removal(tmp_path):
    result = run(
        tmp_path,
        [
            build_summary(seed, small_ratio=1.0, small_defect=4e-6)
            for seed in (31415, 271828, 161803)
        ],
    )
    assert result["variants"]["r0_h8_l1_anchor"]["removable"] is False
    assert result["recommended_variant"] == REFERENCE_VARIANT


def test_one_bad_seed_blocks_removal(tmp_path):
    """Two favourable seeds must not average away a third degraded seed."""

    summaries = []
    for seed, ratio in zip((31415, 271828, 161803), (0.85, 0.85, 1.35)):
        summary = build_summary(seed)
        summary["rollout_mse_ratio_to_reference"]["r0_h8_l1_anchor"][
            "geometric_mean"
        ] = ratio
        summaries.append(summary)
    result = run(tmp_path, summaries)
    reduced = result["variants"]["r0_h8_l1_anchor"]
    assert reduced["pooled_verdict"] == "equivalent"
    assert reduced["per_seed_verdict"]["161803"] == "degraded"
    assert reduced["removable"] is False
    assert result["recommended_variant"] == REFERENCE_VARIANT


def test_mixed_cache_digests_are_rejected(tmp_path):
    summaries = [build_summary(seed) for seed in (31415, 271828, 161803)]
    summaries[1]["data_cache_sha256"] = "other"
    with pytest.raises(ValueError, match="one immutable data cache"):
        run(tmp_path, summaries)
