#!/usr/bin/env python3
"""Verify and aggregate the frozen, exploratory Burgers reference-v2 matrix.

Only completed full runs from the three protocol seeds are accepted. This
reader hashes checkpoint files without unpickling them and keeps zero
composition defects as absolute measurements, never as denominators.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "burgers-paired-v2"
EXPECTED_SEEDS = (31415, 161803, 271828)
ARCHITECTURES = {"small": [0, 8, 1], "large": [2, 32, 2]}
PARAMETERS = {"small": 33, "large": 1313}
TEMPORAL_RULES = ("autonomous", "query")
LEARNED_MODELS = tuple(f"{a}_{t}" for a in ARCHITECTURES for t in TEMPORAL_RULES)
ORACLE = "exact_flux_fd_oracle"
MODELS = (*LEARNED_MODELS, ORACLE)
CELLS = tuple(f"tau={t}:horizon={h}" for t in (.04, .08) for h in (.4, .8))
HORIZONS = ("0.4", "0.8")
SOURCE_FILES = {
    "run_burgers_validated.py", "burgers_reference.py", "prepare_burgers_validated_cache.py",
    "run_burgers_flux_ablation.py", "run_burgers_semigroup_screen.py", "pde_solver.py",
    "seed_utils.py", "experiment_artifacts.py",
}
MATERIAL_RATIO = .90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite statistic")
    if isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _read_json(path: Path, tracked: dict) -> dict:
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    content = path.read_bytes()
    tracked[path] = hashlib.sha256(content).hexdigest()
    value = json.loads(content, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    _finite_tree(value)
    return value


def _number(value, name: str, *, minimum=None, maximum=None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its valid range")
    return float(value)


def _integer(value, name: str, *, minimum=0, maximum=None) -> int:
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer in its valid range")
    return value


def _digest(value, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _same_number(actual, expected, name):
    if not math.isclose(_number(actual, name), expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{name} disagrees with its recorded samples or protocol")


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"incomplete or unexpected {name}")


def _validated_configuration(summary):
    config = summary["configuration"]
    if configuration_digest(config) != _digest(summary["config_sha256"], "configuration hash"):
        raise ValueError("configuration hash mismatch")
    if config.get("architectures") != ARCHITECTURES:
        raise ValueError("architecture specifications differ from protocol")
    settings, cache = config["settings"], config["cache_config"]
    seed = _integer(settings.get("seed"), "seed")
    if seed not in EXPECTED_SEEDS:
        raise ValueError("unexpected seed")
    fixed_settings = {"epochs": 100, "batch_size": 64, "validation_interval": 10,
                      "ode_steps": 12, "lr": .001, "weight_decay": .00001}
    fixed_cache = {"N": 64, "L": 2 * math.pi, "nu": .01, "reference_N": 512,
                   "reference_dt": .0005, "n_train": 1000, "n_val": 50, "n_test": 128}
    for key, value in fixed_settings.items():
        _same_number(settings.get(key), value, f"settings.{key}")
    for key, value in fixed_cache.items():
        _same_number(cache.get(key), value, f"cache_config.{key}")
    for key in ("epochs", "batch_size", "validation_interval", "ode_steps", "threads", "eval_batch_size"):
        _integer(settings.get(key), f"settings.{key}", minimum=1)
    for key in ("N", "reference_N", "n_train", "n_val", "n_test"):
        _integer(cache.get(key), f"cache_config.{key}", minimum=1)
    if settings.get("architectures") != list(ARCHITECTURES):
        raise ValueError("requires both architectures in frozen order")
    expected_lags = {"train_taus": [.025, .05, .075, .1], "test_taus": [.04, .08], "horizons": [.4, .8]}
    if any(cache.get(key) != value for key, value in expected_lags.items()):
        raise ValueError("cache lag/horizon grid differs from protocol")
    if cache.get("data_seeds") != {"train": 2026100301, "val": 2026100302, "test": 2026100303}:
        raise ValueError("data split seeds differ from protocol")
    _same_number(config.get("energy_tolerance"), 1e-8, "energy tolerance")
    normalized = copy.deepcopy(config)
    del normalized["settings"]["seed"]
    normalized["settings"].pop("output_dir", None)
    return seed, normalized


def _sample_list(value, name, count=128, *, allow_none=False):
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{name} must have {count} samples")
    for item in value:
        if item is not None or not allow_none:
            _number(item, name, minimum=0)
    return value


def _error_metrics(row, samples, name):
    mse = _sample_list(samples.get("mse"), f"{name}.mse")
    relative = _sample_list(samples.get("relative_l2"), f"{name}.relative_l2", allow_none=True)
    _same_number(row.get("mse_mean"), math.fsum(mse) / len(mse), f"{name}.mse_mean")
    finite_relative = [value for value in relative if value is not None]
    if row.get("undefined_relative_l2_count") != len(relative) - len(finite_relative):
        raise ValueError(f"{name} undefined relative error count mismatch")
    if finite_relative:
        _same_number(row.get("relative_l2_mean"), math.fsum(finite_relative) / len(finite_relative), f"{name}.relative_l2_mean")
    elif row.get("relative_l2_mean") is not None:
        raise ValueError(f"{name} relative error must be undefined")


def _defects(row, samples, name):
    values = _sample_list(samples, name)
    _same_number(row.get("rms_defect_mean"), math.fsum(values) / len(values), f"{name}.mean")
    _same_number(row.get("rms_defect_max"), max(values), f"{name}.max")


def _validate_evaluation(evaluation, samples):
    sections = ("one_step", "rollouts", "composition_matched_conditioning", "composition_historical_conditioning", "cross_lag", "time_integration_refinement")
    _keys(evaluation, sections, "evaluation sections")
    _keys(samples, ("sample_indices", *sections), "per-sample sections")
    if samples["sample_indices"] != list(range(128)):
        raise ValueError("test sample indices must cover all 128 held-out states")
    for section in sections:
        expected = ("0.04", "0.08") if section == "one_step" else HORIZONS if section == "cross_lag" else CELLS
        _keys(evaluation[section], expected, f"{section} grid")
        _keys(samples[section], expected, f"per-sample {section} grid")
    for lag in ("0.04", "0.08"):
        _error_metrics(evaluation["one_step"][lag], samples["one_step"][lag], f"one_step.{lag}")
    for key in CELLS:
        tau, horizon = [float(part.split("=")[1]) for part in key.split(":")]
        substeps, depth = round(12 * tau / .04), round(horizon / tau)
        work = 4 * substeps * depth
        row, raw = evaluation["rollouts"][key], samples["rollouts"][key]
        _error_metrics(row, raw, key)
        drift = _sample_list(raw.get("mean_drift"), f"{key}.mean_drift")
        _same_number(row.get("mean_drift_max"), max(drift), f"{key}.mean_drift_max")
        monotone = raw.get("energy_monotone")
        if not isinstance(monotone, list) or len(monotone) != 128 or any(type(v) is not bool for v in monotone):
            raise ValueError("energy monotonicity flags must cover all test states")
        _same_number(row.get("energy_monotone_fraction"), sum(monotone) / 128, "energy monotone fraction")
        for field in ("energy_positive_increment_sum_mean", "energy_max_positive_increment"):
            _number(row.get(field), field, minimum=0)
        _number(row.get("energy_horizon_delta_mean"), "energy horizon change")
        _integer(row.get("energy_increase_step_count_total"), "energy increase step count", maximum=128 * depth)
        _same_number(row.get("energy_tolerance"), 1e-8, "energy tolerance")
        if row.get("rk4_substeps_per_call") != substeps or row.get("rhs_evaluations") != work:
            raise ValueError("unmatched rollout RHS work")
        refinement = evaluation["time_integration_refinement"][key]
        raw_refinement = samples["time_integration_refinement"][key]
        refinement_mse = _sample_list(raw_refinement.get("mse_vs_production"), "refinement MSE")
        refinement_rms = _sample_list(raw_refinement.get("rms_vs_production"), "refinement RMS")
        for mse, rms in zip(refinement_mse, refinement_rms):
            _same_number(rms, math.sqrt(mse), "refinement RMS")
        _same_number(refinement.get("mse_vs_production_mean"), math.fsum(refinement_mse) / 128, "refinement MSE mean")
        _same_number(refinement.get("rms_vs_production_mean"), math.fsum(refinement_rms) / 128, "refinement RMS mean")
        _same_number(refinement.get("rms_vs_production_max"), max(refinement_rms), "refinement RMS maximum")
        if refinement.get("production_substeps_per_call") != substeps or refinement.get("refined_substeps_per_call") != 2 * substeps:
            raise ValueError("incorrect time integration refinement work")
        for section in ("composition_matched_conditioning", "composition_historical_conditioning"):
            composition = evaluation[section][key]
            _defects(composition, samples[section][key], f"{section}.{key}")
            conditioning = tau if section == "composition_matched_conditioning" else horizon
            _same_number(composition.get("direct_conditioning_tau"), conditioning, "direct conditioning")
            if composition.get("rhs_evaluations_each_path") != work:
                raise ValueError("unmatched composition RHS work")
    for horizon in HORIZONS:
        row = evaluation["cross_lag"][horizon]
        _defects(row, samples["cross_lag"][horizon], f"cross_lag.{horizon}")
        if row.get("fine_substeps") != 12 or row.get("coarse_substeps") != 24 or row.get("rhs_evaluations_each_path") != round(4 * 12 * float(horizon) / .04):
            raise ValueError("unmatched cross-lag RHS work")


def _read_seed(root: Path, tracked: dict, data_cache=None):
    summary_path = root / "summary.json"
    marker = _read_json(root / "COMPLETE.json", tracked)
    _keys(marker, ("schema_version", "summary_sha256", "cache_sha256", "config_sha256"), "completion marker")
    summary = _read_json(summary_path, tracked)
    if (marker.get("schema_version") != SCHEMA or summary.get("schema_version") != SCHEMA
            or summary.get("scope") != "full" or summary.get("exploratory") is not True
            or summary.get("do_not_use_for_formal") is not True):
        raise ValueError("requires completed full Burgers reference-v2 runs")
    if _digest(marker.get("summary_sha256"), "summary hash") != tracked[summary_path]:
        raise ValueError("completion summary hash mismatch")
    seed, normalized = _validated_configuration(summary)
    if marker["config_sha256"] != summary["config_sha256"]:
        raise ValueError("completion configuration hash mismatch")
    cache_hash = _digest(summary.get("cache_sha256_before"), "cache hash")
    if summary.get("cache_sha256_after") != cache_hash or marker.get("cache_sha256") != cache_hash:
        raise ValueError("immutable cache before/after/completion mismatch")
    cache_path = Path(data_cache if data_cache is not None else summary["configuration"]["settings"]["data_cache"]).expanduser().resolve()
    if not cache_path.is_file() or sha256_file(cache_path) != cache_hash:
        raise ValueError("live cache content does not match the completed run")
    tracked[cache_path] = cache_hash
    expected_hash = summary["configuration"]["settings"].get("expect_cache_sha256")
    if expected_hash is not None and expected_hash != cache_hash:
        raise ValueError("expected cache digest disagrees with run")
    _keys(summary.get("sources"), SOURCE_FILES, "source hash inventory")
    for name, digest in summary["sources"].items():
        _digest(digest, f"source {name}")
    cells = summary.get("cells")
    _keys(cells, MODELS, "model matrix")
    selection = _read_json(root / "selection_complete.json", tracked)
    _keys(selection, LEARNED_MODELS, "selection checkpoint matrix")
    for label, cell in cells.items():
        cell_dir = root / label
        if _read_json(cell_dir / "summary.json", tracked) != cell:
            raise ValueError("cell summary differs from completed seed summary")
        samples = _read_json(cell_dir / "per_sample.json", tracked)
        _validate_evaluation(cell["evaluation"], samples)
        _number(cell.get("eval_seconds"), "evaluation time", minimum=0)
        if label == ORACLE:
            if cell.get("parameter_count") != 0:
                raise ValueError("exact-flux oracle must have no learned parameters")
            continue
        architecture, temporal = label.split("_")
        if cell.get("architecture") != architecture or cell.get("query_conditioned") is not (temporal == "query"):
            raise ValueError("model architecture/temporal identity mismatch")
        if cell.get("parameter_count") != PARAMETERS[architecture]:
            raise ValueError("nominal parameter budget mismatch")
        _digest(cell.get("initial_state_sha256"), "initial state hash")
        _integer(cell.get("selected_epoch"), "selected epoch", minimum=1, maximum=100)
        if cell["selected_epoch"] % 10:
            raise ValueError("selected epoch is outside frozen validation schedule")
        for field in ("best_val_mse", "train_and_validation_seconds", "validation_seconds", "train_seconds_excluding_validation"):
            _number(cell.get(field), field, minimum=0)
        selected_fields = {key: value for key, value in cell.items() if key not in ("eval_seconds", "evaluation")}
        if selection[label] != selected_fields:
            raise ValueError("post-selection training metadata changed")
        for stage in ("selected", "final"):
            checkpoint = cell_dir / f"{stage}.pt"
            digest = _digest(cell.get(f"{stage}_sha256"), f"{stage} checkpoint hash")
            if not checkpoint.is_file() or sha256_file(checkpoint) != digest:
                raise ValueError(f"{stage} checkpoint hash mismatch")
            tracked[checkpoint] = digest
    for architecture in ARCHITECTURES:
        if cells[f"{architecture}_autonomous"]["initial_state_sha256"] != cells[f"{architecture}_query"]["initial_state_sha256"]:
            raise ValueError("paired initial parameter tensors do not match")
    return {"seed": seed, "root": root, "summary": summary, "normalized_configuration": normalized,
            "summary_sha256": tracked[summary_path]}


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("geometric mean requires nonnegative finite values")
    return 0.0 if any(v == 0 for v in values) else math.exp(math.fsum(math.log(v) for v in values) / len(values))


def _mse(run, label, cell):
    return run["summary"]["cells"][label]["evaluation"]["rollouts"][cell]["mse_mean"]


def _ratios(runs, numerator, denominator):
    raw, per_seed = {}, {}
    for run in runs:
        values = []
        for cell in CELLS:
            divisor = _mse(run, denominator, cell)
            if divisor <= 0:
                raise ValueError("MSE ratio denominator must be positive")
            ratio = _mse(run, numerator, cell) / divisor
            _number(ratio, "MSE ratio", minimum=0)
            raw[f"s{run['seed']}:{cell}"] = ratio
            values.append(ratio)
        per_seed[str(run["seed"])] = geometric_mean(values)
    return {"numerator": numerator, "denominator": denominator,
            "geometric_mean_mse_ratio": geometric_mean(raw.values()),
            "per_seed_geometric_mean_mse_ratio": per_seed, "cell_mse_ratios": raw,
            "seeds_favoring_numerator": sum(value < 1 for value in per_seed.values()),
            "cells_favoring_numerator": sum(value < 1 for value in raw.values())}


def _model_metrics(runs, label):
    cells = [run["summary"]["cells"][label] for run in runs]
    evaluations = [cell["evaluation"] for cell in cells]
    rollouts = [row for evaluation in evaluations for row in evaluation["rollouts"].values()]
    result = {
        "parameter_count": cells[0]["parameter_count"],
        "rollout_mse_mean_over_seed_conditions": math.fsum(row["mse_mean"] for row in rollouts) / len(rollouts),
        "rollout_mse_by_condition": {key: {
            "mean_over_seeds": math.fsum(e["rollouts"][key]["mse_mean"] for e in evaluations) / len(evaluations),
            "per_seed": {str(run["seed"]): _mse(run, label, key) for run in runs},
        } for key in CELLS},
        "energy_monotone_fraction_min": min(row["energy_monotone_fraction"] for row in rollouts),
        "spatial_mean_drift_max": max(row["mean_drift_max"] for row in rollouts),
        "energy_max_positive_increment": max(row["energy_max_positive_increment"] for row in rollouts),
        "energy_horizon_delta_mean": math.fsum(row["energy_horizon_delta_mean"] for row in rollouts) / len(rollouts),
        "evaluation_seconds_total": math.fsum(cell["eval_seconds"] for cell in cells),
    }
    for section in ("cross_lag", "composition_matched_conditioning", "composition_historical_conditioning"):
        rows = [row for e in evaluations for row in e[section].values()]
        result[section] = {
            "rms_defect_mean": math.fsum(row["rms_defect_mean"] for row in rows) / len(rows),
            "rms_defect_max": max(row["rms_defect_max"] for row in rows),
            "per_seed_condition_rms_defect_mean": {
                f"s{run['seed']}:{key}": row["rms_defect_mean"]
                for run in runs for key, row in run["summary"]["cells"][label]["evaluation"][section].items()},
        }
    refinement = [row for evaluation in evaluations for row in evaluation["time_integration_refinement"].values()]
    result["time_integration_refinement"] = {
        "mse_vs_production_mean": math.fsum(row["mse_vs_production_mean"] for row in refinement) / len(refinement),
        "rms_vs_production_mean": math.fsum(row["rms_vs_production_mean"] for row in refinement) / len(refinement),
        "rms_vs_production_max": max(row["rms_vs_production_max"] for row in refinement),
        "per_seed_condition_rms_vs_production_mean": {
            f"s{run['seed']}:{key}": row["rms_vs_production_mean"]
            for run in runs for key, row in run["summary"]["cells"][label]["evaluation"]["time_integration_refinement"].items()},
    }
    if label != ORACLE:
        result["selected_epoch_by_seed"] = {str(run["seed"]): run["summary"]["cells"][label]["selected_epoch"] for run in runs}
        result["train_and_validation_seconds_total"] = math.fsum(cell["train_and_validation_seconds"] for cell in cells)
    return result


def aggregate(seed_dirs, data_cache=None):
    """Validate read-only inputs; return a JSON result, CSV rows and receipts."""
    roots = [Path(path).expanduser().resolve() for path in seed_dirs]
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("requires exactly three distinct seed directories")
    tracked = {}
    try:
        runs = sorted((_read_seed(root, tracked, data_cache) for root in roots), key=lambda run: run["seed"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed completed-run artifact: {error}") from error
    if tuple(run["seed"] for run in runs) != EXPECTED_SEEDS:
        raise ValueError("requires exactly the three distinct protocol seeds")
    first = runs[0]
    for run in runs[1:]:
        if run["summary"]["cache_sha256_before"] != first["summary"]["cache_sha256_before"]:
            raise ValueError("seeds did not use the same immutable cache")
        if run["normalized_configuration"] != first["normalized_configuration"]:
            raise ValueError("seed configurations differ beyond seed/output directory")
        if run["summary"]["sources"] != first["summary"]["sources"]:
            raise ValueError("source hashes differ across seeds")
    accuracy = {}
    for architecture in ARCHITECTURES:
        comparison = _ratios(runs, f"{architecture}_autonomous", f"{architecture}_query")
        material = comparison["geometric_mean_mse_ratio"] <= MATERIAL_RATIO and comparison["seeds_favoring_numerator"] >= 2
        accuracy[architecture] = {**comparison, "material_accuracy_advantage": material,
            "decision": "exploratory_material_autonomy_accuracy_advantage" if material else "exploratory_material_autonomy_accuracy_advantage_not_established"}
    sizes = {temporal: _ratios(runs, f"small_{temporal}", f"large_{temporal}") for temporal in TEMPORAL_RULES}
    metrics = {label: _model_metrics(runs, label) for label in MODELS}
    result = {
        "schema_version": "burgers-paired-v2-aggregate", "exploratory": True, "do_not_use_for_formal": True,
        "scope": "full", "seeds": list(EXPECTED_SEEDS), "trained_model_count": 12,
        "oracle_evaluation_count": 3, "learned_rollout_cell_count": 48,
        "material_accuracy_rule": {"geometric_mean_a_over_b_max": MATERIAL_RATIO, "minimum_seeds_favoring_a": 2},
        "accuracy_by_architecture": accuracy, "small_over_large_by_temporal_rule": sizes,
        "model_metrics": metrics, "reference_oracle_mse": metrics[ORACLE]["rollout_mse_by_condition"],
        "provenance": {"cache_sha256": first["summary"]["cache_sha256_before"],
            "source_sha256": first["summary"]["sources"],
            "shared_configuration": first["normalized_configuration"],
            "shared_configuration_sha256": configuration_digest(first["normalized_configuration"]),
            "inputs": [{"seed": run["seed"], "directory": str(run["root"]),
                        "summary_sha256": run["summary_sha256"], "config_sha256": run["summary"]["config_sha256"]} for run in runs]},
        "interpretation": "Fixed-budget, projected N64 Burgers study with independent test data. Composition defects are absolute; projected PDE targets are not assumed to form a closed semigroup. The exact-flux finite-difference oracle is a spatial-discretization comparison, not the fine reference.",
    }
    csv_rows = []
    for run in runs:
        for label in MODELS:
            cell = run["summary"]["cells"][label]
            for condition in CELLS:
                tau, horizon = [part.split("=")[1] for part in condition.split(":")]
                row = cell["evaluation"]["rollouts"][condition]
                architecture, temporal = label.split("_") if label != ORACLE else ("oracle", "fixed")
                ratio_key = f"s{run['seed']}:{condition}"
                csv_rows.append({"seed": run["seed"], "model": label, "tau": tau, "horizon": horizon,
                    "parameter_count": cell["parameter_count"], "selected_epoch": cell.get("selected_epoch", ""),
                    "mse_mean": row["mse_mean"], "relative_l2_mean": row["relative_l2_mean"],
                    "mean_drift_max": row["mean_drift_max"], "energy_monotone_fraction": row["energy_monotone_fraction"],
                    "energy_max_positive_increment": row["energy_max_positive_increment"],
                    "cross_lag_rms_defect_mean": cell["evaluation"]["cross_lag"][horizon]["rms_defect_mean"],
                    "time_refinement_rms_mean": cell["evaluation"]["time_integration_refinement"][condition]["rms_vs_production_mean"],
                    "mse_a_over_b": accuracy[architecture]["cell_mse_ratios"][ratio_key] if label != ORACLE else "",
                    "mse_small_over_large": sizes[temporal]["cell_mse_ratios"][ratio_key] if label != ORACLE else ""})
    for path, digest in tracked.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"input changed during aggregation: {path}")
    json.dumps(result, allow_nan=False)
    return result, csv_rows, tracked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_dirs", nargs=3, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, help="relocated cache path; its bytes must match every recorded digest")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.output_dir.is_symlink() or output.exists():
        raise ValueError("output directory must be new; refusing to overwrite")
    inputs = [path.expanduser().resolve() for path in args.seed_dirs]
    if any(output == path or path in output.parents or output in path.parents for path in inputs):
        raise ValueError("output must not overlap input seed directories")
    result, rows, tracked = aggregate(inputs, data_cache=args.data_cache)
    if any(output == path or path in output.parents or output in path.parents for path in tracked):
        raise ValueError("output must not overlap input artifacts")
    result["provenance"]["aggregator_sha256"] = sha256_file(Path(__file__))
    result["provenance"]["artifact_sha256"] = {str(path): digest for path, digest in sorted(tracked.items())}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.mkdir(parents=True, exist_ok=False)
    (output / "aggregate.json").write_text(payload, encoding="utf-8")
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output_dir": str(output), "accuracy_by_architecture": result["accuracy_by_architecture"]}, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
