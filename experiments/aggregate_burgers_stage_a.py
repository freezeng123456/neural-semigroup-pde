#!/usr/bin/env python3
"""Validate and aggregate the six-arm, three-seed exploratory Burgers study.

Checkpoint files are hashed without unpickling. Paired training seeds are the
replication units; four conditions reuse each seed's same 128 test states.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "burgers-osg-stage-a-v1"
EXPECTED_SEEDS = (31415, 161803, 271828)
TRAINED = ("small_autonomous", "small_query", "osg33_free", "osg33_gdsg",
           "osg1313_free", "osg1313_gdsg")
CONTROLS = ("persistence", "zero", "exact_flux_fd_oracle")
MODELS = (*TRAINED, *CONTROLS)
PARAMETERS = dict(zip(TRAINED, (33, 33, 33, 33, 1313, 1313)))
PAIRS = (("small_autonomous", "small_query"), ("osg33_free", "osg33_gdsg"),
         ("osg1313_free", "osg1313_gdsg"))
CONDITIONS = tuple(f"tau={tau}:horizon={horizon}" for tau in (.04, .08) for horizon in (.4, .8))
MATERIAL_RATIO = .90
SAMPLE_COUNT = 128
SPECTRAL_BANDS = ("k0_3", "k4_15", "k16_32")
CALLS_PER_TRAINING_SAMPLE = dict(zip(TRAINED, (48, 48, 1, 6, 1, 6)))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configuration_digest(configuration):
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()


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


def _read(path, tracked):
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    content = path.read_bytes()
    tracked[path] = hashlib.sha256(content).hexdigest()
    value = json.loads(content, parse_constant=_reject_constant, object_pairs_hook=_unique_pairs)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    _finite_tree(value)
    return value


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the valid range")
    return float(value)


def _integer(value, name, *, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} must be an integer in the valid range")
    return value


def _same_number(actual, expected, name):
    if not math.isclose(_number(actual, name), expected, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError(f"{name} disagrees with its samples or protocol")


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"incomplete or unexpected {name}")


def _samples(values, name, *, allow_none=False):
    if not isinstance(values, list) or len(values) != SAMPLE_COUNT:
        raise ValueError(f"{name} must contain exactly {SAMPLE_COUNT} samples")
    for value in values:
        if value is not None or not allow_none:
            _number(value, name, minimum=0)
    return values


def _validate_rollouts(evaluation, samples):
    _keys(evaluation.get("rollouts"), CONDITIONS, "rollout conditions")
    _keys(samples.get("rollouts"), CONDITIONS, "raw rollout conditions")
    if "sample_indices" in samples and samples["sample_indices"] != list(range(SAMPLE_COUNT)):
        raise ValueError("sample indices must enumerate the common 128 test states")
    for key in CONDITIONS:
        row, raw = evaluation["rollouts"][key], samples["rollouts"][key]
        mse = _samples(raw.get("mse"), f"{key}.mse")
        _same_number(row.get("mse_mean"), math.fsum(mse) / SAMPLE_COUNT, f"{key}.mse_mean")
        relative = _samples(raw.get("relative_l2"), f"{key}.relative_l2", allow_none=True)
        defined = [value for value in relative if value is not None]
        if defined:
            _same_number(row.get("relative_l2_mean"), math.fsum(defined) / len(defined), f"{key}.relative_l2_mean")
        elif row.get("relative_l2_mean") is not None:
            raise ValueError("relative L2 must remain undefined for zero-norm references")
        if "undefined_relative_l2_count" in row and row["undefined_relative_l2_count"] != SAMPLE_COUNT - len(defined):
            raise ValueError("undefined relative error count mismatch")
        drifts = _samples(raw.get("mean_drift"), f"{key}.mean_drift")
        _same_number(row.get("mean_drift_max"), max(drifts), f"{key}.mean_drift_max")
        monotone = raw.get("energy_monotone")
        if not isinstance(monotone, list) or len(monotone) != SAMPLE_COUNT or any(type(value) is not bool for value in monotone):
            raise ValueError("energy monotonicity flags must cover all 128 test states")
        _same_number(row.get("energy_monotone_fraction"), sum(monotone) / SAMPLE_COUNT, f"{key}.energy_monotone_fraction")
        _keys(row.get("spectral_error_mean"), SPECTRAL_BANDS, "spectral summary bands")
        _keys(raw.get("spectral_error"), SPECTRAL_BANDS, "per-sample spectral bands")
        for band in SPECTRAL_BANDS:
            values = _samples(raw["spectral_error"][band], f"{key}.{band}")
            _same_number(row["spectral_error_mean"][band], math.fsum(values) / SAMPLE_COUNT, f"{key}.{band} mean")
    for section, keys in (("cross_lag", ("0.4", "0.8")),
                          ("composition_in_range", ("0.04+0.04=0.08",))):
        _keys(evaluation.get(section), keys, f"{section} summary")
        _keys(samples.get(section), keys, f"{section} samples")
        for key in keys:
            row = evaluation[section][key]
            values = _samples(samples[section][key].get("rms_defect"), f"{section}.{key}")
            _same_number(row.get("rms_defect_mean"), math.fsum(values) / SAMPLE_COUNT, f"{section}.{key} mean")
            if "rms_defect_max" in row:
                _same_number(row["rms_defect_max"], max(values), f"{section}.{key} max")


def _track_expected_file(path, expected, tracked, name):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{name} hash mismatch or missing file: {path}")
    tracked[path] = expected


def _read_seed(root, tracked):
    marker = _read(root / "COMPLETE.json", tracked)
    summary = _read(root / "summary.json", tracked)
    manifest = _read(root / "manifest.json", tracked)
    if summary.get("schema_version") != SCHEMA or marker.get("schema_version") != SCHEMA:
        raise ValueError("requires Stage A schema")
    if summary.get("run_kind") != "full":
        raise ValueError("pilot results cannot enter the full aggregate")
    if summary.get("exploratory") is not True or summary.get("do_not_use_for_formal") is not True:
        raise ValueError("requires exploratory evidence flags")
    if _digest(marker.get("summary_sha256"), "summary hash") != tracked[root / "summary.json"]:
        raise ValueError("completion summary hash mismatch")
    seed = _integer(summary.get("seed"), "seed")
    if seed not in EXPECTED_SEEDS:
        raise ValueError("unexpected training seed")
    for field in ("cache_sha256", "protocol_sha256"):
        _digest(summary.get(field), field)
        if marker.get(field) != summary[field]:
            raise ValueError(f"completion {field} mismatch")
    source_hashes = summary.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("requires a nonempty source hash inventory")
    for name, digest in source_hashes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid source hash inventory key")
        _digest(digest, f"source {name}")
    shared_fields = ("schema_version", "seed", "run_kind", "source_hashes", "cache_sha256", "config_sha256",
                     "protocol_sha256", "configuration", "exploratory", "do_not_use_for_formal")
    for field in shared_fields:
        if manifest.get(field) != summary.get(field):
            raise ValueError(f"manifest/summary {field} mismatch")
    configuration = summary.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("requires a configuration object")
    if configuration_digest(configuration) != _digest(summary.get("config_sha256"), "configuration hash"):
        raise ValueError("configuration hash mismatch")
    for field, expected in (("epochs", 100), ("batch_size", 64)):
        if _integer(configuration.get(field), field, minimum=1) != expected:
            raise ValueError(f"full run {field} differs from frozen protocol")
    for field, expected in (("n_train", 1000), ("n_val", 50), ("n_test", 128),
                            ("validation_interval", 10), ("ode_steps", 12)):
        if field in configuration and _integer(configuration[field], field, minimum=1) != expected:
            raise ValueError(f"full run {field} differs from frozen protocol")
    for field, expected in (("lr", .001), ("weight_decay", 1e-5)):
        if field in configuration:
            _same_number(configuration[field], expected, field)
    normalized = copy.deepcopy(configuration)
    for field in ("seed", "output_dir", "data_cache", "protocol_file"):
        normalized.pop(field, None)
    cells = summary.get("cells")
    _keys(cells, MODELS, "six trained models plus three controls")
    selection = _read(root / "selection_complete.json", tracked)
    _keys(selection, TRAINED, "selection checkpoint matrix")
    per_sample = {}
    for label, cell in cells.items():
        cell_dir = root / label
        if (cell_dir / "summary.json").exists() and _read(cell_dir / "summary.json", tracked) != cell:
            raise ValueError("cell summary differs from completed seed summary")
        samples = _read(cell_dir / "per_sample.json", tracked)
        _validate_rollouts(cell["evaluation"], samples)
        per_sample[label] = samples
        if label in CONTROLS:
            if cell.get("parameter_count") != 0:
                raise ValueError("fixed controls must have no learned parameters")
            continue
        if cell.get("parameter_count") != PARAMETERS[label]:
            raise ValueError("nominal parameter budget mismatch")
        _digest(cell.get("initial_state_sha256"), "initial state hash")
        training = cell.get("training")
        if not isinstance(training, dict):
            raise ValueError("requires training metadata")
        if _integer(training.get("epochs_completed"), "completed epochs") != 100:
            raise ValueError("incomplete training budget")
        if _integer(training.get("optimizer_updates"), "optimizer updates") != 1600:
            raise ValueError("optimizer update count differs from the full budget")
        selected_epoch = _integer(training.get("selected_epoch"), "selected epoch", minimum=1, maximum=100)
        if selected_epoch % 10:
            raise ValueError("selected epoch is outside the validation schedule")
        _digest(training.get("batch_order_sha256"), "batch order hash")
        for field in ("train_seconds", "validation_seconds"):
            _number(training.get(field), field, minimum=0)
        calls = CALLS_PER_TRAINING_SAMPLE[label]
        if _integer(training.get("network_evaluations_per_training_sample"), "network calls per training sample", minimum=1) != calls:
            raise ValueError("network call count per training sample differs from protocol")
        if _integer(training.get("network_evaluations_training"), "training network evaluations", minimum=1) != 100000 * calls:
            raise ValueError("training network evaluation count differs from protocol")
        if training.get("initial_state_sha256") is not None and training["initial_state_sha256"] != cell["initial_state_sha256"]:
            raise ValueError("training/cell initial state hashes differ")
        if "parameters_changed" in training and training["parameters_changed"] is not True:
            raise ValueError("training did not change parameters")
        for field in ("parameter_count", "initial_state_sha256", "training"):
            if selection[label].get(field) != cell.get(field):
                raise ValueError(f"post-selection {field} changed")
        for stage in ("selected", "final"):
            expected = _digest(training.get(f"{stage}_sha256"), f"{stage} checkpoint hash")
            _track_expected_file(cell_dir / f"{stage}.pt", expected, tracked, f"{stage} checkpoint")
    for left, right in PAIRS:
        if cells[left]["initial_state_sha256"] != cells[right]["initial_state_sha256"]:
            raise ValueError("paired initial parameter tensors differ")
        if cells[left]["training"]["batch_order_sha256"] != cells[right]["training"]["batch_order_sha256"]:
            raise ValueError("paired minibatch orders differ")
    return {"root": root, "seed": seed, "summary": summary, "configuration": normalized, "samples": per_sample}


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("geometric mean requires finite nonnegative values")
    return 0.0 if any(value == 0 for value in values) else math.exp(math.fsum(map(math.log, values)) / len(values))


def _comparison(runs, numerator, denominator):
    per_seed, raw = {}, {}
    for run in runs:
        ratios = []
        for condition in CONDITIONS:
            cells = run["summary"]["cells"]
            divisor = cells[denominator]["evaluation"]["rollouts"][condition]["mse_mean"]
            if divisor <= 0:
                raise ValueError("MSE ratio denominator must be positive; no pseudocount is permitted")
            ratio = cells[numerator]["evaluation"]["rollouts"][condition]["mse_mean"] / divisor
            _number(ratio, "MSE ratio", minimum=0)
            ratios.append(ratio)
            raw[f"s{run['seed']}:{condition}"] = ratio
        per_seed[str(run["seed"])] = geometric_mean(ratios)
    pooled = geometric_mean(per_seed.values())
    favorable = sum(value < 1 for value in per_seed.values())
    return {"numerator": numerator, "denominator": denominator,
            "geometric_mean_mse_ratio": pooled, "per_seed_geometric_mean_mse_ratio": per_seed,
            "cell_mse_ratios": raw, "seeds_favoring_numerator": favorable,
            "cells_favoring_numerator": sum(value < 1 for value in raw.values()),
            "material_accuracy_advantage": pooled <= MATERIAL_RATIO and favorable >= 2,
            "decision_scope": "Exploratory accuracy criterion only; physical and composition diagnostics are separate"}


def aggregate(seed_dirs, data_cache=None, protocol_file=None, source_dir=None):
    """Return ``(result, tables, tracked_inputs)`` after read-only validation."""
    roots = [Path(path).expanduser().resolve() for path in seed_dirs]
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("requires exactly three distinct seed directories")
    tracked = {}
    try:
        runs = sorted((_read_seed(root, tracked) for root in roots), key=lambda run: run["seed"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(f"malformed completed-run artifact: {error}") from error
    if tuple(run["seed"] for run in runs) != EXPECTED_SEEDS:
        raise ValueError("requires exactly the three distinct protocol seeds")
    first = runs[0]
    for run in runs[1:]:
        for field in ("source_hashes", "cache_sha256", "protocol_sha256"):
            if run["summary"][field] != first["summary"][field]:
                raise ValueError(f"{field} differs across seeds")
        if run["configuration"] != first["configuration"]:
            raise ValueError("seed configurations differ beyond seed/output/input paths")
    external_checks = {"cache_bytes_verified": False, "protocol_bytes_verified": False,
                       "source_bytes_verified": False}
    for supplied, field, status in ((data_cache, "cache_sha256", "cache_bytes_verified"),
                                    (protocol_file, "protocol_sha256", "protocol_bytes_verified")):
        if supplied is not None:
            _track_expected_file(supplied, first["summary"][field], tracked, field)
            external_checks[status] = True
    if source_dir is not None:
        source_root = Path(source_dir).expanduser().resolve()
        for name, digest in first["summary"]["source_hashes"].items():
            path = (source_root / name).resolve()
            if source_root not in path.parents:
                raise ValueError("source inventory path escapes the supplied source directory")
            _track_expected_file(path, digest, tracked, f"source {name}")
        external_checks["source_bytes_verified"] = True
    primary = {budget: _comparison(runs, f"{budget}_gdsg", f"{budget}_free") for budget in ("osg33", "osg1313")}
    secondary = _comparison(runs, "small_autonomous", "small_query")
    tables = {"cells": [], "costs": [], "per_sample": []}
    physical, structure = {}, {}
    for label in MODELS:
        rows = [run["summary"]["cells"][label]["evaluation"]["rollouts"][condition]
                for run in runs for condition in CONDITIONS]
        physical[label] = {"spatial_mean_drift_max": max(row["mean_drift_max"] for row in rows),
                           "energy_monotone_fraction_min": min(row["energy_monotone_fraction"] for row in rows),
                           "scope": "Sampled projected-state diagnostics; not physical guarantees or accuracy gates"}
        structure[label] = {}
        for section in ("cross_lag", "composition_in_range"):
            defects = {f"s{run['seed']}:{key}": row["rms_defect_mean"]
                for run in runs for key, row in run["summary"]["cells"][label]["evaluation"][section].items()}
            structure[label][section] = {"rms_defect_mean": math.fsum(defects.values()) / len(defects),
                                        "per_seed_condition_rms_defect_mean": defects}
    for run in runs:
        for label in MODELS:
            cell = run["summary"]["cells"][label]
            training = cell.get("training", {})
            cost = {"seed": run["seed"], "model": label, "parameter_count": cell["parameter_count"],
                    "selected_epoch": training.get("selected_epoch", ""),
                    "optimizer_updates": training.get("optimizer_updates", 0),
                    "epochs_completed": training.get("epochs_completed", 0),
                    "train_seconds": training.get("train_seconds", 0.0),
                    "validation_seconds": training.get("validation_seconds", 0.0),
                    "eval_seconds": cell.get("eval_seconds", ""),
                    "network_evaluations_per_training_sample": training.get("network_evaluations_per_training_sample", 0),
                    "network_evaluations_training": training.get("network_evaluations_training", 0),
                    "peak_cuda_memory_bytes": training.get("peak_cuda_memory_bytes", "")}
            for field in ("eval_seconds", "network_evaluations_per_training_sample", "network_evaluations_training", "peak_cuda_memory_bytes"):
                if cost[field] not in ("", None):
                    _number(cost[field], field, minimum=0)
            tables["costs"].append(cost)
            for condition in CONDITIONS:
                tau, horizon = [float(piece.split("=")[1]) for piece in condition.split(":")]
                row = cell["evaluation"]["rollouts"][condition]
                tables["cells"].append({"seed": run["seed"], "model": label, "tau": tau, "horizon": horizon,
                    "parameter_count": cell["parameter_count"], "selected_epoch": training.get("selected_epoch", ""),
                    **{name: row[name] for name in ("mse_mean", "relative_l2_mean", "mean_drift_max", "energy_monotone_fraction")},
                    **{f"spectral_error_{band}": row["spectral_error_mean"][band] for band in SPECTRAL_BANDS},
                    "cross_lag_rms_defect_mean": cell["evaluation"]["cross_lag"][str(horizon)]["rms_defect_mean"],
                    "composition_in_range_rms_defect_mean": cell["evaluation"]["composition_in_range"]["0.04+0.04=0.08"]["rms_defect_mean"],
                    "network_evaluations_per_sample": row.get("network_evaluations_per_sample", "")})
                raw = run["samples"][label]["rollouts"][condition]
                for index in range(SAMPLE_COUNT):
                    tables["per_sample"].append({"seed": run["seed"], "model": label, "tau": tau,
                        "horizon": horizon, "sample_index": index, "mse": raw["mse"][index],
                        "relative_l2": raw["relative_l2"][index], "mean_drift": raw["mean_drift"][index],
                        "energy_monotone": raw["energy_monotone"][index],
                        **{f"spectral_error_{band}": raw["spectral_error"][band][index] for band in SPECTRAL_BANDS}})
    result = {"schema_version": SCHEMA + "-aggregate", "run_kind": "full",
        "exploratory": True, "do_not_use_for_formal": True,
        "seeds": list(EXPECTED_SEEDS), "trained_model_count": 18,
        "fixed_control_evaluation_count": 9, "rollout_cell_count": len(tables["cells"]),
        "test_states_per_seed_condition": SAMPLE_COUNT,
        "primary_osg_comparisons": primary, "secondary_small_autonomous_over_query": secondary,
        "material_accuracy_rule": {"geometric_mean_ratio_max": MATERIAL_RATIO, "minimum_favorable_seeds": 2},
        "physical_diagnostics": physical, "composition_diagnostics": structure, "costs": tables["costs"],
        "provenance": {"source_hashes": first["summary"]["source_hashes"],
            "cache_sha256": first["summary"]["cache_sha256"], "protocol_sha256": first["summary"]["protocol_sha256"],
            "shared_configuration": first["configuration"], "external_input_verification": external_checks,
            "inputs": [{"seed": run["seed"], "directory": str(run["root"]),
                        "summary_sha256": tracked[run["root"] / "summary.json"]} for run in runs]},
        "interpretation": {
            "test_scope": "The reference-v2 test set has previously been opened; this is an exploratory common-data comparison, not a fresh blinded confirmation.",
            "replication": "Three training seeds; four correlated conditions and the same 128 initial states are not independent replications.",
            "capacity": "OSG lambda pairs match architecture and initialization. Cross-family comparisons do not establish capacity matching.",
            "budget": "100 epochs and matched paired sample exposure; network calls and wall time differ. No convergence claim.",
            "objective": "The DUE-style (fit + lambda * GDSG)/(1 + lambda) changes loss scaling as well as adding regularization.",
            "structure": "Prediction accuracy, composition consistency, and sampled physical diagnostics are separate. No composition ratio uses zero as denominator.",
            "external_hashes": "Unprovided cache/protocol/source files have matching recorded digests only; external_input_verification identifies byte-level checks."}}
    for path, digest in tracked.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"input changed during aggregation: {path}")
    json.dumps(result, allow_nan=False)
    return result, tables, tracked


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_dirs", nargs=3, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path)
    parser.add_argument("--protocol-file", type=Path)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if args.output_dir.is_symlink() or output.exists():
        raise ValueError("output directory must be new; refusing to overwrite")
    roots = [path.expanduser().resolve() for path in args.seed_dirs]
    if any(output == root or root in output.parents or output in root.parents for root in roots):
        raise ValueError("output must not overlap input seed directories")
    result, tables, tracked = aggregate(roots, args.data_cache, args.protocol_file, args.source_dir)
    if any(output == path or path in output.parents or output in path.parents for path in tracked):
        raise ValueError("output must not overlap input artifacts")
    result["provenance"]["aggregator_sha256"] = sha256_file(Path(__file__))
    result["provenance"]["artifact_sha256"] = {str(path): digest for path, digest in sorted(tracked.items())}
    output.mkdir(parents=True, exist_ok=False)
    (output / "aggregate.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, rows in tables.items():
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    artifacts = {name: sha256_file(output / name) for name in ("aggregate.json", "cells.csv", "costs.csv", "per_sample.csv")}
    (output / "COMPLETE.json").write_text(json.dumps({"schema_version": SCHEMA + "-aggregate",
        "artifact_sha256": artifacts, "exploratory": True, "do_not_use_for_formal": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "primary_osg_comparisons": result["primary_osg_comparisons"]}, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
