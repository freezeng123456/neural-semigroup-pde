#!/usr/bin/env python3
"""Recover only the summary after the documented final JSON Path-type error.

No model load, forward, prediction, training, or array changes. Original failure
and executed source are retained; every stored scalar is checked against CSV.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import numpy as np
from replay_official_cpu import read_inputs, sha, write_json
from diagnose_semigroup_coverage import parse_args, GROUPS, MODELS, TRAIN_TIMES, TEST_TIMES

EXECUTED_SHA = 'adb35b16a5c16aa8ee38d3a2cd36f84bd5873210bd194025a045ea98c314ef29'


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)): return [clean(v) for v in x]
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, (np.integer, int)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    return x


def stats(x):
    a = np.asarray(x, dtype=np.float64)
    good = a[np.isfinite(a)]
    result = {'defined_count': int(good.size), 'undefined_count': int(a.size - good.size)}
    if len(good):
        result.update(mean=float(good.mean()), median=float(np.median(good)), p10=float(np.quantile(good, .1)),
                      p90=float(np.quantile(good, .9)), p99=float(np.quantile(good, .99)), maximum=float(good.max()))
    return result


def ratio(a, b):
    return a / b if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b)) and b > 0 else None


def main():
    args = parse_args()
    out = args.output_dir.resolve(strict=True)
    original = out / 'executed_diagnose_semigroup_coverage.py'
    assert sha(original) == EXECUTED_SHA
    failure = json.loads((out / 'FAILURE.json').read_text())
    assert "'str' object has no attribute 'open'" in failure['error']
    assert "'script_sha256': sha(__file__)" in failure['traceback']
    assert not (out / 'SEMIGROUP_COVERAGE_DIAGNOSTIC.json').exists()
    fingerprints, contexts = {}, {}
    for name in MODELS:
        source, run, env, receipt, checkpoint, _, hashes, is_sg0 = read_inputs(argparse.Namespace(
            source_root=args.source_root, source_manifest=args.source_manifest,
            run_root=getattr(args, name + '_run_root'), test_mat=args.test_mat))
        assert is_sg0 == (name == 'sg0')
        vpath = getattr(args, name + '_verification')
        verified = json.loads(vpath.read_text())
        assert verified['status'] == 'verified' and verified['engineering']['status'] == 'passed'
        assert Path(verified['run_root']).resolve() == run
        fingerprints.update(hashes)
        fingerprints[str(vpath.resolve())] = sha(vpath)
        contexts[name] = env
    assert sha(args.train_mat) == contexts['gdsg']['data_sha256']['VorticityOSG_train.mat']
    fingerprints[str(args.train_mat.resolve())] = sha(args.train_mat)
    recon_path = args.reconstruction_dir / 'RECONSTRUCTION_REPORT.json'
    reconstruction = json.loads(recon_path.read_text())
    assert reconstruction['status'] == 'VERIFIED_DETERMINISTIC_RECONSTRUCTION'
    assert all(c['passed'] for c in reconstruction['checks'])
    for path, expected in reconstruction['input_sha256'].items():
        assert sha(Path(path)) == expected
        fingerprints[path] = expected
    fingerprints[str(recon_path.resolve())] = sha(recon_path)
    for name in ('fixed_u0_normalized.npy', 'fixed_dt_normalized.npy', 'fixed_substeps_physical.npy'):
        path = args.reconstruction_dir / 'artifacts' / name
        assert sha(path) == reconstruction['artifacts'][name]['sha256']
        fingerprints[str(path.resolve())] = sha(path)
    protected = {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in out.iterdir() if p.is_file()}
    total_count, csv_checks = 0, 0
    groups, results = {}, {}
    with np.load(out / 'per_sample_defects.npz', allow_pickle=False) as data:
        keys = sorted({k.split('__')[2] for k in data.files})
        assert len(keys) == 16 and len(data.files) == 4 * 2 * 16
        for label in GROUPS:
            with np.load(out / (label + '_inputs.npz'), allow_pickle=False) as inputs:
                groups[label] = {k: inputs[k].copy() for k in inputs.files}
            n = 2500 if 'box' in label else 500
            assert groups[label]['normalized_states'].shape == (n, 64, 64, 1)
            assert groups[label]['normalized_states'].dtype == np.float32
            assert groups[label]['normalized_lag_codes'].shape == (n, 3)
            results[label] = {name: {key: data[label + '__' + name + '__' + key].copy() for key in keys} for name in MODELS}
            assert all(a.shape == (n,) for values in results[label].values() for a in values.values())
            total_count += n
    assert total_count == 6000
    seen = set()
    with (out / 'per_sample_defects.csv').open() as f:
        for row in csv.DictReader(f):
            label, name, i = row['group'], row['model'], int(row['sample_index'])
            identity = (label, name, i)
            assert identity not in seen
            seen.add(identity)
            for key in keys:
                expected = results[label][name][key][i]
                assert (row[key] == '' and np.isnan(expected)) or float(row[key]) == expected
                csv_checks += 1
            assert float(row['physical_total']) == float(groups[label]['physical_total'][i, 0])
            assert (row['total_above_tmax'] == 'True') == (groups[label]['physical_total'][i, 0] > reconstruction['tmax'])
    assert len(seen) == 12000 and csv_checks == 192000
    summaries = {}
    for label, data in groups.items():
        total = data['physical_total'][:, 0]
        conditions = {'all': np.ones(len(total), bool), 'total_within_supervised_range': total <= reconstruction['tmax'],
                      'total_above_supervised_range': total > reconstruction['tmax']}
        if 'state_index' in data:
            for t in np.unique(data['state_index']):
                conditions['state_index=' + str(t)] = data['state_index'] == t
                for split in ('total_within_supervised_range', 'total_above_supervised_range'):
                    conditions['state_index=%s:%s' % (t, split)] = (data['state_index'] == t) & conditions[split]
        summaries[label] = {condition: {'count': int(mask.sum()), **{
            name: {key: stats(value[mask]) for key, value in values.items()} for name, values in results[label].items()}}
            for condition, mask in conditions.items()}
    key = 'normalized_relative_l2_symmetric'
    fv, nv = results['gdsg_fixed_box']['gdsg'][key], results['fresh_box']['gdsg'][key]
    fs, ns = stats(fv), stats(nv)
    mean_ratio = ratio(ns.get('mean'), fs.get('mean'))
    defined = fs['undefined_count'] == ns['undefined_count'] == 0 and mean_ratio is not None
    gate = {'metric': key, 'model': 'gdsg', 'fixed_mean': fs.get('mean'), 'fresh_mean': ns.get('mean'),
            'fresh_over_fixed_mean': mean_ratio, 'prespecified_mean_ratio_threshold': 1.5,
            'gate_defined': defined, 'mean_threshold_met': mean_ratio >= 1.5 if defined else None,
            'fixed_undefined_count': fs['undefined_count'], 'fresh_undefined_count': ns['undefined_count'],
            'fresh_over_fixed_median': ratio(ns.get('median'), fs.get('median')),
            'fresh_over_fixed_p90': ratio(ns.get('p90'), fs.get('p90')),
            'fixed_top_1_percent_sum_fraction': ratio(np.sort(fv)[-25:].sum(), fv.sum()),
            'fresh_top_1_percent_sum_fraction': ratio(np.sort(nv)[-25:].sum(), nv.sum())}
    comparisons = {name: {split: {metric: ratio(summaries['fresh_box'][split][name][metric].get('mean'),
        summaries['gdsg_fixed_box'][split][name][metric].get('mean')) for metric in keys}
        for split in ('all', 'total_within_supervised_range', 'total_above_supervised_range')} for name in MODELS}
    for name, info in protected.items():
        assert sha(out / name) == info['sha256'] and (out / name).stat().st_size == info['bytes']
    assert all(sha(Path(path)) == expected for path, expected in fingerprints.items())
    report = {'status': 'SEMIGROUP_COVERAGE_DIAGNOSTIC_RECOVERED_AFTER_REPORT_ERROR',
        'optimizer_updates': 0, 'network_batch_forward_calls': 840, 'sample_equivalent_forward_calls': 84000,
        'additional_midpoint_calls': 240, 'regular_two_order_and_direct_calls': 600,
        'recovery_network_calls': 0, 'recovery_csv_npz_scalar_checks': csv_checks,
        'batch_size': 100, 'device': 'cpu', 'fresh_box_seed': 20261005, 'real_state_lag_seed': 20261006,
        'group_counts': {k: len(v['physical_total']) for k, v in groups.items()},
        'train_state_indices': TRAIN_TIMES, 'test_state_indices': TEST_TIMES,
        'tmin': reconstruction['tmin'], 'tmax': reconstruction['tmax'],
        'model_selection': 'Archived final epoch500 for both arms',
        'metric_definitions': {'normalized_relative_l2': 'Author float32 norm ||composed-direct||/||composed||; st,ts separate, symmetric is their mean.',
            'physical_relative_l2': 'Same denominator after author float32 denormalization; difference and norm in float64.',
            'absolute_rms': 'RMS difference in named space; float64 difference and sums.',
            'midpoint': 'Same s+t total, replaced by two half steps; ordinary float32 time encoding roundoff applies.',
            'undefined': 'Zero denominator gives NaN/blank/null and a count; no epsilon.'},
        'summaries': summaries, 'fresh_over_fixed_mean_ratios': comparisons, 'fixed_vs_fresh_gate': gate,
        'input_sha256': fingerprints, 'original_saved_artifacts': protected,
        'executed_diagnostic_script_sha256': EXECUTED_SHA, 'recovery_script_sha256': sha(Path(__file__)),
        'reconstruction_evidence': reconstruction['evidence_limit'],
        'recovery': {'reason': 'After all 840 forwards, invariant assertions and CSV/NPZ saves, report construction called sha(__file__) with str instead of Path.',
            'original_failure_retained': 'FAILURE.json', 'executed_source_retained': original.name,
            'no_model_reexecution': True,
            'invariant_evidence': 'Executed-source control flow and the recorded failure location place parameter/input/source assertions before successfully saved CSV/NPZ. Current frozen inputs are also rehashed here.',
            'original_inference_wall_seconds': None,
            'wall_time_note': 'Original process did not persist a duration before the report error. No inferred timing is substituted.'},
        'limitations': ['Already inspected public test: exploratory, not independent confirmation.',
            'SG0 did not train on the original GDSG fixed collocations.',
            'One training seed per model, no causal or training-significance claim.',
            'Low semigroup defect does not prove PDE correctness or rollout accuracy.',
            'Random box states generally lie away from physical data; report lag strata separately.',
            'Real training-trajectory states need not be selected burst inputs; state indices are not verified absolute times.',
            'Original point arrays were not directly archived; reconstruction uses historical RNG/model anchors.']}
    write_json(out / 'SEMIGROUP_COVERAGE_DIAGNOSTIC.json', clean(report))
    artifacts = [p for p in out.iterdir() if p.is_file()]
    write_json(out / 'artifact_manifest.json', {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in artifacts})
    print(json.dumps(clean({'status': report['status'], 'scalar_checks': csv_checks, 'gate': gate}), indent=2))


if __name__ == '__main__':
    main()
