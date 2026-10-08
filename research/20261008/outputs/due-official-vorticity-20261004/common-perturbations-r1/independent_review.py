#!/usr/bin/env python3
"""Independent postprocessing audit of existing arrays only; no model inference."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / 'COMMON_PERTURBATIONS_DIAGNOSTIC.json').read_text())
    manifest = json.loads((root / 'artifact_manifest.json').read_text())
    for name, info in manifest.items():
        assert sha(root / name) == info['sha256'] and (root / name).stat().st_size == info['bytes']
    for name, expected in report['input_sha256'].items():
        assert sha(name) == expected
    z = np.load(root / 'common_perturbation_arrays.npz', allow_pickle=False)
    times, amps, signs = list(z['times']), list(z['amplitudes']), list(z['signs'])
    directions = report['directions']
    checks = 0
    for filename, kind in [('one_sided_gains.csv', 'one'), ('symmetric_gains.csv', 'sym')]:
        with (root / filename).open() as f:
            for row in csv.DictReader(f):
                ti = times.index(int(row['input_time']))
                di = directions.index(row['direction'])
                ai = amps.index(float(row['requested_relative_l2_amplitude']))
                i = int(row['test_index'])
                for name in ['sg0', 'gdsg']:
                    if kind == 'one':
                        pairs = [(row[name + '_gain'], z[name + '_one_sided_gain'][ti, di, ai, signs.index(int(row['sign'])), i])]
                    else:
                        pairs = [(row[name + '_symmetric_mean_gain'], z[name + '_symmetric_mean_gain'][ti, di, ai, i]),
                                 (row[name + '_central_secant_gain'], z[name + '_central_secant_gain'][ti, di, ai, i])]
                    for value, expected in pairs:
                        assert (value == '' and np.isnan(expected)) or float(value) == expected
                        checks += 1
    for item in report['conditions']:
        ti = times.index(item['input_time'])
        di = directions.index(item['direction'])
        ai = amps.index(item['requested_relative_l2_amplitude'])
        for metric in ['symmetric_mean_gain', 'central_secant_gain']:
            for name in ['sg0', 'gdsg']:
                a = z[name + '_' + metric][ti, di, ai]
                good = a[np.isfinite(a)]
                if len(good):
                    assert item[metric][name]['mean'] == good.mean()
                    assert item[metric][name]['median'] == np.median(good)
                checks += 1
    pooled = []
    for di, direction in enumerate(directions):
        for ai, amplitude in enumerate(amps):
            item = {'direction': direction, 'relative_amplitude': float(amplitude)}
            for metric in ['symmetric_mean_gain', 'central_secant_gain']:
                s, g = z['sg0_' + metric][:, di, ai].ravel(), z['gdsg_' + metric][:, di, ai].ravel()
                valid = np.isfinite(s) & np.isfinite(g)
                item[metric] = {'paired_count': int(valid.sum()),
                    'sg0_mean': float(s[valid].mean()), 'gdsg_mean': float(g[valid].mean()),
                    'sg0_median': float(np.median(s[valid])), 'gdsg_median': float(np.median(g[valid])),
                    'mean_paired_difference_gdsg_minus_sg0': float((g[valid] - s[valid]).mean()),
                    'gdsg_lower_count': int((g[valid] < s[valid]).sum())}
            pooled.append(item)
    result = {'status': 'INDEPENDENT_POSTPROCESS_AUDIT_PASSED', 'artifact_hashes_checked': len(manifest),
        'input_hashes_unchanged': len(report['input_sha256']), 'csv_and_json_summary_checks': checks,
        'new_network_calls': 0, 'optimizer_updates': 0,
        'pooled_direction_amplitude_summary': pooled,
        'pooling': '100 trajectories × 5 specified state times. Repeated times on the same trajectory are correlated; no independence, confidence interval, or significance claim.',
        'interpretation': 'GDSG response is lower mostly along SG0 error directions; along GDSG error and zero-mean Gaussian directions it is usually higher. These data reject a universal contraction description. Response is not true-PDE accuracy.',
        'input_sha256': {name: sha(root / name) for name in list(manifest) + ['artifact_manifest.json']},
        'script_sha256': sha(__file__)}
    with (root / 'INDEPENDENT_REVIEW.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'checks': checks, 'new_network_calls': 0}))


if __name__ == '__main__':
    main()
