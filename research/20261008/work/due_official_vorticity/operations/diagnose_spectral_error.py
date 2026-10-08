#!/usr/bin/env python3
"""Decompose frozen baseline prediction error using the author's FFT index mask.

This is a descriptive energy decomposition, not evidence that modes outside the
mask cannot be represented: the FNO also has pointwise paths and nonlinearities.
"""
import argparse
import json
from pathlib import Path

from replay_official_cpu import sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verification', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    v = json.loads(a.verification.read_text())
    assert v['status'] == 'verified' and v['engineering']['status'] == 'passed'
    run = Path(v['run_root']).resolve()
    truth_path = Path(v['test_mat']).resolve()
    pred_path = run / 'official_workdir/model/pred.mat'
    hashes = {str(x.resolve()): sha(x) for x in [a.verification, truth_path, pred_path]}
    for x in [truth_path, pred_path]:
        assert hashes[str(x)] == v['input_sha256'][str(x)]
    output = a.output_dir.resolve()
    assert not output.is_relative_to(run)
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from scipy.io import loadmat
    truth = loadmat(truth_path, variable_names=['trajectories'])['trajectories'].astype(np.float32)
    pred = loadmat(pred_path, variable_names=['trajectories'])['trajectories']
    assert pred.shape == truth.shape == (100, 64, 64, 1, 100)
    retained = np.zeros((64, 33), dtype=bool)
    retained[:12, :12] = True
    retained[-12:, :12] = True
    zero = np.zeros_like(retained); zero[0, 0] = True
    masks = {'zero': zero, 'author_spectral_indices_nonzero': retained & ~zero,
             'outside_author_spectral_indices': ~retained}
    assert np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1)
    weight = np.full((64, 33), 2.0); weight[:, 0] = 1.; weight[:, -1] = 1.
    summaries = []
    max_parseval_relative_gap = 0.
    all_arrays = {}
    for t in [0, 1, 40, 60, 80, 99]:
        u = truth[..., 0, t].astype(np.float64)
        y = pred[..., 0, t].astype(np.float64)
        fields = {'truth': u, 'prediction': y, 'error': y-u}
        energies = {}
        for field, values in fields.items():
            spectrum = np.fft.rfft2(values, axes=(-2, -1), norm='ortho')
            weighted = np.abs(spectrum)**2 * weight
            by_group = {name: weighted[:, mask].sum(axis=1) for name, mask in masks.items()}
            spectral_total = sum(by_group.values())
            physical_total = (values**2).sum(axis=(1, 2))
            assert np.allclose(spectral_total, physical_total, rtol=2e-12, atol=1e-20)
            valid = physical_total > 0
            if valid.any():
                max_parseval_relative_gap = max(max_parseval_relative_gap, float(np.max(np.abs(spectral_total[valid]-physical_total[valid])/physical_total[valid])))
            energies[field] = by_group
            for name, array in by_group.items():
                all_arrays[f'{field}_elapsed{t}_{name}'] = array
        total_error = sum(x.sum() for x in energies['error'].values())
        total_truth = sum(x.sum() for x in energies['truth'].values())
        groups = {}
        for name in masks:
            e = float(energies['error'][name].sum())
            u = float(energies['truth'][name].sum())
            y = float(energies['prediction'][name].sum())
            groups[name] = {'pooled_truth_squared_norm': u, 'pooled_prediction_squared_norm': y,
                            'pooled_error_squared_norm': e,
                            'fraction_of_total_error_squared_norm': e/total_error if total_error else None,
                            'fraction_of_total_truth_squared_norm': u/total_truth,
                            'error_rms_over_all_trajectories_and_spatial_points': (e/(100*64*64))**.5}
        summaries.append({'elapsed_time': t, 'groups': groups})
    assert all(sha(Path(path)) == digest for path, digest in hashes.items())
    np.savez_compressed(output / 'per_trajectory_spectral_energies.npz', **all_arrays)
    report = {'status': 'SPECTRAL_ENERGY_DIAGNOSTIC_COMPLETE',
              'run_root': str(run), 'test_trajectories': 100,
              'times': [0,1,40,60,80,99], 'summaries': summaries,
              'source_mask': 'due/networks/fno.py: rfft2 rows[:12,:12] and rows[-12:,:12]; zero mode separated',
              'normalization': 'Orthonormal FFT, real FFT conjugate weight 2 except second-axis zero and Nyquist (weight1). Float32 physical states cast to float64 before subtraction and FFT.',
              'parseval_max_relative_gap': max_parseval_relative_gap,
              'scope': 'Descriptive allocation of squared error across fixed frequency index sets; no inference or training.',
              'limitations': ['FNO pointwise paths and nonlinearities can represent frequencies outside the spectral mask.',
                              'Frequency localization does not establish a causal capacity bottleneck.',
                              'This already inspected test set is used for diagnosis, not independent model selection.'],
              'input_sha256': hashes, 'inputs_unchanged': True, 'script_sha256': sha(Path(__file__))}
    write_json(output / 'SPECTRAL_DIAGNOSTIC.json', report)
    print(json.dumps({'status': report['status'], 'parseval_max_relative_gap': max_parseval_relative_gap,
                      'final': summaries[-1]}, indent=2))


if __name__ == '__main__':
    main()
