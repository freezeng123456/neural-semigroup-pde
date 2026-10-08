#!/usr/bin/env python3
"""Read-only SG0 postprocessing; no model import, inference, or training.

Requires NumPy and SciPy. Run next to SCIENTIFIC_REVIEW.zh.md. An optional
--run-root supports an independently relocated, already verified recovery.
The output is created exclusively and never overwrites earlier diagnostics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.io import loadmat


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def quantiles(values):
    names = ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'max')
    return dict(zip(names, map(float, np.quantile(values, [0, .1, .25, .5, .75, .9, .95, 1], method='linear'))))


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification-json', type=Path, default=here.parent / 'sg0-independent-verification/verification.json')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--test-mat', type=Path)
    parser.add_argument('--output', type=Path, default=here / 'diagnostic_metrics.json')
    args = parser.parse_args()
    verification_path = args.verification_json.resolve(strict=True)
    verification = json.loads(verification_path.read_text())
    assert verification['status'] == 'verified' and verification['engineering']['status'] == 'passed'
    original_root = Path(verification['run_root'])
    root = args.run_root.resolve(strict=True) if args.run_root else original_root.resolve(strict=True)
    truth_path = args.test_mat.resolve(strict=True) if args.test_mat else Path(verification['test_mat']).resolve(strict=True)
    inputs = {
        'independent_verification': verification_path,
        'prediction': root / 'official_workdir/model/pred.mat',
        'truth': truth_path,
        'training_history': root / 'official_workdir/model/training_history.csv',
        'epoch_events': root / 'audit/epoch_events.jsonl',
    }
    hashes = {name: {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)} for name, path in inputs.items()}
    # Bind the analyzed files to the prior verification, without rerunning it.
    prior = verification['input_sha256']
    for name in ('prediction', 'training_history', 'epoch_events'):
        original = original_root / inputs[name].relative_to(root)
        assert hashes[name]['sha256'] == prior[str(original)], f'File differs from verified input: {name}'
    assert hashes['truth']['sha256'] == prior[verification['test_mat']]
    output = args.output.resolve()
    assert not output.is_relative_to(root), 'Diagnostic output must be outside recovered evidence'
    assert output not in inputs.values() and not output.exists()

    # Main accuracy figures retain the author's float32 metric calculation.
    official = verification['independently_recomputed_metrics']['author_float32']
    errors = np.asarray(official['per_trajectory_component_time'], dtype=np.float64)[:, 0, :]
    assert errors.shape == (100, 100) and np.isfinite(errors).all()
    curve = np.asarray(official['mean_over_trajectories_curve'])[0]
    curve_steps = (0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
    final = errors[:, -1]
    order = np.argsort(final)[::-1]
    accuracy = {
        'metric_track': 'author_float32; summaries/quantiles of the recorded per-trajectory values use float64',
        'mean_curve_selected_steps': {str(step): float(curve[step]) for step in curve_steps},
        'terminal_elapsed99_quantiles': quantiles(final),
        'terminal_count_gt_1': int(np.count_nonzero(final > 1)),
        'terminal_count_gt_0_5': int(np.count_nonzero(final > .5)),
        'terminal_top5_share_of_error_sum': float(final[order[:5]].sum() / final.sum()),
        'terminal_top10_share_of_error_sum': float(final[order[:10]].sum() / final.sum()),
        'strict_increases_between_forecast_steps1to99': int(np.count_nonzero(np.diff(curve[1:]) > 0)),
        'first_elapsed_step_mean_exceeds': {str(value): int(np.flatnonzero(curve > value)[0]) if np.any(curve > value) else None for value in (.01, .1, 1)},
        'time_and_trajectory_mean_relative_l2': official['time_and_trajectory_mean_relative_l2'],
        'mean_trajectory_joint_spacetime_relative_l2': official['mean_trajectory_joint_spacetime_relative_l2'],
        'scope': '100 test trajectories, one training seed. Quantiles are descriptive, not confidence intervals.',
    }

    prediction = loadmat(inputs['prediction'], variable_names=['trajectories'], verify_compressed_data_integrity=True)['trajectories']
    truth = loadmat(inputs['truth'], variable_names=['trajectories'], verify_compressed_data_integrity=True)['trajectories']
    assert prediction.shape == truth.shape == (100, 64, 64, 1, 100)
    diagnostics = {}
    for step in (0, 1, 40, 60, 80, 99):
        p = prediction[..., step].astype(np.float64)
        u = truth[..., step].astype(np.float64)
        assert np.isfinite(p).all() and np.isfinite(u).all()
        delta = p - u
        p_rms = np.sqrt(np.mean(p * p, axis=(1, 2, 3)))
        u_rms = np.sqrt(np.mean(u * u, axis=(1, 2, 3)))
        assert np.all(u_rms > 0)
        ratio = p_rms / u_rms
        mean_p, mean_u = p.mean(axis=(1, 2, 3)), u.mean(axis=(1, 2, 3))
        mean_error = mean_p - mean_u
        error_mse = np.mean(delta * delta, axis=(1, 2, 3))
        defined = error_mse > 0
        fractions = mean_error[defined] ** 2 / error_mse[defined]
        row = {
            'truth_rms': quantiles(u_rms),
            'prediction_rms': quantiles(p_rms),
            'prediction_over_truth_rms': quantiles(ratio),
            'truth_max_abs_spatial_mean': float(np.abs(mean_u).max()),
            'prediction_abs_spatial_mean': quantiles(np.abs(mean_p)),
            'constant_component_error_fraction_defined_trajectories': int(defined.sum()),
            'constant_component_error_fraction': quantiles(fractions) if fractions.size else None,
            'pooled_constant_component_error_fraction': float(np.sum(mean_error ** 2) / np.sum(error_mse)) if error_mse.sum() > 0 else None,
        }
        if step == 99:
            row['worst5_by_official_terminal_error'] = [
                {'test_index_zero_based': int(i), 'official_float32_relative_l2': float(final[i]),
                 'truth_rms': float(u_rms[i]), 'prediction_rms': float(p_rms[i]),
                 'prediction_over_truth_rms': float(ratio[i]),
                 'truth_spatial_mean': float(mean_u[i]), 'prediction_spatial_mean': float(mean_p[i]),
                 'constant_component_error_fraction': float(mean_error[i] ** 2 / error_mse[i])}
                for i in order[:5]
            ]
        diagnostics[str(step)] = row

    history = np.loadtxt(inputs['training_history'])
    events = [json.loads(line) for line in inputs['epoch_events'].read_text().splitlines() if line.strip()]
    assert history.shape == (500,) and len(events) == 500 and np.isfinite(history).all()
    history_steps = (1, 100, 200, 300, 400, 450, 490, 500)
    training = {
        'online_objective_selected_epochs': {str(epoch): float(history[epoch - 1]) for epoch in history_steps},
        'initial_over_final_objective': float(history[0] / history[-1]),
        'epoch100_over_final_objective': float(history[99] / history[-1]),
        'epoch500_fraction_of_epoch100_objective': float(history[-1] / history[99]),
        'percent_decrease_epoch450_to500': float(100 * (1 - history[499] / history[449])),
        'percent_decrease_epoch490_to500': float(100 * (1 - history[499] / history[489])),
        'increases_across_all499_epoch_transitions': int(np.count_nonzero(np.diff(history) > 0)),
        'increases_within_last50_epochs': int(np.count_nonzero(np.diff(history[-50:]) > 0)),
        'best_epoch_original_python_float': min(events, key=lambda event: event['training_loss'])['epoch'],
        'lr_after_selected_epochs': {str(epoch): events[epoch - 1]['learning_rate_after_epoch'] for epoch in history_steps},
        'scope': 'Online minibatch objective; cosine learning rate reaches zero. These values do not prove optimization convergence.',
    }
    for name, path in inputs.items():
        assert sha256(path) == hashes[name]['sha256'], f'Input changed during analysis: {name}'
    report = {
        'schema_version': 'official-sg0-diagnostic-v1',
        'scope': 'Read-only postprocessing of verified artifacts; no training, inference, or repeated full verification.',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__)),
        'runtime': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'scipy': scipy.__version__},
        'input_sha256': hashes, 'inputs_unchanged_after_analysis': True,
        'definitions': {
            'time': 'Elapsed since the first supplied state, 0..99; absolute time origin unknown.',
            'diagnostic_precision': 'Prediction and original MAT truth cast to float64 before all amplitude/bias arithmetic.',
            'rms': 'sqrt(mean over all spatial/field elements of value squared), per trajectory and time.',
            'constant_component_fraction': 'mean_spatial(pred-truth)^2 / mean_spatial((pred-truth)^2). The spatially constant component is orthogonal to the zero-mean remainder.',
            'pooled_constant_component_fraction': 'sum_trajectories(mean_spatial(pred-truth)^2) / sum_trajectories(mean_spatial((pred-truth)^2)).',
            'training_percent_decrease': '100 * (1 - objective_at_later_epoch / objective_at_earlier_epoch).',
            'quantiles': 'NumPy quantile(method="linear"); descriptive across 100 test trajectories.',
        },
        'official_accuracy': accuracy,
        'amplitude_and_bias_float64': diagnostics,
        'training': training,
    }
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output': str(output), 'input_count': len(inputs),
                      'elapsed99_constant_component_fraction': diagnostics['99']['pooled_constant_component_error_fraction'],
                      'last50_epoch_percent_decrease': training['percent_decrease_epoch450_to500'],
                      'last10_epoch_percent_decrease': training['percent_decrease_epoch490_to500'],
                      'inputs_unchanged': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
