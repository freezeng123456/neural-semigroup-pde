#!/usr/bin/env python3
"""Verify recovered official DUE VorticityOSG outputs without training or inference.

Reads MAT/CSV/JSON artifacts and the wrapper's plain training-state checkpoint.
Whole-network pickle files are hashed, never imported or executed. Use only on
trusted artifacts from the recorded official run. Output must be outside run-root.

Two metric tracks are reported:
  author_float32: source MAT truth cast as the official loader does; used for CSV parity.
  physical_float64: original source MAT truth with float64 arithmetic; independent report.
Neither an engineering pass nor finite errors proves published accuracy reproduced.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import scipy
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
EXPECTED_COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
EXPECTED_TEST_SHA = 'a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a'
EXPECTED_TRAIN_SHA = '0ebdd969d52292b4304d3583b3cd432c288397b367f75372d06944d629c5d111'
CSV_ATOL, CSV_RTOL = 1e-7, 1e-5  # Set before observing recovered official output.


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_value(value):
    """Represent nonfinite metrics by null plus explicit counts, never invalid JSON."""
    if isinstance(value, np.ndarray):
        return strict_value(value.tolist())
    if isinstance(value, np.generic):
        return strict_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(v) for v in value]
    return value


def metrics(prediction, truth, dtype):
    """For (N,H,W,D,T), return per-(trajectory,component,time) and joint ratios."""
    if prediction.shape != truth.shape or prediction.ndim != 5:
        raise ValueError('Prediction and truth require identical (N,H,W,D,T) shapes')
    n, h, w, d, t = truth.shape
    per_time = np.empty((n, d, t), dtype=dtype)
    per_joint = np.empty((n, d), dtype=dtype)
    zero_denominators = 0
    global_error_squared = np.float64(0)
    global_truth_squared = np.float64(0)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        for i in range(n):
            p = np.asarray(prediction[i], dtype=dtype)
            u = np.asarray(truth[i], dtype=dtype)
            delta = p - u
            denominator = np.linalg.norm(u.reshape(h * w, d, t), ord=2, axis=0)
            numerator = np.linalg.norm(delta.reshape(h * w, d, t), ord=2, axis=0)
            per_time[i] = numerator / denominator
            zero_denominators += int(np.count_nonzero(denominator == 0))
            # Same transpose/flatten axes as due/utils.py pde2dregular_evaluate.
            joint_denominator = np.linalg.norm(u.transpose(3, 0, 1, 2).reshape(-1, d), ord=2, axis=0)
            joint_numerator = np.linalg.norm(delta.transpose(3, 0, 1, 2).reshape(-1, d), ord=2, axis=0)
            per_joint[i] = joint_numerator / joint_denominator
            global_error_squared += np.sum(np.square(delta, dtype=np.float64), dtype=np.float64)
            global_truth_squared += np.sum(np.square(u, dtype=np.float64), dtype=np.float64)
        curve = np.mean(per_time, axis=0)  # (D,T); author CSV stores its transpose.
        return {'per_trajectory_component_time': per_time,
                'mean_over_trajectories_curve': curve,
                'time_and_trajectory_mean_relative_l2': float(np.mean(per_time)),
                'per_trajectory_component_joint_spacetime': per_joint,
                'mean_trajectory_joint_spacetime_relative_l2': float(np.mean(per_joint)),
                'globally_pooled_spacetime_relative_l2': float(np.sqrt(global_error_squared / global_truth_squared)),
                'zero_spatial_truth_norm_count': zero_denominators,
                'nonfinite_per_time_ratio_count': int(np.count_nonzero(~np.isfinite(per_time))),
                'nonfinite_joint_ratio_count': int(np.count_nonzero(~np.isfinite(per_joint)))}


def compare_metrics(actual, expected, atol=CSV_ATOL, rtol=CSV_RTOL):
    same_shape = actual.shape == expected.shape
    if not same_shape:
        return {'same_shape': False, 'matches': False, 'atol': atol, 'rtol': rtol}
    finite = np.isfinite(actual) & np.isfinite(expected)
    equal_nonfinite = ((np.isnan(actual) & np.isnan(expected)) |
                       (np.isposinf(actual) & np.isposinf(expected)) |
                       (np.isneginf(actual) & np.isneginf(expected)))
    finite_close = np.zeros(actual.shape, dtype=bool)
    finite_close[finite] = np.abs(actual[finite] - expected[finite]) <= atol + rtol * np.abs(expected[finite])
    return {'same_shape': True, 'matches': bool(np.all(finite_close | equal_nonfinite)),
            'finite_pair_count': int(finite.sum()), 'matching_nonfinite_pair_count': int(equal_nonfinite.sum()),
            'max_abs_difference_on_finite_pairs': float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else None,
            'atol': atol, 'rtol': rtol}


def self_test():
    truth = np.zeros((2, 2, 1, 1, 3), dtype=np.float64)
    truth[0, 0, 0, 0] = [1, 1, 10]
    truth[1, 0, 0, 0] = [1, 1, 2]
    prediction = truth.copy()
    prediction[0, 0, 0, 0, 1:] += [1, 1]
    prediction[1, 0, 0, 0, 2] += 2
    result = metrics(prediction, truth, np.float64)
    expected_per_time = np.array([[[0, 1, .1]], [[0, 0, 1]]])
    expected_joint = np.array([[math.sqrt(2 / 102)], [math.sqrt(4 / 6)]])
    np.testing.assert_allclose(result['per_trajectory_component_time'], expected_per_time, atol=1e-15)
    np.testing.assert_allclose(result['mean_over_trajectories_curve'], [[0, .5, .55]], atol=1e-15)
    np.testing.assert_allclose(result['per_trajectory_component_joint_spacetime'], expected_joint, atol=1e-15)
    assert math.isclose(result['time_and_trajectory_mean_relative_l2'], .35)
    assert math.isclose(result['globally_pooled_spacetime_relative_l2'], math.sqrt(6 / 108))
    assert not math.isclose(result['time_and_trajectory_mean_relative_l2'], result['mean_trajectory_joint_spacetime_relative_l2'])
    assert not math.isclose(result['globally_pooled_spacetime_relative_l2'], result['mean_trajectory_joint_spacetime_relative_l2'])
    scaled = metrics(7 * prediction, 7 * truth, np.float64)
    np.testing.assert_allclose(scaled['per_trajectory_component_time'], expected_per_time, atol=1e-15)
    direct_official_curve = np.mean(np.linalg.norm((prediction-truth).reshape(2, 2, 1, 3), ord=2, axis=1) / np.linalg.norm(truth.reshape(2, 2, 1, 3), ord=2, axis=1), axis=0)
    direct_official_joint = np.mean(np.linalg.norm((prediction-truth).transpose(0, 4, 1, 2, 3).reshape(2, -1, 1), ord=2, axis=1) / np.linalg.norm(truth.transpose(0, 4, 1, 2, 3).reshape(2, -1, 1), ord=2, axis=1))
    np.testing.assert_allclose(result['mean_over_trajectories_curve'], direct_official_curve, atol=1e-15)
    assert math.isclose(result['mean_trajectory_joint_spacetime_relative_l2'], direct_official_joint)
    native = metrics(prediction.astype(np.float32), truth.astype(np.float32), np.float32)
    assert compare_metrics(native['mean_over_trajectories_curve'], direct_official_curve)['matches']
    corrupted = direct_official_curve.copy(); corrupted[0, 1] += .01
    assert not compare_metrics(corrupted, direct_official_curve)['matches']
    selection_losses = [1.00000005, 1.00000004]
    assert np.float32(selection_losses[0]) == np.float32(selection_losses[1])
    assert int(np.argmin(np.asarray(selection_losses, dtype=np.float32))) == 0
    assert min(enumerate(selection_losses), key=lambda pair: pair[1])[0] == 1
    zero = np.zeros((1, 1, 1, 1, 1))
    undefined = metrics(zero, zero, np.float64)
    assert undefined['zero_spatial_truth_norm_count'] == 1 and undefined['nonfinite_per_time_ratio_count'] == 1
    return {'status': 'passed', 'scope': 'Synthetic arithmetic checks only; no official training, model inference, or fabricated official result.',
            'mean_pointwise_relative_l2': result['time_and_trajectory_mean_relative_l2'],
            'mean_trajectory_joint_spacetime_relative_l2': result['mean_trajectory_joint_spacetime_relative_l2'],
            'globally_pooled_spacetime_relative_l2': result['globally_pooled_spacetime_relative_l2'],
            'checked': ['author formula parity', 'different metric aggregation', 'scale invariance', 'CSV mismatch detection', 'undefined zero-norm denominator reporting', 'original precision checkpoint selection versus rounded history']}


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def inspect_checkpoint_values(value, torch, path='root'):
    """Count tensor/scalar nonfinites without loading the author's model classes."""
    rows = []
    if torch.is_tensor(value):
        rows.append({'path': path, 'shape': list(value.shape), 'dtype': str(value.dtype), 'finite': bool(torch.isfinite(value).all()), 'device': str(value.device)})
    elif isinstance(value, dict):
        for key, child in value.items(): rows.extend(inspect_checkpoint_values(child, torch, f'{path}.{key}'))
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value): rows.extend(inspect_checkpoint_values(child, torch, f'{path}[{i}]'))
    elif isinstance(value, np.ndarray):
        rows.append({'path': path, 'shape': list(value.shape), 'dtype': str(value.dtype), 'finite': bool(np.isfinite(value).all()), 'device': 'numpy_cpu'})
    elif isinstance(value, float):
        rows.append({'path': path, 'shape': [], 'dtype': 'python_float', 'finite': math.isfinite(value), 'device': 'cpu'})
    return rows


def verify(args):
    begin = time.perf_counter()
    root, output = args.run_root.resolve(), args.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError('Output directory must be outside the recovered run tree')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite verification output: {output}')
    checks, fingerprints, quality = {}, {}, {}
    def check(name, condition):
        checks[name] = bool(condition)
    def record(path):
        path = Path(path).resolve()
        fingerprints[str(path)] = sha256(path)
        return path
    def read_json(path):
        return json.loads(record(path).read_text())
    report = {'schema_version': 'due-vorticity-official-output-verification-v1',
              'scope': 'Offline artifact, checkpoint-state and metric verification; no training or model forward pass.',
              'run_root': str(root), 'test_mat': str(args.test_mat.resolve()), 'checks': checks,
              'prediction_quality': quality, 'independently_recomputed_metrics': {},
              'metric_definitions': {
                  'curve': 'e[n,d,k]=||pred[n,:,:,d,k]-truth[n,:,:,d,k]||_2 / ||truth[n,:,:,d,k]||_2; official CSV is mean_n(e), transposed to (T,D).',
                  'joint': 'For each n,d, concatenate all spatial points and times, take the norm ratio, then average those trajectory/component ratios. This equals the author printed Relative error is value.',
                  'pooled': 'One norm ratio after pooling all trajectories, components, times and spatial points; supplementary, not the author printed metric.',
                  'time_axis': '100 supplied states / 99 unit increments. Index 0 is the supplied initial state; relative elapsed times 0..99. Absolute start time is unspecified.',
                  'no_accuracy_gate': 'No acceptance threshold for scientific quality is invented. Engineering completion and finite metrics do not certify the published accuracy or superiority.'},
              'runtime': {'python': sys.version, 'executable': sys.executable, 'numpy': np.__version__, 'scipy': scipy.__version__},
              'script_sha256': sha256(Path(__file__)), 'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        complete = read_json(root / 'OFFICIAL_COMPLETE.json')
        environment = read_json(root / 'audit/environment.json')
        training = read_json(root / 'audit/training_boundary.json')
        plan = read_json(root / 'audit/model_and_training_plan.json')
        checkpoint_replays = read_json(root / 'audit/checkpoint_forward_replays.json')
        prediction_receipt = read_json(root / 'audit/official_batch100_prediction_replay.json')
        source_manifest = read_json(args.source_manifest)
        check('official_completion_receipt', complete['status'] == 'OFFICIAL_500_EPOCHS_COMPLETE' and complete['complete_official_reproduction'] is True)
        check('full_mode_seed_zero', environment['mode'] == 'full' and complete['seed'] == 0 and environment['official_config']['seed'] == 0)
        check('source_commit', environment['source_commit'] == source_manifest['commit'] == EXPECTED_COMMIT)
        check('source_manifest_hash', environment['source_manifest_sha256'] == fingerprints[str(args.source_manifest.resolve())])
        check('wrapper_hash', environment['wrapper_sha256'] == sha256(record(args.wrapper)))
        check('original_data_hashes', environment['data_sha256'] == {'VorticityOSG_train.mat': EXPECTED_TRAIN_SHA, 'VorticityOSG_test.mat': EXPECTED_TEST_SHA})
        check('unchanged_test_mat', sha256(record(args.test_mat)) == EXPECTED_TEST_SHA)
        for name in ('VorticityOSG.py', 'network.py', 'config.yaml'):
            check(f'official_workdir_source:{name}', sha256(record(root / 'official_workdir' / name)) == source_manifest['files']['examples/VorticityOSG/' + name])
        check('training_receipt_consistency', complete['training'] == training and complete['checkpoints'] == checkpoint_replays and complete['prediction'] == prediction_receipt)
        check('completed500epochs', training['completed_epochs'] == training['configured_epochs'] == complete['completed_epochs'] == complete['configured_epochs'] == 500)
        check('observed62500updates', training['optimizer_updates'] == 62500)
        check('training_data_batches', plan['trainX_shape'] == [2500, 64, 64, 2] and plan['trainY_shape'] == [2500, 64, 64, 1] and plan['batches_per_epoch'] == 125 and plan['official_batch_size'] == 20)
        check('parameter_digest_changed', training['initial_parameter_sha256'] == plan['initial_parameter_sha256'] and training['initial_parameter_sha256'] != training['final_parameter_sha256'])
        check('no_failure_receipt', not (root / 'FAILURE.json').exists())
        for name, expected in complete['official_artifact_sha256'].items():
            check(f'official_artifact_hash:{name}', sha256(record(root / 'official_workdir/model' / name)) == expected)
        check('required_artifact_receipts', {'model', 'pred.mat', 'rel_err.csv', 'training_history.csv', 'training_history.png', 'rel_err.png'}.issubset(complete['official_artifact_sha256']))
        checkpoint_path = record(root / 'audit' / training['final_network_file'])
        check('final_checkpoint_hash', sha256(checkpoint_path) == training['final_network_sha256'] == checkpoint_replays['final_epoch']['sha256'])
        check('best_checkpoint_hash', sha256(record(root / 'official_workdir/model/model')) == checkpoint_replays['best_training_objective']['sha256'])
        for key in ('final_epoch', 'best_training_objective'):
            check(f'wrapper_checkpoint_replay:{key}', checkpoint_replays[key]['finite'] is True and checkpoint_replays[key]['reload_replay_passed'] is True and checkpoint_replays[key]['probe_batch_size'] == 20)
        check('wrapper_prediction_scope', prediction_receipt['test_shape'] == [100, 64, 64, 1, 100] and prediction_receipt['prediction_batch_size'] == 100 and prediction_receipt['prediction_steps'] == 99 and prediction_receipt['relative_elapsed_time_range'] == [0, 99])
        history = np.loadtxt(record(root / 'official_workdir/model/training_history.csv'), ndmin=2)
        saved_history = np.loadtxt(record(root / 'audit/epoch500_final_training_history.csv'), ndmin=2)
        check('history500rows', history.shape == (500, 1))
        check('history_finite', bool(np.isfinite(history).all()))
        check('history_snapshot_matches_official', np.array_equal(history, saved_history))
        events = [json.loads(line) for line in record(root / 'audit/epoch_events.jsonl').read_text().splitlines() if line.strip()]
        check('epoch_event_sequence', [row['epoch'] for row in events] == list(range(1, 501)))
        check('epoch_update_sequence', [row['optimizer_updates'] for row in events] == [125 * epoch for epoch in range(1, 501)])
        event_history = np.asarray([row['training_loss'] for row in events], dtype=np.float32).astype(np.float64).reshape(-1, 1)
        check('history_matches_epoch_events', np.array_equal(history, event_history))
        best_event = min(events, key=lambda row: row['training_loss'])
        check('best_selection_receipt', checkpoint_replays['best_training_objective']['selected_epoch'] == best_event['epoch'])
        # The audit snapshot is a tensor dictionary with NumPy RNG metadata, not a DUE model pickle.
        import torch
        state_path = record(root / 'audit/epoch500_final_training_state.pt')
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        checkpoint_values = inspect_checkpoint_values(state, torch)
        check('training_state_all_tensors_scalars_finite', bool(checkpoint_values) and all(row['finite'] for row in checkpoint_values))
        check('checkpoint500epochs62500updates', state['epoch'] == 500 and state['optimizer_updates'] == 62500)
        check('checkpoint_history_consistent', np.array_equal(state['history'].cpu().numpy(), history))
        check('checkpoint_parameter_digest', state_digest(state['network_state_dict']) == training['final_parameter_sha256'])
        optimizer_steps = [float(row['step']) for row in state['optimizer_state_dict']['state'].values()]
        check('adam_tensor_steps62500', len(optimizer_steps) == len(plan['parameter_tensors']) and all(step == 62500 for step in optimizer_steps))
        check('scheduler62500steps', state['scheduler_state_dict']['last_epoch'] == state['scheduler_state_dict']['T_max'] == 62500)
        report['checkpoint_state_verification'] = {'tensor_scalar_inventory': checkpoint_values, 'adam_steps': optimizer_steps, 'torch': torch.__version__, 'whole_network_pickles_executed': False}
        report['training'] = {'epochs': 500, 'updates': 62500, 'batches_per_epoch': 125, 'training_initial_objective': float(history[0, 0]), 'training_final_objective': float(history[-1, 0]), 'training_best_objective': best_event['training_loss'], 'best_training_objective_epoch': best_event['epoch'],
                              'training_best_objective_float32_history': float(np.min(history)),
                              'best_selection_basis': 'Earliest minimum of the epoch events original Python-float objective, matching the author save rule; float32 history can round distinct losses into ties.',
                              'prediction_model': complete['main_prediction_model']}
        del state
        truth_data = loadmat(args.test_mat, verify_compressed_data_integrity=True)
        truth, dt = truth_data['trajectories'], truth_data['dt']
        prediction = loadmat(record(root / 'official_workdir/model/pred.mat'), verify_compressed_data_integrity=True)['trajectories']
        check('test_shape_and_unit_times', truth.shape == (100, 64, 64, 1, 100) and dt.shape == (100, 99) and bool(np.all(dt == 1)))
        check('prediction_shape', prediction.shape == truth.shape)
        check('prediction_dtype_matches_official_single', prediction.dtype == np.dtype('float32'))
        check('truth_finite', bool(np.isfinite(truth).all()))
        quality.update(finite_prediction=bool(np.isfinite(prediction).all()), nonfinite_prediction_values=int(np.count_nonzero(~np.isfinite(prediction))), prediction_dtype=str(prediction.dtype), raw_truth_dtype=str(truth.dtype), published_accuracy_reproduced='not_assessed')
        check('prediction_finiteness_matches_receipt', prediction_receipt['finite'] == quality['finite_prediction'])
        raw_csv = np.loadtxt(record(root / 'official_workdir/model/rel_err.csv'), ndmin=2)
        check('official_csv_shape', raw_csv.shape == (100, 1))
        quality['finite_official_csv'] = bool(np.isfinite(raw_csv).all())
        tracks = {}
        for label, dtype in (('author_float32', np.float32), ('physical_float64', np.float64)):
            tracks[label] = metrics(prediction, truth, dtype)
        parity = compare_metrics(tracks['author_float32']['mean_over_trajectories_curve'].T, raw_csv)
        check('official_rel_err_csv_reproduced', parity['matches'])
        report['official_csv_parity'] = parity
        quality['finite_float64_metrics'] = tracks['physical_float64']['nonfinite_per_time_ratio_count'] == 0 and tracks['physical_float64']['nonfinite_joint_ratio_count'] == 0
        quality['zero_spatial_truth_norm_count'] = tracks['physical_float64']['zero_spatial_truth_norm_count']
        quality['initial_state_max_abs_error'] = float(np.max(np.abs(prediction[..., 0].astype(np.float64) - truth[..., 0])))
        # The supplied initial state is not a forecast; report both inclusive and forecast-only summaries.
        forecast_only = metrics(prediction[..., 1:], truth[..., 1:], np.float64)
        report['independently_recomputed_metrics'] = {**tracks, 'physical_float64_forecast_only_indices1to99': forecast_only}
        log = root / 'official_stdout.log'
        if log.exists():
            numbers = re.findall(r'Relative error is:\s*([^\s]+)', record(log).read_text())
            if len(numbers) == 1:
                printed = float(numbers[0])
                result = compare_metrics(np.asarray([tracks['author_float32']['mean_trajectory_joint_spacetime_relative_l2']]), np.asarray([printed]))
                report['official_printed_joint_parity'] = {'printed_value': printed, **result}
                check('official_printed_joint_reproduced', result['matches'])
            else:
                check('single_official_printed_joint_metric', False)
        for path, before in fingerprints.items():
            check(f'input_unchanged:{path}', sha256(path) == before)
    except Exception as exc:
        check('verification_exception', False)
        report['verification_error'] = {'type': type(exc).__name__, 'message': str(exc)}
    engineering_pass = bool(checks) and all(checks.values())
    quality_ready = quality.get('finite_prediction', False) and quality.get('finite_official_csv', False) and quality.get('finite_float64_metrics', False)
    report['engineering'] = {'status': 'passed' if engineering_pass else 'failed_or_incomplete', 'check_count': len(checks), 'failed_checks': [name for name, passed in checks.items() if not passed]}
    quality['status'] = 'finite_metrics_reported_accuracy_not_judged' if quality_ready else 'nonfinite_undefined_or_unavailable_metrics'
    report['status'] = 'verified' if engineering_pass and quality_ready else ('engineering_passed_quality_issue' if engineering_pass else 'verification_failed_or_incomplete')
    report['input_sha256'] = fingerprints
    report['elapsed_seconds'] = time.perf_counter() - begin
    output.mkdir(parents=True, exist_ok=False)
    (output / 'verification.json').write_text(json.dumps(strict_value(report), indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'engineering': report['engineering'], 'quality_status': quality['status'], 'output': str(output / 'verification.json')}, indent=2))
    return 0 if engineering_pass and quality_ready else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--self-test-output', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--test-mat', type=Path, default=HERE.parent / 'data/VorticityOSG_test.mat')
    parser.add_argument('--source-manifest', type=Path, default=HERE / 'official_source_manifest.json')
    parser.add_argument('--wrapper', type=Path, default=HERE / 'run_official_observed.py')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        if args.self_test_output:
            if args.self_test_output.exists(): raise FileExistsError(args.self_test_output)
            args.self_test_output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    if args.run_root is None or args.output_dir is None:
        parser.error('--run-root and --output-dir are required unless --self-test')
    return verify(args)


if __name__ == '__main__':
    raise SystemExit(main())
