#!/usr/bin/env python3
"""Compare two frozen final models on identical physical states/perturbations.

Preparation protocol: t=20,40,60,80,98; 100 states per time; directions from
SG0 error, GDSG error, and seed-20261004 zero-mean spatial Gaussian vectors;
relative L2 amplitudes .001/.01, both signs. Exactly 130 batch100 predict calls.
This finite-amplitude post-hoc diagnostic performs no optimization.
"""
import argparse
from argparse import Namespace
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

from replay_official_cpu import read_inputs, sha, write_json

TIMES = [20, 40, 60, 80, 98]
AMPLITUDES = [.001, .01]
DIRECTIONS = ['sg0_actual_error', 'gdsg_actual_error', 'zero_mean_gaussian']
SIGNS = [1, -1]
MODELS = ['sg0', 'gdsg']
GAUSSIAN_SEED = 20261004
D1_SHA = 'f2b9b948708427bf76fd9387a8545f325752f2b92bada44eb162ebd2eb54d821'


def parse_args():
    base = Path(__file__).resolve().parents[3]
    out = base / 'outputs/due-official-vorticity-20261004'
    work = base / 'work/due_official_vorticity'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=base / 'work/literature_baselines/due')
    p.add_argument('--source-manifest', type=Path, default=work / 'operations/official_source_manifest.json')
    p.add_argument('--test-mat', type=Path, default=work / 'data/VorticityOSG_test.mat')
    p.add_argument('--sg0-run-root', type=Path, default=out / 'sg0-full-recovery/20261004-due-vorticity-1e62910-r1/runs/sg0-full-r1')
    p.add_argument('--gdsg-run-root', type=Path, default=out / 'gdsg-full-recovery/20261004-due-vorticity-1e62910-r1/runs/official-gdsg-r1')
    p.add_argument('--sg0-verification', type=Path, default=out / 'sg0-independent-verification/verification.json')
    p.add_argument('--gdsg-verification', type=Path, default=out / 'gdsg-independent-verification/verification.json')
    p.add_argument('--sg0-d1-dir', type=Path, default=out / 'sg0-error-propagation')
    p.add_argument('--gdsg-d1-dir', type=Path, default=out / 'gdsg-error-propagation')
    p.add_argument('--output-dir', type=Path, default=out / 'common-perturbations-r1')
    return p.parse_args()


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    for protected in (args.source_root, args.sg0_run_root, args.gdsg_run_root, args.sg0_d1_dir, args.gdsg_d1_dir):
        assert not output.is_relative_to(protected.resolve()), 'Output must be outside immutable evidence/source'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        contexts, fingerprints, d1_reports = {}, {}, {}
        for name in MODELS:
            run_root = getattr(args, name + '_run_root')
            info = read_inputs(Namespace(source_root=args.source_root, source_manifest=args.source_manifest,
                                         run_root=run_root, test_mat=args.test_mat))
            source, run, env, receipt, checkpoint, prediction, hashes, is_sg0 = info
            assert is_sg0 == (name == 'sg0')
            verification_path = getattr(args, name + '_verification')
            verification = json.loads(verification_path.read_text())
            assert verification['status'] == 'verified' and verification['engineering']['status'] == 'passed'
            assert Path(verification['run_root']).resolve() == run
            fingerprints.update(hashes)
            fingerprints[str(verification_path.resolve())] = sha(verification_path)
            directory = getattr(args, name + '_d1_dir')
            manifest_path = directory / 'artifact_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for filename in ('ERROR_PROPAGATION_DIAGNOSTIC.json', 'error_propagation_arrays.npz'):
                path = directory / filename
                assert sha(path) == manifest[filename]['sha256'] and path.stat().st_size == manifest[filename]['bytes']
                fingerprints[str(path.resolve())] = sha(path)
            fingerprints[str(manifest_path.resolve())] = sha(manifest_path)
            diagnostic = json.loads((directory / 'ERROR_PROPAGATION_DIAGNOSTIC.json').read_text())
            assert diagnostic['status'] == 'ERROR_PROPAGATION_DIAGNOSTIC_COMPLETE'
            assert diagnostic['script_sha256'] == D1_SHA
            assert diagnostic['variant'] == ('sg0' if name == 'sg0' else 'default_gdsg')
            assert str(diagnostic['slurm_training_job']) == str(env['slurm_job_id'])
            for path, expected in diagnostic['input_sha256'].items():
                assert sha(Path(path)) == expected, 'D1 input changed: ' + path
                fingerprints[path] = expected
            contexts[name] = {'environment': env, 'receipt': receipt, 'checkpoint': checkpoint,
                              'prediction': prediction, 'd1_directory': directory}
            d1_reports[name] = diagnostic
        sg_env, gd_env = contexts['sg0']['environment'], contexts['gdsg']['environment']
        expected_config = copy.deepcopy(sg_env['effective_config']); expected_config['training']['sg_pairing'] = 1
        assert gd_env['official_config'] == expected_config
        for field in ('source_commit', 'source_manifest_sha256', 'data_sha256', 'packages'):
            assert sg_env[field] == gd_env[field], 'Baseline mismatch: ' + field
        assert contexts['sg0']['receipt']['training']['initial_parameter_sha256'] == contexts['gdsg']['receipt']['training']['initial_parameter_sha256']
        os.environ['MPLCONFIGDIR'] = str(output / 'matplotlib-cache')
        os.environ['MPLBACKEND'] = 'Agg'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        import due
        import hashlib
        import numpy as np
        import torch
        from scipy.io import loadmat
        assert Path(due.__file__).resolve().is_relative_to(source)
        assert torch.__version__.split('+')[0] == '2.5.1' and np.__version__ == '1.26.4'
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        def norm(x): return np.linalg.norm(x.reshape(100, -1), axis=1)
        def array_sha(x):
            h = hashlib.sha256()
            h.update(str(x.dtype).encode()); h.update(str(x.shape).encode()); h.update(np.ascontiguousarray(x).tobytes())
            return h.hexdigest()
        def divide(a, b, mask=None):
            valid = (b > 0) if mask is None else ((b > 0) & mask)
            value = np.full(b.shape, np.nan, dtype=np.float64)
            np.divide(a, b, out=value, where=valid)
            return value
        def clean(x):
            if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)): return [clean(v) for v in x]
            if isinstance(x, (bool, np.bool_)): return bool(x)
            if isinstance(x, (int, np.integer)): return int(x)
            if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
            return x
        def stats(x):
            good = np.asarray(x, dtype=np.float64); finite = good[np.isfinite(good)]
            result = {'defined_count': int(finite.size), 'undefined_count': int(good.size - finite.size)}
            if finite.size: result.update(mean=float(finite.mean()), median=float(np.median(finite)),
                                          p10=float(np.quantile(finite, .1)), p90=float(np.quantile(finite, .9)))
            return result
        def paired(sg, gd):
            valid = np.isfinite(sg) & np.isfinite(gd)
            delta = gd[valid] - sg[valid]
            return {'difference_gdsg_minus_sg0': stats(delta), 'paired_count': int(valid.sum()),
                    'gdsg_lower_response_count': int((delta < 0).sum()), 'equal_response_count': int((delta == 0).sum()),
                    'gdsg_higher_response_count': int((delta > 0).sum()),
                    'gdsg_lower_response_fraction': float((delta < 0).mean()) if delta.size else None}
        raw = loadmat(args.test_mat, variable_names=['trajectories', 'dt'], verify_compressed_data_integrity=True)
        truth = raw['trajectories'].astype('single'); dt = raw['dt'].astype('single'); del raw
        assert truth.shape == (100, 64, 64, 1, 100) and dt.shape == (100, 99) and np.all(dt == 1)
        true_states = np.stack([truth[..., t].copy() for t in TIMES])
        selected_dt = np.stack([dt[:, t:t+1].copy() for t in TIMES])
        del truth, dt
        errors, roundoff_references, d1_masks = {}, {}, {}
        for name in MODELS:
            with np.load(contexts[name]['d1_directory'] / 'error_propagation_arrays.npz', allow_pickle=False) as data:
                indices = [list(data['times']).index(t) for t in TIMES]
                errors[name] = data['e'][indices].copy()
                roundoff_references[name] = data['roundoff_reference_norm'][indices].copy()
                d1_masks[name] = data['gain_interpretable'][indices].astype(bool)
                for i in range(5): np.testing.assert_allclose(norm(errors[name][i]), data['norm_e'][indices[i]], rtol=1e-15, atol=0)
            # Verify that the stored direction really belongs to this archived
            # rollout and the common float32 true state, without new inference.
            archived = loadmat(contexts[name]['prediction'], variable_names=['trajectories'], verify_compressed_data_integrity=True)['trajectories']
            assert archived.shape == (100, 64, 64, 1, 100) and archived.dtype == np.dtype('float32')
            for i, t in enumerate(TIMES):
                assert np.array_equal(errors[name][i], archived[..., t].astype(np.float64) - true_states[i].astype(np.float64))
            del archived
        rng = np.random.default_rng(GAUSSIAN_SEED)
        gaussian = rng.standard_normal((5, 100, 64, 64, 1))
        gaussian -= gaussian.mean(axis=(2, 3, 4), keepdims=True)
        assert np.max(np.abs(gaussian.mean(axis=(2, 3, 4)))) < 1e-14
        models, parameter_states = {}, {}
        for name in MODELS:
            models[name] = torch.load(contexts[name]['checkpoint'], map_location='cpu', weights_only=False).to('cpu')
            assert type(models[name]) is due.networks.fno.osg_fno2d
            parameter_states[name] = {k: v.detach().clone() for k, v in models[name].state_dict().items()}
        calls = 0
        shape = (5, 3, 2, 2, 100)
        gains = {name: np.full(shape, np.nan) for name in MODELS}
        symmetric = {name: np.full((5, 3, 2, 100), np.nan) for name in MODELS}
        central = {name: np.full((5, 3, 2, 100), np.nan) for name in MODELS}
        delta_norm = np.full(shape, np.nan); actual_amplitude = np.full(shape, np.nan)
        central_delta_norm = np.full((5, 3, 2, 100), np.nan)
        symmetry_residual = np.full((5, 3, 2, 100), np.nan)
        direction_valid = np.zeros((5, 3, 100), dtype=bool)
        unit_directions = np.zeros((5, 3, 100, 64, 64, 1), dtype=np.float64)
        input_inventory, one_sided_rows, symmetric_rows, summaries = [], [], [], []
        for ti, t in enumerate(TIMES):
            u = true_states[ti]; u64 = u.astype(np.float64); u_norm = norm(u64)
            assert np.isfinite(u).all() and np.all(u_norm > 0)
            base_outputs = {}
            base_hash = array_sha(u)
            for name in MODELS:
                with contextlib.redirect_stdout(io.StringIO()):
                    base_outputs[name] = models[name].predict(u.copy(), selected_dt[ti].copy(), device='cpu')[..., 1].astype(np.float64)
                calls += 1
                assert array_sha(u) == base_hash and np.isfinite(base_outputs[name]).all()
            input_inventory.append({'input_time': t, 'kind': 'common_base', 'input_sha256': base_hash,
                                    'models_received_identical_array': MODELS})
            for di, direction in enumerate(DIRECTIONS):
                raw_direction = errors[MODELS[di]][ti] if di < 2 else gaussian[ti]
                direction_norm = norm(raw_direction)
                valid = np.isfinite(direction_norm) & (direction_norm > 0)
                if di < 2:
                    valid &= direction_norm > 10 * roundoff_references[MODELS[di]][ti]
                    assert np.array_equal(valid, d1_masks[MODELS[di]][ti])
                direction_valid[ti, di] = valid
                unit = np.zeros_like(raw_direction)
                np.divide(raw_direction, direction_norm[:, None, None, None], out=unit,
                          where=valid[:, None, None, None])
                unit_directions[ti, di] = unit
                for ai, amplitude in enumerate(AMPLITUDES):
                    requested_delta = unit * (amplitude * u_norm)[:, None, None, None]
                    shared_inputs = [(u64 + sign * requested_delta).astype('single') for sign in SIGNS]
                    realized = [x.astype(np.float64) - u64 for x in shared_inputs]
                    outputs = {name: [] for name in MODELS}
                    for si, sign in enumerate(SIGNS):
                        x, d = shared_inputs[si], realized[si]
                        dn = norm(d); valid_actual = valid & (dn > 0)
                        delta_norm[ti, di, ai, si] = dn
                        actual_amplitude[ti, di, ai, si] = dn / u_norm
                        fingerprint = array_sha(x)
                        for name in MODELS:
                            # Both model calls use copies of this same array,
                            # constructed once, with equal shape and batch order.
                            call_input = x.copy()
                            assert array_sha(call_input) == fingerprint
                            with contextlib.redirect_stdout(io.StringIO()):
                                result = models[name].predict(call_input, selected_dt[ti].copy(), device='cpu')[..., 1].astype(np.float64)
                            calls += 1
                            assert array_sha(call_input) == fingerprint and np.isfinite(result).all()
                            outputs[name].append(result)
                            gains[name][ti, di, ai, si] = divide(norm(result - base_outputs[name]), dn, valid_actual)
                        input_inventory.append({'input_time': t, 'direction': direction, 'requested_relative_l2_amplitude': amplitude,
                            'sign': sign, 'input_sha256': fingerprint, 'models_received_identical_array': MODELS,
                            'valid_direction_count': int(valid.sum()), 'valid_actual_perturbation_count': int(valid_actual.sum()),
                            'realized_relative_l2_amplitude': stats(dn[valid_actual] / u_norm[valid_actual])})
                        for sample in range(100):
                            one_sided_rows.append(clean({'test_index': sample, 'input_time': t, 'direction': direction,
                                'requested_relative_l2_amplitude': amplitude, 'sign': sign,
                                'valid_direction': valid[sample], 'actual_delta_norm': dn[sample],
                                'actual_relative_l2_amplitude': (dn / u_norm)[sample],
                                'sg0_gain': gains['sg0'][ti, di, ai, si, sample],
                                'gdsg_gain': gains['gdsg'][ti, di, ai, si, sample],
                                'difference_gdsg_minus_sg0': gains['gdsg'][ti, di, ai, si, sample] - gains['sg0'][ti, di, ai, si, sample]}))
                    span = norm(shared_inputs[0].astype(np.float64) - shared_inputs[1].astype(np.float64))
                    central_delta_norm[ti, di, ai] = span
                    symmetry_residual[ti, di, ai] = divide(norm(realized[0] + realized[1]), norm(realized[0]) + norm(realized[1]), valid)
                    for name in MODELS:
                        symmetric[name][ti, di, ai] = gains[name][ti, di, ai].mean(axis=0)
                        central[name][ti, di, ai] = divide(norm(outputs[name][0] - outputs[name][1]), span, valid)
                    item = {'input_time': t, 'direction': direction, 'requested_relative_l2_amplitude': amplitude,
                            'valid_direction_count': int(valid.sum()),
                            'one_sided': {str(sign): {**{name: stats(gains[name][ti, di, ai, si]) for name in MODELS},
                                                      'paired': paired(gains['sg0'][ti, di, ai, si], gains['gdsg'][ti, di, ai, si])}
                                          for si, sign in enumerate(SIGNS)},
                            'symmetric_mean_gain': {**{name: stats(symmetric[name][ti, di, ai]) for name in MODELS},
                                                   'paired': paired(symmetric['sg0'][ti, di, ai], symmetric['gdsg'][ti, di, ai])},
                            'central_secant_gain': {**{name: stats(central[name][ti, di, ai]) for name in MODELS},
                                                  'paired': paired(central['sg0'][ti, di, ai], central['gdsg'][ti, di, ai])},
                            'float32_input_symmetry_residual_relative': stats(symmetry_residual[ti, di, ai])}
                    summaries.append(item)
                    for sample in range(100):
                        symmetric_rows.append(clean({'test_index': sample, 'input_time': t, 'direction': direction,
                            'requested_relative_l2_amplitude': amplitude, 'valid_direction': valid[sample],
                            'sg0_symmetric_mean_gain': symmetric['sg0'][ti, di, ai, sample],
                            'gdsg_symmetric_mean_gain': symmetric['gdsg'][ti, di, ai, sample],
                            'symmetric_difference_gdsg_minus_sg0': symmetric['gdsg'][ti, di, ai, sample] - symmetric['sg0'][ti, di, ai, sample],
                            'sg0_central_secant_gain': central['sg0'][ti, di, ai, sample],
                            'gdsg_central_secant_gain': central['gdsg'][ti, di, ai, sample],
                            'central_difference_gdsg_minus_sg0': central['gdsg'][ti, di, ai, sample] - central['sg0'][ti, di, ai, sample],
                            'input_symmetry_residual_relative': symmetry_residual[ti, di, ai, sample]}))
            print(f'Common perturbations: time {t} complete; {calls}/130 batch100 predict calls', flush=True)
        assert calls == 130
        for name in MODELS:
            assert all(torch.equal(value, parameter_states[name][key]) for key, value in models[name].state_dict().items())
        np.savez_compressed(output / 'common_perturbation_arrays.npz', times=np.array(TIMES), amplitudes=np.array(AMPLITUDES), signs=np.array(SIGNS),
            true_states_float32=true_states, normalized_spatial_directions=unit_directions, direction_valid=direction_valid,
            actual_delta_norm=delta_norm, actual_relative_l2_amplitude=actual_amplitude,
            central_input_span_norm=central_delta_norm, input_symmetry_residual_relative=symmetry_residual,
            **{name + '_one_sided_gain': gains[name] for name in MODELS},
            **{name + '_symmetric_mean_gain': symmetric[name] for name in MODELS},
            **{name + '_central_secant_gain': central[name] for name in MODELS})
        for filename, rows in [('one_sided_gains.csv', one_sided_rows), ('symmetric_gains.csv', symmetric_rows)]:
            with (output / filename).open('x', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        assert all(sha(Path(path)) == expected for path, expected in fingerprints.items())
        manifest = json.loads(args.source_manifest.read_text())
        assert all(sha(source / path) == expected for path, expected in manifest['files'].items())
        report = {'status': 'COMMON_PERTURBATIONS_DIAGNOSTIC_COMPLETE', 'times': TIMES, 'relative_l2_amplitudes': AMPLITUDES,
            'directions': DIRECTIONS, 'signs': SIGNS, 'models': MODELS, 'gaussian_seed': GAUSSIAN_SEED,
            'gaussian_construction': 'One dedicated np.random.default_rng(20261004) draw with shape (5,100,64,64,1); subtract spatial mean separately per sample/time before L2 normalization.',
            'batch_size': 100, 'device': 'cpu', 'predict_calls': calls, 'state_evaluations': calls * 100, 'optimizer_updates': 0,
            'model_selection': 'Both archived final epoch500 networks; no checkpoint selection or parameter changes.',
            'shared_input_contract': 'Each float32 base/perturbed array constructed once; hashes checked on both copies before and after original predict calls.',
            'metric_precision': 'Float32 physical model states/outputs; realized input differences, output differences and all norms use float64.',
            'definitions': {'one_sided_gain': '||F(x_sign)-F(u)|| / ||x_sign-u||, measured using actual float32-rounded input.',
                'symmetric_mean_gain': '(gain_plus+gain_minus)/2',
                'central_secant_gain': '||F(x_plus)-F(x_minus)|| / ||x_plus-x_minus||; finite-amplitude central secant, not an infinitesimal derivative.',
                'paired_difference': 'GDSG minus SG0 on identical base/perturbed states. Lower response is descriptive, not a claim of greater accuracy.',
                'input_symmetry_residual': '||delta_plus+delta_minus||/(||delta_plus||+||delta_minus||); records float32 rounding asymmetry.'},
            'invalid_direction_rule': 'Actual e must exceed 10*the saved D1 roundoff reference max(initial roundtrip norm,eps32||u||). Zero/near-roundoff directions are flagged, evaluated as unperturbed placeholders to retain batch100, and excluded from gains. Gaussian zero norm is likewise excluded. No denominator epsilon or replacement direction.',
            'array_layout': {'one_sided': '[time,direction,amplitude,sign,test_index]', 'symmetric_and_central': '[time,direction,amplitude,test_index]',
                             'normalized_spatial_directions': '[time,direction,test_index,H,W,D]'},
            'conditions': summaries, 'input_inventory': input_inventory,
            'parameters_unchanged': True, 'inputs_unchanged': True, 'official_source_files_unchanged': len(manifest['files']),
            'input_sha256': fingerprints, 'script_sha256': sha(Path(__file__)),
            'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads()},
            'wall_seconds': time.perf_counter() - started,
            'limitations': ['Post-hoc use of an already inspected public benchmark test set; not a held-out confirmation.',
                'Two finite amplitudes and three specified directions do not give a global Lipschitz bound or a true PDE amplification rate.',
                'Positive/negative and central responses reduce particular directional asymmetries but do not eliminate finite-amplitude nonlinear effects.',
                'One training seed per arm; no training-significance claim. This compares complete author training configurations, not an isolated causal regularizer intervention.']}
        write_json(output / 'COMMON_PERTURBATIONS_DIAGNOSTIC.json', clean(report))
        files = ['COMMON_PERTURBATIONS_DIAGNOSTIC.json', 'common_perturbation_arrays.npz', 'one_sided_gains.csv', 'symmetric_gains.csv']
        write_json(output / 'artifact_manifest.json', {name: {'sha256': sha(output / name), 'bytes': (output / name).stat().st_size} for name in files})
        print(json.dumps({'status': report['status'], 'output': str(output), 'predict_calls': calls, 'wall_seconds': report['wall_seconds']}, indent=2), flush=True)
    except BaseException as error:
        write_json(output / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
