#!/usr/bin/env python3
"""Fixed-time D1 error decomposition of a trusted final-epoch DUE checkpoint.

Twelve original one-step predict calls (six times, true and archived states),
CPU/batch100. No optimization or complete rollout. Arrays stay in physical units.
"""
import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

from replay_official_cpu import read_inputs, sha, write_json

TIMES = [0, 20, 40, 60, 80, 98]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source-root', 'source-manifest', 'run-root', 'test-mat', 'verification', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    assert not out.is_relative_to(args.run_root.resolve()) and not out.is_relative_to(args.source_root.resolve())
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        source, run, env, receipt, checkpoint, prediction_file, hashes, sg0 = read_inputs(args)
        verified = json.loads(args.verification.read_text())
        assert verified['status'] == 'verified' and verified['engineering']['status'] == 'passed'
        assert Path(verified['run_root']).resolve() == run
        hashes[str(args.verification.resolve())] = sha(args.verification)
        os.environ['MPLCONFIGDIR'] = str(out / 'matplotlib-cache')
        os.environ['MPLBACKEND'] = 'Agg'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        import due
        import numpy as np
        import torch
        from scipy.io import loadmat
        assert Path(due.__file__).resolve().is_relative_to(source)
        assert torch.__version__.split('+')[0] == '2.5.1' and np.__version__ == '1.26.4'
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        raw = loadmat(args.test_mat, variable_names=['trajectories', 'dt'], verify_compressed_data_integrity=True)
        truth = raw['trajectories'].astype('single'); dt = raw['dt'].astype('single'); del raw
        archived = loadmat(prediction_file, variable_names=['trajectories'], verify_compressed_data_integrity=True)['trajectories']
        assert truth.shape == archived.shape == (100, 64, 64, 1, 100)
        assert dt.shape == (100, 99) and np.all(dt == 1)
        assert archived.dtype == truth.dtype == np.dtype('float32')
        assert np.isfinite(truth).all() and np.isfinite(archived).all()
        model = torch.load(checkpoint, map_location='cpu', weights_only=False).to('cpu')
        assert type(model) is due.networks.fno.osg_fno2d
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        def norm(value):
            return np.linalg.norm(value.reshape(100, -1), axis=1)
        def ratio(numerator, denominator):
            result = np.full(denominator.shape, np.nan, dtype=np.float64)
            np.divide(numerator, denominator, out=result, where=denominator > 0)
            return result
        def finite_json(value):
            if isinstance(value, dict): return {key: finite_json(v) for key, v in value.items()}
            if isinstance(value, (tuple, list)): return [finite_json(v) for v in value]
            if isinstance(value, (bool, np.bool_)): return bool(value)
            if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
            if isinstance(value, (int, np.integer)): return int(value)
            return value
        def summary(values):
            values = np.asarray(values, dtype=np.float64)
            good = values[np.isfinite(values)]
            if not good.size: return {'defined_count': 0, 'undefined_count': int(values.size)}
            return {'defined_count': int(good.size), 'undefined_count': int(values.size - good.size),
                    'mean': float(good.mean()), 'median': float(np.median(good)),
                    'p10': float(np.quantile(good, .1)), 'p90': float(np.quantile(good, .9)),
                    'minimum': float(good.min()), 'maximum': float(good.max())}
        initial_roundtrip_norm = norm(archived[..., 0].astype(np.float64) - truth[..., 0].astype(np.float64))
        vectors = {key: [] for key in ('a', 'b', 'e', 'interface_difference')}
        scalar_arrays, rows, by_time = {}, [], []
        calls = 0
        for t in TIMES:
            # Separate one-step calls preserve the author's batch100 interface.
            # predict handles normalization and returns denormalized physical states.
            with contextlib.redirect_stdout(io.StringIO()):
                f_true = model.predict(truth[..., t].copy(), dt[:, t:t+1].copy(), device='cpu')[..., 1]
                f_hat = model.predict(archived[..., t].copy(), dt[:, t:t+1].copy(), device='cpu')[..., 1]
            calls += 2
            assert np.isfinite(f_true).all() and np.isfinite(f_hat).all()
            u = truth[..., t].astype(np.float64)
            target = truth[..., t + 1].astype(np.float64)
            hat = archived[..., t].astype(np.float64)
            hat_next = archived[..., t + 1].astype(np.float64)
            f_true = f_true.astype(np.float64); f_hat = f_hat.astype(np.float64)
            a, b, e = f_hat - f_true, f_true - target, hat - u
            total, archived_error = f_hat - target, hat_next - target
            interface = f_hat - hat_next
            persistence = u - target
            an, bn, en, tn = norm(a), norm(b), norm(e), norm(total)
            un, yn, pn = norm(u), norm(target), norm(persistence)
            assert np.all(un > 0) and np.all(yn > 0)
            dot = (a * b).reshape(100, -1).sum(axis=1, dtype=np.float64)
            cross = 2 * dot
            cosine = ratio(dot, an * bn)
            defined = np.isfinite(cosine)
            assert np.all(np.abs(cosine[defined]) <= 1 + 1e-12)
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            raw_gain = ratio(an, en)
            # A flag, never a denominator epsilon: t0 is not interpreted. At
            # later times, exclude perturbations within 10 times the larger of
            # observed initial roundtrip error and eps32*current true-state norm.
            roundoff_scale = np.maximum(initial_roundtrip_norm, np.finfo(np.float32).eps * un)
            valid_gain = (en > 10 * roundoff_scale) & (t != 0)
            effective_gain = np.where(valid_gain, raw_gain, np.nan)
            residual = total - (a + b)
            corrected_residual = archived_error - (a + b - interface)
            squared_residual = tn * tn - (an * an + bn * bn + cross)
            # Verify algebra independently of any model-accuracy judgment.
            vector_scale = max(1.0, float(np.max(np.abs(total))))
            assert float(np.max(np.abs(residual))) <= 1e-12 * vector_scale
            assert float(np.max(np.abs(corrected_residual))) <= 1e-12 * vector_scale
            square_scale = np.maximum(1.0, tn * tn + an * an + bn * bn + np.abs(cross))
            assert np.all(np.abs(squared_residual) <= 1e-12 * square_scale)
            scalars = {
                'norm_a': an, 'norm_b': bn, 'norm_e': en, 'norm_total_mapped_error': tn,
                'norm_archived_next_error': norm(archived_error), 'norm_truth_current': un, 'norm_truth_next': yn,
                'a_relative_to_next_truth': an / yn, 'b_relative_to_next_truth': bn / yn,
                'e_relative_to_current_truth': en / un, 'mapped_total_relative_to_next_truth': tn / yn,
                'archived_next_relative_to_truth': norm(archived_error) / yn,
                'a_dot_b': dot, 'cross_term_2_a_dot_b': cross,
                'cosine_a_b': cosine, 'angle_a_b_degrees': angle,
                'raw_directional_gain': raw_gain, 'interpretable_directional_gain': effective_gain,
                'gain_interpretable': valid_gain.astype(np.int8), 'roundoff_reference_norm': roundoff_scale,
                'initial_roundtrip_norm': initial_roundtrip_norm,
                'interface_difference_norm': norm(interface),
                'interface_difference_relative_to_archived_next': ratio(norm(interface), norm(hat_next)),
                'interface_difference_relative_to_truth_next': norm(interface) / yn,
                'interface_difference_max_abs': np.max(np.abs(interface).reshape(100, -1), axis=1),
                'vector_identity_residual_norm': norm(residual),
                'archived_corrected_identity_residual_norm': norm(corrected_residual),
                'squared_norm_identity_residual': squared_residual,
                'persistence_relative_to_next_truth': pn / yn,
                'local_model_error_over_persistence_error': ratio(bn, pn),
            }
            for key, value in scalars.items(): scalar_arrays.setdefault(key, []).append(value)
            for key, value in [('a', a), ('b', b), ('e', e), ('interface_difference', interface)]: vectors[key].append(value)
            for index in range(100):
                rows.append(finite_json({'test_index': index, 'elapsed_input_time': t, 'elapsed_target_time': t + 1,
                                        **{key: value[index] for key, value in scalars.items()}}))
            item = {'elapsed_input_time': t, 'elapsed_target_time': t + 1,
                    **{key: summary(value) for key, value in scalars.items() if key not in {'gain_interpretable', 'initial_roundtrip_norm', 'roundoff_reference_norm'}},
                    'gain_interpretable_count': int(valid_gain.sum()),
                    'local_model_better_than_persistence_count': int((bn < pn).sum()),
                    'local_model_equal_to_persistence_count': int((bn == pn).sum()),
                    'negative_cross_term_count': int((cross < 0).sum()),
                    'positive_cross_term_count': int((cross > 0).sum())}
            by_time.append(item)
            print(f'D1 input time {t}: 2 batch100 predictions; identities checked', flush=True)
        assert calls == 12 and all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
        np.savez_compressed(out / 'error_propagation_arrays.npz', times=np.array(TIMES), test_indices=np.arange(100),
                            **{key: np.stack(value) for key, value in vectors.items()},
                            **{key: np.stack(value) for key, value in scalar_arrays.items()})
        with (out / 'per_trajectory_error_propagation.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        unchanged = all(sha(Path(path)) == digest for path, digest in hashes.items())
        assert unchanged
        report = {'status': 'ERROR_PROPAGATION_DIAGNOSTIC_COMPLETE', 'variant': 'sg0' if sg0 else 'default_gdsg',
                  'model': 'final epoch500', 'slurm_training_job': env['slurm_job_id'], 'source_commit': env['source_commit'],
                  'elapsed_input_times': TIMES, 'trajectories_per_time': 100, 'device': 'cpu', 'batch_size': 100,
                  'one_step_predict_calls': calls, 'total_states_evaluated': calls * 100, 'optimizer_updates': 0,
                  'parameters_unchanged': True, 'inputs_unchanged': unchanged,
                  'metric_precision': 'Official float32 physical truth/archived states and model outputs; subtraction, inner products and spatial L2 norms in float64.',
                  'definitions': {'b': 'F(u_t)-u_(t+1)', 'a': 'F(uhat_t)-F(u_t)', 'e': 'uhat_t-u_t',
                      'mapped_total': 'F(uhat_t)-u_(t+1)=a+b', 'squared_norm': '||a+b||^2=||a||^2+||b||^2+2<a,b>',
                      'interface_difference': 'F(uhat_t)-archived_uhat_(t+1)',
                      'archived_error_identity': 'archived_uhat_(t+1)-u_(t+1)=a+b-interface_difference',
                      'persistence': 'u_t as a zero-training prediction of u_(t+1); same target norm as b',
                      'directional_gain': '||a||/||e|| along the observed finite perturbation; not a global Lipschitz constant.'},
                  'gain_interpretation_rule': 't=0 always uninterpretable. Later points require ||e|| > 10*max(observed initial roundtrip norm, eps_float32*||u_t||); zero denominators remain undefined. No epsilon is added to a denominator. Raw ratios remain recorded separately.',
                  'interface_difference_scope': 'Includes CPU/GPU arithmetic and extra physical-to-normalized-to-physical conversion, since archived free rollout recurred internally in normalized coordinates; these causes are not individually identified.',
                  'by_time': by_time, 'input_sha256': hashes, 'script_sha256': sha(Path(__file__)),
                  'runtime': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__, 'threads': torch.get_num_threads()},
                  'wall_seconds': time.perf_counter() - start,
                  'limitations': ['Post-hoc, seed0 benchmark diagnosis; not an independent confirmation set.',
                      'Only six fixed times and observed error directions are tested. Small b or large directional gain does not isolate the cause or establish a global bound.',
                      'Persistence is a zero-training reference; no extra model call or fitted baseline is used.',
                      'This step does not repeat the 99-step free rollout or all true-state restarts.']}
        write_json(out / 'ERROR_PROPAGATION_DIAGNOSTIC.json', finite_json(report))
        names = ['ERROR_PROPAGATION_DIAGNOSTIC.json', 'error_propagation_arrays.npz', 'per_trajectory_error_propagation.csv']
        write_json(out / 'artifact_manifest.json', {name: {'sha256': sha(out / name), 'bytes': (out / name).stat().st_size} for name in names})
        print(json.dumps({'status': report['status'], 'variant': report['variant'], 'output': str(out), 'wall_seconds': report['wall_seconds']}, indent=2), flush=True)
    except BaseException as error:
        write_json(out / 'FAILURE.json', {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
