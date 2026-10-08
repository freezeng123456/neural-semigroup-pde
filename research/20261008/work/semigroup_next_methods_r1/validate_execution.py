#!/usr/bin/env python3
"""Bounded actual-update, checkpoint continuation and numerical validation."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from train import Trainer, build_parser, parameter_sha, tensor_sha, sha
from experiment_config import ARMS, SPECS
from flow_numerics import differentiate, quintic_sample, rk4


def snapshot(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: snapshot(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return type(obj)(snapshot(v) for v in obj)
    return copy.deepcopy(obj)


def compare(a, b, rtol=0., atol=0.):
    if isinstance(b, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=rtol, atol=atol)
    elif isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b:
            compare(a[k], b[k], rtol, atol)
    elif isinstance(b, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            compare(x, y, rtol, atol)
    else:
        assert a == b, (a, b)


def numerical_checks():
    checks = []
    t = np.array([0., .6, 1.7, 2.2, 3.6, 4.5, 5.1])
    # Independent analytic polynomial tests catch wrong physical-time scaling
    # and boundary weights; cubic/quartic tests require the five-point stencil.
    for order in (3, 5):
        degree = order-1
        value = np.stack([t**degree, 2*t**degree+3*t+1], axis=1)
        expected1 = np.stack([degree*t**(degree-1), 2*degree*t**(degree-1)+3], axis=1)
        expected2 = np.stack([degree*(degree-1)*t**(degree-2), 2*degree*(degree-1)*t**(degree-2)], axis=1)
        v, acc = differentiate(t, value, order)
        np.testing.assert_allclose(v, expected1, rtol=2e-12, atol=2e-11)
        np.testing.assert_allclose(acc, expected2, rtol=2e-12, atol=2e-11)
        checks.append({'name': f'polynomial_derivatives_{order}_points', 'passed': True,
                       'max_d1_difference': float(np.max(abs(v-expected1))),
                       'max_d2_difference': float(np.max(abs(acc-expected2)))})
    # CFO's actual three-point formulas, independently evaluated on arbitrary
    # non-polynomial data at every interior knot and both boundaries.
    y = np.stack([np.sin(t), np.cos(1.3*t)], axis=1)
    v, acc = differentiate(t, y, 3)
    independent_v, independent_a = np.empty_like(y), np.empty_like(y)
    for j in range(len(t)):
        k = min(max(j-1, 0), len(t)-3)
        h0, h1 = t[k+1]-t[k], t[k+2]-t[k+1]
        a, b, c = y[k:k+3]
        independent_a[j] = 2*(a/(h0*(h0+h1))-b/(h0*h1)+c/(h1*(h0+h1)))
        if j == 0:
            independent_v[j] = -(2*h0+h1)/(h0*(h0+h1))*a+(h0+h1)/(h0*h1)*b-h0/(h1*(h0+h1))*c
        elif j == len(t)-1:
            independent_v[j] = h1/(h0*(h0+h1))*a-(h0+h1)/(h0*h1)*b+(h0+2*h1)/(h1*(h0+h1))*c
        else:
            independent_v[j] = -h1/(h0*(h0+h1))*a+(h1-h0)/(h0*h1)*b+h0/(h1*(h0+h1))*c
    np.testing.assert_allclose(v, independent_v, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(acc, independent_a, rtol=1e-12, atol=1e-12)
    checks.append({'name': 'official_cfo_nonuniform_formula', 'passed': True})
    # A quintic is reconstructed exactly when its analytic endpoint jets are
    # supplied. Interior derivative correctness is checked, not just endpoints.
    y0 = torch.tensor([[1.], [2.]], dtype=torch.float64)
    h = torch.tensor([.7, 1.3], dtype=torch.float64)
    zeros = torch.zeros_like(y0)
    y1 = y0+h[:, None]**5
    for q in (0., .23, .5, 1.):
        fraction = torch.full((2,), q, dtype=torch.float64)
        u, du = quintic_sample(y0, y1, zeros, 5*h[:, None]**4, zeros, 20*h[:, None]**3, h, fraction)
        torch.testing.assert_close(u, y0+(q*h[:, None])**5, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(du, 5*(q*h[:, None])**4, rtol=1e-12, atol=1e-12)
    checks.append({'name': 'analytic_quintic_path_velocity', 'passed': True})
    initial = torch.tensor([[1., 2.], [3., -1.]], dtype=torch.float64)
    rates = torch.tensor([[-.7, .3], [-.5, -.2]], dtype=torch.float64)
    durations = torch.tensor([1.1, .9], dtype=torch.float64)
    exact = initial*torch.exp(rates*durations[:, None])
    errors = []
    for m in (2, 4, 8, 16):
        out = rk4(lambda u, t: rates*u, initial, torch.zeros(2, dtype=torch.float64), durations, m)
        errors.append(float((out-exact).norm()))
    ratios = [a/b for a, b in zip(errors, errors[1:])]
    assert all(14 < r < 21 for r in ratios), ratios
    checks.append({'name': 'rk4_physical_time_fourth_order', 'passed': True, 'errors': errors, 'refinement_ratios': ratios})
    # Nonautonomous du/dt=t has exact endpoint, exposing omitted or incorrectly
    # shifted intermediate RK4 times.
    out = rk4(lambda u, t: t[:, None].expand_as(u), initial, torch.tensor([2., 4.], dtype=torch.float64), durations, 2)
    exact = initial+(torch.tensor([2., 4.], dtype=torch.float64)*durations+.5*durations**2)[:, None]
    torch.testing.assert_close(out, exact, rtol=1e-12, atol=1e-12)
    checks.append({'name': 'rk4_absolute_time_stages', 'passed': True})
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--benchmark-updates', type=int, default=3)
    a = p.parse_args()
    if not 3 <= a.benchmark_updates <= 30:
        raise ValueError('Preflight allows 3..30 actual continuous updates per arm')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    checks, benchmarks, paired, initial_seeds = [], {}, {}, []
    begun = time.perf_counter()
    try:
        checks.extend(numerical_checks())
        for arm in ARMS:
            args = build_parser().parse_args(['--arm', arm, '--seed', '0', '--mode', 'preflight',
                    '--source-root', str(a.source_root), '--source-manifest', str(a.source_manifest),
                    '--data-dir', str(a.data_dir), '--output-dir', str(a.output_dir/(arm+'-continuous')),
                    '--device', a.device])
            trainer = Trainer(args)
            before = parameter_sha(trainer.model)
            expected_parameters = 3712921 if SPECS[arm]['width'] == 40 else 928461
            assert sum(x.numel()*(2 if x.is_complex() else 1) for x in trainer.model.parameters()) == expected_parameters
            observations = {'forward': 0, 'optimizer': 0}
            fhandle = trainer.model.register_forward_pre_hook(lambda module, *_: observations.__setitem__('forward', observations['forward']+1) if module.training else None)
            ohandle = trainer.optimizer.register_step_post_hook(lambda *_: observations.__setitem__('optimizer', observations['optimizer']+1))
            first = trainer.step()
            assert observations == {'forward': SPECS[arm]['calls'], 'optimizer': 1}
            assert parameter_sha(trainer.model) != before
            trainer.save_checkpoint(trainer.out/'partial_state.pt')
            expected_row = trainer.step()
            expected = snapshot({'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                                 'scheduler': trainer.scheduler.state_dict(), 'rng': trainer.rng_state(),
                                 'cursor': (trainer.epoch, trainer.batch, trainer.updates), 'permutation': trainer.permutation})
            expected_prediction = trainer.model.predict(trainer.data['validation_truth'][:2, ..., 0].copy(),
                                                        trainer.data['validation_dt'][:2, :2].copy(), a.device)
            assert expected_prediction.shape == (2, 64, 64, 1, 3) and np.isfinite(expected_prediction).all()
            paired[arm] = {'initial': before, 'indices': first['indices_sha256'],
                           'collocation': trainer.collocation_sha256, 'sampling': first['sampling_sha256']}
            times = [expected_row['training_seconds']]
            for _ in range(a.benchmark_updates-2):
                times.append(trainer.step()['training_seconds'])
            assert observations == {'forward': a.benchmark_updates*SPECS[arm]['calls'], 'optimizer': a.benchmark_updates}
            fhandle.remove(); ohandle.remove()
            peak = torch.cuda.max_memory_allocated() if a.device == 'cuda' else None
            # Short physical rollout sanity after real optimizer updates.
            pred = trainer.model.predict(trainer.data['validation_truth'][:2, ..., 0].copy(),
                                         trainer.data['validation_dt'][:2, :2].copy(), a.device)
            assert pred.shape == (2, 64, 64, 1, 3) and np.isfinite(pred).all()
            if trainer.flow:
                x = trainer.states[:2, 0].to(trainer.device)
                zero, ten = torch.zeros(2, device=trainer.device), torch.full((2,), 10., device=trainer.device)
                with torch.no_grad():
                    delta = float((trainer.model.field(x, zero)-trainer.model.field(x, ten)).abs().max().cpu())
                if trainer.spec['autonomous']:
                    assert delta == 0.
                else:
                    assert delta > 0.
                checks.append({'name': 'field_time_behavior_'+arm, 'passed': True, 'shift_max_difference': delta})
            benchmarks[arm] = {'continuous_updates': a.benchmark_updates, 'resume_updates': 1,
                               'observed_forward_calls': observations['forward'], 'observed_optimizer_updates': observations['optimizer'],
                               'post_warmup_mean_update_seconds': float(np.mean(times)),
                               'post_warmup_max_update_seconds': max(times), 'cuda_peak_allocated_bytes': peak,
                               'estimated_training_hours': float(np.mean(times)*SPECS[arm]['epochs']*100/3600),
                               'estimate_excludes_validation_checkpoint_and_log_overhead': True}
            checkpoint = trainer.out/'partial_state.pt'
            del trainer; gc.collect()
            if a.device == 'cuda':
                torch.cuda.empty_cache()
            args.output_dir = a.output_dir/(arm+'-reload')
            resumed = Trainer(args)
            resumed.load_checkpoint(checkpoint)
            row = resumed.step()
            actual = {'model': resumed.model.state_dict(), 'optimizer': resumed.optimizer.state_dict(),
                      'scheduler': resumed.scheduler.state_dict(), 'rng': resumed.rng_state(),
                      'cursor': (resumed.epoch, resumed.batch, resumed.updates), 'permutation': resumed.permutation}
            # CPU is bit-exact; CUDA tolerates deterministic-library roundoff.
            compare(actual, expected, rtol=2e-6 if a.device == 'cuda' else 0., atol=2e-7 if a.device == 'cuda' else 0.)
            for key in ('update', 'epoch', 'batch', 'indices_sha256', 'sampling_sha256', 'forward_calls', 'lr_used', 'lr_after'):
                assert row[key] == expected_row[key], key
            for key in ('total', 'fit', 'regularizer'):
                np.testing.assert_allclose(row[key], expected_row[key], rtol=2e-6 if a.device == 'cuda' else 0., atol=2e-7 if a.device == 'cuda' else 0.)
            resumed_prediction = resumed.model.predict(resumed.data['validation_truth'][:2, ..., 0].copy(),
                                                      resumed.data['validation_dt'][:2, :2].copy(), a.device)
            np.testing.assert_allclose(resumed_prediction, expected_prediction,
                                       rtol=2e-6 if a.device == 'cuda' else 0., atol=2e-7 if a.device == 'cuda' else 0.)
            checks.append({'name': 'short_predict_reload_'+arm, 'passed': True,
                           'shape': list(resumed_prediction.shape), 'physical_transitions': 2,
                           'maximum_difference': float(np.max(abs(resumed_prediction-expected_prediction)))})
            if resumed.flow:
                from evaluate_generator import evaluate_solver_refinement
                small = {'validation_truth': resumed.data['validation_truth'][:2, ..., :3],
                         'validation_dt': resumed.data['validation_dt'][:2, :2],
                         'train_states': resumed.data['train_states'][:4, :1]}
                audit = evaluate_solver_refinement(resumed.model, small, resumed.out, a.device)
                assert all(item['status'] == 'FINITE' and item['inference_seconds'] > 0 for item in audit['predictions'].values())
                assert [item['nfe_per_trajectory'] for item in audit['predictions'].values()] == [8, 16, 32, 64]
                if resumed.spec['autonomous']:
                    assert audit['autonomous_field_bit_equal'] is True
                checks.append({'name': 'solver_refinement_smoke_'+arm, 'passed': True,
                               'substeps': [1, 2, 4, 8], 'truth_transitions': 2,
                               'primary_solver_self_difference': audit['prediction_relative_differences']['2_vs_4']})
            resumed.verify_source()
            for path, digest in resumed.input_sha256.items():
                assert sha(path) == digest
            checks.append({'name': 'resume_'+arm, 'passed': True, 'parameters_optimizer_scheduler_rng_compared': True,
                           'same_next_batch_and_sampling': True, 'full_shape_batch20': True})
            del resumed, expected, actual; gc.collect()
            if a.device == 'cuda':
                torch.cuda.empty_cache()
            print(json.dumps({'arm': arm, 'resume': 'PASS', 'benchmark': benchmarks[arm]}), flush=True)
        reference = paired['first20e500']
        for arm in ('teacher20e500', 'rollout20e500', 'pushforward20e500', 'teacher20e1000'):
            assert paired[arm]['initial'] == reference['initial'] and paired[arm]['indices'] == reference['indices'] and paired[arm]['collocation'] == reference['collocation']
        assert paired['teacher40e500']['indices'] == reference['indices'] and paired['teacher40e500']['collocation'] == reference['collocation']
        assert paired['auto20e500']['initial'] == paired['time20e500']['initial'] == reference['initial']
        assert paired['auto20e500']['sampling'] == paired['time20e500']['sampling'] and paired['auto20e500']['indices'] == reference['indices']
        checks.append({'name': 'paired_initialization_batches_labels_and_collocation', 'passed': True})
        # Initialization is explicitly seeded after all imports.
        for seed in (0, 1, 2):
            args.arm, args.seed, args.output_dir = 'first20e500', seed, a.output_dir/f'initialization-seed{seed}'
            tr = Trainer(args); initial_seeds.append(tr.initial_parameter_sha256)
            del tr; gc.collect()
        assert len(set(initial_seeds)) == 3 and initial_seeds[0] == reference['initial']
        checks.extend([{'name': 'distinct_initialization_seeds', 'passed': True},
                       {'name': 'frozen_inputs_unchanged', 'passed': True}])
        report = {'schema': 'next_methods_engineering_validation_v1', 'status': 'PASS', 'device': a.device,
                  'slurm_job_id': __import__('os').environ.get('SLURM_JOB_ID'),
                  'gpu_name': torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
                  'gpu_capability': list(torch.cuda.get_device_capability(0)) if a.device == 'cuda' else None,
                  'source_manifest_sha256': sha(a.source_manifest),
                  'data_payload_sha256': sha(a.data_dir/'two_step_data.npz'),
                  'flow_payload_sha256': sha(a.data_dir.parent/'flow-data-r2/flow_data.npz'),
                  'checks': checks, 'benchmarks': benchmarks, 'paired': paired,
                  'actual_optimizer_updates': len(ARMS)*(a.benchmark_updates+1),
                  'duration_seconds': time.perf_counter()-begun,
                  'payload_manifest_sha256': sha(Path(__file__).parent/'PAYLOAD_MANIFEST.json') if (Path(__file__).parent/'PAYLOAD_MANIFEST.json').exists() else None,
                  'interpretation': 'Engineering and proxy-label checks; no trained-model accuracy claim'}
        with (a.output_dir/'VALIDATION_REPORT.json').open('x') as f:
            json.dump(report, f, indent=2, allow_nan=False)
        inventory = {str(p.relative_to(a.output_dir)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(a.output_dir.rglob('*')) if p.is_file() and 'matplotlib-cache' not in p.parts}
        with (a.output_dir/'artifact_manifest.json').open('x') as f:
            json.dump(inventory, f, indent=2)
        print(json.dumps({'status': 'PASS', 'actual_optimizer_updates': report['actual_optimizer_updates'], 'checks': len(checks)}), flush=True)
    except BaseException as e:
        with (a.output_dir/'VALIDATION_FAILURE.json').open('x') as f:
            json.dump({'status': 'FAILED', 'error': repr(e), 'traceback': traceback.format_exc(), 'completed_checks': checks}, f, indent=2)
        raise


if __name__ == '__main__':
    main()
