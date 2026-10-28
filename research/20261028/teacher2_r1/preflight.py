#!/usr/bin/env python3
"""Bounded DUE optimizer, exact next-update resume, paired RNG and reload audit."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from train import Trainer, build_parser, parameter_sha, tensor_sha, json_write, sha, CODE_FILES, load_reference_records


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(snapshot(v) for v in value)
    return copy.deepcopy(value)


def equal_tree(a, b):
    if isinstance(b, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0., atol=0.)
    elif isinstance(b, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for k in b:
            equal_tree(a[k], b[k])
    elif isinstance(b, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b, (a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'source-manifest', 'data-dir', 'output-dir', 'reference-records'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--benchmark-updates', type=int, default=5)
    a = p.parse_args()
    if not 3 <= a.benchmark_updates <= 30:
        raise ValueError('Expected 3..30 benchmark updates per arm')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    checks, benchmarks, paired = [], {}, {}
    started = time.perf_counter()
    def args_for(arm, seed, name):
        return build_parser().parse_args(['--arm', arm, '--seed', str(seed), '--mode', 'preflight',
            '--source-root', str(a.source_root), '--source-manifest', str(a.source_manifest),
            '--data-dir', str(a.data_dir), '--reference-records', str(a.reference_records),
            '--output-dir', str(a.output_dir / name), '--device', a.device])
    def state_for(trainer):
        return snapshot({'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
            'scheduler': trainer.scheduler.state_dict(), 'rng': trainer.rng_state(),
            'cursor': (trainer.epoch, trainer.batch, trainer.updates), 'permutation': trainer.permutation,
            'forward_calls': trainer.forward_calls, 'epoch_sums': trainer.epoch_sums})
    try:
        # Toy objective tests also run here so the GPU gate covers the exact
        # delivered objective source before touching full-budget runs.
        import unittest
        from test_objectives import Objectives
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(Objectives).run(result)
        assert result.wasSuccessful(), str(result.errors + result.failures)
        checks.append({'name': 'independent_objective_values_gradients_calls', 'passed': True, 'tests': result.testsRun})
        for arm in ('teacher2',):
            trainer = Trainer(args_for(arm, 0, arm + '-continuous'))
            before = parameter_sha(trainer.model)
            counts = {'forward': 0, 'optimizer': 0}
            first_inputs = []
            def observe_forward(_model, inputs):
                counts['forward'] += 1
                if len(first_inputs) < 2:
                    first_inputs.append(inputs[0].detach().cpu().clone())
            f_hook = trainer.model.register_forward_pre_hook(observe_forward)
            o_hook = trainer.optimizer.register_step_post_hook(lambda *_: counts.__setitem__('optimizer', counts['optimizer'] + 1))
            first = trainer.step()
            ids = trainer.permutation[:20]
            ti, starts = trainer.trajectory[ids], trainer.starts[ids]
            expected_z0, expected_z1 = trainer.states[ti, starts], trainer.states[ti, starts + 1]
            assert len(first_inputs) == 2
            torch.testing.assert_close(first_inputs[0][..., :1], expected_z0, rtol=0, atol=0)
            torch.testing.assert_close(first_inputs[1][..., :1], expected_z1, rtol=0, atol=0)
            checks.append({'name': 'teacher2_truth_second_input_on_allocated_device', 'passed': True,
                'first_input_sha256': tensor_sha(first_inputs[0][..., :1]),
                'second_truth_input_sha256': tensor_sha(first_inputs[1][..., :1]), 'device': a.device})
            assert before != parameter_sha(trainer.model)
            assert tensor_sha(trainer.permutation) == trainer.reference['permutation_sha256'][0]
            trainer.save_checkpoint(trainer.out / 'partial_state.pt')
            second = trainer.step()
            expected = state_for(trainer)
            paired[arm] = {'initial_parameter_sha256': before, 'first_indices_sha256': first['indices_sha256'],
                           'second_indices_sha256': second['indices_sha256']}
            times = [second['training_seconds']]
            for _ in range(a.benchmark_updates - 2):
                times.append(trainer.step()['training_seconds'])
            expected_counts = {'forward': a.benchmark_updates * (1 if arm == 'first' else 2), 'optimizer': a.benchmark_updates}
            assert counts == expected_counts, counts
            f_hook.remove(); o_hook.remove()
            peak = torch.cuda.max_memory_allocated() if a.device == 'cuda' else None
            pred = trainer.predict(trajectories=2, steps=8)
            assert pred.shape == (2, 64, 64, 1, 9) and pred.dtype == np.float32 and np.isfinite(pred).all()
            trainer.save_checkpoint(trainer.out / 'reload_state.pt')
            checkpoint = trainer.out / 'partial_state.pt'; reload_path = trainer.out / 'reload_state.pt'
            benchmarks[arm] = {'continuous_updates': a.benchmark_updates, 'observed_training_forwards': counts['forward'],
                'observed_optimizer_updates': counts['optimizer'], 'post_warmup_mean_update_seconds': float(np.mean(times)),
                'estimated_training_hours': float(np.mean(times) * 50000 / 3600), 'cuda_peak_allocated_bytes': peak,
                'checkpoint_sha256': sha(reload_path), 'short_rollout_shape': list(pred.shape)}
            del trainer; gc.collect()
            if a.device == 'cuda':
                torch.cuda.empty_cache()
            resumed = Trainer(args_for(arm, 0, arm + '-resumed'))
            resumed.load_checkpoint(checkpoint); got = resumed.step()
            equal_tree(state_for(resumed), expected)
            for k in ('update', 'epoch', 'batch', 'total', 'one_step', 'second_step', 'lr_used', 'lr_after', 'indices_sha256', 'training_forward_calls'):
                assert got[k] == second[k], (k, got[k], second[k])
            checks.append({'name': 'exact_next_optimizer_update_' + arm, 'passed': True})
            del resumed; gc.collect()
            if a.device == 'cuda':
                torch.cuda.empty_cache()
            reloaded = Trainer(args_for(arm, 0, arm + '-reloaded'))
            reloaded.load_checkpoint(reload_path)
            np.testing.assert_array_equal(reloaded.predict(trajectories=2, steps=8), pred)
            reloaded.verify_source()
            assert all(sha(path) == digest for path, digest in reloaded.input_sha256.items())
            checks.append({'name': 'short_rollout_reload_and_frozen_inputs_' + arm, 'passed': True})
            del reloaded; gc.collect()
            if a.device == 'cuda':
                torch.cuda.empty_cache()
        reference_pairs, reference_inputs = load_reference_records(a.reference_records)
        expected_permutation = torch.randperm(2000, generator=torch.Generator().manual_seed(30000))
        assert paired['teacher2']['initial_parameter_sha256'] == reference_pairs[0]['initial_parameter_sha256']
        assert paired['teacher2']['first_indices_sha256'] == tensor_sha(expected_permutation[:20])
        assert paired['teacher2']['second_indices_sha256'] == tensor_sha(expected_permutation[20:40])
        seed0_generator = torch.Generator().manual_seed(30000)
        assert [tensor_sha(torch.randperm(2000, generator=seed0_generator)) for _ in range(500)] == reference_pairs[0]['permutation_sha256']
        checks.append({'name': 'paired_seed0_historical_initialization_and_batch_order', 'passed': True})
        # All planned seeds must initialize identically across arms and differ
        # between seeds; no additional training is performed in these checks.
        initial_hashes = [paired['teacher2']['initial_parameter_sha256']]
        for seed in (1, 2):
            item = Trainer(args_for('teacher2', seed, f'teacher2-seed{seed}-initialization'))
            before = parameter_sha(item.model)
            assert before == reference_pairs[seed]['initial_parameter_sha256']
            clone = torch.Generator(); clone.set_state(item.generators['shuffle'].get_state())
            expected = [tensor_sha(torch.randperm(2000, generator=clone)) for _ in range(500)]
            assert expected == reference_pairs[seed]['permutation_sha256']
            initial_hashes.append(before)
            paired[f'seed{seed}'] = {'initial_parameter_sha256': before, 'full_500_shuffle_sequence_matched': True}
            checks.append({'name': f'paired_seed{seed}_historical_initialization_and_shuffle', 'passed': True})
            del item; gc.collect()
            if a.device == 'cuda':
                torch.cuda.empty_cache()
        assert len(set(initial_hashes)) == 3
        checks.append({'name': 'distinct_seed_initializations', 'passed': True})
        assert all(sha(path) == digest for path, digest in reference_inputs.items())
        checks.append({'name': 'historical_fixed_endpoints_and_source_hashes', 'passed': True, 'historical_runs': 6,
            'reference_input_sha256': reference_inputs})
        report = {'schema': 'teacher2_engineering_preflight_v1', 'status': 'PASS', 'device': a.device,
            'gpu_name': torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if a.device == 'cuda' else None,
            'torch': torch.__version__, 'numpy': np.__version__, 'slurm_job_id': __import__('os').environ.get('SLURM_JOB_ID'),
            'checks': checks, 'benchmarks': benchmarks, 'paired': paired,
            'actual_optimizer_updates': a.benchmark_updates + 1,
            'source_manifest_sha256': sha(a.source_manifest),
            'data_payload_sha256': sha(a.data_dir / 'two_step_data.npz'),
            'code_sha256': {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES},
            'whole_process_wall_seconds': time.perf_counter() - started,
            'scope': 'Teacher2-only bounded optimization and exact resume/reload, true second-input observation, historical initialization/config/input/500-shuffle audit. No baseline retraining and no method-effect claim.'}
        json_write(a.output_dir / 'PREFLIGHT_REPORT.json', report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        json_write(a.output_dir / 'PREFLIGHT_FAILURE.json', {'status': 'FAILED', 'error': repr(error),
            'traceback': traceback.format_exc(), 'checks_completed': checks})
        raise


if __name__ == '__main__':
    main()
