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
from train import Trainer, build_parser, parameter_sha, tensor_sha, json_write, sha, CODE_FILES


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
    for name in ('source-root', 'source-manifest', 'data-dir', 'output-dir'):
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
            '--data-dir', str(a.data_dir), '--output-dir', str(a.output_dir / name), '--device', a.device])
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
        for arm in ('first', 'rollout2'):
            trainer = Trainer(args_for(arm, 0, arm + '-continuous'))
            before = parameter_sha(trainer.model)
            counts = {'forward': 0, 'optimizer': 0}
            f_hook = trainer.model.register_forward_pre_hook(lambda *_: counts.__setitem__('forward', counts['forward'] + 1))
            o_hook = trainer.optimizer.register_step_post_hook(lambda *_: counts.__setitem__('optimizer', counts['optimizer'] + 1))
            first = trainer.step()
            assert before != parameter_sha(trainer.model)
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
        assert paired['first'] == paired['rollout2']
        checks.append({'name': 'paired_seed0_initialization_and_batch_order', 'passed': True})
        # All planned seeds must initialize identically across arms and differ
        # between seeds; no additional training is performed in these checks.
        initial_hashes = [paired['first']['initial_parameter_sha256']]
        for seed in (1, 2):
            pair = []
            for arm in ('first', 'rollout2'):
                item = Trainer(args_for(arm, seed, f'{arm}-seed{seed}-initialization'))
                pair.append((parameter_sha(item.model), tensor_sha(item.generators['shuffle'].get_state())))
                del item; gc.collect()
            assert pair[0] == pair[1]
            initial_hashes.append(pair[0][0])
            checks.append({'name': f'paired_seed{seed}_initialization_and_shuffle', 'passed': True})
        assert len(set(initial_hashes)) == 3
        checks.append({'name': 'distinct_seed_initializations', 'passed': True})
        report = {'schema': 'autoregressive_engineering_preflight_v1', 'status': 'PASS', 'device': a.device,
            'gpu_name': torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
            'gpu_capability': list(torch.cuda.get_device_capability(0)) if a.device == 'cuda' else None,
            'torch': torch.__version__, 'numpy': np.__version__, 'slurm_job_id': __import__('os').environ.get('SLURM_JOB_ID'),
            'checks': checks, 'benchmarks': benchmarks, 'paired': paired,
            'actual_optimizer_updates': 2 * (a.benchmark_updates + 1),
            'source_manifest_sha256': sha(a.source_manifest),
            'data_payload_sha256': sha(a.data_dir / 'two_step_data.npz'),
            'code_sha256': {name: sha(Path(__file__).with_name(name)) for name in CODE_FILES},
            'whole_process_wall_seconds': time.perf_counter() - started,
            'scope': 'Engineering preflight: bounded optimizer updates and short 8-transition rollouts. No full50 untrained-model accuracy gate and no method-effect claim.'}
        json_write(a.output_dir / 'PREFLIGHT_REPORT.json', report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        json_write(a.output_dir / 'PREFLIGHT_FAILURE.json', {'status': 'FAILED', 'error': repr(error),
            'traceback': traceback.format_exc(), 'checks_completed': checks})
        raise


if __name__ == '__main__':
    main()
