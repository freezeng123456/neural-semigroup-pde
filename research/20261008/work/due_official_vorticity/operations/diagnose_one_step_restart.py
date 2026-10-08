#!/usr/bin/env python3
"""Post-hoc baseline diagnosis: restart original predict from every true state.

No optimization or architecture changes. This distinguishes local prediction
error along the test trajectory from accumulated free-rollout error; it does
not identify the causal mechanism or provide a new held-out model selection set.
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

from replay_official_cpu import read_inputs, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['source-root', 'source-manifest', 'run-root', 'test-mat', 'verification', 'output-dir']:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    out = a.output_dir.resolve()
    if out.is_relative_to(a.run_root.resolve()) or out.is_relative_to(a.source_root.resolve()):
        raise ValueError('Diagnostic outputs must be outside immutable inputs')
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source, run, env, receipt, checkpoint, prediction_file, hashes, sg0 = read_inputs(a)
    verified = json.loads(a.verification.read_text())
    assert verified['status'] == 'verified' and verified['engineering']['status'] == 'passed'
    assert Path(verified['run_root']).resolve() == run
    hashes[str(a.verification.resolve())] = sha(a.verification)
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
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    raw = loadmat(a.test_mat, variable_names=['trajectories', 'dt'])
    truth = raw['trajectories'].astype('single')
    dt = raw['dt'].astype('single')
    del raw
    assert truth.shape == (100, 64, 64, 1, 100) and np.all(dt == 1)
    model = torch.load(checkpoint, map_location='cpu', weights_only=False).to('cpu')
    assert type(model) is due.networks.fno.osg_fno2d
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    per_time = np.empty((100, 99), dtype=np.float64)
    first_prediction = None
    for k in range(99):
        # Original predict handles normalization and time scaling; each call
        # starts from truth, never the preceding predicted state.
        with contextlib.redirect_stdout(io.StringIO()):
            predicted = model.predict(truth[..., k].copy(), dt[:, k:k+1].copy(), device='cpu')[..., 1]
        if not np.isfinite(predicted).all():
            np.save(out / f'nonfinite_prediction_transition_{k}.npy', predicted)
            raise FloatingPointError(f'Nonfinite local prediction at transition {k}')
        if k == 0:
            first_prediction = predicted.copy()
        target = truth[..., k + 1].astype(np.float64).reshape(100, -1)
        delta = predicted.astype(np.float64).reshape(100, -1) - target
        denominator = np.linalg.norm(target, axis=1)
        assert np.all(denominator > 0)
        per_time[:, k] = np.linalg.norm(delta, axis=1) / denominator
        if (k + 1) % 10 == 0 or k == 98:
            print(f'Completed true-state restarts: {k + 1}/99, 100 trajectories each', flush=True)
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    # Same physical units, float64 aggregation; official free rollout is GPU,
    # restarts are CPU. The first-step cross-device difference is measured below.
    free = np.empty_like(per_time)
    # Restarts and free rollout both use the same official float32 truth with
    # float64 accumulation; do not mix in the raw-MAT float64 truth variant.
    gpu = loadmat(prediction_file, variable_names=['trajectories'])['trajectories']
    for k in range(99):
        target = truth[..., k+1].astype(np.float64).reshape(100, -1)
        free[:, k] = np.linalg.norm(gpu[..., k+1].astype(np.float64).reshape(100, -1) - target, axis=1) / np.linalg.norm(target, axis=1)
    step1_delta = first_prediction.astype(np.float64) - gpu[..., 1].astype(np.float64)
    step1_parity = {
        'max_abs': float(np.max(np.abs(step1_delta))),
        'relative_rms_to_gpu': float(np.linalg.norm(step1_delta.ravel()) / np.linalg.norm(gpu[..., 1].astype(np.float64).ravel())),
    }
    rows = []
    for k in range(99):
        rows.append({'elapsed_target_time': k+1,
                     'restart_mean': float(per_time[:, k].mean()),
                     'restart_median': float(np.median(per_time[:, k])),
                     'restart_p90': float(np.quantile(per_time[:, k], .9)),
                     'free_rollout_mean': float(free[:, k].mean()),
                     'free_rollout_median': float(np.median(free[:, k]))})
    np.savez_compressed(out / 'per_trajectory_metrics.npz', restart_relative_l2=per_time, free_rollout_relative_l2=free)
    with (out / 'error_by_time.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    unchanged = all(sha(Path(path)) == digest for path, digest in hashes.items())
    assert unchanged
    report = {
        'status': 'TRUE_STATE_RESTART_DIAGNOSTIC_COMPLETE',
        'variant': 'sg0' if sg0 else 'default_gdsg',
        'source_commit': env['source_commit'], 'slurm_training_job': env['slurm_job_id'],
        'model': 'final epoch500', 'device': 'cpu', 'batch_size': 100,
        'test_trajectories': 100, 'transitions_per_trajectory': 99,
        'total_one_step_predictions': 9900, 'optimizer_updates': 0,
        'parameters_unchanged': True, 'inputs_unchanged': unchanged,
        'metric': 'Per-trajectory spatial relative L2 in physical units. Float32 truth/model outputs, float64 subtraction and norm accumulation. Same metric for both curves; not bitwise official float32 CSV.',
        'first_step_cpu_gpu_parity': step1_parity,
        'restart_all_time_mean': float(per_time.mean()),
        'restart_max': float(per_time.max()),
        'selected_times': [rows[k-1] for k in [1,20,40,60,80,99]],
        'wall_seconds': time.perf_counter()-start,
        'input_sha256': hashes, 'script_sha256': sha(Path(__file__)),
        'runtime': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version},
        'limitations': [
            'Post-hoc diagnosis on an already inspected benchmark test set; not a new confirmation set.',
            'True-state restart measures local fit on real states; free rollout includes accumulated error and changed input states.',
            'Their difference does not separately identify capacity, optimization, data coverage, or dynamical sensitivity.',
            'Cross-device numerical differences remain; first-step parity is measured, not assumed.',
        ],
    }
    write_json(out / 'RESTART_DIAGNOSTIC.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
