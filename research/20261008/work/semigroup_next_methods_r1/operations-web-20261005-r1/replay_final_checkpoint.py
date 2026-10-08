#!/usr/bin/env python3
"""Replay an archived scheduled final on CPU, without optimizer updates."""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import torch
from capture_completed_task import digest


def metrics(pred, truth):
    delta = pred.astype(np.float64) - truth.astype(np.float64)
    reference = truth.astype(np.float64)
    return (np.linalg.norm(delta.reshape(20, -1), axis=1) / np.linalg.norm(reference.reshape(20, -1), axis=1),
            np.linalg.norm(delta.reshape(20, -1, 51), axis=1) / np.linalg.norm(reference.reshape(20, -1, 51), axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    gate = json.loads(args.gate.read_text())
    assert gate['status'] == 'PASS'
    root = Path(gate['recovered_root']).resolve(strict=True)
    run = root / 'runs' / gate['run_name']
    assert torch.__version__.split('+')[0] == '2.5.1' and np.__version__ == '1.26.4'
    manifest = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text())
    protected = {args.gate: digest(args.gate), root / 'PAYLOAD_MANIFEST.json': gate['payload_manifest_sha256']}
    assert digest(args.data) == manifest['files']['data-r2/two_step_data.npz']['sha256']
    protected[args.data] = digest(args.data)
    for name, item in manifest['files'].items():
        if not name.endswith('.npz'):
            protected[root / name] = item['sha256']
    artifacts = json.loads((run / 'artifact_manifest.json').read_text())
    protected[run / 'artifact_manifest.json'] = gate['run_manifest_sha256']
    for name in ('final_state.pt', 'latest_state.pt', 'final_weights.pt', 'validation_final_prediction.npy',
                 'validation_final_metrics.npz', 'RUN_RECEIPT.json', 'configuration.json'):
        protected[run / name] = artifacts[name]['sha256']
    for path, sha in protected.items():
        assert digest(path) == sha, str(path)
    with np.load(args.data, allow_pickle=False) as data:
        truth = data['validation_truth'].copy()
        dt = data['validation_dt'].copy()
        vmin, vmax = data['vmin'].copy(), data['vmax'].copy()
        tmin, tmax = data['tmin'][()], data['tmax'][()]
    assert truth.shape == (20, 64, 64, 1, 51) and truth.dtype == np.float32
    gpu = np.load(run / 'validation_final_prediction.npy', allow_pickle=False)
    assert gpu.shape == truth.shape and gpu.dtype == np.float32 and np.isfinite(gpu).all()
    receipt = json.loads((run / 'RUN_RECEIPT.json').read_text())
    gpu_joint, gpu_by_time = metrics(gpu, truth)
    with np.load(run / 'validation_final_metrics.npz', allow_pickle=False) as stored:
        assert np.allclose(gpu_joint, stored['joint_relative_l2'], rtol=1e-12, atol=1e-14)
        assert np.allclose(gpu_by_time, stored['relative_l2_by_snapshot'], rtol=1e-12, atol=1e-14)
    assert np.allclose(gpu_joint, receipt['final_validation']['joint_per_trajectory'], rtol=1e-12, atol=1e-14)
    assert abs(gpu_joint.mean() - receipt['final_validation']['joint_mean']) <= 1e-12
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'due'))
    import due
    assert Path(due.__file__).resolve().is_relative_to(root / 'due')
    from train import parameter_sha
    state = torch.load(run / 'final_state.pt', map_location='cpu', weights_only=False)
    latest = torch.load(run / 'latest_state.pt', map_location='cpu', weights_only=False)
    weights = torch.load(run / 'final_weights.pt', map_location='cpu', weights_only=True)
    config = receipt['config']
    assert state['schema'] == 'next_methods_training_state_v1' and state['config'] == config
    assert state['origin_device'] == 'cuda' and not state['failed'] and state['resume_safe']
    expected_cursor = {'epoch': gate['epochs'], 'batch': 0, 'updates': gate['actual_updates']}
    assert state['cursor'] == latest['cursor'] == expected_cursor
    assert state['forward_calls'] == gate['forward_calls']
    assert state['history'] == receipt['history'] and state['validation_history'] == receipt['validation_history']
    assert state['scheduler_state_dict']['last_epoch'] == gate['actual_updates']
    assert state['scheduler_state_dict']['T_max'] == gate['actual_updates']
    assert all(group['lr'] == 0 for group in state['optimizer_state_dict']['param_groups'])
    assert set(weights) == set(state['model_state_dict']) == set(latest['model_state_dict'])
    assert all(torch.equal(v, weights[k]) and torch.equal(v, latest['model_state_dict'][k])
               and torch.isfinite(v).all() for k, v in state['model_state_dict'].items())
    if config['spec']['family'] == 'flow':
        from generator_model import GeneratorFNO
        net = GeneratorFNO(vmin, vmax, config['network'], autonomous=config['spec']['autonomous'])
    else:
        net = due.networks.fno.osg_fno2d(vmin, vmax, tmin, tmax, config['network'], multiscale=False)
    net.load_state_dict(state['model_state_dict'])
    net.eval()
    before = parameter_sha(net)
    assert before == receipt['final_parameter_sha256']
    real_parameters = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())
    assert real_parameters == gate['parameters_real']
    args.output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({'status': 'CPU_FINAL_REPLAY_STARTED', 'run_name': gate['run_name'],
                      'optimizer_updates': 0, 'trajectories': 20, 'positive_steps': 50}), flush=True)
    with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
        prediction = net.predict(truth[..., 0].copy(), dt.copy(), device='cpu')
    assert prediction.shape == truth.shape and prediction.dtype == np.float32 and np.isfinite(prediction).all()
    assert parameter_sha(net) == before
    with (args.output / 'CPU_REPLAY_FINAL_PREDICTION.npy').open('xb') as f:
        np.save(f, prediction, allow_pickle=False)
    delta = prediction.astype(np.float64) - gpu.astype(np.float64)
    relative = np.linalg.norm(delta.reshape(20, -1), axis=1) / np.linalg.norm(gpu.astype(np.float64).reshape(20, -1), axis=1)
    first = np.linalg.norm(delta[..., 1].reshape(20, -1), axis=1) / np.linalg.norm(gpu[..., 1].astype(np.float64).reshape(20, -1), axis=1)
    cpu_joint, cpu_by_time = metrics(prediction, truth)
    # Cross-device float32 comparisons, fixed before this replay. These are
    # engineering tolerances; they do not establish PDE accuracy or exact resume.
    checks = {'first_step_max_relative_l2': float(first.max()) <= 2e-5,
              'full_rollout_max_relative_l2': float(relative.max()) <= 1e-3,
              'development_joint_mean_difference': abs(float(cpu_joint.mean() - gpu_joint.mean())) <= 1e-3}
    for path, sha in protected.items():
        assert digest(path) == sha, str(path)
    result = {
        'schema': 'next_methods_cpu_final_replay_v1',
        'status': 'PASS' if all(checks.values()) else 'CPU_CUDA_COMPARISON_NEEDS_REVIEW',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_name': gate['run_name'],
        'job': gate['job'], 'optimizer_updates': 0, 'checkpoint_cursor': state['cursor'],
        'parameters_real': real_parameters, 'checkpoint_parameter_sha256': before,
        'gpu_saved_metrics_recomputed': True,
        'cpu_joint_mean': float(cpu_joint.mean()), 'gpu_joint_mean': float(gpu_joint.mean()),
        'cpu_final_step_mean': float(cpu_by_time[:, -1].mean()),
        'gpu_final_step_mean': float(gpu_by_time[:, -1].mean()),
        'first_step_max_relative_l2_to_gpu': float(first.max()),
        'full_rollout_max_relative_l2_to_gpu': float(relative.max()),
        'max_absolute_prediction_difference': float(np.abs(delta).max()),
        'engineering_tolerances': {'first_step_relative_l2': 2e-5, 'full_rollout_relative_l2': 1e-3,
                                  'development_joint_mean_difference': 1e-3},
        'checks': checks, 'protected_inputs_unchanged': len(protected),
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                    'device': 'cpu', 'threads': torch.get_num_threads()},
        'elapsed_seconds': time.monotonic() - started,
        'scope': 'Inference and saved-metric verification; no training and no claim of identical CPU/CUDA continuation',
        'script_sha256': digest(Path(__file__)),
    }
    with (args.output / 'CHECKPOINT_REPLAY_REPORT.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    files = {p.name: {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in args.output.iterdir() if p.is_file()}
    with (args.output / 'artifact_manifest.json').open('x') as f:
        json.dump(files, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
