#!/usr/bin/env python3
"""Verify one recovered final checkpoint and replay two development steps on CPU.

No optimizer update, selection or hyperparameter change. Requires the frozen
archive verifier's COMPLETE receipt; output is exclusive and outside inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), 'Non-finite checkpoint tensor'
        return 1
    if isinstance(value, dict):
        return sum(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(finite_tensors(v) for v in value)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-report', type=Path, required=True)
    parser.add_argument('--payload-manifest', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    transfer = json.loads(args.recovery_report.read_text())
    assert transfer['status'] == 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED'
    assert transfer['mode'] == 'run' and transfer['capture_status'] == 'COMPLETE'
    root = Path(transfer['root']).resolve(strict=True)
    run = Path(transfer['run_root']).resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    out = args.output_dir.resolve()
    assert run.is_relative_to(root)
    assert not out.is_relative_to(root) and not out.is_relative_to(data_dir)
    assert not root.is_relative_to(out) and not data_dir.is_relative_to(out)
    assert not out.exists()

    manifest_sha = sha(args.payload_manifest)
    assert sha(root / 'PAYLOAD_MANIFEST.json') == manifest_sha == transfer['payload_manifest_sha256']
    manifest = json.loads(args.payload_manifest.read_text())
    protected = {root / 'PAYLOAD_MANIFEST.json': manifest_sha}
    for name, item in manifest['files'].items():
        path = (data_dir / 'development_data.npz') if name == 'data-r1/development_data.npz' else root / name
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
        protected[path] = item['sha256']
    artifact_path = run / 'artifact_manifest.json'
    protected[artifact_path] = sha(artifact_path)
    for name, item in json.loads(artifact_path.read_text()).items():
        path = (run / name).resolve(strict=True)
        assert path.is_relative_to(run) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
        protected[path] = item['sha256']
    receipt = json.loads((run / 'RUN_RECEIPT.json').read_text())
    assert receipt['status'] == 'STAGE_B_FULL_COMPLETE'
    config = receipt['config']
    arm, seed = config['arm'], config['seed']
    assert arm in ('gdsg500', 'single500', 'midpoint500', 'gdsg333') and seed in (0, 1, 2)
    assert run.name == f'{arm}-seed{seed}-r1'
    epochs = 333 if arm == 'gdsg333' else 500
    calls = 6 if arm.startswith('gdsg') else 4
    assert receipt['actual_updates'] == epochs * 100 and receipt['forward_calls'] == epochs * 100 * calls
    assert receipt['cursor'] == {'epoch': epochs, 'batch': 0}
    assert receipt['parameters_changed'] and receipt['initial_parameter_sha256'] != receipt['final_parameter_sha256']
    assert receipt['final_learning_rate'] == 0.0
    assert receipt['final_validation']['scheduled_final'] and receipt['final_validation']['valid_metric']

    with np.load(data_dir / 'development_data.npz', allow_pickle=False) as stored:
        data = {key: stored[key].copy() for key in stored.files}
    sys.path.insert(0, str(root / 'due'))
    import due
    assert Path(due.__file__).resolve().is_relative_to(root / 'due')
    torch.set_num_threads(4)
    state = torch.load(run / 'final_state.pt', map_location='cpu', weights_only=False)
    network_saved = torch.load(run / 'final_network.pt', map_location='cpu', weights_only=False)
    assert state['config'] == config
    assert state['cursor'] == {'epoch': epochs, 'batch': 0, 'updates': epochs * 100}
    assert state['resume_safe'] and not state['failed']
    assert state['forward_calls'] == epochs * 100 * calls
    tensor_count = finite_tensors(state)
    adam = state['optimizer_state_dict']
    assert adam['state'] and all(float(slot['step'].item()) == epochs * 100 for slot in adam['state'].values())
    assert adam['param_groups'] and all(group['lr'] == 0.0 for group in adam['param_groups'])
    scheduler = state['scheduler_state_dict']
    assert scheduler['T_max'] == scheduler['last_epoch'] == epochs * 100
    state_dict = state['model_state_dict']
    saved_dict = network_saved.state_dict()
    assert state_dict.keys() == saved_dict.keys()
    assert all(torch.equal(value, saved_dict[key]) for key, value in state_dict.items())
    network = due.networks.fno.osg_fno2d(data['vmin'], data['vmax'], data['tmin'][()], data['tmax'][()], config['network'], multiscale=False)
    network.load_state_dict(state_dict)

    original_prediction = np.load(run / 'validation_final_prediction.npy', allow_pickle=False)
    truth = data['validation_truth']
    assert original_prediction.shape == truth.shape == (20, 64, 64, 1, 51)
    assert original_prediction.dtype == truth.dtype == np.float32 and np.isfinite(original_prediction).all()
    diff = original_prediction.astype(np.float64) - truth.astype(np.float64)
    truth64 = truth.astype(np.float64)
    joint = np.linalg.norm(diff.reshape(20, -1), axis=1) / np.linalg.norm(truth64.reshape(20, -1), axis=1)
    by_snapshot = np.linalg.norm(diff.reshape(20, -1, 51), axis=1) / np.linalg.norm(truth64.reshape(20, -1, 51), axis=1)
    assert np.isfinite(joint).all() and np.isfinite(by_snapshot).all()
    with np.load(run / 'validation_final_metrics.npz', allow_pickle=False) as stored:
        assert np.allclose(joint, stored['joint_relative_l2'], rtol=1e-10, atol=1e-12)
        assert np.allclose(by_snapshot, stored['relative_l2_by_snapshot'], rtol=1e-10, atol=1e-12)
        assert np.array_equal(stored['validation_trajectory_ids'], data['validation_trajectory_ids'])
    assert np.isclose(joint.mean(), receipt['final_validation']['joint_mean'], rtol=1e-10, atol=1e-12)
    before = {key: value.clone() for key, value in network.state_dict().items()}
    prediction = network.predict(truth[..., 0].copy(), data['validation_dt'][:, :2].copy(), device='cpu')
    assert all(torch.equal(value, network.state_dict()[key]) for key, value in before.items())
    assert prediction.shape == (20, 64, 64, 1, 3) and prediction.dtype == np.float32 and np.isfinite(prediction).all()
    reference = original_prediction[..., :3].astype(np.float64)
    difference = prediction.astype(np.float64) - reference
    per_trajectory = np.linalg.norm(difference.reshape(20, -1), axis=1) / np.linalg.norm(reference.reshape(20, -1), axis=1)
    assert np.isfinite(per_trajectory).all() and (per_trajectory <= 2e-5).all(), per_trajectory
    for path, digest in protected.items():
        assert sha(path) == digest, 'Replay changed input: ' + str(path)
    out.mkdir(parents=True, exist_ok=False)
    np.save(out / 'development_first_two_steps_cpu.npy', prediction, allow_pickle=False)
    result = {'schema': 'stage_b_final_cpu_replay_v1', 'status': 'PASS',
              'job_id': transfer['job_id'], 'allocation_job_id_raw': transfer['allocation_job_id_raw'],
              'arm': arm, 'seed': seed, 'archive_sha256': transfer['archive_sha256'],
              'payload_manifest_sha256': manifest_sha, 'checkpoint_sha256': sha(run / 'final_state.pt'),
              'checkpoint_tensor_count_finite': tensor_count, 'final_network_state_exactly_equal': True,
              'all_adam_steps_and_scheduler_endpoint_verified': True,
              'actual_training_updates_verified': epochs * 100, 'actual_training_forwards_verified': epochs * 100 * calls,
              'development_joint_mean_recomputed': float(joint.mean()), 'development_metrics_recomputed': True,
              'cpu_gpu_relative_l2_by_trajectory': per_trajectory.tolist(),
              'cpu_gpu_relative_l2_tolerance': 2e-5, 'cpu_gpu_max_abs': float(np.max(np.abs(difference))),
              'replay_batch_size': 20, 'replay_steps_per_trajectory': 2, 'optimizer_updates_by_verifier': 0,
              'protected_inputs_unchanged': len(protected), 'verifier_sha256': sha(__file__),
              'cpu_replay_sha256': sha(out / 'development_first_two_steps_cpu.npy'),
              'scope': 'Recovered final-state integrity, full development-metric arithmetic and a two-step cross-backend replay; does not independently rerun the complete trajectory.'}
    with (out / 'FINAL_REPLAY_REPORT.json').open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False); handle.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
