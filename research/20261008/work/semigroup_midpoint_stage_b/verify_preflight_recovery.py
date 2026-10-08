#!/usr/bin/env python3
"""Independently inspect recovered tensors and replay four preflight models on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(left, right, rtol=0, atol=0):
    if isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
        a, b = left.detach().cpu().numpy(), right.detach().cpu().numpy()
        assert np.isfinite(a).all() and np.isfinite(b).all()
        if rtol == atol == 0:
            assert np.array_equal(a, b)
        else:
            assert np.all(np.abs(a-b) <= atol + rtol * np.abs(b))
        return 1
    if isinstance(right, np.ndarray):
        assert isinstance(left, np.ndarray) and left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(right, dict):
        assert isinstance(left, dict) and left.keys() == right.keys()
        return sum(compare(left[k], right[k], rtol, atol) for k in right)
    elif isinstance(right, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        return sum(compare(a, b, rtol, atol) for a, b in zip(left, right))
    else:
        assert left == right
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-report', type=Path, required=True)
    p.add_argument('--payload-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    transfer = json.loads(a.recovery_report.read_text())
    assert transfer['status'] == 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED'
    assert transfer['mode'] == 'preflight' and transfer['capture_status'] == 'COMPLETE'
    root, run = Path(transfer['root']), Path(transfer['run_root'])
    root, run = root.resolve(strict=True), run.resolve(strict=True)
    assert run.is_relative_to(root)
    a.output_dir = a.output_dir.resolve()
    assert not a.output_dir.is_relative_to(root) and not a.output_dir.is_relative_to(a.data_dir.resolve())
    a.output_dir.mkdir(parents=True, exist_ok=False)
    payload_sha = sha(a.payload_manifest)
    assert sha(root / 'PAYLOAD_MANIFEST.json') == payload_sha == transfer['payload_manifest_sha256']
    payload = json.loads(a.payload_manifest.read_text())
    protected = {root / 'PAYLOAD_MANIFEST.json': payload_sha}
    for name, item in payload['files'].items():
        path = ((a.data_dir / 'development_data.npz') if name == 'data-r1/development_data.npz' else root / name).resolve(strict=True)
        assert path.is_relative_to(root) or path == (a.data_dir / 'development_data.npz').resolve()
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        protected[path] = item['sha256']
    gate_path = run / 'VALIDATION_REPORT.json'
    gate = json.loads(gate_path.read_text())
    detail = json.loads((run / 'PREFLIGHT_PASSED.json').read_text())
    assert gate['status'] == 'PASS' and gate['device'] == 'cuda' and gate['gpu_capability'] == [7, 0]
    assert 'V100' in gate['gpu_name'] and gate['slurm_job_id'] == transfer['job_id']
    assert gate['payload_manifest_sha256'] == payload_sha and gate['optimizer_updates'] == 13
    assert len(gate['checks']) == 12 and all(c['passed'] for c in gate['checks'])
    assert sha(run / 'PREFLIGHT_PASSED.json') == gate['detailed_report_sha256']
    manifest = json.loads((run / 'artifact_manifest.json').read_text())
    for name, item in manifest.items():
        path = (run / name).resolve(strict=True)
        assert path.is_relative_to(run.resolve()) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        protected[path] = item['sha256']
    with np.load(a.data_dir / 'development_data.npz', allow_pickle=False) as stored:
        data = {k: stored[k].copy() for k in stored.files}
    assert sha(a.data_dir / 'development_data.npz') == gate['data_payload_sha256']
    torch.set_num_threads(4)
    sys.path.insert(0, str(root / 'due'))
    import due
    assert Path(due.__file__).resolve().is_relative_to((root / 'due').resolve())
    evidence = torch.load(run / 'author_update_tensor_evidence.pt', weights_only=False, map_location='cpu')
    oracle_pairs = [('stage_b_gradient', 'author_gradient'), ('stage_b_after', 'author_after'),
                    ('stage_b_Adam', 'author_Adam'), ('stage_b_scheduler', 'author_scheduler')]
    oracle = []
    for left, right in oracle_pairs:
        count = compare(evidence[left], evidence[right], rtol=3e-5, atol=2e-6)
        oracle.append({'comparison': left + '_vs_' + right, 'tensor_count': count, 'passed': True})
    arms = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')
    state_keys = ('config', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'rng_state',
                  'cursor', 'permutation', 'epoch_sums', 'history', 'validation_history', 'forward_calls',
                  'resume_safe', 'failed', 'initial_parameter_sha256', 'collocation_sha256')
    replays = []
    for arm in arms:
        cont_dir, reload_dir = run / (arm + '-continuous'), run / (arm + '-reload')
        cont = torch.load(cont_dir / 'after_update_two.pt', weights_only=False, map_location='cpu')
        reload = torch.load(reload_dir / 'after_reloaded_update_two.pt', weights_only=False, map_location='cpu')
        count = compare({k: cont[k] for k in state_keys}, {k: reload[k] for k in state_keys})
        assert cont['cursor'] == {'epoch': 0, 'batch': 2, 'updates': 2}
        assert cont['config']['arm'] == arm and cont['config']['seed'] == 0
        saved = np.load(cont_dir / 'two_step_original_prediction.npy', allow_pickle=False)
        other = np.load(reload_dir / 'two_step_original_prediction.npy', allow_pickle=False)
        assert saved.shape == (1, 64, 64, 1, 3) and saved.dtype == np.float32
        assert np.isfinite(saved).all() and np.array_equal(saved, other)
        network = due.networks.fno.osg_fno2d(data['vmin'], data['vmax'], data['tmin'][()], data['tmax'][()],
                                            cont['config']['network'], multiscale=False)
        network.load_state_dict(cont['model_state_dict'])
        before = {key: tensor.clone() for key, tensor in network.state_dict().items()}
        prediction = network.predict(data['validation_truth'][:1, ..., 0].copy(),
                                     data['validation_dt'][:1, :2].copy(), device='cpu')
        compare(network.state_dict(), before)
        assert prediction.shape == saved.shape and prediction.dtype == np.float32 and np.isfinite(prediction).all()
        # Backend changes can alter FFT rounding; this is a numerical replay,
        # while the same-device continuous/reloaded arrays above match exactly.
        delta = prediction.astype(np.float64) - saved.astype(np.float64)
        rel = float(np.linalg.norm(delta.ravel()) / np.linalg.norm(saved.astype(np.float64).ravel()))
        assert rel <= 2e-5, (arm, rel)
        np.save(a.output_dir / (arm + '_cpu_replay.npy'), prediction, allow_pickle=False)
        replays.append({'arm': arm, 'complete_state_tensors_bitwise_equal': count,
                        'saved_rollouts_bitwise_equal': True, 'cpu_gpu_prediction_relative_l2': rel,
                        'cpu_gpu_max_abs': float(np.max(np.abs(delta))), 'cpu_gpu_tolerance_relative_l2': 2e-5,
                        'optimizer_updates': 0, 'passed': True})
    for path, digest in protected.items():
        assert sha(path) == digest, 'Replay changed input: ' + str(path)
    checks = [{'name': 'remote_archive_and_exact_inventory', 'passed': True,
               'archive_sha256': transfer['archive_sha256'], 'files': transfer['inventory_files_verified']},
              {'name': 'gpu_validation_report_and_artifact_hashes', 'passed': True, 'checks': len(gate['checks'])},
              {'name': 'independent_author_update_tensor_comparison', 'passed': True, 'comparisons': oracle},
              {'name': 'independent_four_arm_resume_state_and_cpu_replay', 'passed': True, 'arms': replays},
              {'name': 'all_frozen_source_data_and_checkpoint_inputs_unchanged', 'passed': True, 'files': len(protected)}]
    result = {'schema': 'stage_b_local_recovery_gate_v1', 'status': 'PASS',
              'preflight_job_id': transfer['job_id'], 'payload_manifest_sha256': payload_sha,
              'validation_report_sha256': sha(gate_path), 'checks': checks,
              'optimizer_updates': 0, 'replay_transitions': 8, 'verifier_sha256': sha(__file__),
              'scope': 'Integrity, independent checkpoint tensor equality and CPU replay; no long-rollout accuracy conclusion'}
    with (a.output_dir / 'LOCAL_RECOVERY_GATE.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
