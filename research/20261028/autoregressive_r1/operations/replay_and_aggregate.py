#!/usr/bin/env python3
"""Replay the six frozen endpoints and aggregate compact development evidence."""
import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

DATA_SHA = '49869507406029ce4263dda51150726589593ecbe4ff8bbe59da956aedb2023a'
SOURCE_SHA = 'c423cbc0bde6636fd07f157b50a2f01d0430ac06973f5f0dfc2dbec1c483a79c'
COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def finite_tensors(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tensors(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(v) for v in value)
    return True


def parameter_sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def safe_member(root, name):
    p = root / name
    if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
        raise ValueError('Unsafe artifact member: ' + name)
    return p


def verify_inventory(run):
    inventory = json.loads((run / 'artifact_manifest.json').read_text())
    actual = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts}
    if actual != set(inventory) | {'artifact_manifest.json'}:
        raise ValueError('Run inventory set differs from actual files')
    for name, entry in inventory.items():
        p = safe_member(run, name)
        if p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('Run artifact mismatch: ' + name)
    return sha(run / 'artifact_manifest.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    a = p.parse_args()
    root = a.root.resolve(strict=True); output = a.output_dir.resolve()
    if output == root or output.is_relative_to(root / 'runs'):
        raise ValueError('Analysis output must be separate from immutable runs')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    matrix = json.loads((root / 'matrix.json').read_text())
    expected_tasks = [(arm, seed, f'{arm}-seed{seed}') for seed in (0, 1, 2) for arm in ('first', 'rollout2')]
    if [(t['arm'], t['seed'], t['run_name']) for t in matrix['tasks']] != expected_tasks or len(matrix['tasks']) != 6:
        raise ValueError('Expected exactly six frozen tasks')
    manifest = json.loads(a.source_manifest.read_text())
    if sha(a.source_manifest) != SOURCE_SHA or manifest['commit'] != COMMIT or len(manifest['files']) != 75:
        raise ValueError('Wrong source identity')
    for name, digest in manifest['files'].items():
        if sha(a.source_root / name) != digest:
            raise ValueError('Upstream source mismatch: ' + name)
    if sha(a.data_dir / 'two_step_data.npz') != DATA_SHA:
        raise ValueError('Data hash mismatch')
    with np.load(a.data_dir / 'two_step_data.npz', allow_pickle=False) as z:
        truth, dt = z['validation_truth'].copy(), z['validation_dt'].copy()
        vmin, vmax, tmin, tmax = (z[k].copy() for k in ('vmin', 'vmax', 'tmin', 'tmax'))
    sys.path.insert(0, str(a.source_root))
    import due
    if not Path(due.__file__).resolve().is_relative_to(a.source_root.resolve()):
        raise ValueError('Wrong DUE import')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True; torch.backends.cudnn.deterministic = True
    if a.device == 'cuda':
        if torch.cuda.device_count() != 1 or 'V100' not in torch.cuda.get_device_name(0):
            raise ValueError('Replay requires one allocated V100')
    runs = root / 'runs'
    if runs.exists():
        unknown = {x.name for x in runs.iterdir() if x.is_dir()} - {x[2] for x in expected_tasks}
        if unknown:
            raise ValueError('Unexpected run directories: ' + repr(sorted(unknown)))
    entries = []
    for arm, seed, name in expected_tasks:
        run = runs / name
        row = {'run_name': name, 'arm': arm, 'seed': seed, 'status': 'MISSING', 'replay_passed': False}
        try:
            if (run / 'FAILURE.json').exists():
                failure = json.loads((run / 'FAILURE.json').read_text())
                row.update(status='FAILED', failure=failure, failure_sha256=sha(run / 'FAILURE.json'))
                entries.append(row); continue
            if not (run / 'RUN_RECEIPT.json').exists():
                entries.append(row); continue
            inventory_sha = verify_inventory(run)
            receipt = json.loads((run / 'RUN_RECEIPT.json').read_text()); config = receipt['config']
            if receipt['status'] != 'AUTOREGRESSIVE_FULL_COMPLETE' or config['arm'] != arm or config['seed'] != seed:
                raise ValueError('Wrong completed task identity')
            expected_forwards = 50000 * (1 if arm == 'first' else 2)
            if receipt['actual_updates'] != 50000 or receipt['cursor'] != {'epoch': 500, 'batch': 0}:
                raise ValueError('Incomplete scheduled updates')
            if receipt['training_forward_calls'] != expected_forwards or receipt['final_learning_rate'] != 0.:
                raise ValueError('Forward budget or cosine endpoint mismatch')
            if config['regularizer'] is not None or config['source_commit'] != COMMIT or config['data_sha256'] != DATA_SHA:
                raise ValueError('Wrong frozen recipe')
            if config['network']['width'] != 20 or config['network']['modes1'] != 12 or config['network']['modes2'] != 12 or config['network']['depth'] != 4:
                raise ValueError('Wrong backbone')
            for path, digest in receipt['input_sha256'].items():
                if sha(Path(path)) != digest:
                    raise ValueError('Frozen input or code changed after training: ' + str(path))
            cp_entry = receipt['final_checkpoint']; checkpoint = safe_member(run, cp_entry['file'])
            if checkpoint.stat().st_size != cp_entry['bytes'] or sha(checkpoint) != cp_entry['sha256']:
                raise ValueError('Final checkpoint integrity mismatch')
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if state['schema'] != 'autoregressive_training_state_v1' or state['config'] != config or not state['resume_safe'] or state['failed']:
                raise ValueError('Final checkpoint identity/safety mismatch')
            if state['cursor'] != {'epoch': 500, 'batch': 0, 'updates': 50000} or state['forward_calls'] != expected_forwards:
                raise ValueError('Final checkpoint endpoint mismatch')
            history = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines() if line]
            if len(history) != 500 or history != state['history'] or any(row['epoch'] != i + 1 or row['updates'] != (i + 1) * 100 for i, row in enumerate(history)):
                raise ValueError('Incomplete or inconsistent epoch history')
            shuffle_sha = hashlib.sha256(''.join(row['permutation_sha256'] for row in history).encode()).hexdigest()
            if not finite_tensors(state):
                raise FloatingPointError('Non-finite final checkpoint tensor')
            net = due.networks.fno.osg_fno2d(vmin, vmax, tmin[()], tmax[()], config['network'], multiscale=False)
            net.load_state_dict(state['model_state_dict']); net = net.to(a.device)
            if parameter_sha(net) != receipt['final_parameter_sha256'] or receipt['initial_parameter_sha256'] == receipt['final_parameter_sha256']:
                raise ValueError('Parameter identity/change mismatch')
            with contextlib.redirect_stdout(io.StringIO()):
                pred = net.predict(truth[..., 0].copy(), dt.copy(), device=a.device)
            if pred.shape != truth.shape or pred.dtype != np.float32 or not np.isfinite(pred).all():
                raise FloatingPointError('Invalid replay rollout')
            diff, ref = pred.astype(np.float64) - truth.astype(np.float64), truth.astype(np.float64)
            joint = np.linalg.norm(diff.reshape(20, -1), axis=1) / np.linalg.norm(ref.reshape(20, -1), axis=1)
            by_time = np.linalg.norm(diff.reshape(20, -1, 51), axis=1) / np.linalg.norm(ref.reshape(20, -1, 51), axis=1)
            saved = receipt['final_validation']
            if saved['finite'] is not True or not np.isfinite(joint).all() or not np.isfinite(by_time).all():
                raise FloatingPointError('Final metrics not finite')
            # Same CUDA replay should match exactly; CPU replay permits small
            # float32 FFT/backend differences and records the actual discrepancy.
            rtol, atol = ((0., 0.) if a.device == 'cuda' else (3e-5, 1e-7))
            np.testing.assert_allclose(joint, np.array(saved['joint_per_trajectory']), rtol=rtol, atol=atol)
            np.testing.assert_allclose(by_time, np.array(saved['relative_l2_by_snapshot']), rtol=rtol, atol=atol)
            future_joint = np.linalg.norm(diff[..., 1:].reshape(20, -1), axis=1) / np.linalg.norm(ref[..., 1:].reshape(20, -1), axis=1)
            row.update(status='COMPLETE_AND_REPLAYED', replay_passed=True, actual_updates=50000,
                receipt_sha256=sha(run / 'RUN_RECEIPT.json'), artifact_manifest_sha256=inventory_sha,
                checkpoint_sha256=cp_entry['sha256'], initial_parameter_sha256=receipt['initial_parameter_sha256'],
                final_parameter_sha256=receipt['final_parameter_sha256'], all_checkpoint_tensors_finite=True,
                epoch_count=500, shuffle_sequence_sha256=shuffle_sha,
                joint_mean=float(joint.mean()), future_only_joint_mean=float(future_joint.mean()),
                horizon_step_mean={str(k): float(by_time[:, k].mean()) for k in (1, 2, 4, 8, 16, 32, 50)},
                joint_per_trajectory=joint.tolist(), relative_l2_by_snapshot=by_time.tolist(),
                replay_metric_max_abs_difference=float(np.max(abs(by_time - np.array(saved['relative_l2_by_snapshot'])))),
                replay_rtol=rtol, replay_atol=atol, training_seconds=receipt['training_seconds'],
                validation_seconds=receipt['validation_seconds'], training_forward_calls=receipt['training_forward_calls'],
                cuda_peak_allocated_bytes=receipt['cuda_peak_allocated_bytes'])
            del net, state, pred
            if a.device == 'cuda':
                torch.cuda.empty_cache()
        except BaseException as error:
            row.update(status='AUDIT_OR_REPLAY_FAILED', error=repr(error), traceback=traceback.format_exc())
        entries.append(row)
    paired = []; completed = all(x['status'] == 'COMPLETE_AND_REPLAYED' for x in entries)
    for seed in (0, 1, 2):
        one = next(x for x in entries if x['arm'] == 'first' and x['seed'] == seed)
        two = next(x for x in entries if x['arm'] == 'rollout2' and x['seed'] == seed)
        item = {'seed': seed, 'ratio_rollout2_over_first': None, 'paired_initialization': None, 'paired_shuffle': None}
        if one['replay_passed'] and two['replay_passed']:
            item['paired_initialization'] = one['initial_parameter_sha256'] == two['initial_parameter_sha256']
            item['paired_shuffle'] = one['shuffle_sequence_sha256'] == two['shuffle_sequence_sha256']
            if not item['paired_initialization'] or not item['paired_shuffle']:
                completed = False
            item['ratio_rollout2_over_first'] = two['joint_mean'] / one['joint_mean'] if one['joint_mean'] > 0 else None
            item['training_time_ratio'] = two['training_seconds'] / one['training_seconds']
            item['horizon_ratios'] = {k: two['horizon_step_mean'][k] / one['horizon_step_mean'][k] if one['horizon_step_mean'][k] > 0 else None for k in one['horizon_step_mean']}
        paired.append(item)
    ratios = [x['ratio_rollout2_over_first'] for x in paired]
    valid = completed and all(x is not None and math.isfinite(x) and x > 0 for x in ratios)
    gm = math.exp(sum(math.log(x) for x in ratios) / 3) if valid else None
    favored = sum(x < 1 for x in ratios) if valid else None
    report = {'schema': 'autoregressive_replay_aggregate_v1', 'status': 'COMPLETE' if valid else 'INCOMPLETE_OR_FAILED',
        'all_six_scheduled_final_endpoints': completed, 'entries': entries, 'paired': paired,
        'joint_ratio_geometric_mean': gm, 'seeds_favoring_rollout2': favored,
        'practical_gate': {'threshold': .90, 'required_favoring_seeds': 2, 'passed': bool(valid and gm <= .9 and favored >= 2)},
        'replay_device': a.device, 'torch': torch.__version__, 'numpy': np.__version__,
        'source_manifest_sha256': SOURCE_SHA, 'data_sha256': DATA_SHA, 'matrix_sha256': sha(root / 'matrix.json'),
        'script_sha256': sha(__file__), 'whole_process_wall_seconds': time.perf_counter() - started,
        'scope': 'Development/exploratory comparison. Extra second labels and training cost are confounded; no statistical significance or isolated exposure-bias claim. Failed/missing seeds retained without substitutes.'}
    write_json(output / 'AGGREGATE_REPORT.json', report)
    with (output / 'paired_results.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'ratio_rollout2_over_first', 'paired_initialization', 'paired_shuffle', 'training_time_ratio'])
        writer.writeheader(); writer.writerows({k: x.get(k) for k in writer.fieldnames} for x in paired)
    with (output / 'horizon_results.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['run_name', 'arm', 'seed', 'status', 'joint_mean', 'horizon', 'relative_l2_mean'])
        writer.writeheader()
        for entry in entries:
            for horizon, value in entry.get('horizon_step_mean', {'': None}).items():
                writer.writerow({**{k: entry.get(k) for k in ('run_name', 'arm', 'seed', 'status', 'joint_mean')}, 'horizon': horizon, 'relative_l2_mean': value})
    print(json.dumps({k: report[k] for k in ('status', 'joint_ratio_geometric_mean', 'seeds_favoring_rollout2', 'practical_gate')}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
