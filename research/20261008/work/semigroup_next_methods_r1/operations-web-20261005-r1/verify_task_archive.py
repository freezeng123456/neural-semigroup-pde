#!/usr/bin/env python3
"""Verify complete evidence and every scheduled update of one fixed task."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from capture_completed_task import audit_archive, digest

MANIFEST_SHA = 'ffbe1bb1b40c46ba4cd41a9fc1b94ea0dbce130c7ee8c646203fb4f16543d8a1'


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    outer = json.loads(args.receipt.read_text())
    assert outer['payload_manifest_sha256'] == MANIFEST_SHA
    inventory = audit_archive(args.archive, outer)
    assert inventory['payload_manifest_sha256'] == MANIFEST_SHA
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive, 'r:gz') as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts
            assert str(path) == member.name and path.parts[0] == inventory['root_name']
            target = args.output.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f, tar.extractfile(member) as source:
                shutil.copyfileobj(source, f)
    root = args.output / inventory['root_name']
    assert digest(root / 'PAYLOAD_MANIFEST.json') == MANIFEST_SHA
    manifest = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text())
    assert len(manifest['files']) == 99
    for name, item in manifest['files'].items():
        if not name.endswith('.npz'):
            file = root / name
            assert file.stat().st_size == item['bytes'] and digest(file) == item['sha256'], name
    sys.path.insert(0, str(root))
    from experiment_config import SPECS, matrix
    frozen_matrix = json.loads((root / 'matrix.json').read_text())
    assert frozen_matrix == matrix()
    task = frozen_matrix['tasks'][int(outer['job'].split('_')[1])]
    assert outer['run_relative_path'] == 'runs/' + task['run_name']
    run = root / outer['run_relative_path']
    artifacts = json.loads((run / 'artifact_manifest.json').read_text())
    run_files = {str(p.relative_to(run)) for p in run.rglob('*') if p.is_file()}
    assert run_files == set(artifacts) | {'artifact_manifest.json'}
    for name, item in artifacts.items():
        path = PurePosixPath(name)
        assert not path.is_absolute() and '..' not in path.parts and str(path) == name
        file = run / name
        assert file.stat().st_size == item['bytes'] and digest(file) == item['sha256'], name
    receipt = json.loads((run / 'RUN_RECEIPT.json').read_text())
    finite(receipt)
    config = json.loads((run / 'configuration.json').read_text())
    assert receipt['config'] == config
    assert config['arm'] == task['arm'] and config['seed'] == task['seed']
    spec = SPECS[task['arm']]
    assert config['spec'] == spec and config['epochs'] == spec['epochs']
    budget = spec['epochs'] * 100
    assert config['planned_updates'] == budget and config['network']['width'] == spec['width']
    assert receipt['status'] == 'NEXT_METHODS_FULL_COMPLETE' and receipt['actual_updates'] == budget
    assert receipt['cursor'] == {'epoch': spec['epochs'], 'batch': 0}
    assert receipt['forward_calls'] == budget * spec['calls']
    assert receipt['parameters_changed'] and receipt['final_parameter_sha256'] != receipt['initial_parameter_sha256']
    assert receipt['final_learning_rate'] == 0.0
    calls = 0
    count = 0
    with (run / 'updates.jsonl').open() as f:
        for count, line in enumerate(f, 1):
            row = json.loads(line)
            finite(row)
            assert row['update'] == count
            assert row['epoch'] == (count - 1) // 100 + 1 and row['batch'] == (count - 1) % 100 + 1
            assert row['forward_calls'] == spec['calls']
            calls += row['forward_calls']
    assert count == budget and calls == receipt['forward_calls']
    epochs = [json.loads(line) for line in (run / 'epochs.jsonl').read_text().splitlines()]
    finite(epochs)
    assert epochs == receipt['history'] and len(epochs) == spec['epochs']
    assert all(row['epoch'] == i and row['updates'] == i * 100 for i, row in enumerate(epochs, 1))
    final = receipt['final_validation']
    assert final['valid_metric'] and final['status'] == 'FINITE_ROLLOUT'
    assert final['final'] and final['scheduled_final'] and final['epoch'] == spec['epochs'] and final['updates'] == budget
    assert len(final['joint_per_trajectory']) == 20 and len(final['relative_l2_mean_by_snapshot']) == 51
    assert len(receipt['validation_history']) == spec['epochs'] // 25
    assert receipt['validation_history'][-1] == final
    expected_weights = {f'epoch{i:04d}_weights.pt' for i in range(25, spec['epochs'], 25)}
    assert {name for name in artifacts if name.startswith('epoch') and name.endswith('_weights.pt')} == expected_weights
    for name in ('final_state.pt', 'latest_state.pt', 'final_weights.pt', 'validation_final_prediction.npy', 'validation_final_metrics.npz'):
        assert name in artifacts and artifacts[name]['bytes'] > 0
    assert digest(run / final['prediction_file']) == final['prediction_sha256']
    assert digest(run / final['metrics_file']) == final['metrics_sha256']
    env = json.loads((run / 'environment.json').read_text())
    assert env['device'] == 'cuda' and 'V100' in env['gpu_name'] and env['mode'] == 'full'
    assert env['parameter_count_real_scalar_components'] > 0
    sacct = (root / outer['inventory_relative_path']).parent / 'sacct.txt'
    rows = [line.split('|') for line in sacct.read_text().splitlines() if line.split('|')[0] == outer['job']]
    assert len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0']
    if outer.get('immutable_view_used'):
        provenance = json.loads((root / outer['capture_provenance_relative_path']).read_text())
        assert provenance['job'] == outer['job'] and provenance['run_name'] == task['run_name']
        import hashlib
        assert hashlib.sha256(provenance['program_source'].encode()).hexdigest() == provenance['program_sha256']
        assert provenance['frozen_packer_sha256'] == manifest['files']['pack_evidence.py']['sha256']
        assert provenance['original_run_manifest_sha256'] == digest(run / 'artifact_manifest.json')
        assert provenance['run_file_count'] == len(run_files)
    gate = {
        'schema': 'next_methods_complete_task_recovery_v1', 'status': 'PASS',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'job': outer['job'], 'arm': task['arm'], 'seed': task['seed'], 'run_name': task['run_name'],
        'archive_sha256': outer['sha256'], 'archive_bytes': outer['bytes'],
        'archive_members': outer['archive_members'], 'all_inventory_files_verified': len(inventory['files']),
        'payload_manifest_sha256': MANIFEST_SHA, 'actual_updates': count, 'forward_calls': calls,
        'epochs': len(epochs), 'intermediate_weight_checkpoints': len(expected_weights),
        'checkpoint_file_count': len([name for name in artifacts if name.endswith('.pt')]),
        'parameters_real': env['parameter_count_real_scalar_components'],
        'development_joint_mean': final['joint_mean'], 'development_final_step_mean': final['final_step_mean'],
        'training_seconds': receipt['training_seconds'], 'whole_process_wall_seconds': receipt['whole_process_wall_seconds'],
        'recovered_root': str(root.resolve()), 'run_manifest_sha256': digest(run / 'artifact_manifest.json'),
        'selected_endpoint': 'Frozen scheduled final; no checkpoint or seed selection',
        'scope': 'Evidence and engineering checks for one task; method comparison awaits all paired seeds',
    }
    with (args.output / 'LOCAL_TASK_RECOVERY_GATE.json').open('x') as f:
        json.dump(gate, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
