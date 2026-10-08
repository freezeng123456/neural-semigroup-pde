#!/usr/bin/env python3
"""Capture a finished Stage B allocation without replacing existing evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def canonical(name):
    value = PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts or str(value) != name or not value.parts:
        raise ValueError('Unsafe/noncanonical path: ' + name)
    return value


def file_inside(root, name):
    root = root.resolve(strict=True)
    canonical(name)
    path = root / name
    assert path.resolve(strict=True).is_relative_to(root) and path.is_file() and not path.is_symlink(), name
    return path


def verify_complete_run(root, run, mode, job_id, frozen, job_id_raw):
    """Verify the run's earlier manifest; a new inventory cannot replace it."""
    assert not (run / 'FAILURE.json').exists()
    original_path = file_inside(run, 'artifact_manifest.json')
    original = json.loads(original_path.read_text())
    assert isinstance(original, dict) and original
    for name, item in original.items():
        path = file_inside(run, name)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    if mode == 'preflight':
        required = {'VALIDATION_REPORT.json', 'PREFLIGHT_PASSED.json'}
        profile = json.loads((run / 'VALIDATION_REPORT.json').read_text())
        checks = profile.get('checks')
        assert profile['schema'] == 'stage_b_engineering_validation_v1' and profile['status'] == 'PASS'
        assert isinstance(checks, list) and checks and all(c.get('name') and c.get('passed') is True for c in checks)
        assert len({c['name'] for c in checks}) == len(checks)
        assert profile['device'] == 'cuda' and str(profile['slurm_job_id']) == job_id == job_id_raw
        assert profile['payload_manifest_sha256'] == sha(root / 'PAYLOAD_MANIFEST.json')
        assert profile['source_commit'] == frozen['source_commit']
        assert profile['source_manifest_sha256'] == frozen['files']['official_source_manifest.json']['sha256']
        assert profile['data_payload_sha256'] == frozen['files']['data-r1/development_data.npz']['sha256']
        detailed = file_inside(run, profile['detailed_report'])
        assert detailed.name == 'PREFLIGHT_PASSED.json' and sha(detailed) == profile['detailed_report_sha256']
    else:
        required = {'RUN_RECEIPT.json', 'configuration.json', 'environment.json', 'epochs.jsonl', 'updates.jsonl',
                    'final_network.pt', 'final_state.pt', 'latest_state.pt',
                    'validation_final_prediction.npy', 'validation_final_metrics.npz'}
        profile = json.loads((run / 'RUN_RECEIPT.json').read_text())
        match = re.fullmatch(r'(gdsg500|single500|midpoint500|gdsg333)-seed([012])-r1', run.name)
        assert match
        arm, seed = match.group(1), int(match.group(2))
        epochs = 333 if arm == 'gdsg333' else 500
        config = profile['config']
        assert profile['status'] == 'STAGE_B_FULL_COMPLETE'
        assert config['arm'] == arm and config['seed'] == seed and config['epochs'] == epochs
        assert config['planned_updates'] == profile['actual_updates'] == epochs * 100
        assert profile['cursor'] == {'epoch': epochs, 'batch': 0}
        assert profile['forward_calls'] == epochs * 100 * (6 if arm.startswith('gdsg') else 4)
        final = profile['final_validation']
        assert final['valid_metric'] is True and final['final'] is True and final['epoch'] == epochs
        assert final['updates'] == epochs * 100
        assert final['prediction_file'] == 'validation_final_prediction.npy' and final['metrics_file'] == 'validation_final_metrics.npz'
        for key in ('prediction', 'metrics'):
            assert sha(file_inside(run, final[key + '_file'])) == final[key + '_sha256']
        assert json.loads((run / 'configuration.json').read_text()) == config
        environment = json.loads((run / 'environment.json').read_text())
        assert environment['mode'] == 'full' and environment['device'] == 'cuda'
        assert str(environment['slurm']['SLURM_JOB_ID']) == job_id_raw, 'Run belongs to another Slurm allocation'
        assert config['source_commit'] == frozen['source_commit']
        assert config['data_payload_sha256'] == frozen['files']['data-r1/development_data.npz']['sha256']
    assert required <= set(original), 'Run manifest omits mandatory outputs'
    for name in required:
        file_inside(run, name)
    return profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--mode', choices=['preflight', 'run'], required=True)
    p.add_argument('--run-name')
    p.add_argument('--capture', choices=['COMPLETE', 'FAILED'], default='COMPLETE')
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    assert re.fullmatch(r'[0-9]+(?:_[0-9]+)?', a.job_id)
    assert root.name.startswith('20261004-midpoint-stage-b-')
    frozen = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text())
    assert frozen['prefix'] == root.name
    # Reject path aliases before any archive/evidence directory is created.
    for name, item in frozen['files'].items():
        path = file_inside(root, name)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    if a.mode == 'preflight':
        assert a.run_name is None and a.job_id.isdigit()
        run = root / 'preflight' / ('job' + a.job_id)
    else:
        assert re.fullmatch(r'(?:gdsg500|single500|midpoint500|gdsg333)-seed[012]-r1', a.run_name or '')
        run = root / 'runs' / a.run_name
    assert run.is_dir() and not run.is_symlink()
    stem = f'{run.name}-job{a.job_id}-{a.capture}'
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    evidence = recovery / stem
    archive = recovery / (stem + '.tar.gz')
    receipt_path = recovery / (stem + '.archive.json')
    assert not archive.exists() and not evidence.exists() and not receipt_path.exists(), 'Inspect/reuse existing capture; no overwrite'
    command = ['sacct', '-j', a.job_id, '-n', '-P', '--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList,JobIDRaw']
    scheduler = subprocess.check_output(command, text=True)
    records = [line.split('|') for line in scheduler.splitlines() if line.split('|')[0] == a.job_id]
    assert len(records) == 1, 'Need exact completed allocation record'
    record = records[0]
    assert len(record) >= 8 and record[7].isdigit(), 'Need numeric allocation JobIDRaw'
    job_id_raw = record[7]
    terminal = record[2].split()[0].rstrip('+')
    if a.capture == 'COMPLETE':
        assert terminal == 'COMPLETED' and record[3] == '0:0'
        verify_complete_run(root, run, a.mode, a.job_id, frozen, job_id_raw)
    else:
        assert terminal in {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'COMPLETED'}
        assert terminal != 'COMPLETED' or record[3] != '0:0' or (run / 'FAILURE.json').exists()
    files = {path for path in run.rglob('*') if path.is_file() and 'matplotlib-cache' not in path.parts}
    files.add(root / 'PAYLOAD_MANIFEST.json')
    # Inputs already saved locally need not be transferred a second time.
    for name, item in frozen['files'].items():
        path = file_inside(root, name)
        if name != 'data-r1/development_data.npz':
            files.add(path)
    for path in (root / 'logs').glob('*'):
        if path.is_file() and (a.job_id in path.name or a.job_id.split('_')[0] in path.name):
            files.add(path)
    for path in (root / 'operations').glob('*.json'):
        if path.is_file():
            files.add(path)
    evidence.mkdir()
    scheduler_path = evidence / 'sacct.txt'
    with scheduler_path.open('x') as f:
        f.write(scheduler)
    files.add(scheduler_path)
    for path in files:
        assert path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
    inventory_path = evidence / 'INVENTORY.json'
    inventory = {'schema': 'stage_b_evidence_v1', 'job_id': a.job_id, 'allocation_job_id_raw': job_id_raw, 'mode': a.mode,
                 'capture_status': a.capture, 'root_name': root.name,
                 'run_relative_path': str(run.relative_to(root)),
                 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                 'payload_manifest_sha256': sha(root / 'PAYLOAD_MANIFEST.json'),
                 'inventory_relative_path': str(inventory_path.relative_to(root)),
                 'omitted': ['development NPZ already frozen locally', 'dependency binaries', 'matplotlib cache'],
                 'files': [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha(path)}
                           for path in sorted(files)]}
    write(inventory_path, inventory)
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(files | {inventory_path}):
            tar.add(path, arcname=root.name + '/' + str(path.relative_to(root)), recursive=False)
    receipt = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'job_id': a.job_id, 'allocation_job_id_raw': job_id_raw, 'mode': a.mode, 'capture_status': a.capture,
               'inventory_files': len(files), 'archive_members': len(files) + 1,
               'inventory_relative_path': str(inventory_path.relative_to(root)),
               'payload_manifest_sha256': inventory['payload_manifest_sha256']}
    write(receipt_path, receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
