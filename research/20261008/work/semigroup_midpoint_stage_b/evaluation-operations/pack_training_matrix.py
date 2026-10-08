#!/usr/bin/env python3
"""Capture the completed frozen Stage B matrix once, without replacing evidence.

Only the pinned pack_evidence.py creates model archives. Existing complete
captures are verified and reused. Any partial capture or changed shared file
stops the sequence. This is remote capture, not a local recovery attestation.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
BROOT = Path('/data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-r1')
PYTHON = Path('/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python')
ARRAY = '2349407'
PACK_SHA = '4406547114b29f6a99b1a68a067cb3095bda5df70f139815ba347e6afcb08bbc'
PAYLOAD_SHA = 'c30e2208cff36ba79b91da43b947fd85f7f0c23088d5020e2824f906704df466'
ARMS = {'gdsg500', 'single500', 'midpoint500', 'gdsg333'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def safe_file(root, name):
    relative = PurePosixPath(name)
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
            and str(relative) == name, 'Unsafe path: ' + name)
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink in path: ' + str(path))
    require(path.is_file(), 'Missing regular file: ' + str(path))
    return path


def tasks_for(root, frozen):
    spec = frozen['files']['matrix.json']
    path = safe_file(root, 'matrix.json')
    require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Frozen matrix changed')
    tasks = read(path)['tasks']
    require(len(tasks) == 12 and {(t['arm'], t['seed']) for t in tasks}
            == {(arm, seed) for arm in ARMS for seed in range(3)}, 'Need all twelve fixed cells')
    for index, task in enumerate(tasks):
        epochs = 333 if task['arm'] == 'gdsg333' else 500
        require(task['index'] == index and task['run_name'] == f"{task['arm']}-seed{task['seed']}-r1"
                and task['epochs'] == epochs and task['updates'] == epochs * 100, 'Changed matrix identity/budget')
    return tasks


def accounting(raw):
    reader = csv.DictReader(io.StringIO(raw), delimiter='|')
    require({'JobID', 'JobIDRaw', 'State', 'ExitCode'} <= set(reader.fieldnames or []), 'Missing accounting headers')
    rows = [{key: str(value).strip() for key, value in row.items() if key is not None} for row in reader]
    allocations = [r for r in rows if re.fullmatch(ARRAY + r'_\d+', r['JobID'])]
    require(len(allocations) == 12 and {r['JobID'] for r in allocations}
            == {ARRAY + '_' + str(i) for i in range(12)}, 'Missing/duplicate/unexpected allocation')
    require(all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0'
                and r['JobIDRaw'].isdigit() for r in allocations), 'All twelve allocations must be COMPLETED 0:0')
    require(len({r['JobIDRaw'] for r in allocations}) == 12, 'Duplicate raw allocation ID')
    steps = [r for r in rows if re.fullmatch(ARRAY + r'_\d+\..+', r['JobID'])]
    require(all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0' for r in steps),
            'Active or unsuccessful training step/batch remains')
    return {r['JobID']: r for r in allocations}


def query(command, evidence):
    reply = subprocess.run(command, capture_output=True, text=True, timeout=90, check=False)
    evidence.append({'command': command, 'returncode': reply.returncode,
                     'stdout': reply.stdout, 'stderr': reply.stderr,
                     'queried_at_utc': datetime.now(timezone.utc).isoformat()})
    require(reply.returncode == 0, 'Read-only scheduler query failed')
    return reply.stdout


def shared_snapshot(root):
    """Exactly the shared dynamic files selected by the frozen packer for this array."""
    result = {}
    for directory, pattern in [('logs', '*'), ('operations', '*.json')]:
        parent = root / directory
        require(not parent.is_symlink(), 'Shared directory is a symlink')
        for path in sorted(parent.glob(pattern)):
            if path.is_file() and (directory == 'operations' or ARRAY in path.name):
                name = str(path.relative_to(root)); safe_file(root, name)
                result[name] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result


def capture_paths(root, task):
    job = ARRAY + '_' + str(task['index'])
    stem = task['run_name'] + '-job' + job + '-COMPLETE'
    recovery = root / 'recovery'
    return job, recovery / stem, recovery / (stem + '.tar.gz'), recovery / (stem + '.archive.json')


def existing_capture(root, task):
    _, evidence, archive, receipt = capture_paths(root, task)
    present = [p.exists() or p.is_symlink() for p in (evidence, archive, receipt)]
    require(not any(present) or all(present), 'Incomplete existing capture: ' + task['run_name'])
    require(not any(p.is_symlink() for p in (evidence, archive, receipt)), 'Capture symlink forbidden')
    return all(present)


def verify_capture(root, task, row, shared, frozen, packer):
    job, evidence, archive, receipt_path = capture_paths(root, task)
    require(existing_capture(root, task), 'Capture did not produce all three components')
    require(evidence.is_dir(), 'Evidence must be a directory')
    safe_file(root, str(archive.relative_to(root))); safe_file(root, str(receipt_path.relative_to(root)))
    receipt = read(receipt_path)
    relative_inventory = str((evidence / 'INVENTORY.json').relative_to(root))
    expected = {'archive': str(archive), 'job_id': job, 'allocation_job_id_raw': row['JobIDRaw'],
                'mode': 'run', 'capture_status': 'COMPLETE', 'inventory_relative_path': relative_inventory,
                'payload_manifest_sha256': PAYLOAD_SHA}
    require(all(receipt.get(k) == v for k, v in expected.items()), 'Capture receipt identity differs')
    require(archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256'], 'Archive bytes/hash differ')
    inventory_path = safe_file(root, relative_inventory)
    inventory = read(inventory_path)
    require(inventory['schema'] == 'stage_b_evidence_v1' and inventory['root_name'] == root.name
            and inventory['run_relative_path'] == 'runs/' + task['run_name']
            and all(inventory.get(k) == v for k, v in expected.items() if k != 'archive'), 'Inventory identity differs')
    specs = {item['path']: {'bytes': item['bytes'], 'sha256': item['sha256']} for item in inventory['files']}
    require(len(specs) == len(inventory['files']) == receipt['inventory_files'], 'Inventory duplicates/count mismatch')
    run = root / 'runs' / task['run_name']
    expected_paths = {str(path.relative_to(root)) for path in run.rglob('*')
                      if path.is_file() and 'matplotlib-cache' not in path.parts}
    expected_paths |= {'PAYLOAD_MANIFEST.json', str((evidence / 'sacct.txt').relative_to(root))}
    expected_paths |= set(frozen['files']) - {'data-r1/development_data.npz'}
    expected_paths |= set(shared)
    require(set(specs) == expected_paths, 'Captured file membership differs from frozen packer selection')
    require(all(specs.get(name) == item for name, item in shared.items()), 'Captured shared files differ')
    require({name for name in specs if name.startswith('logs/') or name.startswith('operations/')}
            == set(shared), 'Captured shared file membership differs')
    for name, spec in specs.items():
        path = safe_file(root, name)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Captured source changed: ' + name)
    packer.verify_complete_run(root, root / 'runs' / task['run_name'], 'run', job, frozen, row['JobIDRaw'])
    expected_members = {root.name + '/' + name: spec for name, spec in specs.items()}
    expected_members[root.name + '/' + relative_inventory] = {'bytes': inventory_path.stat().st_size, 'sha256': sha(inventory_path)}
    require(receipt['archive_members'] == len(expected_members), 'Archive member count mismatch')
    seen = set()
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            require(member.isfile() and member.sparse is None and member.name in expected_members
                    and member.name not in seen, 'Unsafe/extra/duplicate archive member')
            spec = expected_members[member.name]
            require(member.size == spec['bytes'], 'Archive member size mismatch')
            digest = hashlib.sha256(); source = stream.extractfile(member)
            for block in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(block)
            require(digest.hexdigest() == spec['sha256'], 'Archive member checksum mismatch')
            seen.add(member.name)
    require(seen == set(expected_members), 'Archive missing members')
    return {'receipt': receipt, 'receipt_sha256': sha(receipt_path), 'inventory_sha256': sha(inventory_path),
            'verified_archive_members': len(seen), 'local_recovery_performed': False}


def run(args):
    root = args.root.resolve(strict=True)
    require(root == BROOT, 'Only the frozen Stage B training root is authorized')
    require(sha(safe_file(root, 'PAYLOAD_MANIFEST.json')) == PAYLOAD_SHA, 'Training payload changed')
    require(sha(safe_file(root, 'pack_evidence.py')) == PACK_SHA, 'Frozen packer changed')
    frozen = read(root / 'PAYLOAD_MANIFEST.json'); tasks = tasks_for(root, frozen)
    for name, spec in frozen['files'].items():
        path = safe_file(root, name)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Frozen input changed: ' + name)
    module_spec = importlib.util.spec_from_file_location('frozen_training_packer', root / 'pack_evidence.py')
    packer = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(packer)
    recovery = root / 'recovery'
    require(not recovery.is_symlink(), 'Recovery root symlink forbidden'); recovery.mkdir(exist_ok=True)
    report_path = recovery / 'MATRIX_CAPTURE_REPORT.json'
    require(not report_path.exists() and not report_path.is_symlink(), 'Matrix report exists; inspect/reuse, never overwrite')
    evidence = recovery / 'MATRIX_CAPTURE'; evidence.mkdir(exist_ok=False)
    report = {'schema': 'stage_b_training_matrix_capture_v1', 'status': 'STARTED', 'training_array_job': ARRAY,
              'root': str(root), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'driver_sha256': sha(__file__), 'packer_sha256': PACK_SHA, 'payload_manifest_sha256': PAYLOAD_SHA,
              'scheduler_evidence': [], 'runs': [{'task': task, 'status': 'NOT_ATTEMPTED'} for task in tasks],
              'local_recovery_performed': False}
    write(evidence / 'INTENT.json', report)
    try:
        raw = query(['sacct', '--jobs=' + ARRAY, '--parsable2', '--format=JobID,JobIDRaw,State,ExitCode'], report['scheduler_evidence'])
        rows = accounting(raw)
        queue = query(['squeue', '--array', '--noheader', '--jobs=' + ARRAY, '--format=%i|%T'], report['scheduler_evidence'])
        require(not queue.strip(), 'Training array still appears in active queue')
        shared = shared_snapshot(root); time.sleep(2)
        require(shared_snapshot(root) == shared, 'Shared logs/operations are still changing')
        report['shared_files'] = shared; report['stability_interval_seconds'] = 2
        # Check all preexisting capture components before creating any new archive.
        reuse = {}
        for record in report['runs']:
            try:
                reuse[record['task']['run_name']] = existing_capture(root, record['task'])
            except BaseException as error:
                record.update(status='FAILED_STOPPED', error=repr(error)); raise
        for record in report['runs']:
            task = record['task']; job, _, _, _ = capture_paths(root, task)
            try:
                require(shared_snapshot(root) == shared, 'Shared files changed before capture')
                if not reuse[task['run_name']]:
                    require(sha(root / 'pack_evidence.py') == PACK_SHA, 'Frozen packer changed')
                    command = [str(PYTHON), '-B', '-u', str(root / 'pack_evidence.py'), '--root', str(root),
                               '--job-id', job, '--mode', 'run', '--run-name', task['run_name'], '--capture', 'COMPLETE']
                    record['command'] = command
                    with (evidence / (task['run_name'] + '.stdout.txt')).open('x') as stdout, (evidence / (task['run_name'] + '.stderr.txt')).open('x') as stderr:
                        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
                    record['returncode'] = result.returncode
                    require(result.returncode == 0, 'Frozen packer failed; partial capture retained')
                record.update(verify_capture(root, task, rows[job], shared, frozen, packer))
                require(shared_snapshot(root) == shared, 'Shared files changed during capture')
                record['status'] = 'REUSED_COMPLETE_CAPTURE' if reuse[task['run_name']] else 'CAPTURED_AND_VERIFIED'
                write(evidence / (task['run_name'] + '.json'), record)
                print(json.dumps({'run': task['run_name'], 'status': record['status'], 'archive': record['receipt']['archive'], 'sha256': record['receipt']['sha256'], 'bytes': record['receipt']['bytes']}), flush=True)
            except BaseException as error:
                record.update(status='FAILED_STOPPED', error=repr(error)); raise
        report['status'] = 'ALL_TWELVE_CAPTURES_VERIFIED'
    except BaseException as error:
        report.update(status='STOPPED_INCOMPLETE', error=repr(error))
    finally:
        report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        report['verified_capture_count'] = sum(r['status'] in {'REUSED_COMPLETE_CAPTURE', 'CAPTURED_AND_VERIFIED'} for r in report['runs'])
        write(report_path, report)
    print(json.dumps({'status': report['status'], 'verified_capture_count': report['verified_capture_count'], 'report': str(report_path), 'report_sha256': sha(report_path)}), flush=True)
    return 0 if report['status'] == 'ALL_TWELVE_CAPTURES_VERIFIED' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
