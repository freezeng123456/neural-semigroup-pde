#!/usr/bin/env python3
"""Run the frozen packer in an immutable view of one completed array task.

Other array tasks write their logs every epoch. The frozen packer includes all
logs sharing the parent job number, so use a separate view containing this
completed task's complete logs. Scientific files are unchanged and verified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit_archive(archive, sidecar):
    assert archive.stat().st_size == sidecar['bytes']
    assert digest(archive) == sidecar['sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == sidecar['archive_members']
        assert len({m.name for m in members}) == len(members)
        for m in members:
            p = PurePosixPath(m.name)
            assert m.isfile() and not p.is_absolute() and '..' not in p.parts
        inventory_name = '20261005-next-methods-r1/' + sidecar['inventory_relative_path']
        inv = json.load(tar.extractfile(inventory_name))
        assert inv['job'] == sidecar['job'] and inv['capture'] == 'COMPLETE'
        expected = {inv['root_name'] + '/' + k: v for k, v in inv['files'].items()}
        assert {m.name for m in members} == set(expected) | {inventory_name}
        for name, item in expected.items():
            member = tar.getmember(name)
            assert member.size == item['bytes']
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            assert h.hexdigest() == item['sha256'], name
    return inv


def stable_copy(source, target, root):
    assert source.is_file() and not source.is_symlink()
    assert source.resolve(strict=True).is_relative_to(root)
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    shutil.copy2(source, target)
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), source
    assert target.stat().st_size == after.st_size and digest(target) == digest(source)


def show(sidecar_path, receipt, reused):
    print('CAPTURE_STATUS VERIFIED', 'REUSED' if reused else 'CREATED', flush=True)
    print('ARCHIVE', Path(receipt['archive']).name, flush=True)
    print('ARCHIVE_BYTES', receipt['bytes'], flush=True)
    print('ARCHIVE_SHA_HEAD', receipt['sha256'][:32], flush=True)
    print('ARCHIVE_SHA_TAIL', receipt['sha256'][32:], flush=True)
    print('SIDECAR', sidecar_path.name, flush=True)
    print('SIDECAR_BYTES', sidecar_path.stat().st_size, flush=True)
    sha = digest(sidecar_path)
    print('SIDECAR_SHA_HEAD', sha[:32], flush=True)
    print('SIDECAR_SHA_TAIL', sha[32:], flush=True)
    print('ARCHIVE_MEMBERS', receipt['archive_members'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    assert root.name == '20261005-next-methods-r1'
    assert re.fullmatch(r'\d+_\d+', args.job)
    assert re.fullmatch(r'[a-z0-9]+-seed[012]-r1', args.run_name)
    sys.path.insert(0, str(root))
    from launch_task import verify_payload
    frozen = verify_payload(root)
    assert digest(root / 'PAYLOAD_MANIFEST.json') == 'ffbe1bb1b40c46ba4cd41a9fc1b94ea0dbce130c7ee8c646203fb4f16543d8a1'
    stem = args.run_name + '-job' + args.job + '-COMPLETE'
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    archive = recovery / (stem + '.tar.gz')
    sidecar_path = recovery / (stem + '.archive.json')
    directory = recovery / stem
    if archive.exists() or sidecar_path.exists() or directory.exists():
        assert archive.is_file() and sidecar_path.is_file() and directory.is_dir(), 'Partial capture: investigate without overwriting'
        receipt = json.loads(sidecar_path.read_text())
        assert receipt['job'] == args.job and receipt['run_relative_path'] == 'runs/' + args.run_name
        audit_archive(archive, receipt)
        show(sidecar_path, receipt, True)
        return
    raw = subprocess.check_output(['sacct', '-j', args.job, '-nP', '--format=JobID%40,State%30,ExitCode'], text=True)
    rows = [line.split('|') for line in raw.splitlines() if line.split('|')[0] == args.job]
    assert len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0'], raw
    run = root / 'runs' / args.run_name
    matrix = json.loads((root / 'matrix.json').read_text())
    task = matrix['tasks'][int(args.job.split('_')[1])]
    assert task['run_name'] == args.run_name
    receipt = json.loads((run / 'RUN_RECEIPT.json').read_text())
    assert receipt['status'] == 'NEXT_METHODS_FULL_COMPLETE'
    assert receipt['config']['arm'] == task['arm'] and receipt['config']['seed'] == task['seed']
    original_artifacts = json.loads((run / 'artifact_manifest.json').read_text())
    for name, item in original_artifacts.items():
        file = run / name
        assert file.resolve(strict=True).is_relative_to(run) and not file.is_symlink()
        assert file.stat().st_size == item['bytes'] and digest(file) == item['sha256'], name
    stage = root / 'operations' / 'captures' / ('job' + args.job)
    assert not stage.exists(), 'Existing staging view: investigate without rerunning'
    view = stage / root.name
    view.mkdir(parents=True)
    for name in frozen['files']:
        source, target = root / name, view / name
        assert not source.is_symlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    stable_copy(root / 'PAYLOAD_MANIFEST.json', view / 'PAYLOAD_MANIFEST.json', root)
    run_files = [p for p in run.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts]
    for file in run_files:
        stable_copy(file, view / file.relative_to(root), root)
    log_pattern = re.compile(r'(?:^|[-_.])' + re.escape(args.job) + r'(?:[._-]|$)')
    logs = [p for p in (root / 'logs').glob('*') if p.is_file() and log_pattern.search(p.name)]
    assert logs, 'No complete task log found'
    for file in logs:
        stable_copy(file, view / file.relative_to(root), root)
    for file in (root / 'operations').glob('*.json'):
        if file.is_file():
            stable_copy(file, view / file.relative_to(root), root)
    program_source = globals().get('_CAPTURE_SOURCE')
    assert isinstance(program_source, bytes)
    provenance = {
        'schema': 'completed_task_immutable_capture_v1',
        'original_root': str(root), 'temporary_view': str(view), 'job': args.job,
        'run_name': args.run_name, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scientific_files': 'Original frozen bytes; checked before and by unchanged frozen packer',
        'log_scope': 'All complete logs belonging to this exact array element; active sibling logs excluded',
        'logs': [str(p.relative_to(root)) for p in logs],
        'run_file_count': len(run_files), 'run_bytes': sum(p.stat().st_size for p in run_files),
        'program_source': program_source.decode(),
        'program_sha256': hashlib.sha256(program_source).hexdigest(),
        'frozen_packer_sha256': digest(root / 'pack_evidence.py'),
        'original_run_manifest_sha256': digest(run / 'artifact_manifest.json'),
    }
    provenance_path = view / 'operations' / ('CAPTURE_PROVENANCE_' + args.job + '.json')
    with provenance_path.open('x') as f:
        json.dump(provenance, f, indent=2)
    subprocess.run([sys.executable, '-B', str(view / 'pack_evidence.py'), '--root', str(view),
                    '--job', args.job, '--run-name', args.run_name], check=True)
    staged_archive = view / 'recovery' / archive.name
    staged_sidecar = view / 'recovery' / sidecar_path.name
    staged_receipt = json.loads(staged_sidecar.read_text())
    audit_archive(staged_archive, staged_receipt)
    verify_payload(root)
    shutil.copytree(view / 'recovery' / stem, directory)
    os.link(staged_archive, archive)
    staged_receipt.update({'archive': str(archive), 'capture_provenance_relative_path': str(provenance_path.relative_to(view)),
                           'immutable_view_used': True})
    with sidecar_path.open('x') as f:
        json.dump(staged_receipt, f, indent=2)
    shutil.rmtree(stage)
    show(sidecar_path, staged_receipt, False)


if __name__ == '__main__':
    main()
