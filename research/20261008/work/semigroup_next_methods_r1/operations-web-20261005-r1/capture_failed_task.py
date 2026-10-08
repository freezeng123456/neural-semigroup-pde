#!/usr/bin/env python3
"""Preserve every file and the complete log of a terminal failed task."""
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(archive, sidecar):
    assert archive.stat().st_size == sidecar['bytes'] and sha(archive) == sidecar['sha256']
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == sidecar['archive_members'] == len({m.name for m in members})
        for m in members:
            p = PurePosixPath(m.name)
            assert m.isfile() and not p.is_absolute() and '..' not in p.parts and str(p) == m.name
        prefix = '20261005-next-methods-r1/'
        name = prefix + sidecar['inventory_relative_path']
        inv = json.load(tar.extractfile(name))
        assert inv['capture'] == 'FAILED' and inv['job'] == sidecar['job']
        assert {m.name for m in members} == {prefix + k for k in inv['files']} | {name}
        for path, item in inv['files'].items():
            member = tar.getmember(prefix + path)
            h = hashlib.sha256()
            with tar.extractfile(member) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            assert member.size == item['bytes'] and h.hexdigest() == item['sha256'], path


def copy(source, target, root):
    assert source.is_file() and not source.is_symlink() and source.resolve().is_relative_to(root)
    before = source.stat()
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    shutil.copy2(source, target)
    after = source.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert target.stat().st_size == after.st_size and sha(source) == sha(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--run-name', required=True)
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    assert root.name == '20261005-next-methods-r1' and re.fullmatch(r'\d+_\d+', args.job)
    assert re.fullmatch(r'[a-z0-9]+-seed[012]-r1', args.run_name)
    sys.path.insert(0, str(root))
    from launch_task import verify_payload
    manifest = verify_payload(root)
    assert sha(root / 'PAYLOAD_MANIFEST.json') == 'ffbe1bb1b40c46ba4cd41a9fc1b94ea0dbce130c7ee8c646203fb4f16543d8a1'
    run = root / 'runs' / args.run_name
    task = json.loads((root / 'matrix.json').read_text())['tasks'][int(args.job.split('_')[1])]
    assert task['run_name'] == args.run_name
    raw = subprocess.check_output(['sacct', '-j', args.job, '-nP', '--format=JobID%40,State%30,ExitCode'], text=True)
    rows = [r.split('|') for r in raw.splitlines() if r.split('|')[0] == args.job]
    assert len(rows) == 1 and rows[0][1] in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED')
    failure = json.loads((run / 'FAILURE.json').read_text())
    assert failure['status'] == 'NEXT_METHODS_FAILED'
    stem = args.run_name + '-job' + args.job + '-FAILED'
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    archive, sidecar, directory = recovery / (stem + '.tar.gz'), recovery / (stem + '.archive.json'), recovery / stem
    if archive.exists() or sidecar.exists() or directory.exists():
        assert archive.is_file() and sidecar.is_file() and directory.is_dir(), 'Partial capture: preserve and investigate'
        receipt = json.loads(sidecar.read_text())
        assert receipt['job'] == args.job
        audit(archive, receipt)
    else:
        stage = root / 'operations' / 'captures' / ('job' + args.job + '-failed')
        assert not stage.exists(), 'Existing failed staging view: preserve and investigate'
        view = stage / root.name
        view.mkdir(parents=True)
        for name in manifest['files']:
            source, target = root / name, view / name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not source.is_symlink()
            os.link(source, target)
        copy(root / 'PAYLOAD_MANIFEST.json', view / 'PAYLOAD_MANIFEST.json', root)
        files = [f for f in run.rglob('*') if f.is_file() and 'matplotlib-cache' not in f.parts]
        for file in files:
            copy(file, view / file.relative_to(root), root)
        pattern = re.compile(r'(?:^|[-_.])' + re.escape(args.job) + r'(?:[._-]|$)')
        logs = [f for f in (root / 'logs').glob('*') if f.is_file() and pattern.search(f.name)]
        assert logs
        for file in logs + [f for f in (root / 'operations').glob('*.json') if f.is_file()]:
            copy(file, view / file.relative_to(root), root)
        source = globals()['_FAILED_CAPTURE_SOURCE']
        provenance = {'schema': 'failed_task_immutable_capture_v1', 'job': args.job, 'original_root': str(root),
                      'created_at_utc': datetime.now(timezone.utc).isoformat(), 'failure': failure,
                      'run_file_count': len(files), 'run_bytes': sum(f.stat().st_size for f in files),
                      'logs': [str(f.relative_to(root)) for f in logs], 'log_scope': 'Complete exact task logs; active sibling logs excluded',
                      'program_source': source.decode(), 'program_sha256': hashlib.sha256(source).hexdigest(),
                      'frozen_packer_sha256': sha(root / 'pack_evidence.py')}
        provenance_path = view / 'operations' / ('FAILED_CAPTURE_PROVENANCE_' + args.job + '.json')
        with provenance_path.open('x') as f:
            json.dump(provenance, f, indent=2)
        subprocess.run([sys.executable, '-B', str(view / 'pack_evidence.py'), '--root', str(view),
                        '--job', args.job, '--run-name', args.run_name, '--failed'], check=True)
        staged = view / 'recovery' / archive.name
        receipt = json.loads((view / 'recovery' / sidecar.name).read_text())
        audit(staged, receipt)
        verify_payload(root)
        shutil.copytree(view / 'recovery' / stem, directory)
        os.link(staged, archive)
        receipt.update({'archive': str(archive), 'immutable_view_used': True,
                        'capture_provenance_relative_path': str(provenance_path.relative_to(view))})
        with sidecar.open('x') as f:
            json.dump(receipt, f, indent=2)
        shutil.rmtree(stage)
    print('FAILED_EVIDENCE VERIFIED', flush=True)
    for label, path in (('ARCHIVE', archive), ('SIDECAR', sidecar)):
        digest = sha(path)
        print(label, path.name, 'BYTES', path.stat().st_size, flush=True)
        print(label + '_SHA_HEAD', digest[:32], flush=True)
        print(label + '_SHA_TAIL', digest[32:], flush=True)
    print('ARCHIVE_MEMBERS', receipt['archive_members'], flush=True)


if __name__ == '__main__':
    main()
