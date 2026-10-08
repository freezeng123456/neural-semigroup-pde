#!/usr/bin/env python3
"""Verify a remotely hashed DUE evidence archive and extract to a fresh directory.

This checks transfer and inventory integrity only. Run the independent numerical
verifier and checkpoint replay separately after this script succeeds.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True, help='Digest observed on the remote host')
    p.add_argument('--bytes', type=int, required=True, help='Size observed on the remote host')
    p.add_argument('--job-id', required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--variant', choices=['default', 'sg0'], required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    root_name = '20261004-due-vorticity-1e62910-r1'
    stem = f'{a.run_name}-job{a.job_id}-COMPLETE'
    if not re.fullmatch(r'[a-z0-9-]+', a.run_name) or not a.job_id.isdigit():
        raise ValueError('Invalid run identity')
    if not re.fullmatch(r'[a-f0-9]{64}', a.sha256):
        raise ValueError('Expected a lowercase SHA-256 digest')
    staging = a.output_dir.with_name(a.output_dir.name + '.partial')
    if a.output_dir.exists() or staging.exists() or a.report.exists():
        raise FileExistsError('Output, staging, or report already exists; inspect before retrying')
    if a.archive.stat().st_size != a.bytes or sha(a.archive) != a.sha256:
        raise ValueError('Downloaded archive does not match remote bytes and SHA-256')
    inventory_rel = f'recovery/{stem}/INVENTORY.json'
    inventory_member = f'{root_name}/{inventory_rel}'
    with tarfile.open(a.archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        for m in members:
            parts = PurePosixPath(m.name)
            if (parts.is_absolute() or '..' in parts.parts or str(parts) != m.name or
                    len(parts.parts) < 2 or parts.parts[0] != root_name or not m.isfile()):
                raise ValueError(f'Unexpected archive member: {m.name}')
        inv = json.load(tar.extractfile(inventory_member))
        identity = (inv['run_name'], str(inv['job_id']), inv['variant'], inv['mode'], inv['capture_status'])
        if identity != (a.run_name, a.job_id, a.variant, 'full', 'COMPLETE'):
            raise ValueError('Inventory run identity or completion status mismatch')
        if inv['inventory_relative_path'] != inventory_rel:
            raise ValueError('Inventory identity mismatch')
        for entry in inv['files']:
            relative = PurePosixPath(entry['path'])
            if (relative.is_absolute() or '..' in relative.parts or not relative.parts or
                    str(relative) != entry['path']):
                raise ValueError('Unsafe or noncanonical inventory path: ' + entry['path'])
        expected = {f"{root_name}/{entry['path']}" for entry in inv['files']}
        if len(expected) != len(inv['files']) or set(names) != expected | {inventory_member}:
            raise ValueError('Inventory does not exactly cover archive members')
        staging.mkdir(parents=True)
        for m in members:
            path = staging.joinpath(*PurePosixPath(m.name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src, path.open('xb') as dest:
                shutil.copyfileobj(src, dest)
    for entry in inv['files']:
        path = staging / root_name / entry['path']
        if path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError(f"Extracted inventory mismatch: {entry['path']}")
    staging.rename(a.output_dir)
    report = {
        'status': 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED',
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': a.job_id, 'run_name': a.run_name, 'variant': a.variant,
        'archive': str(a.archive.resolve()), 'archive_sha256': a.sha256,
        'archive_bytes': a.bytes, 'inventory_files_verified': len(inv['files']),
        'source_archive_members': len(members),
        'run_root': str((a.output_dir / root_name / 'runs' / a.run_name).resolve()),
        'script_sha256': sha(Path(__file__)),
        'scope': 'Transfer and inventory integrity; numerical verification and replay are separate',
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
