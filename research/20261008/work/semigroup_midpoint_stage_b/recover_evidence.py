#!/usr/bin/env python3
"""Verify remote archive hash and exact internal inventory into a new directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

from pack_evidence import verify_complete_run


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def canonical(name):
    value = PurePosixPath(name)
    if value.is_absolute() or '..' in value.parts or str(value) != name or not value.parts:
        raise ValueError('Unsafe/noncanonical path: ' + name)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True, help='Observed on remote host')
    p.add_argument('--bytes', type=int, required=True)
    p.add_argument('--job-id', required=True)
    p.add_argument('--payload-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    assert re.fullmatch(r'[0-9]+(?:_[0-9]+)?', a.job_id)
    assert re.fullmatch(r'[0-9a-f]{64}', a.sha256)
    manifest = json.loads(a.payload_manifest.read_text())
    root_name = manifest['prefix']
    assert canonical(root_name).name == root_name
    assert root_name.startswith('20261004-midpoint-stage-b-')
    for name in manifest['files']:
        canonical(name)
    staging = a.output_dir.with_name(a.output_dir.name + '.partial')
    assert not a.output_dir.exists() and not staging.exists() and not a.report.exists()
    assert a.archive.stat().st_size == a.bytes and sha(a.archive) == a.sha256
    with tarfile.open(a.archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        assert len(names) == len(set(names)), 'Duplicate archive members'
        for member in members:
            path = canonical(member.name)
            assert member.isfile() and len(path.parts) >= 2 and path.parts[0] == root_name
        candidates = [m for m in members if m.name.startswith(root_name + '/recovery/') and m.name.endswith('/INVENTORY.json')]
        assert len(candidates) == 1
        inventory_member = candidates[0]
        inventory = json.load(tar.extractfile(inventory_member))
        assert inventory['schema'] == 'stage_b_evidence_v1' and inventory['job_id'] == a.job_id
        assert isinstance(inventory['allocation_job_id_raw'], str) and inventory['allocation_job_id_raw'].isdigit()
        assert inventory['root_name'] == root_name and inventory['payload_manifest_sha256'] == sha(a.payload_manifest)
        assert inventory['mode'] in {'preflight', 'run'} and inventory['capture_status'] in {'COMPLETE', 'FAILED'}
        run_relative = canonical(inventory['run_relative_path'])
        if inventory['mode'] == 'preflight':
            assert a.job_id.isdigit() and str(run_relative) == 'preflight/job' + a.job_id
        else:
            assert len(run_relative.parts) == 2 and run_relative.parts[0] == 'runs'
            assert re.fullmatch(r'(?:gdsg500|single500|midpoint500|gdsg333)-seed[012]-r1', run_relative.parts[1])
        assert root_name + '/' + inventory['inventory_relative_path'] == inventory_member.name
        expected = set()
        for item in inventory['files']:
            canonical(item['path'])
            expected.add(root_name + '/' + item['path'])
        assert len(expected) == len(inventory['files']) and set(names) == expected | {inventory_member.name}
        staging.mkdir(parents=True)
        for member in members:
            path = staging.joinpath(*canonical(member.name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, path.open('xb') as dest:
                shutil.copyfileobj(src, dest)
    recovered = staging / root_name
    for item in inventory['files']:
        path = recovered / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], item['path']
    assert sha(recovered / 'PAYLOAD_MANIFEST.json') == sha(a.payload_manifest)
    source_files = 0
    for name, spec in manifest['files'].items():
        if name != 'data-r1/development_data.npz':
            path = recovered / name
            assert path.resolve(strict=True).is_relative_to(recovered.resolve()) and path.is_file() and not path.is_symlink()
            assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], name
            source_files += 1
    if inventory['capture_status'] == 'COMPLETE':
        verify_complete_run(recovered, recovered / str(run_relative), inventory['mode'], a.job_id, manifest,
                            inventory['allocation_job_id_raw'])
    staging.rename(a.output_dir)
    report = {'status': 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED', 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'job_id': a.job_id, 'allocation_job_id_raw': inventory['allocation_job_id_raw'],
              'mode': inventory['mode'], 'capture_status': inventory['capture_status'],
              'archive': str(a.archive.resolve()), 'archive_sha256': a.sha256, 'archive_bytes': a.bytes,
              'inventory_files_verified': len(inventory['files']), 'frozen_source_and_metadata_files_verified': source_files,
              'source_archive_members': len(members), 'payload_manifest_sha256': sha(a.payload_manifest),
              'root': str((a.output_dir / root_name).resolve()),
              'run_root': str((a.output_dir / root_name / inventory['run_relative_path']).resolve()),
              'script_sha256': sha(__file__),
              'scope': 'Transfer and exact inventory integrity only; scientific and numerical checks are separate'}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    with a.report.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
