#!/usr/bin/env python3
"""Verify remote SHA/bytes, scan a plain-file tar, and stream to a new local directory.

Run beside pack_evaluation_evidence.py. Never extractall, follow links, load models,
or overwrite evidence. A failed verification leaves its exclusive .partial directory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile

from pack_evaluation_evidence import (GRID, canonical, decode_json, inspect, read_json,
    regular, require, scheduler_row, sha, valid_spec, write_json)


def member_path(member, prefix):
    path = canonical(member.name)
    require(member.isfile() and not member.issparse() and member.size >= 0,
            'Only ordinary regular-file members are accepted: ' + member.name)
    require(len(path.parts) >= 2 and path.parts[0] == prefix, 'Wrong archive prefix')
    return '/'.join(path.parts[1:])


def scan_archive(archive, prefix, job_id, attempt_id, deploy_sha):
    """First streaming pass validates paths and reads only the bounded inventory."""
    names, sizes, inventory, inventory_sha = set(), {}, None, None
    expected_inventory = f'recovery/{attempt_id}-job{job_id}/INVENTORY.json'
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            name = member_path(member, prefix)
            require(name not in names, 'Duplicate archive member: ' + name)
            names.add(name); sizes[name] = member.size
            if name == expected_inventory:
                require(member.size <= 16 * 1024 * 1024, 'Inventory JSON is unexpectedly large')
                with tar.extractfile(member) as source:
                    raw_inventory = source.read()
                    inventory = decode_json(raw_inventory)
                    inventory_sha = hashlib.sha256(raw_inventory).hexdigest()
    require(inventory is not None, 'Expected inventory is missing')
    require(inventory['schema'] == 'stage_b_evaluation_evidence_v1' and inventory['root_name'] == prefix
            and inventory['job_id'] == job_id and inventory['attempt_id'] == attempt_id
            and inventory['inventory_relative_path'] == expected_inventory
            and inventory['deploy_manifest_sha256'] == deploy_sha, 'Inventory identity mismatch')
    require(inventory['capture_status'] in {'COMPLETE', 'FAILED'}, 'Unknown capture status')
    require(isinstance(inventory['files'], list) and inventory['files'], 'Empty inventory')
    specs = {}
    for spec in inventory['files']:
        name = str(canonical(spec['path']))
        require(name not in specs and name != expected_inventory and valid_spec(spec), 'Duplicate/invalid inventory entry')
        require(sizes.get(name) == spec['bytes'], 'Declared archive size differs: ' + name)
        specs[name] = spec
    require(names == set(specs) | {expected_inventory}, 'Archive membership differs from exact inventory')
    require('DEPLOY_MANIFEST.json' in specs and specs['DEPLOY_MANIFEST.json']['sha256'] == deploy_sha,
            'Trusted deployment manifest is missing/different')
    for name in names:
        parts = canonical(name).parts
        require(all('/'.join(parts[:index]) not in names for index in range(1, len(parts))), 'File/directory path collision')
    capture_path = f'recovery/{attempt_id}-job{job_id}/CAPTURE_REPORT.json'
    sacct_path = f'recovery/{attempt_id}-job{job_id}/evaluation_sacct.txt'
    require(inventory['capture_report_relative_path'] == capture_path and capture_path in specs
            and inventory['sacct_relative_path'] == sacct_path and sacct_path in specs, 'Missing/moved capture evidence')
    # Bind the self-excluded inventory to exactly the bytes seen on pass one.
    specs[expected_inventory] = {'bytes': sizes[expected_inventory], 'sha256': inventory_sha}
    return inventory, specs


def recover(archive, expected_sha, expected_bytes, job_id, attempt_id, deploy_manifest, output, report_path):
    require(re.fullmatch(r'[0-9a-f]{64}', expected_sha) and type(expected_bytes) is int and expected_bytes > 0, 'Invalid remote SHA/bytes')
    require(re.fullmatch(r'\d+', job_id) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', attempt_id), 'Invalid job/attempt')
    require(not archive.is_symlink() and not deploy_manifest.is_symlink(), 'Input symlinks are forbidden')
    archive, deploy_manifest = archive.resolve(strict=True), deploy_manifest.resolve(strict=True)
    require(archive.is_file() and deploy_manifest.is_file(), 'Input must be a file')
    # Resolve the parent only: never turn an existing final-component symlink into a target.
    output = output.parent.resolve() / output.name
    report_path = report_path.parent.resolve() / report_path.name
    staging = output.with_name(output.name + '.partial')
    require(not any(path.exists() or path.is_symlink() for path in (output, staging, report_path)), 'Recovery/report already exists; no overwrite')
    require(not report_path.is_relative_to(output) and not report_path.is_relative_to(staging), 'Report must be outside output/staging')
    require(not archive.is_relative_to(output) and not deploy_manifest.is_relative_to(output), 'Output overlaps input')
    deploy = read_json(deploy_manifest)
    prefix = str(canonical(deploy['prefix']))
    require('/' not in prefix and re.fullmatch(r'20261004-midpoint-stage-b-evaluation-r\d+', prefix)
            and deploy['schema'] == 'stage_b_evaluation_deployment_v1', 'Unexpected deployment manifest')
    require(archive.stat().st_size == expected_bytes and sha(archive) == expected_sha, 'Remote archive SHA/bytes mismatch')
    inventory, specs = scan_archive(archive, prefix, job_id, attempt_id, sha(deploy_manifest))
    staging.mkdir(parents=True, exist_ok=False)
    try:
        copied = set()
        with tarfile.open(archive, 'r|gz') as tar:
            for member in tar:
                name = member_path(member, prefix)
                require(name in specs and name not in copied and member.size == specs[name]['bytes'], 'Archive changed since scan')
                destination = staging / prefix / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                digest, count = hashlib.sha256(), 0
                with tar.extractfile(member) as source, destination.open('xb') as target:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        target.write(block); digest.update(block); count += len(block)
                require(count == specs[name]['bytes'] and digest.hexdigest() == specs[name]['sha256'], 'File checksum differs: ' + name)
                copied.add(name)
        require(copied == set(specs), 'Not all inventoried files recovered')
        require(archive.stat().st_size == expected_bytes and sha(archive) == expected_sha, 'Archive changed during recovery')
        root = staging / prefix
        require(sha(regular(root, 'DEPLOY_MANIFEST.json')) == sha(deploy_manifest), 'Deployment identity changed')
        recorded = read_json(regular(root, inventory['capture_report_relative_path']))
        raw = regular(root, inventory['sacct_relative_path']).read_bytes()
        row = scheduler_row(raw, job_id)
        recomputed = inspect(root, attempt_id, job_id, row, remote_root=inventory['remote_root'])
        require(recorded['capture_status'] == recomputed['capture_status'] == inventory['capture_status'], 'Capture status does not match recovered evidence')
        require(recorded['job_id'] == job_id and recorded['attempt_id'] == attempt_id
                and recorded['remote_root'] == inventory['remote_root'] and recorded['scheduler_row'] == row, 'Capture receipt/accounting mismatch')
        require(len(recorded['results']) == 12 and [(r['arm'], r['seed'], r['run_name']) for r in recorded['results']] == GRID, 'Capture dropped/reordered an identity')
        for original, current in zip(recorded['results'], recomputed['results']):
            for key in ('arm', 'seed', 'run_name', 'relative_path', 'status', 'quality_status', 'report_sha256', 'artifact_manifest_sha256', 'parameter_sha256'):
                require(original.get(key) == current.get(key), 'Recovered per-result status differs: ' + current['run_name'])
        staging.rename(output)
        report = {'schema': 'stage_b_evaluation_recovery_v1', 'status': 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED',
                  'capture_status': inventory['capture_status'], 'job_id': job_id, 'attempt_id': attempt_id,
                  'archive': str(archive), 'archive_sha256': expected_sha, 'archive_bytes': expected_bytes,
                  'root': str(output / prefix), 'inventory_files_verified': len(inventory['files']),
                  'archive_members_verified': len(specs), 'results': recomputed['results'], 'issues': recomputed['issues'],
                  'deploy_manifest_sha256': sha(deploy_manifest), 'recovery_script_sha256': sha(__file__),
                  'pack_helper_sha256': sha(Path(__file__).with_name('pack_evaluation_evidence.py')),
                  'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'optimizer_updates': 0, 'network_calls': 0,
                  'scope': 'Transfer and engineering integrity only. FAILED captures remain failed; scientific metric checks are separate.'}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(report_path, report)
        return report
    except BaseException as error:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if not report_path.exists():
            write_json(report_path, {'status': 'RECOVERY_FAILED', 'error': repr(error), 'partial_directory': str(staging),
                       'archive': str(archive), 'job_id': job_id, 'attempt_id': attempt_id})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True, help='SHA independently observed on the remote host')
    p.add_argument('--bytes', type=int, required=True, help='Bytes independently observed on the remote host')
    p.add_argument('--job-id', required=True)
    p.add_argument('--attempt-id', required=True)
    p.add_argument('--deploy-manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(recover(a.archive, a.sha256, a.bytes, a.job_id, a.attempt_id,
                            a.deploy_manifest, a.output_dir, a.report), indent=2))


if __name__ == '__main__':
    main()
