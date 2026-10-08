#!/usr/bin/env python3
"""Capture a terminal official DUE run without dependencies or duplicate MAT data.

This standard-library helper reads Slurm accounting itself. It does not submit,
cancel, resume, train, or run inference. Scientific metric recomputation remains
the responsibility of verify_official[_sg0]_results.py after recovery.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile


TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}
RECEIPTS = {
    ('default', 'preflight'): ('PREFLIGHT_PROFILE_PASSED.json', 'PROFILE_AND_ENGINEERING_CHECKS_PASSED'),
    ('default', 'full'): ('OFFICIAL_COMPLETE.json', 'OFFICIAL_500_EPOCHS_COMPLETE'),
    ('sg0', 'preflight'): ('SG0_PREFLIGHT_PROFILE_PASSED.json', 'SG0_PROFILE_AND_ENGINEERING_CHECKS_PASSED'),
    ('sg0', 'full'): ('OFFICIAL_SG0_COMPLETE.json', 'AUTHOR_SUPPORTED_SG0_500_EPOCHS_COMPLETE'),
}
WRAPPERS = {
    'default': ('run_official_observed.py', 'fa2b0951e283df3c6e20a5c4bce74a456504a8cd17428ad8c7ef2d72f6526395'),
    'sg0': ('run_official_sg0_observed.py', '77c5e1c4285ea04f1ede56ca127c8222537cc96daa9c98d2b4c462fa9192b434'),
}
OMIT_DIRS = {'matplotlib-cache', 'local-mpl-cache', 'local-import-deps', '__pycache__',
             '.pytest_cache', '.git', 'deps', 'wheelhouse'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def safe_file(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Unsafe relative path: ' + relative)
    path = root.joinpath(*rel.parts)
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError('Refusing symlink: ' + relative)
    if not path.resolve().is_relative_to(root) or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Expected regular file: ' + relative)
    return path


def accounting(job_id):
    fields = ['JobIDRaw', 'State%40', 'ExitCode', 'JobName%60', 'Elapsed', 'Start',
              'End', 'AllocTRES%200', 'NodeList%80', 'MaxRSS']
    command = ['sacct', '-j', job_id, '-n', '-P', '--format=' + ','.join(fields)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True)
    rows = [line.strip().split('|') for line in result.stdout.splitlines() if line.strip()]
    parent = [row for row in rows if row[0] == job_id]
    if len(parent) != 1 or len(parent[0]) < len(fields):
        raise RuntimeError('sacct must return exactly one complete parent-job row')
    row = parent[0]
    state = row[1].split()[0].rstrip('+')
    if state not in TERMINAL:
        raise RuntimeError('Job is not terminal; refusing even partial capture: ' + row[1])
    return {'command': command, 'stdout': result.stdout, 'stderr': result.stderr,
            'parent': dict(zip([x.split('%')[0] for x in fields], row)),
            'state': state, 'exit_code': row[2]}


def inspect_run(root, run_name, job_id, variant, mode, acct):
    checks, errors = [], []
    def check(name, action):
        try:
            value = action()
            if value is False:
                raise ValueError('condition is false')
            checks.append(name)
            return value
        except Exception as error:
            errors.append({'check': name, 'error': type(error).__name__ + ': ' + str(error)})
            return None
    def read(relative):
        return json.loads(safe_file(root, relative).read_text())
    run = 'runs/' + run_name + '/'
    check('parent_slurm_completed_zero', lambda: acct['state'] == 'COMPLETED' and acct['exit_code'] == '0:0')
    source = check('source_manifest', lambda: read('operations/official_source_manifest.json'))
    if source is not None:
        check('official_commit_and_75_sources', lambda: source['commit'] == '1e629104a12a9c3e3064996cbd4e301cec641ad8' and len(source['files']) == 75)
        for name, digest in source['files'].items():
            check('source_hash:' + name, lambda n=name, d=digest: sha(safe_file(root, 'due/' + n)) == d)
    for relative in ('data/manifest.json', 'data/download_manifest.json', 'operations/setup_complete.json'):
        check(relative, lambda r=relative: read(r))
    receipt_name, status = RECEIPTS[variant, mode]
    receipt = check('completion_receipt', lambda: read(run + receipt_name))
    environment = check('environment', lambda: read(run + 'audit/environment.json'))
    plan = check('model_and_training_plan', lambda: read(run + 'audit/model_and_training_plan.json'))
    check('no_failure_receipt', lambda: not (root / run / 'FAILURE.json').exists())
    wrapper, digest = WRAPPERS[variant]
    check('frozen_wrapper_hash', lambda: sha(safe_file(root, 'operations/' + wrapper)) == digest)
    epochs = 1 if mode == 'preflight' else 500
    if environment is not None:
        check('receipt_job_identity', lambda: str(environment['slurm_job_id']) == job_id)
        check('environment_mode', lambda: environment['mode'] == mode)
        check('environment_v100', lambda: 'V100' in environment['gpu'])
        check('recorded_wrapper_hash', lambda: environment['wrapper_sha256'] == digest)
        check('recorded_source_commit', lambda: environment['source_commit'] == source['commit'])
        check('recorded_source_hash', lambda: environment['source_manifest_sha256'] == sha(safe_file(root, 'operations/official_source_manifest.json')))
        config = environment.get('effective_config' if variant == 'sg0' else 'official_config', {})
        check('configuration', lambda: config['seed'] == 0 and config['training']['epochs'] == 500
              and config['training']['batch_size'] == 20 and config['training']['sg_pairing'] == (0 if variant == 'sg0' else 1))
        if variant == 'sg0':
            check('sg0_variant', lambda: environment['variant'] == 'author_supported_sg_pairing_0')
    if receipt is not None:
        check('completion_status', lambda: receipt['status'] == status)
        check('complete_protocol_flag', lambda: receipt['complete_official_reproduction'] is (mode == 'full'))
        check('epoch_count', lambda: receipt['configured_epochs'] == 500 and receipt['completed_epochs'] == epochs
              and receipt['training']['completed_epochs'] == epochs)
        check('actual_update_count', lambda: receipt['training']['optimizer_updates'] == epochs * 125)
        check('changed_parameters', lambda: receipt['training']['initial_parameter_sha256'] != receipt['training']['final_parameter_sha256'])
        check('batch_plan', lambda: plan['official_batch_size'] == 20 and plan['batches_per_epoch'] == 125
              and plan['trainX_shape'][0] == 2500 and plan['configured_epochs'] == 500)
        check('final_checkpoint_hash', lambda: sha(safe_file(root, run + 'audit/' + receipt['training']['final_network_file'])) == receipt['training']['final_network_sha256'])
        for kind in ('final_epoch', 'best_training_objective'):
            check(kind + '_replay', lambda k=kind: receipt['checkpoints'][k]['finite'] is True
                  and receipt['checkpoints'][k]['reload_replay_passed'] is True)
        check('best_checkpoint_hash', lambda: sha(safe_file(root, run + 'official_workdir/model/model')) == receipt['checkpoints']['best_training_objective']['sha256'])
        check('batch100_prediction_attempt', lambda: receipt['prediction']['prediction_batch_size'] == 100
              and receipt['prediction']['prediction_steps'] == 99 and receipt['prediction']['engineering_checks_passed'] is True)
        if mode == 'preflight':
            check('isolated_resumed_update', lambda: receipt['checkpoints']['isolated_resumed_update']['passed'] is True
                  and receipt['checkpoints']['isolated_resumed_update']['parameters_changed'] is True
                  and receipt['checkpoints']['isolated_resumed_update']['cpu_cuda_numpy_python_rng_restored'] is True)
        else:
            required = {'model', 'training_history.csv', 'training_history.png', 'pred.mat', 'rel_err.csv', 'rel_err.png'}
            check('official_artifact_set', lambda: set(receipt['official_artifact_sha256']) == required)
            for name, expected in receipt.get('official_artifact_sha256', {}).items():
                check('official_artifact_hash:' + name,
                      lambda n=name, d=expected: sha(safe_file(root, run + 'official_workdir/model/' + n)) == d)
        check('saved_training_state', lambda: safe_file(root, run + 'audit/' + ('epoch1_profile' if mode == 'preflight' else 'epoch500_final') + '_training_state.pt'))
        check('epoch_event_count', lambda: len([x for x in safe_file(root, run + 'audit/epoch_events.jsonl').read_text().splitlines() if x.strip()]) == epochs)
    return {'checks_passed': checks, 'errors': errors, 'completion_checks_passed': not errors,
            'completion_receipt': receipt_name, 'prediction_quality': receipt.get('prediction', {}).get('prediction_quality_status') if receipt else None,
            'scope': 'Process and artifact receipt validation; independent numerical metric recomputation is performed after recovery.'}


def collect(root, run_name, job_id):
    files, omitted, errors = set(), [], []
    def add(path):
        rel = str(path.relative_to(root))
        try:
            files.add(safe_file(root, rel))
        except Exception as error:
            errors.append({'path': rel, 'error': str(error)})
    run = root / 'runs' / run_name
    if not run.is_dir() or run.is_symlink():
        errors.append({'path': str(run), 'error': 'run directory missing or symlink'})
    else:
        for parent, directories, names in os.walk(run, followlinks=False):
            for name in list(directories):
                path = Path(parent) / name
                if name in OMIT_DIRS or path.is_symlink():
                    directories.remove(name)
                    omitted.append(str(path.relative_to(root)))
            for name in names:
                path = Path(parent) / name
                if path.is_symlink() or path.suffix == '.pyc':
                    omitted.append(str(path.relative_to(root)))
                elif name in {'VorticityOSG_train.mat', 'VorticityOSG_test.mat'}:
                    omitted.append(str(path.relative_to(root)))
                else:
                    add(path)
    logs = list((root / 'runs').glob('*-' + job_id + '.log'))
    if not logs:
        errors.append({'path': 'runs/*-' + job_id + '.log', 'error': 'Slurm stdout log missing'})
    for path in logs:
        add(path)
    # Top-level operation files contain scripts, manifests and submission receipts.
    # Never recurse into local validation environments or dependency overlays.
    for path in sorted((root / 'operations').iterdir()):
        if path.is_file() and not path.is_symlink() and path.suffix in {
                '.py', '.json', '.sh', '.md', '.txt', '.yaml', '.yml', '.lock', '.diff', '.log', '.tsv', '.csv', '.toml'}:
            add(path)
    for path in sorted((root / 'data').glob('*.json')):
        add(path)
    for pattern in ('*manifest*.json', '*PROTOCOL*.md', '*AUDIT*.md'):
        for path in root.glob(pattern):
            add(path)
    try:
        manifest = json.loads((root / 'operations/official_source_manifest.json').read_text())
        for name in manifest['files']:
            add(root / 'due' / name)
    except Exception as error:
        errors.append({'path': 'official_source_manifest.json', 'error': str(error)})
    return sorted(files), sorted(omitted), errors


def verify_archive(archive, inventory, prefix):
    """Read and hash every tar member; reject omissions, additions and symlinks."""
    expected = {prefix + '/' + row['path']: row for row in inventory['files']}
    inventory_member = prefix + '/' + inventory['inventory_relative_path']
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            if not member.isfile() or member.name in seen:
                raise ValueError('Nonregular or duplicate archive member: ' + member.name)
            seen.add(member.name)
            stream = tar.extractfile(member)
            if member.name == inventory_member:
                if json.load(stream) != inventory:
                    raise ValueError('Embedded inventory mismatch')
                continue
            if member.name not in expected or member.size != expected[member.name]['bytes']:
                raise ValueError('Unexpected archive member or size: ' + member.name)
            h = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != expected[member.name]['sha256']:
                raise ValueError('Archive member hash mismatch: ' + member.name)
    if seen != set(expected) | {inventory_member}:
        raise ValueError('Archive file set is incomplete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--variant', choices=['default', 'sg0'], required=True)
    parser.add_argument('--mode', choices=['preflight', 'full'], required=True)
    parser.add_argument('--partial', action='store_true', help='Explicitly preserve incomplete/failed terminal-run evidence; always labeled PARTIAL')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,99}', args.run_name) or not args.job_id.isdigit():
        parser.error('Invalid run name or numeric job ID')
    root = args.root.resolve(strict=True)
    acct = accounting(args.job_id)
    validation = inspect_run(root, args.run_name, args.job_id, args.variant, args.mode, acct)
    files, omitted, collection_errors = collect(root, args.run_name, args.job_id)
    validation['errors'].extend(collection_errors)
    validation['completion_checks_passed'] = not validation['errors']
    if validation['errors'] and not args.partial:
        print(json.dumps(validation, indent=2, allow_nan=False))
        raise SystemExit('Completion validation failed. No archive created; use --partial explicitly to retain terminal failure evidence.')
    outcome = 'PARTIAL' if args.partial else 'COMPLETE'
    stem = args.run_name + '-job' + args.job_id + '-' + outcome
    recovery = root / 'recovery'
    recovery.mkdir(exist_ok=True)
    stage = recovery / stem
    archive = recovery / (stem + '.tar.gz')
    archive_receipt = recovery / (stem + '.archive.json')
    identity = {'run_name': args.run_name, 'job_id': args.job_id, 'variant': args.variant,
                'mode': args.mode, 'capture_status': outcome}
    if archive.exists() or archive_receipt.exists() or stage.exists():
        if not archive.is_file() or not archive_receipt.is_file() or not (stage / 'INVENTORY.json').is_file():
            raise SystemExit('Incomplete existing recovery artifacts; retained unchanged. Inspect before retrying.')
        receipt = json.loads(archive_receipt.read_text())
        if any(receipt.get(k) != v for k, v in identity.items()) or sha(archive) != receipt['sha256'] or archive.stat().st_size != receipt['bytes']:
            raise SystemExit('Existing archive identity/hash mismatch; refusing overwrite.')
        inventory = json.loads((stage / 'INVENTORY.json').read_text())
        verify_archive(archive, inventory, root.name)
        # Existing captured files must still match. New operation files do not
        # retroactively change the content of an earlier verified capture.
        for row in inventory['files']:
            if sha(safe_file(root, row['path'])) != row['sha256']:
                raise SystemExit('Source changed since capture; refusing reuse/overwrite: ' + row['path'])
        print(json.dumps({**receipt, 'reused_verified_archive': True}, indent=2))
        return
    stage.mkdir()  # Exclusive claim. A failed capture is retained for inspection.
    (stage / 'sacct.txt').write_text(acct['stdout'])
    write_new(stage / 'accounting.json', acct)
    write_new(stage / 'validation.json', {**identity, **validation})
    files += [stage / 'sacct.txt', stage / 'accounting.json', stage / 'validation.json']
    captured_at = datetime.now(timezone.utc).isoformat()
    inventory = {**identity, 'captured_at_utc': captured_at,
        'inventory_relative_path': str((stage / 'INVENTORY.json').relative_to(root)),
        'files': [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(files))],
        'omitted_paths': omitted, 'omitted_categories': ['original MAT datasets', 'deps', 'wheelhouse', 'local dependency environments', 'caches', 'data symlinks'],
        'self_hash_note': 'Inventory omits its own recursive hash; archive receipt hashes the complete tar including inventory.'}
    write_new(stage / 'INVENTORY.json', inventory)
    # x:gz is exclusive: no archive is ever silently replaced.
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tar:
        for row in inventory['files']:
            path = safe_file(root, row['path'])
            tar.add(path, arcname=root.name + '/' + row['path'], recursive=False)
        tar.add(stage / 'INVENTORY.json', arcname=root.name + '/' + inventory['inventory_relative_path'], recursive=False)
    verify_archive(archive, inventory, root.name)
    receipt = {**identity, 'captured_at_utc': captured_at, 'file': archive.name,
        'archive_path': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
        'inventory_files': len(inventory['files']), 'archive_members': len(inventory['files']) + 1,
        'inventory_sha256': sha(stage / 'INVENTORY.json'), 'slurm_parent': acct['parent'],
        'completion_checks_passed': validation['completion_checks_passed'],
        'prediction_quality': validation['prediction_quality'], 'reused_verified_archive': False}
    write_new(archive_receipt, receipt)
    print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
