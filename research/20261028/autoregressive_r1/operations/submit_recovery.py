#!/usr/bin/env python3
"""Submit exactly one dependency-bound replay/compact-recovery job; never train."""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
from launch import ROOT, verify
from scheduler_limits import resources

EXPECTED_ROOT = Path('/data02/run01/scv7tsq/neural-semigroup/20261028-autoregressive-r1')
OLD = Path('/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1')
MAIN_JOB = '2420813'
PREFLIGHT_JOB = '2420812'
PAYLOAD_NAMES = ('operations/submit_recovery.py', 'operations/recover.sh',
                 'operations/replay_and_aggregate.py', 'operations/pack_compact.py',
                 'operations/launch.py', 'operations/scheduler_limits.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def common_identity():
    if getpass.getuser() != 'scv7tsq' or ROOT != EXPECTED_ROOT:
        raise ValueError('Recovery is restricted to the authorized account and experiment root')
    verify()
    core_sha = sha(ROOT / 'CORE_MANIFEST.json')
    receipt_path = ROOT / 'operations/full_submission_receipt.json'
    main = json.loads(receipt_path.read_text())
    if main.get('schema') != 'autoregressive_submission_v1' or main.get('stage') != 'full' or main.get('status') != 'SUBMITTED' or str(main.get('job_id')) != MAIN_JOB:
        raise ValueError('Actual formal submission receipt is not the authorized array')
    command = main['command']
    if command[0] != 'sbatch' or '--array=0-5%2' not in command or command[-3:] != [str(ROOT / 'operations/run.sh'), str(ROOT), 'full']:
        raise ValueError('Main job does not match the frozen six-task launch')
    if main['core_manifest_sha256'] != core_sha or str(main['preflight']['job']) != PREFLIGHT_JOB:
        raise ValueError('Main submission core/preflight provenance differs')
    return {'main_job_id': MAIN_JOB, 'preflight_job_id': PREFLIGHT_JOB,
            'core_manifest_sha256': core_sha, 'main_submission_receipt_sha256': sha(receipt_path),
            'matrix_sha256': sha(ROOT / 'matrix.json')}


def scheduler_snapshot():
    command = ['sacct', '-nP', '-j', MAIN_JOB, '--format=JobIDRaw,State,ExitCode']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Cannot inspect main array accounting: ' + result.stderr)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    matched = [row for row in rows if row[0] == MAIN_JOB or re.fullmatch(MAIN_JOB + r'_\d+', row[0])]
    if not matched:
        raise ValueError('Main array has no actual Slurm accounting rows')
    return {'command': command, 'stdout': result.stdout, 'returncode': result.returncode}


def require_main_not_live():
    # A completed array may already be purged from the controller. Query the
    # user's live queue, then match locally; an exact --jobs filter can return
    # Invalid job id for that normal lifecycle even while sacct has the record.
    command = ['squeue', '--array', '--noheader', '--user=scv7tsq', '--format=%i|%T']
    active = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if active.returncode:
        raise RuntimeError('Cannot inspect current user queue before recovery')
    matched = []
    for line in active.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 2:
            raise ValueError('Unexpected current-user queue columns')
        job_id = fields[0].strip()
        if job_id == MAIN_JOB or job_id.startswith(MAIN_JOB + '_'):
            matched.append(line)
    if matched:
        raise RuntimeError('Main array is still live; recovery must wait for terminal accounting')
    return {'command': command, 'returncode': active.returncode,
            'matched_main_rows': matched}


def submit(a):
    identity = common_identity(); operations = ROOT / 'operations'
    intent_path = operations / 'recovery_submission_intent.json'
    receipt_path = operations / 'recovery_submission_receipt.json'
    if intent_path.exists() or receipt_path.exists():
        raise FileExistsError('Existing recovery submission intent/receipt: reconcile actual Slurm, never resubmit automatically')
    main_snapshot = scheduler_snapshot()
    evidence = []; quota = resources('validate', evidence)
    payload = {name: {'bytes': (ROOT / name).stat().st_size, 'sha256': sha(ROOT / name)} for name in PAYLOAD_NAMES}
    logs = ROOT / 'logs'; logs.mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--partition=gpu', '--account=scv7tsq', '--gpus=1', '--ntasks=1',
        '--cpus-per-task=10', '--time=00:30:00', '--dependency=afterany:' + MAIN_JOB,
        '--job-name=sg-ar-recovery', '--chdir=' + str(ROOT),
        '--output=' + str(logs / 'recovery-%j.log'), str(operations / 'recover.sh'), str(ROOT)]
    record = {'schema': 'autoregressive_recovery_submission_v1', 'stage': 'recovery',
        'requested_at_utc': datetime.now(timezone.utc).isoformat(), 'identity': identity,
        'command': command, 'payload': payload, 'resources': quota,
        'scheduler_evidence': evidence, 'main_accounting_snapshot': main_snapshot,
        'scope': 'One allocated-GPU checkpoint replay and compact packaging job; no optimizer updates or new experiments'}
    if a.audit_only:
        print(json.dumps({'status': 'RECOVERY_AUDIT_PASS', 'command': command, 'identity': identity, 'resources': quota}, indent=2))
        return
    write(intent_path, record)
    receipt = dict(record)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        match = re.fullmatch(r'(\d+)(?:;bscc26a)?', result.stdout.strip())
        receipt.update(status='SUBMITTED' if result.returncode == 0 and match else 'UNKNOWN_OR_FAILED_RECONCILE_SLURM',
            job_id=match.group(1) if match else None, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except BaseException as error:
        receipt.update(status='UNKNOWN_OR_FAILED_RECONCILE_SLURM', error=repr(error))
    write(receipt_path, receipt)
    print(json.dumps({'status': receipt['status'], 'job_id': receipt.get('job_id'), 'command': command}, indent=2), flush=True)
    if receipt['status'] != 'SUBMITTED':
        raise RuntimeError('Recovery submission outcome uncertain/failed: inspect actual Slurm; do not repeat')


def validate_analysis(analysis, identity, payload, allow_new_audit_failure=False):
    report_path = analysis / 'AGGREGATE_REPORT.json'; provenance_path = analysis / 'RECOVERY_PROVENANCE.json'
    if not report_path.is_file() or not provenance_path.is_file():
        raise ValueError('Existing analysis is partial or lacks recovery provenance; preserve it and inspect before any new analysis path')
    report = json.loads(report_path.read_text()); provenance = json.loads(provenance_path.read_text())
    if provenance.get('identity') != identity or provenance.get('recovery_payload') != payload or provenance.get('aggregate_report_sha256') != sha(report_path):
        raise ValueError('Existing analysis identity/report provenance differs')
    if report.get('schema') != 'autoregressive_replay_aggregate_v1' or report.get('matrix_sha256') != identity['matrix_sha256']:
        raise ValueError('Existing analysis matrix/schema differs')
    if report.get('script_sha256') != payload['operations/replay_and_aggregate.py']['sha256']:
        raise ValueError('Existing analysis replay implementation differs')
    matrix = json.loads((ROOT / 'matrix.json').read_text())
    entries = report.get('entries', [])
    if [(r.get('run_name'), r.get('arm'), r.get('seed')) for r in entries] != [(t['run_name'], t['arm'], t['seed']) for t in matrix['tasks']]:
        raise ValueError('Existing analysis has wrong six-task provenance')
    replay_module = load_module('recovery_replay_verify', ROOT / 'operations/replay_and_aggregate.py')
    for row in entries:
        run = ROOT / 'runs' / row['run_name']
        if row['status'] == 'COMPLETE_AND_REPLAYED':
            if not row.get('replay_passed') or row.get('actual_updates') != 50000:
                raise ValueError('Existing endpoint replay is incomplete')
            if sha(run / 'RUN_RECEIPT.json') != row['receipt_sha256'] or replay_module.verify_inventory(run) != row['artifact_manifest_sha256']:
                raise ValueError('Run artifacts changed after existing analysis')
        elif row['status'] == 'FAILED':
            if sha(run / 'FAILURE.json') != row['failure_sha256']:
                raise ValueError('Failure record changed after existing analysis')
        elif row['status'] == 'MISSING':
            if any((run / name).exists() for name in ('FAILURE.json', 'RUN_RECEIPT.json')):
                raise ValueError('Existing partial analysis is stale')
        elif row['status'] == 'AUDIT_OR_REPLAY_FAILED':
            # Fresh failed audits are packaged as evidence. A later execution
            # cannot silently treat them as a completed, reusable replay.
            if not allow_new_audit_failure:
                raise ValueError('Existing analysis contains audit/replay failure; preserve it for diagnosis')
        else:
            raise ValueError('Unrecognized analysis entry status')
    return report


def execute():
    identity = common_identity(); operations = ROOT / 'operations'
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '10' or os.environ.get('SLURM_ARRAY_TASK_ID'):
        raise ValueError('Recovery execution requires its single allocated-GPU Slurm job')
    submission = json.loads((operations / 'recovery_submission_receipt.json').read_text())
    if submission.get('status') != 'SUBMITTED' or str(submission.get('job_id')) != os.environ['SLURM_JOB_ID'] or submission.get('identity') != identity:
        raise ValueError('Executing job does not match durable recovery submission')
    payload = submission['payload']
    for name, item in payload.items():
        path = ROOT / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Recovery source changed after submission: ' + name)
    main_snapshot = scheduler_snapshot()
    main_snapshot['controller_live_queue_check'] = require_main_not_live()
    analysis = ROOT / 'analysis-r1'
    archive = ROOT / 'recovery/autoregressive-compact-r1.tar.gz'
    archive_receipt = archive.with_name(archive.name + '.receipt.json')
    completed = operations / 'RECOVERY_EXECUTION_RECEIPT.json'
    pack_module = load_module('recovery_pack_verify', ROOT / 'operations/pack_compact.py')
    if completed.exists():
        receipt = json.loads(completed.read_text())
        report = validate_analysis(analysis, identity, payload)
        pack_receipt = json.loads(archive_receipt.read_text()); pack_module.verify_archive(archive, pack_receipt)
        if receipt.get('identity') != identity or receipt.get('archive_sha256') != pack_receipt['archive_sha256'] or pack_receipt.get('aggregate_report_sha256') != sha(analysis / 'AGGREGATE_REPORT.json'):
            raise ValueError('Existing completed recovery provenance differs')
        print(json.dumps({'status': 'REUSED_VERIFIED_RECOVERY', 'scientific_status': report['status'], 'archive_sha256': pack_receipt['archive_sha256']}, indent=2))
        return
    intent_path = operations / 'RECOVERY_EXECUTION_INTENT.json'
    if intent_path.exists():
        raise FileExistsError('Recovery execution already started without completion; inspect partial evidence before manual continuation')
    write(intent_path, {'schema': 'autoregressive_recovery_execution_v1', 'identity': identity,
        'recovery_payload': payload, 'slurm_job_id': os.environ['SLURM_JOB_ID'],
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'main_accounting_snapshot': main_snapshot})
    try:
        if analysis.exists():
            report = validate_analysis(analysis, identity, payload)
        else:
            command = [sys.executable, '-B', '-u', str(ROOT / 'operations/replay_and_aggregate.py'),
                '--root', str(ROOT), '--source-root', str(OLD / 'due'), '--source-manifest', str(OLD / 'official_source_manifest.json'),
                '--data-dir', str(OLD / 'data-r2'), '--output-dir', str(analysis), '--device', 'cuda']
            subprocess.run(command, check=True)
            report = json.loads((analysis / 'AGGREGATE_REPORT.json').read_text())
            write(analysis / 'RECOVERY_PROVENANCE.json', {'schema': 'autoregressive_analysis_provenance_v1',
                'identity': identity, 'recovery_payload': payload, 'aggregate_report_sha256': sha(analysis / 'AGGREGATE_REPORT.json'),
                'main_accounting_snapshot': main_snapshot, 'slurm_job_id': os.environ['SLURM_JOB_ID']})
            validate_analysis(analysis, identity, payload, allow_new_audit_failure=True)
        # An incomplete scientific matrix is still a valid partial evidence pack.
        # Audit/replay errors remain visible and cannot be called recovery PASS.
        subprocess.run([sys.executable, '-B', '-u', str(ROOT / 'operations/pack_compact.py'),
            '--root', str(ROOT), '--analysis-dir', str(analysis), '--archive', str(archive)], check=True)
        pack_receipt = json.loads(archive_receipt.read_text()); pack_module.verify_archive(archive, pack_receipt)
        audit_failed = any(row['status'] == 'AUDIT_OR_REPLAY_FAILED' for row in report['entries'])
        receipt = {'schema': 'autoregressive_recovery_execution_v1',
            'status': 'COMPACT_EVIDENCE_WITH_AUDIT_FAILURE' if audit_failed else 'COMPACT_EVIDENCE_RECOVERED',
            'identity': identity, 'slurm_job_id': os.environ['SLURM_JOB_ID'], 'scientific_status': report['status'],
            'scientific_practical_gate': report['practical_gate'], 'aggregate_report_sha256': sha(analysis / 'AGGREGATE_REPORT.json'),
            'archive': str(archive), 'archive_sha256': pack_receipt['archive_sha256'], 'archive_bytes': pack_receipt['archive_bytes'],
            'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'optimizer_updates': 0}
        write(completed, receipt)
        print(json.dumps(receipt, indent=2, allow_nan=False), flush=True)
        if audit_failed:
            raise RuntimeError('Partial evidence pack completed, but checkpoint audit/replay failed; inspect retained report without resubmitting')
    except BaseException as error:
        failure = operations / 'RECOVERY_EXECUTION_FAILURE.json'
        if not failure.exists():
            write(failure, {'schema': 'autoregressive_recovery_failure_v1', 'status': 'FAILED', 'identity': identity,
                'slurm_job_id': os.environ['SLURM_JOB_ID'], 'error': repr(error), 'traceback': traceback.format_exc(),
                'analysis_path': str(analysis), 'archive_path': str(archive), 'optimizer_updates': 0})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--audit-only', action='store_true')
    group.add_argument('--execute', action='store_true')
    a = p.parse_args()
    execute() if a.execute else submit(a)


if __name__ == '__main__':
    main()
