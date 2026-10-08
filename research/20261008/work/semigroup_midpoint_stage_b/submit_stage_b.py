#!/usr/bin/env python3
"""Submit one reviewed Stage B validation job or the bounded 12-task matrix.

Run only on the authorized Beijing login platform. This script does not train.
Any existing submission intent blocks another attempt, including after a failed
or unknown sbatch response; inspect the scheduler before manual recovery.

Full mode also requires a separately produced local recovery gate receipt:
  schema: stage_b_local_recovery_gate_v1, status: PASS,
  preflight_job_id, payload_manifest_sha256, validation_report_sha256,
  checks: nonempty list of {name, passed:true}.
The gate is an explicit attestation from local recovery; this script never
creates it or treats the remote validation report as that independent gate.
"""
import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

USER = ACCOUNT = 'scv7tsq'
CLUSTER = 'bscc26a'
PARENT = Path('/data02/run01/scv7tsq/neural-semigroup')
COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
SOURCE_MANIFEST_SHA = 'c423cbc0bde6636fd07f157b50a2f01d0430ac06973f5f0dfc2dbec1c483a79c'
DATA_SHA = '1092743121a2c6cc733da44d034f0399d61c88d1d6b0016e75c57313bfc7465b'
CPUS, GPUS, FULL_TASKS, CONCURRENCY, WALL_SECONDS = 10, 1, 12, 4, 14400
REQUIRED_CHECKS = {'data_author_loader_replay', 'author_one_update',
                   'loss_cross_branch_identities', 'independent_rng',
                   'resume_gdsg500', 'resume_single500', 'resume_midpoint500', 'resume_gdsg333',
                   'paired_batches_and_initialization', 'two_step_original_predict_after_reload',
                   'distinct_initialization_seeds', 'frozen_inputs_unchanged'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def positive_limit(value):
    value = value.strip()
    if value.upper() in ('', 'UNLIMITED', 'NONE', 'N/A', '-1'):
        return None
    require(value.isdigit(), 'Unrecognized Slurm limit: ' + value)
    return int(value)


def time_seconds(value):
    if value.upper() in ('', 'UNLIMITED', 'NONE', 'N/A', '-1'):
        return None
    days = 0
    if '-' in value:
        day_text, value = value.split('-', 1); days = int(day_text)
    parts = [int(x) for x in value.split(':')]
    if len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    else:
        seconds = parts[0] * 60
    return days * 86400 + seconds


def tres(value):
    result = {}
    for item in value.split(','):
        if '=' not in item:
            continue
        name, count = item.strip().split('=', 1)
        if name == 'cpu' or name == 'gres/gpu' or name.startswith('gres/gpu:'):
            require(count.isdigit(), 'Unrecognized resource limit: ' + item)
            result[name] = int(count)
    return result


def gpu_count(value):
    if value in ('', 'N/A', '(null)', 'None'):
        return 0
    # squeue %b: gres/gpu:1 or gres/gpu:tesla_v100:1, optionally other GRES.
    total = 0
    for entry in value.split(','):
        if entry.startswith(('gres/gpu:', 'gpu:')):
            tail = entry.rsplit(':', 1)[1]
            require(tail.isdigit(), 'Unrecognized requested GRES: ' + value)
            total += int(tail)
    require(total > 0 or 'gpu' not in value, 'Unknown GPU resource spelling')
    return total


def parse_pipe(text, fields):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.strip().split('|')
        if len(values) == len(fields) + 1 and values[-1] == '':
            values.pop()
        require(len(values) == len(fields), 'Unexpected Slurm columns: ' + line)
        rows.append(dict(zip(fields, (v.strip() for v in values))))
    return rows


def parse_queue(text):
    rows = parse_pipe(text, ('id', 'state', 'account', 'partition', 'cpus', 'gres', 'qos'))
    for row in rows:
        require(re.fullmatch(r'\d+(?:_\d+)?', row['id']) is not None,
                'squeue must expand every array task; compressed/unknown ID: ' + row['id'])
        require(row['cpus'].isdigit(), 'Invalid requested CPU count')
        row['cpus'] = int(row['cpus']); row['gpus'] = gpu_count(row['gres'])
    require(len({row['id'] for row in rows}) == len(rows), 'Duplicate squeue task rows')
    return rows


def all_pass(report):
    checks = report.get('checks')
    return (report.get('status') == 'PASS' and isinstance(checks, list) and bool(checks)
            and all(isinstance(c, dict) and c.get('name') and c.get('passed') is True for c in checks)
            and len({c['name'] for c in checks}) == len(checks))


def verify_payload(root):
    manifest_path = root / 'PAYLOAD_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['source_commit'] == COMMIT and manifest['prefix'] == root.name, 'Payload identity mismatch')
    for name, spec in manifest['files'].items():
        path = (root / name).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Payload file changed: ' + name)
    require(sha(root / 'official_source_manifest.json') == SOURCE_MANIFEST_SHA, 'Wrong official source manifest')
    require(sha(root / 'data-r1/development_data.npz') == DATA_SHA, 'Wrong development payload')
    require(sha(Path(__file__).resolve()) == manifest['files']['submit_stage_b.py']['sha256'], 'Submitter differs from frozen payload')
    matrix = json.loads((root / 'matrix.json').read_text())
    require(len(matrix['tasks']) == 12, 'Matrix must contain 12 tasks')
    expected = {(arm, seed) for arm in ('gdsg500', 'single500', 'midpoint500', 'gdsg333') for seed in range(3)}
    require({(task['arm'], task['seed']) for task in matrix['tasks']} == expected, 'Wrong matrix arms/seeds')
    require(matrix['max_concurrency'] == 4 and matrix['gpu_per_task'] == 1 and matrix['cpu_per_task'] == 10 and matrix['allocation_limit_seconds_per_task'] == 14400, 'Wrong matrix resource limits')
    for index, task in enumerate(matrix['tasks']):
        epochs = 333 if task['arm'] == 'gdsg333' else 500
        calls = 6 if task['arm'].startswith('gdsg') else 4
        require(task['index'] == index and task['epochs'] == epochs and task['updates'] == epochs * 100 and task['forwards_per_batch'] == calls and task['total_forward_calls'] == epochs * 100 * calls, 'Invalid matrix row')
        require(re.fullmatch(r'[a-z0-9-]+', task['run_name']) is not None, 'Unsafe run name')
    require(len({task['run_name'] for task in matrix['tasks']}) == 12, 'Duplicate run names')
    return manifest, matrix, sha(manifest_path)


def query(command, evidence):
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
    evidence.append({'command': command, 'returncode': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr})
    require(reply.returncode == 0, 'Read-only Slurm query failed: ' + ' '.join(command))
    return reply.stdout


def verify_preflight(root, job, gate_path, payload_sha, evidence):
    require(job and job.isdigit(), 'Full mode requires numeric --preflight-job')
    require(gate_path is not None, 'Full mode requires separate --local-gate-receipt')
    raw = query(['sacct', '-j', job, '-nP', '--format=JobIDRaw,State,ExitCode'], evidence)
    parents = [r for r in parse_pipe(raw, ('id', 'state', 'exit')) if r['id'] == job]
    require(len(parents) == 1 and parents[0]['state'] == 'COMPLETED' and parents[0]['exit'] == '0:0', 'Preflight parent job must be COMPLETED 0:0')
    report_path = root / 'preflight' / ('job' + job) / 'VALIDATION_REPORT.json'
    report = json.loads(report_path.read_text())
    require(report.get('schema') == 'stage_b_engineering_validation_v1' and all_pass(report), 'Preflight gate must PASS every check')
    checks = {check['name']: check for check in report['checks']}
    require(REQUIRED_CHECKS <= set(checks), 'One or more required engineering gates is missing')
    replay = checks['two_step_original_predict_after_reload']
    require(replay.get('all_bitwise_equal') is True and replay.get('rollouts') == 8 and replay.get('transitions_per_rollout') == 2, 'Short original-predict reload replay is incomplete')
    require(report.get('device') == 'cuda' and 'V100' in str(report.get('gpu_name')) and report.get('gpu_capability') == [7, 0], 'Preflight must be real V100 CUDA')
    require(str(report.get('slurm_job_id')) == job and report.get('optimizer_updates') == 13, 'Wrong preflight job/update count')
    require(report.get('source_commit') == COMMIT and report.get('source_manifest_sha256') == SOURCE_MANIFEST_SHA and report.get('data_payload_sha256') == DATA_SHA and report.get('payload_manifest_sha256') == payload_sha, 'Preflight frozen inputs differ')
    detailed = report_path.with_name('PREFLIGHT_PASSED.json')
    require(sha(detailed) == report.get('detailed_report_sha256'), 'Detailed preflight report changed')
    require(isinstance(report.get('input_sha256'), dict) and bool(report['input_sha256']), 'Preflight input inventory missing')
    for path, expected in report['input_sha256'].items():
        input_path = Path(path).resolve(strict=True)
        original_train = PARENT / '20261004-due-vorticity-1e62910-r1/data/VorticityOSG_train.mat'
        require(input_path.is_relative_to(root) or input_path == original_train, 'Unexpected path in preflight input inventory')
        require(sha(input_path) == expected, 'Preflight input changed: ' + path)
    gate = json.loads(gate_path.read_text())
    require(gate.get('schema') == 'stage_b_local_recovery_gate_v1' and all_pass(gate), 'Separate local recovery gate must PASS')
    require(str(gate.get('preflight_job_id')) == job and gate.get('payload_manifest_sha256') == payload_sha and gate.get('validation_report_sha256') == sha(report_path), 'Local recovery gate identifies different evidence')
    return {'preflight_job_id': job, 'validation_report_sha256': sha(report_path),
            'local_gate_receipt': str(gate_path.resolve()), 'local_gate_sha256': sha(gate_path),
            'checks_passed': [c['name'] for c in report['checks']]}


def resources(mode, evidence):
    # Persist only ClusterName from this broad configuration read; do not copy
    # authentication/configuration fields into the public evidence receipt.
    cluster_reply = subprocess.run(['scontrol', 'show', 'config'], capture_output=True, text=True, timeout=60)
    cluster_match = re.search(r'^\s*ClusterName\s*=\s*(\S+)\s*$', cluster_reply.stdout, re.MULTILINE)
    current_cluster = cluster_match.group(1) if cluster_match else None
    evidence.append({'command': ['scontrol', 'show', 'config'], 'returncode': cluster_reply.returncode,
                     'persisted_fields': {'ClusterName': current_cluster},
                     'stderr': cluster_reply.stderr})
    require(cluster_reply.returncode == 0 and current_cluster == CLUSTER, 'Current Slurm controller is not the authorized Beijing cluster')
    fields = ('Cluster', 'Account', 'User', 'Partition', 'QOS', 'DefaultQOS', 'MaxJobs', 'MaxSubmit', 'GrpTRES', 'MaxTRES', 'MaxWall')
    raw = query(['sacctmgr', '-nP', 'show', 'assoc', 'where', 'user=' + USER, 'account=' + ACCOUNT, 'format=' + ','.join(fields)], evidence)
    assocs = [r for r in parse_pipe(raw, fields) if r['Cluster'] == CLUSTER and r['Account'] == ACCOUNT and r['User'] == USER and r['Partition'] in ('', 'gpu')]
    require(bool(assocs), 'No authorized Beijing GPU association')
    partition_raw = query(['scontrol', 'show', 'partition', 'gpu', '-o'], evidence)
    parts = dict(re.findall(r'(\w+)=([^\s]+)', partition_raw))
    require(parts.get('PartitionName') == 'gpu' and parts.get('State') == 'UP', 'GPU partition is not UP')
    max_time = time_seconds(parts.get('MaxTime', ''))
    require(max_time is None or max_time >= WALL_SECONDS, 'Partition wall limit below four hours')
    qos_names = {parts.get('QoS', parts.get('QOS', ''))}
    for row in assocs:
        qos_names.update(row['QOS'].split(',')); qos_names.add(row['DefaultQOS'])
    qos_names -= {'', 'N/A', '(null)'}
    require(qos_names and all(re.fullmatch(r'[A-Za-z0-9_-]+', q) for q in qos_names), 'Unresolved QoS names')
    qfields = ('Name', 'MaxWall', 'MaxJobsPU', 'MaxSubmitPU', 'MaxTRESPU', 'GrpTRES')
    qos_raw = query(['sacctmgr', '-nP', 'show', 'qos', 'where', 'name=' + ','.join(sorted(qos_names)), 'format=' + ','.join(qfields)], evidence)
    qos_rows = parse_pipe(qos_raw, qfields)
    require({row['Name'] for row in qos_rows} == qos_names, 'Incomplete QoS limits')
    queue_raw = query(['squeue', '--array', '--noheader', '--user=' + USER, '--format=%i|%T|%a|%P|%C|%b|%q'], evidence)
    jobs = parse_queue(queue_raw)
    # --array gives every potential task, so 12 pending elements consume 12
    # MaxSubmit slots even though this matrix can run at most four together.
    active = [row for row in jobs if row['state'] != 'PENDING']
    potential, concurrent = (12, 4) if mode == 'full' else (1, 1)
    maxjobs = [positive_limit(r['MaxJobs']) for r in assocs]
    maxsubmit = [positive_limit(r['MaxSubmit']) for r in assocs]
    cpulimits, gpulimits = [], []
    for row in assocs:
        group = tres(row['GrpTRES'])
        if 'cpu' in group: cpulimits.append(group['cpu'])
        if 'gres/gpu' in group: gpulimits.append(group['gres/gpu'])
        individual = tres(row['MaxTRES'])
        require(individual.get('cpu', CPUS) >= CPUS and individual.get('gres/gpu', GPUS) >= GPUS, 'Per-job association resource limit too small')
        limit = time_seconds(row['MaxWall'])
        require(limit is None or limit >= WALL_SECONDS, 'Association wall limit below four hours')
    for row in qos_rows:
        maxjobs.append(positive_limit(row['MaxJobsPU'])); maxsubmit.append(positive_limit(row['MaxSubmitPU']))
        limits = tres(row['MaxTRESPU'])
        if 'cpu' in limits: cpulimits.append(limits['cpu'])
        if 'gres/gpu' in limits: gpulimits.append(limits['gres/gpu'])
        limit = time_seconds(row['MaxWall'])
        require(limit is None or limit >= WALL_SECONDS, 'QoS wall limit below four hours')
        group = tres(row['GrpTRES'])
        if group:
            # A finite QoS group limit covers other users as well. Inspect only
            # scheduler resource metadata, never names, paths, or commands.
            scope_flag = '--partition=gpu' if row['Name'] == parts.get('QoS', parts.get('QOS')) else '--qos=' + row['Name']
            global_jobs = parse_queue(query(['squeue', '--array', '--noheader', scope_flag, '--format=%i|%T|%a|%P|%C|%b|%q'], evidence))
            group_active = [r for r in global_jobs if r['state'] != 'PENDING']
            require(sum(r['cpus'] for r in group_active) + concurrent * CPUS <= group.get('cpu', 10**12), 'QoS group CPU headroom insufficient')
            require(sum(r['gpus'] for r in group_active) + concurrent * GPUS <= group.get('gres/gpu', 10**12), 'QoS group GPU headroom insufficient')
    maxjobs = [x for x in maxjobs if x is not None]; maxsubmit = [x for x in maxsubmit if x is not None]
    require(maxjobs and maxsubmit and cpulimits and gpulimits, 'Finite account limits must be visible; inherited/unknown limits need inspection')
    limits = {'max_jobs': min(maxjobs), 'max_submit': min(maxsubmit), 'cpus': min(cpulimits), 'gpus': min(gpulimits)}
    usage = {'submitted_tasks': len(jobs), 'active_tasks': len(active),
             'active_cpus': sum(r['cpus'] for r in active), 'active_gpus': sum(r['gpus'] for r in active)}
    require(usage['submitted_tasks'] + potential <= limits['max_submit'], 'MaxSubmit headroom insufficient for all potential array tasks')
    require(usage['active_tasks'] + concurrent <= limits['max_jobs'], 'MaxJobs headroom insufficient')
    require(usage['active_cpus'] + concurrent * CPUS <= limits['cpus'], 'Account CPU headroom insufficient')
    require(usage['active_gpus'] + concurrent * GPUS <= limits['gpus'], 'Account GPU headroom insufficient')
    return {'cluster': CLUSTER, 'account': ACCOUNT, 'partition': 'gpu',
            'limits': limits, 'current_usage': usage, 'new_potential_tasks': potential,
            'new_max_concurrent_tasks': concurrent, 'qos_checked': sorted(qos_names),
            'queue_snapshot': jobs,
            'limitation': 'This is a live pre-submit snapshot, not a reservation; Slurm enforces subsequent changes and physical availability.'}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=('validate', 'full'), required=True)
    p.add_argument('--preflight-job')
    p.add_argument('--local-gate-receipt', type=Path)
    return p


def main():
    args = build_parser().parse_args()
    require(getpass.getuser() == USER, 'Only the authorized Beijing user may submit')
    root = args.root.resolve(strict=True)
    require(root.parent == PARENT and root.name.startswith('20261004-midpoint-stage-b-'), 'Unexpected project root')
    operations = root / 'operations'; operations.mkdir(exist_ok=True)
    intent_path = operations / (args.mode + '_submission_intent.json')
    require(not intent_path.exists(), 'Submission intent already exists; inspect scheduler and previous receipt, never blindly retry')
    evidence = []
    manifest, matrix, payload_sha = verify_payload(root)
    if args.mode == 'full':
        require(all(not (root / 'runs' / task['run_name']).exists() for task in matrix['tasks']), 'A matrix run already exists')
        preflight = verify_preflight(root, args.preflight_job, args.local_gate_receipt, payload_sha, evidence)
    else:
        require(args.preflight_job is None and args.local_gate_receipt is None, 'Validation does not accept full-mode gates')
        preflight = None
    quota = resources(args.mode, evidence)
    logs = root / 'logs'; logs.mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--partition=gpu', '--account=' + ACCOUNT,
               '--gpus=1', '--ntasks=1', '--cpus-per-task=10', '--time=04:00:00',
               '--job-name=sg-mid-' + args.mode, '--chdir=' + str(root),
               '--output=' + str(logs / (args.mode + '-%A_%a.log'))]
    if args.mode == 'full':
        command.append('--array=0-11%4')
    command += [str(root / 'slurm_stage_b.sh'), str(root), args.mode]
    intent = {'schema': 'stage_b_submission_intent_v1', 'mode': args.mode,
              'requested_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': command, 'payload_manifest_sha256': payload_sha,
              'preflight_gate': preflight, 'resource_check': quota,
              'read_only_scheduler_evidence': evidence,
              'retry_policy': 'Any intent, failed return, timeout, exception or unknown response blocks resubmission until manual scheduler inspection.'}
    # The exclusive, durable intent comes before the only mutating Slurm call.
    with intent_path.open('x') as f:
        json.dump(intent, f, indent=2); f.write('\n'); f.flush()
        import os
        os.fsync(f.fileno())
    receipt = dict(intent)
    try:
        reply = subprocess.run(command, capture_output=True, text=True, timeout=90)
        receipt.update(returncode=reply.returncode, stdout=reply.stdout, stderr=reply.stderr)
        match = re.fullmatch(r'(\d+)(?:;([A-Za-z0-9_-]+))?', reply.stdout.strip())
        if reply.returncode == 0 and match and match.group(2) in (None, CLUSTER):
            receipt.update(status='SUBMITTED', job_id=match.group(1))
        else:
            receipt['status'] = 'SUBMISSION_FAILED_OR_UNKNOWN_INSPECT_SCHEDULER'
    except BaseException as error:
        receipt.update(status='SUBMISSION_OUTCOME_UNKNOWN_INSPECT_SCHEDULER', error=repr(error))
        if isinstance(error, subprocess.TimeoutExpired):
            for name in ('stdout', 'stderr'):
                value = getattr(error, name, None)
                receipt[name] = value.decode(errors='replace') if isinstance(value, bytes) else value
    write_exclusive(operations / (args.mode + '_submission_receipt.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    require(receipt['status'] == 'SUBMITTED', 'Submission was not confirmed; inspect saved intent and live scheduler before any recovery')


if __name__ == '__main__':
    main()
