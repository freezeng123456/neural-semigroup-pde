#!/usr/bin/env python3
"""Submit two disjoint waves of the frozen 24-task matrix, with no retries.

This operational adapter keeps the original source and experiment files intact.
Every original GPU-preflight and local-recovery gate remains mandatory.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_config import ARMS, SPECS, matrix
from launch_task import verify_payload, sha
from scheduler_limits import CPUS, GPUS, parse_pipe, resources, tres

ROOT = Path('/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1')
PAYLOAD_SHA = 'ffbe1bb1b40c46ba4cd41a9fc1b94ea0dbce130c7ee8c646203fb4f16543d8a1'
WAVES = {1: list(range(12)), 2: list(range(12, 24))}


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def actual_resources(jobs, evidence):
    """Slurm 20.11 uses TRES, and squeue %b may be N/A for --gpus jobs."""
    usage = {'active_tasks': 0, 'active_cpus': 0, 'active_gpus': 0}
    for job in jobs:
        if job['state'] == 'PENDING':
            continue
        cmd = ['scontrol', 'show', 'job', job['id'], '-o']
        reply = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        assert reply.returncode == 0, 'Cannot resolve live job resources; inspect again'
        fields = dict(re.findall(r'(\w+)=([^\s]+)', reply.stdout))
        raw_tres = fields.get('AllocTRES') or fields.get('TRES') or fields.get('ReqTRES')
        assert raw_tres is not None, 'Cannot resolve live job TRES'
        counts = tres(raw_tres)
        assert 'cpu' in counts and fields.get('JobState') is not None
        # Generic and typed GPU counts describe the same allocation.
        gpus = counts.get('gres/gpu', sum(v for k, v in counts.items() if k.startswith('gres/gpu:')))
        requests = fields.get('TresPerJob', '') + fields.get('TresPerNode', '')
        assert 'gpu' not in requests or gpus > 0, 'Unresolved GPU allocation'
        selected = {k: fields[k] for k in ('JobId', 'JobState', 'NumCPUs', 'TRES', 'AllocTRES', 'ReqTRES', 'TresPerJob') if k in fields}
        evidence.append({'command': cmd, 'returncode': reply.returncode, 'persisted_fields': selected,
                         'resolved_cpus': counts['cpu'], 'resolved_gpus': gpus})
        usage['active_tasks'] += 1
        usage['active_cpus'] += counts['cpu']; usage['active_gpus'] += gpus
    return usage


def quota_for_wave(evidence):
    # Original controller, association, partition, QoS and wall checks.
    basic = resources('validate', evidence)
    current = actual_resources(basic['queue_snapshot'], evidence)
    current['submitted_tasks'] = len(basic['queue_snapshot'])
    limit = basic['limits']
    assert current['submitted_tasks'] + 12 <= limit['max_submit'], '12-task wave exceeds live MaxSubmit headroom'
    assert current['active_tasks'] + 4 <= limit['max_jobs'], 'Four-task concurrency exceeds live MaxJobs headroom'
    assert current['active_cpus'] + 4 * CPUS <= limit['cpus'], 'Insufficient CPU headroom'
    assert current['active_gpus'] + 4 * GPUS <= limit['gpus'], 'Insufficient GPU headroom'
    # Correct any finite QoS group GPU check, including allocations of other users.
    qos_cmd = ['sacctmgr', '-nP', 'show', 'qos', 'where', 'name='+','.join(basic['qos_checked']), 'format=Name,GrpTRES']
    qos_reply = subprocess.run(qos_cmd, capture_output=True, text=True, timeout=60)
    assert qos_reply.returncode == 0
    evidence.append({'command': qos_cmd, 'returncode': 0, 'stdout': qos_reply.stdout, 'stderr': qos_reply.stderr})
    for row in parse_pipe(qos_reply.stdout, ('Name', 'GrpTRES')):
        group = tres(row['GrpTRES'])
        if not group:
            continue
        from scheduler_limits import parse_queue
        scoped = {}
        # Union the GPU partition with all jobs explicitly selecting this QoS.
        # This conservatively covers partition QoS and user-selected QoS.
        for scope in ('--partition=gpu', '--qos='+row['Name']):
            command = ['squeue', '--array', '--noheader', scope, '--format=%i|%T|%a|%P|%C|%b|%q']
            q = subprocess.run(command, capture_output=True, text=True, timeout=60)
            assert q.returncode == 0
            evidence.append({'command': command, 'returncode': 0, 'stdout': q.stdout, 'stderr': q.stderr})
            for job in parse_queue(q.stdout):
                scoped[job['id']] = job
        group_usage = actual_resources(list(scoped.values()), evidence)
        assert group_usage['active_cpus'] + 4 * CPUS <= group.get('cpu', 10**12), 'Finite QoS group CPU limit'
        assert group_usage['active_gpus'] + 4 * GPUS <= group.get('gres/gpu', 10**12), 'Finite QoS group GPU limit'
    return {'cluster': basic['cluster'], 'account': basic['account'], 'partition': basic['partition'],
            'limits': limit, 'current_usage': current, 'new_potential_tasks': 12,
            'new_max_concurrent_tasks': 4, 'queue_snapshot': basic['queue_snapshot'],
            'qos_checked': basic['qos_checked'], 'limitation': basic['limitation']}


def preflight_gate(root, preflight_job, local_gate, frozen, evidence):
    assert preflight_job.isdigit()
    cmd = ['sacct', '-j', preflight_job, '-nP', '--format=JobIDRaw,State,ExitCode']
    raw = subprocess.check_output(cmd, text=True)
    evidence.append({'command': cmd, 'stdout': raw})
    rows = [r.split('|') for r in raw.splitlines() if r.split('|')[0] == preflight_job]
    assert len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0']
    report_path = root/'preflight'/('job'+preflight_job)/'VALIDATION_REPORT.json'
    report = json.loads(report_path.read_text()); checks = report['checks']
    names = {c['name'] for c in checks}
    required = {'resume_'+arm for arm in ARMS} | {'short_predict_reload_'+arm for arm in ARMS} | {
        'official_cfo_nonuniform_formula', 'analytic_quintic_path_velocity', 'rk4_physical_time_fourth_order',
        'rk4_absolute_time_stages', 'paired_initialization_batches_labels_and_collocation', 'frozen_inputs_unchanged',
        'solver_refinement_smoke_auto20e500', 'solver_refinement_smoke_time20e500'}
    assert required <= names and len(names) == len(checks) == 29 and all(c['passed'] is True for c in checks)
    assert report['schema'] == 'next_methods_engineering_validation_v1' and report['status'] == 'PASS'
    assert report['device'] == 'cuda' and 'V100' in report['gpu_name'] and report['gpu_capability'] == [7, 0]
    assert str(report['slurm_job_id']) == preflight_job and report['actual_optimizer_updates'] == 168
    assert report['payload_manifest_sha256'] == PAYLOAD_SHA
    assert report['data_payload_sha256'] == frozen['files']['data-r2/two_step_data.npz']['sha256']
    assert report['flow_payload_sha256'] == frozen['files']['flow-data-r2/flow_data.npz']['sha256']
    assert all(r['estimated_training_hours'] < 4.5 for r in report['benchmarks'].values())
    gate = json.loads(local_gate.read_text())
    assert gate['schema'] == 'next_methods_local_recovery_gate_v1' and gate['status'] == 'PASS'
    assert str(gate['preflight_job_id']) == preflight_job and gate['payload_manifest_sha256'] == PAYLOAD_SHA
    assert gate['validation_report_sha256'] == sha(report_path)
    assert gate['checks'] and all(c['passed'] is True for c in gate['checks'])
    return {'job_id': preflight_job, 'report_sha256': sha(report_path), 'local_gate_sha256': sha(local_gate)}


def predecessor_gate(root, evidence):
    receipt = json.loads((root/'operations'/'full_wave1_submission_receipt.json').read_text())
    assert receipt['status'] == 'SUBMITTED' and receipt['payload_manifest_sha256'] == PAYLOAD_SHA
    assert receipt['task_indices'] == WAVES[1] and receipt['job_id'].isdigit()
    assert receipt['complete_fixed_plan'] == matrix()
    job = receipt['job_id']
    cmd = ['sacct', '-j', job, '-nP', '--format=JobID%40,State%30,ExitCode']
    raw = subprocess.check_output(cmd, text=True)
    evidence.append({'command': cmd, 'stdout': raw})
    rows = parse_pipe(raw, ('JobID', 'State', 'ExitCode'))
    tasks = matrix()['tasks']; completed = []
    for index in WAVES[1]:
        jid = job+'_'+str(index)
        matches = [r for r in rows if r['JobID'] == jid]
        assert len(matches) == 1 and matches[0]['State'] == 'COMPLETED' and matches[0]['ExitCode'] == '0:0', jid+' has not completed successfully'
        task = tasks[index]; run = root/'runs'/task['run_name']
        result = json.loads((run/'RUN_RECEIPT.json').read_text())
        assert result['status'] == 'NEXT_METHODS_FULL_COMPLETE' and result['actual_updates'] == task['updates']
        assert result['forward_calls'] == task['total_forward_calls'] and result['parameters_changed'] is True
        assert result['config']['arm'] == task['arm'] and result['config']['seed'] == task['seed']
        assert result['config']['spec'] == SPECS[task['arm']]
        assert result['final_validation']['valid_metric'] is True
        artifacts = json.loads((run/'artifact_manifest.json').read_text())
        for name, item in artifacts.items():
            path = (run/name).resolve(strict=True)
            assert path.is_relative_to(run) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
        completed.append({'job': jid, 'run_name': task['run_name'], 'receipt_sha256': sha(run/'RUN_RECEIPT.json')})
    return {'wave1_job': job, 'completed_tasks': completed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wave', type=int, choices=(1, 2), required=True)
    parser.add_argument('--preflight-job', required=True)
    parser.add_argument('--local-gate', type=Path, required=True)
    parser.add_argument('--audit-only', action='store_true')
    a = parser.parse_args(); root = a.root.resolve(strict=True)
    assert getpass.getuser() == 'scv7tsq' and root == ROOT
    assert sha(root/'PAYLOAD_MANIFEST.json') == PAYLOAD_SHA
    frozen = verify_payload(root); operations = root/'operations'; operations.mkdir(exist_ok=True)
    assert not (operations/'full_submission_intent.json').exists(), 'Inspect existing original full-matrix intent before another submission'
    intent_file = operations/f'full_wave{a.wave}_submission_intent.json'
    assert not intent_file.exists(), 'Existing intent: inspect receipt and real Slurm, never resubmit automatically'
    evidence = []; preflight = preflight_gate(root, a.preflight_job, a.local_gate, frozen, evidence)
    previous = predecessor_gate(root, evidence) if a.wave == 2 else None
    assert all(not (root/'runs'/task['run_name']).exists() for task in matrix()['tasks'][WAVES[a.wave][0]:WAVES[a.wave][-1]+1])
    if a.wave == 1:
        assert all(not (root/'runs'/task['run_name']).exists() for task in matrix()['tasks'])
    quota = quota_for_wave(evidence)
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    start, end = WAVES[a.wave][0], WAVES[a.wave][-1]
    command = ['sbatch', '--parsable', '--partition=gpu', '--account=scv7tsq', '--gpus=1', '--ntasks=1',
               '--cpus-per-task=10', '--time=06:00:00', f'--job-name=sg-next-w{a.wave}', '--chdir='+str(root),
               '--output='+str(logs/f'full-wave{a.wave}-%A_%a.log'), f'--array={start}-{end}%4',
               str(root/'slurm_next.sh'), str(root), 'full']
    intent = {'schema': 'next_methods_fixed_wave_submission_v1', 'mode': 'full', 'wave': a.wave,
              'task_indices': WAVES[a.wave], 'complete_fixed_plan': matrix(), 'command': command,
              'requested_at_utc': datetime.now(timezone.utc).isoformat(), 'payload_manifest_sha256': PAYLOAD_SHA,
              'preflight': preflight, 'predecessor': previous, 'resource_check': quota,
              'read_only_scheduler_evidence': evidence,
              'operational_launcher_sha256': sha(Path(__file__)), 'operational_launcher_source': Path(__file__).read_text()}
    if a.audit_only:
        intent['status'] = 'AUDIT_PASS_NO_SUBMISSION'
        print(json.dumps({k:v for k,v in intent.items() if k not in ('operational_launcher_source', 'read_only_scheduler_evidence', 'complete_fixed_plan')}, indent=2))
        return
    save(intent_file, intent)
    receipt = dict(intent)
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=90)
        match = re.fullmatch(r'(\d+)(?:;bscc26a)?', r.stdout.strip())
        receipt.update(status='SUBMITTED' if r.returncode == 0 and match else 'UNKNOWN_OR_FAILED_INSPECT_SLURM',
                       returncode=r.returncode, stdout=r.stdout, stderr=r.stderr, job_id=match.group(1) if match else None)
    except BaseException as error:
        receipt.update(status='UNKNOWN_OR_FAILED_INSPECT_SLURM', error=repr(error))
    save(operations/f'full_wave{a.wave}_submission_receipt.json', receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('operational_launcher_source', 'read_only_scheduler_evidence', 'complete_fixed_plan')}, indent=2), flush=True)
    assert receipt['status'] == 'SUBMITTED', 'Inspect durable intent and Slurm; do not retry'


if __name__ == '__main__':
    main()
