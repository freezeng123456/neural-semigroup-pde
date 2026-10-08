#!/usr/bin/env python3
"""Submit the engineering preflight, or the fixed 24-task research matrix.

One durable intent precedes sbatch. An uncertain response never causes an
automatic repeated submission. Full training requires recovered V100 evidence.
"""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
from experiment_config import ARMS, matrix
from launch_task import verify_payload, sha
from scheduler_limits import resources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=('validate', 'full'), required=True)
    p.add_argument('--preflight-job')
    p.add_argument('--local-gate', type=Path)
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    assert getpass.getuser() == 'scv7tsq'
    assert root.parent == Path('/data02/run01/scv7tsq/neural-semigroup') and root.name.startswith('20261005-next-methods-')
    frozen = verify_payload(root)
    payload_sha = sha(root/'PAYLOAD_MANIFEST.json')
    operations = root/'operations'; operations.mkdir(exist_ok=True)
    intent_file = operations/(a.mode+'_submission_intent.json')
    assert not intent_file.exists(), 'Existing intent: inspect Slurm and receipt before any retry'
    preflight = None
    if a.mode == 'full':
        assert a.preflight_job and a.preflight_job.isdigit() and a.local_gate
        raw = subprocess.check_output(['sacct', '-j', a.preflight_job, '-nP', '--format=JobIDRaw,State,ExitCode'], text=True)
        rows = [r.split('|') for r in raw.splitlines() if r.split('|')[0] == a.preflight_job]
        assert len(rows) == 1 and rows[0][1:3] == ['COMPLETED', '0:0']
        report_path = root/'preflight'/('job'+a.preflight_job)/'VALIDATION_REPORT.json'
        report = json.loads(report_path.read_text()); checks = report['checks']
        names = {c['name'] for c in checks}
        required = {'resume_'+arm for arm in ARMS} | {'short_predict_reload_'+arm for arm in ARMS} | {
                    'official_cfo_nonuniform_formula', 'analytic_quintic_path_velocity', 'rk4_physical_time_fourth_order',
                    'rk4_absolute_time_stages', 'paired_initialization_batches_labels_and_collocation', 'frozen_inputs_unchanged'}
        required |= {'solver_refinement_smoke_auto20e500', 'solver_refinement_smoke_time20e500'}
        assert required <= names and len(names) == len(checks) and all(c['passed'] is True for c in checks)
        assert report['schema'] == 'next_methods_engineering_validation_v1' and report['status'] == 'PASS'
        assert report['device'] == 'cuda' and 'V100' in report['gpu_name'] and report['gpu_capability'] == [7, 0]
        assert str(report['slurm_job_id']) == a.preflight_job and report['actual_optimizer_updates'] == 168
        assert report['payload_manifest_sha256'] == payload_sha
        assert report['data_payload_sha256'] == frozen['files']['data-r2/two_step_data.npz']['sha256']
        assert report['flow_payload_sha256'] == frozen['files']['flow-data-r2/flow_data.npz']['sha256']
        # Leave generous headroom for checkpoint, validation and final refinement.
        assert all(r['estimated_training_hours'] < 4.5 for r in report['benchmarks'].values()), 'Measured training may exceed six-hour allocation; revise schedule before full launch'
        gate = json.loads(a.local_gate.read_text())
        assert gate['schema'] == 'next_methods_local_recovery_gate_v1' and gate['status'] == 'PASS'
        assert str(gate['preflight_job_id']) == a.preflight_job and gate['payload_manifest_sha256'] == payload_sha and gate['validation_report_sha256'] == sha(report_path)
        assert gate['checks'] and all(c['passed'] is True for c in gate['checks'])
        assert all(not (root/'runs'/task['run_name']).exists() for task in matrix()['tasks'])
        preflight = {'job_id': a.preflight_job, 'report_sha256': sha(report_path), 'local_gate_sha256': sha(a.local_gate)}
    else:
        assert a.preflight_job is None and a.local_gate is None
    evidence = []
    quota = resources(a.mode, evidence)
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    command = ['sbatch', '--parsable', '--partition=gpu', '--account=scv7tsq', '--gpus=1', '--ntasks=1',
               '--cpus-per-task=10', '--time=06:00:00', '--job-name=sg-next-'+a.mode, '--chdir='+str(root),
               '--output='+str(logs/(a.mode+'-%A_%a.log'))]
    if a.mode == 'full':
        command.append('--array=0-23%4')
    command += [str(root/'slurm_next.sh'), str(root), a.mode]
    intent = {'schema': 'next_methods_submission_v1', 'mode': a.mode, 'command': command,
              'requested_at_utc': datetime.now(timezone.utc).isoformat(), 'payload_manifest_sha256': payload_sha,
              'preflight': preflight, 'resource_check': quota, 'read_only_scheduler_evidence': evidence}
    with intent_file.open('x') as f:
        json.dump(intent, f, indent=2); f.flush(); os.fsync(f.fileno())
    receipt = dict(intent)
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=90)
        match = re.fullmatch(r'(\d+)(?:;bscc26a)?', r.stdout.strip())
        receipt.update(status='SUBMITTED' if r.returncode == 0 and match else 'UNKNOWN_OR_FAILED_INSPECT_SLURM',
                       returncode=r.returncode, stdout=r.stdout, stderr=r.stderr, job_id=match.group(1) if match else None)
    except BaseException as e:
        receipt.update(status='UNKNOWN_OR_FAILED_INSPECT_SLURM', error=repr(e))
    with (operations/(a.mode+'_submission_receipt.json')).open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)
    assert receipt['status'] == 'SUBMITTED', 'Do not retry: inspect the recorded intent and Slurm'


if __name__ == '__main__':
    main()
