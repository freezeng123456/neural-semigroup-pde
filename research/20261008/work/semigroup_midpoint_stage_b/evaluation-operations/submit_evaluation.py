#!/usr/bin/env python3
"""Submit the one authorized Stage B final-evaluation allocation after fresh gates.

Fixed training array 2349407; one GPU, ten CPUs, two hours, no array. The existing
worker verifies one visible V100 before inference. This file does not load models.
It invokes the frozen helper's read-only gate with a new attempt ID, then queries
live resource limits. Any submission intent blocks retries, even with new IDs.
"""
import argparse
import csv
from datetime import datetime, timezone
import getpass
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.dont_write_bytecode = True
PARENT = Path('/data02/run01/scv7tsq/neural-semigroup')
BROOT = PARENT / '20261004-midpoint-stage-b-r1'
EROOT = PARENT / '20261004-midpoint-stage-b-evaluation-r1'
PYTHON = Path('/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python')
ARRAY_JOB = '2349407'
USER = ACCOUNT = 'scv7tsq'
CLUSTER = 'bscc26a'
PAYLOAD_SHA = 'c30e2208cff36ba79b91da43b947fd85f7f0c23088d5020e2824f906704df466'
DEPLOY_SHA = 'b19cef09e9b4dbcdd6d175cde0dd898c19d447ba689d4f872f959f0a9645ca0b'
RESOURCE_HELPER_SHA = '037438e0944660ccc8d2755c9985a0b1f0f8903a4c684a1d32f3a46a96752d86'
MATRIX_HELPER_SHA = '81c1b1c7b8c25476dab7f355b54c06f99c283d2f5e9254afeed97804a0e97ec0'
SHELL_SHA = 'c19b1bb67c965aa79b3eb2df60b458642f6c447b512af76824d62ac8ea515957'
EVALUATOR_SHA = 'f1027d00f040d2c53ec70cda8513f216192f07b60699cf3bde7d6f46d183fa2a'
FREEZE_SHA = 'c6b07aa712957f63c2fd60ca09ad32752ee81692b0ff5337582825d7efbba63b'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def safe_file(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and relative.parts and '..' not in relative.parts
            and str(relative) == name, 'Invalid file path')
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink in evidence/input path')
    require(path.is_file(), 'Missing input file: ' + str(path))
    return path


def verify_manifest(root, specs, exact=False):
    require(isinstance(specs, dict) and specs, 'Missing manifest')
    for name, spec in specs.items():
        path = safe_file(root, name)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Changed evidence/input: ' + str(path))
    if exact:
        actual = {str(path.relative_to(root)) for path in root.rglob('*')
                  if path.is_file() and 'matplotlib-cache' not in path.relative_to(root).parts
                  and path != root / 'artifact_manifest.json'}
        require(actual == set(specs), 'Gate inventory is not exact')


def verify_deployment(training, evaluation):
    require(sha(safe_file(training, 'PAYLOAD_MANIFEST.json')) == PAYLOAD_SHA, 'Training payload manifest changed')
    require(sha(safe_file(evaluation, 'DEPLOY_MANIFEST.json')) == DEPLOY_SHA, 'Evaluation deployment manifest changed')
    deployment = read_json(evaluation / 'DEPLOY_MANIFEST.json')
    require(deployment['schema'] == 'stage_b_evaluation_deployment_v1'
            and deployment['prefix'] == evaluation.name and deployment['training_array_job'] == ARRAY_JOB
            and deployment['training_payload_manifest_sha256'] == PAYLOAD_SHA, 'Unapproved deployment/array')
    verify_manifest(evaluation, deployment['files'])
    operations = evaluation / 'evaluation-operations'
    for name, expected in {'evaluate_matrix.py': MATRIX_HELPER_SHA, 'slurm_evaluate_matrix.sh': SHELL_SHA,
                           'evaluate_final.py': EVALUATOR_SHA, 'EVALUATOR_FREEZE.json': FREEZE_SHA}.items():
        require(sha(safe_file(operations, name)) == expected, 'Frozen evaluation source changed')
    require(sha(safe_file(training, 'submit_stage_b.py')) == RESOURCE_HELPER_SHA, 'Resource checker changed')


def resource_module(training):
    path = safe_file(training, 'submit_stage_b.py')
    require(sha(path) == RESOURCE_HELPER_SHA, 'Only the pinned read-only resource checker may be imported')
    spec = importlib.util.spec_from_file_location('frozen_stage_b_resource_checker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require((module.USER, module.ACCOUNT, module.CLUSTER, module.CPUS, module.GPUS)
            == (USER, ACCOUNT, CLUSTER, 10, 1), 'Resource checker constants differ')
    # The frozen function already supports a one-job request via mode=validate.
    # Only its in-memory budget parameter changes; no file or training code is edited.
    module.WALL_SECONDS = 7200
    return module


def verify_gate(training, evaluation, gate_id):
    evidence = evaluation / 'evaluation-operations/evidence' / gate_id
    require(evidence.is_dir() and not evidence.is_symlink(), 'Gate evidence missing')
    verify_manifest(evidence, read_json(safe_file(evidence, 'artifact_manifest.json')), exact=True)
    require(not (evidence / 'OPERATIONS_FAILURE.json').exists(), 'Gate failure marker present')
    report = read_json(safe_file(evidence, 'OPERATIONS_REPORT.json'))
    require(report['schema'] == 'stage_b_matrix_evaluation_operations_v1' and report['mode'] == 'gate'
            and report['status'] == 'ALL_TRAINING_GATES_PASSED_NO_INFERENCE' and report['gate_task_count'] == 12
            and report['optimizer_updates'] == 0 and report['evaluation_started'] is False
            and report['results'] == [], 'Only a successful read-only twelve-model gate is accepted')
    require(report['training_array_job'] == ARRAY_JOB and report['training_root'] == str(training)
            and report['evaluation_root'] == str(evaluation) and report['helper_sha256'] == MATRIX_HELPER_SHA
            and report['payload_manifest_sha256'] == PAYLOAD_SHA and report['evaluator_sha256'] == EVALUATOR_SHA
            and report['evaluator_freeze_sha256'] == FREEZE_SHA, 'Gate identifies different inputs')
    gate = read_json(safe_file(evidence, 'ALL_TRAINING_GATE.json'))
    tasks = read_json(training / 'matrix.json')['tasks']
    require(len(tasks) == 12 and len(gate['tasks']) == 12 and gate['status'] == 'PASS'
            and gate['unique_raw_job_ids'] == 12 and gate['all_tasks_required_before_any_inference'] is True, 'All twelve gates must pass')
    raw_path = safe_file(evidence, 'sacct_with_header.txt')
    query = read_json(safe_file(evidence, 'sacct_query.json'))
    command = ['sacct', '--jobs=' + ARRAY_JOB, '--parsable2',
               '--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,AllocTRES']
    require(query['command'] == command and query['returncode'] == 0 and query['raw_sacct_sha256'] == sha(raw_path), 'Missing genuine gate accounting query evidence')
    rows = list(csv.DictReader(io.StringIO(raw_path.read_text()), delimiter='|'))
    allocation_rows = [row for row in rows if re.fullmatch(ARRAY_JOB + r'_\d+', row.get('JobID', ''))]
    expected_ids = {ARRAY_JOB + '_' + str(i) for i in range(12)}
    require(len(allocation_rows) == 12 and {row['JobID'] for row in allocation_rows} == expected_ids, 'Accounting omits/adds/duplicates array tasks')
    mapped = {row['JobID']: row for row in allocation_rows}
    raw_ids = set()
    for position, (task, result) in enumerate(zip(tasks, gate['tasks'])):
        require(task['index'] == position and result['task'] == task and result['passed'] is True, 'Gate task order/identity/status mismatch')
        row = mapped[ARRAY_JOB + '_' + str(position)]
        require(row['State'] == 'COMPLETED' and row['ExitCode'] == '0:0' and row['JobIDRaw'].isdigit(), 'Training task is not COMPLETED 0:0')
        require(result['scheduler_row'] == row, 'Gate/accounting row mismatch')
        require(result['run_dir'] == str(training / 'runs' / task['run_name']), 'Wrong training run path')
        run = training / 'runs' / task['run_name']
        require(sha(safe_file(run, 'artifact_manifest.json')) == result['artifact_manifest_sha256']
                and sha(safe_file(run, 'RUN_RECEIPT.json')) == result['run_receipt_sha256'], 'Training receipt/inventory changed after gate')
        require(str(read_json(safe_file(run, 'environment.json'))['slurm']['SLURM_JOB_ID']) == row['JobIDRaw'], 'Accounting raw ID does not belong to this trained model')
        raw_ids.add(row['JobIDRaw'])
    require(len(raw_ids) == 12, 'Duplicate raw allocation ID')
    return {'gate_attempt_id': gate_id, 'status': 'PASS', 'training_array_job': ARRAY_JOB, 'training_tasks': 12,
            'operations_report_sha256': sha(evidence / 'OPERATIONS_REPORT.json'),
            'all_training_gate_sha256': sha(evidence / 'ALL_TRAINING_GATE.json'),
            'artifact_manifest_sha256': sha(evidence / 'artifact_manifest.json'), 'raw_sacct_sha256': sha(raw_path)}


def command_for(training, evaluation, evaluate_id):
    return ['sbatch', '--parsable', '--partition=gpu', '--account=' + ACCOUNT,
            '--gpus=1', '--ntasks=1', '--cpus-per-task=10', '--time=02:00:00',
            '--job-name=sg-mid-eval', '--chdir=' + str(training),
            '--output=' + str(evaluation / 'logs' / (evaluate_id + '-%j.log')),
            str(evaluation / 'evaluation-operations/slurm_evaluate_matrix.sh'),
            str(training), str(evaluation), ARRAY_JOB, evaluate_id]


def submit_once(operations, intent):
    intent_path = operations / 'evaluation_submission_intent.json'
    receipt_path = operations / 'evaluation_submission_receipt.json'
    require(not receipt_path.exists() and not receipt_path.is_symlink(), 'Submission receipt already exists')
    write_json(intent_path, intent)  # Exclusive and fsynced before the only mutating scheduler call.
    receipt = {'schema': 'stage_b_evaluation_submission_receipt_v1', 'intent_sha256': sha(intent_path),
               'command': intent['command'], 'gate_attempt_id': intent['gate_attempt_id'],
               'evaluate_attempt_id': intent['evaluate_attempt_id'], 'training_array_job': ARRAY_JOB}
    try:
        reply = subprocess.run(intent['command'], capture_output=True, text=True, timeout=90, check=False)
        receipt.update(returncode=reply.returncode, stdout=reply.stdout, stderr=reply.stderr)
        match = re.fullmatch(r'(\d+)(?:;([A-Za-z0-9_-]+))?', reply.stdout.strip())
        if reply.returncode == 0 and match and match.group(2) in (None, CLUSTER):
            receipt.update(status='SUBMITTED', job_id=match.group(1))
        else:
            receipt['status'] = 'SUBMISSION_FAILED_OR_UNKNOWN_INSPECT_SCHEDULER'
    except BaseException as error:
        receipt.update(status='SUBMISSION_OUTCOME_UNKNOWN_INSPECT_SCHEDULER', error=repr(error))
        if isinstance(error, subprocess.TimeoutExpired):
            for key in ('stdout', 'stderr'):
                value = getattr(error, key, None)
                receipt[key] = value.decode(errors='replace') if isinstance(value, bytes) else value
    receipt['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(receipt_path, receipt)
    return receipt


def run(args):
    require(getpass.getuser() == USER, 'Only the authorized Beijing cluster account may submit')
    training, evaluation = args.training_root.resolve(strict=True), args.evaluation_root.resolve(strict=True)
    require(training == BROOT and evaluation == EROOT, 'Only the already deployed training/evaluation roots are authorized')
    operations = evaluation / 'evaluation-operations'
    require(Path(__file__).resolve().parent == operations, 'Install this submitter in EROOT/evaluation-operations')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.gate_attempt_id)
            and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.evaluate_attempt_id)
            and args.gate_attempt_id != args.evaluate_attempt_id, 'Use two different canonical attempt IDs')
    for name in ('evaluation_submission_intent.json', 'evaluation_submission_receipt.json'):
        require(not (operations / name).exists() and not (operations / name).is_symlink(), 'Prior submission evidence exists; inspect scheduler, never retry blindly')
    for attempt in (args.gate_attempt_id, args.evaluate_attempt_id):
        path = operations / 'evidence' / attempt
        require(not path.exists() and not path.is_symlink(), 'Attempt already exists: ' + attempt)
    require(not (operations / 'evidence').is_symlink() and not (evaluation / 'results').is_symlink(), 'Output root symlink forbidden')
    require(not (evaluation / 'results').exists() or not any((evaluation / 'results').iterdir()), 'A final evaluation output already exists')
    verify_deployment(training, evaluation)
    checker = resource_module(training)
    checks_root = operations / 'submission-checks'
    require(not checks_root.is_symlink(), 'Submission evidence root symlink forbidden')
    checks_root.mkdir(exist_ok=True)
    checks = checks_root / args.gate_attempt_id
    checks.mkdir(exist_ok=False)
    query_evidence = []
    try:
        gate_command = [str(PYTHON), '-B', '-u', str(operations / 'evaluate_matrix.py'),
                        '--training-root', str(training), '--evaluation-root', str(evaluation),
                        '--training-array-job', ARRAY_JOB, '--attempt-id', args.gate_attempt_id, '--mode', 'gate']
        write_json(checks / 'GATE_COMMAND.json', {'command': gate_command, 'inference': False})
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        with (checks / 'gate.stdout.txt').open('x') as stdout, (checks / 'gate.stderr.txt').open('x') as stderr:
            reply = subprocess.run(gate_command, stdout=stdout, stderr=stderr, env=environment, timeout=900, check=False)
        write_json(checks / 'GATE_EXIT.json', {'returncode': reply.returncode})
        require(reply.returncode == 0, 'Read-only all-model gate failed; no submission occurred')
        gate = verify_gate(training, evaluation, args.gate_attempt_id)
        quota = checker.resources('validate', query_evidence)
        require(quota['new_potential_tasks'] == quota['new_max_concurrent_tasks'] == 1, 'Resource check must cover exactly one new job')
        write_json(checks / 'RESOURCE_CHECK.json', {'wall_seconds': 7200, 'resource_check': quota,
                    'read_only_scheduler_evidence': query_evidence, 'frozen_checker_sha256': RESOURCE_HELPER_SHA})
        verify_deployment(training, evaluation)
        require(not (evaluation / 'results').exists() or not any((evaluation / 'results').iterdir()), 'Evaluation outputs appeared during gates')
        require(not (operations / 'evidence' / args.evaluate_attempt_id).exists(), 'Evaluation attempt appeared during gates')
        logs = evaluation / 'logs'
        require(not logs.is_symlink(), 'Log directory symlink forbidden')
        logs.mkdir(exist_ok=True)
        intent = {'schema': 'stage_b_evaluation_submission_intent_v1', 'training_array_job': ARRAY_JOB,
                  'gate_attempt_id': args.gate_attempt_id, 'evaluate_attempt_id': args.evaluate_attempt_id,
                  'command': command_for(training, evaluation, args.evaluate_attempt_id),
                  'created_at_utc': datetime.now(timezone.utc).isoformat(), 'gate': gate,
                  'resource_check': quota, 'read_only_scheduler_evidence': query_evidence,
                  'requested_resources': {'gpus': 1, 'cpus': 10, 'wall_seconds': 7200, 'array': False,
                                          'gpu_validation': 'Frozen worker requires one V100 sm70 before inference'},
                  'payload_manifest_sha256': PAYLOAD_SHA, 'deploy_manifest_sha256': DEPLOY_SHA,
                  'submitter_sha256': sha(__file__), 'resource_checker_sha256': RESOURCE_HELPER_SHA,
                  'retry_policy': 'Any intent/failed/unknown sbatch outcome forbids automatic resubmission; inspect scheduler manually.'}
        receipt = submit_once(operations, intent)
        print(json.dumps(receipt, indent=2), flush=True)
        require(receipt['status'] == 'SUBMITTED', 'Submission outcome not confirmed; inspect persisted intent/receipt and live scheduler')
        return receipt
    except BaseException as error:
        write_json(checks / 'STOPPED.json', {'status': 'STOPPED', 'error': repr(error),
                   'submission_intent_present': (operations / 'evaluation_submission_intent.json').exists(),
                   'read_only_scheduler_evidence': query_evidence})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--evaluation-root', type=Path, required=True)
    p.add_argument('--gate-attempt-id', required=True)
    p.add_argument('--evaluate-attempt-id', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
