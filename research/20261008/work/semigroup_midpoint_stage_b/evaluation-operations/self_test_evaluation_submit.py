#!/usr/bin/env python3
"""Pure tiny gate and mocked scheduler fixtures; never submit or load a model."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import submit_evaluation as subject


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def inventory(path):
    return {str(p.relative_to(path)): {'sha256': subject.sha(p), 'bytes': p.stat().st_size}
            for p in path.rglob('*') if p.is_file() and p.name != 'artifact_manifest.json'}


def gate_fixture(parent):
    training, evaluation = parent / 'training', parent / 'evaluation'
    training.mkdir(parents=True)
    evidence = evaluation / 'evaluation-operations/evidence/gate-fixture'
    evidence.mkdir(parents=True)
    matrix = subject.read_json(Path(__file__).parent.parent / 'matrix.json')
    save(training / 'matrix.json', matrix)
    fields = ('JobID', 'JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES')
    records, rows = [], []
    for index, task in enumerate(matrix['tasks']):
        run = training / 'runs' / task['run_name']; run.mkdir(parents=True)
        save(run / 'RUN_RECEIPT.json', {'synthetic': True})
        save(run / 'environment.json', {'slurm': {'SLURM_JOB_ID': str(3000 + index)}})
        save(run / 'artifact_manifest.json', inventory(run))
        row = dict(zip(fields, [subject.ARRAY_JOB + '_' + str(index), str(3000 + index), 'COMPLETED', '0:0', '60', 'fixture', 'gres/gpu=1']))
        rows.append(row)
        records.append({'task': task, 'passed': True, 'scheduler_row': row, 'run_dir': str(run),
                        'artifact_manifest_sha256': subject.sha(run / 'artifact_manifest.json'),
                        'run_receipt_sha256': subject.sha(run / 'RUN_RECEIPT.json')})
    raw = evidence / 'sacct_with_header.txt'
    raw.write_text('|'.join(fields) + '\n' + ''.join('|'.join(row[k] for k in fields) + '\n' for row in rows))
    save(evidence / 'sacct_query.json', {'command': ['sacct', '--jobs=' + subject.ARRAY_JOB, '--parsable2',
        '--format=' + ','.join(fields)], 'returncode': 0, 'raw_sacct_sha256': subject.sha(raw)})
    save(evidence / 'ALL_TRAINING_GATE.json', {'tasks': records, 'status': 'PASS', 'unique_raw_job_ids': 12,
                                            'all_tasks_required_before_any_inference': True})
    save(evidence / 'OPERATIONS_REPORT.json', {'schema': 'stage_b_matrix_evaluation_operations_v1', 'mode': 'gate',
        'status': 'ALL_TRAINING_GATES_PASSED_NO_INFERENCE', 'gate_task_count': 12, 'optimizer_updates': 0,
        'evaluation_started': False, 'results': [], 'training_array_job': subject.ARRAY_JOB,
        'training_root': str(training), 'evaluation_root': str(evaluation), 'helper_sha256': subject.MATRIX_HELPER_SHA,
        'payload_manifest_sha256': subject.PAYLOAD_SHA, 'evaluator_sha256': subject.EVALUATOR_SHA,
        'evaluator_freeze_sha256': subject.FREEZE_SHA})
    save(evidence / 'artifact_manifest.json', inventory(evidence))
    return training, evaluation, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    def passed(name):
        checks.append({'name': name, 'passed': True})
    def rejects(name, operation):
        try:
            operation()
        except (ValueError, RuntimeError, FileExistsError):
            passed(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    with tempfile.TemporaryDirectory(prefix='evaluation-submit-fixture-') as temp_name:
        temp = Path(temp_name).resolve()
        training, evaluation, evidence = gate_fixture(temp / 'gates')
        base_files = {p: p.read_bytes() for p in evidence.iterdir()}
        result = subject.verify_gate(training, evaluation, 'gate-fixture')
        assert result['status'] == 'PASS' and result['training_tasks'] == 12
        passed('all_twelve_completed_zero_exit_unique_raw_ids_gate')
        for name, filename, mutate in [
            ('wrong_array', 'OPERATIONS_REPORT.json', lambda d: d.update(training_array_job='999999')),
            ('inference_mode_cannot_replace_gate', 'OPERATIONS_REPORT.json', lambda d: d.update(mode='evaluate')),
            ('gate_did_model_work', 'OPERATIONS_REPORT.json', lambda d: d.update(evaluation_started=True)),
            ('missing_cell', 'ALL_TRAINING_GATE.json', lambda d: d['tasks'].pop()),
            ('failed_cell', 'ALL_TRAINING_GATE.json', lambda d: d['tasks'][4].update(passed=False)),
            ('duplicate_task', 'ALL_TRAINING_GATE.json', lambda d: d['tasks'].__setitem__(5, copy.deepcopy(d['tasks'][4]))),
            ('changed_helper_identity', 'OPERATIONS_REPORT.json', lambda d: d.update(helper_sha256='0' * 64))]:
            for path, raw in base_files.items(): path.write_bytes(raw)
            data = subject.read_json(evidence / filename); mutate(data); save(evidence / filename, data)
            save(evidence / 'artifact_manifest.json', inventory(evidence))
            rejects(name, lambda: subject.verify_gate(training, evaluation, 'gate-fixture'))
        for path, raw in base_files.items(): path.write_bytes(raw)
        raw = evidence / 'sacct_with_header.txt'
        raw.write_text(raw.read_text().replace('|COMPLETED|0:0|', '|RUNNING|0:0|', 1))
        query = subject.read_json(evidence / 'sacct_query.json'); query['raw_sacct_sha256'] = subject.sha(raw)
        save(evidence / 'sacct_query.json', query); save(evidence / 'artifact_manifest.json', inventory(evidence))
        rejects('running_training_task_blocks_even_when_gate_says_pass', lambda: subject.verify_gate(training, evaluation, 'gate-fixture'))
        for path, raw in base_files.items(): path.write_bytes(raw)
        env_path = training / 'runs' / subject.read_json(training / 'matrix.json')['tasks'][0]['run_name'] / 'environment.json'
        save(env_path, {'slurm': {'SLURM_JOB_ID': '9000'}})
        rejects('raw_allocation_id_bound_to_model_environment', lambda: subject.verify_gate(training, evaluation, 'gate-fixture'))
        command = subject.command_for(subject.BROOT, subject.EROOT, 'evaluate-fixture')
        assert '--gpus=1' in command and '--cpus-per-task=10' in command and '--time=02:00:00' in command
        assert not any(value.startswith('--array') for value in command)
        assert command[-4:] == [str(subject.BROOT), str(subject.EROOT), '2349407', 'evaluate-fixture']
        passed('fixed_one_gpu_ten_cpus_two_hours_no_array_existing_worker')
        for index, (label, reply, expected) in enumerate([
            ('valid', subprocess.CompletedProcess(command, 0, '8888;bscc26a\n', ''), 'SUBMITTED'),
            ('wrong_cluster', subprocess.CompletedProcess(command, 0, '8888;other\n', ''), 'SUBMISSION_FAILED_OR_UNKNOWN_INSPECT_SCHEDULER'),
            ('invalid_stdout', subprocess.CompletedProcess(command, 0, 'accepted maybe', 'fixture'), 'SUBMISSION_FAILED_OR_UNKNOWN_INSPECT_SCHEDULER'),
            ('nonzero', subprocess.CompletedProcess(command, 1, '', 'fixture failed'), 'SUBMISSION_FAILED_OR_UNKNOWN_INSPECT_SCHEDULER'),
            ('timeout', subprocess.TimeoutExpired(command, 90, output=b'8888', stderr=b'fixture timeout'), 'SUBMISSION_OUTCOME_UNKNOWN_INSPECT_SCHEDULER')]):
            ops = temp / ('submission-' + label); ops.mkdir()
            intent = {'command': command, 'gate_attempt_id': 'gate-fixture', 'evaluate_attempt_id': 'evaluate-fixture'}
            calls = []
            def mocked(command, **kwargs):
                assert (ops / 'evaluation_submission_intent.json').is_file()
                calls.append(command)
                if isinstance(reply, BaseException): raise reply
                return reply
            with patch.object(subject.subprocess, 'run', side_effect=mocked):
                receipt = subject.submit_once(ops, intent)
                assert receipt['status'] == expected and len(calls) == 1
                assert 'stderr' in receipt
                passed('intent_precedes_single_sbatch_and_persists_' + label)
                rejects('never_retries_' + label, lambda: subject.submit_once(ops, intent))
                assert len(calls) == 1
        ops = temp / 'intent-only'; ops.mkdir(); save(ops / 'evaluation_submission_intent.json', {'unknown': True})
        with patch.object(subject.subprocess, 'run', side_effect=AssertionError('No call allowed')):
            rejects('unknown_outcome_intent_without_receipt_blocks_retry', lambda: subject.submit_once(ops, intent))
        checker = subject.resource_module(Path(__file__).parent.parent)
        assert checker.WALL_SECONDS == 7200
        state = {'cluster': 'bscc26a', 'wall': '02:00:00', 'queue': ''}
        def config_reply(command, **kwargs):
            assert command == ['scontrol', 'show', 'config']
            return SimpleNamespace(returncode=0, stdout='ClusterName = ' + state['cluster'] + '\n', stderr='')
        def query_reply(command, evidence):
            if command[:4] == ['sacctmgr', '-nP', 'show', 'assoc']:
                return 'bscc26a|scv7tsq|scv7tsq||normal||16|16|cpu=80,gres/gpu=8||\n'
            if command[:3] == ['scontrol', 'show', 'partition']:
                return 'PartitionName=gpu State=UP MaxTime=' + state['wall'] + ' QoS=gpu\n'
            if command[:4] == ['sacctmgr', '-nP', 'show', 'qos']:
                return 'gpu||||||\nnormal||||||\n'
            assert command[0] == 'squeue' and '--array' in command
            return state['queue']
        with patch.object(checker.subprocess, 'run', side_effect=config_reply), patch.object(checker, 'query', side_effect=query_reply):
            quota = checker.resources('validate', [])
            assert quota['new_potential_tasks'] == quota['new_max_concurrent_tasks'] == 1
            passed('pinned_resource_checker_accepts_single_job_at_two_hour_boundary')
            state['wall'] = '01:59:59'
            rejects('resource_checker_rejects_below_two_hour_budget', lambda: checker.resources('validate', []))
            state['wall'] = '02:00:00'; state['cluster'] = 'other'
            rejects('wrong_live_cluster_rejected', lambda: checker.resources('validate', []))
            state['cluster'] = 'bscc26a'; state['queue'] = ''.join(f'{i}|RUNNING|scv7tsq|gpu|10|gres/gpu:1|normal\n' for i in range(8))
            rejects('full_cpu_gpu_account_headroom_blocks_new_job', lambda: checker.resources('validate', []))
        assert subject.sha(Path(__file__).parent.parent / 'submit_stage_b.py') == subject.RESOURCE_HELPER_SHA
        passed('resource_helper_source_unmodified')
    report = {'status': 'SYNTHETIC_EVALUATION_SUBMISSION_TESTS_PASS', 'checks_passed': len(checks), 'checks': checks,
        'submitter_sha256': subject.sha(Path(__file__).with_name('submit_evaluation.py')),
        'test_script_sha256': subject.sha(__file__), 'real_scheduler_calls': 0, 'network_calls': 0, 'optimizer_updates': 0,
        'scope': 'Tiny synthetic gate JSON and mocked scheduler process results only; no platform or model access.'}
    subject.write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'checks_passed': len(checks), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
