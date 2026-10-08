#!/usr/bin/env python3
"""Gate and sequentially evaluate the twelve completed Stage B final models.

This operations helper never submits jobs, trains, or imports the evaluator.
gate mode only queries sacct and checks files; evaluate mode additionally requires
a Slurm allocation with exactly one V100 and invokes the frozen evaluator twelve
times sequentially. A fresh attempt directory is mandatory. Existing model
evaluation outputs are never overwritten or silently skipped.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
COMMIT = '1e629104a12a9c3e3064996cbd4e301cec641ad8'
PAYLOAD_SHA = 'c30e2208cff36ba79b91da43b947fd85f7f0c23088d5020e2824f906704df466'
EVALUATOR_SHA = 'f1027d00f040d2c53ec70cda8513f216192f07b60699cf3bde7d6f46d183fa2a'
FREEZE_SHA = 'c6b07aa712957f63c2fd60ca09ad32752ee81692b0ff5337582825d7efbba63b'
DATA_SHA = '1092743121a2c6cc733da44d034f0399d61c88d1d6b0016e75c57313bfc7465b'
TEST_SHA = 'a1e0a3a0fdf63c1ec3622bc21bd7889bf4a14078f86d63e07d8ae2bab74d926a'
PROJECT_PARENT = Path('/data02/run01/scv7tsq/neural-semigroup')
PYTHON = Path('/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python')
DEPS = PROJECT_PARENT / '20261004-due-vorticity-1e62910-r1/deps'
DEFAULT_TEST = PROJECT_PARENT / '20261004-due-vorticity-1e62910-r1/data/VorticityOSG_test.mat'
ARMS = ('gdsg500', 'single500', 'midpoint500', 'gdsg333')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, readonly=False):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    if readonly:
        Path(path).chmod(0o444)


def write_bytes(path, value):
    with Path(path).open('xb') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    Path(path).chmod(0o444)


def safe_file(root, relative):
    name = PurePosixPath(relative)
    require(not name.is_absolute() and '..' not in name.parts and str(name) == relative, 'Unsafe manifest path')
    path = root / relative
    require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), 'Missing or escaping file: ' + str(path))
    return path


def inventory(root, specs, complete=False):
    require(isinstance(specs, dict) and bool(specs), 'Empty file manifest')
    for name, spec in specs.items():
        path = safe_file(root, name)
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Artifact mismatch: ' + str(path))
    if complete:
        actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()
                  and 'matplotlib-cache' not in path.relative_to(root).parts
                  and path != root / 'artifact_manifest.json'}
        require(actual == set(specs), 'Incomplete run inventory: ' + str(root))
    return len(specs)


def scheduler_rows(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8')), delimiter='|'))
    require(rows and {'JobID', 'JobIDRaw', 'State', 'ExitCode'} <= set(rows[0]), 'sacct must return the four named headers')
    return [{key: str(row[key]).strip() for key in ('JobID', 'JobIDRaw', 'State', 'ExitCode', 'ElapsedRaw', 'NodeList', 'AllocTRES')
             if key in row} for row in rows]


def matrix_tasks(matrix):
    tasks = matrix['tasks']
    require(len(tasks) == 12, 'Frozen matrix must have twelve tasks')
    require({(task['arm'], task['seed']) for task in tasks} == {(arm, seed) for arm in ARMS for seed in range(3)}, 'Missing/duplicate matrix cell')
    require(len({task['run_name'] for task in tasks}) == 12, 'Duplicate run directory')
    for index, task in enumerate(tasks):
        require(task['index'] == index and re.fullmatch(r'[a-z0-9-]+', task['run_name']), 'Invalid matrix ordering/name')
        epochs = 333 if task['arm'] == 'gdsg333' else 500
        calls = 6 if task['arm'].startswith('gdsg') else 4
        require(task['epochs'] == epochs and task['updates'] == epochs * 100
                and task['forwards_per_batch'] == calls and task['total_forward_calls'] == epochs * 100 * calls, 'Changed matrix budget')
    return tasks


def training_gate(root, task, master_job, rows, payload):
    """No torch/checkpoint loading here; final evaluator performs strict loads."""
    job_id = master_job + '_' + str(task['index'])
    matches = [row for row in rows if row['JobID'] == job_id]
    require(len(matches) == 1, 'Expected exactly one sacct allocation row: ' + job_id)
    row = matches[0]
    require(row['State'] == 'COMPLETED' and row['ExitCode'] == '0:0', 'Training task has not cleanly completed: ' + job_id)
    require(re.fullmatch(r'\d+', row['JobIDRaw']), 'Expected numeric JobIDRaw')
    run = root / 'runs' / task['run_name']
    specs = read_json(run / 'artifact_manifest.json')
    required = {'configuration.json', 'environment.json', 'RUN_RECEIPT.json', 'final_state.pt', 'final_network.pt', 'updates.jsonl', 'epochs.jsonl', 'validation_final_prediction.npy', 'validation_final_metrics.npz'}
    require(required <= set(specs), 'Missing final artifacts: ' + task['run_name'])
    count = inventory(run, specs, complete=True)
    receipt, environment, config = (read_json(run / name) for name in ('RUN_RECEIPT.json', 'environment.json', 'configuration.json'))
    require(str(environment['slurm']['SLURM_JOB_ID']) == row['JobIDRaw'], 'Array task JobIDRaw does not match training environment')
    require(environment['mode'] == 'full' and environment['device'] == 'cuda' and 'V100' in environment['gpu_name'], 'Not a full V100 training run')
    require(receipt['status'] == 'STAGE_B_FULL_COMPLETE' and receipt['config'] == config, 'Training did not produce a final completion receipt')
    expected = {'arm': task['arm'], 'seed': task['seed'], 'epochs': task['epochs'], 'planned_updates': task['updates'], 'batch_size': 20, 'ntrain': 2000, 'batches_per_epoch': 100, 'lambda': 1.0, 'Q': 1, 'optimizer': 'adam', 'lr': .001, 'loss': 'rel_l2', 'scheduler': 'cosine', 'scheduler_T_max': task['updates'], 'multiscale': False, 'dtype': 'float32', 'data_payload_sha256': DATA_SHA, 'source_commit': COMMIT, 'selection': 'scheduled final epoch only'}
    require(all(config[key] == value for key, value in expected.items()), 'Frozen training configuration differs')
    require(receipt['actual_updates'] == task['updates'] and receipt['forward_calls'] == task['total_forward_calls']
            and receipt['cursor'] == {'epoch': task['epochs'], 'batch': 0} and receipt['final_learning_rate'] == 0, 'Scheduled training budget not complete')
    require(receipt['parameters_changed'] is True and receipt['initial_parameter_sha256'] != receipt['final_parameter_sha256'], 'Parameters did not change')
    final = receipt['final_validation']
    require(final['scheduled_final'] is True and final['valid_metric'] is True and final['status'] == 'FINITE_ROLLOUT'
            and final['epoch'] == task['epochs'] and final['updates'] == task['updates'], 'Final development rollout did not complete')
    require(len(receipt['history']) == task['epochs'] and all(item['epoch'] == index and item['updates'] == index * 100 for index, item in enumerate(receipt['history'], 1)), 'Incomplete epoch history')
    require(receipt['input_sha256'] == environment['input_sha256'], 'Training input identities disagree')
    expected_inputs = {'official_source_manifest.json', 'data-r1/DATA_MANIFEST.json', 'data-r1/development_data.npz', 'train.py', 'losses.py', 'PROTOCOL.zh.md'}
    require(set(receipt['input_sha256']) == {str(root / relative) for relative in expected_inputs}, 'Training input paths do not bind this BROOT')
    for relative in expected_inputs:
        require(receipt['input_sha256'][str(root / relative)] == payload['files'][relative]['sha256'], 'Training input content differs from frozen payload')
    require(sha(run / final['prediction_file']) == final['prediction_sha256'] and sha(run / final['metrics_file']) == final['metrics_sha256'], 'Final validation file identity differs')
    return {'passed': True, 'task': task, 'run_dir': str(run), 'scheduler_row': row,
            'artifact_count': count, 'artifact_manifest_sha256': sha(run / 'artifact_manifest.json'),
            'run_receipt_sha256': sha(run / 'RUN_RECEIPT.json'),
            'final_parameter_sha256': receipt['final_parameter_sha256'],
            'checkpoint_strict_load_deferred_to_frozen_evaluator': True}


def runtime_identity():
    import platform
    import numpy
    import scipy
    import matplotlib
    import yaml
    import torch
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                'matplotlib': matplotlib.__version__, 'PyYAML': yaml.__version__, 'torch': torch.__version__}
    require(versions == {'python': '3.11.7', 'numpy': '1.26.4', 'scipy': '1.11.4', 'matplotlib': '3.8.4', 'PyYAML': '6.0.2', 'torch': '2.5.1+cu121'}, 'Runtime version drift')
    require(Path(sys.executable).resolve() == PYTHON.resolve(), 'Wrong frozen runtime interpreter')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Exactly one visible GPU is required')
    require('V100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (7, 0), 'Expected V100 sm70')
    require(re.fullmatch(r'\d+', os.environ.get('SLURM_JOB_ID', '')), 'Evaluation mode requires a Slurm allocation')
    return {'versions': versions, 'executable': sys.executable, 'executable_sha256': sha(Path(sys.executable)),
            'gpu_name': torch.cuda.get_device_name(0), 'gpu_capability': list(torch.cuda.get_device_capability(0)),
            'visible_device_count': torch.cuda.device_count(), 'cuda_runtime': torch.version.cuda,
            'slurm': {key: os.environ.get(key) for key in ('SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_NODELIST', 'CUDA_VISIBLE_DEVICES')},
            'module_paths': {module.__name__: module.__file__ for module in (numpy, scipy, matplotlib, yaml, torch)}}


def output_manifest(root):
    return {str(path.relative_to(root)): {'sha256': sha(path), 'bytes': path.stat().st_size}
            for path in sorted(root.rglob('*')) if path.is_file() and 'matplotlib-cache' not in path.relative_to(root).parts}


def run(args):
    started = time.perf_counter()
    root, evaluation = args.training_root.resolve(strict=True), args.evaluation_root.resolve(strict=True)
    require(root.parent == PROJECT_PARENT and root.name.startswith('20261004-midpoint-stage-b-'), 'Unexpected training root')
    require(evaluation.parent == PROJECT_PARENT and evaluation != root and evaluation.name.startswith('20261004-midpoint-stage-b-evaluation-'), 'Evaluation must use its separate sibling root')
    operations = evaluation / 'evaluation-operations'
    require(Path(__file__).resolve().parent == operations, 'Place this helper in EROOT/evaluation-operations')
    require(re.fullmatch(r'\d+', args.training_array_job) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.attempt_id), 'Invalid job or attempt identifier')
    evidence = operations / 'evidence' / args.attempt_id
    evidence.mkdir(parents=True, exist_ok=False)
    rows, results = [], []
    status = {'schema': 'stage_b_matrix_evaluation_operations_v1', 'mode': args.mode,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'training_root': str(root),
              'evaluation_root': str(evaluation), 'training_array_job': args.training_array_job,
              'helper_sha256': sha(Path(__file__)), 'optimizer_updates': 0,
              'evaluation_started': False, 'results': results}
    try:
        write_json(evidence / 'INTENT.json', status, readonly=True)
        require(sha(root / 'PAYLOAD_MANIFEST.json') == PAYLOAD_SHA, 'Frozen training payload manifest changed')
        payload = read_json(root / 'PAYLOAD_MANIFEST.json')
        require(payload['source_commit'] == COMMIT and payload['runtime_python'] == str(PYTHON) and payload['shared_dependency_root'] == str(DEPS), 'Unexpected training environment provenance')
        inventory(root, payload['files'])
        require(sha(operations / 'EVALUATOR_FREEZE.json') == FREEZE_SHA, 'Evaluator freeze receipt changed')
        freeze = read_json(operations / 'EVALUATOR_FREEZE.json')
        require(freeze['schema'] == 'stage_b_final_evaluator_freeze_v1' and freeze['training_payload_changed'] is False, 'Unexpected evaluator freeze')
        inventory(operations, freeze['files'])
        require(sha(operations / 'evaluate_final.py') == EVALUATOR_SHA, 'Frozen evaluator changed')
        test = args.test_mat.resolve(strict=True)
        require(sha(test) == TEST_SHA, 'Original test MAT identity differs')
        tasks = matrix_tasks(read_json(root / 'matrix.json'))
        status.update(payload_manifest_sha256=PAYLOAD_SHA, evaluator_freeze_sha256=FREEZE_SHA, evaluator_sha256=EVALUATOR_SHA, test_mat_sha256=TEST_SHA)
        # A real read-only query is mandatory for every attempt. No cached or
        # headerless archive text is substituted for this evidence.
        command = ['sacct', '--jobs=' + args.training_array_job, '--parsable2',
                   '--format=JobID,JobIDRaw,State,ExitCode,ElapsedRaw,NodeList,AllocTRES']
        query_started = datetime.now(timezone.utc).isoformat()
        completed = subprocess.run(command, capture_output=True, timeout=90, check=False)
        raw_path = evidence / 'sacct_with_header.txt'
        write_bytes(raw_path, completed.stdout)
        write_bytes(evidence / 'sacct_stderr.txt', completed.stderr)
        write_json(evidence / 'sacct_query.json', {'command': command, 'started_at_utc': query_started,
                   'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'returncode': completed.returncode,
                   'raw_sacct_sha256': sha(raw_path)}, readonly=True)
        require(completed.returncode == 0, 'sacct query failed; inference was not started')
        rows = scheduler_rows(completed.stdout)
        gates = []
        for task in tasks:
            try:
                gates.append(training_gate(root, task, args.training_array_job, rows, payload))
            except Exception as error:
                gates.append({'passed': False, 'task': task, 'error': repr(error)})
        raw_ids = [item['scheduler_row']['JobIDRaw'] for item in gates if item['passed']]
        all_pass = all(item['passed'] for item in gates) and len(set(raw_ids)) == 12
        write_json(evidence / 'ALL_TRAINING_GATE.json', {'status': 'PASS' if all_pass else 'BLOCKED', 'tasks': gates,
                   'unique_raw_job_ids': len(set(raw_ids)), 'all_tasks_required_before_any_inference': True}, readonly=True)
        require(all_pass, 'At least one training task/accounting/artifact gate failed; no model evaluation started')
        receipts = evidence / 'completion-receipts'
        receipts.mkdir()
        for gate in gates:
            task, row = gate['task'], gate['scheduler_row']
            write_json(receipts / (task['run_name'] + '.json'), {
                'job_id': row['JobID'], 'job_id_raw': row['JobIDRaw'], 'state': 'COMPLETED', 'exit_code': '0:0',
                'sacct_command': command, 'raw_sacct_sha256': sha(raw_path), 'root': str(root),
                'run_name': task['run_name'], 'queried_at_utc': query_started,
                'elapsed_raw_seconds': int(row['ElapsedRaw']) if row.get('ElapsedRaw', '').isdigit() else None,
                'node_list': row.get('NodeList'), 'allocated_tres': row.get('AllocTRES')}, readonly=True)
        if args.mode == 'gate':
            status.update(status='ALL_TRAINING_GATES_PASSED_NO_INFERENCE', gate_task_count=12)
        else:
            os.environ['MPLBACKEND'] = 'Agg'
            os.environ['MPLCONFIGDIR'] = str(evidence / 'matplotlib-cache')
            status['runtime'] = runtime_identity()
            result_root = evaluation / 'results'
            require(all(not (result_root / task['run_name']).exists() for task in tasks), 'An evaluation output already exists; refuse overwrite or silent skip')
            result_root.mkdir(exist_ok=True)
            status['evaluation_started'] = True
            # Persist the complete twelve-cell index BEFORE any model process,
            # so cancellation or a wall-time limit cannot hide unstarted seeds.
            write_json(evidence / 'run_index.json', {'schema': 'stage_b_matrix_analysis_inputs_v1',
                       'runs': [{'arm': task['arm'], 'seed': task['seed'],
                                 'run_dir': str(root / 'runs' / task['run_name']),
                                 'evaluation_dir': str(result_root / task['run_name'])}
                                for task in tasks]}, readonly=True)
            write_json(evidence / 'EVALUATION_START.json', {**status, 'scheduled_models': 12,
                       'execution': 'Sequential; one evaluator process active at a time'}, readonly=True)
            # Every run keeps stdout/stderr, its output directory and an explicit
            # return status. One failure never removes a seed or hides later cells.
            with (evidence / 'progress.jsonl').open('x', buffering=1) as progress:
                for task, gate in zip(tasks, gates):
                    run_name = task['run_name']
                    record = {'arm': task['arm'], 'seed': task['seed'], 'run_name': run_name,
                              'run_dir': str(root / 'runs' / run_name), 'evaluation_dir': str(result_root / run_name)}
                    process_started = time.perf_counter()
                    try:
                        require(sha(operations / 'evaluate_final.py') == EVALUATOR_SHA and sha(root / 'PAYLOAD_MANIFEST.json') == PAYLOAD_SHA, 'Freeze changed between model evaluations')
                        require(sha(root / 'runs' / run_name / 'artifact_manifest.json') == gate['artifact_manifest_sha256'], 'Training artifact manifest changed after all-model gate')
                        eval_command = [str(PYTHON), '-B', '-u', str(operations / 'evaluate_final.py'),
                            '--run-dir', record['run_dir'], '--source-root', str(root / 'due'),
                            '--source-manifest', str(root / 'official_source_manifest.json'),
                            '--training-code-root', str(root), '--data-dir', str(root / 'data-r1'),
                            '--test-mat', str(test), '--completion-receipt', str(receipts / (run_name + '.json')),
                            '--raw-sacct', str(raw_path), '--output-dir', record['evaluation_dir'], '--device', 'cuda']
                        record['command'] = eval_command
                        # Preserve scheduler CUDA_VISIBLE_DEVICES; never replace it.
                        child_environment = os.environ.copy()
                        child_environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                            PYTHONPATH=str(DEPS) + ':' + str(root / 'due'), OMP_NUM_THREADS='1',
                            MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
                        with (evidence / (run_name + '.stdout.txt')).open('xb') as stdout, (evidence / (run_name + '.stderr.txt')).open('xb') as stderr:
                            reply = subprocess.run(eval_command, cwd=root, env=child_environment, stdout=stdout, stderr=stderr, check=False)
                        record['returncode'] = reply.returncode
                        require(reply.returncode == 0, 'Frozen evaluator exited unsuccessfully')
                        destination = result_root / run_name
                        specs = read_json(destination / 'artifact_manifest.json')
                        inventory(destination, specs, complete=True)
                        report = read_json(destination / 'EVALUATION_REPORT.json')
                        require(report['schema'] == 'stage_b_final_evaluation_v1' and report['status'] == 'EVALUATION_COMPLETE', 'Incomplete evaluator report')
                        require(report['arm'] == task['arm'] and report['seed'] == task['seed'] and report['optimizer_updates'] == 0
                                and report['parameter_sha256_before'] == report['parameter_sha256_after'] == gate['final_parameter_sha256'], 'Evaluator report not bound to this final model')
                        require(report['checks'] and all(item['passed'] is True for item in report['checks']), 'Evaluator engineering checks failed')
                        record.update(status='EVALUATION_COMPLETE', quality_status=report['quality_status'],
                                      report_sha256=sha(destination / 'EVALUATION_REPORT.json'),
                                      artifact_manifest_sha256=sha(destination / 'artifact_manifest.json'))
                    except Exception as error:
                        record.update(status='EVALUATION_FAILED', error=repr(error), traceback=traceback.format_exc())
                    record['elapsed_seconds'] = time.perf_counter() - process_started
                    results.append(record)
                    progress.write(json.dumps(record, allow_nan=False) + '\n')
                    progress.flush()
                    os.fsync(progress.fileno())
                    print(json.dumps({key: record.get(key) for key in ('run_name', 'status', 'quality_status', 'elapsed_seconds')}), flush=True)
            completed_count = sum(item['status'] == 'EVALUATION_COMPLETE' for item in results)
            status.update(status='ALL_EVALUATIONS_COMPLETE' if completed_count == 12 else 'PARTIAL_EVALUATION_FAILURE',
                          evaluation_complete_count=completed_count, evaluation_failure_count=12 - completed_count,
                          numerical_quality='Read each quality_status and raw predictions separately; execution completion does not imply accuracy or finite diagnostics.')
        status['elapsed_seconds'] = time.perf_counter() - started
        write_json(evidence / 'OPERATIONS_REPORT.json', status, readonly=True)
        write_json(evidence / 'artifact_manifest.json', output_manifest(evidence), readonly=True)
        return 0 if status['status'] != 'PARTIAL_EVALUATION_FAILURE' else 2
    except BaseException as error:
        status.update(status='OPERATIONS_FAILED', error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started)
        write_json(evidence / 'OPERATIONS_FAILURE.json', status, readonly=True)
        write_json(evidence / 'artifact_manifest.json', output_manifest(evidence), readonly=True)
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root', type=Path, required=True)
    p.add_argument('--evaluation-root', type=Path, required=True)
    p.add_argument('--training-array-job', required=True)
    p.add_argument('--attempt-id', required=True)
    p.add_argument('--mode', choices=('gate', 'evaluate'), required=True)
    p.add_argument('--test-mat', type=Path, default=DEFAULT_TEST)
    return p


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
