#!/usr/bin/env python3
"""Local synthetic regression and real-frozen-packer format integration; no Slurm calls."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


driver = load('synthetic_capture_r2', ROOT / 'pack_training_matrix_r2.py')
packer = load('real_frozen_training_packer', ROOT.parent / 'pack_evidence.py')
checks = []


def require(ok):
    assert ok


def check(name, function):
    function(); checks.append({'name': name, 'passed': True})


def rejected(function):
    try:
        function()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('Expected rejection')


def write(path, value):
    path.write_text(json.dumps(value))


def metadata(path):
    return {'bytes': path.stat().st_size, 'sha256': driver.sha(path)}


def scheduler_text():
    rows = ['JobID|JobIDRaw|State|ExitCode']
    for index in range(12):
        rows.append(f'{driver.ARRAY}_{index}|{90000 + index}|COMPLETED|0:0')
        rows.append(f'{driver.ARRAY}_{index}.batch|{90000 + index}.batch|COMPLETED|0:0')
    return '\n'.join(rows) + '\n'


def fixture(temp):
    root = Path(temp).resolve() / driver.BROOT.name; root.mkdir()
    (root / 'pack_evidence.py').write_bytes((ROOT.parent / 'pack_evidence.py').read_bytes())
    (root / 'matrix.json').write_bytes((ROOT.parent / 'matrix.json').read_bytes())
    (root / 'official_source_manifest.json').write_text('{}')
    (root / 'data-r1').mkdir(); (root / 'data-r1/development_data.npz').write_bytes(b'Synthetic data identity only')
    frozen = {'prefix': root.name, 'source_commit': 'synthetic-commit', 'files': {
        name: metadata(root / name) for name in ('pack_evidence.py', 'matrix.json',
                                               'official_source_manifest.json', 'data-r1/development_data.npz')}}
    write(root / 'PAYLOAD_MANIFEST.json', frozen)
    tasks = json.loads((root / 'matrix.json').read_text())['tasks']
    for task in tasks:
        run = root / 'runs' / task['run_name']; run.mkdir(parents=True)
        config = {'arm': task['arm'], 'seed': task['seed'], 'epochs': task['epochs'],
                  'planned_updates': task['updates'], 'source_commit': frozen['source_commit'],
                  'data_payload_sha256': frozen['files']['data-r1/development_data.npz']['sha256']}
        write(run / 'configuration.json', config)
        write(run / 'environment.json', {'mode': 'full', 'device': 'cuda', 'slurm': {'SLURM_JOB_ID': str(90000 + task['index'])}})
        for name in ('epochs.jsonl', 'updates.jsonl', 'final_network.pt', 'final_state.pt', 'latest_state.pt',
                     'validation_final_prediction.npy', 'validation_final_metrics.npz'):
            (run / name).write_bytes(b'Synthetic bytes; no model or numerical data loaded')
        final = {'valid_metric': True, 'final': True, 'epoch': task['epochs'], 'updates': task['updates'],
                 'prediction_file': 'validation_final_prediction.npy', 'metrics_file': 'validation_final_metrics.npz',
                 'prediction_sha256': driver.sha(run / 'validation_final_prediction.npy'),
                 'metrics_sha256': driver.sha(run / 'validation_final_metrics.npz')}
        write(run / 'RUN_RECEIPT.json', {'status': 'STAGE_B_FULL_COMPLETE', 'config': config,
              'actual_updates': task['updates'], 'cursor': {'epoch': task['epochs'], 'batch': 0},
              'forward_calls': task['total_forward_calls'], 'final_validation': final})
        write(run / 'artifact_manifest.json', {path.name: metadata(path) for path in run.iterdir()})
    (root / 'logs').mkdir(); (root / 'logs' / ('job-' + driver.ARRAY + '.log')).write_text('Synthetic completed training log')
    (root / 'operations').mkdir(); write(root / 'operations/submission.json', {'synthetic': True})
    recovery = root / 'recovery'; recovery.mkdir(); (recovery / 'MATRIX_CAPTURE').mkdir()
    write(recovery / 'MATRIX_CAPTURE/INTENT.json', {'synthetic_old_intent': True})
    (recovery / 'capture-r1.stdout.txt').write_text('Synthetic preserved old output')
    prior = {'status': 'STOPPED_INCOMPLETE', 'verified_capture_count': 0,
             'driver_sha256': driver.PRIOR_DRIVER_SHA, 'training_array_job': driver.ARRAY, 'root': str(root),
             'runs': [{'task': task, 'status': 'NOT_ATTEMPTED'} for task in tasks],
             'scheduler_evidence': [{'returncode': 0}, {'command': ['squeue', '--array', '--noheader',
                 '--jobs=' + driver.ARRAY, '--format=%i|%T'], 'returncode': 1, 'stdout': '',
                 'stderr': 'slurm_load_jobs error: Invalid job id specified\n'}]}
    write(recovery / 'MATRIX_CAPTURE_REPORT.json', prior)
    return root, tasks


def invoke_frozen_packer(root, task, stdout=None):
    job = driver.ARRAY + '_' + str(task['index'])
    command = [str(root / 'pack_evidence.py'), '--root', str(root), '--job-id', job,
               '--mode', 'run', '--run-name', task['run_name'], '--capture', 'COMPLETE']
    def fake_accounting(actual, **kwargs):
        require(actual == ['sacct', '-j', job, '-n', '-P', '--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList,JobIDRaw'])
        return f'{job}|synthetic|COMPLETED|0:0|00:01:00||testnode|{90000 + task["index"]}\n'
    with patch.object(sys, 'argv', command), patch.object(packer.subprocess, 'check_output', side_effect=fake_accounting), contextlib.redirect_stdout(stdout or io.StringIO()):
        packer.main()


def flow(queue_text='99999|RUNNING\n23494070_2|PENDING\n', queue_return=0, reuse=False, partial=False):
    with tempfile.TemporaryDirectory(prefix='stage-b-capture-r2-') as temp:
        root, tasks = fixture(temp)
        protected = {path: driver.sha(path) for path in (root / 'recovery').rglob('*') if path.is_file()}
        if reuse:
            invoke_frozen_packer(root, tasks[0])
            protected.update({path: driver.sha(path) for path in (root / 'recovery').rglob('*') if path.is_file()})
        if partial:
            driver.capture_paths(root, tasks[0])[2].write_bytes(b'Incomplete synthetic archive')
        calls = []; pack_calls = []
        def fake_run(command, **kwargs):
            calls.append(command)
            if command[0] == 'sacct':
                return subprocess.CompletedProcess(command, 0, scheduler_text(), '')
            if command[0] == 'squeue':
                require(command == ['squeue', '--array', '--noheader', '--user=scv7tsq', '--format=%i|%T'])
                return subprocess.CompletedProcess(command, queue_return, queue_text, 'synthetic query failure' if queue_return else '')
            require(command[:4] == [str(driver.PYTHON), '-B', '-u', str(root / 'pack_evidence.py')])
            name = command[command.index('--run-name') + 1]; task = next(task for task in tasks if task['run_name'] == name)
            pack_calls.append(name); invoke_frozen_packer(root, task, kwargs['stdout'])
            return subprocess.CompletedProcess(command, 0)
        with patch.object(driver, 'BROOT', root), patch.object(driver, 'PAYLOAD_SHA', driver.sha(root / 'PAYLOAD_MANIFEST.json')), patch.object(driver, 'PRIOR_REPORT_SHA', driver.sha(root / 'recovery/MATRIX_CAPTURE_REPORT.json')), patch.object(driver.subprocess, 'run', side_effect=fake_run), patch.object(driver.time, 'sleep', return_value=None), contextlib.redirect_stdout(io.StringIO()):
            code = driver.run(SimpleNamespace(root=root))
            report = json.loads((root / 'recovery/MATRIX_CAPTURE_REPORT_R2.json').read_text())
            if queue_return or queue_text.startswith(driver.ARRAY + '_') or partial:
                require(code == 2 and report['verified_capture_count'] == 0 and not pack_calls)
            else:
                require(code == 0 and report['verified_capture_count'] == 12)
                require(len(pack_calls) == (11 if reuse else 12))
                require(all(item['status'] in {'CAPTURED_AND_VERIFIED', 'REUSED_COMPLETE_CAPTURE'} for item in report['runs']))
                if reuse:
                    require(report['runs'][0]['status'] == 'REUSED_COMPLETE_CAPTURE')
            require(len(report['runs']) == 12 and all(driver.sha(path) == value for path, value in protected.items()))
            count = len(calls); rejected(lambda: driver.run(SimpleNamespace(root=root))); require(len(calls) == count)


def main():
    check('r1_source_unchanged', lambda: require(driver.sha(ROOT / 'pack_training_matrix.py') == driver.PRIOR_DRIVER_SHA))
    check('frozen_packer_source_unchanged', lambda: require(driver.sha(ROOT.parent / 'pack_evidence.py') == driver.PACK_SHA))
    check('empty_user_queue_accepted', lambda: require(driver.no_target_in_queue('') == []))
    check('nonempty_unrelated_jobs_accepted', lambda: require(len(driver.no_target_in_queue('99999|RUNNING\n23494070_2|PENDING\n')) == 2))
    def reproduce_old_error():
        command = ['squeue', '--array', '--noheader', '--jobs=' + driver.ARRAY, '--format=%i|%T']
        evidence = []
        with patch.object(driver.subprocess, 'run', return_value=subprocess.CompletedProcess(command, 1, '', 'slurm_load_jobs error: Invalid job id specified\n')):
            rejected(lambda: driver.query(command, evidence))
        require(evidence[0]['returncode'] == 1 and evidence[0]['stderr'] == 'slurm_load_jobs error: Invalid job id specified\n')
    check('actual_r1_invalid_purged_job_error_reproduced_and_preserved', reproduce_old_error)
    for name, raw in [('base_target', '2349407|COMPLETING\n'), ('indexed_target', '2349407_11|PENDING\n'),
                      ('compressed_target', '2349407_[0-11%4]|PENDING\n'), ('heterogeneous_target', '2349407+0|RUNNING\n'),
                      ('missing_field', '99999\n'), ('extra_field', '99999|RUNNING|x\n'),
                      ('bad_job', 'hello|RUNNING\n'), ('bad_state', '99999|garbage\n'),
                      ('blank_row', '\n'), ('duplicate_row', '99999|RUNNING\n99999|RUNNING\n')]:
        check('reject_' + name, lambda raw=raw: rejected(lambda: driver.no_target_in_queue(raw)))
    check('real_frozen_packer_all_twelve_tiny_archives_then_reuse_validation', lambda: flow())
    check('existing_complete_capture_reused_without_repacking', lambda: flow(reuse=True))
    check('partial_existing_capture_stops_before_packer', lambda: flow(partial=True))
    check('user_queue_failure_is_not_ignored', lambda: flow(queue_return=1, queue_text=''))
    check('target_queue_row_blocks_all_capture', lambda: flow(queue_text='2349407_0|COMPLETING\n'))
    report = {'schema': 'stage_b_training_matrix_capture_r2_synthetic_tests_v1', 'status': 'PASS',
              'driver_sha256': driver.sha(ROOT / 'pack_training_matrix_r2.py'), 'test_script_sha256': driver.sha(__file__),
              'prior_driver_sha256': driver.PRIOR_DRIVER_SHA, 'prior_failure_report_sha256': driver.PRIOR_REPORT_SHA,
              'checks_passed': len(checks), 'checks': checks, 'real_scheduler_calls': 0, 'network_calls': 0,
              'optimizer_updates': 0, 'scope': 'Synthetic metadata/placeholder bytes only. Actual frozen packer main and verification executed under mocked Slurm calls, including 12 complete tar captures and reuse. No models, numerical arrays, GPU or platform access.',
              'boundary': 'Format integration cannot prove real remote archive or scientific results; live sacct, successful user-wide squeue and frozen run/artifact validation remain mandatory.'}
    driver.write(ROOT / 'SYNTHETIC_MATRIX_CAPTURE_R2_TESTS_FINAL.json', report)
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'driver_sha256': report['driver_sha256']}))


if __name__ == '__main__':
    main()
