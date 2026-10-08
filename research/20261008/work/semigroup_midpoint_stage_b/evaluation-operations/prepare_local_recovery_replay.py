#!/usr/bin/env python3
"""Prepare exact, guarded local recovery/replay commands; never execute them.

Requires an authentic remote matrix capture report and its separately observed
SHA-256. Retains all twelve cells, refuses partial existing outputs, and rechecks
complete existing outputs. Frozen recovery/replay programs stay unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tarfile

sys.dont_write_bytecode = True
BASE = Path('/Users/Apple_501/Documents/Codex/2026-10-03/cha')
WORK = BASE / 'work/semigroup_midpoint_stage_b'
OUT = BASE / 'outputs/semigroup-midpoint-stage-b-20261004'
REMOTE = '/data02/run01/scv7tsq/neural-semigroup/20261004-midpoint-stage-b-r1'
PYTHON = Path('/Users/Apple_501/Documents/Codex/2026-10-01/zha/work/apolarity-test-venv/bin/python')
OVERLAY = BASE / 'work/due_official_vorticity/operations/local-import-deps'
PREFIX = Path(REMOTE).name
PINNED = {'PAYLOAD_MANIFEST.json': 'c30e2208cff36ba79b91da43b947fd85f7f0c23088d5020e2824f906704df466',
          'pack_evidence.py': '4406547114b29f6a99b1a68a067cb3095bda5df70f139815ba347e6afcb08bbc',
          'recover_evidence.py': '58d8a9e9a0de99a5365cfb1e2f182a548bc0972f23163046b0f0e32c172d07f2',
          'replay_final_checkpoint.py': '942102528eeccb9ba45319c25fdc27673b667ae5c7c4116d41c6b7ad0c1d420d'}
ENV = {'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '0', 'PYTHONPATH': str(OVERLAY),
       'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MPLBACKEND': 'Agg'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_file(root, name):
    relative = PurePosixPath(name)
    require(relative.parts and not relative.is_absolute() and '..' not in relative.parts
            and str(relative) == name, 'Unsafe relative path')
    path = root
    for part in relative.parts:
        path = path / part
        require(not path.is_symlink(), 'Symlink forbidden: ' + str(path))
    require(path.is_file(), 'Missing file: ' + str(path))
    return path


def command(script, arguments, replay=False):
    # Enforce actual interpreter/library versions inside each execution, including
    # a replay resumed much later. runpy executes the unchanged frozen program.
    code = ('import hashlib,runpy,sys; from pathlib import Path; '
            'assert sys.version_info[:2] == (3,12); '
            f'p=Path({str(WORK / script)!r}); '
            f'assert hashlib.sha256(p.read_bytes()).hexdigest()=={PINNED[script]!r}; ')
    if replay:
        code += "import torch,numpy; assert torch.__version__=='2.5.1'; assert numpy.__version__=='1.26.4'; "
    else:
        # recover_evidence imports the frozen packer from the same source folder.
        code += (f'assert hashlib.sha256(Path({str(WORK / "pack_evidence.py")!r}).read_bytes()).hexdigest()=={PINNED["pack_evidence.py"]!r}; '
                 f'sys.path.insert(0,{str(WORK)!r}); ')
    code += f'sys.argv={[str(WORK / script)] + arguments!r}; runpy.run_path(str(p),run_name="__main__")'
    argv = ['/usr/bin/env'] + [key + '=' + value for key, value in ENV.items()] + [str(PYTHON), '-B', '-c', code]
    return {'argv': argv, 'shell': shlex.join(argv), 'frozen_script_sha256': PINNED[script]}


def local_paths(task):
    name = task['run_name']
    return {'archive': OUT / 'archives' / Path(task['expected_complete_archive']).name,
            'recovery': OUT / task['local_recovery_directory'],
            'recovery_report': OUT / 'training-recovery-reports' / (name + '.json'),
            'replay': OUT / 'replay' / name}


def receipt_for(task, record):
    task_keys = ('index', 'arm', 'seed', 'run_name', 'epochs', 'updates', 'forwards_per_batch', 'total_forward_calls')
    require(record['task'] == {key: task[key] for key in task_keys}, 'Remote task differs from local plan')
    require(record['status'] in {'CAPTURED_AND_VERIFIED', 'REUSED_COMPLETE_CAPTURE'}, 'Remote capture incomplete')
    receipt = record['receipt']
    expected = {'archive': task['expected_complete_archive'], 'job_id': task['job_id'], 'mode': 'run',
                'capture_status': 'COMPLETE', 'payload_manifest_sha256': PINNED['PAYLOAD_MANIFEST.json']}
    require(all(receipt.get(key) == value for key, value in expected.items()), 'Remote receipt identity differs')
    require(isinstance(receipt['bytes'], int) and receipt['bytes'] > 0
            and re.fullmatch(r'[a-f0-9]{64}', receipt['sha256'])
            and re.fullmatch(r'[0-9]+', receipt['allocation_job_id_raw']), 'Invalid receipt archive/raw ID')
    expected_inventory = 'recovery/' + Path(task['expected_complete_archive']).name[:-7] + '/INVENTORY.json'
    require(receipt['inventory_relative_path'] == expected_inventory, 'Wrong receipt inventory path')
    serialized = (json.dumps(receipt, indent=2, allow_nan=False) + '\n').encode()
    require(hashlib.sha256(serialized).hexdigest() == record['receipt_sha256'], 'Embedded receipt does not match frozen packer serialization/hash')
    return receipt


def verify_recovery(task, receipt, paths, manifest, packer):
    require(paths['recovery'].is_dir() and paths['recovery_report'].is_file(), 'Incomplete recovery outputs; inspect, do not rerun')
    result = read(paths['recovery_report'])
    root = paths['recovery'] / PREFIX; run = root / 'runs' / task['run_name']
    expected = {'status': 'ARCHIVE_AND_ALL_MEMBERS_VERIFIED', 'job_id': task['job_id'],
                'allocation_job_id_raw': receipt['allocation_job_id_raw'], 'mode': 'run', 'capture_status': 'COMPLETE',
                'archive': str(paths['archive'].resolve()), 'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes'],
                'payload_manifest_sha256': PINNED['PAYLOAD_MANIFEST.json'], 'root': str(root.resolve()),
                'run_root': str(run.resolve()), 'script_sha256': PINNED['recover_evidence.py']}
    require(all(result.get(key) == value for key, value in expected.items()), 'Recovery report binding differs')
    inventory_path = safe_file(root, receipt['inventory_relative_path'])
    with tarfile.open(paths['archive'], 'r:gz') as archive:
        archived_inventory = archive.extractfile(PREFIX + '/' + receipt['inventory_relative_path']).read()
    require(inventory_path.read_bytes() == archived_inventory, 'Recovered inventory differs from authentic archive')
    inventory = read(inventory_path)
    require(inventory['run_relative_path'] == 'runs/' + task['run_name'] and inventory['job_id'] == task['job_id']
            and inventory['allocation_job_id_raw'] == receipt['allocation_job_id_raw'], 'Recovered inventory run binding differs')
    names = set()
    for item in inventory['files']:
        path = safe_file(root, item['path']); names.add(item['path'])
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Recovered artifact changed: ' + item['path'])
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    require(len(names) == len(inventory['files']) == receipt['inventory_files']
            and actual == names | {receipt['inventory_relative_path']}, 'Recovered membership differs')
    require(sha(root / 'PAYLOAD_MANIFEST.json') == PINNED['PAYLOAD_MANIFEST.json'], 'Recovered payload changed')
    for name, spec in manifest['files'].items():
        if name != 'data-r1/development_data.npz':
            path = safe_file(root, name)
            require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Recovered frozen source changed')
    packer.verify_complete_run(root, run, 'run', task['job_id'], manifest, receipt['allocation_job_id_raw'])
    return root, run


def verify_replay(task, receipt, paths, run):
    directory = paths['replay']
    report_path = safe_file(directory, 'FINAL_REPLAY_REPORT.json')
    prediction = safe_file(directory, 'development_first_two_steps_cpu.npy')
    require({path.name for path in directory.iterdir()} == {report_path.name, prediction.name}, 'Unexpected replay output membership')
    report = read(report_path)
    expected = {'schema': 'stage_b_final_cpu_replay_v1', 'status': 'PASS', 'job_id': task['job_id'],
                'allocation_job_id_raw': receipt['allocation_job_id_raw'], 'arm': task['arm'], 'seed': task['seed'],
                'archive_sha256': receipt['sha256'], 'payload_manifest_sha256': PINNED['PAYLOAD_MANIFEST.json'],
                'checkpoint_sha256': sha(run / 'final_state.pt'), 'verifier_sha256': PINNED['replay_final_checkpoint.py'],
                'cpu_replay_sha256': sha(prediction), 'actual_training_updates_verified': task['updates'],
                'actual_training_forwards_verified': task['total_forward_calls'],
                'final_network_state_exactly_equal': True, 'all_adam_steps_and_scheduler_endpoint_verified': True,
                'development_metrics_recomputed': True, 'replay_batch_size': 20, 'replay_steps_per_trajectory': 2,
                'optimizer_updates_by_verifier': 0, 'cpu_gpu_relative_l2_tolerance': 2e-5}
    require(all(report.get(key) == value for key, value in expected.items()), 'Replay report binding/checks differ')
    errors = report['cpu_gpu_relative_l2_by_trajectory']
    require(len(errors) == 20 and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 2e-5 for x in errors), 'Replay tolerance failed')
    require(report['checkpoint_tensor_count_finite'] > 0 and report['protected_inputs_unchanged'] > 0, 'Replay input/tensor checks absent')
    return {'report_sha256': sha(report_path), 'prediction_sha256': sha(prediction), 'status': 'PASS_RECORD_AND_INPUTS_REVERIFIED'}


def prepare(args):
    require(not sys.flags.optimize, 'Assertions must stay enabled in the frozen tools')
    require(re.fullmatch(r'[a-f0-9]{64}', args.capture_sha256) and sha(args.capture_report) == args.capture_sha256,
            'Remote capture report must match separately observed SHA-256')
    for name, digest in PINNED.items():
        require(sha(WORK / name) == digest, 'Frozen local tool/payload changed: ' + name)
    require(PYTHON.is_file() and OVERLAY.is_dir(), 'Required local interpreter/dependency overlay unavailable')
    manifest = read(WORK / 'PAYLOAD_MANIFEST.json')
    data = WORK / 'data-r1/development_data.npz'; data_spec = manifest['files']['data-r1/development_data.npz']
    require(data.stat().st_size == data_spec['bytes'] and sha(data) == data_spec['sha256'], 'Frozen development dataset changed')
    plan = read(OUT / 'RECOVERY_PLAN.json'); tasks = plan['tasks']; matrix = read(WORK / 'matrix.json')['tasks']
    require(plan['parent_job_id'] == '2349407' and plan['remote_root'] == REMOTE and len(tasks) == 12
            and all({key: task[key] for key in original} == original for task, original in zip(tasks, matrix)), 'Recovery plan differs from frozen matrix')
    for task in tasks:
        job = '2349407_' + str(task['index'])
        stem = task['run_name'] + '-job' + job + '-COMPLETE'
        require(task['job_id'] == job and task['remote_run_root'] == REMOTE + '/runs/' + task['run_name']
                and task['expected_complete_archive'] == REMOTE + '/recovery/' + stem + '.tar.gz'
                and task['expected_archive_receipt'] == REMOTE + '/recovery/' + stem + '.archive.json'
                and task['local_recovery_directory'] == 'training-recovery/' + task['run_name'],
                'Recovery plan paths/job differ from fixed matrix')
    remote = read(args.capture_report)
    require(remote['schema'] == 'stage_b_training_matrix_capture_v1' and remote['root'] == REMOTE
            and remote['training_array_job'] == '2349407' and remote['status'] == 'ALL_TWELVE_CAPTURES_VERIFIED'
            and remote['verified_capture_count'] == 12 and len(remote['runs']) == 12, 'Need complete authentic remote capture report')
    module_spec = importlib.util.spec_from_file_location('frozen_local_capture_verifier', WORK / 'pack_evidence.py')
    packer = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(packer)
    result = {'schema': 'stage_b_local_recovery_replay_command_plan_v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'capture_report_sha256': args.capture_sha256, 'capture_report': str(args.capture_report.resolve()),
              'generator_sha256': sha(__file__), 'frozen_files': PINNED, 'development_data': str(data),
              'public_test_data_used': False, 'commands_executed': 0, 'optimizer_updates': 0, 'runs': []}
    raw_ids = set()
    for task, remote_record in zip(tasks, remote['runs']):
        record = {'run_name': task['run_name'], 'job_id': task['job_id'], 'commands': [], 'status': 'NOT_CHECKED'}
        result['runs'].append(record)
        try:
            receipt = receipt_for(task, remote_record)
            require(receipt['allocation_job_id_raw'] not in raw_ids, 'Duplicate remote raw ID'); raw_ids.add(receipt['allocation_job_id_raw'])
            paths = local_paths(task); record['paths'] = {key: str(value) for key, value in paths.items()}
            if not paths['archive'].exists():
                record['status'] = 'AWAITING_ARCHIVE_DOWNLOAD'; continue
            safe_file(OUT, str(paths['archive'].relative_to(OUT)))
            require(paths['archive'].stat().st_size == receipt['bytes'] and sha(paths['archive']) == receipt['sha256'], 'Downloaded archive SHA/bytes differ')
            for key in ('recovery', 'recovery_report', 'replay'):
                require(not paths[key].is_symlink(), 'Output symlink forbidden')
            staging = paths['recovery'].with_name(paths['recovery'].name + '.partial')
            require(not staging.exists() and not staging.is_symlink(), 'Partial extraction exists; inspect, do not retry')
            any_recovery = paths['recovery'].exists() or paths['recovery_report'].exists()
            if any_recovery:
                _, run = verify_recovery(task, receipt, paths, manifest, packer)
                record['recovery_status'] = 'COMPLETED_OUTPUTS_REVERIFIED'
            else:
                require(not paths['replay'].exists(), 'Replay output exists without verified recovery')
                arguments = ['--archive', str(paths['archive']), '--sha256', receipt['sha256'], '--bytes', str(receipt['bytes']),
                             '--job-id', task['job_id'], '--payload-manifest', str(WORK / 'PAYLOAD_MANIFEST.json'),
                             '--output-dir', str(paths['recovery']), '--report', str(paths['recovery_report'])]
                record['commands'].append({'phase': 'recover', **command('recover_evidence.py', arguments)})
                record['recovery_status'] = 'NEW_OUTPUT_REQUIRED'; run = None
            if paths['replay'].exists():
                record['replay_verification'] = verify_replay(task, receipt, paths, run)
                record['status'] = 'RECOVERY_AND_REPLAY_REVERIFIED'
            else:
                arguments = ['--recovery-report', str(paths['recovery_report']), '--payload-manifest', str(WORK / 'PAYLOAD_MANIFEST.json'),
                             '--data-dir', str(WORK / 'data-r1'), '--output-dir', str(paths['replay'])]
                record['commands'].append({'phase': 'replay', **command('replay_final_checkpoint.py', arguments, replay=True)})
                record['status'] = 'COMMANDS_READY_EXECUTE_SEQUENTIALLY'
        except Exception as error:
            record.update(status='BLOCKED_INSPECT_RETAIN_OUTPUTS', error=repr(error), commands=[])
    result['completion_verified'] = all(record['status'] == 'RECOVERY_AND_REPLAY_REVERIFIED' for record in result['runs'])
    result['execution_rule'] = 'Run each ready cell in listed phase order. Save return code/stdout/stderr outside recovered/replay inputs; do not run its replay after recovery failure. Never repeat failed commands automatically. Regenerate a NEW plan after execution to verify existing complete outputs.'
    result['reuse_limit'] = 'Rechecks authentic archive/current recovered hashes and bound PASS replay outputs; it does not independently repeat prior numerical inference or prove the environment used by externally generated historical replay records.'
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    with args.output_plan.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'plan': str(args.output_plan), 'plan_sha256': sha(args.output_plan),
                      'states': {state: sum(record['status'] == state for record in result['runs']) for state in sorted({r['status'] for r in result['runs']})},
                      'completion_verified': result['completion_verified'], 'commands_executed': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-report', type=Path, required=True)
    parser.add_argument('--capture-sha256', required=True)
    parser.add_argument('--output-plan', type=Path, required=True)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
