#!/usr/bin/env python3
"""Launch the preflight or one exact frozen wave task inside its GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from common import ROOT, PYTHON, load, now, require_remote, sha, verify_core, wave_tasks, write


def failure(error):
    marker = ROOT / 'operations/FULL_FAILURE.json'
    record = {'schema': 'burgers_ks_full_failure_v1', 'status': 'FAILED',
              'recorded_at_utc': now(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
              'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
              'error': repr(error), 'traceback': traceback.format_exc(),
              'action': 'Block all subsequent experiment launches; inspect actual Slurm. No retry/requeue/retraining.'}
    try:
        write(marker, record)
    except FileExistsError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'full'))
    parser.add_argument('--wave', type=int, choices=(1, 2))
    args = parser.parse_args()
    require_remote(); verify_core()
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('SLURM_CPUS_PER_TASK') != '10':
        raise ValueError('Launch requires the allocated10CPU/1GPU Slurm job')
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Launch requires the pinned PyTorch interpreter')
    if args.mode == 'preflight':
        if args.wave is not None or os.environ.get('SLURM_ARRAY_TASK_ID'):
            raise ValueError('Preflight is a single engineering allocation')
        runtime = load(ROOT / 'RUNTIME_CONFIG.json')
        if runtime['training_python'] != PYTHON:
            raise ValueError('Frozen runtime configuration differs')
        command = runtime['data_generation_command']
        if not isinstance(command, list) or not all(isinstance(x, str) for x in command) or len(command) < 2:
            raise ValueError('Exact data generation argv required')
        if not Path(command[0]).is_absolute() or str(ROOT / 'prepare_data.py') != command[1]:
            raise ValueError('Data generation must use frozen local script and separate absolute interpreter')
        data_env = dict(os.environ, JAX_PLATFORM_NAME='cpu', CUDA_VISIBLE_DEVICES='')
        print(json.dumps({'stage': 'FROZEN_REFERENCE_DATA_PREPARATION', 'command': command}), flush=True)
        subprocess.run(command, check=True, env=data_env)
        command = [sys.executable, '-B', '-u', str(ROOT / 'preflight.py'),
                   '--data-root', str(ROOT / 'data'), '--device', 'cuda', '--benchmark-updates', '10',
                   '--output-dir', str(ROOT / 'preflight' / ('job' + os.environ['SLURM_JOB_ID']))]
    else:
        try:
            if args.wave is None or (ROOT / 'operations/FULL_FAILURE.json').exists():
                raise ValueError('Wave identity required and no prior failure allowed')
            receipt = load(ROOT / 'operations' / f'full_wave{args.wave}_submission_receipt.json')
            if receipt['status'] != 'SUBMITTED' or str(receipt['job_id']) != os.environ.get('SLURM_ARRAY_JOB_ID'):
                raise ValueError('Allocation does not match the durable submitted wave')
            for name, item in receipt['operational_payload'].items():
                path = ROOT / name
                if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                    raise ValueError('Operational payload changed after submission: ' + name)
            rows = wave_tasks(args.wave)
            index = int(os.environ['SLURM_ARRAY_TASK_ID'])
            if not 0 <= index < 9:
                raise ValueError('Wave array index outside the fixed task set')
            task = rows[index]
            if list((ROOT / 'runs').glob('*/FAILURE.json')):
                raise ValueError('An earlier training failure blocks new launches')
            command = [sys.executable, '-B', '-u', str(ROOT / 'train.py'), '--pde', task['pde'],
                       '--arm', task['arm'], '--seed', str(task['seed']), '--mode', 'full', '--device', 'cuda',
                       '--data-dir', str(ROOT / 'data' / task['pde']),
                       '--output-dir', str(ROOT / 'runs' / task['run_name'])]
        except BaseException as error:
            failure(error); raise
    print(json.dumps({'status': 'LAUNCHING', 'command': command, 'slurm_job_id': os.environ['SLURM_JOB_ID']}), flush=True)
    try:
        subprocess.run(command, check=True)
    except BaseException as error:
        if args.mode == 'full':
            failure(error)
        raise


if __name__ == '__main__':
    main()
