#!/usr/bin/env python3
"""Validate the frozen payload and dispatch one bounded Stage B task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def runtime_identity():
    import platform
    import numpy
    import scipy
    import matplotlib
    import yaml
    import torch
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__,
                'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__,
                'PyYAML': yaml.__version__, 'torch': torch.__version__}
    expected = {'python': '3.11.7', 'numpy': '1.26.4', 'scipy': '1.11.4',
                'matplotlib': '3.8.4', 'PyYAML': '6.0.2', 'torch': '2.5.1+cu121'}
    assert versions == expected, versions
    return {'versions': versions, 'executable': sys.executable,
            'executable_sha256': sha(Path(sys.executable)),
            'module_paths': {module.__name__: module.__file__ for module in
                             [numpy, scipy, matplotlib, yaml, torch]},
            'cuda_runtime': torch.version.cuda}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--mode', required=True, choices=['full', 'validate'])
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    manifest = json.loads((root / 'PAYLOAD_MANIFEST.json').read_text())
    assert manifest['source_commit'] == '1e629104a12a9c3e3064996cbd4e301cec641ad8'
    for name, spec in manifest['files'].items():
        path = (root / name).resolve(strict=True)
        assert path.is_relative_to(root) and path.is_file()
        assert path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], name
    assert os.environ.get('SLURM_JOB_ID'), 'Requires a Slurm allocation'
    runtime = runtime_identity()
    common = ['--source-root', str(root / 'due'), '--source-manifest', str(root / 'official_source_manifest.json'),
              '--data-dir', str(root / 'data-r1'), '--device', 'cuda']
    if a.mode == 'full':
        index = int(os.environ['SLURM_ARRAY_TASK_ID'])
        matrix = json.loads((root / 'matrix.json').read_text())
        assert len(matrix['tasks']) == 12 and 0 <= index < 12
        task = matrix['tasks'][index]
        assert task['index'] == index
        run = root / 'runs' / task['run_name']
        assert not run.exists(), 'Existing run; inspect its state before any restart'
        command = [sys.executable, '-B', '-u', str(root / 'train.py'), *common,
                   '--arm', task['arm'], '--seed', str(task['seed']), '--mode', 'full', '--output-dir', str(run)]
    else:
        run = root / 'preflight' / ('job' + os.environ['SLURM_JOB_ID'])
        assert not run.exists()
        command = [sys.executable, '-B', '-u', str(root / 'validate_stage_b.py'), *common,
                   '--train-mat', '/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/data/VorticityOSG_train.mat',
                   '--output-dir', str(run)]
    print(json.dumps({'slurm_job_id': os.environ['SLURM_JOB_ID'], 'mode': a.mode,
                      'payload_manifest_sha256': sha(root / 'PAYLOAD_MANIFEST.json'),
                      'runtime': runtime, 'command': command}, indent=2), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
