#!/usr/bin/env python3
"""Verify frozen files and runtime, then dispatch one authorized Slurm task."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from experiment_config import matrix


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_payload(root):
    manifest = json.loads((root/'PAYLOAD_MANIFEST.json').read_text())
    assert manifest['prefix'] == root.name
    for name, item in manifest['files'].items():
        path = (root/name).resolve(strict=True)
        assert path.is_relative_to(root) and not path.is_symlink() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
    assert json.loads((root/'matrix.json').read_text()) == matrix()
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--mode', choices=('validate', 'full'), required=True)
    a = p.parse_args()
    root = a.root.resolve(strict=True)
    verify_payload(root)
    import numpy, scipy, matplotlib, yaml, torch
    actual = {'python': platform.python_version(), 'numpy': numpy.__version__, 'scipy': scipy.__version__,
              'matplotlib': matplotlib.__version__, 'yaml': yaml.__version__, 'torch': torch.__version__}
    expected = {'python': '3.11.7', 'numpy': '1.26.4', 'scipy': '1.11.4', 'matplotlib': '3.8.4', 'yaml': '6.0.2', 'torch': '2.5.1+cu121'}
    assert actual == expected, actual
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    common = ['--source-root', str(root/'due'), '--source-manifest', str(root/'official_source_manifest.json'),
              '--data-dir', str(root/'data-r2'), '--device', 'cuda']
    if a.mode == 'validate':
        output = root/'preflight'/('job'+os.environ['SLURM_JOB_ID'])
        command = [sys.executable, '-B', '-u', str(root/'validate_execution.py'), *common,
                   '--benchmark-updates', '20', '--output-dir', str(output)]
    else:
        task = matrix()['tasks'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
        output = root/'runs'/task['run_name']
        command = [sys.executable, '-B', '-u', str(root/'train.py'), *common,
                   '--arm', task['arm'], '--seed', str(task['seed']), '--mode', 'full', '--output-dir', str(output)]
    assert not output.exists(), 'Existing run: inspect instead of overwriting'
    print(json.dumps({'job_id': os.environ['SLURM_JOB_ID'], 'mode': a.mode, 'runtime': actual, 'command': command}), flush=True)
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
