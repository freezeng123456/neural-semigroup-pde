#!/usr/bin/env python3
"""Build an explicit finite payload; preserve earlier freezes and data failures."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile
from experiment_config import matrix


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--cpu-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', default='20261005-next-methods-r1')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    assert a.prefix.startswith('20261005-next-methods-') and '/' not in a.prefix
    assert not a.output.exists() and not (here/'PAYLOAD_MANIFEST.json').exists()
    report = json.loads(a.cpu_report.read_text())
    assert report['status'] == 'PASS' and report['device'] == 'cpu' and len(report['checks']) == 29
    assert report['actual_optimizer_updates'] == 32 and all(c['passed'] is True for c in report['checks'])
    source = json.loads(a.source_manifest.read_text())
    assert source['commit'] == '1e629104a12a9c3e3064996cbd4e301cec641ad8' and len(source['files']) == 75
    with (here/'matrix.json').open('x') as f:
        json.dump(matrix(), f, indent=2)
    names = ['NEXT_EXPERIMENT_PROTOCOL.zh.md', 'GENERATOR_PROTOCOL.zh.md', 'two_step_objectives.py',
             'experiment_config.py', 'flow_numerics.py', 'generator_model.py', 'prepare_flow_data.py',
             'train.py', 'validate_execution.py', 'evaluate_generator.py', 'launch_task.py',
             'slurm_next.sh', 'submit_experiments.py', 'scheduler_limits.py', 'pack_evidence.py',
             'recover_preflight.py', 'build_payload.py', 'matrix.json',
             'data-r2/DATA_REPORT.json', 'data-r2/two_step_data.npz',
             'flow-data-r2/FLOW_DATA_REPORT.json', 'flow-data-r2/flow_data.npz']
    included = {name: here/name for name in names}
    for name, path in included.items():
        assert path.is_file() and not path.is_symlink(), name
        if name.endswith('.py'):
            ast.parse(path.read_text())
    for name, digest in source['files'].items():
        path = a.source_root/name; assert sha(path) == digest
        included['due/'+name] = path
    included['official_source_manifest.json'] = a.source_manifest
    included['CPU_ENGINEERING_REPORT.json'] = a.cpu_report
    manifest = {'schema': 'next_methods_payload_v1', 'prefix': a.prefix, 'source_commit': source['commit'],
                'scope': 'Two-step map controls: 18 runs; autonomous/time-conditioned generator: 6 runs',
                'cpu_checks': 29, 'cpu_optimizer_updates': 32,
                'runtime_python': '/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python',
                'dependency_root': '/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps',
                'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size} for name, path in sorted(included.items())}}
    manifest_file = here/'PAYLOAD_MANIFEST.json'
    with manifest_file.open('x') as f:
        json.dump(manifest, f, indent=2)
    a.output.parent.mkdir(exist_ok=True, parents=True)
    with tarfile.open(a.output, 'x:gz') as tar:
        for name, path in sorted(included.items()):
            tar.add(path, arcname=a.prefix+'/'+name, recursive=False)
        tar.add(manifest_file, arcname=a.prefix+'/PAYLOAD_MANIFEST.json', recursive=False)
    receipt = {'archive': str(a.output.resolve()), 'prefix': a.prefix, 'bytes': a.output.stat().st_size,
               'sha256': sha(a.output), 'manifest_sha256': sha(manifest_file), 'archive_members': len(included)+1}
    with a.output.with_suffix(a.output.suffix+'.json').open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
