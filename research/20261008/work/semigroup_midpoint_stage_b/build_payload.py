#!/usr/bin/env python3
"""Build a new, finite Stage B upload from explicit reviewed files."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', default='20261004-midpoint-stage-b-r1')
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    assert a.prefix.startswith('20261004-midpoint-stage-b-') and '/' not in a.prefix
    assert not a.output.exists()
    manifest_file = here / 'PAYLOAD_MANIFEST.json'
    assert not manifest_file.exists(), 'Existing freeze must be inspected before rebuilding'
    source_manifest = json.loads(a.source_manifest.read_text())
    assert source_manifest['commit'] == '1e629104a12a9c3e3064996cbd4e301cec641ad8'
    assert len(source_manifest['files']) == 75
    included = {}
    for name, digest in source_manifest['files'].items():
        path = a.source_root / name
        assert sha(path) == digest
        included['due/' + name] = path
    names = ['PROTOCOL.zh.md', 'prepare_data.py', 'losses.py', 'train.py',
             'validate_stage_b.py', 'slurm_stage_b.sh', 'launch_task.py',
             'matrix.json', 'build_payload.py', 'submit_stage_b.py',
             'pack_evidence.py', 'recover_evidence.py',
             'data-r1/DATA_MANIFEST.json', 'data-r1/development_data.npz']
    for name in names:
        path = here / name
        assert path.is_file() and not path.is_symlink()
        included[name] = path
    included['official_source_manifest.json'] = a.source_manifest
    data_manifest = json.loads((here / 'data-r1/DATA_MANIFEST.json').read_text())
    assert sha(here / 'data-r1/development_data.npz') == data_manifest['payload_sha256']
    manifest = {'source_commit': source_manifest['commit'], 'prefix': a.prefix,
                'scope': 'Stage B four arms x three seeds, new fixed 80/20 development split',
                'shared_dependency_root': '/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1/deps',
                'runtime_python': '/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python',
                'files': {name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                          for name, path in sorted(included.items())}}
    with manifest_file.open('x') as f:
        json.dump(manifest, f, indent=2); f.write('\n')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(a.output, 'x:gz') as tar:
        for name, path in sorted(included.items()):
            tar.add(path, arcname=a.prefix + '/' + name, recursive=False)
        tar.add(manifest_file, arcname=a.prefix + '/PAYLOAD_MANIFEST.json', recursive=False)
    receipt = {'archive': str(a.output.resolve()), 'bytes': a.output.stat().st_size,
               'sha256': sha(a.output), 'prefix': a.prefix, 'listed_files': len(included),
               'archive_members': len(included) + 1, 'manifest_sha256': sha(manifest_file)}
    with a.output.with_suffix(a.output.suffix + '.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
