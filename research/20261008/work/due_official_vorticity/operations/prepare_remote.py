"""Verify and unpack frozen inputs, then install dependencies into this run only."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tarfile

ROOT = Path('/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1')
ARCHIVE = ROOT.parent / 'due-vorticity-inputs-1e62910.tar'
EXPECTED_SIZE = 356085760
EXPECTED_SHA = '5ab7f7d59c35fe5910caf4d15a251264263e73446fc32cfd5ad7799db2db0698'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def unpack(path, dest):
    with tarfile.open(path) as archive:
        for item in archive.getmembers():
            target = (dest / item.name).resolve()
            if not target.is_relative_to(dest.resolve()) or item.issym() or item.islnk():
                raise ValueError(f'Unsafe archive member: {item.name}')
        archive.extractall(dest)

assert ARCHIVE.stat().st_size == EXPECTED_SIZE, 'Input upload incomplete'
assert sha(ARCHIVE) == EXPECTED_SHA, 'Input archive hash mismatch'
assert not ROOT.exists(), 'Run root already exists; inspect before any retry'
ROOT.mkdir()
unpack(ARCHIVE, ROOT)
(ROOT / 'due').mkdir()
unpack(ROOT / 'due-source-1e62910.tar', ROOT / 'due')
source = json.loads((ROOT / 'operations/official_source_manifest.json').read_text())
for name, expected in source['files'].items():
    assert sha(ROOT / 'due' / name) == expected, name
data = json.loads((ROOT / 'data/manifest.json').read_text())
for item in data['files']:
    assert sha(ROOT / 'data' / item['filename']) == item['sha256'], item['filename']
wheels = json.loads((ROOT / 'operations/wheelhouse_manifest.json').read_text())
for item in wheels['packages']:
    assert sha(ROOT / 'wheelhouse' / item['file']) == item['sha256'], item['file']
(ROOT / 'runs').mkdir()
env = dict(os.environ, PYTHONNOUSERSITE='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
result = subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
    '--find-links', str(ROOT / 'wheelhouse'), '--target', str(ROOT / 'deps'),
    '-r', str(ROOT / 'operations/requirements-linux-cp311.lock'),
], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(ROOT / 'operations/dependency_install.log').write_text(result.stdout)
print(result.stdout, flush=True)
result.check_returncode()
env.update(PYTHONPATH=f'{ROOT}/deps:{ROOT}/due', MPLBACKEND='Agg')
check = subprocess.run([
    sys.executable, '-c',
    'import json,sys,torch,numpy,scipy,matplotlib,yaml,due; '
    'print(json.dumps({"python":sys.version,"torch":torch.__version__, '
    '"numpy":numpy.__version__,"scipy":scipy.__version__, '
    '"matplotlib":matplotlib.__version__,"yaml":yaml.__version__, '
    '"due_path":due.__file__},indent=2))',
], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(ROOT / 'operations/import_check.log').write_text(check.stdout)
print(check.stdout, flush=True)
check.check_returncode()
(ROOT / 'operations/setup_complete.json').write_text(json.dumps({
    'status': 'passed', 'input_archive_sha256': EXPECTED_SHA,
    'source_files_verified': len(source['files']), 'data_files_verified': len(data['files']),
    'wheels_verified': len(wheels['packages']), 'python': sys.executable,
    'dependency_location': str(ROOT / 'deps'), 'shared_runtime_modified': False,
}, indent=2) + '\n')
print('SETUP_COMPLETE', ROOT, flush=True)
