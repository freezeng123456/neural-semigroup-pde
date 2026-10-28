"""Bounded identities, exact file checks and durable records for this benchmark."""
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path('/data02/run01/scv7tsq/neural-semigroup/20261028-burgers-ks-r1')
PDES = ('burgers', 'ks')
ARMS = ('first', 'teacher2', 'rollout2')
PYTHON = '/data02/run01/scv7tsq/apolarity/20261002-v100-paper-r1/env/runtime/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def now():
    return datetime.now(timezone.utc).isoformat()


def require_remote():
    if getpass.getuser() != 'scv7tsq' or ROOT != EXPECTED_ROOT:
        raise ValueError('Restricted to the authorized Beijing account and benchmark root')


def verify_core():
    manifest = load(ROOT / 'CORE_MANIFEST.json')
    for name, item in manifest['files'].items():
        path = ROOT / name
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Unsafe core path: ' + name)
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Frozen core changed: ' + name)
    return manifest


def tasks():
    matrix = load(ROOT / 'matrix.json'); rows = matrix['tasks']
    identities = {(pde, arm, seed) for pde in PDES for seed in range(3) for arm in ARMS}
    if len(rows) != 18 or {(r['pde'], r['arm'], r['seed']) for r in rows} != identities:
        raise ValueError('Expected exactly18 frozen PDE/arm/seed tasks')
    if [r['index'] for r in rows] != list(range(18)) or len({r['run_name'] for r in rows}) != 18:
        raise ValueError('Task indices/names must be unique and contiguous')
    if any(not re.fullmatch(r'[a-z0-9-]+', r['run_name']) for r in rows):
        raise ValueError('Unsafe run name')
    if matrix['epochs'] != 500 or matrix['updates_per_epoch'] != 100 or matrix['batch_size'] != 20 or matrix['max_concurrency'] != 2:
        raise ValueError('Scientific endpoint or concurrency differs')
    return rows


def wave_tasks(wave):
    if wave not in (1, 2):
        raise ValueError('Expected fixed wave1 or2')
    selected = [row for row in tasks() if row['pde'] == PDES[wave - 1]]
    if len(selected) != 9:
        raise ValueError('Each fixed PDE wave must contain9 tasks')
    return selected


def file_inventory(run):
    inventory = load(run / 'artifact_manifest.json')
    actual = {p.relative_to(run).as_posix() for p in run.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts}
    if actual != set(inventory) | {'artifact_manifest.json'}:
        raise ValueError('Run inventory differs from actual files: ' + str(run))
    for name, item in inventory.items():
        path = run / name
        if path.is_symlink() or not path.resolve().is_relative_to(run.resolve()):
            raise ValueError('Unsafe artifact path')
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Artifact bytes/hash mismatch: ' + str(path))
    return sha(run / 'artifact_manifest.json')


def query(command):
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Slurm query failed: ' + ' '.join(command) + ': ' + result.stderr)
    return {'command': command, 'returncode': result.returncode, 'stdout': result.stdout}


def accounting(job, expected_count, require_success=True):
    if not str(job).isdigit():
        raise ValueError('Actual numeric job identity required')
    snapshot = query(['sacct', '-nP', '-j', str(job), '--format=JobID,JobIDRaw,State,ExitCode'])
    rows = [line.split('|') for line in snapshot['stdout'].splitlines() if line.strip()]
    allocations = {r[0]: r for r in rows if re.fullmatch(str(job) + r'_\d+', r[0])}
    if require_success:
        if set(allocations) != {str(job) + '_' + str(i) for i in range(expected_count)}:
            raise ValueError('Expected exact terminal array allocation set')
        if any(row[2:4] != ['COMPLETED', '0:0'] for row in allocations.values()):
            raise ValueError('A frozen training allocation did not complete successfully')
    snapshot['allocation_rows'] = allocations
    return snapshot


def live_queue(jobs):
    result = query(['squeue', '--array', '--noheader', '--user=scv7tsq', '--format=%i|%T'])
    matched = []
    for line in result['stdout'].splitlines():
        fields = line.split('|')
        if len(fields) != 2:
            raise ValueError('Unexpected live queue columns')
        if any(fields[0] == job or fields[0].startswith(job + '_') for job in jobs):
            matched.append(line)
    # Keep only this experiment's queue rows in its compact record.
    return {'command': result['command'], 'returncode': 0, 'matched_rows': matched}


def operations_payload():
    return {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted((ROOT / 'operations').iterdir()) if p.is_file() and (p.suffix in {'.py', '.sh'} or p.name == 'PREFLIGHT_CONTRACT.json')}
