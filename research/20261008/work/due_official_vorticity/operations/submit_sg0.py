from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import subprocess
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['preflight', 'full'], required=True)
mode = parser.parse_args().mode
root = Path('/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1')
expected_wrapper = '77c5e1c4285ea04f1ede56ca127c8222537cc96daa9c98d2b4c462fa9192b434'
wrapper = root / 'operations/run_official_sg0_observed.py'
if not wrapper.exists():
    assert mode == 'preflight'
    payload = root / 'recovery/due-vorticity-sg0-runtime-v1.tar.gz'
    assert payload.stat().st_size == 12737
    assert hashlib.sha256(payload.read_bytes()).hexdigest() == '279a142316a20b196296fb581cbd46e57e1a32bf741598e13843473219de0f0b'
    with tarfile.open(payload) as archive:
        for m in archive.getmembers():
            assert (root / m.name).resolve().is_relative_to(root.resolve())
            assert not m.issym() and not m.islnk()
            assert not (root / m.name).exists()
        archive.extractall(root)
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == expected_wrapper
manifest = json.loads((root / 'operations/sg0_preparation_manifest.json').read_text())
for name, item in manifest['files'].items():
    assert hashlib.sha256((root / 'operations' / name).read_bytes()).hexdigest() == item['sha256']
assert hashlib.sha256((root / 'operations/run_official_observed.py').read_bytes()).hexdigest() == 'fa2b0951e283df3c6e20a5c4bce74a456504a8cd17428ad8c7ef2d72f6526395'
run_name = f'sg0-{mode}-r1'
assert not (root / 'runs' / run_name).exists()
intent_path = root / f'operations/sg0_{mode}_submission_intent.json'
assert not intent_path.exists(), 'Prior intent exists; inspect scheduler before retry'
projection = None
preflight_job = None
if mode == 'preflight':
    limit_seconds = 900
else:
    prior = json.loads((root / 'operations/sg0_preflight_submission.json').read_text())
    preflight_job = prior['job_id']
    rows = subprocess.check_output(['sacct', '-j', preflight_job, '--format=JobID,State,ExitCode', '-n', '-P'], text=True)
    assert any(r.strip().split('|')[:3] == [preflight_job, 'COMPLETED', '0:0'] for r in rows.splitlines())
    profile = json.loads((root / 'runs/sg0-preflight-r1/SG0_PREFLIGHT_PROFILE_PASSED.json').read_text())
    assert profile['status'] == 'SG0_PROFILE_AND_ENGINEERING_CHECKS_PASSED'
    assert profile['training']['optimizer_updates'] == 125
    assert profile['checkpoints']['isolated_resumed_update']['passed']
    assert profile['prediction']['engineering_checks_passed']
    projection = profile['full_training_linear_projection_seconds']
    limit_seconds = max(3600, math.ceil((projection * 1.8 + 300) / 900) * 900)
    assert limit_seconds <= 14400, 'Unexpectedly slow preflight; inspect before allocating'
walltime = f'{limit_seconds // 3600:02d}:{(limit_seconds % 3600) // 60:02d}:00'
command = ['sbatch', '--parsable', '--partition=gpu', '--gpus=1', '--ntasks=1',
    '--cpus-per-task=10', '--time=' + walltime, '--job-name=due-sg0-' + mode,
    '--chdir=' + str(root), '--output=' + str(root / f'runs/sg0-{mode}-%j.log'),
    str(root / 'operations/slurm_sg0.sh'), str(root), mode, run_name]
intent = {'requested_at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
    'variant': 'author_supported_sg_pairing_0', 'configuration_change': {'training.sg_pairing': {'from': 1, 'to': 0}},
    'mode': mode, 'wrapper_sha256': expected_wrapper, 'preflight_job_id': preflight_job,
    'linear_projection_seconds': projection, 'allocation_limit_seconds': limit_seconds,
    'expected_optimizer_updates': 125 if mode == 'preflight' else 62500}
intent_path.write_text(json.dumps(intent, indent=2) + '\n')
reply = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
receipt = {**intent, 'returncode': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr}
if reply.returncode == 0:
    job = reply.stdout.strip().split(';')[0]
    assert job.isdigit(), 'Unknown submission response; inspect before retry'
    receipt['job_id'] = job
(root / f'operations/sg0_{mode}_submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
reply.check_returncode()
