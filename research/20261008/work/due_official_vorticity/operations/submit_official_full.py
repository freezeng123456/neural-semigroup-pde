from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

root = Path('/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1')
wrapper = root / 'operations/run_official_observed.py'
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == 'fa2b0951e283df3c6e20a5c4bce74a456504a8cd17428ad8c7ef2d72f6526395'
profile = json.loads((root / 'runs/preflight-r1/PREFLIGHT_PROFILE_PASSED.json').read_text())
assert profile['status'] == 'PROFILE_AND_ENGINEERING_CHECKS_PASSED'
assert profile['training']['optimizer_updates'] == 125
assert profile['checkpoints']['isolated_resumed_update']['passed']
assert profile['prediction']['finite'] and profile['prediction']['exact_replay_equal']
accounting = subprocess.check_output(['sacct', '-j', '2349393', '--format=JobID,State,ExitCode', '-n', '-P'], text=True)
assert any(line.strip().split('|')[:3] == ['2349393', 'COMPLETED', '0:0'] for line in accounting.splitlines())
assert not (root / 'runs/official-gdsg-r1').exists()
intent_path = root / 'operations/official_submission_intent.json'
assert not intent_path.exists(), 'Prior submission intent exists; inspect scheduler rather than resubmit'
command = ['sbatch', '--parsable', '--partition=gpu', '--gpus=1', '--ntasks=1',
    '--cpus-per-task=10', '--time=04:00:00', '--job-name=due-vort-500',
    '--chdir=' + str(root), '--output=' + str(root / 'runs/official-gdsg-%j.log'),
    str(root / 'operations/slurm_official.sh'), str(root), 'full', 'official-gdsg-r1']
intent = {'requested_at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
    'scope': 'Unchanged author VorticityOSG default, 500 epochs, seed0, batch20, Q=lambda=1',
    'preflight_job_id': '2349393', 'preflight_local_archive_verified': True,
    'preflight_archive_sha256': '851c8e4e7adf9f2686ddce3a2a3c9efd8beb1a552794cb389f1997e7ffb05af2',
    'preflight_first_epoch_seconds': profile['training']['observed_train_seconds'],
    'linear_projection_seconds': profile['full_training_linear_projection_seconds'],
    'allocation_limit_seconds': 14400, 'expected_optimizer_updates': 62500}
intent_path.write_text(json.dumps(intent, indent=2) + '\n')
reply = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
receipt = {**intent, 'returncode': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr}
if reply.returncode == 0:
    job = reply.stdout.strip().split(';')[0]
    assert job.isdigit(), 'Unknown submission response; inspect before retry'
    receipt['job_id'] = job
(root / 'operations/official_submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
reply.check_returncode()
