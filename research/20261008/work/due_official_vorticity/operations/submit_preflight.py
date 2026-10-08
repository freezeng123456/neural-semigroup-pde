"""Verify the reviewed runtime payload and submit exactly one bounded GPU preflight."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone

root = Path('/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1')
payload = root.parent / 'due-vorticity-runtime-v1.tar.gz'
expected = 'bda930d493ca6daad9d149427e3b76a8a6992bf0bdac3e2b0e1fedeec3e2f3a4'
assert payload.stat().st_size == 22968
assert hashlib.sha256(payload.read_bytes()).hexdigest() == expected
assert not (root / 'operations/run_official_observed.py').exists()
assert not (root / 'operations/preflight_submission_intent.json').exists()
assert json.loads((root / 'operations/setup_complete.json').read_text())['status'] == 'passed'
with tarfile.open(payload) as archive:
    for member in archive.getmembers():
        assert (root / member.name).resolve().is_relative_to(root.resolve())
        assert not member.issym() and not member.islnk()
    archive.extractall(root)
manifest = json.loads((root / 'operations/runtime_payload_manifest.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
command = ['sbatch', '--parsable', '--partition=gpu', '--gpus=1', '--ntasks=1',
    '--cpus-per-task=10', '--time=00:30:00', '--job-name=due-vort-pre',
    '--chdir=' + str(root), '--output=' + str(root / 'runs/preflight-%j.log'),
    str(root / 'operations/slurm_official.sh'), str(root), 'preflight', 'preflight-r1']
intent = {'requested_at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
    'runtime_payload_sha256': expected, 'wrapper_sha256': manifest['files']['operations/run_official_observed.py'],
    'scope': 'One complete original epoch, full batch100 prediction and checkpoint engineering tests; not the full 500-epoch baseline.'}
(root / 'operations/preflight_submission_intent.json').write_text(json.dumps(intent, indent=2) + '\n')
reply = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
receipt = {**intent, 'returncode': reply.returncode, 'stdout': reply.stdout, 'stderr': reply.stderr}
if reply.returncode == 0:
    job = reply.stdout.strip().split(';')[0]
    assert job.isdigit(), 'Unexpected scheduler response; inspect without resubmitting'
    receipt['job_id'] = job
(root / 'operations/preflight_submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
reply.check_returncode()
