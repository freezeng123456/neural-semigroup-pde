#!/usr/bin/env python3
"""Read scheduler and progress metadata; persist one non-overwriting snapshot."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/data02/run01/scv7tsq/neural-semigroup/20261005-next-methods-r1')
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
records = []
for wave in (1, 2):
    receipt_path = ROOT/'operations'/f'full_wave{wave}_submission_receipt.json'
    if not receipt_path.exists():
        continue
    receipt = json.loads(receipt_path.read_text())
    job = receipt['job_id']
    assert receipt['status'] == 'SUBMITTED' and job.isdigit()
    evidence = {}
    for kind, command in (
        ('squeue', ['squeue', '--array', '--noheader', '-j', job, '--format=%i|%T|%M|%R']),
        ('sacct', ['sacct', '-j', job, '-nP', '--format=JobID%40,State%30,ExitCode,Elapsed,NodeList']),
    ):
        r = subprocess.run(command, capture_output=True, text=True, timeout=60)
        evidence[kind] = {'command': command, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
        assert r.returncode == 0, 'Actual scheduler query failed'
    records.append({'wave': wave, 'job_id': job, **evidence})

tasks = json.loads((ROOT/'matrix.json').read_text())['tasks']
progress = []
for task in tasks:
    run = ROOT/'runs'/task['run_name']
    if not run.exists():
        continue
    path = run/'epochs.jsonl'
    lines = path.read_text().splitlines() if path.exists() else []
    epoch = None
    for line in reversed(lines):
        try:
            epoch = json.loads(line); break
        except json.JSONDecodeError:
            continue
    env_path = run/'environment.json'
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    failure_path = run/'FAILURE.json'
    result_path = run/'RUN_RECEIPT.json'
    row = {'index': task['index'], 'arm': task['arm'], 'seed': task['seed'], 'run_name': task['run_name'],
           'planned_updates': task['updates'], 'last_epoch': epoch, 'gpu_name': env.get('gpu_name'),
           'failure': json.loads(failure_path.read_text()) if failure_path.exists() else None,
           'complete_receipt_present': result_path.exists()}
    if result_path.exists():
        result = json.loads(result_path.read_text())
        row['completion'] = {k: result.get(k) for k in ('status', 'actual_updates', 'forward_calls', 'parameters_changed', 'final_validation')}
    progress.append(row)

value = {'schema': 'next_methods_heartbeat_snapshot_v1', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
         'scheduler': records, 'progress': progress,
         'wave2_intent_present': (ROOT/'operations/full_wave2_submission_intent.json').exists(),
         'wave2_receipt_present': (ROOT/'operations/full_wave2_submission_receipt.json').exists()}
if '_INSPECTION_SOURCE' in globals():
    value['inspection_program_sha256'] = hashlib.sha256(_INSPECTION_SOURCE).hexdigest()
    value['inspection_program_source'] = _INSPECTION_SOURCE.decode()
path = ROOT/'operations'/('heartbeat_'+STAMP+'.json')
with path.open('x') as f:
    json.dump(value, f, indent=2, allow_nan=False)
print('UTC', value['timestamp_utc'])
for record in records:
    print('WAVE', record['wave'], 'JOB', record['job_id']); print(record['squeue']['stdout'])
print('INDEX ARM EPOCH UPDATES GPU FAILURE COMPLETE')
for row in progress:
    epoch = row['last_epoch'] or {}
    print(row['index'], row['arm'], epoch.get('epoch'), epoch.get('updates'), row['gpu_name'],
          row['failure']['error'] if row['failure'] else 'NONE', row['complete_receipt_present'])
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print('SNAPSHOT', path.name, 'BYTES', path.stat().st_size)
print('SHA_HEAD', digest[:32]); print('SHA_TAIL', digest[32:])
print('WAVE2_INTENT', value['wave2_intent_present'], 'WAVE2_RECEIPT', value['wave2_receipt_present'])
