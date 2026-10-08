#!/usr/bin/env python3
"""Capture completed or failed allocation evidence, with no archive overwrites."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tarfile
from launch_task import sha, verify_payload
from experiment_config import SPECS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--run-name')
    p.add_argument('--failed', action='store_true')
    a = p.parse_args()
    root = a.root.resolve(strict=True); verify_payload(root)
    assert re.fullmatch(r'\d+(?:_\d+)?', a.job)
    if a.run_name:
        match = re.fullmatch(r'([a-z0-9]+)-seed([012])-r1', a.run_name)
        assert match and match[1] in SPECS
        run = root/'runs'/a.run_name
    else:
        assert a.job.isdigit(); run = root/'preflight'/('job'+a.job)
    assert run.is_dir() and not run.is_symlink()
    raw = subprocess.check_output(['sacct', '-j', a.job, '-nP', '--format=JobID,State,ExitCode,Elapsed,NodeList,JobIDRaw'], text=True)
    rows = [row.split('|') for row in raw.splitlines() if row.split('|')[0] == a.job]
    assert len(rows) == 1 and rows[0][1].split()[0].rstrip('+') in {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'}
    if not a.failed:
        assert rows[0][1:3] == ['COMPLETED', '0:0']
        prior = json.loads((run/'artifact_manifest.json').read_text())
        for name, item in prior.items():
            path = (run/name).resolve(strict=True)
            assert path.is_relative_to(run) and path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        if a.run_name:
            receipt = json.loads((run/'RUN_RECEIPT.json').read_text())
            spec = SPECS[match[1]]
            assert receipt['status'] == 'NEXT_METHODS_FULL_COMPLETE' and receipt['actual_updates'] == spec['epochs']*100
            assert receipt['forward_calls'] == spec['epochs']*100*spec['calls']
            assert receipt['final_validation']['valid_metric'] is True
        else:
            receipt = json.loads((run/'VALIDATION_REPORT.json').read_text())
            assert receipt['status'] == 'PASS' and receipt['device'] == 'cuda' and str(receipt['slurm_job_id']) == a.job
    recovery = root/'recovery'; recovery.mkdir(exist_ok=True)
    stem = run.name+'-job'+a.job+('-FAILED' if a.failed else '-COMPLETE')
    archive, sidecar, directory = recovery/(stem+'.tar.gz'), recovery/(stem+'.archive.json'), recovery/stem
    assert not archive.exists() and not sidecar.exists() and not directory.exists(), 'Existing capture: reuse and verify'
    directory.mkdir()
    (directory/'sacct.txt').write_text(raw)
    frozen = json.loads((root/'PAYLOAD_MANIFEST.json').read_text())
    files = {p for p in run.rglob('*') if p.is_file() and 'matplotlib-cache' not in p.parts}
    files |= {root/name for name in frozen['files'] if not name.endswith('.npz')}
    files.add(root/'PAYLOAD_MANIFEST.json'); files.add(directory/'sacct.txt')
    files |= {p for p in (root/'operations').glob('*.json') if p.is_file()}
    files |= {p for p in (root/'logs').glob('*') if p.is_file() and a.job.split('_')[0] in p.name}
    for path in files:
        assert path.resolve(strict=True).is_relative_to(root) and not path.is_symlink()
    inventory = {'schema': 'next_methods_evidence_v1', 'root_name': root.name, 'job': a.job,
                 'run_relative_path': str(run.relative_to(root)), 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                 'payload_manifest_sha256': sha(root/'PAYLOAD_MANIFEST.json'),
                 'capture': 'FAILED' if a.failed else 'COMPLETE',
                 'files': {str(p.relative_to(root)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(files)}}
    inventory_path = directory/'INVENTORY.json'
    with inventory_path.open('x') as f:
        json.dump(inventory, f, indent=2)
    with tarfile.open(archive, 'x:gz') as tar:
        for path in sorted(files | {inventory_path}):
            tar.add(path, arcname=root.name+'/'+str(path.relative_to(root)), recursive=False)
    receipt = {'archive': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
               'inventory_relative_path': str(inventory_path.relative_to(root)), 'archive_members': len(files)+1,
               'job': a.job, 'run_relative_path': str(run.relative_to(root)), 'payload_manifest_sha256': inventory['payload_manifest_sha256']}
    with sidecar.open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
