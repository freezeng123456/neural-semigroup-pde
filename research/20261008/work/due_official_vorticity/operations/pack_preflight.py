from pathlib import Path
import hashlib
import json
import subprocess
import tarfile

root = Path('/data02/run01/scv7tsq/neural-semigroup/20261004-due-vorticity-1e62910-r1')
recovery = root / 'recovery'
recovery.mkdir(exist_ok=True)
archive = recovery / 'preflight-evidence-2349393.tar.gz'
assert not archive.exists(), 'Evidence archive already exists; reuse it'
profile = json.loads((root / 'runs/preflight-r1/PREFLIGHT_PROFILE_PASSED.json').read_text())
assert profile['status'] == 'PROFILE_AND_ENGINEERING_CHECKS_PASSED'
assert profile['training']['optimizer_updates'] == 125
assert profile['checkpoints']['isolated_resumed_update']['passed']
sacct = subprocess.check_output(['sacct', '-j', '2349393', '--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS', '-n', '-P'], text=True)
assert any(line.startswith('2349393|') and '|COMPLETED|0:0|' in line for line in sacct.splitlines())
(recovery / 'preflight-slurm-2349393.txt').write_text(sacct)
files = []
for p in (root / 'runs/preflight-r1').rglob('*'):
    if p.is_file() and not p.is_symlink() and 'matplotlib-cache' not in p.parts:
        files.append(p)
files.append(root / 'runs/preflight-2349393.log')
files.extend(p for p in (root / 'operations').iterdir() if p.is_file())
for n in json.loads((root / 'operations/official_source_manifest.json').read_text())['files']:
    files.append(root / 'due' / n)
files.extend([root / 'data/manifest.json', root / 'data/download_manifest.json',
    root / 'OFFICIAL_PROTOCOL_AUDIT.md', root / 'inputs_archive_manifest.json',
    recovery / 'preflight-slurm-2349393.txt'])
files = sorted(set(files))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
inventory = {'job_id': '2349393', 'files': [
    {'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]}
inventory_file = recovery / 'preflight-inventory-2349393.json'
inventory_file.write_text(json.dumps(inventory, indent=2) + '\n')
with tarfile.open(archive, 'w:gz') as tar:
    for p in files + [inventory_file]:
        tar.add(p, arcname=str(p.relative_to(root)), recursive=False)
receipt = {'file': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha(archive),
    'inventory_files': len(files), 'omitted': ['original MAT files already verified locally', 'dependency binaries', 'matplotlib cache']}
(recovery / 'preflight-archive-2349393.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
